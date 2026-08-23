"""M2CDuetRehearsal -- drum-prefix rehearsal context + interleaved AR suffix.

Replaces the retired M2CIntraCrossAttnRecon (#3) with a genuine
loss/architecture conditioning baseline. The "rehearsal" is implemented
architecturally: the entire drum stream is prepended as a bidirectional
context block, then the model runs standard DuetAttn-style interleaved
AR on the suffix with full visibility into the prefix.

Sequence layout (length 3T):

  default (shift-1, the conditional scheme):
  [ a_0, a_1, ..., a_{T-1},   sos_m, a_0, b_0, a_1, b_1, ..., a_{T-1} ]
   └─ prefix (T pos) ────┘   └──── shifted interleaved suffix (2T pos) ────┘
   bidirectional within        causal within; sees ALL of the prefix
   never sees the suffix       slot i predicts x_i and HOLDS x_{i-1},
   no targets, no loss         so the slot predicting b_k holds a_k

  legacy (shift-2, --suffix_shift1 0): the suffix opens [sos_m, sos_c,
  a_0, b_0, ...] and slot i holds x_{i-2}, so b_k's slot holds b_{k-1}
  and a_k is in that slot's masked future.

The prefix is the same drum content as appears at the suffix's drum
slots; it's just made available before AR begins so nondrum predictions
in the suffix can attend to FUTURE drum content (not just past drum).
That's the rehearsal: the model knows what drum is going to be when it
predicts nondrum.

Loss = CE on the mod_b slots of the suffix (--target_only_loss, the
default) + recon_weight * the Brier retrieval term on the mod_a slots +
the MoE aux. Token-level CE on mod_a is dropped -- it is trivially
satisfiable and dilutes val_loss -- but the Brier term is KEPT, because
it is not a copy: the slot predicting mod_a[k] holds mod_a[k-1], so it
has to retrieve mod_a[k] out of the prefix, over the same k_m/v_m the
conditioning path reads. The signal is mod_b's CE, and b_k sees ALL of
mod_a via the prefix instead of only past mod_a (DuetAttn's behaviour).

Architecture: 2 SDPA passes per block (intra + cross) with per-modality
Q/K/V/O, per-modality cross gate, shared MoE FFN -- same machinery as
DuetAttn (#2). Only the attention mask shape and the input layout
differ.

Warm-start: same per-modality remap as DuetAttn (mod-a and mod-b
projections initialized from the single-stream pretrained backbone).
Cross gate bias = -10 at init -> cross stream silent at step 0 -> the
suffix collapses to per-modality self-attention on the clean shifted
positions, and the drum prefix's only effect at init is providing
extra same-modality keys to suffix drum attention. Not exact
warm-start equivalence with DuetAttn (prefix changes the attention
distribution), but close to it.

Inference: cp_transformer_m2c_duet_rehearsal_inference.py. It keeps a
prefix buffer of committed mod_a frames separate from the suffix's
shift-trick buffer, and mirrors the ckpt's shift -- under shift-1 it
commits a_t BEFORE sampling b_t, so the query slot holds what it held
in training.

CONDITIONAL-MODEL CORRECTIONS (all ON by default; each rides in the
state_dict, so a checkpoint decodes and trains under the scheme it was
built with and older ckpts are unaffected by these defaults).

This variant inherited its layout, shift and loss from the symmetric
CO-GENERATION lineage (DuetAttn), where both streams are predicted and
neither is given. As a CONDITIONAL model -- mod_a given in full, mod_b
generated -- three of those inherited choices were wrong:

--suffix_shift1  Under the inherited shift-2 the slot predicting mod_b[k]
    holds mod_b[k-1], and mod_a[k] sits at slot 2k+2, in that slot's
    MASKED FUTURE. Both streams at frame k were predicted from strictly
    earlier frames -- correct for co-generation, wrong here: mod_b[k]
    could not see the very frame it is conditioned on except by reaching
    back into the prefix. Shift-1 (single SOS) puts mod_a[k] in the query
    slot itself, so in the interleaved [a, b, a, b, ...] region every b
    is conditioned on its own a.
    THE GATE MUST BE OPENED TO MATCH. The intra/cross split is keyed on
    SLOT parity, and under shift-1 slot parity is the opposite of content
    parity: odd ("mod-b") slots hold mod_a, even ("mod-a") slots hold
    mod_b. So for the slot predicting b_k the intra path reaches mod_a,
    and mod_b's OWN AUTOREGRESSIVE HISTORY moves onto the cross path --
    together with the prefix. With the inherited gate_init_bias of -10
    (sigmoid ~ 4.5e-5) the model would start with no access to its own
    history at all. train_duet_rehearsal.sbatch therefore defaults
    GATE_INIT_BIAS to 0.0 (sigmoid = 0.5) whenever SUFFIX_SHIFT1=1, and
    keeps -10 for shift-2, where the intra path already carries mod_b and
    the closed gate is the intended warm-start trick.
    The projections themselves are fine either way: the assignment is
    consistent (odd always mod_a, even always mod_b), and the warm start
    replicates the SAME pretrained q/k/v into both branches
    (init_pretrained_into_jointattn._map_global_key), so neither branch
    is initialised for a role it does not get.

--target_only_loss  CE on the mod_b slots only. Token-level CE on mod_a
    is trivially satisfiable and dilutes val_loss, so it is dropped.
    THIS DOES NOT TOUCH THE BRIER TERM, which is controlled separately by
    recon_weight and stays ON by default -- the two were coupled at first
    and that was a mistake. The Brier term is not a copy task: under the
    shift the slot predicting mod_a[k] holds mod_a[k-1], so satisfying it
    requires RETRIEVING mod_a[k] from the prefix by position, and that
    retrieval runs over k_m/v_m at the prefix positions, which are the
    same keys and values a mod_b query reads on the cross path. It
    therefore trains the shared half of the conditioning mechanism, which
    is the point of the rehearsal.
    With recon_weight > 0, val_loss carries that term, so select
    checkpoints on val_ce_loss_nondrum (--ckpt_monitor) to rank on the
    conditional objective alone.
    Note the PREFIX itself never had a loss and still does not: logits
    are read only from suffix positions.

--prefix_stride2  Fixes a UNIT MISMATCH. The interleaved suffix advances
    TWO rotary units per musical frame; the mod_a-only prefix advanced
    ONE. The same frame carried two rotary coordinates that drift apart,
    so the distance from mod_b[k]'s query to mod_a[k] in the prefix grew
    as k+1: one frame at the start of the sequence, T at the end. Exactly
    where the rehearsal should pay off (late frames, distant future
    mod_a), RoPE placed the answer furthest from the question.
    The rule: the prefix copy of frame j takes THE SAME ROTARY INDEX AS
    THE SUFFIX SLOT THAT HOLDS IT, so one musical event has one rotary
    coordinate. That slot depends on the shift -- 2j+1 under shift-1,
    2j+2 under shift-2 -- so the offset tracks suffix_shift1 rather than
    being a fixed constant. A bare 2j is NOT equivalent: under shift-1 it
    lands on the even suffix slots, which hold mod_b, making the prefix
    copy of mod_a[j] positionally indistinguishable from the suffix slot
    holding mod_b[j-1] while both are keyed with the mod-a projections.
    With the matched offset there is no collision (prefix odd, mod_b
    slots even under shift-1) and the query-to-prefix distance is a
    constant 0 (shift-1) or -1 (shift-2) for every k.
    The rationale this came from ("give the prefix a self-consistent
    0-indexed RoPE for prefix-INTERNAL relative time") is preserved:
    stride-2 only scales those internal distances by 2.
    The RoPE table is sized from the indices actually used, since the
    widened prefix reaches 2(T-1)+offset, past the suffix's 2T-1.

Pass 0 to any of them to reproduce a pre-fix run. The run directory name
carries _sh1/_tgt/_ps2 so checkpoints of different schemes never share a
directory (resolve_best_ckpt scans by directory).

M2CDuetPrefix (C.2) needed none of these: it is single-stream per block,
already shifts by one over a pure mod_b block, already scores mod_b
alone, and already advances one rotary unit per frame.

Under target_only_loss this model's val_loss IS comparable to
M2CDuetPrefix's -- both are then mod_b CE plus the MoE aux term. Under
the old joint objective it was not, which is why val_ce_loss_nondrum
exists; see --ckpt_monitor for selecting checkpoints on it.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_transformer_m2c_moe import (
    RoFormerSymbolicTransformer, FramedDataset, TRAIN_LENGTH, MAX_STEPS,
)
from cp_transformer_m2c_jointattn import (
    _rope_freqs, _apply_rope, SimpleMoEFFN,
)
from tasks import get_task, TASKS


# ---------------------------------------------------------------------------
# Per-layer 2-pass block (intra + cross) with prefix+suffix masks
# ---------------------------------------------------------------------------

class M2CDuetRehearsalLayer(nn.Module):
    """One transformer block, post-LN, with per-modality Q/K/V/O and two
    masked SDPA passes under the rehearsal mask scheme.

    Input layout assumed:
      h = [ drum prefix (T pos, mod-a),
            shifted interleaved suffix (2T pos: alternating mod-a, mod-b) ]
      total length L = 3T.
    """

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 moe_num_experts, moe_topk, moe_intermediate_size,
                 dropout=0.0, gate_init_bias=-10.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Per-modality Q/K/V/O. m = drum (mod-a), c = nondrum (mod-b).
        self.q_m = nn.Linear(hidden_size, hidden_size)
        self.k_m = nn.Linear(hidden_size, hidden_size)
        self.v_m = nn.Linear(hidden_size, hidden_size)
        self.o_m = nn.Linear(hidden_size, hidden_size)
        self.q_c = nn.Linear(hidden_size, hidden_size)
        self.k_c = nn.Linear(hidden_size, hidden_size)
        self.v_c = nn.Linear(hidden_size, hidden_size)
        self.o_c = nn.Linear(hidden_size, hidden_size)

        # Per-modality cross gate, bias=-10 at init (cross stream silent).
        self.gate_m = nn.Linear(hidden_size, 1)
        self.gate_c = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            for g in (self.gate_m, self.gate_c):
                g.weight.zero_()
                g.bias.fill_(gate_init_bias)

        self.ln_attn = nn.LayerNorm(hidden_size)
        self.ln_ffn = nn.LayerNorm(hidden_size)
        self.use_moe = moe_num_experts > 1
        ffn_inter = moe_intermediate_size or intermediate_size
        if self.use_moe:
            self.ffn = SimpleMoEFFN(hidden_size, ffn_inter,
                                     num_experts=moe_num_experts,
                                     topk=moe_topk)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ffn_inter),
                nn.GELU(),
                nn.Linear(ffn_inter, hidden_size),
            )
        self.drop = nn.Dropout(dropout)

        # Mask cache keyed by (T, device). seq_len always = 3T at training.
        self._mask_cache_key = None
        self._mask_intra = None
        self._mask_cross = None

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * d)

    def _build_masks(self, T, device):
        """Build the rehearsal intra/cross masks for seq_len = 3T.

        Positions [0, T)       : drum prefix (mod-a), bidirectional within.
        Positions [T, 3T)      : interleaved suffix.
                                 mod-a at offset 0 (suffix pos T+2k),
                                 mod-b at offset 1 (suffix pos T+2k+1).
        """
        L = 3 * T
        cache_key = (T, str(device))
        if self._mask_cache_key == cache_key:
            return self._mask_intra, self._mask_cross

        pos = torch.arange(L, device=device)
        is_prefix = pos < T
        is_suffix = ~is_prefix

        # Modality per position. Prefix is all mod-a; suffix alternates.
        suffix_offset = (pos - T).clamp_min(0)
        modality = torch.where(
            is_prefix,
            torch.zeros_like(pos),
            suffix_offset % 2,
        )

        # Broadcast.
        m_p = modality[:, None]
        m_q = modality[None, :]
        same_mod = (m_p == m_q)
        diff_mod = ~same_mod

        # Within-suffix causal: q <= p (only meaningful when both in suffix).
        causal_pos = pos[None, :] <= pos[:, None]

        prefix_p = is_prefix[:, None]
        prefix_q = is_prefix[None, :]
        suffix_p = is_suffix[:, None]
        suffix_q = is_suffix[None, :]

        # Visibility per-row.
        # Row p in PREFIX:
        #   - q in prefix: yes (bidirectional within).
        #   - q in suffix: no (prefix never sees suffix).
        # Row p in SUFFIX:
        #   - q in prefix: yes (full visibility).
        #   - q in suffix: q <= p (causal within suffix).
        visible = (
            (prefix_p & prefix_q)
            | (suffix_p & prefix_q)
            | (suffix_p & suffix_q & causal_pos)
        )

        # Now split visible into intra and cross by modality.
        mask_intra = visible & same_mod
        mask_cross = visible & diff_mod

        # Empty-row safeguard for mask_cross. Prefix-only positions have
        # no other-modality keys at all (prefix is all mod-a, can't see
        # suffix). Add diagonal to be safe -- wrong-modality self-attn at
        # init is harmless because gate_m.bias = -10 -> sigmoid ~ 0.
        diag = torch.eye(L, dtype=torch.bool, device=device)
        mask_cross = mask_cross | diag

        self._mask_cache_key = cache_key
        self._mask_intra = mask_intra
        self._mask_cross = mask_cross
        return mask_intra, mask_cross

    def forward(self, h, T, cos, sin, gate_offset=0.0):
        """h: [B, 3T, H]. Returns (h_out, aux_loss).

        gate_offset is added to the gate PRE-ACTIVATION, so the
        caller can ramp the cross path open on a schedule without
        touching the learned bias.
        """
        B, L, H = h.shape
        assert L == 3 * T

        # Split per modality.
        # Mod-a positions: prefix [0, T) AND suffix mod-a [T, 3T, step 2 from T].
        # Mod-b positions: suffix mod-b [T+1, 3T, step 2 from T+1].
        h_d_prefix = h[:, :T]                   # [B, T, H]
        h_d_suffix = h[:, T::2]                 # [B, T, H]  (positions T, T+2, ..., 3T-2)
        h_c_suffix = h[:, T+1::2]               # [B, T, H]  (positions T+1, T+3, ..., 3T-1)
        h_m_all = torch.cat([h_d_prefix, h_d_suffix], dim=1)   # [B, 2T, H]
        h_c_all = h_c_suffix                                    # [B, T, H]

        # Per-modality Q/K/V projections.
        q_m = self._split_heads(self.q_m(h_m_all))
        k_m = self._split_heads(self.k_m(h_m_all))
        v_m = self._split_heads(self.v_m(h_m_all))
        q_c = self._split_heads(self.q_c(h_c_all))
        k_c = self._split_heads(self.k_c(h_c_all))
        v_c = self._split_heads(self.v_c(h_c_all))

        # Scatter back into flat [B, h, L, d_k] in prefix+suffix layout.
        def _scatter(t_m, t_c):
            B_, hd, _, dk = t_m.shape
            out = torch.zeros(B_, hd, L, dk, device=t_m.device, dtype=t_m.dtype)
            # Mod-a: positions 0..T-1 (prefix) get t_m[:T];
            #        positions T, T+2, ... get t_m[T:].
            out[:, :, :T] = t_m[:, :, :T]
            out[:, :, T::2] = t_m[:, :, T:]
            # Mod-b: positions T+1, T+3, ... get t_c.
            out[:, :, T+1::2] = t_c
            return out

        q = _scatter(q_m, q_c)
        k = _scatter(k_m, k_c)
        v = _scatter(v_m, v_c)

        # RoPE on Q, K. The caller pre-built cos/sin with per-position
        # local indexing (prefix 0..T-1, suffix 0..2T-1), shape
        # [1, 1, L, head_dim], so no further slicing here.
        q, k = _apply_rope(q, k, cos, sin)

        # Two SDPA passes.
        mask_intra, mask_cross = self._build_masks(T, q.device)
        out_intra = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_intra)
        out_cross = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_cross)
        out_intra = self._merge_heads(out_intra)
        out_cross = self._merge_heads(out_cross)

        # Gather per modality for the gates + output projection.
        def _gather_m(t):
            return torch.cat([t[:, :T], t[:, T::2]], dim=1)   # [B, 2T, H]

        def _gather_c(t):
            return t[:, T+1::2]                                # [B, T, H]

        u_intra_m = _gather_m(out_intra)
        u_intra_c = _gather_c(out_intra)
        u_cross_m = _gather_m(out_cross)
        u_cross_c = _gather_c(out_cross)

        # Gates.
        g_m = torch.sigmoid(self.gate_m(h_m_all) + gate_offset)
        g_c = torch.sigmoid(self.gate_c(h_c_all) + gate_offset)
        self._last_gate_m = g_m.detach()
        self._last_gate_c = g_c.detach()

        o_m = self.o_m(u_intra_m + g_m * u_cross_m)
        o_c = self.o_c(u_intra_c + g_c * u_cross_c)

        # Scatter outputs back.
        out_flat = torch.zeros_like(h)
        out_flat[:, :T] = o_m[:, :T]
        out_flat[:, T::2] = o_m[:, T:]
        out_flat[:, T+1::2] = o_c

        h = self.ln_attn(h + self.drop(out_flat))

        if self.use_moe:
            ffn_out, aux_loss = self.ffn(h)
        else:
            ffn_out = self.ffn(h)
            aux_loss = torch.zeros((), device=h.device, dtype=h.dtype)
        h = self.ln_ffn(h + self.drop(ffn_out))
        return h, aux_loss


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class M2CDuetRehearsal(RoFormerSymbolicTransformer):
    """DuetAttn-style joint interleaved AR with a bidirectional drum
    prefix prepended as rehearsal context. See module docstring.
    """

    def __init__(self, *args, moe_num_experts=4, moe_topk=2,
                 moe_intermediate_size=None, global_num_layers=None,
                 global_dropout=0.0, preserve_program=True,
                 gate_init_bias=-10.0, recon_weight=1.0,
                 prefix_stride2=True, suffix_shift1=True,
                 target_only_loss=True, gate_target_bias=0.0,
                 gate_warmup_steps=1000, **kwargs):
        super().__init__(
            *args,
            moe_num_experts=moe_num_experts,
            moe_topk=moe_topk,
            moe_intermediate_size=moe_intermediate_size,
            global_num_layers=global_num_layers,
            global_dropout=global_dropout,
            preserve_program=preserve_program,
            **kwargs,
        )
        # Weight on the Brier-style drum reconstruction loss
        # ||softmax(drum_logits) - one_hot(drum_target)||^2. With the
        # rehearsal prefix giving full drum visibility this term is
        # closely related to CE on drum (both push softmax -> one-hot),
        # but it provides an explicit "match the drum I just saw in
        # the prefix" gradient that's useful as the explicit rehearsal
        # supervision and as a logged quantity for analysis.
        self.recon_weight = float(recon_weight)
        # Conditional model: mod_a is GIVEN, so nothing is learned by
        # scoring the model on reproducing it. Under target_only_loss the
        # CE is restricted to mod_b slots and the Brier drum term is
        # logged but not optimised.
        self.target_only_loss = bool(target_only_loss)
        del self.global_roformer
        del self.gate_m
        del self.gate_c

        ffn_inter = moe_intermediate_size or self.intermediate_size
        self.global_layers = nn.ModuleList([
            M2CDuetRehearsalLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                moe_num_experts=moe_num_experts,
                moe_topk=moe_topk,
                moe_intermediate_size=ffn_inter,
                dropout=global_dropout,
                gate_init_bias=gate_init_bias,
            )
            for _ in range(self.global_num_layers)
        ])

        # SOS offsets per modality (matches the DuetAttn lineage).
        self.sos_offset_m = nn.Parameter(torch.zeros(self.hidden_size))
        self.sos_offset_c = nn.Parameter(torch.zeros(self.hidden_size))

        # Geometry flags ride in the state_dict so a checkpoint decodes
        # under the scheme it was TRAINED with, no CLI needed at inference
        # (same pattern as A.1's time_rope_aligned_flag).
        self.register_buffer(
            'prefix_stride2_flag',
            torch.tensor(1 if prefix_stride2 else 0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            'suffix_shift1_flag',
            torch.tensor(1 if suffix_shift1 else 0, dtype=torch.long),
            persistent=True,
        )
        # GATE WARMUP. The cross gate starts at sigmoid(gate_init_bias) so
        # the model begins as the pretrained single-stream LM. At -10 that
        # is 4.5e-5 -- and sigmoid'(-10) is the SAME 4.5e-5, so the gate
        # both passes almost no signal and receives almost none to change.
        # Waiting for gradients to escape that saturation is a gamble,
        # especially on a corpus that overfits in ~1k steps. Instead ramp
        # an ADDITIVE offset on the pre-activation from 0 to
        # (gate_target_bias - gate_init_bias) over gate_warmup_steps: the
        # warm start is exact at step 0, and the prefix is guaranteed to
        # come online by the end of the ramp regardless of gradient flow.
        # The learned bias keeps training on top of it.
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.register_buffer(
            'gate_offset_total',
            torch.tensor(float(gate_target_bias) - float(gate_init_bias)),
            persistent=True,
        )
        # How far the ramp got. Persisted so INFERENCE reproduces the
        # effective gate the model was last trained with -- without this a
        # decode would silently run at the raw learned bias.
        self.register_buffer('gate_ramp', torch.tensor(1.0), persistent=True)

        # Modality info carried by per-modality QKVO; freeze token type
        # embedding as a no-op.
        with torch.no_grad():
            self.token_type_embeddings.weight.zero_()
        self.token_type_embeddings.weight.requires_grad = False

    @property
    def prefix_stride2(self):
        return bool(self.prefix_stride2_flag.item())

    @property
    def suffix_shift1(self):
        return bool(self.suffix_shift1_flag.item())

    def _assemble_sos(self, batch_size, device, dtype):
        sos_m = (self.global_sos + self.sos_offset_m).view(1, 1, -1)
        sos_c = (self.global_sos + self.sos_offset_c).view(1, 1, -1)
        sos = torch.cat([sos_m, sos_c], dim=1).expand(batch_size, -1, -1)
        return sos.to(device=device, dtype=dtype)

    def _assemble_sos1(self, batch_size, device, dtype):
        """Single SOS for the shift-by-1 suffix. Suffix slot 0 predicts
        drum_0, so the one prepended token is the mod-a SOS."""
        sos_m = (self.global_sos + self.sos_offset_m).view(1, 1, -1)
        return sos_m.expand(batch_size, 1, -1).to(device=device, dtype=dtype)

    def build_suffix(self, h):
        """h: [B, 2T, H] encoded interleaved frames -> the shifted suffix.

        shift-2 (legacy): [sos_m, sos_c, x_0 .. x_{2T-3}]. Slot i predicts
            x_i and HOLDS x_{i-2}, so the slot predicting nondrum_k holds
            nondrum_{k-1} and drum_k -- which sits at slot 2k+2 -- is in
            the slot's future and therefore masked. Both streams at frame
            k are predicted from strictly earlier frames: a symmetric
            CO-GENERATION shift, which is what this variant inherited.
        shift-1: [sos_m, x_0 .. x_{2T-2}]. Slot i predicts x_i and HOLDS
            x_{i-1}, so the slot predicting nondrum_k holds drum_k itself.
            That is the correct teacher forcing for a CONDITIONAL model,
            where drum is given: the frame being conditioned on is in the
            query slot rather than an unreachable future position.
        """
        B = h.shape[0]
        if self.suffix_shift1:
            sos = self._assemble_sos1(B, h.device, h.dtype)
            return torch.cat([sos, h[:, :-1]], dim=1)
        sos = self._assemble_sos(B, h.device, h.dtype)
        return torch.cat([sos, h[:, :-2]], dim=1)

    def _gate_means(self):
        """Mean cross-gate value per modality, averaged over layers.

        The gates decide whether the rehearsal prefix reaches the
        generated stream at all: for a mod_b query the prefix is
        cross-modality, so g_c scales ALL of it. At gate_init_bias=-10,
        g_c = 4.5e-5 and sigmoid'(-10) is the same 4.5e-5 -- the gate
        passes almost no signal AND receives almost no gradient to
        change. Whether it climbs out of that is the single most
        important thing to watch in a C.1 run, so log it.

        Reads the values cached by the last forward; returns None before
        one has run.
        """
        gm = [l._last_gate_m.mean() for l in self.global_layers
              if getattr(l, '_last_gate_m', None) is not None]
        gc = [l._last_gate_c.mean() for l in self.global_layers
              if getattr(l, '_last_gate_c', None) is not None]
        if not gc:
            return None, None
        return torch.stack(gm).mean(), torch.stack(gc).mean()

    def _run_global_stack(self, h_full, T):
        """h_full: [B, 3T, H] prefix + shifted suffix. Returns (h_global, aux).

        RoPE is applied SEGMENT-WISE: prefix positions [0, T) get rotations
        0..T-1; suffix positions [T, 3T) get rotations 0..2T-1. This
        preserves the pretrained backbone's RoPE phase on the suffix
        (where it runs standard DuetAttn AR) and gives the prefix its
        own self-consistent 0-indexed RoPE so the model can learn
        prefix-internal relative-time structure independently of the
        suffix's RoPE phase.
        """
        B, L, H = h_full.shape
        assert L == 3 * T
        head_dim = H // self.num_attention_heads
        # Base RoPE table for up to 2T positions (max length of any segment).
        positions = torch.arange(L, device=h_full.device)
        if self.prefix_stride2:
            # Give the prefix copy of frame j the SAME rotary index as the
            # SUFFIX SLOT THAT HOLDS IT, so one musical event has one
            # rotary coordinate. Which slot that is depends on the shift:
            #   shift-1: suffix idx 2j+1 holds mod_a[j]
            #   shift-2: suffix idx 2j+2 holds mod_a[j]
            # Getting this wrong is not merely a constant offset. Indexing
            # the prefix at a bare 2j under shift-1 lands it on the EVEN
            # suffix slots, which hold mod_b -- so the prefix copy of
            # mod_a[j] and the suffix slot holding mod_b[j-1] become
            # positionally indistinguishable, and both are keyed with the
            # mod-a projections. With the matched offset there is no such
            # collision: prefix indices are odd under shift-1, and the
            # suffix's mod_b slots are even.
            offset = 1 if self.suffix_shift1 else 2
            prefix_pos = positions * 2 + offset
        else:
            prefix_pos = positions
        # Build per-position local index: prefix as above; suffix -> pos - T.
        #
        # prefix_stride2 fixes a UNIT MISMATCH. The suffix interleaves two
        # streams, so it advances TWO rotary units per musical frame, while
        # the mod_a-only prefix advances one. Same frame j therefore had
        # two rotary coordinates that drift apart, and the distance from
        # mod_b[k]'s query slot to mod_a[k] in the prefix grew as k+1 --
        # one frame at the start of the sequence, T at the end. Exactly
        # where the rehearsal is supposed to pay off (late frames, distant
        # future mod_a), RoPE placed the answer furthest from the question.
        local_pos = torch.where(positions < T, prefix_pos, positions - T)
        # Size the table from what is actually indexed: the widened prefix
        # reaches 2(T-1)+offset, which exceeds the suffix's 2T-1 when
        # offset is 2.
        max_pos = int(local_pos.max().item()) + 1
        cos_base, sin_base = _rope_freqs(
            max_pos, head_dim, device=h_full.device, dtype=h_full.dtype,
        )
        cos = cos_base[:, :, local_pos]   # [1, 1, L, head_dim]
        sin = sin_base[:, :, local_pos]
        if self.training and self.gate_warmup_steps > 0:
            step = float(getattr(self, 'global_step', 0) or 0)
            self.gate_ramp.fill_(min(1.0, step / self.gate_warmup_steps))
        gate_offset = self.gate_offset_total * self.gate_ramp
        total_aux = torch.zeros((), device=h_full.device, dtype=h_full.dtype)
        for layer in self.global_layers:
            h_full, aux = layer(h_full, T, cos, sin, gate_offset=gate_offset)
            total_aux = total_aux + aux
        return h_full, total_aux / max(len(self.global_layers), 1)

    def forward(self, x):
        """x: [B, 2T_full, subseq_len]  interleaved [drum_0, nondrum_0, ...].

        Returns logits [B, 2T_full, subseq_len, vocab], aux_loss.
        Logits are over the SUFFIX positions of the global stack (matching
        the standard DuetAttn target alignment).
        """
        batch_size, seq_len, subseq_len = x.shape
        assert seq_len % 2 == 0
        T_full = seq_len // 2

        # Token type ids for local encoder (frame-level even=drum, odd=nondrum).
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2).long()
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, seq_len, subseq_len
        )
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, seq_len, 1
        )
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)

        h, emb = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)

        # mod_a prefix: T_full positions, no shift.
        h_drum = h[:, 0::2]                          # [B, T_full, H]
        # Suffix: interleaved, shifted by 1 or 2 per suffix_shift1.
        h_suffix = self.build_suffix(h)                 # [B, 2T_full, H]

        # Concat.
        h_full = torch.cat([h_drum, h_suffix], dim=1)   # [B, 3T_full, H]

        h_global, aux_loss = self._run_global_stack(h_full, T=T_full)

        # Read off suffix hiddens; local_decode against the original emb.
        h_suffix_global = h_global[:, T_full:T_full + seq_len]   # [B, 2T_full, H]
        logits = self.local_decode(h_suffix_global, emb)
        return logits, aux_loss

    def loss(self, x_mel, x_acc, batch_pitch_shift):
        """CE over the suffix. Under target_only_loss (the default) only
        the mod_b slots are scored -- mod_a is given, so reproducing it
        teaches nothing. The prefix contributes via attention only and
        never carries targets under either setting."""
        x_mel, x_acc = self.preprocess(x_mel, batch_pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape

        # Anti-exposure-bias augmentation (see base _ctx_corrupt_acc):
        # silent runs in the acc INPUT context; CE targets from clean x.
        x_acc_in = self._ctx_corrupt_acc(x_acc)
        stacked = torch.stack([x_mel, x_acc], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)
        if x_acc_in is x_acc:
            x_in = x
        else:
            x_in = torch.stack([x_mel, x_acc_in], dim=2).view(
                batch_size, seq_len * 2, subseq_len)
        logits, aux_loss = self.forward(x_in)

        full_seq_len = seq_len * 2

        per_token = F.cross_entropy(
            logits.reshape(-1, self.tokenizer.n_tokens),
            x.reshape(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none',
        ).view(batch_size, full_seq_len, subseq_len)

        non_pad = (x != self.tokenizer.pad_token).float()
        is_eos = (x == self.tokenizer.eos_token).float() * non_pad
        is_content = non_pad * (1.0 - is_eos)

        frame_idx = torch.arange(full_seq_len, device=x.device)
        # Even frames are mod_a (given), odd are mod_b (generated).
        mel_w = 0.0 if self.target_only_loss else self.mel_loss_weight
        frame_w = torch.where(
            frame_idx % 2 == 0,
            torch.as_tensor(mel_w, device=x.device),
            torch.as_tensor(self.acc_loss_weight, device=x.device),
        )
        w = frame_w.view(1, full_seq_len, 1).expand(batch_size, -1, subseq_len)
        ttw = 1.0 + (self.eos_loss_weight - 1.0) * is_eos
        weighted = per_token * w * ttw * non_pad
        normalizer = (w * ttw * non_pad).sum().clamp_min(1.0)
        ce_loss = weighted.sum() / normalizer

        # Scope the content/eos split to whatever the objective covers,
        # so the logged numbers describe the loss actually being minimised.
        scope = (frame_idx % 2 == 1).float().view(1, full_seq_len, 1) \
            if self.target_only_loss else torch.ones_like(non_pad)
        is_content = is_content * scope
        is_eos_s = is_eos * scope
        content_n = is_content.sum().clamp_min(1.0)
        eos_n = is_eos_s.sum().clamp_min(1.0)
        ce_loss_content = (per_token * is_content).sum() / content_n
        ce_loss_eos = (per_token * is_eos_s).sum() / eos_n

        # Diagnostic: drum-side vs nondrum-side CE. drum CE collapses fast
        # because the prefix gives away the answer; nondrum CE is the
        # useful signal.
        frame_is_drum = (frame_idx % 2 == 0).float()
        frame_is_nondrum = 1.0 - frame_is_drum
        w_drum = frame_is_drum.view(1, full_seq_len, 1).expand_as(non_pad)
        w_nondrum = frame_is_nondrum.view(1, full_seq_len, 1).expand_as(non_pad)
        drum_n = (w_drum * non_pad).sum().clamp_min(1.0)
        nondrum_n = (w_nondrum * non_pad).sum().clamp_min(1.0)
        ce_loss_drum = (per_token * w_drum * non_pad).sum() / drum_n
        ce_loss_nondrum = (per_token * w_nondrum * non_pad).sum() / nondrum_n

        # ----- Brier-style MSE recon loss on drum (suffix-drum slots) -----
        # ||softmax(drum_logits) - one_hot(drum_target)||^2 averaged over
        # non-PAD drum slots. This is the explicit "match the drum you
        # just saw in the prefix" supervision. With the prefix giving
        # full drum visibility, this term should drop quickly alongside
        # CE_drum; tracked as a separate signal for ablation analysis.
        # local_decode returns logits with the batch and frame dims
        # collapsed, so reshape to [B, 2T_full, S, V] before slicing
        # the drum (even-frame) rows.
        # Grad only when the term is actually optimised: softmax + one_hot
        # each materialise a [B, T_full, S, vocab] float tensor, and keeping
        # them in the autograd graph for a discarded term costs real VRAM on
        # a model whose sequences are already 3T long.
        V = self.tokenizer.n_tokens
        with torch.set_grad_enabled(
                torch.is_grad_enabled() and self.recon_weight != 0.0):
            logits_4d = logits.view(batch_size, full_seq_len, subseq_len, V)
            drum_logits = logits_4d[:, 0::2]           # [B, T_full, S, V]
            drum_targets = x[:, 0::2]                  # [B, T_full, S]
            drum_non_pad = (drum_targets != self.tokenizer.pad_token).float()
            drum_probs = F.softmax(drum_logits, dim=-1)
            safe_targets = drum_targets.clamp(min=0, max=V - 1)
            one_hot = F.one_hot(safe_targets, num_classes=V).float()
            mse_per_slot = ((drum_probs - one_hot) ** 2).sum(dim=-1)
            recon_loss = (
                (mse_per_slot * drum_non_pad).sum()
                / drum_non_pad.sum().clamp_min(1.0)
            )

        self._last_ce_loss = ce_loss.detach()
        self._last_ce_loss_content = ce_loss_content.detach()
        self._last_ce_loss_eos = ce_loss_eos.detach()
        self._last_ce_loss_drum = ce_loss_drum.detach()
        self._last_ce_loss_nondrum = ce_loss_nondrum.detach()
        self._last_recon_loss = recon_loss.detach()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
        else:
            aux_loss = ce_loss.new_zeros(())

        # The Brier term is INDEPENDENT of target_only_loss, and is kept on
        # by default. It is not a copy task: under the shift the slot
        # predicting mod_a[k] holds mod_a[k-1], so satisfying it requires
        # RETRIEVING mod_a[k] out of the prefix, indexed by position. That
        # retrieval runs over k_m/v_m on the prefix positions -- the same
        # keys and values a mod_b query reads on the cross path -- so it
        # trains the shared half of exactly the mechanism the conditioning
        # depends on. Set recon_weight=0 to drop it.
        total_loss = (ce_loss
                      + self.recon_weight * recon_loss
                      + self.aux_loss_weight * aux_loss)
        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_ce_loss', self._last_ce_loss)
        self.log('train_ce_loss_content', self._last_ce_loss_content)
        self.log('train_ce_loss_eos', self._last_ce_loss_eos)
        self.log('train_ce_loss_drum', self._last_ce_loss_drum)
        self.log('train_ce_loss_nondrum', self._last_ce_loss_nondrum)
        self.log('train_recon_loss', self._last_recon_loss)
        self.log('train_moe_aux_loss', aux_loss.detach())
        gm, gc = self._gate_means()
        if gc is not None:
            # gate_c is the channel the prefix reaches mod_b through.
            self.log('train_gate_c', gc)
            self.log('train_gate_m', gm)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ce_loss', self._last_ce_loss)
        self.log('val_ce_loss_content', self._last_ce_loss_content)
        self.log('val_ce_loss_eos', self._last_ce_loss_eos)
        self.log('val_ce_loss_drum', self._last_ce_loss_drum)
        self.log('val_ce_loss_nondrum', self._last_ce_loss_nondrum)
        self.log('val_recon_loss', self._last_recon_loss)
        self.log('val_moe_aux_loss', aux_loss.detach())
        gm, gc = self._gate_means()
        if gc is not None:
            # gate_c is the channel the prefix reaches mod_b through.
            self.log('val_gate_c', gc)
            self.log('val_gate_m', gm)
        return loss


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    try:
        import lightning as L
        from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
    except ImportError:
        import pytorch_lightning as L
        from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

    parser = argparse.ArgumentParser(
        description='Train M2CDuetRehearsal (drum prefix + interleaved AR suffix).',
    )
    parser.add_argument('--task', type=str, required=True, choices=sorted(TASKS))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_size', type=str, default='large',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str, default=None)
    parser.add_argument('--mod_a_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--global_num_layers', type=int, default=None)
    parser.add_argument('--mel_loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_loss_weight', type=float, default=1.0)
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--val_check_interval', type=int, default=500,
                        help='steps between val evaluations. On the small '
                             'melchord corpora the val minimum arrives '
                             'within ~1k steps; 500 resolves it with only '
                             'one or two points.')
    parser.add_argument('--preserve_program', action='store_true', default=True)
    parser.add_argument('--hardcode_program', dest='preserve_program',
                        action='store_false')
    parser.add_argument('--wandb_dir', type=str, default='/tmp/wandb')
    parser.add_argument('--ckpt_monitor', '--monitor', dest='monitor',
                        type=str, default='val_loss',
                        help='metric ModelCheckpoint selects on. NAMED '
                             '--ckpt_monitor because torchrun parses the '
                             'multi-GPU command line first and abbreviation-'
                             'matches a bare --monitor against its own '
                             '--monitor-interval / --monitor_interval, '
                             'failing with "ambiguous option" before the '
                             'trainer ever sees it. --monitor still works '
                             'for single-GPU runs, which invoke python '
                             'directly. NOTE what '
                             "val_loss is for THIS variant: CE averaged over "
                             'BOTH streams plus recon_weight * the Brier drum '
                             'term. The drum side is a copy of the prefix, so '
                             'both of those collapse to ~0 early and dilute '
                             'the only quantity the model is evaluated on -- '
                             'nondrum CE. Use val_ce_loss_nondrum to select '
                             'on the conditional task itself. Default is left '
                             'at val_loss so resuming an existing run keeps '
                             'its selection rule.')
    parser.add_argument('--save_top_k', type=int, default=2)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--lr_total_steps', type=int, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)
    parser.add_argument('--eos_loss_weight', type=float, default=1.0)
    parser.add_argument('--ctx_corrupt_prob', type=float, default=0.0,
                        help='prob per frame that a silent run of '
                             'ctx_corrupt_len frames begins in the acc '
                             'INPUT context (targets stay clean)')
    parser.add_argument('--ctx_corrupt_len', type=int, default=8)
    parser.add_argument('--silence_augment_prob', type=float, default=0.0)
    parser.add_argument('--moe_monitor_every_n_steps', type=int, default=0)
    parser.add_argument('--moe_monitor_n_samples', type=int, default=4)
    parser.add_argument('--dump_samples_dir', type=str, default=None)
    parser.add_argument('--dump_samples_n', type=int, default=4)
    parser.add_argument('--dump_samples_every_n_epochs', type=int, default=None)
    parser.add_argument('--max_polyphony', type=int, default=16)
    parser.add_argument('--gate_init_bias', type=float, default=-10.0)
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Weight on the Brier-style MSE drum '
                             'reconstruction loss applied to suffix-drum '
                             'logits. Total loss = CE + recon_weight * '
                             'MSE_drum + aux_loss_weight * aux.')
    parser.add_argument('--prefix_stride2', type=int, default=1,
                        help='1 = count the drum prefix in HALF-FRAMES, the '
                             'same unit the interleaved suffix uses. The '
                             'suffix advances 2 rotary units per musical '
                             'frame and the drum-only prefix advanced 1, so '
                             "drum_k's distance from nondrum_k's query grew "
                             'as k+1 -- the rehearsal signal decayed across '
                             'the sequence. With this on it is a constant 1. '
                             'ON by default: the old behaviour was a unit '
                             'inconsistency, not a trade-off -- stride-2 '
                             'preserves prefix-internal relative time just '
                             'as well (scaled by 2), which is all the '
                             'original segment-wise rationale protected. '
                             'Pass 0 only to reproduce a pre-fix run. '
                             'Existing ckpts are unaffected either way: the '
                             'flag is a buffer, so they decode under the '
                             'scheme they were trained with.')
    parser.add_argument('--suffix_shift1', type=int, default=1,
                        help='1 = shift the interleaved suffix by ONE slot '
                             '(single SOS) instead of two. ON BY DEFAULT: '
                             'under shift-2 the slot predicting mod_b at '
                             'frame k holds mod_b[k-1], and mod_a[k] sits in '
                             "that slot's masked future -- correct teacher "
                             'forcing for symmetric CO-GENERATION, wrong for '
                             'a conditional model where mod_a is given. '
                             'Shift-1 puts mod_a[k] in the query slot, so '
                             'mod_b[k] is conditioned on its own frame of '
                             'mod_a. Pass 0 to reproduce a pre-fix run.')
    parser.add_argument('--target_only_loss', type=int, default=1,
                        help='1 = CE on the mod_b (generated) slots only, '
                             'and the Brier mod_a term logged but not '
                             'optimised. ON BY DEFAULT: mod_a is GIVEN, so '
                             'scoring the model on reproducing it teaches '
                             'nothing and spends capacity copying the '
                             'prefix. Pass 0 for the old joint objective '
                             '(CE over both streams + recon_weight * Brier).')
    parser.add_argument('--gate_target_bias', type=float, default=0.0,
                        help='effective cross-gate bias the warmup ramps '
                             'TO. 0.0 -> sigmoid 0.5, i.e. the prefix fully '
                             'available to the generated stream by the end '
                             'of the ramp. Set equal to --gate_init_bias to '
                             'disable the ramp and rely on gradients alone.')
    parser.add_argument('--gate_warmup_steps', type=int, default=1000,
                        help='steps over which the cross gate opens from '
                             'gate_init_bias to gate_target_bias. The ramp '
                             'is an ADDITIVE offset on the pre-activation, '
                             'so the learned bias trains on top of it and '
                             'step 0 is still the exact warm start. 0 '
                             'disables the ramp (jump straight to target).')
    parser.add_argument('--fresh_schedule', action='store_true', default=False)
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    task = get_task(args.task)
    mod_a_path = args.mod_a_path if args.mod_a_path is not None else task.mod_a_path
    mod_b_path = args.path_to_dataset if args.path_to_dataset is not None else task.mod_b_path

    tag = f'_{args.run_tag}' if args.run_tag else ''
    # The two shift schemes are separate VARIANTS, not settings of one
    # model -- they differ in what the ungated intra path carries, which
    # gate bias is coherent, and how the prefix must be indexed. Name them
    # apart so resolve_best_ckpt (which scans a directory) can never treat
    # one as the other, and so E3 tables can report them as distinct rows.
    #
    #   C1A  shift-2: the slot predicting mod_b[k] holds mod_b[k-1]; the
    #        intra path carries mod_b's own history, so gate -10 is the
    #        coherent warm start. mod_a[k] arrives ONLY via the prefix.
    #   C1B  shift-1: that slot holds mod_a[k] itself; the intra path
    #        carries mod_a and mod_b's history moves to the gated cross
    #        path, so the gate must start open (0.0).
    variant = 'C1B' if args.suffix_shift1 else 'C1A'
    # Anything that deviates from the variant's canonical config gets
    # spelled out, so a non-standard run is never mistaken for the real one.
    dev = ''
    if not args.prefix_stride2:
        dev += '_nops2'
    if not args.target_only_loss:
        dev += '_jointce'
    if args.recon_weight == 0.0:
        dev += '_norecon'
    default_name = (f"m2c_duet_rehearsal_{variant}_{args.model_size}_"
                    f"gnl{gnl}_{task.name}{dev}{tag}_"
                    f"batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  mod_a={task.mod_a_label}  mod_b={task.mod_b_label}')
    print(f'[variant] {variant}  '
          f'({"shift-1" if args.suffix_shift1 else "shift-2"} suffix)')
    print(f'[scheme] prefix_stride2={bool(args.prefix_stride2)} '
          f'suffix_shift1={bool(args.suffix_shift1)} '
          f'target_only_loss={bool(args.target_only_loss)} '
          f'recon_weight={args.recon_weight}')
    print(f'[gate] init_bias={args.gate_init_bias} -> '
          f'target={args.gate_target_bias} over {args.gate_warmup_steps} '
          f'steps (sigmoid {1/(1+2.718281828**-args.gate_init_bias):.5f} -> '
          f'{1/(1+2.718281828**-args.gate_target_bias):.3f})')

    net = M2CDuetRehearsal(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
        preserve_program=args.preserve_program,
        max_lr=args.max_lr,
        lr_total_steps=args.lr_total_steps,
        aux_loss_weight=args.aux_loss_weight,
        silence_augment_prob=args.silence_augment_prob,
        eos_loss_weight=args.eos_loss_weight,
        ctx_corrupt_prob=args.ctx_corrupt_prob,
        ctx_corrupt_len=args.ctx_corrupt_len,
        gate_init_bias=args.gate_init_bias,
        recon_weight=args.recon_weight,
        prefix_stride2=bool(args.prefix_stride2),
        suffix_shift1=bool(args.suffix_shift1),
        target_only_loss=bool(args.target_only_loss),
        gate_target_bias=args.gate_target_bias,
        gate_warmup_steps=args.gate_warmup_steps,
    )
    print(f'Architecture: M2CDuetRehearsal  drum-prefix (T pos, bidirectional) + '
          f'interleaved AR suffix (2T pos) + per-mod Q/K/V/O + cross gate + shared MoE FFN '
          f'({args.moe_num_experts}E, topk={args.moe_topk})')
    print(f'Global depth: {gnl}   gate_init_bias: {args.gate_init_bias}   '
          f'recon_weight: {args.recon_weight}')

    train_set = FramedDataset(mod_b_path, TRAIN_LENGTH,
                              args.batch_size, split='train',
                              mel_path=mod_a_path)
    val_set = FramedDataset(mod_b_path, TRAIN_LENGTH,
                            args.batch_size, split='val',
                            mel_path=mod_a_path)
    train_set_loader = DataLoader(train_set, batch_size=None, num_workers=0)
    val_set_loader = DataLoader(val_set, batch_size=None, num_workers=0)

    global_batch = args.batch_size * n_gpus
    steps_per_epoch = max(1, train_set.valid_song_count // global_batch)
    if args.lr_total_steps is not None:
        implied_epochs = args.lr_total_steps / max(1, steps_per_epoch)
        print(f'[lr] valid_train_songs={train_set.valid_song_count}  '
              f'global_batch={global_batch}  steps_per_epoch={steps_per_epoch}  '
              f'lr_total_steps={args.lr_total_steps}  '
              f'implied_epochs={implied_epochs:.2f}')

    ckpt_dir = args.ckpt_dir or f'ckpt/{model_name}'
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor=args.monitor, save_top_k=args.save_top_k, save_last=True,
        enable_version_counter=False,
        dirpath=ckpt_dir,
        filename=model_name + '.{epoch:02d}.{' + args.monitor + ':.5f}',
    )
    print(f'[select] checkpointing on {args.monitor}')
    if args.monitor == 'val_loss':
        print('[select] NOTE: val_loss here = CE over BOTH streams + '
              f'{args.recon_weight} * Brier(drum). The drum side is copied '
              'from the prefix, so those terms collapse early and the number '
              'is NOT comparable to M2CDuetPrefix\'s val_loss, which is '
              'nondrum CE alone. --monitor val_ce_loss_nondrum selects on '
              'the conditional task itself.')

    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(
            timeout=datetime.timedelta(hours=2),
            find_unused_parameters=True,
        )
    else:
        strategy = 'auto'

    extra_callbacks = []
    if args.moe_monitor_every_n_steps > 0:
        from moe_routing_monitor import MoERoutingMonitor
        extra_callbacks.append(
            MoERoutingMonitor(
                every_n_steps=args.moe_monitor_every_n_steps,
                n_samples=args.moe_monitor_n_samples,
            ).as_callback()
        )
    if args.dump_samples_dir is not None:
        from dump_train_samples import DumpInputSamplesCallback
        extra_callbacks.append(
            DumpInputSamplesCallback(
                out_dir=args.dump_samples_dir,
                n_samples=args.dump_samples_n,
                max_polyphony=args.max_polyphony,
                every_n_epochs=args.dump_samples_every_n_epochs,
            ).as_callback()
        )

    trainer = L.Trainer(
        devices=n_gpus,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=(args.lr_total_steps if args.lr_total_steps is not None else MAX_STEPS),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback] + extra_callbacks,
        val_check_interval=args.val_check_interval,
        limit_val_batches=25,
        check_val_every_n_epoch=None,
        gradient_clip_val=(args.gradient_clip_val if args.gradient_clip_val > 0 else None),
        logger=(
            WandbLogger(
                name=model_name, project='MusicMOE',
                save_dir=args.wandb_dir,
                config={
                    'batch_size': args.batch_size,
                    'model_size': args.model_size,
                    'train_length': TRAIN_LENGTH,
                    'variant': 'm2c_duet_rehearsal',
                    'task': task.name,
                    'mod_a_label': task.mod_a_label,
                    'mod_b_label': task.mod_b_label,
                    'global_num_layers': gnl,
                    'moe_num_experts': args.moe_num_experts,
                    'moe_topk': args.moe_topk,
                    'gate_init_bias': args.gate_init_bias,
                    'recon_weight': args.recon_weight,
                    'run_tag': args.run_tag,
                },
            ) if args.wandb else TensorBoardLogger('tb_logs', name=model_name)
        ),
        num_sanity_val_steps=0 if args.checkpoint_path is not None else 2,
        strategy=strategy,
    )

    ckpt_path_for_resume = None
    if args.checkpoint_path is not None:
        loaded = torch.load(args.checkpoint_path, map_location='cpu',
                             weights_only=False)
        has_lightning_meta = (
            isinstance(loaded, dict)
            and 'pytorch-lightning_version' in loaded
        )
        if has_lightning_meta and not args.fresh_schedule:
            ckpt_path_for_resume = args.checkpoint_path
        else:
            sd = loaded['state_dict'] if isinstance(loaded, dict) and 'state_dict' in loaded else loaded
            # The geometry flags are buffers, so they ride in the state
            # dict -- a warm-start ckpt built under the old scheme would
            # silently reset a --prefix_stride2 / --suffix_shift1 run back
            # to legacy geometry. The CLI wins on a fresh init.
            sd = dict(sd)
            sd.pop('prefix_stride2_flag', None)
            sd.pop('suffix_shift1_flag', None)
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                print(f'[init] {len(missing)} missing (first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected (first few: {unexpected[:3]})')

    # Effective values, not the CLI's: a Lightning RESUME restores the
    # buffers from the ckpt, which is correct (continue the same geometry)
    # but means args no longer describe what is running.
    print(f'[scheme] effective prefix_stride2={net.prefix_stride2} '
          f'suffix_shift1={net.suffix_shift1}')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
    torch.save(net.state_dict(), f'{ckpt_dir}/{model_name}.fin.ckpt')
