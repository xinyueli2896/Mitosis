"""DEPRECATED -- RETIRED MODEL. Do not use for experiments.

M2CDuetBlockAttn is the model the CODE calls "A.2". The PAPER's A.2 is
M2CDuetBlockDiffusion (code name "A.3", cp_transformer_m2c_duet_block_
diffusion.py). Two different models, one string -- check the class, not
the label.

Retired because it collapses at inference: its query slots are trained
ONLY in the both-fully-masked regime, so at decode each slot predicts
its frame conditionally independently of the other, given the past.
That is "equalize by removing" -- neither stream sees the other's
current-frame value. Fixing it is exactly why M2CDuetBlockDiffusion
exists (it trains the slots across noise levels so refinement rounds
give real mutual within-frame conditioning).

Per EXPERIMENTS.md it "plays NO role in any experiment", and nothing
warm-starts from it -- M2CDuetBlockDiffusion inits straight from the
pretrained single-stream CP transformer via
init_pretrained_into_duet_block_diffusion.py.

The file stays only because M2CDuetBlockDiffusion subclasses
M2CDuetBlockAttn, so the class must remain importable.

M2CDuetBlockAttn -- DuetAttn extended with appended next-frame query slots
and a third "frame" attention pass for symmetric same-instant coupling.

Conceptual diff vs M2CIntraCrossAttn:

  * Same per-modality Q/K/V/O projections, same MoE FFN, same per-block
    cross gate. *Adds* a per-block frame gate (gate_f_m, gate_f_c, bias=-10).

  * Three SDPA passes per block instead of two:
       intra : same-modality, causal (== existing)
       cross : other-modality, STRICTLY PAST FRAMES (tightened)
       frame : other-modality, SAME PREDICTION-FRAME, bidirectional (new)

  * Sequence is augmented at the global-stack input with two appended
    "query slots" at the end:
         h_in_full = [shifted_clean of length 2T_full,  mask_m,  mask_c]
    The two appended slots carry learned mask embeddings (mask_m_emb,
    mask_c_emb) and exist to predict the chosen target frame T_query in
    parallel with bidirectional same-frame attention between the two
    slots. T_query is sampled uniformly per batch.

  * Loss = standard AR cross-entropy on the clean shifted positions
    (preserves pretrained behaviour, anchors the gates open gently)
       + lambda_query * cross-entropy on the two query slots vs frame
         T_query's ground-truth tokens (the new joint-prediction signal).

  * Inference (Stage 2): use simple AR decoding off the clean shifted
    positions. The query slots are training-time only for now; Stage 3
    will swap inference to a denoising loop reading off the query slots.

Warm-start (init_pretrained_into_duet_block.py): per-layer remap is
identical to intra-cross-attn. gate_f_m / gate_f_c biases are baked in
by this constructor (-10) so the frame pass starts silent. The new
mask_m_emb / mask_c_emb parameters are zero-initialised; they remain
unused by clean-stream AR loss at step 0, so warm-start equivalence on
the AR term is preserved. The query-stream loss starts from cold init
on these parameters and learns from scratch -- this is the "adaptation
finetune" cost we accept.
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
# Per-layer 3-pass block (intra + cross + frame), 2 gates (gate_c, gate_f)
# ---------------------------------------------------------------------------

class M2CDuetBlockLayer(nn.Module):
    """One transformer block, post-LN, with per-modality Q/K/V/O, three
    key-masked SDPA passes (intra + cross + frame), per-modality cross
    and frame gates, and a shared MoE FFN.

    Three masks built per-call from (clean_len, T_query, device):
      mask_intra : clean queries -> same-mod clean causal;
                    query slots -> same-mod clean with frame < T_query.
      mask_cross : clean queries -> other-mod clean with frame < query frame;
                    query slots -> other-mod clean with frame < T_query.
      mask_frame : clean queries -> other-mod clean SAME prediction-frame;
                    query slots -> the OTHER query slot only.

    Clean tokens never see the query slots (no info leak).
    """

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 moe_num_experts, moe_topk, moe_intermediate_size,
                 dropout=0.0, gate_init_bias=-10.0,
                 moe_modality_bias=False, moe_modality_gates=False):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Per-modality Q/K/V/O projections.
        self.q_m = nn.Linear(hidden_size, hidden_size)
        self.k_m = nn.Linear(hidden_size, hidden_size)
        self.v_m = nn.Linear(hidden_size, hidden_size)
        self.o_m = nn.Linear(hidden_size, hidden_size)
        self.q_c = nn.Linear(hidden_size, hidden_size)
        self.k_c = nn.Linear(hidden_size, hidden_size)
        self.v_c = nn.Linear(hidden_size, hidden_size)
        self.o_c = nn.Linear(hidden_size, hidden_size)

        # Cross gate (existing). Per-modality scalar gate.
        self.gate_m = nn.Linear(hidden_size, 1)
        self.gate_c = nn.Linear(hidden_size, 1)
        # Frame gate (new). Per-modality scalar gate.
        self.gate_fm = nn.Linear(hidden_size, 1)
        self.gate_fc = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            for g in (self.gate_m, self.gate_c, self.gate_fm, self.gate_fc):
                g.weight.zero_()
                g.bias.fill_(gate_init_bias)

        self.ln_attn = nn.LayerNorm(hidden_size)
        self.ln_ffn = nn.LayerNorm(hidden_size)
        self.use_moe = moe_num_experts > 1
        ffn_inter = moe_intermediate_size or intermediate_size
        if self.use_moe:
            self.ffn = SimpleMoEFFN(hidden_size, ffn_inter,
                                     num_experts=moe_num_experts,
                                     topk=moe_topk,
                                     modality_bias=moe_modality_bias,
                                     modality_gates=moe_modality_gates)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ffn_inter),
                nn.GELU(),
                nn.Linear(ffn_inter, hidden_size),
            )
        self.drop = nn.Dropout(dropout)

        # Mask cache keyed by (clean_len, T_query, device).
        self._mask_cache_key = None
        self._mask_intra = None
        self._mask_cross = None
        self._mask_frame = None

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * d)

    def _build_masks(self, clean_len, T_query, device):
        """Build the three [L, L] boolean masks where L = clean_len + 2.

        Positions 0 .. clean_len-1 are clean (alternating m/c).
        Position clean_len is the m-query slot for frame T_query.
        Position clean_len+1 is the c-query slot for frame T_query.
        """
        L = clean_len + 2
        cache_key = (clean_len, T_query, str(device))
        if self._mask_cache_key == cache_key:
            return self._mask_intra, self._mask_cross, self._mask_frame

        pos = torch.arange(L, device=device)
        is_query = pos >= clean_len
        is_clean = ~is_query

        # Modality per position. Clean: pos % 2 (0=m, 1=c).
        # Query: (pos - clean_len) % 2 (so clean_len -> m, clean_len+1 -> c).
        modality = torch.where(
            is_clean,
            pos % 2,
            (pos - clean_len) % 2,
        )

        # Prediction-frame per position.
        # Clean: pos // 2.
        # Query: T_query.
        pred_frame = torch.where(
            is_clean,
            pos // 2,
            torch.full_like(pos, T_query),
        )

        # Broadcast.
        m_p = modality[:, None]
        m_q = modality[None, :]
        same_mod = (m_p == m_q)
        diff_mod = ~same_mod

        f_p = pred_frame[:, None]
        f_q = pred_frame[None, :]
        same_frame = (f_p == f_q)
        strict_past_frame = (f_q < f_p)
        # For query rows (p is_query), "frame(q) < T_query" -- because f_p=T_query.
        # This is identical to strict_past_frame when q is clean, so we can
        # reuse strict_past_frame uniformly for cross/intra "past" tests on q.

        q_clean = is_clean[None, :]
        q_query = is_query[None, :]
        p_clean = is_clean[:, None]
        p_query = is_query[:, None]

        # clean-clean causal (for intra): q <= p AND both clean.
        clean_clean = p_clean & q_clean
        causal_pos = (pos[None, :] <= pos[:, None])

        # mask_intra:
        #   clean p AND clean q AND same_mod AND q <= p
        # OR
        #   query p AND clean q AND same_mod AND frame(q) < T_query
        mask_intra = (
            (clean_clean & same_mod & causal_pos)
            | (p_query & q_clean & same_mod & strict_past_frame)
        )

        # mask_cross:
        #   clean p AND clean q AND diff_mod AND frame(q) < frame(p)
        # OR
        #   query p AND clean q AND diff_mod AND frame(q) < T_query
        # (the two cases collapse since frame(p) = T_query for query rows
        # and frame(p) = pred_frame for clean rows.)
        mask_cross = (
            (clean_clean & diff_mod & strict_past_frame)
            | (p_query & q_clean & diff_mod & strict_past_frame)
        )

        # mask_frame:
        #   clean p AND clean q AND diff_mod AND same prediction-frame
        # OR
        #   query p AND query q AND p != q
        diag = torch.eye(L, dtype=torch.bool, device=device)
        mask_frame = (
            (clean_clean & diff_mod & same_frame)
            | (p_query & q_query & ~diag)
        )

        # Empty-row safeguard: pos 0 and pos 1 have empty mask_cross
        # (no past frames). Add diagonal to mask_cross so SDPA softmax
        # has at least one allowed key per row. Wrong-modality self-attn
        # at init is harmless because gate_c.bias = -10 -> sigmoid ~ 0.
        # (Same trick as intra_cross_attn's _build_masks.)
        mask_cross = mask_cross | diag

        # Sanity: clean tokens must NEVER see query slots (no info leak).
        # By construction p_clean rows with q_query columns are all False
        # in intra/cross/frame above; no extra masking needed.

        self._mask_cache_key = cache_key
        self._mask_intra = mask_intra
        self._mask_cross = mask_cross
        self._mask_frame = mask_frame
        return mask_intra, mask_cross, mask_frame

    def forward(self, h, T_query, cos, sin, clean_len):
        """h: [B, L, H] with L = clean_len + 2. The last 2 positions are
        the query slots (m-query, c-query) for frame T_query.

        Returns (h_out, aux_loss).
        """
        B, L, H = h.shape
        assert L == clean_len + 2

        # Split into per-modality streams.
        # Clean stream: alternating m/c at positions 0..clean_len-1.
        # Query stream: position clean_len (m), clean_len+1 (c).
        h_m_clean = h[:, 0:clean_len:2]    # [B, T_full, H]  (mod-a clean)
        h_c_clean = h[:, 1:clean_len:2]    # [B, T_full, H]  (mod-b clean)
        h_qm = h[:, clean_len:clean_len+1]   # [B, 1, H]  (m-query)
        h_qc = h[:, clean_len+1:clean_len+2] # [B, 1, H]  (c-query)

        # Per-modality Q/K/V build, then re-interleave to flat [B, L, H].
        # Modality-A positions: 0, 2, 4, ..., clean_len-2, clean_len.
        # Modality-B positions: 1, 3, 5, ..., clean_len-1, clean_len+1.
        h_m_all = torch.cat([h_m_clean, h_qm], dim=1)   # [B, T_full+1, H]
        h_c_all = torch.cat([h_c_clean, h_qc], dim=1)   # [B, T_full+1, H]

        q_m = self._split_heads(self.q_m(h_m_all))
        k_m = self._split_heads(self.k_m(h_m_all))
        v_m = self._split_heads(self.v_m(h_m_all))
        q_c = self._split_heads(self.q_c(h_c_all))
        k_c = self._split_heads(self.k_c(h_c_all))
        v_c = self._split_heads(self.v_c(h_c_all))

        # Scatter back into flat [B, h, L, d_k] in the interleaved+appended
        # layout. Mod-A occupies even clean positions then position clean_len;
        # mod-B occupies odd clean positions then position clean_len+1.
        def _scatter(t_m, t_c):
            B_, hd, _, dk = t_m.shape
            out = torch.zeros(B_, hd, L, dk, device=t_m.device, dtype=t_m.dtype)
            out[:, :, 0:clean_len:2] = t_m[:, :, :clean_len // 2]
            out[:, :, 1:clean_len:2] = t_c[:, :, :clean_len // 2]
            out[:, :, clean_len:clean_len+1] = t_m[:, :, clean_len // 2:clean_len // 2 + 1]
            out[:, :, clean_len+1:clean_len+2] = t_c[:, :, clean_len // 2:clean_len // 2 + 1]
            return out

        q = _scatter(q_m, q_c)
        k = _scatter(k_m, k_c)
        v = _scatter(v_m, v_c)

        # Apply RoPE to Q and K. Use position indexing 0..L-1 (token-level).
        cos_L = cos[:, :, :L]
        sin_L = sin[:, :, :L]
        q, k = _apply_rope(q, k, cos_L, sin_L)

        # Three SDPA passes.
        m_intra, m_cross, m_frame = self._build_masks(clean_len, T_query, q.device)
        out_intra = F.scaled_dot_product_attention(q, k, v, attn_mask=m_intra)
        out_cross = F.scaled_dot_product_attention(q, k, v, attn_mask=m_cross)
        # Empty-row guard for mask_frame: clean tokens at frame 0 have only
        # their same-frame partner -- non-empty. Query rows always non-empty
        # (the other query). So no extra guard needed here.
        out_frame = F.scaled_dot_product_attention(q, k, v, attn_mask=m_frame)

        out_intra = self._merge_heads(out_intra)   # [B, L, H]
        out_cross = self._merge_heads(out_cross)
        out_frame = self._merge_heads(out_frame)

        # Gather per-modality slices for the gates and per-mod output proj.
        # mod-A query positions: even clean + position clean_len.
        # mod-B query positions: odd clean + position clean_len+1.
        def _gather_m(t):
            return torch.cat(
                [t[:, 0:clean_len:2], t[:, clean_len:clean_len+1]], dim=1,
            )

        def _gather_c(t):
            return torch.cat(
                [t[:, 1:clean_len:2], t[:, clean_len+1:clean_len+2]], dim=1,
            )

        u_intra_m = _gather_m(out_intra)
        u_cross_m = _gather_m(out_cross)
        u_frame_m = _gather_m(out_frame)
        u_intra_c = _gather_c(out_intra)
        u_cross_c = _gather_c(out_cross)
        u_frame_c = _gather_c(out_frame)

        # Per-modality gates (cross + frame), each a scalar in (0, 1).
        h_m_for_gate = torch.cat([h_m_clean, h_qm], dim=1)
        h_c_for_gate = torch.cat([h_c_clean, h_qc], dim=1)
        g_c_m = torch.sigmoid(self.gate_m(h_m_for_gate))     # [B, T_full+1, 1]
        g_c_c = torch.sigmoid(self.gate_c(h_c_for_gate))
        g_f_m = torch.sigmoid(self.gate_fm(h_m_for_gate))
        g_f_c = torch.sigmoid(self.gate_fc(h_c_for_gate))

        # Cache for diagnostics.
        self._last_gate_c_m = g_c_m.detach()
        self._last_gate_c_c = g_c_c.detach()
        self._last_gate_f_m = g_f_m.detach()
        self._last_gate_f_c = g_f_c.detach()

        o_m = self.o_m(u_intra_m + g_c_m * u_cross_m + g_f_m * u_frame_m)
        o_c = self.o_c(u_intra_c + g_c_c * u_cross_c + g_f_c * u_frame_c)

        # Scatter outputs back to flat [B, L, H], add residual + LN.
        out_flat = torch.zeros_like(h)
        out_flat[:, 0:clean_len:2] = o_m[:, :clean_len // 2]
        out_flat[:, 1:clean_len:2] = o_c[:, :clean_len // 2]
        out_flat[:, clean_len:clean_len+1] = o_m[:, clean_len // 2:clean_len // 2 + 1]
        out_flat[:, clean_len+1:clean_len+2] = o_c[:, clean_len // 2:clean_len // 2 + 1]

        h = self.ln_attn(h + self.drop(out_flat))

        # Shared MoE FFN over the flat L-length sequence.
        if self.use_moe:
            if getattr(self.ffn, 'needs_modality_ids', False):
                # Per-position modality, matching _build_masks: even
                # clean positions and the first query slot are mod_a (0),
                # odd clean positions and the second slot are mod_b (1).
                pos = torch.arange(L, device=h.device)
                mod_ids = torch.where(pos < clean_len, pos % 2,
                                      (pos - clean_len) % 2)
                ffn_out, aux_loss = self.ffn(h, modality_ids=mod_ids)
            else:
                ffn_out, aux_loss = self.ffn(h)
        else:
            ffn_out = self.ffn(h)
            aux_loss = torch.zeros((), device=h.device, dtype=h.dtype)
        h = self.ln_ffn(h + self.drop(ffn_out))

        return h, aux_loss


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class M2CDuetBlockAttn(RoFormerSymbolicTransformer):
    """DuetAttn with appended next-frame query slots and a third 'frame'
    attention pass. See module docstring."""

    def __init__(self, *args, moe_num_experts=4, moe_topk=2,
                 moe_intermediate_size=None, global_num_layers=None,
                 global_dropout=0.0, preserve_program=True,
                 gate_init_bias=-10.0, query_loss_weight=1.0,
                 moe_modality_bias=False, moe_modality_gates=False,
                 **kwargs):
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
        # Drop the inherited single-backbone global stack and top-level
        # gate_m/gate_c (legacy from parent).
        del self.global_roformer
        del self.gate_m
        del self.gate_c

        ffn_inter = moe_intermediate_size or self.intermediate_size
        self.global_layers = nn.ModuleList([
            M2CDuetBlockLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                moe_num_experts=moe_num_experts,
                moe_topk=moe_topk,
                moe_intermediate_size=ffn_inter,
                dropout=global_dropout,
                gate_init_bias=gate_init_bias,
                moe_modality_bias=moe_modality_bias,
                moe_modality_gates=moe_modality_gates,
            )
            for _ in range(self.global_num_layers)
        ])

        # Per-modality SOS offsets (matches intra-cross-attn).
        self.sos_offset_m = nn.Parameter(torch.zeros(self.hidden_size))
        self.sos_offset_c = nn.Parameter(torch.zeros(self.hidden_size))

        # Mask embeddings for the two appended query slots. Zero-init so
        # they don't perturb the AR forward at step 0; their meaningful
        # representation is learned from scratch via the query loss.
        self.mask_m_emb = nn.Parameter(torch.zeros(self.hidden_size))
        self.mask_c_emb = nn.Parameter(torch.zeros(self.hidden_size))

        # Token type embedding zeroed + frozen (same rationale as
        # intra-cross-attn: modality info carried by per-modality QKVO).
        with torch.no_grad():
            self.token_type_embeddings.weight.zero_()
        self.token_type_embeddings.weight.requires_grad = False

        # Loss weight for the query-slot joint prediction term, balanced
        # against the standard AR CE.
        self.query_loss_weight = query_loss_weight

    def _assemble_sos(self, batch_size, device, dtype):
        sos_m = (self.global_sos + self.sos_offset_m).view(1, 1, -1)
        sos_c = (self.global_sos + self.sos_offset_c).view(1, 1, -1)
        sos = torch.cat([sos_m, sos_c], dim=1).expand(batch_size, -1, -1)
        return sos.to(device=device, dtype=dtype)

    def _run_global_stack(self, h, T_query):
        """h: [B, L, H] with L = clean_len + 2 (last 2 are query slots).

        Returns (h_global, aux_loss).
        """
        B, L, H = h.shape
        clean_len = L - 2
        head_dim = H // self.num_attention_heads
        cos, sin = _rope_freqs(L, head_dim, device=h.device, dtype=h.dtype)
        total_aux = torch.zeros((), device=h.device, dtype=h.dtype)
        for layer in self.global_layers:
            h, aux = layer(h, T_query, cos, sin, clean_len)
            total_aux = total_aux + aux
        return h, total_aux / max(len(self.global_layers), 1)

    def forward(self, x, T_query=None):
        """x: [B, 2T_full, subseq_len]  ground-truth flat sequence.
        T_query (optional, int): which frame's tokens the appended query
            slots predict. If None, defaults to T_full - 1 (predict last
            frame). For training a random T_query is sampled in loss().

        Returns:
            ar_logits     : [B, 2T_full, subseq_len, vocab]   standard AR
            query_logits  : [B, 2, subseq_len, vocab]         m_T, c_T
            aux_loss      : scalar
        """
        batch_size, seq_len, subseq_len = x.shape
        assert seq_len % 2 == 0
        T_full = seq_len // 2

        if T_query is None:
            T_query = T_full - 1
        T_query = int(T_query)
        assert 1 <= T_query < T_full, (
            f'T_query={T_query} out of valid range [1, {T_full})'
        )

        # Local encode + token type ids (identical to parent forward).
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

        # Standard shift for clean stream.
        sos = self._assemble_sos(batch_size, h.device, h.dtype)
        h_clean = torch.cat([sos, h[:, :-2]], dim=1)   # [B, 2T_full, H]

        # Append 2 query slots (mask_m, mask_c) at the END.
        mask_m_expand = self.mask_m_emb.view(1, 1, -1).expand(batch_size, 1, -1)
        mask_c_expand = self.mask_c_emb.view(1, 1, -1).expand(batch_size, 1, -1)
        h_full = torch.cat([h_clean, mask_m_expand, mask_c_expand], dim=1)
        # h_full: [B, 2T_full + 2, H]

        # Run the global stack with the 3-pass masks.
        h_global, aux_loss = self._run_global_stack(h_full, T_query=T_query)

        # Split outputs.
        h_clean_global = h_global[:, :seq_len]            # [B, 2T_full, H]
        h_query_global = h_global[:, seq_len:seq_len+2]   # [B, 2, H]

        # AR logits over the clean stream (standard local_decode).
        ar_logits = self.local_decode(h_clean_global, emb)

        # Query logits: local_decode takes the query slot hidden states
        # as seeds and uses the ground-truth emb at frame T_query to
        # teacher-force the inner AR over subseq positions.
        # emb has shape [B*seq_len, subseq_len, H] from local_encode.
        emb_reshape = emb.view(batch_size, seq_len, subseq_len, -1)
        # Frame T_query's mod-a is at position 2*T_query, mod-b at 2*T_query+1.
        emb_query_m = emb_reshape[:, 2 * T_query:2 * T_query + 1]   # [B, 1, subseq_len, H]
        emb_query_c = emb_reshape[:, 2 * T_query + 1:2 * T_query + 2]
        emb_query = torch.cat([emb_query_m, emb_query_c], dim=1)    # [B, 2, subseq_len, H]
        emb_query_flat = emb_query.view(batch_size * 2, subseq_len, -1)
        # local_decode expects h of shape [B_eff, H_total] roughly; mimic
        # the way the parent calls it: pass the query hidden states as h.
        query_logits = self.local_decode(h_query_global, emb_query_flat)

        return ar_logits, query_logits, aux_loss

    def loss(self, x_mel, x_acc, batch_pitch_shift):
        """Override: AR loss on clean stream + joint loss on query slots."""
        # Preprocess + interleave (matches parent's loss).
        x_mel, x_acc = self.preprocess(x_mel, batch_pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape

        stacked = torch.stack([x_mel, x_acc], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)
        T_full = seq_len  # number of frames per modality
        full_seq_len = seq_len * 2  # = 2 * T_full

        # Sample one T_query per batch, uniform in [1, T_full - 1].
        # Use torch RNG for determinism with seed_everything if set.
        if self.training:
            T_query = int(torch.randint(
                low=1, high=T_full, size=(1,), device=x.device,
            ).item())
        else:
            # Eval: deterministic last-frame prediction for comparability.
            T_query = T_full - 1

        ar_logits, query_logits, aux_loss = self.forward(x, T_query=T_query)
        targets_ar = x  # [B, 2T_full, subseq_len]
        # Query targets: frame T_query's m and c.
        targets_query = torch.stack([
            x[:, 2 * T_query],          # [B, subseq_len]   m_{T_query}
            x[:, 2 * T_query + 1],      # [B, subseq_len]   c_{T_query}
        ], dim=1)                       # [B, 2, subseq_len]

        # ----- AR loss (standard CE on every clean position) -----
        per_token_ar = F.cross_entropy(
            ar_logits.reshape(-1, self.tokenizer.n_tokens),
            targets_ar.reshape(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none',
        ).view(batch_size, full_seq_len, subseq_len)

        non_pad_ar = (targets_ar != self.tokenizer.pad_token).float()
        is_eos_ar = (targets_ar == self.tokenizer.eos_token).float() * non_pad_ar
        is_content_ar = non_pad_ar * (1.0 - is_eos_ar)

        # Frame stream weights (mel even, acc odd), as in parent.
        frame_idx = torch.arange(full_seq_len, device=x.device)
        frame_w = torch.where(
            frame_idx % 2 == 0,
            torch.as_tensor(self.mel_loss_weight, device=x.device),
            torch.as_tensor(self.acc_loss_weight, device=x.device),
        )
        w_ar = frame_w.view(1, full_seq_len, 1).expand(batch_size, -1, subseq_len)
        ttw_ar = 1.0 + (self.eos_loss_weight - 1.0) * is_eos_ar
        weighted_ar = per_token_ar * w_ar * ttw_ar * non_pad_ar
        norm_ar = (w_ar * ttw_ar * non_pad_ar).sum().clamp_min(1.0)
        ar_loss = weighted_ar.sum() / norm_ar

        content_n_ar = is_content_ar.sum().clamp_min(1.0)
        eos_n_ar = is_eos_ar.sum().clamp_min(1.0)
        ar_loss_content = (per_token_ar * is_content_ar).sum() / content_n_ar
        ar_loss_eos = (per_token_ar * is_eos_ar).sum() / eos_n_ar

        # ----- Query loss (CE on the 2 appended slots) -----
        per_token_q = F.cross_entropy(
            query_logits.reshape(-1, self.tokenizer.n_tokens),
            targets_query.reshape(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none',
        ).view(batch_size, 2, subseq_len)
        non_pad_q = (targets_query != self.tokenizer.pad_token).float()
        norm_q = non_pad_q.sum().clamp_min(1.0)
        query_loss = (per_token_q * non_pad_q).sum() / norm_q

        # Stash for logging.
        self._last_ar_loss = ar_loss.detach()
        self._last_ar_loss_content = ar_loss_content.detach()
        self._last_ar_loss_eos = ar_loss_eos.detach()
        self._last_query_loss = query_loss.detach()
        self._last_T_query = T_query

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
        else:
            aux_loss = ar_loss.new_zeros(())

        total_loss = (
            ar_loss
            + self.query_loss_weight * query_loss
            + self.aux_loss_weight * aux_loss
        )
        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_ar_loss', self._last_ar_loss)
        self.log('train_ar_loss_content', self._last_ar_loss_content)
        self.log('train_ar_loss_eos', self._last_ar_loss_eos)
        self.log('train_query_loss', self._last_query_loss)
        self.log('train_moe_aux_loss', aux_loss.detach())
        self.log('train_T_query', float(self._last_T_query))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ar_loss', self._last_ar_loss)
        self.log('val_ar_loss_content', self._last_ar_loss_content)
        self.log('val_ar_loss_eos', self._last_ar_loss_eos)
        self.log('val_query_loss', self._last_query_loss)
        self.log('val_moe_aux_loss', aux_loss.detach())
        return loss


# ---------------------------------------------------------------------------
# Training entry point (mirrors intra_cross_attn, adds --query_loss_weight)
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
        description='Train M2CDuetBlockAttn (DuetAttn + appended next-frame '
                    'query slots + 3-pass intra/cross/frame attention).',
    )
    parser.add_argument('--task', type=str, required=True,
                        choices=sorted(TASKS))
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
    parser.add_argument('--preserve_program', action='store_true', default=True)
    parser.add_argument('--hardcode_program', dest='preserve_program',
                        action='store_false')
    parser.add_argument('--wandb_dir', type=str, default='/tmp/wandb')
    parser.add_argument('--save_top_k', type=int, default=2)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--lr_total_steps', type=int, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)
    parser.add_argument('--eos_loss_weight', type=float, default=1.0)
    parser.add_argument('--silence_augment_prob', type=float, default=0.0)
    parser.add_argument('--moe_monitor_every_n_steps', type=int, default=0)
    parser.add_argument('--moe_monitor_n_samples', type=int, default=4)
    parser.add_argument('--dump_samples_dir', type=str, default=None)
    parser.add_argument('--dump_samples_n', type=int, default=4)
    parser.add_argument('--dump_samples_every_n_epochs', type=int, default=None)
    parser.add_argument('--max_polyphony', type=int, default=16)
    parser.add_argument('--gate_init_bias', type=float, default=-10.0)
    parser.add_argument('--query_loss_weight', type=float, default=1.0,
                        help='Weight on the query-slot joint prediction CE '
                             'term. Total loss = AR_CE + lambda_query * Q_CE '
                             '+ lambda_aux * aux.')
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
    default_name = (f"m2c_duet_block_v1.0_{args.model_size}_"
                    f"gnl{gnl}_{task.name}{tag}_"
                    f"batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  mod_a={task.mod_a_label}  mod_b={task.mod_b_label}')

    net = M2CDuetBlockAttn(
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
        gate_init_bias=args.gate_init_bias,
        query_loss_weight=args.query_loss_weight,
    )
    print(f'Architecture: M2CDuetBlockAttn  3-pass (intra/cross/frame) + '
          f'2 gates (cross, frame) + appended query slots + shared MoE FFN '
          f'({args.moe_num_experts}E, topk={args.moe_topk})')
    print(f'Global depth: {gnl}   gate_init_bias: {args.gate_init_bias}   '
          f'query_loss_weight: {args.query_loss_weight}')

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
        monitor='val_loss', save_top_k=args.save_top_k, save_last=True,
        enable_version_counter=False,
        dirpath=ckpt_dir,
        filename=model_name + '.{epoch:02d}.{val_loss:.5f}',
    )

    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        # find_unused_parameters=True is required because the appended
        # query slots + MoE router routing can leave some params with no
        # gradient on a given step (e.g. an expert not selected this batch,
        # or a gate whose contribution flows entirely through the silent
        # zero of g*u_cross at init). The autograd-graph scan is a small
        # per-step overhead but necessary for DDP correctness here.
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
        val_check_interval=500,
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
                    'variant': 'm2c_duet_block',
                    'task': task.name,
                    'mod_a_label': task.mod_a_label,
                    'mod_b_label': task.mod_b_label,
                    'global_num_layers': gnl,
                    'moe_num_experts': args.moe_num_experts,
                    'moe_topk': args.moe_topk,
                    'gate_init_bias': args.gate_init_bias,
                    'query_loss_weight': args.query_loss_weight,
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
            print(f'[resume] full Lightning ckpt at {args.checkpoint_path}')
            ckpt_path_for_resume = args.checkpoint_path
        else:
            if has_lightning_meta and args.fresh_schedule:
                print(f'[fresh-schedule] loading model weights only from '
                       f'{args.checkpoint_path}')
            else:
                print(f'[init] bare warm-start ckpt at {args.checkpoint_path}')
            sd = loaded['state_dict'] if isinstance(loaded, dict) and 'state_dict' in loaded else loaded
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                print(f'[init] {len(missing)} missing keys (first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected keys (first few: {unexpected[:3]})')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
    torch.save(net.state_dict(), f'{ckpt_dir}/{model_name}.fin.ckpt')
