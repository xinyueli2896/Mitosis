"""M2CDuetRehearsal -- drum-prefix rehearsal context + interleaved AR suffix.

Replaces the retired M2CIntraCrossAttnRecon (#3) with a genuine
loss/architecture conditioning baseline. The "rehearsal" is implemented
architecturally: the entire drum stream is prepended as a bidirectional
context block, then the model runs standard DuetAttn-style interleaved
AR on the suffix with full visibility into the prefix.

Sequence layout (length 3T):

  [ drum_0, drum_1, ..., drum_{T-1},   sos_m, sos_c, drum_0, nondrum_0, ..., drum_{T-2}, nondrum_{T-2} ]
   └──── drum prefix (T pos) ────┘   └────────────── shifted interleaved suffix (2T pos) ──────────────┘
   bidirectional within prefix       standard DuetAttn shifted causal AR
   invisible to nothing in suffix    sees all prefix + causal-within-suffix
   no targets / no loss              CE targets = the original interleaved x

The prefix is the same drum content as appears at the suffix's drum
slots; it's just made available before AR begins so nondrum predictions
in the suffix can attend to FUTURE drum content (not just past drum).
That's the rehearsal: the model knows what drum is going to be when it
predicts nondrum.

Loss = standard CE on the entire 2T-length suffix. Drum-side CE
collapses fast because the model can trivially copy drum_k from the
prefix to the suffix's drum slot. That's expected. The useful signal
lives in the nondrum CE -- nondrum_k's prediction now sees ALL drum
(via the prefix) instead of only PAST drum (DuetAttn's behaviour).

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

Inference: deferred. Generation needs a custom loop that maintains a
prefix buffer of committed drum frames separately from the suffix's
shift-trick buffer. See IMPLEMENTATION_REPORT.md outstanding-work
table.
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

    def forward(self, h, T, cos, sin):
        """h: [B, 3T, H]. Returns (h_out, aux_loss)."""
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

        # RoPE on Q, K.
        cos_t = cos[:, :, :L]
        sin_t = sin[:, :, :L]
        q, k = _apply_rope(q, k, cos_t, sin_t)

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
        g_m = torch.sigmoid(self.gate_m(h_m_all))
        g_c = torch.sigmoid(self.gate_c(h_c_all))
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
                 gate_init_bias=-10.0, **kwargs):
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

        # Modality info carried by per-modality QKVO; freeze token type
        # embedding as a no-op.
        with torch.no_grad():
            self.token_type_embeddings.weight.zero_()
        self.token_type_embeddings.weight.requires_grad = False

    def _assemble_sos(self, batch_size, device, dtype):
        sos_m = (self.global_sos + self.sos_offset_m).view(1, 1, -1)
        sos_c = (self.global_sos + self.sos_offset_c).view(1, 1, -1)
        sos = torch.cat([sos_m, sos_c], dim=1).expand(batch_size, -1, -1)
        return sos.to(device=device, dtype=dtype)

    def _run_global_stack(self, h_full, T):
        """h_full: [B, 3T, H] prefix + shifted suffix. Returns (h_global, aux)."""
        B, L, H = h_full.shape
        assert L == 3 * T
        head_dim = H // self.num_attention_heads
        cos, sin = _rope_freqs(L, head_dim, device=h_full.device, dtype=h_full.dtype)
        total_aux = torch.zeros((), device=h_full.device, dtype=h_full.dtype)
        for layer in self.global_layers:
            h_full, aux = layer(h_full, T, cos, sin)
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

        # Drum prefix: T_full positions, no shift.
        h_drum = h[:, 0::2]                          # [B, T_full, H]
        # Suffix: standard DuetAttn shifted interleaved.
        sos = self._assemble_sos(batch_size, h.device, h.dtype)
        h_suffix = torch.cat([sos, h[:, :-2]], dim=1)   # [B, 2T_full, H]

        # Concat.
        h_full = torch.cat([h_drum, h_suffix], dim=1)   # [B, 3T_full, H]

        h_global, aux_loss = self._run_global_stack(h_full, T=T_full)

        # Read off suffix hiddens; local_decode against the original emb.
        h_suffix_global = h_global[:, T_full:T_full + seq_len]   # [B, 2T_full, H]
        logits = self.local_decode(h_suffix_global, emb)
        return logits, aux_loss

    def loss(self, x_mel, x_acc, batch_pitch_shift):
        """Standard CE on the 2T-length suffix. Same loss structure as
        DuetAttn (#2); the prefix contributes via attention only, not via
        the loss."""
        x_mel, x_acc = self.preprocess(x_mel, batch_pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape

        stacked = torch.stack([x_mel, x_acc], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)
        logits, aux_loss = self.forward(x)

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
        frame_w = torch.where(
            frame_idx % 2 == 0,
            torch.as_tensor(self.mel_loss_weight, device=x.device),
            torch.as_tensor(self.acc_loss_weight, device=x.device),
        )
        w = frame_w.view(1, full_seq_len, 1).expand(batch_size, -1, subseq_len)
        ttw = 1.0 + (self.eos_loss_weight - 1.0) * is_eos
        weighted = per_token * w * ttw * non_pad
        normalizer = (w * ttw * non_pad).sum().clamp_min(1.0)
        ce_loss = weighted.sum() / normalizer

        content_n = is_content.sum().clamp_min(1.0)
        eos_n = is_eos.sum().clamp_min(1.0)
        ce_loss_content = (per_token * is_content).sum() / content_n
        ce_loss_eos = (per_token * is_eos).sum() / eos_n

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

        self._last_ce_loss = ce_loss.detach()
        self._last_ce_loss_content = ce_loss_content.detach()
        self._last_ce_loss_eos = ce_loss_eos.detach()
        self._last_ce_loss_drum = ce_loss_drum.detach()
        self._last_ce_loss_nondrum = ce_loss_nondrum.detach()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
        else:
            aux_loss = ce_loss.new_zeros(())

        total_loss = ce_loss + self.aux_loss_weight * aux_loss
        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_ce_loss', self._last_ce_loss)
        self.log('train_ce_loss_content', self._last_ce_loss_content)
        self.log('train_ce_loss_eos', self._last_ce_loss_eos)
        self.log('train_ce_loss_drum', self._last_ce_loss_drum)
        self.log('train_ce_loss_nondrum', self._last_ce_loss_nondrum)
        self.log('train_moe_aux_loss', aux_loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ce_loss', self._last_ce_loss)
        self.log('val_ce_loss_content', self._last_ce_loss_content)
        self.log('val_ce_loss_eos', self._last_ce_loss_eos)
        self.log('val_ce_loss_drum', self._last_ce_loss_drum)
        self.log('val_ce_loss_nondrum', self._last_ce_loss_nondrum)
        self.log('val_moe_aux_loss', aux_loss.detach())
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
    default_name = (f"m2c_duet_rehearsal_v1.0_{args.model_size}_"
                    f"gnl{gnl}_{task.name}{tag}_"
                    f"batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  mod_a={task.mod_a_label}  mod_b={task.mod_b_label}')

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
        gate_init_bias=args.gate_init_bias,
    )
    print(f'Architecture: M2CDuetRehearsal  drum-prefix (T pos, bidirectional) + '
          f'interleaved AR suffix (2T pos) + per-mod Q/K/V/O + cross gate + shared MoE FFN '
          f'({args.moe_num_experts}E, topk={args.moe_topk})')
    print(f'Global depth: {gnl}   gate_init_bias: {args.gate_init_bias}')

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
                    'variant': 'm2c_duet_rehearsal',
                    'task': task.name,
                    'mod_a_label': task.mod_a_label,
                    'mod_b_label': task.mod_b_label,
                    'global_num_layers': gnl,
                    'moe_num_experts': args.moe_num_experts,
                    'moe_topk': args.moe_topk,
                    'gate_init_bias': args.gate_init_bias,
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
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                print(f'[init] {len(missing)} missing (first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected (first few: {unexpected[:3]})')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
    torch.save(net.state_dict(), f'{ckpt_dir}/{model_name}.fin.ckpt')
