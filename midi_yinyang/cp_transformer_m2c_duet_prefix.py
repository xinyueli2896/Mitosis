"""M2CDuetPrefix -- prefix-LM variant: all drum first (bidirectional within),
then all nondrum (causal within + full attention back to drum).

Sequence layout in the global stack (length 2T):

    [drum_0, drum_1, ..., drum_{T-1},  sos_nondrum, nondrum_0, ..., nondrum_{T-2}]
     └────── drum block (T) ──────┘    └──────── nondrum block (T) ─────────┘
     bidirectional within drum         strict causal within nondrum
     no peeking at nondrum             full attention to drum

Predictions are read off ONLY the nondrum block positions; position T (sos_nondrum)
predicts nondrum_0, position T+t (= h_in[nondrum_{t-1}]) predicts nondrum_t.
The drum block contributes via its K/V projections (carries drum context to
nondrum queries) but is never asked to predict anything -- it's pure
conditioning.

Two SDPA passes per block (matches DuetAttn lineage so the per-block gate
can keep the warm-start story honest):
    intra : same-modality.  Drum-drum BIDIRECTIONAL.  Nondrum-nondrum causal.
    cross : nondrum query attends to ALL drum keys (one direction only).
            Drum query has no cross keys -- empty rows guarded by diagonal.

Per-modality cross gate `gate_c` (nondrum side) controls how much the
nondrum block reads the drum block. At init bias = -10 -> sigmoid ~ 0,
the nondrum block runs in isolation as standard causal AR. Training
opens the gate to read drum.

`gate_m` (drum side) exists for symmetry with the DuetAttn layer
structure but is functionally inert here -- drum's cross stream is
empty (it can't see nondrum), so g_m * u_cross_m has no useful signal.

Inference: drum→nondrum conditional. Given drum prompt (whole sequence
or partial), greedy/sampled AR decode of the nondrum block. Not
streaming in the drum→nondrum sense (drum prefix must be committed
before nondrum decoding starts), but trivially KV-cacheable across
nondrum positions.

Loss = CE on the T nondrum predictions + lambda_aux * MoE aux.
No silence augmentation. No co-generation training (this variant is
one-way drum→nondrum by design; reverse-direction needs a separate
model or a swap-augmented training scheme).
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
# Per-layer block: 2 SDPA passes with prefix-LM masks + per-modality gate
# ---------------------------------------------------------------------------

class M2CDuetPrefixLayer(nn.Module):
    """One transformer block, post-LN, with per-modality Q/K/V/O, two
    masked SDPA passes (intra + cross) under the prefix-LM mask scheme,
    per-modality cross gate, and a shared MoE FFN.

    Input layout assumed:
        h = [drum_0, ..., drum_{T-1}, sos_nondrum, nondrum_0, ..., nondrum_{T-2}]
        positions [0, T)  -> drum block        (mod-a / "m" projections)
        positions [T, 2T) -> nondrum block    (mod-b / "c" projections)
    """

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 moe_num_experts, moe_topk, moe_intermediate_size,
                 dropout=0.0, gate_init_bias=-10.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Per-modality Q/K/V/O projections. m = drum, c = nondrum.
        self.q_m = nn.Linear(hidden_size, hidden_size)
        self.k_m = nn.Linear(hidden_size, hidden_size)
        self.v_m = nn.Linear(hidden_size, hidden_size)
        self.o_m = nn.Linear(hidden_size, hidden_size)
        self.q_c = nn.Linear(hidden_size, hidden_size)
        self.k_c = nn.Linear(hidden_size, hidden_size)
        self.v_c = nn.Linear(hidden_size, hidden_size)
        self.o_c = nn.Linear(hidden_size, hidden_size)

        # Cross gate (only gate_c is functionally meaningful -- it controls
        # nondrum's read of drum). gate_m is here for symmetry; its
        # u_cross_m is essentially self-attn through the safeguard diagonal
        # and is silent at init.
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

        # Mask cache keyed by (seq_len, T, device).
        self._mask_cache_key = None
        self._mask_intra = None
        self._mask_cross = None

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * d)

    def _build_masks(self, seq_len, T, device):
        """Build the prefix-LM intra/cross masks.

        Layout: positions [0, T) drum (mod-a), [T, seq_len) nondrum (mod-b).
        Typical seq_len == 2T but we keep them separate for safety.
        """
        cache_key = (seq_len, T, str(device))
        if self._mask_cache_key == cache_key:
            return self._mask_intra, self._mask_cross

        pos = torch.arange(seq_len, device=device)
        is_drum_p = (pos < T)[:, None]    # query rows
        is_drum_q = (pos < T)[None, :]    # key columns
        is_nondrum_p = ~is_drum_p
        is_nondrum_q = ~is_drum_q
        causal_pos = pos[None, :] <= pos[:, None]   # q <= p

        # mask_intra:
        #   drum query <-> drum key      : bidirectional (all-to-all within drum)
        #   nondrum query -> nondrum key : causal (q <= p)
        mask_intra = (
            (is_drum_p & is_drum_q)
            | (is_nondrum_p & is_nondrum_q & causal_pos)
        )

        # mask_cross:
        #   nondrum query -> drum key   : ALL allowed (full read of prefix)
        #   drum query -> nondrum key   : BLOCKED (no peeking at suffix)
        mask_cross = is_nondrum_p & is_drum_q

        # Empty-row safeguard for mask_cross: drum query rows are empty
        # (drum can't see nondrum). Add diagonal so SDPA softmax has at
        # least one allowed key per row. Wrong-modality self-attn at init
        # is harmless because gate_m.bias = -10 -> sigmoid ~ 0.
        diag = torch.eye(seq_len, dtype=torch.bool, device=device)
        mask_cross = mask_cross | diag

        self._mask_cache_key = cache_key
        self._mask_intra = mask_intra
        self._mask_cross = mask_cross
        return mask_intra, mask_cross

    def forward(self, h, T, cos, sin):
        """h: [B, seq_len, H] in prefix layout, seq_len typically = 2T.

        Returns (h_out, aux_loss).
        """
        B, seq_len, H = h.shape

        # Per-modality slices.
        h_d = h[:, :T]          # drum block
        h_n = h[:, T:]          # nondrum block (already SOS-shifted by caller)

        # Per-modality Q/K/V projections.
        q_d = self._split_heads(self.q_m(h_d))
        k_d = self._split_heads(self.k_m(h_d))
        v_d = self._split_heads(self.v_m(h_d))
        q_n = self._split_heads(self.q_c(h_n))
        k_n = self._split_heads(self.k_c(h_n))
        v_n = self._split_heads(self.v_c(h_n))

        # Concat to flat layout [B, h, seq_len, d_k].
        q = torch.cat([q_d, q_n], dim=2)
        k = torch.cat([k_d, k_n], dim=2)
        v = torch.cat([v_d, v_n], dim=2)

        # RoPE on Q, K. Positions 0..seq_len-1 (token-level).
        cos_t = cos[:, :, :seq_len]
        sin_t = sin[:, :, :seq_len]
        q, k = _apply_rope(q, k, cos_t, sin_t)

        # Build masks and run the two SDPA passes.
        mask_intra, mask_cross = self._build_masks(seq_len, T, q.device)
        out_intra = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_intra)
        out_cross = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_cross)
        out_intra = self._merge_heads(out_intra)   # [B, seq_len, H]
        out_cross = self._merge_heads(out_cross)

        # Slice per modality.
        u_intra_d = out_intra[:, :T]
        u_intra_n = out_intra[:, T:]
        u_cross_d = out_cross[:, :T]   # essentially self-attn via safeguard
        u_cross_n = out_cross[:, T:]   # nondrum reading drum -- the useful one

        # Per-modality cross gate.
        g_d = torch.sigmoid(self.gate_m(h_d))
        g_n = torch.sigmoid(self.gate_c(h_n))
        self._last_gate_d = g_d.detach()
        self._last_gate_n = g_n.detach()

        o_d = self.o_m(u_intra_d + g_d * u_cross_d)
        o_n = self.o_c(u_intra_n + g_n * u_cross_n)

        # Reassemble + residual + LN.
        o = torch.cat([o_d, o_n], dim=1)
        h = self.ln_attn(h + self.drop(o))

        # Shared MoE FFN over the flat sequence.
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

class M2CDuetPrefix(RoFormerSymbolicTransformer):
    """Prefix-LM variant: drum block (bidirectional) then nondrum block
    (causal + reads all drum). One-way drum->nondrum. See module docstring.
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
            M2CDuetPrefixLayer(
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

        # SOS offsets per modality, same pattern as DuetAttn / DuetBlock.
        # sos_offset_c is the only one that matters (used as the SOS token
        # at the boundary between drum prefix and nondrum suffix).
        # sos_offset_m is kept for state-dict compatibility with siblings.
        self.sos_offset_m = nn.Parameter(torch.zeros(self.hidden_size))
        self.sos_offset_c = nn.Parameter(torch.zeros(self.hidden_size))

        # Modality info carried by per-modality QKVO; token type embedding
        # is redundant, freeze it as a no-op.
        with torch.no_grad():
            self.token_type_embeddings.weight.zero_()
        self.token_type_embeddings.weight.requires_grad = False

    def _run_global_stack(self, h_prefix, T):
        """h_prefix: [B, 2T, H] in prefix layout (drum block then shifted
        nondrum block). Returns (h_global, aux_loss).
        """
        B, seq_len, H = h_prefix.shape
        head_dim = H // self.num_attention_heads
        cos, sin = _rope_freqs(seq_len, head_dim, device=h_prefix.device,
                                dtype=h_prefix.dtype)
        total_aux = torch.zeros((), device=h_prefix.device, dtype=h_prefix.dtype)
        for layer in self.global_layers:
            h_prefix, aux = layer(h_prefix, T, cos, sin)
            total_aux = total_aux + aux
        return h_prefix, total_aux / max(len(self.global_layers), 1)

    def forward(self, x):
        """x: [B, 2T_full, subseq_len]  interleaved [drum_0, nondrum_0, ...].

        Returns:
            logits     : [B, T_full, subseq_len, vocab]   nondrum predictions
            aux_loss   : scalar
        """
        batch_size, seq_len, subseq_len = x.shape
        assert seq_len % 2 == 0
        T_full = seq_len // 2

        # Token type ids for the local encoder (frame-level even=drum, odd=nondrum).
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
        # h is in interleaved order: h[:, 0::2] = drum frames, h[:, 1::2] = nondrum.

        h_drum = h[:, 0::2]      # [B, T_full, H]
        h_nondrum = h[:, 1::2]   # [B, T_full, H]

        # Build SOS for the nondrum block. global_sos + sos_offset_c gives a
        # [H] vector; expand to [B, 1, H] and use as the first position of
        # the nondrum suffix.
        sos_n = (self.global_sos + self.sos_offset_c).view(1, 1, -1)
        sos_n = sos_n.expand(batch_size, 1, -1).to(
            device=h.device, dtype=h.dtype,
        )

        # Shift the nondrum block by 1 (SOS at front, drop last).
        h_nondrum_shifted = torch.cat([sos_n, h_nondrum[:, :-1]], dim=1)
        # Concat to prefix layout.
        h_prefix = torch.cat([h_drum, h_nondrum_shifted], dim=1)
        # h_prefix: [B, 2T_full, H]

        h_global, aux_loss = self._run_global_stack(h_prefix, T=T_full)

        # Extract nondrum hidden states (predict the original nondrum frames).
        h_nondrum_global = h_global[:, T_full:]   # [B, T_full, H]

        # Local decode using the nondrum frame embeddings as the AR
        # teacher-forcing context.
        emb_reshape = emb.view(batch_size, seq_len, subseq_len, -1)
        emb_nondrum = emb_reshape[:, 1::2]       # [B, T_full, subseq_len, H]
        emb_nondrum_flat = emb_nondrum.reshape(
            batch_size * T_full, subseq_len, -1,
        )
        logits = self.local_decode(h_nondrum_global, emb_nondrum_flat)
        return logits, aux_loss

    def loss(self, x_mel, x_acc, batch_pitch_shift):
        """CE on nondrum positions only + MoE aux."""
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

        # Targets: nondrum frames only.
        targets_nondrum = x[:, 1::2]   # [B, T_full, subseq_len]

        per_token = F.cross_entropy(
            logits.reshape(-1, self.tokenizer.n_tokens),
            targets_nondrum.reshape(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none',
        ).view(batch_size, seq_len, subseq_len)

        non_pad = (targets_nondrum != self.tokenizer.pad_token).float()
        is_eos = (targets_nondrum == self.tokenizer.eos_token).float() * non_pad
        is_content = non_pad * (1.0 - is_eos)

        token_type_weight = 1.0 + (self.eos_loss_weight - 1.0) * is_eos
        weighted = per_token * token_type_weight * non_pad
        normalizer = (token_type_weight * non_pad).sum().clamp_min(1.0)
        ce_loss = weighted.sum() / normalizer

        content_n = is_content.sum().clamp_min(1.0)
        eos_n = is_eos.sum().clamp_min(1.0)
        ce_loss_content = (per_token * is_content).sum() / content_n
        ce_loss_eos = (per_token * is_eos).sum() / eos_n

        self._last_ce_loss = ce_loss.detach()
        self._last_ce_loss_content = ce_loss_content.detach()
        self._last_ce_loss_eos = ce_loss_eos.detach()

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
        self.log('train_moe_aux_loss', aux_loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ce_loss', self._last_ce_loss)
        self.log('val_ce_loss_content', self._last_ce_loss_content)
        self.log('val_ce_loss_eos', self._last_ce_loss_eos)
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
        description='Train M2CDuetPrefix (prefix-LM: drum block + causal '
                    'nondrum reading drum, one-way drum->nondrum).',
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
    parser.add_argument('--val_check_interval', type=int, default=500,
                        help='steps between val evaluations. On the small '
                             'melchord corpora the val minimum arrives '
                             'within ~1k steps; 500 resolves it with only '
                             'one or two points.')
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
    default_name = (f"m2c_duet_prefix_v1.0_{args.model_size}_"
                    f"gnl{gnl}_{task.name}{tag}_"
                    f"batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  mod_a={task.mod_a_label}  mod_b={task.mod_b_label}')

    net = M2CDuetPrefix(
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
    )
    print(f'Architecture: M2CDuetPrefix  prefix-LM (drum bidirectional + '
          f'causal nondrum reading drum) + per-mod cross gate + shared MoE FFN '
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
        # find_unused_parameters=True for the same reason as DuetBlock:
        # gate_m's u_cross_m is largely inert (cross is empty for drum
        # queries), so drum-side cross gate sometimes has zero grad.
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
                    'variant': 'm2c_duet_prefix',
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
