"""M2CIntraCrossAttn -- per-modality Q/K/V/O + intra/cross-stream split with
per-block gated cross-stream mix + shared MoE FFN.

Replaces M2CJointAttn's single joint causal attention (which co-normalizes
intra- and cross-stream contributions in one softmax) with two separately
normalized attention passes per block:

    Pass M: queries attend ONLY to mel keys (causal & key_is_mel)
            -> u_mm (mel query, mel keys)
            -> u_cm (chord query, mel keys)
    Pass C: queries attend ONLY to chord keys (causal & key_is_chord)
            -> u_mc (mel query, chord keys)
            -> u_cc (chord query, chord keys)

Then per-modality gated mix (intra unconditional, cross gated):

    o_m = W_O^m ( u_mm + sigmoid(gate_m(m)) * u_mc )
    o_c = W_O^c ( u_cc + sigmoid(gate_c(c)) * u_cm )

`gate_m`, `gate_c` are per-block nn.Linear(H, 1). At init bias = -10
(sigmoid ~ 4.5e-5), so each modality reproduces the pretrained block's
behaviour in isolation at step 0 (warm-start equivalence). During
training, the gates learn to open as cross-stream conditioning becomes
useful.

Compute trade-off vs jointattn: ~2x SDPA per block, but same K/V cache
at inference (the two passes differ only by mask). Parameter delta is
trivial (~2*(H+1) per layer).

Warm-start via init_pretrained_into_intra_cross_attn.py: same per-layer
remap as the jointattn init (Wq -> q_m AND q_c, ditto K/V/O; dense FFN
replicated to all K MoE experts; post-LN copies). New `gate_m`, `gate_c`
biases are baked in by the constructor and not overridden by the init.
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
# Per-layer intra/cross-attention block
# ---------------------------------------------------------------------------

class M2CIntraCrossAttnLayer(nn.Module):
    """One transformer block, post-LN, with per-modality Q/K/V/O, two
    key-masked SDPA passes (intra + cross), per-block gated cross-stream
    mix, and a shared MoE FFN.

    Per-modality parameters: q_m/q_c, k_m/k_c, v_m/v_c, o_m/o_c, gate_m, gate_c.
    Shared parameters: ln_attn, ln_ffn, ffn (MoE or dense).
    """

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 moe_num_experts, moe_topk, moe_intermediate_size,
                 dropout=0.0, gate_init_bias=-10.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Per-modality Q/K/V/O projections (untied between mel and chord).
        self.q_m = nn.Linear(hidden_size, hidden_size)
        self.k_m = nn.Linear(hidden_size, hidden_size)
        self.v_m = nn.Linear(hidden_size, hidden_size)
        self.o_m = nn.Linear(hidden_size, hidden_size)
        self.q_c = nn.Linear(hidden_size, hidden_size)
        self.k_c = nn.Linear(hidden_size, hidden_size)
        self.v_c = nn.Linear(hidden_size, hidden_size)
        self.o_c = nn.Linear(hidden_size, hidden_size)

        # Per-block gates over cross-stream contribution. bias=-10 ->
        # sigmoid ~ 4.5e-5 at init, so the cross term contributes ~0 at
        # step 0 and each modality reproduces pretrained behaviour in
        # isolation. Weight is zeroed so the gate value at init is exactly
        # sigmoid(bias) regardless of input.
        self.gate_m = nn.Linear(hidden_size, 1)
        self.gate_c = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.gate_m.weight.zero_()
            self.gate_c.weight.zero_()
            self.gate_m.bias.fill_(gate_init_bias)
            self.gate_c.bias.fill_(gate_init_bias)

        # Shared post-LN modules and shared (MoE or dense) FFN.
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

        # Cached masks keyed by (two_T, device); built lazily on first call.
        self._mask_cache_key = None
        self._mask_M = None
        self._mask_C = None

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * d)

    def _build_masks(self, two_T, device):
        cache_key = (two_T, str(device))
        if self._mask_cache_key == cache_key:
            return self._mask_M, self._mask_C

        pos = torch.arange(two_T, device=device)
        # Standard causal: query i attends to key j if j <= i.
        causal = pos[None, :] <= pos[:, None]                        # [2T, 2T]
        mel_keys = (pos % 2 == 0)
        chord_keys = (pos % 2 == 1)

        mask_M = causal & mel_keys[None, :]
        mask_C = causal & chord_keys[None, :]

        # Empty-row safeguard: under mask_C, the very first mel query (pos 0)
        # has no prior chord keys, so its row is all-False -> softmax NaN.
        # Add the diagonal to both masks so every row has at least one
        # allowed key. Wrong-modality self attendance at init is harmless
        # because gate.bias = -10 means the cross term is multiplied by
        # sigmoid(-10) ~ 0. Matches the parent's _global_interaction
        # safeguard at cp_transformer_m2c_moe.py:594-596.
        diag = torch.eye(two_T, dtype=torch.bool, device=device)
        mask_M = mask_M | diag
        mask_C = mask_C | diag

        self._mask_cache_key = cache_key
        self._mask_M = mask_M
        self._mask_C = mask_C
        return mask_M, mask_C

    def forward(self, m, c, cos, sin):
        """m, c: [B, T, H]. cos, sin: RoPE buffers covering 2T positions.

        Returns (m_out, c_out, aux_loss)."""
        B, T, H = m.shape
        assert c.shape == m.shape
        two_T = 2 * T

        # 1. Per-modality Q/K/V projection.
        q_m = self._split_heads(self.q_m(m))
        k_m = self._split_heads(self.k_m(m))
        v_m = self._split_heads(self.v_m(m))
        q_c = self._split_heads(self.q_c(c))
        k_c = self._split_heads(self.k_c(c))
        v_c = self._split_heads(self.v_c(c))

        # 2. Interleave Q/K/V into 2T sequence (even=mel, odd=chord).
        def interleave(a_m, a_c):
            return torch.stack([a_m, a_c], dim=3).reshape(
                B, self.num_heads, two_T, self.head_dim
            )
        q = interleave(q_m, q_c)
        k = interleave(k_m, k_c)
        v = interleave(v_m, v_c)

        # 3. Apply RoPE to Q, K.
        cos_2t = cos[:, :, :two_T]
        sin_2t = sin[:, :, :two_T]
        q, k = _apply_rope(q, k, cos_2t, sin_2t)

        # 4. Two SDPA passes with key-modality masks.
        mask_M, mask_C = self._build_masks(two_T, q.device)
        out_M = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_M)
        out_C = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_C)
        out_M = self._merge_heads(out_M)                            # [B, 2T, H]
        out_C = self._merge_heads(out_C)

        # 5. Slice into four sub-outputs.
        u_mm = out_M[:, 0::2, :]   # mel query, mel keys (intra-mel)
        u_cm = out_M[:, 1::2, :]   # chord query, mel keys (cross into chord)
        u_mc = out_C[:, 0::2, :]   # mel query, chord keys (cross into mel)
        u_cc = out_C[:, 1::2, :]   # chord query, chord keys (intra-chord)

        # 6. Per-token gated mix. Intra is unconditional; cross is gated.
        g_m = torch.sigmoid(self.gate_m(m))                          # [B, T, 1]
        g_c = torch.sigmoid(self.gate_c(c))
        # Cache for diagnostics (detached). Mean over batch and time gives a
        # scalar "how open is this gate on average" for wandb logging.
        self._last_gate_m = g_m.detach()
        self._last_gate_c = g_c.detach()

        o_m = self.o_m(u_mm + g_m * u_mc)
        o_c = self.o_c(u_cc + g_c * u_cm)

        # 7. Residual + post-LN (shared LN_attn).
        m = self.ln_attn(m + self.drop(o_m))
        c = self.ln_attn(c + self.drop(o_c))

        # 8. Shared FFN over re-interleaved 2T, split, residual, post-LN.
        stacked = torch.stack([m, c], dim=2).reshape(B, two_T, H)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(stacked)
        else:
            ffn_out = self.ffn(stacked)
            aux_loss = torch.zeros((), device=m.device, dtype=m.dtype)
        ffn_m = ffn_out[:, 0::2]
        ffn_c = ffn_out[:, 1::2]
        m = self.ln_ffn(m + self.drop(ffn_m))
        c = self.ln_ffn(c + self.drop(ffn_c))
        return m, c, aux_loss


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class M2CIntraCrossAttn(RoFormerSymbolicTransformer):
    """Per-modality Q/K/V/O + intra/cross-stream split with per-block gated
    cross-stream mix + shared MoE FFN. See module docstring for the math.

    Inherits local encoder/decoder, tokenizer, embeddings, loss, sampling,
    and the multi-program preserve_program=True override path from
    RoFormerSymbolicTransformer / M2CJointAttn lineage."""

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
        # Drop the inherited single-backbone global stack; we replace it
        # with our custom intra/cross-attn stack below.
        del self.global_roformer
        # Drop the inherited top-level gate_m/gate_c. They were used by the
        # parent's _global_interaction (two RoFormer passes + sigmoid gate);
        # this variant has its own per-block gates inside
        # M2CIntraCrossAttnLayer, so the top-level ones are dead weight.
        del self.gate_m
        del self.gate_c

        ffn_inter = moe_intermediate_size or self.intermediate_size
        self.global_layers = nn.ModuleList([
            M2CIntraCrossAttnLayer(
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

        # Per-modality SOS offsets, zero-init (same as jointattn).
        self.sos_offset_m = nn.Parameter(torch.zeros(self.hidden_size))
        self.sos_offset_c = nn.Parameter(torch.zeros(self.hidden_size))

        # Modality information is already carried by the per-modality
        # Q/K/V/O projections (and, through them, into the MoE router's
        # input statistics). The inherited binary token_type_embedding
        # is redundant for this variant. Zero it and freeze so it stays
        # a no-op additive bias without requiring changes to the parent's
        # local_encode call signature.
        with torch.no_grad():
            self.token_type_embeddings.weight.zero_()
        self.token_type_embeddings.weight.requires_grad = False

    def _assemble_sos(self, batch_size, device, dtype):
        sos_m = (self.global_sos + self.sos_offset_m).view(1, 1, -1)
        sos_c = (self.global_sos + self.sos_offset_c).view(1, 1, -1)
        sos = torch.cat([sos_m, sos_c], dim=1).expand(batch_size, -1, -1)
        return sos.to(device=device, dtype=dtype)

    def _global_interaction(self, h):
        """h: [B, 2T, H] interleaved [m_0, c_0, m_1, c_1, ...].

        Returns (h_out, aux_loss).
        """
        B, two_T, H = h.shape
        assert two_T % 2 == 0
        T = two_T // 2

        m = h[:, 0::2]
        c = h[:, 1::2]

        head_dim = H // self.num_attention_heads
        cos, sin = _rope_freqs(two_T, head_dim, device=h.device, dtype=h.dtype)

        total_aux = torch.zeros((), device=h.device, dtype=h.dtype)
        for layer in self.global_layers:
            m, c, aux = layer(m, c, cos, sin)
            total_aux = total_aux + aux

        out = torch.stack([m, c], dim=2).reshape(B, two_T, H)
        return out, total_aux / max(len(self.global_layers), 1)


# ---------------------------------------------------------------------------
# Training entry point (mirrors jointattn's CLI; --variant tag differs)
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
        description='Train M2CIntraCrossAttn (per-modality Q/K/V/O + '
                    'intra/cross-attn split + per-block gates + shared MoE FFN).',
    )
    parser.add_argument('--task', type=str, required=True,
                        choices=sorted(TASKS),
                        help='Which two-stream task to train on. Resolves '
                             'mod_a/mod_b dataset paths, default programs, '
                             'and user-facing display labels from tasks.py. '
                             'melchord: mod_a=mel, mod_b=chord. drumnondrum: '
                             'mod_a=drum, mod_b=nondrum.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str, default=None,
                        help='OPTIONAL override of the mod_b path resolved '
                             'from --task. mod_a path will still be resolved '
                             'from the task config unless --mod_a_path is '
                             'also passed.')
    parser.add_argument('--mod_a_path', type=str, default=None,
                        help='OPTIONAL override of the mod_a path.')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--global_num_layers', type=int, default=None,
                        help='Default: 12 if large else 6.')
    # Loss weighting is intentionally off by default (all weights = 1.0).
    # Flags retained for parent-class compatibility; override only if you
    # have a specific reason. Note: setting any of these != 1.0 makes
    # val_loss incomparable across runs with different weights.
    parser.add_argument('--mel_loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_loss_weight', type=float, default=1.0)
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--preserve_program', action='store_true', default=True)
    parser.add_argument('--hardcode_program', dest='preserve_program',
                        action='store_false')
    parser.add_argument('--wandb_dir', type=str, default='/tmp/wandb')
    parser.add_argument('--save_top_k', type=int, default=2)
    parser.add_argument('--val_check_interval', type=int, default=500,
                        help='steps between val evaluations. On small corpora '
                             'the val minimum can arrive within the first ~1k '
                             'steps, where the stock 500 resolves it with only '
                             'one or two points.')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--lr_total_steps', type=int, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)
    parser.add_argument('--eos_loss_weight', type=float, default=1.0)
    parser.add_argument('--silence_augment_prob', type=float, default=0.0)
    parser.add_argument('--ctx_corrupt_prob', type=float, default=0.0,
                        help='prob per frame that a silent run of '
                             'ctx_corrupt_len frames begins in the acc '
                             'INPUT context (targets stay clean) -- '
                             'anti-exposure-bias augmentation for the '
                             'sparse stream')
    parser.add_argument('--ctx_corrupt_len', type=int, default=8)
    parser.add_argument('--moe_monitor_every_n_steps', type=int, default=0)
    parser.add_argument('--moe_monitor_n_samples', type=int, default=4)
    parser.add_argument('--dump_samples_dir', type=str, default=None,
                        help='If set, write the first training batch as '
                             '.mid files into this directory at start of '
                             'training (sanity-check listening). Per-sample '
                             'mod_a, mod_b, and combined files are emitted.')
    parser.add_argument('--dump_samples_n', type=int, default=4,
                        help='How many samples from the first batch to dump.')
    parser.add_argument('--dump_samples_every_n_epochs', type=int, default=None,
                        help='If set, also re-dump the first batch of every '
                             'Nth epoch (uses a different filename tag per '
                             'epoch).')
    parser.add_argument('--max_polyphony', type=int, default=16,
                        help='Polyphony cap used by the dump callback. '
                             'Match your preprocess setting (typically 16).')
    parser.add_argument('--gate_init_bias', type=float, default=-10.0,
                        help='Init bias for per-block cross-stream gates. '
                             '-10 -> sigmoid(bias) ~ 4.5e-5, near-off at '
                             'step 0 (preserves per-modality warm-start '
                             'equivalence). Raise to -4 (sigmoid ~ 0.018) '
                             'if you observe gates pinned at init through '
                             'LR warmup.')
    parser.add_argument('--fresh_schedule', action='store_true', default=False,
                        help='When resuming from a Lightning ckpt, load '
                             'ONLY the model weights and start a fresh '
                             'OneCycleLR schedule + fresh optimizer state. '
                             'Use this when --lr_total_steps differs from '
                             'the saved ckpt -- otherwise PyTorch overwrites '
                             'the new schedule with the old one and you get '
                             '"Tried to step N times" or LR stuck at zero.')
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    task = get_task(args.task)
    mod_a_path = args.mod_a_path if args.mod_a_path is not None else task.mod_a_path
    mod_b_path = args.path_to_dataset if args.path_to_dataset is not None else task.mod_b_path

    tag = f'_{args.run_tag}' if args.run_tag else ''
    default_name = (f"m2c_intra_cross_attn_v1.0_{args.model_size}_"
                    f"gnl{gnl}_{task.name}{tag}_"
                    f"batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  '
          f'mod_a={task.mod_a_label} (program {task.mod_a_default_program}, '
          f'{mod_a_path})  '
          f'mod_b={task.mod_b_label} (program {task.mod_b_default_program}, '
          f'{mod_b_path})')

    net = M2CIntraCrossAttn(
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
        ctx_corrupt_prob=args.ctx_corrupt_prob,
        ctx_corrupt_len=args.ctx_corrupt_len,
        eos_loss_weight=args.eos_loss_weight,
        gate_init_bias=args.gate_init_bias,
    )
    print(f'Architecture: per-modality Q/K/V/O + intra/cross-attn (2 SDPA) + '
          f'per-block gates + shared MoE FFN '
          f'({args.moe_num_experts} experts, topk={args.moe_topk})')
    print(f'Global depth: {gnl} layers   gate_init_bias: {args.gate_init_bias}')

    train_set = FramedDataset(mod_b_path, TRAIN_LENGTH,
                              args.batch_size, split='train',
                              mel_path=mod_a_path)
    val_set = FramedDataset(mod_b_path, TRAIN_LENGTH,
                            args.batch_size, split='val',
                            mel_path=mod_a_path)
    train_set_loader = DataLoader(train_set, batch_size=None, num_workers=0)
    val_set_loader = DataLoader(val_set, batch_size=None, num_workers=0)

    # Log implied epoch count so it's obvious whether --lr_total_steps is
    # over/under-shooting the actual training duration. steps_per_epoch is
    # global (across all ranks) -- IterableDataset doesn't shard, so each
    # rank sees the full pool and global step rate is the per-rank rate.
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
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
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
        print(f'[dump] training-sample dumps will be written to '
              f'{args.dump_samples_dir}')

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
                    'variant': 'm2c_intra_cross_attn',
                    'task': task.name,
                    'mod_a_label': task.mod_a_label,
                    'mod_b_label': task.mod_b_label,
                    'mod_a_path': mod_a_path,
                    'mod_b_path': mod_b_path,
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
            print(f'[resume] full Lightning ckpt at {args.checkpoint_path}')
            ckpt_path_for_resume = args.checkpoint_path
        else:
            if has_lightning_meta and args.fresh_schedule:
                print(f'[fresh-schedule] Lightning ckpt at {args.checkpoint_path}; '
                       'loading model weights only (skipping optimizer + '
                       'scheduler state) so the new OneCycleLR with '
                       f'lr_total_steps={args.lr_total_steps} is used.')
            else:
                print(f'[init] bare warm-start ckpt at {args.checkpoint_path}; '
                       'loading state_dict only (no Lightning metadata).')
            sd = loaded['state_dict'] if isinstance(loaded, dict) and 'state_dict' in loaded else loaded
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                print(f'[init] {len(missing)} missing keys (fresh-init, '
                       f'first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected keys (ignored, '
                       f'first few: {unexpected[:3]})')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
    torch.save(net.state_dict(), f'{ckpt_dir}/{model_name}.fin.ckpt')
