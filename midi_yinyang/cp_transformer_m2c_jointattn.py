"""M2CJointAttn -- per-modality Q/K/V/O + joint self-attention + shared MoE FFN.

This is the "Option A" variant of the two-backbones idea, designed to be
strictly closer to one-backbone equivalence than the cross-attention
adapter variant in cp_transformer_m2c_two_backbones_crossattn.py.

Per layer:
    1. Pre-LN per modality.
    2. Per-modality Q/K/V projections (untied between mel and chord).
    3. Apply RoPE to Q, K (positions = global timestep, identical
       within a {m_t, c_t} pair so that the rotation is consistent with
       the interleaved 2T layout).
    4. Interleave Q, K, V into a single 2T sequence and run a SINGLE
       causal self-attention. Cross-modality information flows through
       the off-diagonal blocks of the attention score matrix -- no
       separate cross-attention module.
    5. Split the output back to per-modality and apply per-modality
       output projection + residual.
    6. Shared MoE FFN over the interleaved 2T (with type embeddings so
       the router can specialize).

Warm-start (via init_pretrained_into_jointattn.py): copy the pretrained
one-backbone's per-layer Wq/Wk/Wv/Wo into BOTH mel and chord
projections, and replicate the pretrained dense FFN into all K experts
of the shared MoE FFN. At step 0 the model is provably equivalent to
the pretrained one-backbone evaluated on the interleaved input.

The local encoder/decoder, tokenizer, embeddings, preprocess, loss,
and sampling are inherited from M2CMoE unchanged.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import lightning as L_modern  # noqa: F401  # if installed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from cp_transformer_m2c_moe import (
    M2CMoE, FramedDataset, TRAIN_LENGTH, MAX_STEPS,
)


# ---------------------------------------------------------------------------
# RoPE (matches RoFormer convention with base=10000)
# ---------------------------------------------------------------------------

def _rope_freqs(seq_len: int, head_dim: int, device, dtype=torch.float32,
                base: float = 10000.0):
    """Return (cos, sin) of shape [1, 1, seq_len, head_dim], matching
    HF RoFormer's rotary convention (pair-wise rotation in the head_dim)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)               # [L, half]
    emb = torch.cat([freqs, freqs], dim=-1)                     # [L, head_dim]
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Switch-style MoE FFN (shared between modalities)
# ---------------------------------------------------------------------------

class SimpleMoEFFN(nn.Module):
    """Switch Transformer-style MoE FFN with top-k routing.

    Each expert is a 2-layer MLP (Linear -> GELU -> Linear), matching the
    dense FFN shape of the pretrained one-backbone so warm-start is a
    direct weight copy.
    """

    def __init__(self, hidden_size, intermediate_size, num_experts, topk):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.fc1 = nn.ModuleList(
            [nn.Linear(hidden_size, intermediate_size) for _ in range(num_experts)]
        )
        self.fc2 = nn.ModuleList(
            [nn.Linear(intermediate_size, hidden_size) for _ in range(num_experts)]
        )

    def forward(self, x):
        """x: [B, L, H]. Returns (out, aux_loss)."""
        B, L, H = x.shape
        x_flat = x.reshape(-1, H)                                   # [N, H]
        N = x_flat.size(0)
        logits = self.gate(x_flat)                                  # [N, E]
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = probs.topk(self.topk, dim=-1)           # [N, k]
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)

        out_flat = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            # tokens whose top-k routes include expert e
            mask = (top_idx == e).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            x_e = x_flat[idx]
            h = F.gelu(self.fc1[e](x_e))
            y_e = self.fc2[e](h)
            # weight = sum of top_vals for this expert (handles topk>1)
            w_e = (top_vals * (top_idx == e).float()).sum(dim=-1)[idx].unsqueeze(-1)
            out_flat.index_add_(0, idx, y_e * w_e)

        # Switch-style load balancing loss:
        # f_i = fraction of tokens whose top-1 is expert i
        # P_i = mean softmax probability of expert i
        top1 = top_idx[:, 0]
        f = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        f.scatter_add_(0, top1, torch.ones_like(top1, dtype=x.dtype))
        f = f / max(N, 1)
        P = probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()

        return out_flat.view(B, L, H), aux_loss


# ---------------------------------------------------------------------------
# Per-layer joint-attention block
# ---------------------------------------------------------------------------

class M2CJointAttnLayer(nn.Module):
    """One transformer block: per-modality Q/K/V/O + joint attention +
    shared MoE FFN (with type embeddings to disambiguate streams to the
    router)."""

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 moe_num_experts, moe_topk, moe_intermediate_size,
                 dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Per-modality pre-LN and Q/K/V/O projections
        self.ln_attn_m = nn.LayerNorm(hidden_size)
        self.ln_attn_c = nn.LayerNorm(hidden_size)
        self.q_m = nn.Linear(hidden_size, hidden_size)
        self.k_m = nn.Linear(hidden_size, hidden_size)
        self.v_m = nn.Linear(hidden_size, hidden_size)
        self.o_m = nn.Linear(hidden_size, hidden_size)
        self.q_c = nn.Linear(hidden_size, hidden_size)
        self.k_c = nn.Linear(hidden_size, hidden_size)
        self.v_c = nn.Linear(hidden_size, hidden_size)
        self.o_c = nn.Linear(hidden_size, hidden_size)

        # FFN side: shared LN + shared (MoE or dense) FFN
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

    def _split_heads(self, x):
        B, L, H = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * d)

    def forward(self, m, c, cos, sin):
        """m, c: [B, T, H]. cos, sin: RoPE buffers covering 2T positions.

        Returns (m_out, c_out, aux_loss)."""
        B, T, H = m.shape
        assert c.shape == m.shape

        # 1. Pre-LN + per-modality QKV projection
        m_n = self.ln_attn_m(m)
        c_n = self.ln_attn_c(c)
        q_m = self._split_heads(self.q_m(m_n))                      # [B, h, T, d]
        k_m = self._split_heads(self.k_m(m_n))
        v_m = self._split_heads(self.v_m(m_n))
        q_c = self._split_heads(self.q_c(c_n))
        k_c = self._split_heads(self.k_c(c_n))
        v_c = self._split_heads(self.v_c(c_n))

        # 2. Interleave Q/K/V into 2T sequence so position 2t = mel_t,
        #    2t+1 = chord_t. RoPE applies to the 2T positions globally.
        def interleave(a_m, a_c):
            # [B, h, T, d] each -> [B, h, 2T, d]
            return torch.stack([a_m, a_c], dim=3).reshape(B, self.num_heads, 2 * T, self.head_dim)
        q = interleave(q_m, q_c)
        k = interleave(k_m, k_c)
        v = interleave(v_m, v_c)

        # 3. Apply RoPE to Q, K
        cos_2t = cos[:, :, : 2 * T]
        sin_2t = sin[:, :, : 2 * T]
        q, k = _apply_rope(q, k, cos_2t, sin_2t)

        # 4. Joint causal self-attention (PyTorch SDPA handles flash/cuDNN
        #    paths; is_causal=True applies the lower-triangular mask)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = self._merge_heads(out)                                # [B, 2T, H]

        # 5. Split, per-modality output projection + residual
        out_m = out[:, 0::2, :]
        out_c = out[:, 1::2, :]
        m = m + self.drop(self.o_m(out_m))
        c = c + self.drop(self.o_c(out_c))

        # 6. Shared (MoE) FFN over interleaved 2T sequence
        m_n = self.ln_ffn(m)
        c_n = self.ln_ffn(c)
        stacked = torch.stack([m_n, c_n], dim=2).reshape(B, 2 * T, H)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(stacked)
        else:
            ffn_out, aux_loss = self.ffn(stacked), torch.zeros((), device=m.device)
        ffn_m = ffn_out[:, 0::2]
        ffn_c = ffn_out[:, 1::2]
        m = m + self.drop(ffn_m)
        c = c + self.drop(ffn_c)
        return m, c, aux_loss


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class M2CJointAttn(M2CMoE):
    """One-backbone-equivalent two-backbones variant: per-modality Q/K/V/O,
    joint self-attention, shared MoE FFN. Inherits local encoder/decoder,
    tokenizer, embeddings, preprocess, loss, and sampling from M2CMoE."""

    def __init__(self, *args, moe_num_experts=4, moe_topk=2,
                 moe_intermediate_size=None, global_num_layers=None,
                 global_dropout=0.0, **kwargs):
        super().__init__(
            *args,
            moe_num_experts=moe_num_experts,
            moe_topk=moe_topk,
            moe_intermediate_size=moe_intermediate_size,
            global_num_layers=global_num_layers,
            global_dropout=global_dropout,
            **kwargs,
        )
        # Drop the inherited single-backbone global stack -- we replace it
        # with our custom joint-attn stack below.
        del self.global_roformer

        ffn_inter = moe_intermediate_size or self.intermediate_size
        self.global_layers = nn.ModuleList([
            M2CJointAttnLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                moe_num_experts=moe_num_experts,
                moe_topk=moe_topk,
                moe_intermediate_size=ffn_inter,
                dropout=global_dropout,
            )
            for _ in range(self.global_num_layers)
        ])
        # Final LN before sending to the local decoder (matches RoFormer
        # post-norm-style finalize used in the original stack).
        self.global_final_ln = nn.LayerNorm(self.hidden_size)

        # Type embedding added to the FFN input so the shared router can
        # condition on modality (mel=0, chord=1). Same shape as the
        # parent's token_type_embeddings but a SEPARATE module so we can
        # warm-start to zero (no init bias).
        self.modality_type_emb = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.modality_type_emb.weight.zero_()

    # ------------------------------------------------------------------
    # Override: global interaction is our custom joint-attn stack
    # ------------------------------------------------------------------

    def _global_interaction(self, h):
        """h: [B, 2T, H] interleaved [m_0, c_0, m_1, c_1, ...] (already
        shift-by-2'd by the caller).

        Returns (h_out, aux_loss) with the same shape as h.
        """
        B, two_T, H = h.shape
        assert two_T % 2 == 0
        T = two_T // 2

        # Split interleaved input into per-modality streams.
        m = h[:, 0::2]                                              # [B, T, H]
        c = h[:, 1::2]

        # Add modality type embedding once at entry so the routers see it.
        mod_m = self.modality_type_emb(torch.zeros(1, dtype=torch.long, device=h.device))
        mod_c = self.modality_type_emb(torch.ones(1, dtype=torch.long, device=h.device))
        m = m + mod_m
        c = c + mod_c

        # Precompute RoPE buffers for 2T positions.
        head_dim = H // self.num_attention_heads
        cos, sin = _rope_freqs(2 * T, head_dim, device=h.device, dtype=h.dtype)

        total_aux = torch.zeros((), device=h.device, dtype=h.dtype)
        for layer in self.global_layers:
            m, c, aux = layer(m, c, cos, sin)
            total_aux = total_aux + aux
        m = self.global_final_ln(m)
        c = self.global_final_ln(c)

        # Re-interleave for downstream local decoder.
        out = torch.stack([m, c], dim=2).reshape(B, 2 * T, H)
        return out, total_aux / max(len(self.global_layers), 1)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train M2CJointAttn (per-modality Q/K/V/O + joint '
                    'self-attention + shared MoE FFN).',
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--global_num_layers', type=int, default=None,
                        help='Default: 12 if large else 6.')
    parser.add_argument('--mel_loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_loss_weight', type=float, default=3.0)
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--wandb_dir', type=str, default='/tmp/wandb')
    parser.add_argument('--save_top_k', type=int, default=2)
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help="Where to write checkpoints. Default 'ckpt/{model_name}'.")
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    tag = f'_{args.run_tag}' if args.run_tag else ''
    default_name = (f"m2c_jointattn_v1.0_{args.model_size}_"
                    f"gnl{gnl}{tag}_batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CJointAttn(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
    )
    print(f'Architecture: per-modality Q/K/V/O + joint self-attn + '
          f'shared MoE FFN ({args.moe_num_experts} experts, topk={args.moe_topk})')
    print(f'Global depth: {gnl} layers')

    train_set_loader = DataLoader(
        FramedDataset(args.path_to_dataset, TRAIN_LENGTH, args.batch_size, split='train'),
        batch_size=None, num_workers=0,
    )
    val_set_loader = DataLoader(
        FramedDataset(args.path_to_dataset, TRAIN_LENGTH, args.batch_size, split='val'),
        batch_size=None, num_workers=0,
    )

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

    trainer = L.Trainer(
        devices=n_gpus,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=MAX_STEPS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        val_check_interval=500,
        limit_val_batches=25,
        check_val_every_n_epoch=None,
        logger=(
            WandbLogger(
                name=model_name, project='MusicMOE',
                save_dir=args.wandb_dir,
                config={
                    'batch_size': args.batch_size,
                    'model_size': args.model_size,
                    'train_length': TRAIN_LENGTH,
                    'variant': 'm2c_jointattn',
                    'global_num_layers': gnl,
                    'moe_num_experts': args.moe_num_experts,
                    'moe_topk': args.moe_topk,
                    'run_tag': args.run_tag,
                },
            ) if args.wandb else TensorBoardLogger('tb_logs', name=model_name)
        ),
        num_sanity_val_steps=0 if args.checkpoint_path is not None else 2,
        strategy=strategy,
    )
    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=args.checkpoint_path)
    torch.save(net.state_dict(), f'{ckpt_dir}/{model_name}.fin.ckpt')
