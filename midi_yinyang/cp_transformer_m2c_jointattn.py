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
and sampling are inherited from RoFormerSymbolicTransformer unchanged.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_transformer_m2c_moe import (
    RoFormerSymbolicTransformer, FramedDataset, TRAIN_LENGTH, MAX_STEPS,
)


# ---------------------------------------------------------------------------
# RoPE (matches HF RoFormer convention -- interleaved pairs, base=10000)
#
# RoFormer treats each consecutive pair (x_{2i}, x_{2i+1}) as a complex number
# and rotates it by angle theta_i = pos * base^(-2i/D). This is the original
# RoPE paper convention. It is NOT the same as LLaMA's "rotate half" trick
# (which splits the head_dim down the middle); HF RoFormer's pretrained
# weights are calibrated for interleaved pairs, so we must match here for the
# warm-start to be numerically equivalent.
#
# Layout of cos/sin buffer at indices (2i, 2i+1):
#   cos[..., 2i]   = cos(pos * theta_i)
#   cos[..., 2i+1] = cos(pos * theta_i)   # same value
#   (same for sin)
# Layout of rotate_pairs(x):
#   out[..., 2i]   = -x[..., 2i+1]
#   out[..., 2i+1] =  x[..., 2i]
# So the operation is: rotated = x * cos + rotate_pairs(x) * sin, exactly
# matching `apply_rotary_position_embeddings` in HF's modeling_roformer.py.
# ---------------------------------------------------------------------------

def _rope_freqs(seq_len: int, head_dim: int, device, dtype=torch.float32,
                base: float = 10000.0):
    """Return (cos, sin) of shape [1, 1, seq_len, head_dim]. Values at the
    even and odd index of each (2i, 2i+1) pair are identical -- that's how
    the interleaved-pair convention is encoded as a single elementwise
    multiplicand."""
    assert head_dim % 2 == 0, 'head_dim must be even for RoPE'
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)                # [L, half]
    # Interleave: emb[:, 2i] = emb[:, 2i+1] = freqs[:, i]
    emb = freqs.repeat_interleave(2, dim=-1)                    # [L, head_dim]
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def _rotate_pairs(x):
    """Pair-wise 90-degree rotation matching HF RoFormer's
    `rotate_half_query_layer`: (a, b) -> (-b, a) for each adjacent pair."""
    # x: [..., head_dim] -> view as [..., head_dim/2, 2], swap, sign-flip,
    # then flatten back.
    shape = x.shape
    x_pairs = x.view(*shape[:-1], -1, 2)
    x_a = x_pairs[..., 0:1]                                      # [..., half, 1]
    x_b = x_pairs[..., 1:2]
    rotated = torch.cat([-x_b, x_a], dim=-1)                    # (-b, a)
    return rotated.view(*shape)


def _apply_rope(q, k, cos, sin):
    return q * cos + _rotate_pairs(q) * sin, k * cos + _rotate_pairs(k) * sin


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
        """x: [B, L, H]. Returns (out, aux_loss).

        Caches the most recent routing probabilities at self._last_routing_probs
        with shape [B, L, num_experts] so a diagnostic callback can read them
        (split by interleaved position parity to compare drum vs non-drum).
        Detached to avoid graph buildup; replaced every forward."""
        B, L, H = x.shape
        x_flat = x.reshape(-1, H)                                   # [N, H]
        N = x_flat.size(0)
        logits = self.gate(x_flat)                                  # [N, E]
        probs = F.softmax(logits, dim=-1)
        # Cache for diagnostics. Reshape back to [B, L, E].
        self._last_routing_probs = probs.detach().view(B, L, self.num_experts)
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
    """One transformer block matching HF RoFormer's post-LN structure but
    with per-modality Q/K/V/O projections and a shared MoE FFN.

    Order (matches transformers/models/roformer/modeling_roformer.py):
        attn_out = self_attention(x)              # joint over interleaved 2T
        x = LN_attn(x + Wo(attn_out))              # POST-LN, residual first
        ffn_out = FFN(x)                           # GELU MLP (or MoE)
        x = LN_ffn(x + ffn_out)                    # POST-LN

    Per-modality parameters: q_m/q_c, k_m/k_c, v_m/v_c, o_m/o_c.
    Shared parameters: ln_attn, ln_ffn, ffn (MoE or dense).
    """

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 moe_num_experts, moe_topk, moe_intermediate_size,
                 dropout=0.0):
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

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * d)

    def forward(self, m, c, cos, sin):
        """m, c: [B, T, H]. cos, sin: RoPE buffers covering 2T positions.

        Returns (m_out, c_out, aux_loss)."""
        B, T, H = m.shape
        assert c.shape == m.shape

        # 1. Per-modality Q/K/V projection (NO pre-LN; RoFormer is post-LN).
        q_m = self._split_heads(self.q_m(m))                        # [B, h, T, d]
        k_m = self._split_heads(self.k_m(m))
        v_m = self._split_heads(self.v_m(m))
        q_c = self._split_heads(self.q_c(c))
        k_c = self._split_heads(self.k_c(c))
        v_c = self._split_heads(self.v_c(c))

        # 2. Interleave Q/K/V into 2T sequence so position 2t = mel_t,
        #    2t+1 = chord_t. RoPE applies to the 2T positions globally.
        def interleave(a_m, a_c):
            return torch.stack([a_m, a_c], dim=3).reshape(
                B, self.num_heads, 2 * T, self.head_dim
            )
        q = interleave(q_m, q_c)
        k = interleave(k_m, k_c)
        v = interleave(v_m, v_c)

        # 3. Apply RoPE to Q, K (V is not rotated).
        cos_2t = cos[:, :, : 2 * T]
        sin_2t = sin[:, :, : 2 * T]
        q, k = _apply_rope(q, k, cos_2t, sin_2t)

        # 4. Joint causal self-attention.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = self._merge_heads(out)                                # [B, 2T, H]

        # 5. Split, per-modality output projection, residual, POST-LN (shared).
        out_m = out[:, 0::2, :]
        out_c = out[:, 1::2, :]
        m = self.ln_attn(m + self.drop(self.o_m(out_m)))
        c = self.ln_attn(c + self.drop(self.o_c(out_c)))

        # 6. Shared FFN over the interleaved 2T sequence, then split, residual,
        #    POST-LN (shared).
        stacked = torch.stack([m, c], dim=2).reshape(B, 2 * T, H)
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

class M2CJointAttn(RoFormerSymbolicTransformer):
    """One-backbone-equivalent two-backbones variant: per-modality Q/K/V/O,
    joint self-attention, shared MoE FFN. Inherits local encoder/decoder,
    tokenizer, embeddings, loss, and sampling from RoFormerSymbolicTransformer.

    Overrides preprocess() to PRESERVE the original program in the a-slot
    token. The parent's preprocess hardcodes a-slot to program 24 (mel) and
    program 0 (chord) for with_velocity=False, which is fine for POP909
    (single melody + single chord track) but disastrous for LAMD's
    multi-instrument drum / non-drum streams: drums get tagged as
    Classical Guitar and every pitched instrument collapses into Grand
    Piano. The vocab already covers programs 0..127 (n_normal_tokens =
    24*128 + 256), so this override only changes which token IDs are
    emitted, not the embedding size."""

    def __init__(self, *args, moe_num_experts=4, moe_topk=2,
                 moe_intermediate_size=None, global_num_layers=None,
                 global_dropout=0.0, preserve_program=True, **kwargs):
        # Default preserve_program=True here: LAMD's multi-instrument streams
        # are the primary target for this variant. Override to False if you
        # want the POP909 hardcoded-program behavior.
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
        # NOTE: HF RoFormer's post-LN stack already applies a LayerNorm at the
        # end of each layer's FFN, so we do NOT add an extra final LN here.
        # Per-modality differentiation is achieved by the per-modality Q/K/V/O
        # projections alone -- a modality type embedding would shift the input
        # distribution away from one-backbone equivalence at step 0.

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

        # Precompute RoPE buffers for 2T positions.
        head_dim = H // self.num_attention_heads
        cos, sin = _rope_freqs(2 * T, head_dim, device=h.device, dtype=h.dtype)

        total_aux = torch.zeros((), device=h.device, dtype=h.dtype)
        for layer in self.global_layers:
            m, c, aux = layer(m, c, cos, sin)
            total_aux = total_aux + aux

        # Re-interleave for downstream local decoder.
        out = torch.stack([m, c], dim=2).reshape(B, 2 * T, H)
        return out, total_aux / max(len(self.global_layers), 1)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Lightning / DataLoader imports are deferred to here so the module can
    # be imported by init scripts that don't have lightning installed.
    from torch.utils.data import DataLoader
    try:
        import lightning as L
        from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
    except ImportError:
        import pytorch_lightning as L
        from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

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
    parser.add_argument('--preserve_program', action='store_true', default=True,
                        help='Preserve actual per-note program in the a-slot '
                             'token. Default True for M2CJointAttn (LAMD is '
                             'the primary target). Use --hardcode_program for '
                             'POP909-style behavior.')
    parser.add_argument('--hardcode_program', dest='preserve_program',
                        action='store_false',
                        help='Hardcode a-slot to program 24 (mel) / 0 (chord). '
                             'POP909 default of the legacy m2c models.')
    parser.add_argument('--wandb_dir', type=str, default='/tmp/wandb')
    parser.add_argument('--save_top_k', type=int, default=2)
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help="Where to write checkpoints. Default 'ckpt/{model_name}'.")
    parser.add_argument('--max_lr', type=float, default=1e-4,
                        help='Peak LR for the OneCycleLR schedule. For '
                             'warm-started training (the common case), '
                             '5e-5 is often safer than the default 1e-4.')
    parser.add_argument('--lr_total_steps', type=int, default=None,
                        help='Total step budget the OneCycleLR schedule '
                             'plans for. Default: MAX_STEPS (1M). Set to '
                             'your actual training budget so the LR '
                             'finishes cosine-decaying right around when '
                             'you stop.')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Per-step gradient norm clip. 1.0 is the '
                             'MoE safety default. Set to 0 to disable.')
    parser.add_argument('--aux_loss_weight', type=float, default=0.01,
                        help='Weight on MoE load-balancing aux loss. '
                             'Lower (e.g. 0) to allow more expert '
                             'specialization; higher to force more '
                             'uniformity.')
    parser.add_argument('--silence_augment_prob', type=float, default=0.0,
                        help='Per-sample probability of silencing each '
                             'modality during training. Trains the model '
                             'to handle mel_only / chord_only inference '
                             'in-distribution. Try 0.1 (10%% drum-silenced, '
                             '10%% non-drum-silenced, 80%% normal). '
                             'Default 0 = co-only training (legacy).')
    parser.add_argument('--moe_monitor_every_n_steps', type=int, default=0,
                        help='If > 0, every N training steps run a forward '
                             'on a cached held-out drum/non-drum batch and '
                             'log per-layer per-modality routing probabilities '
                             'to wandb. Diagnoses whether MoE experts are '
                             'specializing per modality (drum vs non-drum).')
    parser.add_argument('--moe_monitor_n_samples', type=int, default=4)
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
        preserve_program=args.preserve_program,
        max_lr=args.max_lr,
        lr_total_steps=args.lr_total_steps,
        aux_loss_weight=args.aux_loss_weight,
        silence_augment_prob=args.silence_augment_prob,
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

    extra_callbacks = []
    if args.moe_monitor_every_n_steps > 0:
        from moe_routing_monitor import MoERoutingMonitor
        extra_callbacks.append(
            MoERoutingMonitor(
                every_n_steps=args.moe_monitor_every_n_steps,
                n_samples=args.moe_monitor_n_samples,
            ).as_callback()
        )

    trainer = L.Trainer(
        devices=n_gpus,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=MAX_STEPS,
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
    # Differentiate "warm-start from a bare init ckpt" vs "resume from a full
    # Lightning ckpt". Warm-start ckpts (from init_pretrained_into_jointattn.py)
    # contain only {state_dict, ...} and lack Lightning's run metadata
    # (pytorch-lightning_version, epoch, optimizer state). Pass the bare ckpt
    # to trainer.fit(ckpt_path=...) and Lightning crashes with KeyError on the
    # missing version key.
    ckpt_path_for_resume = None
    if args.checkpoint_path is not None:
        loaded = torch.load(args.checkpoint_path, map_location='cpu',
                             weights_only=False)
        has_lightning_meta = (
            isinstance(loaded, dict)
            and 'pytorch-lightning_version' in loaded
        )
        if has_lightning_meta:
            # Full Lightning ckpt -> resume mode (restores optimizer, lr,
            # step counter, etc.).
            print(f'[resume] full Lightning ckpt at {args.checkpoint_path}')
            ckpt_path_for_resume = args.checkpoint_path
        else:
            # Bare warm-start ckpt -> load weights, start training from step 0.
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
