"""Per-layer fusion + mask-predict variant.

Combines:
  - M2CPerLayerFusion's architecture (per-block dual-pass gated cross-stream
    coupling; same-step cross allowed; decompartmentalizable via per-block
    gates).
  - M2CMaskMoE's mask-predict training and iterative inference (random per-
    block-per-modality masking, loss only on masked positions, MaskGIT-style
    iterative refinement at inference).

What this gets you (relative to the AR-only per-layer fusion variant):

  - True bidirectional mutual coupling at the SAME timestep: each refinement
    iteration sees the partner modality's latest committed value.
  - All 5 modes (co / mel2chord / chord2mel / mel_only / chord_only) handled
    cleanly via masking patterns -- no silence-frame hack for single-stream.
  - Decompartmentalization preserved: per-block gate_m / gate_c are unchanged,
    gate-zero still drops the cross-attention adapter per layer.

Trade-off: inference cost is ~n_refine_steps x N global forwards per timestep
(vs. one AR forward). Block-by-block AR over timesteps is unchanged; the
refinement is inside each block.

Run:
    python cp_transformer_m2c_per_layer_fusion_mask.py \\
        --batch_size 8 --model_size large \\
        --path_to_dataset data/pop909_chord_cp4_v2.pt \\
        --moe_num_experts 4 --moe_topk 2 \\
        --wandb
"""

# Vendored transformers fork on sys.path BEFORE anything else imports.
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from cp_transformer_m2c_moe import (
    FramedDataset,
    TRAIN_LENGTH,
    MAX_STEPS,
)
from cp_transformer_m2c_per_layer_fusion import M2CPerLayerFusion
from cp_transformer_m2c_samestep import TimestepRoPE


class M2CPerLayerFusionMask(M2CPerLayerFusion):
    """Per-layer-fusion architecture trained with mask-predict.
    Inference is block-by-block AR with K-step iterative refinement per block.

    Inherits `_global_interaction` (per-block dual-pass + per-block gates)
    from M2CPerLayerFusion. Adds vocab extension, mask embeddings, mask-aware
    forward, masked-position loss, and MaskGIT-style sampling -- the same
    mechanics as M2CMaskMoE but layered on the per-layer fusion backbone
    instead of the SameStep backbone.
    """

    def __init__(self, *args, mel_loss_weight=1.0, acc_loss_weight=1.0, **kwargs):
        super().__init__(*args, mel_loss_weight=mel_loss_weight,
                         acc_loss_weight=acc_loss_weight, **kwargs)

        # Extend the CP vocab by 1 to add a [MASK] token.
        old_n = self.tokenizer.n_tokens
        self.mask_token_id = old_n
        self.vocab_size = old_n + 1

        new_emb = nn.Embedding(self.vocab_size, self.hidden_size)
        with torch.no_grad():
            new_emb.weight[:old_n] = self.local_embedding.weight
            nn.init.normal_(new_emb.weight[old_n:], std=0.02)
        self.local_embedding = new_emb

        new_dec = nn.Linear(self.hidden_size, self.vocab_size)
        with torch.no_grad():
            new_dec.weight[:old_n] = self.final_decoder.weight
            new_dec.bias[:old_n] = self.final_decoder.bias
            nn.init.normal_(new_dec.weight[old_n:], std=0.02)
            nn.init.zeros_(new_dec.bias[old_n:])
        self.final_decoder = new_dec

        # Learned global-axis mask embeddings, one per modality. Replace the
        # per-block summary at masked positions so the global tower can see
        # "this slot is unknown" without leaking the masked token's identity.
        self.mask_emb_m = nn.Parameter(torch.randn(self.hidden_size))
        self.mask_emb_c = nn.Parameter(torch.randn(self.hidden_size))

        # Symmetric same-step RoPE for every fusion block's two encoders.
        # Without this, mel-Q -> chord-K and chord-Q -> mel-K have RoPE
        # relative rotations -1 and +1 respectively (asymmetric); with it,
        # positions 2t and 2t+1 share the same rotation R(t), so all
        # relative rotations are R(t - t') -- self and both cross-attention
        # directions match. Mask-predict's bidirectional symmetric coupling
        # depends on this; without it, "m_t given c_t" and "c_t given m_t"
        # learn out-of-alignment conditionals.
        for block in self.fusion_blocks:
            for enc in (block.layer_mel_pass, block.layer_chord_pass):
                old = enc.embed_positions
                new = TimestepRoPE(
                    num_positions=old.num_embeddings,
                    embedding_dim=old.embedding_dim,
                )
                enc.embed_positions = new

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def _apply_token_mask(self, x, mask):
        """Replace tokens at masked blocks with the [MASK] id.
        x: [B, T, subseq]; mask: [B, T] bool."""
        if mask is None:
            return x
        return torch.where(
            mask.unsqueeze(-1),
            torch.full_like(x, self.mask_token_id),
            x,
        )

    def forward(self, x_m, x_c, mask_m=None, mask_c=None):
        """No shift-by-2. Positions align with targets; masking is what hides
        the answer. Returns (logits_m, logits_c, aux_loss)."""
        B, T, subseq = x_m.shape
        device = x_m.device

        x_m_in = self._apply_token_mask(x_m, mask_m)
        x_c_in = self._apply_token_mask(x_c, mask_c)

        type_m = torch.zeros(B, T, subseq + 1, dtype=torch.long, device=device)
        type_c = torch.ones(B, T, subseq + 1, dtype=torch.long, device=device)

        h_m, emb_m = self.local_encode(x_m_in, type_m)
        h_c, emb_c = self.local_encode(x_c_in, type_c)
        h_m = h_m.view(B, T, -1)
        h_c = h_c.view(B, T, -1)

        # Override block summary at masked positions with the learned global
        # mask embedding.
        if mask_m is not None:
            h_m = torch.where(
                mask_m.unsqueeze(-1),
                self.mask_emb_m.view(1, 1, -1), h_m,
            )
        if mask_c is not None:
            h_c = torch.where(
                mask_c.unsqueeze(-1),
                self.mask_emb_c.view(1, 1, -1), h_c,
            )

        # Interleave -> [B, 2T, H] = [m_0, c_0, m_1, c_1, ...].
        h_pairs = torch.stack([h_m, h_c], dim=2).reshape(B, 2 * T, -1)

        # Per-layer fusion's _global_interaction: stack of fusion blocks, each
        # with dual-pass attention + per-stream MoE + per-block gated combine.
        h_global, aux_loss = self._global_interaction(h_pairs)
        h_global = h_global.view(B, T, 2, -1)
        h_m_out = h_global[:, :, 0]
        h_c_out = h_global[:, :, 1]

        logits_m = self.local_decode(h_m_out, emb_m).view(B, T, subseq, -1)
        logits_c = self.local_decode(h_c_out, emb_c).view(B, T, subseq, -1)
        return logits_m, logits_c, aux_loss

    def loss(self, x_m_raw, x_c_raw, pitch_shift):
        x_m, x_c = self.preprocess(x_m_raw, pitch_shift, y=x_c_raw)
        B, T, subseq = x_m.shape

        # Per-sample mask ratio uniform in [0, 1]; per-(block, modality)
        # Bernoulli(p) independently. Guarantee at least one masked position.
        p = torch.rand(B, device=x_m.device)
        mask_m = torch.rand(B, T, device=x_m.device) < p.unsqueeze(1)
        mask_c = torch.rand(B, T, device=x_m.device) < p.unsqueeze(1)
        if not (mask_m.any() or mask_c.any()):
            mask_m[0, 0] = True

        logits_m, logits_c, aux_loss = self(x_m, x_c, mask_m, mask_c)

        pad = self.tokenizer.pad_token
        loss_m_per_tok = F.cross_entropy(
            logits_m.reshape(-1, self.vocab_size),
            x_m.reshape(-1),
            ignore_index=pad,
            reduction='none',
        ).view(B, T, subseq)
        loss_c_per_tok = F.cross_entropy(
            logits_c.reshape(-1, self.vocab_size),
            x_c.reshape(-1),
            ignore_index=pad,
            reduction='none',
        ).view(B, T, subseq)

        block_m = mask_m.unsqueeze(-1).float()
        block_c = mask_c.unsqueeze(-1).float()
        nonpad_m = (x_m != pad).float()
        nonpad_c = (x_c != pad).float()

        denom_m = (block_m * nonpad_m).sum().clamp_min(1.0)
        denom_c = (block_c * nonpad_c).sum().clamp_min(1.0)
        loss_m = (loss_m_per_tok * block_m).sum() / denom_m
        loss_c = (loss_c_per_tok * block_c).sum() / denom_c

        mw = self.mel_loss_weight
        cw = self.acc_loss_weight
        ce_loss = (mw * loss_m + cw * loss_c) / max(mw + cw, 1e-8)

        if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
            aux_loss = aux_loss.mean()
            total_loss = ce_loss + 0.01 * aux_loss
        else:
            aux_loss = ce_loss.new_zeros(())
            total_loss = ce_loss

        self._last_loss_m = loss_m.detach()
        self._last_loss_c = loss_c.detach()
        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_loss_melody', self._last_loss_m, on_step=True)
        self.log('train_loss_chord', self._last_loss_c, on_step=True)
        self.log('moe_aux_loss', aux_loss, on_step=True, on_epoch=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('training/lr', lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_loss_melody', self._last_loss_m)
        self.log('val_loss_chord', self._last_loss_c)
        self.log('val_moe_aux_loss', aux_loss, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Inference: block-by-block AR with K-step iterative refinement per block
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_tokens(self, logits, temperature):
        """logits: [..., vocab]. Returns sampled token ids of shape logits[...:-1].
        [MASK] is excluded from sampling."""
        logits = logits.clone()
        logits[..., self.mask_token_id] = float('-inf')
        if temperature == 0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        return torch.multinomial(flat, 1).squeeze(-1).view(*logits.shape[:-1])

    @torch.no_grad()
    def mask_predict_sampling(self, x_m_prompt, x_c_prompt, max_seq_len=384,
                              n_refine_steps=2, temperature=1.0):
        """Block-by-block generation with K within-block refinement passes.

        n_refine_steps:
          1 = pure parallel (m_t and c_t sampled independently each block).
          2 = sample more-confident slot first, refine the other given it.
              This gives true same-timestep bidirectional dependency.
          >2 = no extra benefit with 2 modalities.
        """
        B, S0, subseq = x_m_prompt.shape
        device = x_m_prompt.device

        x_m_full = x_m_prompt.clone()
        x_c_full = x_c_prompt.clone()
        mask_m_full = torch.zeros(B, S0, dtype=torch.bool, device=device)
        mask_c_full = torch.zeros(B, S0, dtype=torch.bool, device=device)

        for t in range(S0, max_seq_len):
            if t % 10 == 0:
                print(f'[mask-predict] block {t}/{max_seq_len}')

            # Append a fully-masked block at position t.
            x_m_full = torch.cat([
                x_m_full,
                torch.full((B, 1, subseq), self.mask_token_id,
                           dtype=torch.long, device=device),
            ], dim=1)
            x_c_full = torch.cat([
                x_c_full,
                torch.full((B, 1, subseq), self.mask_token_id,
                           dtype=torch.long, device=device),
            ], dim=1)
            mask_m_full = torch.cat([
                mask_m_full,
                torch.ones(B, 1, dtype=torch.bool, device=device),
            ], dim=1)
            mask_c_full = torch.cat([
                mask_c_full,
                torch.ones(B, 1, dtype=torch.bool, device=device),
            ], dim=1)

            for k in range(n_refine_steps):
                logits_m, logits_c, _ = self(
                    x_m_full, x_c_full, mask_m_full, mask_c_full,
                )
                m_block_logits = logits_m[:, t]
                c_block_logits = logits_c[:, t]

                m_sample = self._sample_tokens(m_block_logits, temperature)
                c_sample = self._sample_tokens(c_block_logits, temperature)

                last = (k == n_refine_steps - 1)
                if last:
                    m_active = mask_m_full[:, t].unsqueeze(-1)
                    c_active = mask_c_full[:, t].unsqueeze(-1)
                    x_m_full[:, t] = torch.where(m_active, m_sample, x_m_full[:, t])
                    x_c_full[:, t] = torch.where(c_active, c_sample, x_c_full[:, t])
                    mask_m_full[:, t] = False
                    mask_c_full[:, t] = False
                else:
                    # Sample the more-confident slot first; leave the other
                    # masked so the next refinement pass sees the partner's
                    # just-committed sample as context.
                    conf_m = F.softmax(m_block_logits.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                    conf_c = F.softmax(c_block_logits.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                    pick_m = (conf_m >= conf_c) & mask_m_full[:, t]
                    pick_c = (conf_m < conf_c) & mask_c_full[:, t]
                    only_m = mask_m_full[:, t] & ~mask_c_full[:, t]
                    only_c = mask_c_full[:, t] & ~mask_m_full[:, t]
                    pick_m = pick_m | only_m
                    pick_c = pick_c | only_c

                    x_m_full[:, t] = torch.where(pick_m.unsqueeze(-1), m_sample, x_m_full[:, t])
                    x_c_full[:, t] = torch.where(pick_c.unsqueeze(-1), c_sample, x_c_full[:, t])
                    mask_m_full[:, t] = mask_m_full[:, t] & ~pick_m
                    mask_c_full[:, t] = mask_c_full[:, t] & ~pick_c

        mel_frames = [x_m_full[:, t, :] for t in range(x_m_full.shape[1])]
        chord_frames = [x_c_full[:, t, :] for t in range(x_c_full.shape[1])]
        return mel_frames, chord_frames


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the per-layer fusion + mask-predict M2C MoE '
                    'variant. Combines per-block gated cross-stream coupling '
                    'with random masking objective for true bidirectional '
                    'same-step coupling via iterative refinement at inference.',
    )

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str,
                        help='path to the chord .pt; melody .pt is found by '
                             'replacing "chord" -> "melody"')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--global_num_layers', type=int, default=None,
                        help='Default: 12 if model_size=large else 6.')
    parser.add_argument('--mel_loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_loss_weight', type=float, default=1.0)
    args = parser.parse_args()

    batch_size = args.batch_size
    model_size = args.model_size
    dataset = args.path_to_dataset
    checkpoint_path = args.checkpoint_path

    with_velocity = False
    n_gpus = max(torch.cuda.device_count(), 1)

    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if model_size == 'large' else 6

    default_name = (f"m2c_perlayer_fusion_mask_v1.0_{model_size}_gnl{gnl}"
                    f"_batch_{batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CPerLayerFusionMask(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
    )
    print(f'MoE enabled per block: {net.fusion_blocks[0].layer_mel_pass.config.moe}')
    print(f'Global depth: {gnl} fusion blocks '
          f'(each = 2 single-layer transformers + per-block gates)')
    print(f'Training objective: mask-predict (random per-block-per-modality '
          f'masking, loss only on masked positions)')
    print(f'Loss weights: mel={args.mel_loss_weight}, '
          f'chord={args.acc_loss_weight}')

    train_set_loader = DataLoader(
        FramedDataset(dataset, TRAIN_LENGTH, batch_size, split='train'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    val_set_loader = DataLoader(
        FramedDataset(dataset, TRAIN_LENGTH, batch_size, split='val'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss', save_top_k=5, save_last=True,
        enable_version_counter=False,
        dirpath=f'ckpt/{model_name}',
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
                config={
                    'batch_size': batch_size,
                    'model_size': model_size,
                    'train_length': TRAIN_LENGTH,
                    'variant': 'm2c_per_layer_fusion_mask',
                    'global_num_layers': gnl,
                    'mel_loss_weight': args.mel_loss_weight,
                    'acc_loss_weight': args.acc_loss_weight,
                },
            ) if args.wandb else TensorBoardLogger('tb_logs', name=model_name)
        ),
        num_sanity_val_steps=0 if checkpoint_path is not None else 2,
        strategy=strategy,
    )
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    torch.save(
        {
            'state_dict': net.state_dict(),
            'hyper_parameters': {
                'model_size': model_size,
                'with_velocity': with_velocity,
                'moe_num_experts': args.moe_num_experts,
                'moe_topk': args.moe_topk,
                'moe_intermediate_size': args.moe_intermediate_size,
                'global_num_layers': gnl,
                'mel_loss_weight': args.mel_loss_weight,
                'acc_loss_weight': args.acc_loss_weight,
                'variant': 'm2c_per_layer_fusion_mask',
            },
        },
        f'ckpt/{model_name}.pt',
    )
