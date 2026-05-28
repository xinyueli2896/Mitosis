"""Variant of cp_transformer_m2c_moe with same-timestep cross-stream attention.

The original model's _global_interaction ran TWO masked attention passes (one
restricting keys to melody positions, one to chord positions), then combined
the outputs via per-token sigmoid gates `gate_m`, `gate_c`. It also explicitly
blocked same-timestep cross-stream attention except on the diagonal.

This variant keeps EVERYTHING ELSE identical:
  - local encoder / decoder (dense RoFormer, 3 layers)
  - token type embeddings (mel=0, chord=1)
  - global RoFormer with MoE (4 experts, top-2 by default)
  - shift-by-2 teacher forcing on the interleaved global axis
  - loss (mel_loss_weight=1.0, acc_loss_weight=3.0)
  - FramedDataset (same chord/.pt + melody/.pt pairing)
  - CLI / wandb / tensorboard wiring

and replaces only `_global_interaction` with a SINGLE attention pass whose
mask is plain timestep-causal — any position attends to any other position
in the same or an earlier timestep, including its same-timestep partner.

What changes for the model semantically:

  - For a melody slot at buffer-timestep t (predicting m_t), it can now
    directly read the chord-input at position 2t+1 (which carries c_{t-1}).
    Symmetric for chord slots.
  - The within-block attention is bidirectional across modalities at every
    layer, so the past timestep summaries (m_{k-1}, c_{k-1}) jointly
    contextualize each other inside the global transformer.
  - At inference m_t and c_t are still independent given history (they're
    outputs of one parallel pass, neither is in the input). If you want
    joint sampling, use MaskGIT-style refinement on top.

The `gate_m` and `gate_c` parameters from the parent class are still
constructed (we subclass) but go unused. They're a few hundred parameters,
immaterial; if you want to strip them, do it in a follow-up.

Run from midi_yinyang/, same CLI as cp_transformer_m2c_moe.py:

    python cp_transformer_m2c_samestep.py \
        --batch_size 8 \
        --model_size small \
        --path_to_dataset data/pop909_chord_cp4_v2.pt \
        --moe_num_experts 4 --moe_topk 2 \
        --wandb
"""

import argparse
import os
import sys

import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from cp_transformer_m2c_moe import (
    RoFormerSymbolicTransformer,
    FramedDataset,
    TRAIN_LENGTH,
    MAX_STEPS,
)


class M2CSameStep(RoFormerSymbolicTransformer):
    """RoFormerSymbolicTransformer with a single-pass, same-step-allowed
    global interaction (replaces the dual-mask + gate mechanism)."""

    def _global_interaction(self, h):
        """h: [B, 2T, H], interleaved [m_0, c_0, m_1, c_1, ...].
        Returns (out, aux_loss) with the same contract as the parent."""
        B, L, H = h.shape
        assert L % 2 == 0
        device = h.device

        # Timestep-causal mask. NO same-step cross-stream block. So:
        #   - mel position at buffer-timestep t can attend to chord position at
        #     buffer-timestep t (i.e., c_{t-1} via position 2t+1 in the
        #     shift-by-2 setup), and vice versa.
        #   - All past-timestep positions are visible to both modalities.
        idx = torch.arange(L, device=device)
        t = idx // 2
        mask = torch.zeros((L, L), device=device, dtype=torch.float32)
        mask.masked_fill_(t[None, :] > t[:, None], float("-inf"))

        out = self.global_roformer(h, attention_mask=mask)
        # The vendored fork's RoFormerEncoder doesn't expose a router
        # load-balance loss, so synthesize a zero for the parent's loss
        # assembly path.
        aux_loss = h.new_zeros(())
        return out.last_hidden_state, aux_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the M2C-MoE variant that allows same-timestep '
                    'cross-stream attention.',
    )

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str,
                        help='path to the chord .pt; the melody .pt is found '
                             'by replacing "chord" -> "melody" in the path')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)

    args = parser.parse_args()

    batch_size = args.batch_size
    model_size = args.model_size
    dataset = args.path_to_dataset
    checkpoint_path = args.checkpoint_path

    with_velocity = False
    n_gpus = max(torch.cuda.device_count(), 1)

    default_name = (f"m2c_samestep_moe_v1.0_{model_size}"
                    f"_batch_{batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CSameStep(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
    )
    print(f'MoE enabled: {net.global_roformer.config.moe}')
    print('Same-step cross-stream attention: ALLOWED (no same-step block)')

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
                    'variant': 'same_step_cross_stream',
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
                'variant': 'same_step_cross_stream',
            },
        },
        f'ckpt/{model_name}.pt',
    )
