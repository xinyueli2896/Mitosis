"""Variant of cp_transformer_m2c_moe where same-timestep cross-stream
attention is SYMMETRIC and FAIR.

Two changes vs the original m2c MoE model:

1) Drop the same-step cross-stream block in the attention mask. Both slots
   of a timestep are now mutually visible (m and c at the same buffer
   timestep can attend to each other).

2) Make the global RoPE positional encoding map positions 2t and 2t+1 to
   the SAME timestep index t. Without this, RoPE would rotate Q/K of the
   two slots by different absolute angles, giving Q_m -> K_c a relative
   rotation of -1 and Q_c -> K_m a relative rotation of +1 -- the dot
   products would not match even though the mask allowed both attentions.
   With floor(p/2) positions, same-step pairs get relative rotation 0
   (identity), so attention scores are truly symmetric across the two
   directions.

Modality identity is still encoded -- it just doesn't ride on the RoPE
position any more. The model gets that signal from the token_type
embeddings added inside `local_encode` (mel=0, chord=1), which propagate
into the per-block H-vectors that the global model consumes.

Everything else (CP local model, MoE FFN, shift-by-2 teacher forcing,
loss, FramedDataset, CLI / wandb / tensorboard wiring, sanity check) is
reused unchanged from cp_transformer_m2c_moe.py. The `gate_m`, `gate_c`
parameters from the parent constructor exist but go unused.

Run from midi_yinyang/, same CLI as cp_transformer_m2c_moe.py:

    python cp_transformer_m2c_samestep.py \
        --batch_size 8 \
        --model_size small \
        --path_to_dataset data/pop909_chord_cp4_v2.pt \
        --moe_num_experts 4 --moe_topk 2 \
        --wandb
"""

# Inject the vendored transformers fork BEFORE anything else imports.
# Without this, indirect transformers loads from pytorch_lightning / wandb
# can cache the system package and shadow the fork (would manifest as
# `ImportError: cannot import name 'RoFormerMoE'` later on).
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import os
import sys

import torch
import torch.nn as nn
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
# Imported after cp_transformer_m2c_moe so the vendored fork is on sys.path.
from transformers.models.roformer.modeling_roformer import (
    RoFormerSinusoidalPositionalEmbedding,
)


class TimestepRoPE(RoFormerSinusoidalPositionalEmbedding):
    """RoPE position table indexed by timestep = position // 2.

    On a length-2T interleaved [m_0, c_0, m_1, c_1, ...] global axis, both
    slots of a timestep get the SAME rotation. Cross-timestep relative
    positions are then `t1 - t2`, the natural timestep gap. Same-timestep
    cross-stream attention has relative position 0 (identity), so
    Q_m_k . K_c_k == Q_c_k . K_m_k under symmetric content -- truly fair.
    """

    @torch.no_grad()
    def forward(self, input_ids_shape, past_key_values_length=0):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        ) // 2
        return nn.Embedding.forward(self, positions)


class M2CSameStep(RoFormerSymbolicTransformer):
    """RoFormerSymbolicTransformer with same-timestep cross-stream attention
    that's both UNBLOCKED in the mask AND symmetric under RoPE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the global RoFormer's positional embedding so that
        # positions 2t and 2t+1 share an identical RoPE rotation.
        old = self.global_roformer.embed_positions
        new = TimestepRoPE(
            num_positions=old.num_embeddings,
            embedding_dim=old.embedding_dim,
        )
        # parent class re-runs the sinusoidal _init_weight inside __init__
        # so `new` already has the correct fixed table.
        self.global_roformer.embed_positions = new

    def _global_interaction(self, h):
        """h: [B, 2T, H]. Single attention pass with a plain timestep-causal
        mask -- no same-step cross-stream block. Combined with the override
        above, mel <-> chord attention at the same timestep is symmetric."""
        B, L, H = h.shape
        assert L % 2 == 0
        device = h.device

        idx = torch.arange(L, device=device)
        t = idx // 2
        mask = torch.zeros((L, L), device=device, dtype=torch.float32)
        mask.masked_fill_(t[None, :] > t[:, None], float("-inf"))

        out = self.global_roformer(h, attention_mask=mask)
        # The vendored fork's RoFormerEncoder doesn't expose the MoE
        # router's balance loss; synthesize zero.
        aux_loss = h.new_zeros(())
        return out.last_hidden_state, aux_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the M2C-MoE variant with symmetric same-timestep '
                    'cross-stream attention (mask + RoPE).',
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

    default_name = (f"m2c_samestep_sym_moe_v1.0_{model_size}"
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
    print('Same-step cross-stream attention: ALLOWED and SYMMETRIC '
          '(mask + RoPE share-position)')

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
                    'variant': 'same_step_symmetric_rope',
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
                'variant': 'same_step_symmetric_rope',
            },
        },
        f'ckpt/{model_name}.pt',
    )
