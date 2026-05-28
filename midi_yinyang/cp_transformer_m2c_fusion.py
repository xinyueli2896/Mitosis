"""Two-CP-model fusion variant of the m2c MoE architecture.

This is a SEPARATE file from cp_transformer_m2c_moe.py so the original
weight-shared architecture (one global RoFormer used twice) stays unchanged.

What's different here:

  - Two PHYSICALLY SEPARATE global RoFormer backbones, `global_roformer_m`
    and `global_roformer_c`, one per stream. Each backbone is loaded from the
    same pretrained CP transformer checkpoint by init_pretrained_into_fusion.py,
    so they start bitwise-identical, then drift during training to specialize
    per modality.

  - Two-phase finetuning support via `freeze_backbone`:
      phase 1 (warmup): freeze both backbones + local_*, train ONLY the
                        cross-attention adapter (gate_m, gate_c, token_type_emb,
                        global_sos). Gates learn where cross-modality info helps
                        against a stable pretrained backbone.
      phase 2 (full):   unfreeze everything, joint finetune. Backbones diverge,
                        MoE experts within each can differentiate.

  - Inherits everything else (local encoder/decoder, CP tokenizer, dual-mask +
    gated cross-attention, loss, sampling, FramedDataset) from the parent
    `RoFormerSymbolicTransformer` in cp_transformer_m2c_moe.py.

Run from midi_yinyang/, same CLI structure as cp_transformer_m2c_moe.py plus
two new flags:

    python cp_transformer_m2c_fusion.py \\
        --batch_size 8 \\
        --model_size large \\
        --path_to_dataset data/pop909_chord_cp4_v2.pt \\
        --moe_num_experts 4 --moe_topk 2 \\
        --checkpoint_path pretrained/m2c_fusion_init_from_<name>.pt \\
        --freeze_backbone \\
        --wandb

Then phase 2 (without --freeze_backbone), resuming from a phase-1 checkpoint:

    python cp_transformer_m2c_fusion.py \\
        ... \\
        --checkpoint_path ckpt/<phase1_run_name>/last.ckpt
"""

# Vendored transformers fork goes on sys.path BEFORE anything else imports.
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import os

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
from transformers.models.roformer.modeling_roformer import RoFormerEncoder


class M2CFusion(RoFormerSymbolicTransformer):
    """Two-CP-model fusion: independent backbones per stream, joined by the
    parent's gated cross-attention adapter.

    Architecturally:
      shared: local_embedding, local_encoder, local_decoder, final_decoder,
              token_type_embeddings, global_sos, gate_m, gate_c
      per-stream: global_roformer_m, global_roformer_c

    The shared local CP encoder/decoder keep the embedding spaces of the two
    streams compatible -- without that, cross-attention between drifted
    streams would be very hard to learn.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build the second global backbone with the same RoFormerConfig the
        # parent already used. Both start from random init here; they get
        # filled with identical pretrained weights by the init script.
        global_config = self.global_roformer.config
        self.global_roformer_m = self.global_roformer
        self.global_roformer_c = RoFormerEncoder(global_config)
        # `self.global_roformer` is intentionally kept as an alias to
        # `global_roformer_m` so any inherited code (e.g. attribute reads
        # by subclasses) doesn't crash. The forward path here uses _m / _c
        # explicitly and never touches `global_roformer` again.

    def _global_interaction(self, h):
        """Same dual-mask + gated cross-attention adapter as the parent, but
        each pass runs through ITS OWN backbone.

        h: [B, 2T, H], interleaved [m_0, c_0, m_1, c_1, ...].
        Returns (out, aux_loss).
        """
        B, L, H = h.shape
        assert L % 2 == 0
        device = h.device

        # Build the masks exactly like the parent does.
        idx = torch.arange(L, device=device)
        t = idx // 2
        mask_bool = t[None, :] > t[:, None]
        base_mask = torch.zeros((L, L), device=device, dtype=torch.float32)
        base_mask[mask_bool] = float("-inf")
        same_step = t[None, :] == t[:, None]
        diag = torch.eye(L, device=device, dtype=torch.bool)
        base_mask[same_step & ~diag] = float("-inf")

        mel_keys = (idx % 2 == 0)
        chord_keys = ~mel_keys

        mask_mel = base_mask.clone()
        mask_mel[:, ~mel_keys] = float("-inf")
        mask_chord = base_mask.clone()
        mask_chord[:, ~chord_keys] = float("-inf")
        diag_idx = torch.arange(L, device=device)
        mask_mel[diag_idx, diag_idx] = 0.0
        mask_chord[diag_idx, diag_idx] = 0.0

        # KEY DIFFERENCE vs parent: separate backbones per stream.
        out_mel = self.global_roformer_m(h, attention_mask=mask_mel)
        out_chord = self.global_roformer_c(h, attention_mask=mask_chord)

        out_mel_states = out_mel.last_hidden_state
        out_chord_states = out_chord.last_hidden_state
        u_mm = out_mel_states[:, 0::2, :]    # mel Q, mel K, MEL backbone
        u_cm = out_mel_states[:, 1::2, :]    # chord Q, mel K, MEL backbone  -> feeds o_c
        u_mc = out_chord_states[:, 0::2, :]  # mel Q, chord K, CHORD backbone -> feeds o_m
        u_cc = out_chord_states[:, 1::2, :]  # chord Q, chord K, CHORD backbone

        h_m_in = h[:, 0::2, :]
        h_c_in = h[:, 1::2, :]

        o_m = u_mm + torch.sigmoid(self.gate_m(h_m_in)) * u_mc
        o_c = u_cc + torch.sigmoid(self.gate_c(h_c_in)) * u_cm

        out = torch.zeros_like(h)
        out[:, 0::2, :] = o_m
        out[:, 1::2, :] = o_c
        return out, h.new_zeros(())

    # ------------------------------------------------------------------
    # Freeze / unfreeze for two-phase finetuning
    # ------------------------------------------------------------------
    def _adapter_param_names(self):
        """The 'cross-attention adapter' = everything the fusion adds on top
        of the two pretrained CP backbones. Trained alone in phase 1."""
        return (
            'gate_m.', 'gate_c.',
            'token_type_embeddings.',
            'global_sos',  # learned vector, modality-neutral but fusion-specific
        )

    def freeze_backbone(self):
        """Phase 1: train ONLY the adapter. Everything else frozen.

        Sets `requires_grad` accordingly and stashes the trainable-param
        filter so configure_optimizers picks it up.
        """
        keep_prefixes = self._adapter_param_names()
        n_frozen = n_trained = 0
        for name, p in self.named_parameters():
            trainable = any(name == k.rstrip('.') or name.startswith(k)
                            for k in keep_prefixes)
            p.requires_grad = trainable
            if trainable:
                n_trained += p.numel()
            else:
                n_frozen += p.numel()
        print(f'[freeze] adapter-only training: '
              f'frozen={n_frozen:,}  trainable={n_trained:,}  '
              f'({100*n_trained/(n_frozen+n_trained):.2f}% trainable)')

    def unfreeze_all(self):
        """Phase 2: enable full finetune."""
        for p in self.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        """Override so AdamW only allocates state for trainable params; without
        this, --freeze_backbone wastes ~all the optimizer memory on frozen
        backbones."""
        max_lr = 1e-4
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr,
            total_steps=MAX_STEPS, pct_start=0.005,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the two-model M2C fusion: independent per-stream '
                    'global backbones joined by the gated cross-attention adapter.',
    )

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str,
                        help='path to the chord .pt; melody .pt is found by '
                             'replacing "chord" -> "melody"')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Lightning .ckpt or .pt to resume from. For phase 1 '
                             'this is typically a pretrained-init .pt from '
                             'init_pretrained_into_fusion.py. For phase 2 it is '
                             'the last.ckpt of the phase 1 run.')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--global_num_layers', type=int, default=None,
                        help='Override global transformer depth. Default: '
                             '12 if model_size=large else 6.')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Phase 1: train only the cross-attention adapter '
                             '(gate_m/c, token_type_embeddings, global_sos). '
                             'Everything else frozen.')
    parser.add_argument('--mel_loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_loss_weight', type=float, default=3.0)
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

    phase_tag = 'phase1adapter' if args.freeze_backbone else 'phase2full'
    default_name = (f"m2c_fusion_v1.0_{model_size}_gnl{gnl}_{phase_tag}"
                    f"_batch_{batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CFusion(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
    )
    print(f'MoE enabled: {net.global_roformer_m.config.moe}')
    print(f'Global backbones: 2 (untied, one per stream)')
    print(f'Loss weights: mel={args.mel_loss_weight}, '
          f'chord={args.acc_loss_weight}')

    if args.freeze_backbone:
        net.freeze_backbone()

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
                    'variant': 'm2c_fusion_two_backbones',
                    'global_num_layers': gnl,
                    'freeze_backbone': args.freeze_backbone,
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
                'freeze_backbone': args.freeze_backbone,
                'mel_loss_weight': args.mel_loss_weight,
                'acc_loss_weight': args.acc_loss_weight,
                'variant': 'm2c_fusion_two_backbones',
            },
        },
        f'ckpt/{model_name}.pt',
    )
