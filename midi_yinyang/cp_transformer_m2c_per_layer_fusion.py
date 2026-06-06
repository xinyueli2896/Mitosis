"""Per-layer fusion variant of the m2c MoE model.

What's different from cp_transformer_m2c_moe.py:

  - Stream fusion happens at EVERY global layer instead of only at the end.
    Each fusion block contains TWO single-layer (attn + MoE + LN) sublayers
    (one per attention pass), gates the two outputs together, and feeds the
    fused result to the next block.
  - Same-timestep cross-stream attention is ALLOWED (m2c MoE blocked it). At
    timestep t, A's cross-attention reads B_{0..t} including B at the same
    timestep, and symmetrically for B. Matches the user spec:

        u_{2t-1}  = SelfAttn(h_1, h_3, ..., h_{2t-1})    # A self,  reads A_{1..t}
        u'_{2t-1} = SelfAttn(h_2, h_4, ..., h_{2t})      # A cross, reads B_{1..t}
        u_{2t}    = SelfAttn(h_2, h_4, ..., h_{2t})      # B self,  reads B_{1..t}
        u'_{2t}   = SelfAttn(h_1, h_3, ..., h_{2t-1})    # B cross, reads A_{1..t}

        p_{2t-1}  = MoE(u_{2t-1})
        p'_{2t-1} = MoE(u'_{2t-1})
        p_{2t}    = MoE(u_{2t})
        p'_{2t}   = MoE(u'_{2t})

        o_{2t-1}  = p_{2t-1} + sigmoid(gate_m) * p'_{2t-1}     # PER LAYER
        o_{2t}    = p_{2t}   + sigmoid(gate_c) * p'_{2t}       # PER LAYER

Implementation note: the dual-mask trick from m2c MoE gives us all four
outputs naturally:
  mel-keys pass at mel rows -> u_mm (A self)
  mel-keys pass at chord rows -> u_cm (B cross, i.e. u'_{2t} in the spec)
  chord-keys pass at mel rows -> u_mc (A cross, i.e. u'_{2t-1})
  chord-keys pass at chord rows -> u_cc (B self)
So we use two RoFormerLayer-inside-Encoder modules per fusion block. Their
internal MoE FFNs handle p (self) and p' (cross) for one stream each. The
per-block gates combine self and cross outputs into the next block's input.

Inference works with the same shift-by-2 sampling loop the m2c MoE inference
script uses. The "same-step cross-stream attention" the model allows at
training only ever reads ALREADY-SAMPLED tokens at inference: positions 2t
and 2t+1 of the inference buffer both hold the previously sampled m_{t-1}
and c_{t-1}, so when predicting m_t and c_t in parallel from the last buffer
pair, neither prediction depends on the not-yet-sampled counterpart. See
cp_transformer_m2c_per_layer_fusion_inference.py.

Run:
    python cp_transformer_m2c_per_layer_fusion.py \\
        --batch_size 8 \\
        --model_size large \\
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
import copy
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


class PerLayerFusionBlock(nn.Module):
    """One block of per-layer fusion.

    Internally holds two single-layer RoFormerEncoders -- one for each
    attention pass (mel-keys-only, chord-keys-only) -- each with its own
    attention + MoE FFN + LayerNorms (residual handled by the encoder). The
    block:

      1. runs both passes on the interleaved input h [B, 2T, H]
      2. slices the four outputs (u_mm, u_cm, u_mc, u_cc)
      3. gates per stream: o_m = u_mm + sigmoid(gate_m(h_m)) * u_mc, sym for c
      4. reassembles the interleaved output [B, 2T, H] for the next block

    Mapping to the user spec:
      mel-pass at mel rows   = u_mm = p_{2t-1}  (A self)
      mel-pass at chord rows = u_cm = p'_{2t}   (B cross, reads A keys)
      chord-pass at mel rows = u_mc = p'_{2t-1} (A cross, reads B keys)
      chord-pass at chord rows = u_cc = p_{2t}  (B self)

    The MoE FFN inside each encoder layer handles p (self) and p' (cross)
    for ONE stream within the same pass -- p is at one row, p' is at the
    other, but they share Q/K/V and MoE weights within the pass.
    """

    def __init__(self, single_layer_config):
        super().__init__()
        # Each "encoder" here is a 1-layer wrapper that gives us attention +
        # MoE + LN with RoPE inside -- exactly one transformer sublayer.
        self.layer_mel_pass = RoFormerEncoder(single_layer_config)
        self.layer_chord_pass = RoFormerEncoder(single_layer_config)
        H = single_layer_config.hidden_size
        self.gate_m = nn.Linear(H, 1)
        self.gate_c = nn.Linear(H, 1)

    def forward(self, h, mask_mel, mask_chord):
        out_mel = self.layer_mel_pass(h, attention_mask=mask_mel).last_hidden_state
        out_chord = self.layer_chord_pass(h, attention_mask=mask_chord).last_hidden_state

        u_mm = out_mel[:, 0::2, :]      # A self
        u_cm = out_mel[:, 1::2, :]      # B cross (B-Q reads A-K)
        u_mc = out_chord[:, 0::2, :]    # A cross (A-Q reads B-K)
        u_cc = out_chord[:, 1::2, :]    # B self

        h_m_in = h[:, 0::2, :]
        h_c_in = h[:, 1::2, :]

        o_m = u_mm + torch.sigmoid(self.gate_m(h_m_in)) * u_mc
        o_c = u_cc + torch.sigmoid(self.gate_c(h_c_in)) * u_cm

        out = torch.zeros_like(h)
        out[:, 0::2, :] = o_m
        out[:, 1::2, :] = o_c
        return out


class M2CPerLayerFusion(RoFormerSymbolicTransformer):
    """Per-layer-fusion variant. Stacks `global_num_layers` PerLayerFusionBlocks
    in place of the parent's single L-layer global_roformer.

    Reuses everything else from the parent: local encoder/decoder, CP
    tokenizer, embeddings, final decoder, loss, sampling harness,
    FramedDataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build a SINGLE-layer copy of the parent's global config to use
        # inside each PerLayerFusionBlock.
        global_config = self.global_roformer.config
        single_layer_config = copy.copy(global_config)
        single_layer_config.num_hidden_layers = 1

        # Replace the L-layer global backbone with L fusion blocks. The parent
        # already built `global_roformer` -- we keep it allocated but unused
        # so its parameters are still visible to optimizer state (they're
        # garbage-collected on first state_dict load if a fresh checkpoint
        # is loaded, but for now this is a fresh build).
        # Actually delete it to avoid wasted memory and confusing state_dict.
        del self.global_roformer

        self.fusion_blocks = nn.ModuleList([
            PerLayerFusionBlock(single_layer_config)
            for _ in range(self.global_num_layers)
        ])

        # The parent's gate_m / gate_c are now unused (the gates live inside
        # each fusion block). Delete them to keep state_dict clean.
        # Their per-block replacements have the same role.
        del self.gate_m
        del self.gate_c

    def _global_interaction(self, h):
        """Stack of per-layer fusion blocks.

        h: [B, 2T, H], interleaved [m_0, c_0, m_1, c_1, ...]
        Returns (out, aux_loss).
        """
        B, L, H = h.shape
        assert L % 2 == 0
        device = h.device

        # Build the per-pass key-restriction masks once. Same-step
        # cross-stream is NOT blocked here (unlike m2c MoE), so each
        # stream can read the other at the same timestep.
        idx = torch.arange(L, device=device)
        t = idx // 2
        base_mask = torch.zeros((L, L), device=device, dtype=torch.float32)
        base_mask[t[None, :] > t[:, None]] = float("-inf")   # cross-timestep causal

        mel_keys = (idx % 2 == 0)
        chord_keys = ~mel_keys

        mask_mel = base_mask.clone()
        mask_mel[:, ~mel_keys] = float("-inf")               # only mel keys visible

        mask_chord = base_mask.clone()
        mask_chord[:, ~chord_keys] = float("-inf")           # only chord keys visible

        # Run the fusion stack. Each block fuses self+cross per stream and
        # passes the combined interleaved output forward.
        h_cur = h
        for block in self.fusion_blocks:
            h_cur = block(h_cur, mask_mel, mask_chord)

        # Collect Switch-Transformer load-balancing losses from every MoE
        # block in the fusion stack (the fork stores the per-MoE loss on
        # last_balance_loss after each forward). Sum into a single aux_loss
        # so the existing loss path picks it up with the 0.01 coefficient.
        # Without this, the router has no signal to spread tokens across
        # experts and tends to collapse to 1-2 always-on experts.
        aux_loss = h.new_zeros(())
        n_collected = 0
        for m in self.modules():
            bl = getattr(m, 'last_balance_loss', None)
            if bl is not None:
                aux_loss = aux_loss + bl
                n_collected += 1
        if n_collected > 0:
            aux_loss = aux_loss / n_collected  # average across MoE blocks
        return h_cur, aux_loss


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the per-layer fusion M2C MoE model. Cross-stream '
                    'information is gated into each layer instead of only at '
                    'end-of-stack, and same-step cross-stream attention is '
                    'allowed (symmetric).',
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
    parser.add_argument('--acc_loss_weight', type=float, default=3.0)
    parser.add_argument('--preserve_program', action='store_true', default=False,
                        help='Preserve actual per-note program in the a-slot '
                             'token. Use for LAMD-style multi-instrument '
                             'streams. Default False (POP909-style hardcode).')
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

    default_name = (f"m2c_perlayer_fusion_v1.0_{model_size}_gnl{gnl}"
                    f"_batch_{batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CPerLayerFusion(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
        preserve_program=args.preserve_program,
    )
    print(f'MoE enabled per block: {net.fusion_blocks[0].layer_mel_pass.config.moe}')
    print(f'Global depth: {gnl} fusion blocks '
          f'(each = 2 single-layer transformers + gates)')
    print(f'Same-step cross-stream attention: ALLOWED (symmetric)')
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
                    'variant': 'm2c_per_layer_fusion',
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
                'variant': 'm2c_per_layer_fusion',
            },
        },
        f'ckpt/{model_name}.pt',
    )
