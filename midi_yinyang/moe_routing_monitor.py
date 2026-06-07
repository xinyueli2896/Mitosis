"""MoE routing-probability monitor for M2CJointAttn.

Captures one held-out (drum, non-drum) batch at training start, then
every --moe-monitor-every-n-steps runs a forward pass and reads
`self._last_routing_probs` from every SimpleMoEFFN layer. Splits by
position parity (drum = even, non-drum = odd in the interleaved 2T
sequence) and logs per-layer per-modality mean expert-prob to wandb.

What to look for:
  - At init: drum and non-drum should have NEARLY IDENTICAL routing
    distributions (the gate hasn't trained yet).
  - As training progresses, if experts specialize per modality, the
    distributions should diverge: e.g. drum tokens concentrate on
    expert 0/1, non-drum on 2/3. If they remain identical, the
    router is staying uniform and the MoE FFN is wasted parameters
    (it behaves like a dense FFN with K*ffn_dim hidden size).

Loaded automatically by cp_transformer_m2c_jointattn.py when the
--moe-monitor-every-n-steps flag is positive.
"""

from __future__ import annotations

import torch


class MoERoutingMonitor:
    """Lightning callback. Lazy-imports lightning so importing this module
    doesn't require it (mirrors DumpInputSamplesCallback)."""

    def __init__(self, every_n_steps=500, n_samples=4):
        try:
            from lightning.pytorch.callbacks import Callback as _Callback
        except ImportError:
            from pytorch_lightning.callbacks import Callback as _Callback
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self._heldout = None

        outer = self

        class _Cb(_Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if outer._heldout is None:
                    # Cache the first batch for repeated re-use. Take only
                    # the first n_samples so the monitor forward is cheap.
                    x_mel, x_chord, pitch_shift = batch
                    n = min(outer.n_samples, x_mel.size(0))
                    outer._heldout = (
                        x_mel[:n].detach().clone(),
                        x_chord[:n].detach().clone(),
                        pitch_shift[:n].detach().clone(),
                    )

                step = trainer.global_step
                if step == 0 or step % outer.every_n_steps != 0:
                    return

                outer._log_routing(trainer, pl_module)

        self._cb = _Cb()

    def _log_routing(self, trainer, pl_module):
        x_mel, x_chord, pitch_shift = self._heldout
        device = next(pl_module.parameters()).device
        x_mel = x_mel.to(device)
        x_chord = x_chord.to(device)
        pitch_shift = pitch_shift.to(device)

        pl_module.eval()
        try:
            with torch.no_grad():
                x_m, x_c = pl_module.preprocess(x_mel, pitch_shift, y=x_chord)
                stacked = torch.stack([x_m, x_c], dim=2)
                B, T, _, S = stacked.shape
                x = stacked.view(B, T * 2, S)
                pl_module(x)
        finally:
            pl_module.train()

        # Walk through global_layers and read _last_routing_probs from each
        # MoE FFN. Layer.ffn.* is either SimpleMoEFFN (if use_moe) or a
        # plain Sequential (dense). Only the former has the cached attr.
        scalars = {}
        for layer_idx, layer in enumerate(pl_module.global_layers):
            ffn = getattr(layer, 'ffn', None)
            probs = getattr(ffn, '_last_routing_probs', None) if ffn is not None else None
            if probs is None:
                continue
            # probs: [B, 2T, E]. Even = drum (mel), odd = non-drum (chord).
            drum_probs = probs[:, 0::2, :].mean(dim=(0, 1))          # [E]
            nondrum_probs = probs[:, 1::2, :].mean(dim=(0, 1))       # [E]
            E = drum_probs.size(0)
            for e in range(E):
                scalars[f'routing/L{layer_idx:02d}/drum_e{e}'] = drum_probs[e].item()
                scalars[f'routing/L{layer_idx:02d}/nondrum_e{e}'] = nondrum_probs[e].item()
            # Also log the L1 distance between drum and non-drum routing
            # distributions at this layer. 0 = identical, 2 = disjoint.
            l1 = (drum_probs - nondrum_probs).abs().sum().item()
            scalars[f'routing/L{layer_idx:02d}/drum_vs_nondrum_l1'] = l1

        if scalars and trainer.logger is not None:
            # Use the trainer's logger so the scalars land in the wandb
            # run (or whichever logger is active).
            try:
                pl_module.log_dict(scalars, on_step=True, on_epoch=False)
            except Exception:
                # Fallback: direct wandb.log if log_dict isn't available
                # outside a training_step.
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(scalars, step=trainer.global_step)
                except Exception:
                    pass

    def as_callback(self):
        return self._cb
