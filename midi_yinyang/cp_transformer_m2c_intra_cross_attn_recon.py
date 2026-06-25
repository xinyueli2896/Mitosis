"""M2CIntraCrossAttnRecon — intra/cross-attn + Brier-style MSE on drum logits.

Role in the lineup: **loss-shape ablation on top of DuetAttn (#2)**, NOT
a true conditioning baseline. See VARIANTS.md / IMPLEMENTATION_REPORT.md
for the full framing. Short version:

    L_total = CE_drum + CE_nondrum + lambda_recon * MSE_drum

    MSE_drum = mean_{b,t,s, non-PAD} || softmax(drum_logits_{b,t,s})
                                       - one_hot(drum_target_{b,t,s}) ||_2^2

The MSE_drum term operates on the **same** drum logits CE already
supervises and conditions on the **same** causal past CE conditions on.
Both terms push softmax(drum_logits) toward one_hot(drum_target) -- CE
via -log(p_target), Brier MSE via squared error. Different gradient
profiles, same task, same context.

Originally framed (incorrectly) as a "rehearsal-style conditioning
baseline" where the model would learn to use drum context better. That
framing was wrong: the model never sees drum context it didn't already
have under plain NTP, because the loss change doesn't change the
information available to the forward pass. To get true rehearsal-style
conditioning, the drum stream needs to be **architecturally** visible
as a prefix when nondrum is predicted -- see the proposed
M2CDuetRehearsal (#6) or the implemented M2CDuetPrefix (#5).

What this variant still answers:
  * Does the Brier-MSE gradient profile on drum logits do anything CE
    alone doesn't? (Loss-shape only ablation against #2.)

What this variant does NOT answer:
  * Anything about drum-as-condition for nondrum generation.

Architecture identical to M2CIntraCrossAttn; forward pass unchanged.
Inference uses the existing cp_transformer_m2c_intra_cross_attn_*.py
scripts with a ckpt trained here (state-dict-compatible with #2).

ckpt-incompatible with standard m2c_intra_cross_attn ONLY in the sense
that training trajectory differs (and the recon_weight attribute is on
the module); use the warm-start init from
init_pretrained_into_intra_cross_attn_recon.py to start a fresh run.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_transformer_m2c_moe import (
    FramedDataset, TRAIN_LENGTH, MAX_STEPS,
)
from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn
from tasks import get_task, TASKS


class M2CIntraCrossAttnRecon(M2CIntraCrossAttn):
    """Same architecture as M2CIntraCrossAttn, plus a Brier-style MSE
    reconstruction loss on drum (mod_a) tokens added inside loss().

    LOSS-SHAPE ABLATION, NOT A CONDITIONING BASELINE. The MSE term acts
    on the same logits CE acts on, using the same context, with a
    different gradient profile. See module docstring for the full
    framing.

    The MSE term:
      softmax(drum_logits) -- the model's predicted drum distribution
      one_hot(drum_target) -- the actual drum token at that position
      || softmax(...) - one_hot(...) ||_2^2  summed over vocab, averaged
                                              over non-PAD slots.

    Net effect on training: the drum head receives a sharper gradient
    near the correct token (Brier MSE has a wider zero-gradient region
    far from the target than CE, so the gradient near 0 confidence is
    smaller relative to CE), which may bias the head toward harder
    commitments on its top-1.

    Empirical question this variant answers: does that gradient-profile
    bias change drum prediction quality vs CE alone? It does NOT answer
    "is the model better at using drum-as-condition" -- the model never
    sees drum-as-condition under this variant.
    """

    def __init__(self, *args, recon_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Saved on the module so it shows up in trainer hyperparameters
        # and wandb config.
        self.recon_weight = float(recon_weight)

    # ------------------------------------------------------------------
    # Override loss(): standard logic from RoFormerSymbolicTransformer.loss
    # plus the drum recon term. Mostly copied from
    # cp_transformer_m2c_moe.py::RoFormerSymbolicTransformer.loss so the
    # parent's per-modality weighting and EOS/content breakdown remain.
    # ------------------------------------------------------------------

    def loss(self, x_mel, x_acc, batch_pitch_shift):
        # Steps 1-N: replicate the parent's loss method, then add MSE on
        # drum (mel-slot) logits at the end. Keeping the body inline
        # rather than calling super().loss() because the parent doesn't
        # expose logits / targets externally and we'd otherwise have to
        # forward twice.

        x_mel, x_acc = self.preprocess(x_mel, batch_pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape

        # Silence augmentation (unchanged from parent).
        if self.training and self.silence_augment_prob > 0:
            p = self.silence_augment_prob
            rand = torch.rand(batch_size, device=x_mel.device)
            silence_mel_mask = (rand < p)
            silence_acc_mask = (rand >= p) & (rand < 2 * p)
            silence_frame = torch.full(
                (1, 1, subseq_len), self.tokenizer.pad_token,
                dtype=x_mel.dtype, device=x_mel.device,
            )
            silence_frame[..., 0] = self.tokenizer.eos_token
            x_mel = torch.where(
                silence_mel_mask[:, None, None], silence_frame, x_mel,
            )
            x_acc = torch.where(
                silence_acc_mask[:, None, None], silence_frame, x_acc,
            )

        stacked = torch.stack([x_mel, x_acc], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        logits, aux_loss = self(x)
        targets = x

        # Per-frame stream weight (mel even / acc odd).
        frame_idx = torch.arange(seq_len * 2, device=targets.device)
        frame_weights = torch.where(
            frame_idx % 2 == 0,
            torch.as_tensor(self.mel_loss_weight, device=targets.device),
            torch.as_tensor(self.acc_loss_weight, device=targets.device),
        )
        weights = frame_weights.view(1, seq_len * 2, 1).expand(
            batch_size, -1, subseq_len,
        )

        per_token_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.n_tokens),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none',
        ).view(batch_size, seq_len * 2, subseq_len)

        non_pad_mask = (targets != self.tokenizer.pad_token).float()
        is_eos_mask = (targets == self.tokenizer.eos_token).float() * non_pad_mask
        is_content_mask = non_pad_mask * (1.0 - is_eos_mask)

        token_type_weight = 1.0 + (self.eos_loss_weight - 1.0) * is_eos_mask
        weighted_loss = per_token_loss * weights * token_type_weight * non_pad_mask
        normalizer = (weights * token_type_weight * non_pad_mask).sum().clamp_min(1.0)
        ce_loss = weighted_loss.sum() / normalizer

        content_n = is_content_mask.sum().clamp_min(1.0)
        eos_n = is_eos_mask.sum().clamp_min(1.0)
        ce_loss_content = (per_token_loss * is_content_mask).sum() / content_n
        ce_loss_eos = (per_token_loss * is_eos_mask).sum() / eos_n

        # ------------------------------------------------------------------
        # NEW: token-level L2 reconstruction on drum (mel-slot) tokens.
        # The transformer flattens (2T, S) into a single sequence dim, so
        # logits comes back as [B, 2T*S, V]. Fold it to [B, 2T, S, V] before
        # picking the drum (even-frame) rows.
        # ------------------------------------------------------------------
        V = self.tokenizer.n_tokens
        logits_4d = logits.view(batch_size, seq_len * 2, subseq_len, V)
        drum_logits = logits_4d[:, 0::2, :, :]               # [B, T, S, V]
        drum_targets = targets[:, 0::2, :]                   # [B, T, S]
        drum_non_pad = (drum_targets != self.tokenizer.pad_token).float()

        # Brier-style MSE: softmax(probs) vs one-hot target. Numerically
        # well-behaved (probs in [0,1]) unlike raw-logit MSE.
        drum_probs = F.softmax(drum_logits, dim=-1)          # [B, T, S, V]
        # F.one_hot ignores PAD index by clamp + mask later. Clamp negative
        # indices first if PAD ever shows up as something out-of-range
        # (it shouldn't but be defensive).
        safe_targets = drum_targets.clamp(min=0, max=self.tokenizer.n_tokens - 1)
        one_hot = F.one_hot(safe_targets, num_classes=self.tokenizer.n_tokens).float()
        mse_per_slot = ((drum_probs - one_hot) ** 2).sum(dim=-1)   # [B, T, S]
        recon_loss = (mse_per_slot * drum_non_pad).sum() / drum_non_pad.sum().clamp_min(1.0)

        # ------------------------------------------------------------------
        # Stash for training_step / validation_step to log.
        # ------------------------------------------------------------------
        self._last_ce_loss = ce_loss.detach()
        self._last_ce_loss_content = ce_loss_content.detach()
        self._last_ce_loss_eos = ce_loss_eos.detach()
        self._last_recon_loss = recon_loss.detach()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
        else:
            aux_loss = ce_loss.new_zeros(())

        total_loss = (
            ce_loss
            + self.aux_loss_weight * aux_loss
            + self.recon_weight * recon_loss
        )

        return total_loss, aux_loss

    # ------------------------------------------------------------------
    # Override training/validation hooks so recon_loss shows up on wandb.
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_ce_loss', self._last_ce_loss)
        self.log('train_ce_loss_content', self._last_ce_loss_content)
        self.log('train_ce_loss_eos', self._last_ce_loss_eos)
        self.log('train_recon_loss', self._last_recon_loss)
        if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad is False:
            pass
        self.log('train_moe_aux_loss', aux_loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ce_loss', self._last_ce_loss)
        self.log('val_ce_loss_content', self._last_ce_loss_content)
        self.log('val_ce_loss_eos', self._last_ce_loss_eos)
        self.log('val_recon_loss', self._last_recon_loss)
        self.log('val_moe_aux_loss', aux_loss.detach())
        return loss


# ---------------------------------------------------------------------------
# Training entry point (mirrors intra-cross-attn, adds --recon_weight)
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
        description='Train M2CIntraCrossAttnRecon (intra-cross-attn + '
                    'token-level L2 drum reconstruction).',
    )
    parser.add_argument('--task', type=str, required=True,
                        choices=sorted(TASKS))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_size', type=str, default='small',
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
    parser.add_argument('--silence_augment_prob', type=float, default=0.0)
    parser.add_argument('--moe_monitor_every_n_steps', type=int, default=0)
    parser.add_argument('--moe_monitor_n_samples', type=int, default=4)
    parser.add_argument('--dump_samples_dir', type=str, default=None)
    parser.add_argument('--dump_samples_n', type=int, default=4)
    parser.add_argument('--dump_samples_every_n_epochs', type=int, default=None)
    parser.add_argument('--max_polyphony', type=int, default=16)
    parser.add_argument('--gate_init_bias', type=float, default=-10.0)
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Weight on the Brier-style MSE drum '
                             'reconstruction loss. Default 1.0. '
                             'Total loss = CE + lambda_aux * aux + '
                             'recon_weight * MSE_drum.')
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    task = get_task(args.task)
    mod_a_path = args.mod_a_path if args.mod_a_path is not None else task.mod_a_path
    mod_b_path = args.path_to_dataset if args.path_to_dataset is not None else task.mod_b_path

    tag = f'_{args.run_tag}' if args.run_tag else ''
    default_name = (f"m2c_intra_cross_attn_recon_v1.0_{args.model_size}_"
                    f"gnl{gnl}_{task.name}{tag}_"
                    f"batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  '
          f'mod_a={task.mod_a_label} (program {task.mod_a_default_program}, '
          f'{mod_a_path})  '
          f'mod_b={task.mod_b_label} (program {task.mod_b_default_program}, '
          f'{mod_b_path})')

    net = M2CIntraCrossAttnRecon(
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
        gate_init_bias=args.gate_init_bias,
        recon_weight=args.recon_weight,
    )
    print(f'Architecture: M2CIntraCrossAttn + drum-recon MSE')
    print(f'Global depth: {gnl} layers   gate_init_bias: {args.gate_init_bias}')
    print(f'recon_weight: {args.recon_weight}')

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

    trainer = L.Trainer(
        devices=n_gpus,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=(args.lr_total_steps if args.lr_total_steps is not None else MAX_STEPS),
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
                    'variant': 'm2c_intra_cross_attn_recon',
                    'task': task.name,
                    'mod_a_label': task.mod_a_label,
                    'mod_b_label': task.mod_b_label,
                    'mod_a_path': mod_a_path,
                    'mod_b_path': mod_b_path,
                    'global_num_layers': gnl,
                    'moe_num_experts': args.moe_num_experts,
                    'moe_topk': args.moe_topk,
                    'gate_init_bias': args.gate_init_bias,
                    'recon_weight': args.recon_weight,
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
        if has_lightning_meta:
            print(f'[resume] full Lightning ckpt at {args.checkpoint_path}')
            ckpt_path_for_resume = args.checkpoint_path
        else:
            print(f'[init] bare warm-start ckpt at {args.checkpoint_path}; '
                   'loading state_dict only (no Lightning metadata).')
            sd = loaded['state_dict'] if isinstance(loaded, dict) and 'state_dict' in loaded else loaded
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                print(f'[init] {len(missing)} missing keys (first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected keys (first few: {unexpected[:3]})')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
    torch.save(net.state_dict(), f'{ckpt_dir}/{model_name}.fin.ckpt')
