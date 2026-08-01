"""M2CDuetAnticipatory -- DuetAttn (#2) with drum shifted AHEAD of nondrum
by k frames in the interleaved input.

Inspired by the Anticipatory Music Transformer (Thickstun et al. 2023):
when one modality is conditionally salient (drum as a "score" or
control track), feeding it AHEAD of the other in the AR sequence lets
the model plan the conditioned modality (nondrum) with knowledge of
near-future drum context. No new parameters, no new attention math --
just a data-side reindexing.

Sequence semantics (length 2T_full, same as DuetAttn):

   Standard DuetAttn (#2):
     [ drum_0, nondrum_0, drum_1, nondrum_1, …, drum_{T-1}, nondrum_{T-1} ]

   Anticipatory (this variant, k frames ahead):
     [ drum_k, nondrum_0, drum_{k+1}, nondrum_1, …, drum_{T-1}, nondrum_{T-k-1},
       PAD,   nondrum_{T-k}, PAD, nondrum_{T-k+1}, …, PAD, nondrum_{T-1} ]

At interleaved position 2t the slot contains drum_{t+k} (or PAD if
t+k ≥ T). At interleaved position 2t+1 the slot contains nondrum_t.

Standard causal AR is preserved on this reindexed sequence: position p
attends to positions ≤ p. Concretely:

  * nondrum_t (at position 2t+1) attends to drum_k..drum_{t+k} -- i.e.
    the model can use drum context up through k frames ahead when
    predicting nondrum_t.
  * drum_{t+k} (at position 2t) attends to drum_k..drum_{t-1+k} +
    nondrum_0..nondrum_{t-1} -- standard drum AR, just relabeled.

Architecture identical to M2CIntraCrossAttn (#2): per-modality Q/K/V/O,
2 SDPA passes per block (intra + cross), per-block cross gate, shared
MoE FFN. Loss is standard CE; PAD targets are ignored.

Warm-start ckpts of M2CIntraCrossAttn (#2) load directly into this
variant -- the state dict is identical. Only the loss-time data
reindexing differs.

Inference note: the model now predicts drum k frames AHEAD of the
nondrum timeline. Either generate drum ahead and feed it as
"anticipated" context, or accept that the output drum stream is k
frames offset from the nondrum stream and trim accordingly. Custom
inference loop not yet written.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_transformer_m2c_moe import (
    FramedDataset, TRAIN_LENGTH, MAX_STEPS,
)
from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn
from tasks import get_task, TASKS


class M2CDuetAnticipatory(M2CIntraCrossAttn):
    """DuetAttn with drum shifted ahead by `anticipation_frames` frames in
    the interleaved input. Architecture is identical to the parent;
    only `preprocess()` is overridden to do the shift.

    `anticipation_frames` is in units of frames per modality (e.g. 16
    if your dataset has 1 frame per 16th note and you want 1 bar
    lookahead). The shift applies to the drum (mod-a / "x_mel") stream
    only; nondrum (mod-b / "x_acc") is unchanged.
    """

    def __init__(self, *args, anticipation_frames=16, **kwargs):
        super().__init__(*args, **kwargs)
        # Stored on the module so it shows up in trainer hyperparameters
        # and on the wandb config dump.
        self.anticipation_frames = int(anticipation_frames)

    def preprocess(self, x, pitch_shift, tuple_size=4, y=None):
        """Run the parent's preprocess, then shift the drum (x = mod-a)
        stream ahead by `anticipation_frames` frames. PAD-fill the
        trailing slots that fall off the end."""
        result = super().preprocess(x, pitch_shift, tuple_size=tuple_size, y=y)
        # Parent returns either x_proc (y=None) or (x_proc, y_proc).
        #
        # ONLY apply the anticipatory shift in the PAIRED case (training
        # batches, where x is guaranteed to be the mod-a/drum stream of
        # an interleaved pair). Single-stream calls (y=None) come from
        # generic tokenization -- e.g. _load_prompt_tokens at inference,
        # which loads DRUM and NONDRUM prompt files through this same
        # method. Shifting those corrupted every inference load: the
        # nondrum prompt lost its first k frames (heard as "the
        # beginning of the piece is missing") and the drum condition
        # was shifted TWICE (load-time + decode-loop), feeding
        # drum_{t+2k} instead of drum_{t+k}.
        if y is None:
            return result
        x_proc, y_proc = result

        k = self.anticipation_frames
        if k > 0:
            B, T, S = x_proc.shape
            if k < T:
                pad_frames = torch.full(
                    (B, k, S), self.tokenizer.pad_token,
                    dtype=x_proc.dtype, device=x_proc.device,
                )
                x_proc = torch.cat([x_proc[:, k:], pad_frames], dim=1)
            else:
                # k >= T: every drum slot becomes PAD. Should never
                # happen in practice (T is 192+ frames, k is small),
                # but guard anyway.
                x_proc = torch.full_like(x_proc, self.tokenizer.pad_token)

        if y is None:
            return x_proc
        return x_proc, y_proc


# ---------------------------------------------------------------------------
# Training entry point (mirrors intra-cross-attn CLI; adds --anticipation_frames)
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
        description='Train M2CDuetAnticipatory (DuetAttn with drum '
                    'shifted ahead of nondrum by k frames).',
    )
    parser.add_argument('--task', type=str, required=True, choices=sorted(TASKS))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_size', type=str, default='large',
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
    parser.add_argument('--val_check_interval', type=int, default=500,
                        help='steps between val evaluations. On the small '
                             'melchord corpora the val minimum can arrive '
                             'within the first ~1k steps, which 500 '
                             'resolves with only one or two points -- too '
                             'coarse to tell a real minimum from a '
                             'monotonic rise.')
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
    parser.add_argument('--anticipation_frames', type=int, default=16,
                        help='Number of frames per modality by which to '
                             'shift the drum stream ahead of the nondrum '
                             'stream in the interleaved input. Default 16 '
                             '(= 1 bar at 16th-note resolution).')
    parser.add_argument('--fresh_schedule', action='store_true', default=False)
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    task = get_task(args.task)
    mod_a_path = args.mod_a_path if args.mod_a_path is not None else task.mod_a_path
    mod_b_path = args.path_to_dataset if args.path_to_dataset is not None else task.mod_b_path

    tag = f'_{args.run_tag}' if args.run_tag else ''
    default_name = (
        f"m2c_duet_anticipatory_v1.0_{args.model_size}_"
        f"gnl{gnl}_{task.name}_k{args.anticipation_frames}{tag}_"
        f"batch_{args.batch_size * n_gpus}_schedule"
    )
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  mod_a={task.mod_a_label}  mod_b={task.mod_b_label}')

    net = M2CDuetAnticipatory(
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
        anticipation_frames=args.anticipation_frames,
    )
    print(f'Architecture: M2CDuetAnticipatory (DuetAttn architecture; drum '
          f'shifted +{args.anticipation_frames} frames in interleaved input)')
    print(f'Global depth: {gnl}   gate_init_bias: {args.gate_init_bias}   '
          f'anticipation_frames: {args.anticipation_frames}')

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
        strategy = strategies.DDPStrategy(
            timeout=datetime.timedelta(hours=2),
            find_unused_parameters=False,
        )
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
                    'variant': 'm2c_duet_anticipatory',
                    'task': task.name,
                    'mod_a_label': task.mod_a_label,
                    'mod_b_label': task.mod_b_label,
                    'global_num_layers': gnl,
                    'moe_num_experts': args.moe_num_experts,
                    'moe_topk': args.moe_topk,
                    'gate_init_bias': args.gate_init_bias,
                    'anticipation_frames': args.anticipation_frames,
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
            ckpt_path_for_resume = args.checkpoint_path
        else:
            sd = loaded['state_dict'] if isinstance(loaded, dict) and 'state_dict' in loaded else loaded
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                print(f'[init] {len(missing)} missing (first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected (first few: {unexpected[:3]})')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
    torch.save(net.state_dict(), f'{ckpt_dir}/{model_name}.fin.ckpt')
