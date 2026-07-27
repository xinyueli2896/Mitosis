"""Finetune the pretrained single-stream CP transformer (S0) on POP909
melody+chord, with wandb logging -- FOR THE UPSTREAM midi-function-alignment
REPO. cp_transformer.py itself needs NO edits: this driver imports its
model/dataset/trainer pieces unchanged and overrides, from the outside,
exactly the four things that make the stock __main__ unsuitable for
finetuning:

  1. data:      POP909 (one .pt, proper train/val splits) instead of the
                hardcoded LA-train / RWC-val pair (both split='all').
  2. schedule:  OneCycle over --lr_total_steps (default 20k) instead of
                MAX_STEPS=2M (patched via cp_transformer.MAX_STEPS before
                the model is built).
  3. init:      WEIGHTS-ONLY load of the pretrained ckpt instead of the
                stock full-Lightning resume (which would restore the
                finished pretraining schedule at LR ~= 0).
  4. run name:  ..._pop909ft_... so ckpts never collide with the
                pretrained run's directory.

Plus wandb logging (--wandb_project / WANDB_PROJECT env; --no_wandb for
TensorBoard fallback).

Usage (single node, uses all visible GPUs; Lightning spawns DDP itself):
    python finetune_pop909.py \
        --pretrained ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import cp_transformer
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, TRAIN_LENGTH


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/pop909_melchord_cp16_v2.pt')
    p.add_argument('--pretrained',
                   default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt')
    p.add_argument('--resume', default=None,
                   help='full Lightning resume of a PRIOR finetune run')
    p.add_argument('--batch_size', type=int, default=12,
                   help='per-GPU; 12 x 4 GPUs = pretraining global batch 48')
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--max_lr', type=float, default=2e-5)
    p.add_argument('--lr_total_steps', type=int, default=20000)
    p.add_argument('--run_tag', default=None)
    p.add_argument('--save_top_k', type=int, default=5)
    p.add_argument('--val_check_interval', type=int, default=250)
    p.add_argument('--wandb_project',
                   default=os.environ.get('WANDB_PROJECT', 'MusicMOE'))
    p.add_argument('--wandb_dir', default=os.environ.get('WANDB_DIR', '/tmp/wandb'))
    p.add_argument('--no_wandb', action='store_true', default=False)
    args = p.parse_args()

    # OneCycleLR in configure_optimizers spans cp_transformer.MAX_STEPS;
    # rescale it to the finetune horizon before building the model.
    cp_transformer.MAX_STEPS = args.lr_total_steps

    n_gpus = max(torch.cuda.device_count(), 1)
    tag = f'_{args.run_tag}' if args.run_tag else ''
    model_name = (f'cp_transformer_v0.42_size{args.model_size}_pop909ft{tag}_'
                  f'batch_{args.batch_size * n_gpus}_schedule')

    net = RoFormerSymbolicTransformer(size=args.model_size, max_lr=args.max_lr)

    if args.resume is None:
        if not os.path.exists(args.pretrained):
            raise SystemExit(f'pretrained ckpt not found: {args.pretrained}')
        ck = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
        net.load_state_dict(state, strict=True)
        print(f'[init] loaded pretrained weights from {args.pretrained}')

    train_loader = DataLoader(
        FramedDataset(args.data, TRAIN_LENGTH, args.batch_size, split='train'),
        batch_size=None, num_workers=1, persistent_workers=True)
    val_loader = DataLoader(
        FramedDataset(args.data, TRAIN_LENGTH, args.batch_size, split='val'),
        batch_size=None, num_workers=1, persistent_workers=True)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss', save_top_k=args.save_top_k, save_last=True,
        enable_version_counter=False,
        dirpath=f'ckpt/{model_name}',
        filename=model_name + '.{epoch:02d}.{val_loss:.5f}')

    if args.no_wandb:
        logger = TensorBoardLogger('tb_logs', name=model_name)
    else:
        os.makedirs(args.wandb_dir, exist_ok=True)
        logger = WandbLogger(
            name=model_name, project=args.wandb_project, save_dir=args.wandb_dir,
            config={
                'task': 'pop909_melchord_finetune',
                'data': args.data,
                'pretrained': args.pretrained,
                'batch_size': args.batch_size,
                'global_batch': args.batch_size * n_gpus,
                'model_size': args.model_size,
                'max_lr': args.max_lr,
                'lr_total_steps': args.lr_total_steps,
                'train_length': TRAIN_LENGTH,
            })

    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'

    trainer = L.Trainer(
        devices=-1 if torch.cuda.is_available() else 1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=args.lr_total_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        val_check_interval=args.val_check_interval,
        limit_val_batches=10,
        check_val_every_n_epoch=None,
        logger=logger,
        num_sanity_val_steps=0 if args.resume is not None else 2,
        strategy=strategy)
    trainer.fit(net, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
