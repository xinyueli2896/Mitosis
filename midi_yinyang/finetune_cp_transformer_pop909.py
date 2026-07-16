"""Finetune the pretrained single-stream CP transformer on POP909
melody+chord (the merged, program-tagged combined dataset).

Pipeline (see the sbatch for one-command usage):
  1. merge_melody_chord.py --chord-program 48
         POP909-melody + POP909-chord -> POP909-melody-chord-tagged
     (distinct programs; the single-stream tokenizer separates streams by
      program ONLY, so same-program streams would fuse irreversibly)
  2. preprocess_large_midi_dataset.py pop909_melchord
         -> data/pop909_melchord_cp16_v2.pt
  3. this script: load the pretrained ckpt's weights, run a SHORT fresh
     OneCycle schedule (--lr_total_steps, default 20k) at a low LR
     (--max_lr, default 2e-5), val on a held-out POP909 split
     (song_index % 10 == 1, the FramedDataset 'val' convention).

The base trainer in cp_transformer.py hardcodes LA data and a 2M-step
schedule; this entrypoint replaces the data and rescales the schedule but
touches nothing about the model.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

import cp_transformer
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, TRAIN_LENGTH


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/pop909_melchord_cp16_v2.pt')
    p.add_argument('--pretrained',
                   default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt',
                   help='ckpt whose WEIGHTS initialize the finetune')
    p.add_argument('--resume', default=None,
                   help='full Lightning resume of a prior finetune run '
                        '(optimizer+scheduler restored; --pretrained ignored)')
    p.add_argument('--batch_size', type=int, default=12,
                   help='per-GPU; 12 x 4 GPUs = the pretraining global 48')
    p.add_argument('--model_size', type=int, default=1)
    p.add_argument('--max_lr', type=float, default=2e-5,
                   help='finetune peak LR (pretraining used 1e-4)')
    p.add_argument('--lr_total_steps', type=int, default=20000)
    p.add_argument('--run_tag', default=None)
    p.add_argument('--save_top_k', type=int, default=5)
    p.add_argument('--val_check_interval', type=int, default=250)
    p.add_argument('--scratch', action='store_true',
                   help='skip the pretrained init (debugging only)')
    args = p.parse_args()

    # configure_optimizers builds OneCycleLR over cp_transformer.MAX_STEPS;
    # rescale it to the finetune's horizon BEFORE the model is constructed.
    cp_transformer.MAX_STEPS = args.lr_total_steps

    n_gpus = max(torch.cuda.device_count(), 1)
    tag = f'_{args.run_tag}' if args.run_tag else ''
    model_name = (f'cp_transformer_v0.42_size{args.model_size}_pop909ft{tag}_'
                  f'batch_{args.batch_size * n_gpus}_schedule')

    net = RoFormerSymbolicTransformer(size=args.model_size, max_lr=args.max_lr)

    if args.resume is None and not args.scratch:
        if not os.path.exists(args.pretrained):
            raise SystemExit(f'pretrained ckpt not found: {args.pretrained}')
        ck = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
        net.load_state_dict(state, strict=True)
        print(f'[init] loaded pretrained weights from {args.pretrained}')
    elif args.scratch:
        print('[init] --scratch: random init (debugging)')

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

    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'

    print(f'[finetune] {model_name}')
    print(f'[finetune] data={args.data}  max_lr={args.max_lr}  '
          f'lr_total_steps={args.lr_total_steps}  global_batch='
          f'{args.batch_size * n_gpus}')

    try:
        logger = TensorBoardLogger('tb_logs', name=model_name)
    except ModuleNotFoundError:
        print('[finetune] tensorboard unavailable; logging disabled')
        logger = False

    trainer = L.Trainer(
        devices=-1 if torch.cuda.is_available() else 1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=args.lr_total_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        val_check_interval=args.val_check_interval,
        limit_val_batches=10,
        check_val_every_n_epoch=None,
        gradient_clip_val=None,
        logger=logger,
        num_sanity_val_steps=0 if args.resume is not None else 2,
        strategy=strategy)
    trainer.fit(net, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
