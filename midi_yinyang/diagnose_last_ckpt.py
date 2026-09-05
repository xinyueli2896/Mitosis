"""Does ModelCheckpoint(save_last=True) keep writing last.ckpt once the
monitored metric stops improving?

The A.8 run (job 206327) finished all 75000 steps with validation every
500, yet nothing in its checkpoint directory was written after step
9500 -- the last time val_loss improved. The best-k files are correct
by construction; last.ckpt was the safety net and it froze at the last
best-k save, so the converged weights never reached disk.

This is a 40-line reproduction with NO model from the repo, so it
isolates the callback from everything else: a toy LightningModule whose
monitored metric is a CONSTANT (so top-k saves exactly save_top_k
times, then never again), the same ModelCheckpoint arguments the
training script uses, step-based validation, epoch stuck at 0 (more
steps per epoch than max_steps, as with the repo's iterable dataset).

After fit it lists the directory and reads global_step back out of
last.ckpt. Expected: last.ckpt holds global_step == max_steps.
Reproduced: it holds the step of the last top-k save.

Run via diagnose_last_ckpt.sbatch; NGPUS=0/1/2 selects cpu, one gpu,
or 2-GPU DDP -- the A.8 run was DDP, so both single and DDP matter.
"""
import os
import sys
import time

import torch
import pytorch_lightning as L

OUT = sys.argv[1]
NGPUS = int(sys.argv[2])
MAX_STEPS, VAL_EVERY = 600, 50


class Toy(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(4, 1)

    def training_step(self, b, i):
        return self.w(b).pow(2).mean()

    def validation_step(self, b, i):
        self.log('val_const', 1.0)                   # never improves
        self.log('val_seen_step', float(self.global_step))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), 0.01)

    def train_dataloader(self):                      # 1250 steps/epoch
        return torch.utils.data.DataLoader(torch.randn(10000, 4), batch_size=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(torch.randn(64, 4), batch_size=8)


if __name__ == '__main__':
    print(f'pytorch_lightning {L.__version__}  torch {torch.__version__}')
    cb = L.callbacks.ModelCheckpoint(
        monitor='val_const', save_top_k=2, save_last=True,
        enable_version_counter=False, dirpath=OUT,
        filename='toy.{epoch:02d}.{step}.{val_const:.5f}',
    )
    trainer = L.Trainer(
        max_steps=MAX_STEPS, val_check_interval=VAL_EVERY,
        check_val_every_n_epoch=None, limit_val_batches=4,
        accelerator='gpu' if NGPUS > 0 else 'cpu',
        devices=NGPUS if NGPUS > 0 else 1,
        callbacks=[cb], logger=False, enable_progress_bar=False,
    )
    trainer.fit(Toy())
    if trainer.is_global_zero:
        print(f'\n--- {OUT} after fit (max_steps={MAX_STEPS}, '
              f'val every {VAL_EVERY}) ---')
        for f in sorted(os.listdir(OUT)):
            st = os.stat(os.path.join(OUT, f))
            print(f'  {f:<40} {st.st_size:>10}  '
                  f'{time.strftime("%H:%M:%S", time.localtime(st.st_mtime))}')
        last = os.path.join(OUT, 'last.ckpt')
        ck = torch.load(last, map_location='cpu', weights_only=False)
        gs = ck.get('global_step')
        print(f'\n  last.ckpt global_step = {gs}   (expected {MAX_STEPS})')
        print('  is symlink:', os.path.islink(last))
        print('  -> ' + ('REPRODUCED: last.ckpt froze at the last top-k save'
                         if gs != MAX_STEPS else 'not reproduced here'))
