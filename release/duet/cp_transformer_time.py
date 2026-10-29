"""The cp transformer on time-based frames (preprocess_time_grid.py).

Same architecture as cp_transformer.py (global RoFormer over frames, local
encoder and decoder over each frame's tokens); only the tokenizer
differs. A note is two tokens:

  A = program + 128 * (velocity_bin + 16 * onset_offset)      2048*K values
  B = 2048*K + 128 * duration_index + pitch                    3072 values

with 16 velocity bins (velocity // 8), the onset offset 0..K-1 in codec
frames inside the symbolic frame, and the duration index over
DURATION_TEMPLATES in 20 ms units. K = 1 is the existing with_velocity
vocabulary layout. sos, eos and pad follow.

  python cp_transformer_time.py train  --data data/<corpus> ...
  python cp_transformer_time.py sample --ckpt <ckpt> --data data/<corpus> --out <dir>

Validation logs val_loss (all tokens, as cp_transformer.py) and
val_note_loss (note tokens only): with time frames many targets are an
easy EOS, so val_note_loss is the number that tracks the music.
"""
import argparse
import datetime
import os

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cp_transformer
from cp_transformer import RoFormerSymbolicTransformer
from preprocess_time_grid import SLOT, EOS, PAD, TimeGridDataset, roll_to_notes, write_midi, DURATION_TEMPLATES


class TimeGridTokenizer:
    def __init__(self, frame_codec):
        self.frame_codec = frame_codec
        self.n_first = 128 * 16 * frame_codec
        self.n_normal_tokens = self.n_first + 128 * len(DURATION_TEMPLATES)
        self.n_tokens = self.n_normal_tokens + 3
        self.sos_token = self.n_normal_tokens
        self.eos_token = self.n_normal_tokens + 1
        self.pad_token = self.n_normal_tokens + 2
        self.with_velocity = True


class TimeGridSymbolicTransformer(RoFormerSymbolicTransformer):

    def __init__(self, size=1, frame_codec=6, max_polyphony=16, max_position_embeddings=1536, max_lr=1e-4,
                 total_steps=cp_transformer.MAX_STEPS):
        super().__init__(size=size, max_position_embeddings=max_position_embeddings, with_velocity=True, max_lr=max_lr)
        self.tokenizer = TimeGridTokenizer(frame_codec)
        self.local_embedding = nn.Embedding(self.tokenizer.n_tokens, self.hidden_size)
        self.final_decoder = nn.Linear(self.hidden_size, self.tokenizer.n_tokens)
        self.max_polyphony = max_polyphony
        self.total_steps = total_steps
        self.register_buffer('frame_codec_flag', torch.tensor(frame_codec))  # recorded in the checkpoint
        self.register_buffer('max_polyphony_flag', torch.tensor(max_polyphony))

    def preprocess(self, x, pitch_shift, tuple_size=SLOT):
        b, t, sub = x.shape
        x = x.long().view(b, t, sub // SLOT, SLOT)
        prog, pitch, dur, vel, off = x.unbind(-1)
        out = torch.zeros(b, t, sub // SLOT, 2, dtype=torch.long, device=x.device)
        out[..., 0] = prog + 128 * (vel // 8 + 16 * off)
        out[..., 1] = self.tokenizer.n_first + 128 * dur + pitch + pitch_shift[:, None, None] * (prog != 127)
        out[pitch == PAD] = self.tokenizer.pad_token
        out[..., 0][prog == EOS] = self.tokenizer.eos_token
        return out.view(b, t, sub // SLOT * 2)

    def tokens_to_roll(self, y):
        """Token frames [T, 2P] -> roll [T, P*SLOT] uint8 (for writing samples)."""
        y = y.view(len(y), -1, 2).cpu().numpy()
        roll = np.full((len(y), y.shape[1], SLOT), PAD, np.uint8)
        nf = self.tokenizer.n_first
        for s in range(len(y)):
            for j, (a, b) in enumerate(y[s]):
                if a >= self.tokenizer.n_normal_tokens or a >= nf or not (nf <= b < self.tokenizer.n_normal_tokens):
                    roll[s, j, 0] = EOS
                    break
                q = (b - nf)
                roll[s, j] = [a % 128, q % 128, q // 128, (a // 128 % 16) * 8 + 4, a // 2048]
        return roll.reshape(len(y), -1)

    def _loss(self, x, pitch_shift):
        x = self.preprocess(x, pitch_shift)
        y = self(x)
        tok = x.view(-1)
        ce = F.cross_entropy(y.view(-1, self.tokenizer.n_tokens), tok, ignore_index=self.tokenizer.pad_token,
                             reduction='none')
        valid = tok != self.tokenizer.pad_token
        note = valid & (tok != self.tokenizer.eos_token)
        return ce[valid].mean(), ce[note].mean() if note.any() else ce.new_zeros(())

    def loss(self, x, pitch_shift):
        return self._loss(x, pitch_shift)[0]

    def validation_step(self, batch, batch_idx):
        loss, note_loss = self._loss(*batch)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_note_loss', note_loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.total_steps,
                                                        pct_start=0.005)
        return [optimizer], [scheduler]


def train(args):
    n_gpus = max(torch.cuda.device_count(), 1)
    name = args.run_name or (f'cp_time_{os.path.basename(args.data)}_{args.frame_codec * 20}ms_cp{args.max_polyphony}'
                             f'_size{args.size}_batch_{args.batch_size * n_gpus}')
    net = TimeGridSymbolicTransformer(size=args.size, frame_codec=args.frame_codec, max_polyphony=args.max_polyphony,
                                      max_position_embeddings=args.max_positions, max_lr=args.max_lr,
                                      total_steps=args.max_steps)
    kw = dict(frame_codec=args.frame_codec, max_polyphony=args.max_polyphony)
    train_set = TimeGridDataset(args.data, args.train_length, args.batch_size, split='train', stretch=args.stretch, **kw)
    val_set = TimeGridDataset(args.data, args.train_length, args.batch_size, split='val', **kw)
    print(f'run {name}: {args.data}, {args.frame_codec * 20} ms frames, {args.train_length} frames per crop '
          f'({args.train_length * args.frame_codec * 0.02:.1f} s), cp{args.max_polyphony}')
    ckpt = L.callbacks.ModelCheckpoint(monitor='val_note_loss', save_top_k=5, save_last=True,
                                       enable_version_counter=False, dirpath=f'ckpt/{name}',
                                       filename=name + '.{step}.{val_note_loss:.5f}')
    resume = f'ckpt/{name}/last.ckpt'
    resume = resume if os.path.isfile(resume) else None
    if resume:
        print('resuming', resume)
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project='mitosis-time-grid', name=name)
    else:
        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
        logger = TensorBoardLogger('tb_logs', name=name)
    strategy = 'auto'
    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    trainer = L.Trainer(devices=-1 if torch.cuda.is_available() else 1,
                        precision='bf16-mixed' if torch.cuda.is_available() else 32,
                        max_steps=args.max_steps, accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        callbacks=[ckpt], val_check_interval=args.val_every, limit_val_batches=args.val_batches,
                        check_val_every_n_epoch=None, gradient_clip_val=1.0 if args.size >= 2 else None,
                        logger=logger, num_sanity_val_steps=0 if resume else 2, strategy=strategy)
    loader = lambda ds: DataLoader(ds, batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0)
    trainer.fit(net, loader(train_set), loader(val_set), ckpt_path=resume)
    torch.save(net.state_dict(), f'ckpt/{name}.pt')


def load(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    hidden = sd['global_sos'].shape[0]
    size = [512, 768, 1024, 1280].index(hidden)
    net = TimeGridSymbolicTransformer(size=size, frame_codec=int(sd['frame_codec_flag']),
                                      max_polyphony=int(sd['max_polyphony_flag']),
                                      max_position_embeddings=sd['model.embed_positions.weight'].shape[0]
                                      if 'model.embed_positions.weight' in sd else 1536)
    net.load_state_dict(sd)
    return net.to(device).eval()


@torch.no_grad()
def sample(args):
    """Continue held-out prompts and write prompt+continuation MIDI files."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load(args.ckpt, device)
    K, P = net.tokenizer.frame_codec, net.max_polyphony
    total = args.prompt_frames + args.gen_frames
    ds = TimeGridDataset(args.data, total, args.n, frame_codec=K, max_polyphony=P, split='test', random_order=True,
                         repeat=False)
    rolls, _ = next(iter(ds))
    x = net.preprocess(rolls[:, :args.prompt_frames].to(device), torch.zeros(len(rolls), dtype=torch.long, device=device))
    y = net.global_sampling(x, max_seq_len=total, temperature=args.temperature)
    y = torch.stack(y, dim=1)
    os.makedirs(args.out, exist_ok=True)
    for b in range(len(y)):
        write_midi(roll_to_notes(net.tokens_to_roll(y[b]), K, P), os.path.join(args.out, f'sample{b}.mid'))
        write_midi(roll_to_notes(rolls[b].numpy(), K, P), os.path.join(args.out, f'reference{b}.mid'))
    print(f'wrote {len(y)} samples ({args.prompt_frames * K * 0.02:.1f} s prompt + '
          f'{args.gen_frames * K * 0.02:.1f} s generated) and their references to {args.out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True, help='corpus prefix, e.g. data/time_piano')
    t.add_argument('--frame_codec', type=int, default=6, help='codec frames (20 ms) per symbolic frame')
    t.add_argument('--max_polyphony', type=int, default=16)
    t.add_argument('--size', type=int, default=1, choices=[0, 1, 2, 3])
    t.add_argument('--batch_size', type=int, default=8, help='per GPU')
    t.add_argument('--train_length', type=int, default=384, help='frames per crop (384 x 120 ms = 46 s)')
    t.add_argument('--max_positions', type=int, default=1536)
    t.add_argument('--max_lr', type=float, default=None, help='default 1e-4, 5e-5 for size >= 2')
    t.add_argument('--max_steps', type=int, default=500000)
    t.add_argument('--stretch', type=float, default=0.1, help='random time stretch +-, training crops only')
    t.add_argument('--val_every', type=int, default=2500)
    t.add_argument('--val_batches', type=int, default=25)
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--run_name', default=None)
    t.add_argument('--wandb', type=int, default=0)
    s = sub.add_parser('sample')
    s.add_argument('--ckpt', required=True)
    s.add_argument('--data', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--n', type=int, default=4)
    s.add_argument('--prompt_frames', type=int, default=64)
    s.add_argument('--gen_frames', type=int, default=192)
    s.add_argument('--temperature', type=float, default=1.0)
    args = ap.parse_args()
    if args.stage == 'train':
        if args.max_lr is None:
            args.max_lr = 5e-5 if args.size >= 2 else 1e-4
        assert args.train_length < args.max_positions, 'the global stack holds max_positions - 1 frames'
        train(args)
    else:
        sample(args)


if __name__ == '__main__':
    main()
