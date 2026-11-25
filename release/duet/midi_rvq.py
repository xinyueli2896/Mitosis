"""A residual-VQ autoencoder for piano MIDI at EnCodec's frame rate.

MusicGen's codec, EnCodec 32 kHz, turns audio into 4 codes per 20 ms
frame (residual vector quantization, 4 codebooks of 2048). This model
does the same for MIDI: a piano roll at 50 Hz goes in, N_Q codes per
frame come out, and the decoder turns codes back into notes. MIDI frame
f and codec frame f then cover the same 20 ms, with the same shape, so
a transformer can model both streams side by side, e.g. as extra
codebooks next to MusicGen's.

Input per frame and per piano key (88 keys, A0..C8; drums and other
programs are folded onto the keys or dropped, see --keep_drums):
  onset     1 where a note starts
  frame     1 while a note sounds (pedal already folded into durations)
  velocity  the onset's velocity / 127, 0 elsewhere
Output per frame and key: onset and frame logits, and 16 velocity bins
(velocity // 8, as the cp tokenizer bins it). Notes are read back from
the output: an onset starts a note, which lasts while the frame output
stays on, until the next onset of that key.

Encoder and decoder are 1-D over time with the 88x3 key features as
channels: a dilated residual conv stack then a small bidirectional
transformer, no downsampling (one code vector per 20 ms frame, like
EnCodec). The quantizer is EMA residual VQ with k-means init and dead
code restarts, synchronized across DDP ranks.

What decides whether the codes are usable is reconstruction, so
validation decodes notes and reports note F1 (onset within 2 frames =
40 ms, near mir_eval's 50 ms), onset+offset F1, onset+velocity F1, and
codebook usage. The target is note F1 well above 0.99: a wrong code in
audio is a little noise, in MIDI it is a wrong note.

Data is a note corpus from preprocess_time_grid.py (milliseconds,
memory-mapped); MAESTRO's packed splits work the same way.

  python midi_rvq.py train  --data data/<corpus> [--val_data data/maestro_validation]
  python midi_rvq.py eval   --ckpt <ckpt> --data data/maestro_test
  python midi_rvq.py export --ckpt <ckpt> --data data/maestro_train   # -> <data>.midi_codes.npy
"""
import argparse
import datetime
import json
import math
import os

import numpy as np
import pytorch_lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from preprocess_time_grid import CODEC_MS, Corpus

LOW, N_KEYS = 21, 88  # A0 .. C8
N_VEL = 16


# ---------------------------------------------------------------- rolls

def notes_to_roll(notes, start_ms, n_frames, stretch=1.0, shift=0, keep_drums=False):
    """onset, frame, velocity rolls [3, 88, n_frames] float32 for a window.

    Notes that started before the window but still sound in it are on in
    the frame roll (no onset). Times are multiplied by stretch first
    (start_ms is in stretched time); pitches are moved by shift and notes
    pushed off the keyboard are dropped.
    """
    roll = np.zeros((3, N_KEYS, n_frames), np.float32)
    if len(notes) == 0:
        return roll
    if not keep_drums:
        notes = notes[notes['program'] != 127]
    on = notes['onset_ms'].astype(np.float64) * stretch - start_ms
    end = on + notes['dur_ms'] * stretch
    c = np.rint(on / CODEC_MS).astype(np.int64)
    e = np.maximum(np.rint(end / CODEC_MS).astype(np.int64), c + 1)
    k = notes['pitch'].astype(np.int64) + shift - LOW
    sel = (e > 0) & (c < n_frames) & (k >= 0) & (k < N_KEYS)
    c, e, k, vel = c[sel], e[sel], k[sel], notes['velocity'][sel]
    diff = np.zeros((N_KEYS, n_frames + 1), np.int32)
    np.add.at(diff, (k, np.clip(c, 0, n_frames)), 1)
    np.add.at(diff, (k, np.clip(e, 0, n_frames)), -1)
    roll[1] = np.cumsum(diff, axis=1)[:, :n_frames] > 0
    o = c >= 0
    roll[0, k[o], c[o]] = 1
    roll[2, k[o], c[o]] = vel[o] / 127.0
    return roll


def outputs_to_notes(onset_p, frame_p, vel_bin, threshold=0.5, min_gap=0):
    """Notes [(onset frame, end frame, key, velocity)] from per-key outputs [88, T].

    An onset is a run of frames above threshold; with min_gap > 0, an onset
    within min_gap frames after the previous one on the same key is
    dropped (a smeared onset read twice, not a repeated note).
    """
    on = onset_p > threshold
    on[:, 1:] &= ~on[:, :-1]  # a run of onset frames is one onset
    if min_gap > 0:
        for k in np.flatnonzero(on.any(1)):
            last = -10 ** 9
            for t in np.flatnonzero(on[k]):
                if t - last <= min_gap:
                    on[k, t] = False
                else:
                    last = t
    act = frame_p > threshold
    T = on.shape[1]
    notes = []
    for k, t in zip(*np.nonzero(on)):
        nxt = np.flatnonzero(on[k, t + 1:])
        stop = t + 1 + (nxt[0] if len(nxt) else T - t - 1)
        off = np.flatnonzero(~act[k, t + 1:stop])
        e = t + 1 + (off[0] if len(off) else stop - t - 1)
        notes.append((int(t), int(max(e, t + 1)), int(k), int(vel_bin[k, t]) * 8 + 4))
    return notes


def roll_to_note_list(roll):
    """Reference notes from an input roll, read exactly as outputs_to_notes reads outputs."""
    return outputs_to_notes(roll[0], roll[1], np.floor(roll[2] * 127 / 8).astype(np.int64))


def note_f1(ref, est, onset_tol=2, with_offset=False, with_velocity=False):
    """Greedy one-to-one matching per key; returns precision, recall, F1."""
    if not ref and not est:
        return 1.0, 1.0, 1.0
    by_key = {}
    for n in est:
        by_key.setdefault(n[2], []).append(n)
    used = set()
    tp = 0
    for r in ref:
        best = None
        for j, n in enumerate(by_key.get(r[2], [])):
            if (r[2], j) in used or abs(n[0] - r[0]) > onset_tol:
                continue
            if with_offset and abs(n[1] - r[1]) > max(onset_tol, 0.2 * (r[1] - r[0])):
                continue
            if with_velocity and abs(n[3] // 8 - r[3] // 8) > 1:
                continue
            if best is None or abs(n[0] - r[0]) < abs(by_key[r[2]][best][0] - r[0]):
                best = j
        if best is not None:
            used.add((r[2], best))
            tp += 1
    p = tp / max(1, len(est))
    r_ = tp / max(1, len(ref))
    return p, r_, 2 * p * r_ / max(1e-9, p + r_)


class RollDataset(IterableDataset):
    """Random windows of a note corpus as [B, 3, 88, T] rolls (train: stretch and pitch shift)."""

    def __init__(self, prefix, n_frames, batch_size, split='train', stretch=0.1, max_shift=5, keep_drums=False,
                 split_ratio=10, seed=None):
        self.prefix, self.T, self.B, self.split = prefix, n_frames, batch_size, split
        self.stretch = stretch if split == 'train' else 0.0
        self.max_shift = max_shift if split == 'train' else 0
        self.keep_drums, self.seed = keep_drums, seed
        z = np.load(prefix + '.pieces.npz')
        idx = np.arange(len(z['count']))
        ok = z['length_ms'] * (1 - self.stretch) >= n_frames * CODEC_MS
        sel = {'all': np.ones_like(ok), 'train': idx % split_ratio > 1,
               'val': idx % split_ratio == 1, 'test': idx % split_ratio == 0}[split]
        self.valid = idx[ok & sel]
        if len(self.valid) == 0:
            raise ValueError(f'{prefix} [{split}]: no piece is {n_frames * CODEC_MS / 1000:.1f} s long; lower --frames')
        self.corpus = None
        print(f'{prefix} [{split}]: {len(self.valid)} pieces of >= {n_frames * CODEC_MS / 1000:.1f} s')

    def window(self, sid, rng):
        """One random (stretched, shifted) window of piece sid as a [3, 88, T] roll."""
        if self.corpus is None:
            self.corpus = Corpus(self.prefix)
        a = rng.uniform(1 - self.stretch, 1 + self.stretch) if self.stretch else 1.0
        span = max(0.0, self.corpus.length_ms[sid] * a - self.T * CODEC_MS)
        s0 = round(rng.uniform(0, span) / CODEC_MS) * CODEC_MS
        sh = int(rng.integers(-self.max_shift, self.max_shift + 1)) if self.max_shift else 0
        return notes_to_roll(self.corpus.piece(sid), s0, self.T, a, sh, self.keep_drums)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            order = rng.permutation(self.valid)
            order = np.resize(order, -(-len(order) // self.B) * self.B)  # a short split still fills batches
            for i in range(0, len(order), self.B):
                yield torch.from_numpy(np.stack([self.window(sid, rng) for sid in order[i:i + self.B]]))


class MixedRollDataset(IterableDataset):
    """Batches from a main corpus with a fixed share of windows from a second one.

    Pieces are drawn uniformly within each corpus, so a small corpus mixed
    in by concatenation would almost never be seen (MAESTRO train is ~960
    of ~372k pieces next to Aria). Here each window comes from the mixed
    corpus with probability `frac`, whatever the corpus sizes.
    """

    def __init__(self, main, mix, frac, seed=None):
        self.main, self.mix, self.frac, self.seed = main, mix, frac, seed
        self.B = main.B

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            rolls = [self.mix.window(rng.choice(self.mix.valid), rng) if rng.random() < self.frac
                     else self.main.window(rng.choice(self.main.valid), rng) for _ in range(self.B)]
            yield torch.from_numpy(np.stack(rolls))


# ---------------------------------------------------------------- model

class ResBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(1, d), nn.GELU(), nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation),
                                 nn.GroupNorm(1, d), nn.GELU(), nn.Conv1d(d, d, 1))

    def forward(self, x):
        return x + self.net(x)


class Stack(nn.Module):
    """[B, C_in, T] -> [B, C_out, T]: conv in, dilated residual blocks, bidirectional transformer, conv out."""

    def __init__(self, c_in, c_out, d, n_res, n_attn, heads=8):
        super().__init__()
        self.inp = nn.Conv1d(c_in, d, 3, padding=1)
        self.res = nn.Sequential(*[ResBlock(d, 2 ** (i % 4)) for i in range(n_res)])
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, n_attn) if n_attn else None
        self.out = nn.Sequential(nn.GroupNorm(1, d), nn.GELU(), nn.Conv1d(d, c_out, 1))

    def forward(self, x):
        h = self.res(self.inp(x))
        if self.attn is not None:
            T = h.shape[-1]
            pos = torch.arange(T, device=h.device)[:, None] / 10000 ** (torch.arange(0, h.shape[1], 2, device=h.device) / h.shape[1])
            pe = torch.zeros(T, h.shape[1], device=h.device, dtype=h.dtype)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos), torch.cos(pos)
            h = self.attn(h.transpose(1, 2) + pe).transpose(1, 2)
        return self.out(h)


class EMACodebook(nn.Module):
    def __init__(self, n_codes, dim, decay=0.99, eps=1e-5, dead_threshold=1.0):
        super().__init__()
        self.decay, self.eps, self.dead = decay, eps, dead_threshold
        self.register_buffer('embed', torch.randn(n_codes, dim) * 0.01)
        self.register_buffer('cluster_size', torch.zeros(n_codes))
        self.register_buffer('embed_sum', torch.zeros(n_codes, dim))
        self.register_buffer('inited', torch.tensor(False))

    @staticmethod
    def _all_reduce(t):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t)
        return t

    def _kmeans_init(self, x, iters=10):
        n = self.embed.shape[0]
        idx = torch.randint(0, len(x), (n,), device=x.device)
        c = x[idx].clone()
        for _ in range(iters):
            a = torch.cdist(x, c).argmin(1)
            s = torch.zeros_like(c).index_add_(0, a, x)
            cnt = torch.bincount(a, minlength=n).float()
            c = torch.where(cnt[:, None] > 0, s / cnt.clamp(min=1)[:, None], c)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(c, 0)
        self.embed.copy_(c)
        self.embed_sum.copy_(c)
        self.cluster_size.fill_(1.0)
        self.inited.fill_(True)

    def codes(self, x):
        return torch.cdist(x, self.embed).argmin(-1)

    def forward(self, x):
        """x [N, D] -> quantized [N, D] (straight-through), codes [N]."""
        x = x.float()
        if self.training and not bool(self.inited):
            self._kmeans_init(x.detach())
        idx = self.codes(x)
        q = self.embed[idx]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.embed.shape[0]).float()
                size = self._all_reduce(onehot.sum(0))
                total = self._all_reduce(onehot.t() @ x)
                self.cluster_size.mul_(self.decay).add_(size, alpha=1 - self.decay)
                self.embed_sum.mul_(self.decay).add_(total, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                size = (self.cluster_size + self.eps) / (n + self.embed.shape[0] * self.eps) * n
                self.embed.copy_(self.embed_sum / size[:, None])
                dead = self.cluster_size < self.dead
                if dead.any():  # restart unused codes on random current inputs
                    r = x[torch.randint(0, len(x), (int(dead.sum()),), device=x.device)]
                    if dist.is_available() and dist.is_initialized():
                        dist.broadcast(r, 0)
                    self.embed[dead] = r
                    self.embed_sum[dead] = r
                    self.cluster_size[dead] = 1.0
        return x + (q - x).detach(), idx


class ResidualVQ(nn.Module):
    def __init__(self, n_q, n_codes, dim):
        super().__init__()
        self.layers = nn.ModuleList([EMACodebook(n_codes, dim) for _ in range(n_q)])

    def forward(self, z):
        """z [B, D, T] -> quantized [B, D, T], codes [B, T, n_q], commitment loss."""
        x = z.transpose(1, 2).reshape(-1, z.shape[1]).float()
        res, out, codes = x, torch.zeros_like(x), []
        for layer in self.layers:
            q, idx = layer(res)
            out = out + q
            res = res - q.detach()
            codes.append(idx)
        commit = F.mse_loss(x, out.detach())
        B, D, T = z.shape
        return out.view(B, T, D).transpose(1, 2).to(z.dtype), torch.stack(codes, -1).view(B, T, -1), commit

    def decode_codes(self, codes):
        """codes [B, T, n_q] -> [B, D, T]."""
        q = sum(layer.embed[codes[..., i]] for i, layer in enumerate(self.layers))
        return q.transpose(1, 2)


class MidiRVQ(L.LightningModule):
    def __init__(self, n_q=4, n_codes=1024, dim=128, d=512, n_res=8, n_attn=4, lr=3e-4, total_steps=200000,
                 onset_pos_weight=10.0, commit_weight=0.25):
        super().__init__()
        self.save_hyperparameters()
        self.enc = Stack(3 * N_KEYS, dim, d, n_res, n_attn)
        self.rvq = ResidualVQ(n_q, n_codes, dim)
        self.dec = Stack(dim, N_KEYS * (2 + N_VEL), d, n_res, n_attn)

    def encode(self, roll):
        """roll [B, 3, 88, T] -> codes [B, T, n_q]."""
        return self.rvq(self.enc(roll.flatten(1, 2)))[1]

    def _heads(self, h):
        B, _, T = h.shape
        h = h.view(B, N_KEYS, 2 + N_VEL, T)
        return h[:, :, 0], h[:, :, 1], h[:, :, 2:]  # onset, frame [B, 88, T]; velocity [B, 88, 16, T]

    def decode(self, codes):
        """codes [B, T, n_q] -> onset prob, frame prob [B, 88, T], velocity bin [B, 88, T]."""
        on, fr, vel = self._heads(self.dec(self.rvq.decode_codes(codes)))
        return torch.sigmoid(on), torch.sigmoid(fr), vel.argmax(2)

    def forward(self, roll):
        z = self.enc(roll.flatten(1, 2))
        q, codes, commit = self.rvq(z)
        return self._heads(self.dec(q)), codes, commit

    def losses(self, roll):
        (on, fr, vel), codes, commit = self(roll)
        on_t, fr_t = roll[:, 0], roll[:, 1]
        l_on = F.binary_cross_entropy_with_logits(on.float(), on_t, pos_weight=torch.tensor(self.hparams.onset_pos_weight,
                                                                                           device=on.device))
        l_fr = F.binary_cross_entropy_with_logits(fr.float(), fr_t)
        m = on_t > 0
        vel_t = (roll[:, 2] * 127 / 8).floor().long().clamp(0, N_VEL - 1)
        l_vel = F.cross_entropy(vel.float().permute(0, 1, 3, 2)[m], vel_t[m]) if m.any() else on.new_zeros(())
        loss = l_on + l_fr + 0.5 * l_vel + self.hparams.commit_weight * commit
        return loss, dict(onset=l_on, frame=l_fr, velocity=l_vel, commit=commit), (on, fr, vel), codes

    def training_step(self, roll, i):
        loss, parts, _, _ = self.losses(roll)
        self.log('train_loss', loss, prog_bar=True)
        for k, v in parts.items():
            self.log(f'train/{k}', v)
        return loss

    def validation_step(self, roll, i):
        loss, parts, (on, fr, vel), codes = self.losses(roll)
        self.log('val_loss', loss, sync_dist=True)
        on, fr, vel = torch.sigmoid(on).float().cpu().numpy(), torch.sigmoid(fr).float().cpu().numpy(), vel.argmax(2).cpu().numpy()
        f = np.array([[note_f1(roll_to_note_list(r), est, **kw)[2] for kw in
                       ({}, {'with_offset': True}, {'with_velocity': True})]
                      for r, est in ((roll[b].float().cpu().numpy(), outputs_to_notes(on[b], fr[b], vel[b]))
                                     for b in range(len(roll)))])
        self.log('val_note_f1', float(f[:, 0].mean()), sync_dist=True)
        self.log('val_note_offset_f1', float(f[:, 1].mean()), sync_dist=True)
        self.log('val_note_velocity_f1', float(f[:, 2].mean()), sync_dist=True)
        for q in range(codes.shape[-1]):
            self.log(f'val_codes_used_q{q}', float(codes[..., q].unique().numel()), sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr, total_steps=self.hparams.total_steps,
                                                    pct_start=0.02)
        return [opt], [{'scheduler': sched, 'interval': 'step'}]


# ---------------------------------------------------------------- whole pieces

@torch.no_grad()
def encode_piece(model, notes, n_frames, device, window=1000, context=100):
    """Codes [n_frames, n_q] for a whole piece, in windows with `context` frames of real input on each side."""
    out = np.zeros((n_frames, model.hparams.n_q), np.int16)
    for s in range(0, n_frames, window):
        a, b = max(0, s - context), min(n_frames, s + window + context)
        roll = torch.from_numpy(notes_to_roll(notes, a * CODEC_MS, b - a))[None].to(device)
        codes = model.encode(roll)[0].cpu().numpy()
        e = min(s + window, n_frames)
        out[s:e] = codes[s - a:e - a]
    return out


@torch.no_grad()
def decode_piece(model, codes, device, window=1000, context=100, threshold=0.5, min_gap=0):
    """Notes [(onset s, end s, pitch, velocity)] from codes [T, n_q], decoded in overlapping windows."""
    on_p, fr_p, vel = decode_probs(model, codes, device, window, context)
    return [(t0 * CODEC_MS / 1000, t1 * CODEC_MS / 1000, k + LOW, v)
            for t0, t1, k, v in outputs_to_notes(on_p, fr_p, vel, threshold, min_gap)]


@torch.no_grad()
def decode_probs(model, codes, device, window=1000, context=100):
    """Onset and frame probabilities and velocity bins [88, T] from codes [T, n_q]."""
    T = len(codes)
    on_p, fr_p, vel = (np.zeros((N_KEYS, T), np.float32), np.zeros((N_KEYS, T), np.float32), np.zeros((N_KEYS, T), np.int64))
    for s in range(0, T, window):
        a, b = max(0, s - context), min(T, s + window + context)
        o, f, v = model.decode(torch.from_numpy(np.asarray(codes[a:b], np.int64))[None].to(device))
        e = min(s + window, T)
        on_p[:, s:e], fr_p[:, s:e], vel[:, s:e] = (t[0, :, s - a:e - a].float().cpu().numpy() for t in (o, f, v))
    return on_p, fr_p, vel


def load(ckpt, device):
    model = MidiRVQ.load_from_checkpoint(ckpt, map_location='cpu', weights_only=False)
    return model.to(device).eval()


# ---------------------------------------------------------------- stages

def train(args):
    n_gpus = max(torch.cuda.device_count(), 1)
    mix = (f'+{os.path.basename(args.mix_data)}{args.mix_frac:g}' if args.mix_data and args.mix_frac > 0 else '')
    name = args.run_name or (f'midi_rvq_{os.path.basename(args.data)}{mix}_q{args.n_q}x{args.n_codes}_d{args.d}'
                             f'_batch{args.batch_size * n_gpus}')
    model = MidiRVQ(n_q=args.n_q, n_codes=args.n_codes, dim=args.dim, d=args.d, n_res=args.n_res, n_attn=args.n_attn,
                    lr=args.lr, total_steps=args.max_steps)
    # with a separate validation corpus the main corpus needs no held-out pieces
    tr = RollDataset(args.data, args.frames, args.batch_size, split='all' if args.val_data else 'train',
                     stretch=args.stretch)
    if args.mix_data and args.mix_frac > 0:
        # the mixed corpus is training data in full (split 'all'); validate on a separate corpus
        tr = MixedRollDataset(tr, RollDataset(args.mix_data, args.frames, args.batch_size, split='all',
                                              stretch=args.stretch), args.mix_frac)
        print(f'training windows: {1 - args.mix_frac:.0%} {args.data}, {args.mix_frac:.0%} {args.mix_data}')
    va = (RollDataset(args.val_data, args.frames, args.batch_size, split='all', seed=0) if args.val_data
          else RollDataset(args.data, args.frames, args.batch_size, split='val', seed=0))
    ckpt = L.callbacks.ModelCheckpoint(monitor='val_note_f1', mode='max', save_top_k=3, save_last=True,
                                       enable_version_counter=False, dirpath=f'ckpt/{name}',
                                       filename=name + '.{step}.{val_note_f1:.4f}')
    resume = f'ckpt/{name}/last.ckpt'
    resume = resume if os.path.isfile(resume) else None
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project='mitosis-midi-rvq', name=name)
    else:
        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
        logger = TensorBoardLogger('tb_logs', name=name)
    strategy = 'auto'
    if n_gpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(timeout=datetime.timedelta(hours=2))
    gpu = torch.cuda.is_available()
    trainer = L.Trainer(devices=-1 if gpu else 1, accelerator='gpu' if gpu else 'cpu',
                        precision='bf16-mixed' if gpu else 32, max_steps=args.max_steps, callbacks=[ckpt],
                        val_check_interval=args.val_every, limit_val_batches=args.val_batches,
                        check_val_every_n_epoch=None, gradient_clip_val=1.0, logger=logger,
                        num_sanity_val_steps=0 if resume else 1, strategy=strategy)
    print(f'run {name}: {args.data}, {args.frames} frames ({args.frames * CODEC_MS / 1000:.0f} s) per window, '
          f'{args.n_q} x {args.n_codes} codes per 20 ms', flush=True)
    dl = lambda ds: DataLoader(ds, batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0)
    trainer.fit(model, dl(tr), dl(va), ckpt_path=resume)


def listen(args):
    """Original and reconstructed MIDI side by side for a few pieces: <out>/<n>_<piece>_{orig,recon}.mid."""
    from preprocess_time_grid import write_midi
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load(args.ckpt, device)
    c = Corpus(args.data)
    os.makedirs(args.out, exist_ok=True)
    ids = np.random.default_rng(args.seed).permutation(len(c))[:args.n]
    for n, i in enumerate(ids):
        notes = c.piece(i)
        n_frames = int(min(c.length_ms[i], args.seconds * 1000) // CODEC_MS) + 1
        notes = notes[notes['onset_ms'] < n_frames * CODEC_MS]
        codes = encode_piece(model, notes, n_frames, device)
        est = decode_piece(model, codes, device, threshold=args.threshold, min_gap=args.min_gap)
        ref = roll_to_note_list(notes_to_roll(notes, 0, n_frames))
        f1 = note_f1(ref, [(round(t0 * 50), round(t1 * 50), p - LOW, v) for t0, t1, p, v in est])[2]
        orig = [(int(a['onset_ms']) / 1000, (int(a['onset_ms']) + int(a['dur_ms'])) / 1000, int(a['pitch']),
                 int(a['velocity']), 0) for a in notes if a['program'] != 127]
        stem = os.path.join(args.out, f'{n:02d}_piece{i}')
        write_midi(orig, stem + '_orig.mid')
        write_midi([(t0, t1, p, v, 0) for t0, t1, p, v in est], stem + '_recon.mid')
        print(f'{stem}_{{orig,recon}}.mid: {len(ref)} notes, note F1 {f1:.4f}', flush=True)


def evaluate(args):
    """Whole-piece reconstruction: encode, decode, note F1 against the input (read at 20 ms).

    Each piece is decoded once and read out at every (threshold, min_gap)
    pair; the JSON reports the pair with the best note F1 at top level and
    every pair under "sweep".
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load(args.ckpt, device)
    c = Corpus(args.data)
    ids = np.random.default_rng(0).permutation(len(c))[:args.max_pieces]
    combos = [(th, g) for th in args.thresholds for g in args.min_gaps]
    rows = []  # [pieces, combos, 3 metrics, P/R/F]
    for i in ids:
        n_frames = int(c.length_ms[i] // CODEC_MS) + 1
        codes = encode_piece(model, c.piece(i), n_frames, device)
        on_p, fr_p, vel = decode_probs(model, codes, device)
        ref = roll_to_note_list(notes_to_roll(c.piece(i), 0, n_frames))
        row = []
        for th, g in combos:
            est = outputs_to_notes(on_p, fr_p, vel, th, g)
            row.append([note_f1(ref, est, **kw) for kw in ({}, {'with_offset': True}, {'with_velocity': True})])
        rows.append(row)
        print(f'piece {i}: {len(ref)} notes, note F1 ' + ' '.join(f'{x[0][2]:.4f}' for x in row), flush=True)
    r = np.array(rows).mean(0)  # [combos, 3, 3]
    names = ('note', 'note_offset', 'note_velocity')
    sweep = [{'threshold': th, 'min_gap': g, **{m: dict(zip(('precision', 'recall', 'f1'), np.round(r[k, j], 4).tolist()))
                                                for j, m in enumerate(names)}} for k, (th, g) in enumerate(combos)]
    best = max(range(len(combos)), key=lambda k: r[k, 0, 2])
    res = {'pieces': len(ids), 'ckpt': args.ckpt, 'data': args.data, **sweep[best], 'sweep': sweep}
    print(json.dumps({k: v for k, v in res.items() if k != 'sweep'}, indent=1))
    for x in sweep:
        print(f"threshold {x['threshold']:.2f} min_gap {x['min_gap']}: note P/R/F1 {x['note']['precision']:.4f} "
              f"{x['note']['recall']:.4f} {x['note']['f1']:.4f}, +offset F1 {x['note_offset']['f1']:.4f}")
    out = args.ckpt.rsplit('.', 1)[0] + f'.eval_{os.path.basename(args.data)}.json'
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', out)


def export(args):
    """Codes for every piece of a corpus: <data>.midi_codes.npy [sum F, n_q] int16 and <data>.midi_code_start.npy.

    For a MAESTRO split packed by maestro_paired.py the rows follow its
    .codes.npy exactly (same start and length per piece), so row i of the
    MIDI codes and row i of the EnCodec codes are the same 20 ms.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load(args.ckpt, device)
    c = Corpus(args.data)
    paired = os.path.isfile(args.data + '.code_start.npy')
    if paired:
        start = np.load(args.data + '.code_start.npy')
        total = len(np.load(args.data + '.codes.npy', mmap_mode='r'))
        length = np.diff(np.r_[start, total])
    else:
        length = c.length_ms // CODEC_MS + 1
        start = np.cumsum(np.r_[0, length[:-1]])
    out = np.lib.format.open_memmap(args.data + '.midi_codes.npy', mode='w+', dtype=np.int16,
                                    shape=(int(length.sum()), model.hparams.n_q))
    for i in range(len(c)):
        out[start[i]:start[i] + length[i]] = encode_piece(model, c.piece(i), int(length[i]), device)
        if i % 50 == 0:
            print(f'{i}/{len(c)}', flush=True)
    out.flush()
    np.save(args.data + '.midi_code_start.npy', start.astype(np.int64))
    print(f'wrote {args.data}.midi_codes.npy {out.shape}' + (' (rows aligned with .codes.npy)' if paired else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True, help='note corpus prefix (preprocess_time_grid.py), e.g. data/time_piano')
    t.add_argument('--mix_data', default=None, help='a second training corpus mixed in at --mix_frac, e.g. data/maestro_train_midi')
    t.add_argument('--mix_frac', type=float, default=0.15, help='share of training windows drawn from --mix_data')
    t.add_argument('--val_data', default=None, help='a separate corpus to validate on, e.g. data/maestro_validation')
    t.add_argument('--n_q', type=int, default=4, help='codebooks (EnCodec 32 kHz has 4)')
    t.add_argument('--n_codes', type=int, default=1024)
    t.add_argument('--dim', type=int, default=128, help='code vector size')
    t.add_argument('--d', type=int, default=512, help='encoder/decoder width')
    t.add_argument('--n_res', type=int, default=8)
    t.add_argument('--n_attn', type=int, default=4)
    t.add_argument('--frames', type=int, default=500, help='frames per window (500 = 10 s)')
    t.add_argument('--batch_size', type=int, default=16, help='per GPU')
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--max_steps', type=int, default=200000)
    t.add_argument('--stretch', type=float, default=0.1)
    t.add_argument('--val_every', type=int, default=2000)
    t.add_argument('--val_batches', type=int, default=20)
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--run_name', default=None)
    t.add_argument('--wandb', type=int, default=0)
    e = sub.add_parser('eval')
    e.add_argument('--ckpt', required=True)
    e.add_argument('--data', required=True)
    e.add_argument('--max_pieces', type=int, default=50)
    e.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8, 0.9])
    e.add_argument('--min_gaps', type=int, nargs='+', default=[0, 2], help='frames; see outputs_to_notes')
    li = sub.add_parser('listen')
    li.add_argument('--ckpt', required=True)
    li.add_argument('--data', required=True)
    li.add_argument('--out', required=True)
    li.add_argument('--n', type=int, default=8)
    li.add_argument('--seconds', type=float, default=60.0, help='first N seconds of each piece')
    li.add_argument('--seed', type=int, default=0)
    li.add_argument('--threshold', type=float, default=0.5)
    li.add_argument('--min_gap', type=int, default=0)
    x = sub.add_parser('export')
    x.add_argument('--ckpt', required=True)
    x.add_argument('--data', required=True)
    args = ap.parse_args()
    {'train': train, 'eval': evaluate, 'listen': listen, 'export': export}[args.stage](args)


if __name__ == '__main__':
    main()
