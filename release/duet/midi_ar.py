"""An autoregressive transformer over MIDI RVQ codes, MusicGen-style.

The MIDI codec (midi_rvq.py) turns piano MIDI into K codes per 20 ms
frame, the frame rate and shape of MusicGen's EnCodec 32 kHz codes. This
model is MusicGen's language model on top of it: a decoder-only
transformer that predicts the K codebooks of each frame with the delay
pattern (codebook k is shifted right by k steps, so codebook k of frame f
is predicted after codebooks 0..k-1 of that frame are known), summed
codebook embeddings at the input, and one output head per codebook. A
frame here is the same 20 ms as a MusicGen frame, so a step of this model
lines up with a step of MusicGen.

Training codes are made on the fly by the frozen codec: each window is
encoded with CONTEXT extra frames of real input on each side (the codec
is bidirectional) and the middle FRAMES are kept, so no corpus-wide code
export is needed and stretch / pitch shift augmentation still applies.

  python midi_ar.py train  --codec <codec.ckpt> --data data/<corpus> [--mix_data ...] [--val_data ...]
  python midi_ar.py sample --ckpt <ar.ckpt> --data data/maestro_val_midi --out temp/ar_samples
"""
import argparse
import datetime
import math
import os

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import midi_rvq
from midi_rvq import MixedRollDataset, RollDataset, notes_to_roll
from preprocess_time_grid import CODEC_MS, Corpus


# ---------------------------------------------------------------- delay pattern

def delay(codes, special):
    """codes [B, T, K] -> [B, T + K - 1, K]; codebook k shifted right by k, the rest `special`."""
    B, T, K = codes.shape
    out = codes.new_full((B, T + K - 1, K), special)
    for k in range(K):
        out[:, k:k + T, k] = codes[..., k]
    return out


def undelay(seq, K):
    """[B, S, K] -> [B, S - K + 1, K], the inverse of delay()."""
    T = seq.shape[1] - K + 1
    return torch.stack([seq[:, k:k + T, k] for k in range(K)], -1)


# ---------------------------------------------------------------- transformer

def rope(x, pos):
    """Rotary position embedding on x [B, H, S, Dh] at positions pos [S]."""
    d = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(d, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pos, cache=None):
        """x [B, S, d]; cache: None (training, causal) or a dict holding past k/v (one new step at a time)."""
        B, S, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        if cache is not None:
            if 'k' in cache:
                k, v = torch.cat([cache['k'], k], 2), torch.cat([cache['v'], v], 2)
            cache['k'], cache['v'] = k, v
        a = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None,
                                           dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, S, D)))
        return x + self.drop(self.ff(self.n2(x)))


class MidiAR(L.LightningModule):
    def __init__(self, codec_ckpt, d=1024, layers=24, heads=16, dropout=0.1, lr=3e-4, total_steps=500000,
                 warmup=4000, context=50):
        super().__init__()
        self.save_hyperparameters()
        self.codec = midi_rvq.MidiRVQ.load_from_checkpoint(codec_ckpt, map_location='cpu', weights_only=False)
        self.codec.requires_grad_(False)
        self.K, self.V = self.codec.hparams.n_q, self.codec.hparams.n_codes
        self.special = self.V  # pad / start token of every codebook
        self.emb = nn.ModuleList([nn.Embedding(self.V + 1, d) for _ in range(self.K)])
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, self.V, bias=False) for _ in range(self.K)])
        for m in (self.emb, self.blocks, self.heads):  # not the codec
            m.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def codes(self, roll):
        """Roll windows [B, 3, 88, T + 2C] -> codes of the middle T frames [B, T, K]."""
        c = self.hparams.context
        self.codec.eval()  # frozen: in train mode its EMA codebooks would update (Lightning re-enters train mode)
        return self.codec.encode(roll)[:, c:roll.shape[-1] - c]

    def forward(self, inp, pos=None, caches=None):
        """inp [B, S, K] token ids (delayed) -> logits [B, S, K, V]."""
        if pos is None:
            pos = torch.arange(inp.shape[1], device=inp.device)
        x = sum(e(inp[..., k]) for k, e in enumerate(self.emb))
        for i, b in enumerate(self.blocks):
            x = b(x, pos, None if caches is None else caches[i])
        x = self.norm(x)
        return torch.stack([h(x) for h in self.heads], 2)

    def losses(self, roll):
        tgt = delay(self.codes(roll).long(), self.special)  # [B, S, K]
        inp = torch.cat([torch.full_like(tgt[:, :1], self.special), tgt[:, :-1]], 1)
        logits = self(inp).float()
        mask = tgt != self.special
        ce = F.cross_entropy(logits.permute(0, 3, 1, 2), tgt.masked_fill(~mask, -100), reduction='none')  # [B, S, K]
        per_k = (ce * mask).sum((0, 1)) / mask.sum((0, 1))
        return per_k.mean(), per_k

    def training_step(self, roll, i):
        loss, per_k = self.losses(roll)
        self.log('train_loss', loss, prog_bar=True)
        for k, v in enumerate(per_k):
            self.log(f'train/ce_q{k}', v)
        return loss

    def validation_step(self, roll, i):
        loss, per_k = self.losses(roll)
        self.log('val_loss', loss, sync_dist=True)
        for k, v in enumerate(per_k):
            self.log(f'val/ce_q{k}', v, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}],
                                lr=self.hparams.lr, betas=(0.9, 0.95))
        w, n = self.hparams.warmup, self.hparams.total_steps
        f = lambda s: min(1.0, (s + 1) / w) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, s / n))))
        return [opt], [{'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, f), 'interval': 'step'}]

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def generate(self, n_frames, prompt=None, batch=1, temperature=1.0, top_k=250):
        """Codes [B, n_frames, K]: continue `prompt` codes [B, P, K] (or start from nothing) with the delay pattern."""
        K, dev = self.K, self.device
        S = n_frames + K - 1
        seq = torch.full((batch if prompt is None else len(prompt), S, K), self.special, dtype=torch.long, device=dev)
        known = torch.zeros_like(seq, dtype=torch.bool)
        if prompt is not None:
            P = prompt.shape[1]
            d = delay(prompt.to(dev).long(), self.special)
            for k in range(K):
                seq[:, k:k + P, k] = d[:, k:k + P, k]
                known[:, k:k + P, k] = True
        caches = [dict() for _ in self.blocks]
        tok = torch.full_like(seq[:, :1], self.special)  # start token
        for s in range(S):
            logits = self(tok, torch.tensor([s], device=dev), caches)[:, 0].float() / max(temperature, 1e-5)  # [B, K, V]
            if top_k:
                kth = logits.topk(min(top_k, self.V), -1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, -float('inf'))
            nxt = torch.multinomial(logits.softmax(-1).flatten(0, 1), 1).view(-1, K)
            for k in range(K):  # outside codebook k's frames the delay pattern holds the special token
                if not 0 <= s - k < n_frames:
                    nxt[:, k] = self.special
            nxt = torch.where(known[:, s], seq[:, s], nxt)
            seq[:, s] = nxt
            tok = nxt[:, None]
        return undelay(seq, K)


# ---------------------------------------------------------------- stages

def train(args):
    n_gpus = max(torch.cuda.device_count(), 1)
    mix = (f'+{os.path.basename(args.mix_data)}{args.mix_frac:g}' if args.mix_data and args.mix_frac > 0 else '')
    model = MidiAR(args.codec, d=args.d, layers=args.layers, heads=args.heads, dropout=args.dropout, lr=args.lr,
                   total_steps=args.max_steps, warmup=args.warmup, context=args.context)
    # the name holds no GPU count, so a resubmit on another number of GPUs still resumes
    name = args.run_name or f'midi_ar_{os.path.basename(args.data)}{mix}_q{model.K}_d{args.d}x{args.layers}'
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    win = args.frames + 2 * args.context
    tr = RollDataset(args.data, win, args.batch_size, split='all' if args.val_data else 'train', stretch=args.stretch)
    if args.mix_data and args.mix_frac > 0:
        tr = MixedRollDataset(tr, RollDataset(args.mix_data, win, args.batch_size, split='all', stretch=args.stretch),
                              args.mix_frac)
        print(f'training windows: {1 - args.mix_frac:.0%} {args.data}, {args.mix_frac:.0%} {args.mix_data}')
    va = (RollDataset(args.val_data, win, args.batch_size, split='all', seed=0) if args.val_data
          else RollDataset(args.data, win, args.batch_size, split='val', seed=0))
    ckpt = L.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3, save_last=True,
                                       enable_version_counter=False, dirpath=f'ckpt/{name}',
                                       filename=name + '.{step}.{val_loss:.4f}')
    resume = f'ckpt/{name}/last.ckpt'
    resume = resume if os.path.isfile(resume) else None
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project='mitosis-midi-ar', name=name)
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger('logs_csv', name=name)
    strategy = 'auto'
    if n_gpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(timeout=datetime.timedelta(hours=2))
    gpu = torch.cuda.is_available()
    trainer = L.Trainer(devices=-1 if gpu else 1, accelerator='gpu' if gpu else 'cpu',
                        precision='bf16-mixed' if gpu else 32, max_steps=args.max_steps, callbacks=[ckpt],
                        val_check_interval=args.val_every, limit_val_batches=args.val_batches,
                        check_val_every_n_epoch=None, gradient_clip_val=1.0, logger=logger,
                        accumulate_grad_batches=args.accumulate,
                        num_sanity_val_steps=0 if resume else 1, strategy=strategy)
    print(f'run {name}: {n_params / 1e6:.0f}M parameters, codec {args.codec} ({model.K} x {model.V} codes), '
          f'{args.frames} frames ({args.frames * CODEC_MS / 1000:.0f} s) per window, '
          f'batch {args.batch_size} x {n_gpus} GPU x {args.accumulate}' + (f', resuming {resume}' if resume else ''),
          flush=True)
    dl = lambda ds: DataLoader(ds, batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0)
    trainer.fit(model, dl(tr), dl(va), ckpt_path=resume)


def sample(args):
    """N pieces as MIDI: <out>/<n>_{gen,prompt_orig}.mid; with --prompt_seconds, continuations of EVAL pieces."""
    from preprocess_time_grid import write_midi
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # the codec's weights travel in the AR checkpoint; --codec only names a file to build it from when the
    # path recorded at training time does not exist here (e.g. on another machine)
    over = {'codec_ckpt': args.codec} if args.codec else {}
    model = MidiAR.load_from_checkpoint(args.ckpt, map_location='cpu', weights_only=False, **over).to(device).eval()
    os.makedirs(args.out, exist_ok=True)
    n_frames = int(args.seconds * 1000 / CODEC_MS)
    P = int(args.prompt_seconds * 1000 / CODEC_MS)
    c = Corpus(args.data) if P else None
    rng = np.random.default_rng(args.seed)
    ids = rng.permutation(np.flatnonzero(c.length_ms >= (P + 100) * CODEC_MS))[:args.n] if P else range(args.n)
    ctx = model.hparams.context
    for n, i in enumerate(ids):
        prompt = None
        if P:
            notes = c.piece(i)
            roll = torch.from_numpy(notes_to_roll(notes, 0, P + ctx))[None].to(device)
            prompt = model.codec.encode(roll)[:, :P]
            orig = [(int(a['onset_ms']) / 1000, (int(a['onset_ms']) + int(a['dur_ms'])) / 1000, int(a['pitch']),
                     int(a['velocity']), 0) for a in notes
                    if a['program'] != 127 and a['onset_ms'] < n_frames * CODEC_MS]
            write_midi(orig, os.path.join(args.out, f'{n:02d}_piece{i}_orig.mid'))
        half = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device, dtype=half, enabled=device == 'cuda'):
            codes = model.generate(n_frames, prompt, temperature=args.temperature, top_k=args.top_k)
        est = midi_rvq.decode_piece(model.codec, codes[0].cpu().numpy(), device, threshold=args.threshold,
                                    min_gap=args.min_gap, frame_threshold=args.frame_threshold)
        stem = os.path.join(args.out, f'{n:02d}' + (f'_piece{i}_cont{args.prompt_seconds:g}s' if P else '_uncond'))
        write_midi([(t0, t1, p, v, 0) for t0, t1, p, v in est], stem + '.mid')
        print(f'{stem}.mid: {len(est)} notes', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    t = sub.add_parser('train')
    t.add_argument('--codec', required=True, help='MIDI RVQ checkpoint (midi_rvq.py), kept frozen')
    t.add_argument('--data', required=True, help='note corpus prefix, e.g. data/aria_dedup')
    t.add_argument('--mix_data', default=None, help='a second corpus mixed in at --mix_frac, e.g. data/maestro_train_midi')
    t.add_argument('--mix_frac', type=float, default=0.15)
    t.add_argument('--val_data', default=None, help='e.g. data/maestro_val_midi')
    t.add_argument('--d', type=int, default=1024)
    t.add_argument('--layers', type=int, default=24)
    t.add_argument('--heads', type=int, default=16)
    t.add_argument('--dropout', type=float, default=0.1)
    t.add_argument('--frames', type=int, default=1500, help='frames per window (1500 = 30 s, as MusicGen)')
    t.add_argument('--context', type=int, default=50, help='extra codec input frames on each side, dropped after encoding')
    t.add_argument('--batch_size', type=int, default=8, help='per GPU')
    t.add_argument('--accumulate', type=int, default=1)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--warmup', type=int, default=4000)
    t.add_argument('--max_steps', type=int, default=500000)
    t.add_argument('--stretch', type=float, default=0.1)
    t.add_argument('--val_every', type=int, default=2000)
    t.add_argument('--val_batches', type=int, default=20)
    t.add_argument('--workers', type=int, default=6)
    t.add_argument('--run_name', default=None)
    t.add_argument('--wandb', type=int, default=0)
    s = sub.add_parser('sample')
    s.add_argument('--ckpt', required=True)
    s.add_argument('--codec', default=None, help='codec checkpoint to build the model from when the training-time path is missing (weights still come from --ckpt)')
    s.add_argument('--data', default=None, help='corpus for prompts (with --prompt_seconds)')
    s.add_argument('--out', required=True)
    s.add_argument('--n', type=int, default=8)
    s.add_argument('--seconds', type=float, default=30.0, help='length of each sample, prompt included')
    s.add_argument('--prompt_seconds', type=float, default=0.0, help='0: unconditional')
    s.add_argument('--temperature', type=float, default=1.0)
    s.add_argument('--top_k', type=int, default=250)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--threshold', type=float, default=0.8)
    s.add_argument('--min_gap', type=int, default=2)
    s.add_argument('--frame_threshold', type=float, default=0.5)
    args = ap.parse_args()
    if args.stage == 'sample' and args.prompt_seconds and not args.data:
        ap.error('--prompt_seconds needs --data')
    torch.manual_seed(getattr(args, 'seed', 0))
    {'train': train, 'sample': sample}[args.stage](args)


if __name__ == '__main__':
    main()
