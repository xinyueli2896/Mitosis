"""MAESTRO v3 as a paired corpus: performance MIDI frames + EnCodec 32 kHz codes.

The symbolic side is the time-grid tokenization of preprocess_time_grid.py
(the repository's cp format, frames of fixed absolute time) and the audio
side EnCodec 32 kHz (MusicGen's codec: 50 Hz, 4 codebooks of 2048), kept
as one continuous stream. MAESTRO has no tempo, so the frame is not a
note value: by default FRAME_HZ = 50, one symbolic frame per codec frame,
and frame f on both sides covers the same 20 ms. Other rates still work:
symbolic frame s owns codec frames [n(s), n(s+1)) with
n(s) = floor(s * 50 / FRAME_HZ). Both sides start at t = 0 of the
recording (the Disklavier MIDI is aligned to about 3 ms).

Stages (each run under SLURM, see slurm/maestro_*.sbatch):

  encode  WAV -> codes/<idx>.npy   int16 [F, 4], one file per piece.
          Resumable, and shardable over GPUs with --shard i --num_shards n.
  pack    MIDI -> rolls, and the per-piece codes, into one file per split:
            data/<name>_<split>.pt                  rolls   [sum T, P*4] uint8
            data/<name>_<split>.codes.pt            codes   [sum n(T), 4] int16 (= T at 50 Hz)
            data/<name>_<split>.length.pt           T per piece
            data/<name>_<split>.pitch_shift_range.pt zeros (audio cannot be shifted)
            data/<name>_<split>.txt                 idx, csv split, midi path
          so FramedDataset can read the symbolic half on its own.

MaestroPairedDataset yields aligned random crops of both halves.
"""
import argparse
import csv
import json
import math
from fractions import Fraction
import os
import sys

import numpy as np
import torch
from torch.utils.data import IterableDataset

from preprocess_time_grid import notes_from_midi, notes_to_rows, rows_to_roll

CODEC_HZ = 50
CODEC_SR = 32000
CODEC_HOP = CODEC_SR // CODEC_HZ  # 640 samples per code frame
N_CODEBOOKS = 4
SPLITS = ('train', 'validation', 'test')


def frames_per_step(frame_hz):
    """Codec frames per symbolic frame, exact (50 Hz -> 1)."""
    r = Fraction(CODEC_HZ) / Fraction(str(frame_hz))
    if r < 1:
        raise ValueError(f'FRAME_HZ={frame_hz}: a symbolic frame must cover at least one codec frame (FRAME_HZ <= 50)')
    return r


def step_bounds(s, r):
    """n(s): first codec frame of step s (works on ints and long tensors)."""
    return s * r.numerator // r.denominator


def read_index(root):
    """Rows of maestro-v3.0.0.csv, in csv order; idx is the row number."""
    with open(os.path.join(root, 'maestro-v3.0.0.csv'), newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r['idx'] = i
    return rows


# ---------------------------------------------------------------- audio

def load_wav_32k_mono(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, x = wavfile.read(path, mmap=True)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != CODEC_SR:
        g = math.gcd(sr, CODEC_SR)
        x = resample_poly(x, CODEC_SR // g, sr // g).astype(np.float32)
    return x


@torch.no_grad()
def encode_codes(model, wav, device, window_s=60.0, context_s=2.0, batch=8):
    """EnCodec codes [F, 4] for a whole recording, F = ceil(len / 640).

    The recording is cut into windows of window_s seconds, each encoded
    with context_s seconds of real audio on both sides (the encoder has
    an LSTM, so cutting cold would change codes near every cut); only
    the codes of the window itself are kept.
    """
    n_frames = math.ceil(len(wav) / CODEC_HOP)
    win = int(window_s * CODEC_HZ)
    ctx = int(context_s * CODEC_HZ)
    padded = np.zeros((n_frames + 2 * ctx) * CODEC_HOP + win * CODEC_HOP, dtype=np.float32)
    padded[ctx * CODEC_HOP: ctx * CODEC_HOP + len(wav)] = wav
    starts = list(range(0, n_frames, win))
    span = (win + 2 * ctx) * CODEC_HOP
    out = np.zeros((n_frames, N_CODEBOOKS), dtype=np.int16)
    for b in range(0, len(starts), batch):
        chunk = starts[b:b + batch]
        x = np.stack([padded[s * CODEC_HOP: s * CODEC_HOP + span] for s in chunk])
        x = torch.from_numpy(x).to(device)[:, None, :]
        codes = model.encode(x).audio_codes  # [1, B, 4, win + 2*ctx]
        codes = codes[0, :, :, ctx:ctx + win].transpose(1, 2).cpu().numpy()  # [B, win, 4]
        for s, c in zip(chunk, codes):
            e = min(s + win, n_frames)
            out[s:e] = c[:e - s]
    return out


def stage_encode(args):
    from transformers import EncodecModel
    rows = read_index(args.root)
    rows = [r for r in rows if r['idx'] % args.num_shards == args.shard]
    os.makedirs(args.codes_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EncodecModel.from_pretrained(args.codec).to(device).eval()
    assert model.config.sampling_rate == CODEC_SR, model.config.sampling_rate
    todo = [r for r in rows if not os.path.isfile(os.path.join(args.codes_dir, f"{r['idx']}.npy"))]
    print(f'shard {args.shard}/{args.num_shards}: {len(rows)} pieces, {len(rows) - len(todo)} already encoded', flush=True)
    for k, r in enumerate(todo):
        wav = load_wav_32k_mono(os.path.join(args.root, r['audio_filename']))
        codes = encode_codes(model, wav, device, batch=args.batch)
        dst = os.path.join(args.codes_dir, f"{r['idx']}.npy")
        np.save(dst + '.tmp.npy', codes)
        os.replace(dst + '.tmp.npy', dst)  # a killed job never leaves a short file behind
        print(f'[{k + 1}/{len(todo)}] {r["idx"]} {len(wav) / CODEC_SR:7.1f}s -> {codes.shape}', flush=True)


# ---------------------------------------------------------------- MIDI

def midi_to_roll(path, frame_hz, max_polyphony, n_frames, pedal=True):
    """Roll [n_frames, P*4] uint8 (preprocess_time_grid's tokenization), dropped and kept note counts."""
    import pretty_midi
    notes = notes_from_midi(pretty_midi.PrettyMIDI(path), pedal=pedal)
    rows, dropped, _ = notes_to_rows(notes, frame_hz, max_polyphony, n_frames)
    return rows_to_roll(rows, n_frames, max_polyphony), dropped, len(rows)


def stage_pack(args):
    from joblib import Parallel, delayed
    R = frames_per_step(args.frame_hz)
    grid_hz = float(args.frame_hz)
    rows = read_index(args.root)
    missing = [r['idx'] for r in rows if not os.path.isfile(os.path.join(args.codes_dir, f"{r['idx']}.npy"))]
    if missing:
        sys.exit(f'{len(missing)} pieces have no codes in {args.codes_dir} (first: {missing[:5]}); run the encode stage first')
    os.makedirs('data', exist_ok=True)
    summary = {'frame_hz': args.frame_hz, 'codec_frames_per_step': [R.numerator, R.denominator], 'max_polyphony': args.max_polyphony,
               'pedal': args.pedal, 'codec': args.codec, 'splits': {}}

    def one(r):
        codes = np.load(os.path.join(args.codes_dir, f"{r['idx']}.npy"))
        T = len(codes) * R.denominator // R.numerator  # whole steps the audio covers
        roll, dropped, kept = midi_to_roll(os.path.join(args.root, r['midi_filename']),
                                           grid_hz, args.max_polyphony, T, pedal=args.pedal)
        return roll, codes[:step_bounds(T, R)], dropped, kept

    for split in SPLITS:
        part = [r for r in rows if r['split'] == split]
        res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(r) for r in part)
        name = f'{args.name}_{split}'
        torch.save(torch.from_numpy(np.concatenate([x[0] for x in res])), f'data/{name}.pt')
        torch.save(torch.from_numpy(np.concatenate([x[1] for x in res])), f'data/{name}.codes.pt')
        torch.save(torch.tensor([len(x[0]) for x in res]), f'data/{name}.length.pt')
        torch.save(torch.zeros(len(res), 2, dtype=torch.int8), f'data/{name}.pitch_shift_range.pt')
        with open(f'data/{name}.txt', 'w') as f:
            for r in part:
                f.write(f"{r['idx']}\t{r['split']}\t{r['midi_filename']}\n")
        dropped = sum(x[2] for x in res)
        kept = sum(x[3] for x in res)
        hours = sum(len(x[0]) for x in res) / grid_hz / 3600
        summary['splits'][split] = {'pieces': len(res), 'hours': round(hours, 2),
                                    'notes': kept, 'notes_dropped_over_polyphony': dropped}
        print(f'{split}: {len(res)} pieces, {hours:.1f} h, {kept} notes, '
              f'{dropped} dropped over cp{args.max_polyphony} ({100 * dropped / max(1, kept + dropped):.3f}%)', flush=True)
    with open(f'data/{args.name}.json', 'w') as f:
        json.dump(summary, f, indent=1)


# ---------------------------------------------------------------- loader

class MaestroPairedDataset(IterableDataset):
    """Aligned random crops of rolls and codes, in FramedDataset's style.

    Yields (rolls [B, L, P*4] uint8, codes [B, L*R, 4] long,
    bounds [B, L+1] long) per batch, where L = target_length steps and
    R codec frames per step. bounds[b, j] is the first codec frame of
    step j inside the crop, so step j owns codes[b, bounds[b, j]:bounds[b, j+1]]
    (the bridge mask and the per-step pooling read it). L*R must be a
    whole number (any L at 50 Hz, where R = 1) so every crop has the
    same audio length. Crops start on multiples of R's
    denominator by default, which fixes the phase of a 6, 6, 6, 7
    pattern; bounds stays correct for any sample_step. Splits are MAESTRO's own (one file per split),
    so there is no modulo split, and no pitch shift: the audio is fixed.
    """

    def __init__(self, file_path, target_length, batch_size, random_order=True, repeat=True, sample_step=None):
        self.file_path = file_path
        with open(file_path.rsplit('_', 1)[0] + '.json') as f:
            meta = json.load(f)
        self.R = Fraction(*meta['codec_frames_per_step'])
        if (target_length * self.R).denominator != 1:
            raise ValueError(f'target_length={target_length} x {self.R} codec frames per step is not whole; '
                             f'use a multiple of {self.R.denominator}')
        self.length = torch.load(file_path[:-3] + '.length.pt', weights_only=True)
        self.start = torch.cumsum(self.length, dim=0) - self.length
        self.valid_indices = torch.arange(len(self.length))[self.length >= target_length]
        self.target_length = target_length
        self.batch_size = batch_size
        self.random_order = random_order
        self.repeat = repeat
        self.sample_step = sample_step or self.R.denominator
        self.code_start = torch.cumsum(step_bounds(self.length, self.R), dim=0) - step_bounds(self.length, self.R)
        self.rolls = self.codes = None
        print('Metadata for dataset', file_path, 'loaded. Valid pieces:', len(self.valid_indices),
              'of', len(self.length), 'codec frames per frame:', self.R)

    def __iter__(self):
        if self.rolls is None:
            self.rolls = torch.load(self.file_path, weights_only=True)
            self.codes = torch.load(self.file_path[:-3] + '.codes.pt', weights_only=True)
            assert len(self.codes) == int(step_bounds(self.length, self.R).sum()), (len(self.codes), self.R)
            print('Data for dataset', self.file_path, 'loaded.')
        L, R = self.target_length, self.R
        n_aud = int(L * R)
        while True:
            order = torch.randperm(len(self.valid_indices)) if self.random_order else torch.arange(len(self.valid_indices))
            for i in range(0, len(order), self.batch_size):
                ids = self.valid_indices[order[i:i + self.batch_size]]
                local = torch.floor(torch.rand(len(ids)) * ((self.length[ids] - L) // self.sample_step + 1)).long() \
                    * self.sample_step
                sym = (self.start[ids] + local)[:, None] + torch.arange(L)[None]
                a0 = step_bounds(local, R)
                aud = (self.code_start[ids] + a0)[:, None] + torch.arange(n_aud)[None]
                bounds = step_bounds(local[:, None] + torch.arange(L + 1)[None], R) - a0[:, None]
                yield self.rolls[sym], self.codes[aud].long(), bounds
            if not self.repeat:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['encode', 'pack'])
    ap.add_argument('--root', required=True, help='unzipped maestro-v3.0.0 folder (holds maestro-v3.0.0.csv)')
    ap.add_argument('--codes_dir', default='data/maestro_encodec32k')
    ap.add_argument('--codec', default='facebook/encodec_32khz')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8, help='encode: windows per forward pass')
    ap.add_argument('--frame_hz', type=float, default=50.0, help='symbolic frames per second; 50 = one per codec frame')
    ap.add_argument('--max_polyphony', type=int, default=16)
    ap.add_argument('--pedal', type=int, default=1, help='pack: extend offsets through the sustain pedal')
    ap.add_argument('--name', default=None, help='pack: default maestro_t<FRAME_HZ>_cp<P>')
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()
    if args.stage == 'encode':
        stage_encode(args)
    else:
        frames_per_step(args.frame_hz)
        if args.name is None:
            args.name = f"maestro_t{args.frame_hz:g}_cp{args.max_polyphony}".replace('.', 'p')
        stage_pack(args)


if __name__ == '__main__':
    main()
