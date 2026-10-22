"""MAESTRO v3 as a paired corpus: performance MIDI frames + EnCodec 32 kHz codes.

The symbolic side uses the repository's own roll format (the format
preprocess_large_midi_dataset.py writes and FramedDataset reads):
[T, P*4] uint8 rows of (program, pitch, duration index, velocity) per
note slot, 254 marking EOS and 255 padding, so a cp checkpoint's
tokenizer reads it unchanged. MAESTRO has no beat grid, so a frame is
a fixed slice of time, GRID_HZ frames per second. The default, 8 Hz, is
a sixteenth note at 120 BPM (0.125 s), so durations keep the meaning
the cp checkpoint learned: the MIDI is read as if it were at 120 BPM.

The audio side is EnCodec 32 kHz (MusicGen's codec): 50 Hz, 4
codebooks of 2048, kept as one continuous stream. Symbolic step s owns
codec frames [n(s), n(s+1)) with n(s) = floor(s * 50 / GRID_HZ); at
8 Hz that is 6.25 frames per step, a repeating 6, 6, 6, 7 pattern. Both
sides start at t = 0 of the recording (MAESTRO's MIDI is captured by
the Disklavier in the same session, aligned to about 3 ms).

Stages (each run under SLURM, see slurm/maestro_*.sbatch):

  encode  WAV -> codes/<idx>.npy   int16 [F, 4], one file per piece.
          Resumable, and shardable over GPUs with --shard i --num_shards n.
  pack    MIDI -> rolls, and the per-piece codes, into one file per split:
            data/<name>_<split>.pt                  rolls   [sum T, P*4] uint8
            data/<name>_<split>.codes.pt            codes   [sum n(T), 4] int16
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

from preprocess_large_midi_dataset import DURATION_TEMPLATES

CODEC_HZ = 50
CODEC_SR = 32000
CODEC_HOP = CODEC_SR // CODEC_HZ  # 640 samples per code frame
N_CODEBOOKS = 4
SPLITS = ('train', 'validation', 'test')


def frames_per_grid(grid_hz):
    """Codec frames per symbolic step, as an exact fraction (8 Hz -> 25/4)."""
    r = Fraction(CODEC_HZ) / Fraction(str(grid_hz))
    if r < 1:
        raise ValueError(f'GRID_HZ={grid_hz}: a step must cover at least one codec frame (GRID_HZ <= 50)')
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

def sustained_notes(midi, pedal_threshold=64):
    """Notes with offsets extended through the sustain pedal (CC64).

    A note released while the pedal is down keeps sounding until the
    pedal lifts, or until the same key is struck again, which is what
    the recording hears. Returns (start, end, pitch, velocity) arrays.
    """
    notes = []
    for ins in midi.instruments:
        if ins.is_drum:
            continue
        cc = sorted((c.time, c.value >= pedal_threshold) for c in ins.control_changes if c.number == 64)
        downs = []  # [down, up) intervals
        t_down = None
        for t, on in cc:
            if on and t_down is None:
                t_down = t
            elif not on and t_down is not None:
                downs.append((t_down, t))
                t_down = None
        if t_down is not None:
            downs.append((t_down, float('inf')))
        d_starts = np.array([d[0] for d in downs]) if downs else np.zeros(0)
        for n in ins.notes:
            end = n.end
            k = np.searchsorted(d_starts, end, side='right') - 1
            if k >= 0 and downs[k][0] <= end < downs[k][1]:
                end = downs[k][1]
            notes.append([n.start, end, n.pitch, n.velocity])
    if not notes:
        return np.zeros((0, 4))
    notes = np.array(notes, dtype=np.float64)
    notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
    # a re-struck key cuts the previous sounding note of that pitch, and
    # a pedal never lifted ends with the recording
    for p in np.unique(notes[:, 2]):
        idx = np.where(notes[:, 2] == p)[0]
        notes[idx[:-1], 1] = np.minimum(notes[idx[:-1], 1], notes[idx[1:], 0])
    notes[:, 1] = np.minimum(notes[:, 1], midi.get_end_time())
    return notes


def midi_to_roll(path, grid_hz, max_polyphony, n_frames, pedal=True):
    """Roll [n_frames, P*4] uint8 in the repository's format, plus overflow count."""
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(path)
    if pedal:
        notes = sustained_notes(midi)
    else:
        notes = np.array([[n.start, n.end, n.pitch, n.velocity]
                          for ins in midi.instruments if not ins.is_drum for n in ins.notes], dtype=np.float64).reshape(-1, 4)
        notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
    boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    roll = np.full((n_frames, max_polyphony, 4), 255, dtype=np.uint8)
    count = np.zeros(n_frames, dtype=np.int32)
    dropped = 0
    for start, end, pitch, vel in notes:
        t = int(round(start * grid_hz))
        if t < 0 or t >= n_frames:
            continue
        if count[t] >= max_polyphony:
            dropped += 1
            continue
        dur = max(1, int(round(end * grid_hz)) - t)
        # piano is program 0; the grid is time, so the duration index
        # counts GRID_HZ frames, not sixteenth notes
        roll[t, count[t]] = [0, int(pitch), int(np.searchsorted(boundaries, dur)), int(vel)]
        count[t] += 1
    for t in np.nonzero(count)[0]:
        c = count[t]
        roll[t, :c] = roll[t, :c][np.lexsort((roll[t, :c, 2], roll[t, :c, 1], roll[t, :c, 0], roll[t, :c, 3]))]
    eos = count < max_polyphony
    roll[eos, count[eos], 0] = 254
    return roll.reshape(n_frames, -1), dropped, int(count.sum())


def stage_pack(args):
    from joblib import Parallel, delayed
    R = frames_per_grid(args.grid_hz)
    rows = read_index(args.root)
    missing = [r['idx'] for r in rows if not os.path.isfile(os.path.join(args.codes_dir, f"{r['idx']}.npy"))]
    if missing:
        sys.exit(f'{len(missing)} pieces have no codes in {args.codes_dir} (first: {missing[:5]}); run the encode stage first')
    os.makedirs('data', exist_ok=True)
    summary = {'grid_hz': args.grid_hz, 'codec_frames_per_step': [R.numerator, R.denominator], 'max_polyphony': args.max_polyphony,
               'pedal': args.pedal, 'codec': args.codec, 'splits': {}}

    def one(r):
        codes = np.load(os.path.join(args.codes_dir, f"{r['idx']}.npy"))
        T = len(codes) * R.denominator // R.numerator  # whole steps the audio covers
        roll, dropped, kept = midi_to_roll(os.path.join(args.root, r['midi_filename']),
                                           args.grid_hz, args.max_polyphony, T, pedal=args.pedal)
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
        hours = sum(len(x[0]) for x in res) / args.grid_hz / 3600
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
    whole number (at 8 Hz, L a multiple of 4) so every crop has the
    same audio length. Crops start on multiples of R's denominator by
    default, which fixes the 6, 6, 6, 7 phase; bounds stays correct
    for any sample_step. Splits are MAESTRO's own (one file per split),
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
    ap.add_argument('--grid_hz', type=float, default=8.0, help='steps per second; 8 = a 16th at 120 BPM')
    ap.add_argument('--max_polyphony', type=int, default=16)
    ap.add_argument('--pedal', type=int, default=1, help='pack: extend offsets through the sustain pedal')
    ap.add_argument('--name', default=None, help='pack: default maestro_g<grid>_cp<P>')
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()
    if args.stage == 'encode':
        stage_encode(args)
    else:
        frames_per_grid(args.grid_hz)
        if args.name is None:
            args.name = f"maestro_g{args.grid_hz:g}_cp{args.max_polyphony}".replace('.', 'p')
        stage_pack(args)


if __name__ == '__main__':
    main()
