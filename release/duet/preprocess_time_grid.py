"""Time-grid tokenization for the cp transformer: one frame per EnCodec frame.

The original corpus is tokenized in beat time (one frame per sixteenth,
preprocess_large_midi_dataset.py). Performance MIDI such as MAESTRO has
no beat grid, so here a frame is a fixed slice of absolute time,
FRAME_HZ frames per second. The default, 50 Hz (20 ms), is exactly one
frame of MusicGen's EnCodec 32 kHz, so symbolic frame f and codec frame f
cover the same 20 ms. Everything else is the repository's cp format:
up to P notes per frame as (program, pitch, duration index, velocity),
durations counted in frames and snapped to DURATION_TEMPLATES (1 frame =
20 ms up to 4096 = 82 s), drums on program 127, so the model
architecture and tokenizer are unchanged.

At 50 Hz most frames are empty, and a dense [T, P*4] roll of a large
corpus would be several hundred GB. The corpus is therefore stored
sparse, one row per note, and SparseFramedDataset densifies random crops
on the fly into exactly what FramedDataset yields:

  data/<name>.notes.pt              int32 [N, 6]: frame, slot, program, pitch, duration index, velocity
  data/<name>.note_count.pt         notes per piece
  data/<name>.length.pt             frames per piece
  data/<name>.pitch_shift_range.pt  per piece, as in the beat-time corpora
  data/<name>.txt                   index, MIDI path
  data/<name>.json                  frame rate, polyphony, overflow statistics

Notes are sorted by piece, frame, then slot; slot is the note's place in
its frame, already truncated to P.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import IterableDataset

DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
                               1536, 2048, 3072, 4096])  # as preprocess_large_midi_dataset.py
DURATION_BOUNDARIES = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2


def _pedal_intervals(ins, threshold=64):
    cc = sorted((c.time, c.value >= threshold) for c in ins.control_changes if c.number == 64)
    downs, t_down = [], None
    for t, on in cc:
        if on and t_down is None:
            t_down = t
        elif not on and t_down is not None:
            downs.append((t_down, t))
            t_down = None
    if t_down is not None:
        downs.append((t_down, float('inf')))
    return downs


def notes_from_midi(midi, pedal=True):
    """[N, 5] float64 rows (start, end, pitch, velocity, program) in seconds.

    With pedal, a note released while its track's sustain pedal (CC64)
    is down keeps sounding until the pedal lifts or the same key on the
    same track is struck again, which is what a recording hears. Drums
    are program 127 and never sustained.
    """
    out = []
    end_time = midi.get_end_time()
    for ins in midi.instruments:
        if not ins.notes:
            continue
        program = 127 if ins.is_drum else ins.program
        n = np.array([[x.start, x.end, x.pitch, x.velocity, program] for x in ins.notes], dtype=np.float64)
        if pedal and not ins.is_drum:
            downs = _pedal_intervals(ins)
            if downs:
                d_start = np.array([d[0] for d in downs])
                d_end = np.array([d[1] for d in downs])
                k = np.searchsorted(d_start, n[:, 1], side='right') - 1
                held = (k >= 0) & (n[:, 1] < d_end[np.maximum(k, 0)])
                n[held, 1] = d_end[k[held]]
            n = n[np.lexsort((n[:, 2], n[:, 0]))]
            for p in np.unique(n[:, 2]):
                idx = np.where(n[:, 2] == p)[0]
                n[idx[:-1], 1] = np.minimum(n[idx[:-1], 1], n[idx[1:], 0])
            n[:, 1] = np.minimum(n[:, 1], end_time)
        out.append(n)
    return np.concatenate(out) if out else np.zeros((0, 5))


def notes_to_rows(notes, frame_hz, max_polyphony, n_frames=None):
    """Note rows int32 [M, 6] (frame, slot, program, pitch, duration index, velocity).

    A note's frame is round(start * frame_hz) and its duration the
    rounded end frame minus that, at least 1. Inside a frame notes are
    ordered as preprocess_midi orders them (velocity, then program,
    pitch, duration), and only the first max_polyphony are kept.
    Returns the rows, the number of notes dropped, and the frame count.
    """
    if n_frames is None:
        n_frames = int(np.ceil(notes[:, 1].max() * frame_hz)) + 1 if len(notes) else 0
    t = np.rint(notes[:, 0] * frame_hz).astype(np.int64)
    dur = np.maximum(1, np.rint(notes[:, 1] * frame_hz).astype(np.int64) - t)
    keep = (t >= 0) & (t < n_frames)
    t, dur, notes = t[keep], dur[keep], notes[keep]
    rows = np.stack([t, np.zeros_like(t), notes[:, 4].astype(np.int64), notes[:, 2].astype(np.int64),
                     np.searchsorted(DURATION_BOUNDARIES, dur), notes[:, 3].astype(np.int64)], axis=1)
    rows = rows[np.lexsort((rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 5], rows[:, 0]))]
    if len(rows):
        first = np.r_[0, np.flatnonzero(np.diff(rows[:, 0])) + 1]
        rows[:, 1] = np.arange(len(rows)) - np.repeat(first, np.diff(np.r_[first, len(rows)]))
    kept = rows[:, 1] < max_polyphony
    return rows[kept].astype(np.int32), int((~kept).sum()), n_frames


def rows_to_roll(rows, n_frames, max_polyphony, frame_offset=0):
    """Dense roll [n_frames, P*4] uint8 in the repository's format (254 EOS, 255 pad)."""
    roll = np.full((n_frames, max_polyphony, 4), 255, dtype=np.uint8)
    count = np.zeros(n_frames, dtype=np.int64)
    if len(rows):
        f = rows[:, 0] - frame_offset
        roll[f, rows[:, 1]] = rows[:, 2:6]
        np.add.at(count, f, 1)
    eos = count < max_polyphony
    roll[eos, count[eos], 0] = 254
    return roll.reshape(n_frames, -1)


def _one(path, frame_hz, max_polyphony, pedal):
    import pretty_midi
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None
    notes = notes_from_midi(midi, pedal=pedal)
    if len(notes) == 0:
        return None
    rows, dropped, n_frames = notes_to_rows(notes, frame_hz, max_polyphony)
    pitched = rows[rows[:, 2] != 127, 3]
    shift = [-int(pitched.min()), 127 - int(pitched.max())] if len(pitched) else [0, 0]
    return rows, dropped, n_frames, shift


def build_corpus(folder, name, frame_hz, max_polyphony, pedal, max_frames, jobs, max_files=None):
    from joblib import Parallel, delayed
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
                   if f.lower().endswith(('.mid', '.midi')))
    if max_files:
        files = files[:max_files]
    print(f'{len(files)} MIDI files under {folder}', flush=True)
    res = Parallel(n_jobs=jobs, verbose=5, batch_size=64)(
        delayed(_one)(p, frame_hz, max_polyphony, pedal) for p in files)
    kept = [(p, r) for p, r in zip(files, res) if r is not None and r[2] <= max_frames]
    skipped = len(files) - len(kept)
    os.makedirs('data', exist_ok=True)
    rows = np.concatenate([r[0] for _, r in kept])
    torch.save(torch.from_numpy(rows), f'data/{name}.notes.pt')
    torch.save(torch.tensor([len(r[0]) for _, r in kept]), f'data/{name}.note_count.pt')
    torch.save(torch.tensor([r[2] for _, r in kept]), f'data/{name}.length.pt')
    torch.save(torch.tensor([r[3] for _, r in kept], dtype=torch.int8), f'data/{name}.pitch_shift_range.pt')
    with open(f'data/{name}.txt', 'w') as f:
        for i, (p, _) in enumerate(kept):
            f.write(f'{i}\t{os.path.relpath(p, folder)}\n')
    dropped = sum(r[1] for _, r in kept)
    frames = sum(r[2] for _, r in kept)
    meta = {'frame_hz': frame_hz, 'max_polyphony': max_polyphony, 'pedal': pedal, 'source': folder,
            'pieces': len(kept), 'skipped_unreadable_empty_or_too_long': skipped, 'notes': int(len(rows)),
            'notes_dropped_over_polyphony': int(dropped), 'hours': round(frames / frame_hz / 3600, 1)}
    with open(f'data/{name}.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)


class SparseFramedDataset(IterableDataset):
    """FramedDataset over a sparse note corpus: same splits, crops and pitch shift.

    Yields (rolls [B, L, P*4] uint8, pitch_shift [B]) exactly like
    FramedDataset, so the cp transformer trains on it unchanged.
    """

    def __init__(self, file_path, target_length, batch_size, split='all', split_ratio=10, random_order=True,
                 repeat=True):
        assert file_path.endswith('.notes.pt'), file_path
        self.base = file_path[:-len('.notes.pt')]
        with open(self.base + '.json') as f:
            self.max_polyphony = json.load(f)['max_polyphony']
        self.length = torch.load(self.base + '.length.pt', weights_only=True)
        counts = torch.load(self.base + '.note_count.pt', weights_only=True)
        self.note_start = torch.cumsum(counts, 0) - counts
        self.note_count = counts
        idx = torch.arange(len(self.length))
        ok = self.length >= target_length
        sel = {'all': torch.ones_like(ok), 'train': idx % split_ratio > 1,
               'val': idx % split_ratio == 1, 'test': idx % split_ratio == 0}[split]
        self.valid_indices = idx[ok & sel]
        self.split, self.target_length, self.batch_size = split, target_length, batch_size
        self.random_order, self.repeat = random_order, repeat
        self.notes = self.pitch_shift_range = None
        print('Metadata for dataset', file_path, 'split', split, 'loaded. Number of valid songs:',
              len(self.valid_indices), 'first 20:', self.valid_indices[:20])

    def __iter__(self):
        if self.notes is None:
            self.notes = torch.load(self.base + '.notes.pt', weights_only=True).numpy()
            r = torch.load(self.base + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2).long()
            r[:, 0].clamp_(min=-5)
            r[:, 1].clamp_(max=6)
            if self.split in ('val', 'test'):
                r.zero_()
            self.pitch_shift_range = r
            print('Data for dataset', self.base + '.notes.pt', 'loaded.')
        L, P = self.target_length, self.max_polyphony
        while True:
            order = torch.randperm(len(self.valid_indices)) if self.random_order else torch.arange(len(self.valid_indices))
            for i in range(0, len(order), self.batch_size):
                ids = self.valid_indices[order[i:i + self.batch_size]]
                starts = torch.floor(torch.rand(len(ids)) * (self.length[ids] - L + 1)).long()
                rolls = []
                for sid, s0 in zip(ids.tolist(), starts.tolist()):
                    a = int(self.note_start[sid])
                    piece = self.notes[a:a + int(self.note_count[sid])]
                    lo, hi = np.searchsorted(piece[:, 0], [s0, s0 + L])
                    rolls.append(rows_to_roll(piece[lo:hi], L, P, frame_offset=s0))
                rng = self.pitch_shift_range[ids]
                shift = torch.floor(torch.rand(len(ids)) * (rng[:, 1] - rng[:, 0] + 1)).long() + rng[:, 0]
                yield torch.from_numpy(np.stack(rolls)), shift
            if not self.repeat:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--midi_dir', required=True, help='folder searched recursively for .mid/.midi')
    ap.add_argument('--name', required=True, help='output data/<name>.*')
    ap.add_argument('--frame_hz', type=float, default=50.0, help='frames per second; 50 = one EnCodec 32 kHz frame')
    ap.add_argument('--max_polyphony', type=int, default=16)
    ap.add_argument('--pedal', type=int, default=1)
    ap.add_argument('--max_minutes', type=float, default=30.0, help='skip longer pieces (usually broken files)')
    ap.add_argument('--max_files', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()
    build_corpus(args.midi_dir, args.name, args.frame_hz, args.max_polyphony, bool(args.pedal),
                 int(args.max_minutes * 60 * args.frame_hz), args.jobs, args.max_files)


if __name__ == '__main__':
    main()
