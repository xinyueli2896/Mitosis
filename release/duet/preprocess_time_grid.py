"""Time-based tokenization for the cp transformer, aligned to EnCodec frames.

The original backbone is tokenized in beat time: one frame per sixteenth,
read from MIDI ticks (preprocess_large_midi_dataset.py). That cannot be
aligned with audio, and performance MIDI such as MAESTRO has no beat grid
at all. Here every time value is in milliseconds, and the frame is a fixed
slice of time tied to MusicGen's EnCodec 32 kHz codec, whose frames are
20 ms:

  one symbolic frame = K codec frames = 20*K ms        (FRAME_CODEC = K)

so symbolic frame s covers exactly codec frames K*s .. K*s+K-1. Each note
records its onset offset 0..K-1 inside the frame, so onsets keep 20 ms
precision for any K; durations are counted in 20 ms units and snapped to
DURATION_TEMPLATES (20 ms .. 82 s). K = 6 (120 ms) is the default: frames
are dense enough that most hold notes, 1536 positions cover about three
minutes, and a frame is the 6-codec-frame block the joint decode drafts.
K = 1 gives one symbolic frame per codec frame.

A roll slot is 5 bytes (program, pitch, duration index, velocity, onset
offset), 254 in the program byte marking EOS and 255 padding; drums are
program 127. Inside a frame, notes are ordered by onset offset, then
pitch, then program (time order), and only the first max_polyphony are
kept. cp_transformer_time.py turns slots into tokens.

Corpora are stored once, untokenized, as one note record per note in
milliseconds (onset, duration, pitch, velocity, program), memory-mapped,
so a corpus of any size loads instantly and the frame size, polyphony
and time-stretch augmentation are all chosen at load time:

  data/<name>.notes.bin   NOTE_DTYPE records, piece after piece, by onset
  data/<name>.pieces.npz  per piece: note start, note count, length_ms,
                          pitch shift range, has_pedal, distinct velocities
  data/<name>.txt         index, source, MIDI path
  data/<name>.json        summary

Stages: `prep` builds a corpus from MIDI folders (and optionally MAESTRO's
train split); `stats` reports, per frame size, how full frames are and
how many notes the polyphony budget drops, which is what to look at before
fixing FRAME_CODEC and MAX_POLYPHONY for a pretraining run.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import IterableDataset

CODEC_MS = 20  # one EnCodec 32 kHz frame
DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
                               1536, 2048, 3072, 4096])  # in codec frames; as preprocess_large_midi_dataset.py
DURATION_BOUNDARIES = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
NOTE_DTYPE = np.dtype([('onset_ms', '<u4'), ('dur_ms', '<u2'), ('pitch', 'u1'), ('velocity', 'u1'), ('program', 'u1')])
SLOT = 5  # program, pitch, duration index, velocity, onset offset
EOS, PAD = 254, 255


# ---------------------------------------------------------------- MIDI -> notes

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
    """Note records (NOTE_DTYPE, sorted by onset) and whether any pedal was found.

    Times come from the MIDI's own tempo map, so they are seconds of
    performance, not beats. With pedal, a note released while its track's
    sustain pedal (CC64) is down keeps sounding until the pedal lifts or
    the same key on that track is struck again, which is what a recording
    hears. Drums are program 127 and never sustained.
    """
    out, has_pedal = [], False
    end_time = midi.get_end_time()
    for ins in midi.instruments:
        if not ins.notes:
            continue
        program = 127 if ins.is_drum else ins.program
        n = np.array([[x.start, x.end, x.pitch, x.velocity] for x in ins.notes], dtype=np.float64)
        n = n[np.lexsort((n[:, 2], n[:, 0]))]
        if pedal and not ins.is_drum:
            downs = _pedal_intervals(ins)
            has_pedal |= bool(downs)
            if downs:
                d_start = np.array([d[0] for d in downs])
                d_end = np.array([d[1] for d in downs])
                k = np.searchsorted(d_start, n[:, 1], side='right') - 1
                held = (k >= 0) & (n[:, 1] < d_end[np.maximum(k, 0)])
                n[held, 1] = d_end[k[held]]
            for p in np.unique(n[:, 2]):
                idx = np.where(n[:, 2] == p)[0]
                n[idx[:-1], 1] = np.minimum(n[idx[:-1], 1], n[idx[1:], 0])
            n[:, 1] = np.minimum(n[:, 1], end_time)
        rec = np.zeros(len(n), NOTE_DTYPE)
        on = np.maximum(np.rint(n[:, 0] * 1000), 0)
        rec['onset_ms'] = on
        rec['dur_ms'] = np.clip(np.rint(n[:, 1] * 1000) - on, 1, 65535)
        rec['pitch'] = n[:, 2]
        rec['velocity'] = n[:, 3]
        rec['program'] = program
        out.append(rec)
    if not out:
        return np.zeros(0, NOTE_DTYPE), has_pedal
    notes = np.concatenate(out)
    return notes[np.argsort(notes['onset_ms'], kind='stable')], has_pedal


def _piece(path, pedal):
    import pretty_midi
    try:
        midi = pretty_midi.PrettyMIDI(path)
        notes, has_pedal = notes_from_midi(midi, pedal)
    except Exception:
        return None
    if len(notes) == 0:
        return None
    pitched = notes['pitch'][notes['program'] != 127]
    shift = (-int(pitched.min()), 127 - int(pitched.max())) if len(pitched) else (0, 0)
    length_ms = int((notes['onset_ms'].astype(np.int64) + notes['dur_ms']).max())
    return notes, length_ms, shift, has_pedal, len(np.unique(notes['velocity']))


# ---------------------------------------------------------------- corpus files

class Corpus:
    """A memory-mapped note corpus written by build_corpus."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.notes = np.memmap(prefix + '.notes.bin', dtype=NOTE_DTYPE, mode='r')
        z = np.load(prefix + '.pieces.npz')
        self.start, self.count, self.length_ms = z['start'], z['count'], z['length_ms']
        self.shift, self.has_pedal, self.n_velocities = z['shift'], z['has_pedal'], z['n_velocities']

    def __len__(self):
        return len(self.count)

    def piece(self, i):
        return self.notes[self.start[i]:self.start[i] + self.count[i]]


def write_corpus(prefix, items):
    """items: iterable of (source, path, piece tuple). Returns the summary dict."""
    start, count, length_ms, shift, pedal, nvel, names = [], [], [], [], [], [], []
    n = 0
    with open(prefix + '.notes.bin', 'wb') as f:
        for source, path, (notes, ln, sh, hp, nv) in items:
            f.write(notes.tobytes())
            start.append(n)
            count.append(len(notes))
            n += len(notes)
            length_ms.append(ln)
            shift.append(sh)
            pedal.append(hp)
            nvel.append(nv)
            names.append((source, path))
    np.savez(prefix + '.pieces.npz', start=np.array(start, np.int64), count=np.array(count, np.int64),
             length_ms=np.array(length_ms, np.int64), shift=np.array(shift, np.int8).reshape(-1, 2),
             has_pedal=np.array(pedal, bool), n_velocities=np.array(nvel, np.int16))
    with open(prefix + '.txt', 'w') as f:
        for i, (s, p) in enumerate(names):
            f.write(f'{i}\t{s}\t{p}\n')
    return {'pieces': len(count), 'notes': int(n), 'hours': round(sum(length_ms) / 3.6e6, 1),
            'share_with_pedal': round(float(np.mean(pedal)), 3) if pedal else 0.0,
            'share_flat_velocity': round(float(np.mean(np.array(nvel) <= 2)), 3) if nvel else 0.0}


def list_inputs(midi_dirs, maestro_root):
    """(source, absolute path) for every MIDI file; MAESTRO contributes its train split only."""
    files = []
    for d in midi_dirs:
        files += [(os.path.basename(os.path.normpath(d)), os.path.join(r, f))
                  for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(('.mid', '.midi'))]
    if maestro_root:
        with open(os.path.join(maestro_root, 'maestro-v3.0.0.csv'), newline='', encoding='utf-8') as f:
            files += [('maestro-train', os.path.join(maestro_root, r['midi_filename']))
                      for r in csv.DictReader(f) if r['split'] == 'train']
    return sorted(files)


def build_corpus(name, midi_dirs, maestro_root=None, pedal=True, max_minutes=30.0, max_files=None, jobs=-1,
                 chunk=4096):
    from joblib import Parallel, delayed
    files = list_inputs(midi_dirs, maestro_root)
    if max_files:
        files = files[:max_files]
    print(f'{len(files)} MIDI files', flush=True)
    stats = {'unreadable_or_empty': 0, 'too_long': 0}

    def items():
        with Parallel(n_jobs=jobs) as par:
            for c in range(0, len(files), chunk):
                part = files[c:c + chunk]
                res = par(delayed(_piece)(p, pedal) for _, p in part)
                for (src, p), r in zip(part, res):
                    if r is None:
                        stats['unreadable_or_empty'] += 1
                    elif r[1] > max_minutes * 60000:
                        stats['too_long'] += 1
                    else:
                        yield src, p, r
                print(f'{min(c + chunk, len(files))}/{len(files)} files read', flush=True)

    os.makedirs('data', exist_ok=True)
    meta = write_corpus(f'data/{name}', items())
    meta.update(skipped=stats, pedal_folded=pedal, sources=sorted({s for s, _ in files}))
    with open(f'data/{name}.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)


# ---------------------------------------------------------------- tokenization

def notes_to_roll(notes, start_ms, n_frames, frame_codec, max_polyphony, stretch=1.0):
    """Roll [n_frames, P*SLOT] uint8 for the notes whose onset falls in the window.

    The window starts at start_ms and covers n_frames frames of
    20*frame_codec ms. With stretch != 1 every time is multiplied by it
    first (start_ms is in stretched time). Returns the roll and the
    number of notes dropped over the polyphony budget.
    """
    on = notes['onset_ms'].astype(np.float64) * stretch - start_ms
    c = np.rint(on / CODEC_MS).astype(np.int64)  # codec frame of the onset
    keep = (c >= 0) & (c < n_frames * frame_codec)
    notes, on, c = notes[keep], on[keep], c[keep]
    end_c = np.rint((on + notes['dur_ms'] * stretch) / CODEC_MS).astype(np.int64)
    rows = np.stack([c // frame_codec, notes['program'], notes['pitch'],
                     np.searchsorted(DURATION_BOUNDARIES, np.maximum(1, end_c - c)), notes['velocity'],
                     c % frame_codec], axis=1)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 2], rows[:, 5], rows[:, 0]))]  # frame, offset, pitch, program
    roll = np.full((n_frames, max_polyphony, SLOT), PAD, dtype=np.uint8)
    count = np.zeros(n_frames, dtype=np.int64)
    dropped = 0
    if len(rows):
        first = np.r_[0, np.flatnonzero(np.diff(rows[:, 0])) + 1]
        slot = np.arange(len(rows)) - np.repeat(first, np.diff(np.r_[first, len(rows)]))
        ok = slot < max_polyphony
        dropped = int((~ok).sum())
        roll[rows[ok, 0], slot[ok]] = rows[ok, 1:]
        np.add.at(count, rows[ok, 0], 1)
    eos = count < max_polyphony
    roll[eos, count[eos], 0] = EOS
    return roll.reshape(n_frames, -1), dropped


def roll_to_notes(roll, frame_codec, max_polyphony):
    """Inverse of notes_to_roll for listening: (onset s, end s, pitch, velocity, program) rows."""
    out = []
    r = np.asarray(roll).reshape(len(roll), max_polyphony, SLOT)
    for s in range(len(r)):
        for prog, pitch, d, vel, off in r[s]:
            if prog >= EOS:
                break
            t = (s * frame_codec + int(off)) * CODEC_MS / 1000
            out.append((t, t + DURATION_TEMPLATES[min(int(d), len(DURATION_TEMPLATES) - 1)] * CODEC_MS / 1000,
                        int(pitch), int(vel), int(prog)))
    return out


def write_midi(notes, path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    tracks = {}
    for t0, t1, pitch, vel, prog in notes:
        if prog not in tracks:
            tracks[prog] = pretty_midi.Instrument(program=0 if prog == 127 else prog, is_drum=prog == 127)
        tracks[prog].notes.append(pretty_midi.Note(velocity=max(1, min(127, vel)), pitch=pitch, start=t0, end=t1))
    pm.instruments = list(tracks.values())
    pm.write(path)


class TimeGridDataset(IterableDataset):
    """Random crops of a Corpus, tokenized on the fly.

    Yields (rolls [B, L, P*SLOT] uint8, pitch_shift [B]) like FramedDataset:
    L frames of 20*frame_codec ms. Splits are by piece index as in
    FramedDataset (val: index % 10 == 1, test: == 0, train: the rest).
    Training crops get a random time stretch in [1-stretch, 1+stretch]
    (playing faster or slower; a beat-time corpus cannot do this) and
    the usual pitch shift; val and test get neither.
    """

    def __init__(self, prefix, target_length, batch_size, frame_codec=6, max_polyphony=16, split='train',
                 stretch=0.1, random_order=True, repeat=True, split_ratio=10):
        self.prefix, self.L, self.batch_size = prefix, target_length, batch_size
        self.K, self.P, self.split = frame_codec, max_polyphony, split
        self.stretch = stretch if split == 'train' else 0.0
        self.random_order, self.repeat = random_order, repeat
        z = np.load(prefix + '.pieces.npz')
        idx = np.arange(len(z['count']))
        window_ms = target_length * frame_codec * CODEC_MS
        ok = z['length_ms'] * (1 - self.stretch) >= window_ms
        sel = {'all': np.ones_like(ok), 'train': idx % split_ratio > 1,
               'val': idx % split_ratio == 1, 'test': idx % split_ratio == 0}[split]
        self.valid_indices = idx[ok & sel]
        self.corpus = None
        print('Metadata for dataset', prefix, 'split', split, 'loaded. Number of valid songs:',
              len(self.valid_indices), f'({frame_codec * CODEC_MS} ms frames, {target_length} per crop)')

    def __iter__(self):
        if self.corpus is None:
            self.corpus = Corpus(self.prefix)
            r = self.corpus.shift.astype(np.int64)
            r[:, 0] = np.maximum(r[:, 0], -5)
            r[:, 1] = np.minimum(r[:, 1], 6)
            self.shift_range = r if self.split == 'train' else np.zeros_like(r)
            print('Data for dataset', self.prefix, 'loaded.')
        window_ms = self.L * self.K * CODEC_MS
        rng = np.random.default_rng()
        while True:
            order = rng.permutation(len(self.valid_indices)) if self.random_order else np.arange(len(self.valid_indices))
            for i in range(0, len(order), self.batch_size):
                ids = self.valid_indices[order[i:i + self.batch_size]]
                rolls, shifts = [], []
                for sid in ids:
                    a = rng.uniform(1 - self.stretch, 1 + self.stretch) if self.stretch else 1.0
                    span = max(0.0, self.corpus.length_ms[sid] * a - window_ms)
                    s0 = rng.uniform(0, span) if self.random_order else 0.0
                    s0 = round(s0 / CODEC_MS) * CODEC_MS  # crops start on a codec frame
                    rolls.append(notes_to_roll(self.corpus.piece(sid), s0, self.L, self.K, self.P, a)[0])
                    lo, hi = self.shift_range[sid]
                    shifts.append(rng.integers(lo, hi + 1))
                yield torch.from_numpy(np.stack(rolls)), torch.tensor(shifts, dtype=torch.long)
            if not self.repeat:
                break


# ---------------------------------------------------------------- stats

def corpus_stats(prefix, frame_codecs, max_polyphony, max_pieces=2000, seed=0):
    """Frame fill and polyphony overflow per frame size, over whole pieces."""
    c = Corpus(prefix)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(c))[:max_pieces]
    out = {'pieces_sampled': len(ids), 'share_with_pedal': round(float(c.has_pedal[ids].mean()), 3),
           'share_flat_velocity': round(float((c.n_velocities[ids] <= 2).mean()), 3), 'per_frame_size': {}}
    for K in frame_codecs:
        frames = notes = dropped = empty = full = 0
        per_frame = []
        for i in ids:
            n_frames = int(c.length_ms[i] // (K * CODEC_MS)) + 1
            roll, d = notes_to_roll(c.piece(i), 0, n_frames, K, max_polyphony)
            cnt = (roll.reshape(n_frames, max_polyphony, SLOT)[:, :, 0] < EOS).sum(1)
            frames += n_frames
            notes += int(cnt.sum()) + d
            dropped += d
            empty += int((cnt == 0).sum())
            full += int((cnt == max_polyphony).sum())
            per_frame.append(cnt[cnt > 0])
        pf = np.concatenate(per_frame) if per_frame else np.zeros(1)
        out['per_frame_size'][f'{K * CODEC_MS}ms'] = {
            'frame_codec': K, 'empty_frames': round(empty / frames, 3), 'frames_at_budget': round(full / frames, 4),
            'notes_dropped': round(dropped / max(1, notes), 4), 'notes_per_nonempty_frame_mean': round(float(pf.mean()), 2),
            'notes_per_nonempty_frame_p99': int(np.percentile(pf, 99)),
            'context_of_1536_positions_s': round(1536 * K * CODEC_MS / 1000, 1)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prep', help='MIDI folders -> data/<name> note corpus')
    p.add_argument('--name', required=True)
    p.add_argument('--midi_dir', nargs='*', default=[], help='folders searched recursively for .mid/.midi')
    p.add_argument('--maestro_root', default=None, help='also add MAESTRO v3 train-split MIDI (val/test are kept out)')
    p.add_argument('--pedal', type=int, default=1)
    p.add_argument('--max_minutes', type=float, default=30.0, help='skip longer pieces (usually broken files)')
    p.add_argument('--max_files', type=int, default=None)
    p.add_argument('--jobs', type=int, default=-1)
    s = sub.add_parser('stats', help='frame fill and polyphony overflow per frame size')
    s.add_argument('--name', required=True)
    s.add_argument('--frame_codec', type=int, nargs='+', default=[1, 3, 6, 12])
    s.add_argument('--max_polyphony', type=int, default=16)
    s.add_argument('--max_pieces', type=int, default=2000)
    args = ap.parse_args()
    if args.stage == 'prep':
        if not args.midi_dir and not args.maestro_root:
            sys.exit('give --midi_dir and/or --maestro_root')
        build_corpus(args.name, args.midi_dir, args.maestro_root, bool(args.pedal), args.max_minutes,
                     args.max_files, args.jobs)
    else:
        res = corpus_stats(f'data/{args.name}', args.frame_codec, args.max_polyphony, args.max_pieces)
        with open(f'data/{args.name}.stats_cp{args.max_polyphony}.json', 'w') as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
