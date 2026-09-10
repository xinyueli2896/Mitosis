"""Diagnose the bar-phase of the aligned POP909 files (2026-09-10).

preprocess_pop909_align.py writes audio beat i at output tick (i + 1) * PPQ
(a one-beat lead-in segment occupies ticks 0..PPQ) and never reads the
downbeat column of beat_midi.txt. With bars counted from tick 0, the
first downbeat of a song lands on a bar boundary only when its beat
index d0 satisfies (d0 + 1) % 4 == 0. Otherwise the whole song -- and
everything built from it: melody/chord folders, training .pt files,
E1 prompts -- is shifted by (d0 + 1) % 4 beats relative to the grid.

This script reads the RAW beat_midi.txt of each song, reports d0 and the
predicted shift, histograms the shift over the corpus, and, when the
aligned melody folder is given, checks the aligned file: the tick of
beat d0 must be (d0 + 1) * PPQ under the current aligner.

--chord-dir mode (2026-09-10): the raw annotations are not on the
cluster for most songs, so measure the phase on the files actually
used. POP909 chords change on downbeats and half-bars far more than
elsewhere, so the histogram of chord-change positions modulo 4 beats,
on each chord file's own tick grid, locates the bar start: the modal
position is the shift of that file (0 = on the grid). Two-beat
ambiguity (0 vs 2) is resolved by preferring the larger count; the
report prints all four counts so you can see how sharp the mode is.
Also reports whether the file carries the aligner's lead-in signature
(first tempo event above 1000 bpm).

Usage (via check_downbeat_phase.sbatch, CPU):
    python check_downbeat_phase.py --raw /home/xinyue.li/POP909-Dataset/POP909 \\
        --aligned /home/xinyue.li/POP909-Dataset/POP909-aligned \\
        --chord-dir /home/xinyue.li/POP909-Dataset/POP909-chord \\
        --ids 004 046 136 326 456 466 746 816
"""
import argparse
import os
from collections import Counter
from glob import glob

import numpy as np


def read_beats(path):
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    return rows


def first_downbeat(rows):
    """Index of the first row whose LAST column is 1 (the downbeat flag in
    the 3-column format; in a 2-column file the second column)."""
    flag = rows[:, -1]
    idx = np.where(flag >= 0.5)[0]
    return int(idx[0]) if len(idx) else None


def chord_phase(chord_dir, detail):
    import mido
    files = sorted(glob(os.path.join(chord_dir, '*.mid')))
    print(f'=== chord-change phase on the files in {chord_dir} ({len(files)} files) ===')
    print(f'{"song":>5} {"lead-in":>8} {"changes":>8}  counts at beat 0/1/2/3 of the 4-beat grid   mode  (shift)')
    hist = Counter()
    n = 0
    for fp in files:
        sid = os.path.splitext(os.path.basename(fp))[0]
        mid = mido.MidiFile(fp)
        ppq = mid.ticks_per_beat
        first_bpm = None
        # sounding pitch-class set per beat (quantised to the beat), then changes
        notes = []
        for tr in mid.tracks:
            t = 0
            on = {}
            for msg in tr:
                t += msg.time
                if msg.type == 'set_tempo' and first_bpm is None:
                    first_bpm = 6e7 / msg.tempo if msg.tempo > 0 else float('inf')
                if msg.type == 'note_on' and msg.velocity > 0:
                    on[msg.note] = t
                elif msg.type in ('note_off', 'note_on'):
                    if msg.note in on:
                        notes.append((on.pop(msg.note), t, msg.note))
        if not notes:
            continue
        last = max(e for _, e, _ in notes)
        n_beats = int(last // ppq) + 1
        pcs = [set() for _ in range(n_beats)]
        for s0, e0, pitch in notes:
            b0 = int(s0 // ppq)
            b1 = max(b0, int((e0 - 1) // ppq))
            for b in range(b0, min(b1, n_beats - 1) + 1):
                pcs[b].add(pitch % 12)
        counts = [0, 0, 0, 0]
        prev = None
        n_ch = 0
        for b, pc in enumerate(pcs):
            if pc and pc != prev:
                if prev is not None:
                    counts[b % 4] += 1
                    n_ch += 1
                prev = pc
            elif pc:
                prev = pc
        if n_ch < 4:
            continue
        mode = max(range(4), key=lambda k: counts[k])
        hist[mode] += 1
        n += 1
        if sid in detail:
            lead = 'yes' if (first_bpm or 0) > 1000 else 'no'
            print(f'{sid:>5} {lead:>8} {n_ch:>8}  {counts[0]:>5} {counts[1]:>5} {counts[2]:>5} {counts[3]:>5}          {mode}  ({mode} beat{"s" if mode != 1 else ""} late)')
    print()
    print(f'corpus: {n} chord files with >= 4 changes; modal chord-change position on the 4-beat grid:')
    for k in range(4):
        print(f'  beat {k}: {hist.get(k, 0):>4}  ({100.0 * hist.get(k, 0) / max(n, 1):.1f}%)')
    print('  beat 0 = bars start on the tick grid; beat 1 = the file is one beat late; beat 3 = one beat early')
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='/home/xinyue.li/POP909-Dataset/POP909')
    p.add_argument('--aligned', default=None, help='POP909-aligned folder (optional check)')
    p.add_argument('--ids', nargs='*', default=None, help='songs to print in full (default: histogram only)')
    p.add_argument('--chord-dir', default=None, help='POP909-chord folder: measure the phase from chord changes')
    a = p.parse_args()

    if a.chord_dir:
        chord_phase(a.chord_dir, set(a.ids or []))

    dirs = sorted(d for d in glob(os.path.join(a.raw, '*')) if os.path.isdir(d))
    hist = Counter()
    d0s = Counter()
    n = 0
    detail = set(a.ids or [])
    print(f'{"song":>5} {"beats":>6} {"d0":>3} {"shift":>5}  first rows of beat_midi.txt')
    for d in dirs:
        sid = os.path.basename(d)
        bp = os.path.join(d, 'beat_midi.txt')
        if not os.path.exists(bp):
            continue
        rows = read_beats(bp)
        d0 = first_downbeat(rows)
        if d0 is None:
            hist['no-downbeat-flag'] += 1
            continue
        shift = (d0 + 1) % 4
        hist[shift] += 1
        d0s[d0] += 1
        n += 1
        if sid in detail:
            head = ' | '.join(' '.join(f'{v:g}' for v in r) for r in rows[:6])
            print(f'{sid:>5} {len(rows):>6} {d0:>3} {shift:>5}  {head}')
    print()
    print(f'corpus: {n} songs with a downbeat flag')
    print('first-downbeat index d0:', dict(sorted(d0s.items())))
    print('predicted bar shift under the current aligner, (d0+1) mod 4 beats:',
          {k: v for k, v in sorted(hist.items(), key=lambda kv: str(kv[0]))})
    on_grid = hist.get(0, 0)
    print(f'  on the grid (shift 0): {on_grid} / {n} = {100.0 * on_grid / max(n, 1):.1f}%')
    print(f'  one beat late (shift 1): {hist.get(1, 0)} / {n} = {100.0 * hist.get(1, 0) / max(n, 1):.1f}%')

    if a.aligned and detail:
        try:
            import mido
        except ImportError:
            print('[aligned] mido not available; skipping the file check')
            return
        print()
        print('aligned-file check (tick of beat d0 should equal (d0+1)*PPQ under the current aligner):')
        for sid in sorted(detail):
            mp = os.path.join(a.aligned, f'{sid}.mid')
            if not os.path.exists(mp):
                print(f'  {sid}: aligned file missing')
                continue
            mid = mido.MidiFile(mp)
            ppq = mid.ticks_per_beat
            tempos = []
            t = 0
            for msg in mid.tracks[0]:
                t += msg.time
                if msg.type == 'set_tempo':
                    tempos.append((t, msg.tempo))
            first_bpm = (6e7 / tempos[0][1] if tempos and tempos[0][1] > 0
                         else float('inf') if tempos else float('nan'))
            # first note tick over all tracks
            first_note = None
            for tr in mid.tracks:
                t = 0
                for msg in tr:
                    t += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        first_note = t if first_note is None else min(first_note, t)
                        break
            rows = read_beats(os.path.join(a.raw, sid, 'beat_midi.txt'))
            d0 = first_downbeat(rows)
            print(f'  {sid}: PPQ={ppq}  first tempo event {first_bpm:.0f} bpm  '
                  f'first note at tick {first_note} = beat {first_note / ppq:.2f}  '
                  f'd0={d0} -> downbeat at tick {(d0 + 1) * ppq} = beat {d0 + 1} '
                  f'({"on" if (d0 + 1) % 4 == 0 else "OFF"} the 4-beat grid)')


if __name__ == '__main__':
    main()
