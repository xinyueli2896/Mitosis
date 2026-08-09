"""Certify that a polyphony budget (cp) does not clip a corpus.

Fairness protocol for cross-representation comparisons: cp is part of
the representation under test, so differing budgets are legitimate --
PROVIDED no side's budget drops real notes. This measures that directly:
per folder (or per summed group of folders), the distribution of
simultaneous onsets per 16th-note frame, the worst case, and the clip
rate at a given budget.

Two modes, covering both directions the asymmetry can point:

  * per-stream folders (duet side): each folder measured alone against
    its per-stream budget, e.g. melody and chord at cp4.
  * --pooled group (merged side): folders summed per song before
    measuring, against the pooled budget -- e.g. drums+nondrum against
    the merged baseline's cp16, which can clip even when each stream
    alone fits.

Frame step follows each file's own initial tempo (16th note), matching
preprocess_midi's grid.

Usage:
    # duet melchord side: each stream vs cp4
    python cp_capacity_check.py --budget 4 \
        input/pop909_split/melody input/pop909_split/chord

    # merged drumnondrum baseline: drums+nondrum pooled vs cp16
    python cp_capacity_check.py --budget 16 --pooled \
        input/rwc_test_prompts_split/drum input/rwc_test_prompts_split/nondrum

Exit status 1 if any measured group clips at the given budget, so it can
gate a pipeline.
"""

import argparse
import os
import sys
import warnings
from collections import Counter
from glob import glob

import pretty_midi


def frame_counts(path):
    """-> Counter{frame_index: n_onsets} for one midi, 16th-note grid."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pm = pretty_midi.PrettyMIDI(path)
    _, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    if bpm <= 0:
        bpm = 120.0
    step = 60.0 / bpm / 4.0
    counts = Counter()
    for inst in pm.instruments:
        for n in inst.notes:
            counts[int(round(n.start / step))] += 1
    return counts


def midis(folder):
    return sorted(glob(os.path.join(folder, '*.mid'))
                  + glob(os.path.join(folder, '*.MID')))


def report(label, per_song_counts, budget):
    hist = Counter()
    worst = 0
    worst_song = None
    clipped_frames = 0
    clipped_notes = 0
    total_frames = 0
    total_notes = 0
    for song, counts in per_song_counts:
        for c in counts.values():
            hist[c] += 1
            total_frames += 1
            total_notes += c
            if c > worst:
                worst, worst_song = c, song
            if c > budget:
                clipped_frames += 1
                clipped_notes += c - budget
    print(f'\n[{label}]  (budget cp{budget})')
    print(f'  non-empty frames : {total_frames}   notes: {total_notes}')
    print(f'  worst onsets/frame: {worst}'
          + (f'  ({worst_song})' if worst_song else ''))
    print(f'  frames > budget  : {clipped_frames} '
          f'({100.0 * clipped_frames / max(total_frames, 1):.3f}%)')
    print(f'  notes dropped    : {clipped_notes} '
          f'({100.0 * clipped_notes / max(total_notes, 1):.3f}%)')
    top = sorted(hist.items())[-6:]
    print('  tail of the onsets/frame histogram: '
          + ', '.join(f'{k}:{v}' for k, v in top))
    return clipped_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folders', nargs='+')
    p.add_argument('--budget', type=int, required=True,
                   help='polyphony slots per frame to certify against')
    p.add_argument('--pooled', action='store_true',
                   help='sum the folders per song (matched by filename '
                        'stem) before measuring -- the MERGED-stream view')
    args = p.parse_args()

    any_clip = 0
    if args.pooled:
        per_song = {}
        for folder in args.folders:
            for f in midis(folder):
                sid = os.path.splitext(os.path.basename(f))[0]
                c = frame_counts(f)
                if sid in per_song:
                    per_song[sid].update(c)
                else:
                    per_song[sid] = c
        label = 'POOLED: ' + ' + '.join(args.folders)
        any_clip += report(label, sorted(per_song.items()), args.budget)
    else:
        for folder in args.folders:
            songs = [(os.path.basename(f), frame_counts(f))
                     for f in midis(folder)]
            if not songs:
                print(f'\n[{folder}]  NO MIDIS FOUND')
                continue
            any_clip += report(folder, songs, args.budget)

    print()
    if any_clip:
        print('VERDICT: CLIPPING at this budget -- disclose the rate or '
              're-tokenize at a larger cp before comparing across '
              'representations.')
    else:
        print('VERDICT: no clipping -- the budget captures this corpus '
              'losslessly; the cp difference is a documented design '
              'property, not a fairness issue.')
    return 1 if any_clip else 0


if __name__ == '__main__':
    sys.exit(main())
