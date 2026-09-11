"""A/B two MIDI folders on BEAT ALIGNMENT, in tick space.

Answers three questions per song, for a reference folder A and a
candidate folder B holding the same song ids:

  1. Is each one ON THE GRID? Fraction of note onsets landing exactly on
     the 16th (and 8th, 4th) grid, measured in TICK space -- ticks/ppq
     gives beats regardless of what tempo map the file carries. This is
     the only domain the CP tokenizer reads, and it is why
     check_beat_alignment.py (which measures SECONDS through the tempo
     map) is the wrong instrument for beat-synced files.

  2. Are they the SAME MUSIC, SHIFTED? The onset patterns are
     cross-correlated over lags of +-16 sixteenths, and the best lag is
     reported with its match score. A folder that is systematically a
     beat early shows up as a consistent lag of -4 (or +4) across songs
     -- which no per-file grid check would ever reveal.

  3. Do MELODY and CHORD agree with each other? Optional chord folders
     are checked the same way, and the melody-vs-chord lag WITHIN each
     folder is reported: a folder whose two streams disagree is broken
     in a way that a folder-vs-folder comparison alone would miss.

Also reports tempo-event counts, since a single tempo means the file is
metrically exact while one-per-beat means it carries a performance
curve, and the two compare very differently by ear.

Usage (via compare_alignment.sbatch, CPU):
    python compare_alignment.py --a <ref mel dir> --b <cand mel dir> \\
        [--a-chord <dir> --b-chord <dir>] [--ids 004 136] [--out x.tsv]
"""
import argparse
import csv
import os
from glob import glob

import mido
import numpy as np

SUB = 4                     # sixteenths per beat
MAX_LAG = 16                # +-4 beats


def onsets_in_beats(path):
    """Note-on beats (tick/ppq) and the file's tempo-event count."""
    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat
    ons, n_tempo = [], 0
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == 'set_tempo':
                n_tempo += 1
            elif msg.type == 'note_on' and msg.velocity > 0:
                ons.append(t / ppq)
    return np.array(sorted(ons)), n_tempo, ppq


def grid_frac(beats, sub):
    if not len(beats):
        return float('nan')
    unit = 1.0 / sub
    d = np.abs(((beats + unit / 2) % unit) - unit / 2)
    return float(np.mean(d < 1e-6))


def onset_vector(beats, length=None):
    """Binary presence on the 16th grid, for cross-correlation."""
    if not len(beats):
        return np.zeros(1)
    idx = np.round(beats * SUB).astype(int)
    idx = idx[idx >= 0]
    n = (length or (idx.max() + 1))
    v = np.zeros(int(n))
    v[idx[idx < n]] = 1.0
    return v


def best_lag(a_beats, b_beats):
    """Lag in SIXTEENTHS that best aligns b onto a, and the match score
    (matched onsets / min(len)). Positive lag = b is LATE by that many."""
    if not len(a_beats) or not len(b_beats):
        return 0, float('nan'), float('nan')
    n = int(max(a_beats.max(), b_beats.max()) * SUB) + 1 + MAX_LAG
    va, vb = onset_vector(a_beats, n), onset_vector(b_beats, n)
    denom = max(min(va.sum(), vb.sum()), 1.0)
    scores = {}
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        shifted = np.roll(vb, -lag)
        if lag > 0:
            shifted[-lag:] = 0
        elif lag < 0:
            shifted[:-lag] = 0
        scores[lag] = float((va * shifted).sum() / denom)
    lag = max(scores, key=scores.get)
    return lag, scores[lag], scores[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True, help='reference melody folder')
    p.add_argument('--b', required=True, help='candidate melody folder')
    p.add_argument('--a-chord', default=None)
    p.add_argument('--b-chord', default=None)
    p.add_argument('--ids', nargs='*', default=None)
    p.add_argument('--out', default=None, help='per-song TSV')
    a = p.parse_args()

    def ids_of(d):
        return {os.path.splitext(os.path.basename(f))[0]
                for f in glob(os.path.join(d, '*.mid'))
                + glob(os.path.join(d, '*.MID'))}
    ids = sorted(ids_of(a.a) & ids_of(a.b))
    if a.ids:
        ids = [i for i in ids if i in set(a.ids)]
    if not ids:
        raise SystemExit(f'no shared song ids between {a.a} and {a.b}')
    print(f'{len(ids)} shared songs\n')

    rows = []
    for sid in ids:
        def load(d):
            for ext in ('.mid', '.MID'):
                f = os.path.join(d, sid + ext)
                if os.path.exists(f):
                    return onsets_in_beats(f)
            return np.zeros(0), 0, 0
        ab, a_tempo, a_ppq = load(a.a)
        bb, b_tempo, b_ppq = load(a.b)
        lag, sc, sc0 = best_lag(ab, bb)
        r = dict(id=sid, a_notes=len(ab), b_notes=len(bb),
                 a_tempo_events=a_tempo, b_tempo_events=b_tempo,
                 a_ppq=a_ppq, b_ppq=b_ppq,
                 a_on16=f'{grid_frac(ab, 4):.4f}',
                 b_on16=f'{grid_frac(bb, 4):.4f}',
                 a_on4=f'{grid_frac(ab, 1):.4f}',
                 b_on4=f'{grid_frac(bb, 1):.4f}',
                 best_lag_16ths=lag, best_lag_beats=f'{lag / SUB:+.2f}',
                 match_at_best=f'{sc:.4f}', match_at_zero=f'{sc0:.4f}')
        rows.append(r)

    # chord folders, same treatment, plus melody-vs-chord inside each
    if a.a_chord and a.b_chord:
        for r in rows:
            sid = r['id']

            def loadc(d):
                for ext in ('.mid', '.MID'):
                    f = os.path.join(d, sid + ext)
                    if os.path.exists(f):
                        return onsets_in_beats(f)[0]
                return np.zeros(0)
            ac, bc = loadc(a.a_chord), loadc(a.b_chord)
            lc, scc, _ = best_lag(ac, bc)
            r['chord_lag_16ths'] = lc
            r['chord_lag_beats'] = f'{lc / SUB:+.2f}'
            r['chord_match'] = f'{scc:.4f}'
            r['a_chord_on4'] = f'{grid_frac(ac, 1):.4f}'
            r['b_chord_on4'] = f'{grid_frac(bc, 1):.4f}'

    hdr = ['id', 'a_notes', 'b_notes', 'a_tempo_events', 'b_tempo_events',
           'a_on16', 'b_on16', 'best_lag_beats', 'match_at_best',
           'match_at_zero']
    print(' '.join(f'{h:>15}' for h in hdr))
    for r in rows[:40]:
        print(' '.join(f'{str(r[h]):>15}' for h in hdr))
    if len(rows) > 40:
        print(f'  ... {len(rows) - 40} more (see --out)')

    lags = np.array([r['best_lag_16ths'] for r in rows])
    m_best = np.array([float(r['match_at_best']) for r in rows])
    m_zero = np.array([float(r['match_at_zero']) for r in rows])
    a16 = np.array([float(r['a_on16']) for r in rows])
    b16 = np.array([float(r['b_on16']) for r in rows])
    print(f'\n--- summary over {len(rows)} songs ---')
    print(f'  A on the 16th grid: mean {np.nanmean(a16):.3f}   '
          f'B: mean {np.nanmean(b16):.3f}')
    print(f'  A tempo events: median '
          f'{int(np.median([r["a_tempo_events"] for r in rows]))}   '
          f'B: median {int(np.median([r["b_tempo_events"] for r in rows]))}')
    u, c = np.unique(lags, return_counts=True)
    print('  best lag (16ths) histogram: '
          f'{ {int(k): int(v) for k, v in zip(u, c)} }')
    print(f'  songs already aligned (best lag 0): {int((lags == 0).sum())}'
          f' / {len(rows)}')
    print(f'  match at best lag: mean {m_best.mean():.3f}; '
          f'at zero lag: mean {m_zero.mean():.3f}')
    if (lags != 0).any():
        print(f'  NOTE a non-zero lag means the two folders hold the same '
              f'music at different bar/beat offsets.')
    if a.a_chord and a.b_chord:
        cl = np.array([r['chord_lag_16ths'] for r in rows])
        u, c = np.unique(cl, return_counts=True)
        print('  CHORD best lag histogram: '
              f'{ {int(k): int(v) for k, v in zip(u, c)} }')

    if a.out:
        with open(a.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                               delimiter='\t')
            w.writeheader()
            w.writerows(rows)
        print(f'\nper-song TSV -> {a.out}')


if __name__ == '__main__':
    main()
