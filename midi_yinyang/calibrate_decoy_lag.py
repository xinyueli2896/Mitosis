"""A7 calibration: harmonic agreement of (mel_t, chord_{t+lag}) vs lag.

The A.7 corruption replaces a query slot's frame with the same stream's
frame from t +- lag, using the commitment level k to pick the lag range:
larger k -> larger lag -> less harmonic agreement with the partner,
while every corrupted state stays a real, self-contained frame of the
song. This script measures the decay curve that schedule relies on, on
the POP909 TRAINING songs (test ids excluded), using the exact same
machinery as the eval harness (Stream / chord_tone_coverage /
_shift_stream from eval_metrics, so 'agreement' here means the same
thing harmonic_coupling means at eval time).

Outputs:
  * per-lag table: mean coverage, and normalized decoherence
    d(lag) = (cov(0) - cov(lag)) / (cov(0) - floor), where floor is the
    mean coverage under random large shifts (the k=K endpoint);
  * suggested bin edges: the lags where d crosses 0.25 / 0.50 / 0.75,
    i.e. k=1..3 ranges equally spaced in MEASURED decoherence;
  * a monotonicity verdict (the schedule only makes sense if d rises
    with lag up to the floor).

Usage (via calibrate_decoy_lag.sbatch):
    python calibrate_decoy_lag.py \
        --melody-dir <POP909-melody> --chord-dir <POP909-chord> \
        --exclude 001 002 003 004 005 --out results/decoy_lag_curve.csv
"""

import argparse
import csv
import math
import os

import numpy as np
import pretty_midi

from eval_metrics import (
    BEAT_DIV, FRAMES_PER_BAR, Stream, _file_tempo, _shift_stream,
    chord_tone_coverage,
)

# Lag grid in frames (16 per bar). Dense where the curve moves fastest.
LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256]
N_RANDOM = 8          # random draws for the floor estimate
MIN_FRAMES = 512      # skip songs shorter than this (32 bars)
RNG_SEED = 1234


def load_pair(mel_path, chord_path):
    """(mel Stream, chord Stream) over the common frame span."""
    streams = []
    n_frames = None
    for p in (mel_path, chord_path):
        pm = pretty_midi.PrettyMIDI(p)
        step = 60.0 / _file_tempo(pm) / BEAT_DIV
        notes = [n for inst in pm.instruments if not inst.is_drum
                 for n in inst.notes]
        if not notes:
            return None
        n = int(math.ceil(max(nt.end for nt in notes) / step)) + 1
        n_frames = n if n_frames is None else min(n_frames, n)
        streams.append((notes, step))
    if n_frames < MIN_FRAMES:
        return None
    return (Stream(streams[0][0], streams[0][1], n_frames),
            Stream(streams[1][0], streams[1][1], n_frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--melody-dir', required=True)
    ap.add_argument('--chord-dir', required=True)
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='song ids to skip (the held-out test split)')
    ap.add_argument('--max-songs', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    mel_files = {os.path.splitext(f)[0]: f
                 for f in os.listdir(args.melody_dir)
                 if f.lower().endswith('.mid')}
    chord_files = {os.path.splitext(f)[0]: f
                   for f in os.listdir(args.chord_dir)
                   if f.lower().endswith('.mid')}
    stems = sorted(set(mel_files) & set(chord_files))
    stems = [s for s in stems
             if not any(x in s for x in args.exclude)]
    if args.max_songs > 0:
        stems = stems[:args.max_songs]
    print(f'[songs] {len(stems)} train songs '
          f'(excluded: {args.exclude or "none"})')

    per_lag = {lag: [] for lag in LAGS}
    cov0_all, floor_all = [], []
    used = skipped = 0
    for stem in stems:
        pair = load_pair(os.path.join(args.melody_dir, mel_files[stem]),
                         os.path.join(args.chord_dir, chord_files[stem]))
        if pair is None:
            skipped += 1
            continue
        mel, chord = pair
        cov0 = chord_tone_coverage(mel, chord)
        if math.isnan(cov0):
            skipped += 1
            continue
        used += 1
        cov0_all.append(cov0)
        # Symmetric: average the +lag and -lag circular shifts.
        for lag in LAGS:
            vals = [chord_tone_coverage(mel, _shift_stream(chord, d))
                    for d in (lag, -lag)]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                per_lag[lag].append(float(np.mean(vals)))
        # Floor: random large shifts, away from both ends.
        lo, hi = 4 * FRAMES_PER_BAR, mel.n_frames - 4 * FRAMES_PER_BAR
        draws = rng.integers(lo, hi, size=N_RANDOM)
        fvals = [chord_tone_coverage(mel, _shift_stream(chord, int(d)))
                 for d in draws]
        fvals = [v for v in fvals if not math.isnan(v)]
        if fvals:
            floor_all.append(float(np.mean(fvals)))
        if used % 100 == 0:
            print(f'  ...{used} songs scored', flush=True)

    cov0 = float(np.mean(cov0_all))
    floor = float(np.mean(floor_all))
    span = cov0 - floor
    print(f'\n[curve] songs used={used} skipped={skipped}')
    print(f'[curve] cov(0) = {cov0:.4f}   floor(random lag) = {floor:.4f}'
          f'   span = {span:.4f}')
    if span <= 0.005:
        print('[VERDICT] NO usable span: agreement under random shifts is '
              'as high as the true pairing. The lag axis does not grade '
              'decoherence on this data -- do not train A7 on this '
              'schedule.')
        return

    rows = [(0, cov0, 0.0)]
    print(f'\n{"lag":>5} {"bars":>6} {"coverage":>9} {"decoherence":>12}')
    print(f'{0:>5} {0.0:>6.2f} {cov0:>9.4f} {0.0:>12.3f}')
    for lag in LAGS:
        c = float(np.mean(per_lag[lag]))
        d = (cov0 - c) / span
        rows.append((lag, c, d))
        print(f'{lag:>5} {lag / FRAMES_PER_BAR:>6.2f} {c:>9.4f} {d:>12.3f}')
    print(f'{"rand":>5} {"":>6} {floor:>9.4f} {1.0:>12.3f}')

    # Monotonicity: allow small noise, flag real inversions.
    ds = [d for _, _, d in rows]
    inversions = [(rows[i][0], rows[i + 1][0])
                  for i in range(len(ds) - 1) if ds[i + 1] < ds[i] - 0.05]

    # Bin edges: first lag whose decoherence crosses each threshold.
    def first_cross(th):
        for lag, _, d in rows:
            if d >= th:
                return lag
        return None

    e25, e50, e75 = (first_cross(t) for t in (0.25, 0.50, 0.75))
    print('\n[bins] equal spacing in measured decoherence:')
    print(f'  k=1 : lag in [1, {e25})          (decoherence < 0.25)')
    print(f'  k=2 : lag in [{e25}, {e50})      (0.25 - 0.50)')
    print(f'  k=3 : lag in [{e50}, {e75 if e75 else "rand"})'
          f'      (0.50 - 0.75)')
    print(f'  k=4 : random lag                 (the floor; plus the '
          f'residual mask draw)')
    if inversions:
        print(f'[VERDICT] WARNING: non-monotone at {inversions} -- '
              f'inspect the curve before fixing bins.')
    else:
        print('[VERDICT] decay is monotone; the lag axis grades '
              'decoherence as required. Bins above are the A7 defaults.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['lag_frames', 'coverage', 'decoherence'])
        for lag, c, d in rows:
            w.writerow([lag, f'{c:.5f}', f'{d:.4f}'])
        w.writerow(['random', f'{floor:.5f}', '1.0000'])
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
