"""How many samples per song does this evaluation actually need?

Per-song scores are means over N_SAMPLES generations, and the paired test
runs on per-song differences. Raising N_SAMPLES shrinks ONLY the sampling
component of those differences:

    Var(d_i) = sigma^2_between  +  (sigma^2_within_A + sigma^2_within_B) / N

sigma^2_between is real song-to-song variation and no amount of sampling
touches it. So if the second term is already small, more samples are pure
cost -- and on a test set whose song count is fixed, cost is the only
thing they buy.

This decomposes the two terms from an existing metrics CSV and reports
what Var(d) would be at other sample counts, so the choice is measured
rather than guessed.

Usage:
    python sample_size_check.py --csv results/e1_melchord_p64_metrics.csv \
        --system A.2 --baseline S1
"""

import argparse
import csv
import math
from collections import defaultdict


def load(csv_path):
    """-> per_sample[(system, mode, song)] = {metric: [v per sample]}"""
    acc = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row.get('system', '?'), row.get('mode', '-'),
                   row.get('song', '?'))
            for k, v in row.items():
                if k in ('system', 'mode', 'song', 'sample'):
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(fv):
                    acc[key][k].append(fv)
    return acc


def var(xs):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def pooled_within(acc, system, mode, metric):
    """Mean within-song variance across songs, and the sample count used."""
    vs, counts = [], []
    for (sy, md, _song), metrics in acc.items():
        if sy != system or md != mode or metric not in metrics:
            continue
        xs = metrics[metric]
        v = var(xs)
        if v is not None:
            vs.append(v)
            counts.append(len(xs))
    if not vs:
        return None, 0
    return sum(vs) / len(vs), (sum(counts) / len(counts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--system', required=True, help='e.g. A.2')
    p.add_argument('--baseline', required=True, help='e.g. S1')
    p.add_argument('--mode', default='co')
    p.add_argument('--targets', default='3,5,10,20',
                   help='sample counts to project')
    args = p.parse_args()

    acc = load(args.csv)
    metrics = sorted({m for (sy, md, _), ms in acc.items()
                      if md == args.mode for m in ms})
    targets = [int(x) for x in args.targets.split(',')]

    print(f'{args.system} vs {args.baseline}  (mode={args.mode})\n')
    print(f'{"metric":<26}{"N":>3}{"sd(d)":>9}{"%from":>7}   '
          + ''.join(f'sd@{t:<5}' for t in targets))
    print(f'{"":<26}{"":>3}{"":>9}{"sampl":>7}   ')
    print('-' * (45 + 8 * len(targets)))

    for metric in metrics:
        wa, na = pooled_within(acc, args.system, args.mode, metric)
        wb, nb = pooled_within(acc, args.baseline, args.mode, metric)
        if wa is None or wb is None:
            continue
        n_now = max(1.0, min(na, nb))

        # observed per-song differences at the current sample count
        songs = sorted({s for (sy, md, s), ms in acc.items()
                        if md == args.mode and metric in ms and sy == args.system}
                       & {s for (sy, md, s), ms in acc.items()
                          if md == args.mode and metric in ms and sy == args.baseline})
        diffs = []
        for s in songs:
            a = acc[(args.system, args.mode, s)][metric]
            b = acc[(args.baseline, args.mode, s)][metric]
            diffs.append(sum(a) / len(a) - sum(b) / len(b))
        vd = var(diffs)
        if vd is None or vd <= 0:
            continue

        sampling_now = (wa + wb) / n_now
        between = max(0.0, vd - sampling_now)
        frac = min(1.0, sampling_now / vd)

        cells = ''
        for t in targets:
            projected = between + (wa + wb) / t
            cells += f'{math.sqrt(projected):<8.4f}'
        print(f'{metric:<26}{len(songs):>3}{math.sqrt(vd):>9.4f}'
              f'{100 * frac:>6.0f}%   {cells}')

    print('\nsd(d) = sd of the per-song paired differences at the CURRENT sample')
    print('count. "%from sampl" = share of its VARIANCE that comes from')
    print('generation sampling; only that share shrinks with more samples.')
    print('sd@N columns project sd(d) at N samples per song.')
    print('\nRule of thumb: if "%from sampl" is small (<~25%), the differences')
    print('are dominated by real song-to-song variation and extra samples are')
    print('mostly wasted GPU time. If it is large, more samples genuinely')
    print('sharpen the test -- the paired test needs every song to agree in')
    print('sign, and sampling noise is what flips signs.')


if __name__ == '__main__':
    main()
