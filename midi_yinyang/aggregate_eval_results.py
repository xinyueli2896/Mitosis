"""Aggregate eval_metrics.py's per-sample CSV into per-mode comparison
tables with paired significance tests (EXPERIMENTS.md 4.4 / 5.3).

Pipeline position:
    inference outputs -> build_eval_manifest.py -> eval_metrics.py
    -> THIS SCRIPT -> the E1/E2/E3 result tables.

Statistics, per EXPERIMENTS.md:
  * samples are averaged per (system, song) FIRST, so n = songs, not
    songs x samples (the 3 samples per cell are not independent draws
    of a new prompt);
  * systems are compared to a baseline on the SAME songs (paired);
  * Wilcoxon signed-rank on the per-song paired differences;
  * metrics are printed in the pre-registered priority order
    H3 > H2 > H1, primary endpoints marked '*'.

With ~10 songs only large effects clear p<0.05; the plan treats these
as directional evidence with the listening test carrying the headline.
A tie-heavy or tiny-n comparison prints n and the test statistic so a
non-significant result is visibly under-powered rather than silently
reported as "no difference".

Usage:
    python aggregate_eval_results.py --csv results/metrics_melchord.csv \
        --task melchord --baseline S1 [--mode co] [--out-md table.md]
"""

import argparse
import csv
import math
from collections import defaultdict

from eval_metrics import H_GROUPS, PRIMARY


def _try_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def metric_target(metric):
    """Value a metric is trying to hit, or None if its direction is not
    unambiguous from eval_metrics' own legend ('deltas/JSD: closer to 0 is
    better; ratios: closer to 1 is better').

    Only metrics with a known target can be tested one-sided: the
    directional hypothesis is 'this system lands CLOSER to the target than
    the baseline', which is a test on |value - target|, not on the raw
    value. That distinction matters -- for a delta metric a raw one-sided
    'less' test would score -0.065 as an improvement over -0.013 when it
    is actually further from 0, i.e. worse.

    Anything not listed stays two-sided. Guessing a direction and being
    wrong silently inverts the conclusion, so absence of metadata is
    treated as absence of a hypothesis.
    """
    if 'jsd' in metric or metric.endswith('_delta'):
        return 0.0
    if 'ratio' in metric:
        return 1.0
    if metric.startswith('survival'):
        # fraction of active bars; the reference continuation is ~1.0
        return 1.0
    return None


def wilcoxon_signed_rank(diffs, alternative='two-sided'):
    """Wilcoxon signed-rank test. Returns (W, p, n_used).

    alternative:
      'two-sided' -- differences are non-zero (default)
      'less'      -- differences are negative (system scores BELOW baseline)
      'greater'   -- differences are positive (system scores ABOVE baseline)

    Exact enumeration for n<=12 (our regime: ~10 songs), normal
    approximation with continuity correction above that. scipy is not
    assumed present in the training env, so this is self-contained.
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return float('nan'), float('nan'), 0

    # rank |d| with average ranks for ties
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    W = min(w_plus, w_minus)

    if n <= 12:
        # exact: enumerate all sign assignments of the observed ranks
        from itertools import product
        total = 0
        hits = 0
        for signs in product((0, 1), repeat=n):
            s = sum(r for r, sign in zip(ranks, signs) if sign)
            total += 1
            if alternative == 'two-sided':
                if min(s, sum(ranks) - s) <= W:
                    hits += 1
            else:
                # One-sided: only arrangements at least as extreme in the
                # PREDICTED direction count. Halves the attainable p, which
                # is what makes n=5 usable at all (floor 1/2^5 = 0.031 vs
                # the two-sided 2/2^5 = 0.0625, the latter above 0.05 at
                # ANY effect size).
                if (s <= w_plus if alternative == 'less' else
                        sum(ranks) - s <= w_minus):
                    hits += 1
        p = hits / total
    else:
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        z = (abs(W - mean_w) - 0.5) / math.sqrt(var_w)
        p = 2.0 * 0.5 * math.erfc(z / math.sqrt(2.0))
        if alternative != 'two-sided':
            p *= 0.5
        p = min(1.0, p)
    return W, p, n


def load(csv_path):
    """-> per_song[(system, mode, song)][metric] = mean over samples"""
    acc = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row.get('system', '?'), row.get('mode', '-'),
                   row.get('song', '?'))
            for k, v in row.items():
                if k in ('system', 'mode', 'song', 'sample'):
                    continue
                fv = _try_float(v)
                if fv is not None:
                    acc[key][k].append(fv)
    per_song = {}
    for key, metrics in acc.items():
        per_song[key] = {k: sum(vs) / len(vs) for k, vs in metrics.items()}
    return per_song


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='eval_metrics.py --out file')
    p.add_argument('--task', required=True, choices=sorted(PRIMARY))
    p.add_argument('--baseline', required=True,
                   help='system id to compare others against (e.g. S1, S0)')
    p.add_argument('--mode', default=None,
                   help='restrict to one mode (default: all modes present)')
    p.add_argument('--out-md', default=None, help='also write a markdown table')
    p.add_argument('--one-sided', action='store_true',
                   help='test the PRE-REGISTERED direction (system lands '
                        'closer to the metric target than the baseline) '
                        'instead of mere difference. Only valid because '
                        'H1-H3 fix the direction in advance in '
                        'EXPERIMENTS.md -- picking the direction after '
                        'seeing the data would just be a two-sided test '
                        'with the alpha doubled. Halves attainable p, '
                        'which is the difference between reachable and '
                        'unreachable significance at n=5 (0.031 vs 0.062). '
                        'Metrics with no unambiguous target stay two-sided; '
                        'one-sided cells are marked p>.')
    p.add_argument('--complete-cases', action='store_true',
                   help='score every system on the SAME songs (per metric): '
                        'intersect the songs each system defines the metric '
                        'on. Removes the survivorship bias of grading a '
                        'system only on the songs it managed to handle, and '
                        'makes rows comparable. n becomes equal across '
                        'systems, but equal to the MINIMUM -- it cannot '
                        'raise a system that produced empty streams.')
    args = p.parse_args()

    per_song = load(args.csv)
    modes = sorted({m for (_, m, _) in per_song})
    if args.mode:
        modes = [m for m in modes if m == args.mode]
        if not modes:
            raise SystemExit(f'no rows with mode={args.mode!r} in {args.csv}')

    md_lines = []
    for mode in modes:
        systems = sorted({s for (s, m, _) in per_song if m == mode})
        if args.baseline not in systems:
            print(f'\n[{mode}] baseline {args.baseline!r} absent '
                  f'(systems: {systems}) -- printing means only')
        header = f'\n=============== mode={mode} (baseline={args.baseline}) ==============='
        print(header)
        md_lines.append(f'\n### mode={mode} (baseline={args.baseline})\n')

        cols = [s for s in systems]
        print('metric'.ljust(26) + ''.join(c.ljust(30) for c in cols))
        md_lines.append('| metric | ' + ' | '.join(cols) + ' |')
        md_lines.append('|' + '---|' * (len(cols) + 1))

        for h in ('H3', 'H2', 'H1'):
            print(f'--- {h} ---')
            md_lines.append(f'| **{h}** | ' + ' | '.join('' for _ in cols) + ' |')
            for metric in H_GROUPS[h]:
                # songs where the metric exists for every system present
                song_sets = []
                for s in systems:
                    song_sets.append({song for (sy, m, song), vals
                                      in per_song.items()
                                      if sy == s and m == mode and metric in vals})
                if not song_sets or not any(song_sets):
                    continue
                # Complete-case mode: score every system on the SAME songs.
                # By default each system is summarised over whatever songs
                # it happens to define the metric on, which is survivorship
                # bias -- a system that emits an empty stream has that song
                # silently removed rather than penalised, so it is graded
                # only on the songs it handled. It also makes rows
                # mutually incomparable, since each uses its own subset.
                # Intersecting equalises n across systems (at the minimum,
                # not at the maximum: songs no system can score are gone).
                common_all = set.intersection(*song_sets) if song_sets else set()
                if args.complete_cases:
                    dropped = {s: len(ss - common_all)
                               for s, ss in zip(systems, song_sets) if ss - common_all}
                    if dropped:
                        print(f'    [complete-cases] {metric}: keeping '
                              f'{len(common_all)} song(s); dropped ' +
                              ', '.join(f'{s}:{n}' for s, n in sorted(dropped.items())))
                star = '*' if metric in PRIMARY[args.task][h] else ' '
                line = (star + metric).ljust(26)
                md_cells = []
                base_songs = None
                if args.baseline in systems:
                    base_songs = (common_all if args.complete_cases else
                                  {song for (sy, m, song), vals in per_song.items()
                                   if sy == args.baseline and m == mode
                                   and metric in vals})
                for s in systems:
                    songs = (common_all if args.complete_cases else
                             {song for (sy, m, song), vals in per_song.items()
                              if sy == s and m == mode and metric in vals})
                    vals = [per_song[(s, mode, song)][metric]
                            for song in sorted(songs)]
                    if not vals:
                        line += '--'.ljust(30)
                        md_cells.append('--')
                        continue
                    mean = sum(vals) / len(vals)
                    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
                        if len(vals) > 1 else 0.0
                    cell = f'{mean:+.3f}±{sd:.3f}'
                    if (base_songs is not None and s != args.baseline):
                        common = sorted(songs & base_songs)
                        if len(common) >= 2:
                            target = metric_target(metric) if args.one_sided else None
                            if target is None:
                                diffs = [per_song[(s, mode, song)][metric]
                                         - per_song[(args.baseline, mode, song)][metric]
                                         for song in common]
                                alt = 'two-sided'
                            else:
                                # distance to target: smaller = better, so the
                                # pre-registered prediction 'this system beats
                                # the baseline' is diffs < 0.
                                diffs = [
                                    abs(per_song[(s, mode, song)][metric] - target)
                                    - abs(per_song[(args.baseline, mode, song)][metric]
                                          - target)
                                    for song in common]
                                alt = 'less'
                            _, pval, n_used = wilcoxon_signed_rank(diffs, alt)
                            if not math.isnan(pval):
                                tag = '' if alt == 'two-sided' else '>'
                                cell += f' p{tag}={pval:.3f}(n={n_used})'
                    line += cell.ljust(30)
                    md_cells.append(cell)
                print(line)
                md_lines.append(f'| {star}`{metric}` | ' + ' | '.join(md_cells) + ' |')

    print('\n* = pre-registered primary endpoint. Deltas/JSD: closer to 0 is '
          'better; ratios: closer to 1. p = two-sided Wilcoxon signed-rank on '
          'per-song paired differences vs the baseline (n = songs with both '
          'systems present, zero-differences dropped).')
    if args.out_md:
        import os
        os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
        with open(args.out_md, 'w') as f:
            f.write('\n'.join(md_lines) + '\n')
        print(f'wrote markdown table -> {args.out_md}')


if __name__ == '__main__':
    main()
