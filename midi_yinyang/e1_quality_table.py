"""The general-quality sheet as a LaTeX table.

Every metric on that sheet is a corpus-pooled divergence from the
ground-truth continuation -- Frechet Music Distance and the four
groove/scale-consistency JSDs -- so the reference level is 0 for all of
them and a figure with a zero line on every panel says nothing a table
does not. One block per family (ours / internal baselines / external
baselines / not ranked), each opened by a bold header row and closed by
a rule; one row per system, one column per metric, the 95% bootstrap
interval over songs on a scriptsize row beneath each value (--ci
stacked, the default, which fits one column of a two-column template);
the best system per column in bold (least divergence), the unranked
block included unless --ranked-only. The reference
split-half null, what a perfect system scores at this sample size, is
the last row.

Reads the same files the figure does: <tag>_pooled.csv from
eval_metrics and, if present beside it, <tag>_fmd.csv from eval_fmd.

Usage (via plot_e1_box.sbatch, block quality):
  python e1_quality_table.py --pooled results/E1_p80_v5b93_pooled.csv \
      --out results/E1_p80_v5b93_quality.tex
"""

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_e1_box import GROUPS, read_pooled          # noqa: E402

METRICS = [
    ('fmd',     r'FMD'),
    ('js_gc_a', r'$\mathrm{JS}_{\mathrm{GC}}^{\mathrm{mel}}$'),
    ('js_gc_b', r'$\mathrm{JS}_{\mathrm{GC}}^{\mathrm{chd}}$'),
    ('js_sc_a', r'$\mathrm{JS}_{\mathrm{SC}}^{\mathrm{mel}}$'),
    ('js_sc_b', r'$\mathrm{JS}_{\mathrm{SC}}^{\mathrm{chd}}$'),
]

# display names for the paper, one line each (the figure's two-line
# labels joined)
NAMES = {s: d.replace('\n', ' ').replace('\u2192', r'$\to$')
         for _g, _r, members in GROUPS for s, d, _sh in members}


def fmt(v, digits):
    if v is None or math.isnan(v):
        return 'n/a'
    v = max(v, 0.0) if abs(v) < 10 ** -digits else v     # no "-0.000"
    return f'{v:.{digits}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pooled', required=True)
    ap.add_argument('--fmd', default=None,
                    help='FMD CSV; default: --pooled with _pooled.csv -> _fmd.csv')
    ap.add_argument('--out', required=True, help='.tex path')
    ap.add_argument('--exclude', default='',
                    help='comma-separated systems to leave out')
    ap.add_argument('--digits', type=int, default=3)
    ap.add_argument('--fmd-digits', type=int, default=1)
    ap.add_argument('--ci', choices=['stacked', 'inline', 'none'],
                    default='stacked',
                    help='where the 95%% interval goes: stacked = a '
                         'scriptsize row under each value (fits one '
                         'column), inline = beside it (needs the page '
                         'width), none = values only')
    ap.add_argument('--no-ci', action='store_true', help='same as --ci none')
    ap.add_argument('--label', default='tab:e1_quality')
    ap.add_argument('--ranked-only', action='store_true',
                    help='bold only among the ranked families, as on the '
                         'figure; default bolds the best of every system')
    args = ap.parse_args()

    pooled = read_pooled(args.pooled)
    fc = args.fmd or re.sub(r'_pooled\.csv$', '_fmd.csv', args.pooled)
    if fc != args.pooled and os.path.exists(fc):
        for k, v in read_pooled(fc).items():
            pooled[k] = v
    null = {}
    for k, v in pooled.items():
        if '_null_ref_split' in v:
            null[k] = v['_null_ref_split'][0]
    metrics = [(m, lab) for m, lab in METRICS if m in pooled]
    if not metrics:
        raise SystemExit('none of the quality metrics is in the pooled CSV')
    missing = [lab for m, lab in METRICS if m not in pooled]
    if missing:
        print(f'[warn] not in the CSVs, column dropped: {" ".join(missing)}',
              file=sys.stderr)

    excluded = {x.strip() for x in args.exclude.split(',') if x.strip()}
    blocks = []      # (family, ranked, [(sys, name)])
    for family, ranked, members in GROUPS:
        rows = [(s, NAMES[s]) for s, _d, _sh in members
                if s not in excluded and any(s in pooled[m] for m, _ in metrics)]
        if rows:
            blocks.append((family, ranked, rows))

    # best system per column: least divergence (0 is the reference).
    # Every system competes, the unranked block too (by request,
    # 2026-09-16); --ranked-only restores the figure's rule. Ties at the
    # printed precision are all bold -- picking one of three systems
    # that print 0.000 would be arbitrary.
    best = {}
    for m, _ in metrics:
        d = args.fmd_digits if m == 'fmd' else args.digits
        cands = [(round(pooled[m][s][0], d), s) for _f, ranked, rows in blocks
                 if ranked or not args.ranked_only
                 for s, _n in rows if s in pooled[m]
                 and not math.isnan(pooled[m][s][0])]
        if cands:
            lo = min(v for v, _s in cands)
            best[m] = {s for v, s in cands if v == lo}

    if args.no_ci:
        args.ci = 'none'

    def value(m, s):
        rec = pooled[m].get(s)
        if rec is None or math.isnan(rec[0]):
            return '--'
        d = args.fmd_digits if m == 'fmd' else args.digits
        txt = fmt(rec[0], d)
        if s in best.get(m, ()):
            txt = r'\textbf{' + txt + '}'
        if args.ci == 'inline' and not math.isnan(rec[1]):
            txt += r' {\scriptsize[' + fmt(rec[1], d) + ', ' + fmt(rec[2], d) + ']}'
        return txt

    def interval(m, s):
        rec = pooled[m].get(s)
        if rec is None or math.isnan(rec[0]) or math.isnan(rec[1]):
            return ''
        d = args.fmd_digits if m == 'fmd' else args.digits
        return r'{\scriptsize[' + fmt(rec[1], d) + ', ' + fmt(rec[2], d) + ']}'

    ncol = len(metrics) + 1
    L = []
    L.append(r'\begin{table}[t]')
    L.append(r'\centering')
    L.append(r'\caption{General quality: corpus-pooled divergence from the '
             r'ground-truth continuation ($\downarrow$, 0 is the '
             r'reference)'
             + (r'; brackets: 95\% bootstrap interval over songs'
                if args.ci != 'none' else '')
             + r'. Bold: best per column'
             + (' among the ranked systems' if args.ranked_only else '')
             + r'. The last row is the split-half null of the reference itself, '
             r'the value a perfect system scores at this sample size.}')
    L.append(r'\label{' + args.label + '}')
    L.append(r'\small')
    L.append(r'\setlength{\tabcolsep}{3.5pt}')
    L.append(r'\begin{tabular}{@{}l' + 'c' * len(metrics) + '@{}}')
    L.append(r'\toprule')
    L.append('System & ' + ' & '.join(lab for _m, lab in metrics) + r' \\')
    L.append(r'\midrule')
    for i, (family, ranked, rows) in enumerate(blocks):
        if i:
            L.append(r'\midrule')
        L.append(r'\multicolumn{' + str(ncol) + r'}{@{}l}{\textbf{'
                 + family + r'}} \\')
        for s, name in rows:
            L.append(name + ' & ' + ' & '.join(value(m, s) for m, _ in metrics)
                     + r' \\')
            if args.ci == 'stacked':
                L.append(' & ' + ' & '.join(interval(m, s) for m, _ in metrics)
                         + r' \\')
    if null:
        L.append(r'\midrule')
        L.append(r'\textit{reference split-half null} & '
                 + ' & '.join(fmt(null.get(m, float('nan')),
                                  args.fmd_digits if m == 'fmd' else args.digits)
                              for m, _ in metrics) + r' \\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    with open(args.out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'wrote {args.out}')
    print('\n'.join(L))
    n = {m: next(iter(pooled[m].values()))[3] for m, _ in metrics}
    print(f'\nsongs per column: {n}', file=sys.stderr)


if __name__ == '__main__':
    main()
