"""The general-quality sheet as a LaTeX table.

Every metric on that sheet is a corpus-pooled divergence from the
ground-truth continuation -- Frechet Music Distance and the four
groove/scale-consistency JSDs -- so the reference level is 0 for all of
them and a figure with a zero line on every panel says nothing a table
does not. One row per system, one column per metric, the value with
its 95% bootstrap interval over songs beneath it, and the best RANKED
system per column in bold (least divergence). The unranked system is
listed after a rule and never bolded, as on the figure. The reference
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
    ('js_gc_a', r'$\mathrm{JS}_{\mathrm{GC}}$ mel.'),
    ('js_gc_b', r'$\mathrm{JS}_{\mathrm{GC}}$ chd.'),
    ('js_sc_a', r'$\mathrm{JS}_{\mathrm{SC}}$ mel.'),
    ('js_sc_b', r'$\mathrm{JS}_{\mathrm{SC}}$ chd.'),
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
    ap.add_argument('--fmd-digits', type=int, default=2)
    ap.add_argument('--no-ci', action='store_true',
                    help='values only, no interval line under each')
    ap.add_argument('--label', default='tab:e1_quality')
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

    # best RANKED system per column: least divergence (0 is the reference)
    best = {}
    for m, _ in metrics:
        cands = [(pooled[m][s][0], s) for _f, ranked, rows in blocks if ranked
                 for s, _n in rows if s in pooled[m]
                 and not math.isnan(pooled[m][s][0])]
        if cands:
            best[m] = min(cands)[1]

    def cell(m, s):
        rec = pooled[m].get(s)
        if rec is None or math.isnan(rec[0]):
            return '--'
        v, lo, hi = rec[0], rec[1], rec[2]
        d = args.fmd_digits if m == 'fmd' else args.digits
        txt = fmt(v, d)
        if best.get(m) == s:
            txt = r'\textbf{' + txt + '}'
        if not args.no_ci and not math.isnan(lo):
            txt += r' {\scriptsize[' + fmt(lo, d) + ', ' + fmt(hi, d) + ']}'
        return txt

    L = []
    L.append(r'\begin{table}[t]')
    L.append(r'\centering')
    L.append(r'\caption{General quality: corpus-pooled divergence from the '
             r'ground-truth continuation, lower is better and 0 is the '
             r'reference. Brackets: 95\% bootstrap interval over songs. '
             r'Bold: best among the ranked systems per column. The last '
             r'row is the split-half null of the reference itself, the '
             r'value a perfect system scores at this sample size.}')
    L.append(r'\label{' + args.label + '}')
    L.append(r'\small')
    L.append(r'\begin{tabular}{l' + 'c' * len(metrics) + '}')
    L.append(r'\toprule')
    L.append('System & ' + ' & '.join(lab for _m, lab in metrics) + r' \\')
    L.append(r'\midrule')
    for i, (family, ranked, rows) in enumerate(blocks):
        if i and not ranked:
            L.append(r'\midrule')
        L.append(r'\multicolumn{' + str(len(metrics) + 1) + r'}{l}{\textit{'
                 + family + r'}} \\')
        for s, name in rows:
            L.append(name + ' & ' + ' & '.join(cell(m, s) for m, _ in metrics)
                     + r' \\')
    if null:
        L.append(r'\midrule')
        L.append(r'reference split-half null & '
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
