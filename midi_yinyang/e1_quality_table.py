"""The general-quality sheet as a LaTeX table.

Every metric on that sheet is a corpus-pooled divergence from the
ground-truth continuation -- Frechet Music Distance and the four
groove/scale-consistency JSDs -- so the reference level is 0 for all of
them and a figure with a zero line on every panel says nothing a table
does not. One block per family (ours / internal baselines / external
baselines / not ranked), each opened by a bold header row and closed by
a rule; one row per system, one column per metric, the 95% bootstrap
interval over songs beside each value (--span page, the default: a
table* across both columns of the template) or on a scriptsize row
beneath it (--span column). Rules are plain \hline by default so the
table needs no package beyond the template's (--rules booktabs for
toprule/midrule);
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

METRIC_SETS = {
    # the groove / scale consistency sheet (plus FMD when scored)
    'gcsc': [
        ('fmd',     r'FMD'),
        ('js_gc_a', r'$\mathrm{JS}_{\mathrm{GC}}^{\mathrm{mel}}$'),
        ('js_gc_b', r'$\mathrm{JS}_{\mathrm{GC}}^{\mathrm{chd}}$'),
        ('js_sc_a', r'$\mathrm{JS}_{\mathrm{SC}}^{\mathrm{mel}}$'),
        ('js_sc_b', r'$\mathrm{JS}_{\mathrm{SC}}^{\mathrm{chd}}$'),
    ],
    # the stream-grammar divergences (H3), corpus-pooled
    'h3': [
        ('harmonic_rhythm_jsd', r'Harm.\ rhythm'),
        ('onset_grid_jsd_a',    r'Onset grid, mel.'),
        ('onset_grid_jsd_b',    r'Onset grid, chd.'),
        ('duration_jsd_a',      r'Duration, mel.'),
        ('duration_jsd_b',      r'Duration, chd.'),
    ],
}
METRICS = METRIC_SETS['gcsc']

# display names for the paper, one line each (the figure's two-line
# labels joined)
NAMES = {s: d.replace('\n', ' ').replace('\u2192', r'$\to$')
          .replace('\u2212', '$-$')
         for _g, _r, members in GROUPS for s, d, _sh in members}
# (base, qualifier) per system: a variant row directly under its base
# model's row prints only the qualifier, indented and small
PARTS = {s: (d.split('\n', 1) + [''])[:2]
         for _g, _r, members in GROUPS for s, d, _sh in members}


def row_name(s, prev_base):
    base, qual = PARTS[s]
    # only a true ablation qualifier ("w/o ...") nests under its base
    # row; a bracketed variant such as "(finetuned)" keeps its full name
    if qual and base == prev_base and qual.startswith('w/o'):
        return r'\hspace{1em}{\footnotesize ' + qual + '}'
    return NAMES[s]


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
                    default=None,
                    help='where the 95%% interval goes: stacked = a '
                         'scriptsize row under each value (fits one '
                         'column), inline = beside it (needs the page '
                         'width), none = values only')
    ap.add_argument('--no-ci', action='store_true', help='same as --ci none')
    ap.add_argument('--label', default='tab:e1_quality')
    ap.add_argument('--span', choices=['column', 'page'], default='page',
                    help='page (default): table* across both columns, '
                         'intervals inline; column: single-column table')
    ap.add_argument('--rules', choices=['hline', 'booktabs'], default='hline',
                    help='hline (default) needs no package; booktabs '
                         'uses toprule/midrule/bottomrule')
    ap.add_argument('--ranked-only', action='store_true',
                    help='bold only among the ranked families, as on the '
                         'figure; default bolds the best of every system')
    ap.add_argument('--no-bold', action='store_true',
                    help='mark no best system at all (2026-09-23 default '
                         'in plot_e1_box.sbatch, BOLD_BEST=0)')
    ap.add_argument('--bold-exclude', default='',
                    help='internal system names left out of the best-per-'
                         'column contest (still printed), e.g. S1')
    ap.add_argument('--metrics', choices=sorted(METRIC_SETS), default='gcsc',
                    help='which pooled metrics form the columns: gcsc '
                         '(default) or h3, the stream-grammar divergences')
    ap.add_argument('--group-column', action='store_true',
                    help='put the family (Ours / Internal baselines / '
                         'External baselines) in a first column, printed '
                         'on the first row of its block, instead of a '
                         'header row per family: shorter table')
    ap.add_argument('--per-sample', action='store_true',
                    help='print the PER-SAMPLE pooled divergence, mean +- '
                         'std over the sample indices (ps_mean/ps_std, '
                         'eval_metrics 2026-09-23), instead of the '
                         'song-mixed point estimate with its bootstrap CI')
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
    METRICS = METRIC_SETS[args.metrics]
    if args.per_sample:
        # swap the per-sample mean / std into the (value, lo, hi) slots
        # the rest of the script reads; the interval then prints as
        # mean -+ std, and a CSV without the fields drops the cell
        for m in list(pooled):
            for s, rec in list(pooled[m].items()):
                if s == '_null_ref_split' or len(rec) < 10:
                    continue
                mu, sd = rec[7], rec[8]
                pooled[m][s] = (mu, mu - sd, mu + sd) + tuple(rec[3:])
    metrics = [(m, lab) for m, lab in METRICS if m in pooled]
    if not metrics:
        raise SystemExit('none of the quality metrics is in the pooled CSV')
    missing = [lab for m, lab in METRICS if m not in pooled]
    if missing:
        print(f'[warn] not in the CSVs, column dropped: {" ".join(missing)}',
              file=sys.stderr)

    excluded = {x for x in re.split(r'[,\s]+', args.exclude) if x}   # commas OR spaces: sbatch --export splits on commas
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
    bold_excl = {x for x in re.split(r'[,\s]+', args.bold_exclude) if x}
    for m, _ in metrics if not args.no_bold else []:
        d = args.fmd_digits if m == 'fmd' else args.digits
        cands = [(round(pooled[m][s][0], d), s) for _f, ranked, rows in blocks
                 if ranked or not args.ranked_only
                 for s, _n in rows if s in pooled[m] and s not in bold_excl
                 and not math.isnan(pooled[m][s][0])]
        if cands:
            lo = min(v for v, _s in cands)
            best[m] = {s for v, s in cands if v == lo}

    if args.ci is None:
        args.ci = 'inline' if args.span == 'page' else 'stacked'
    if args.no_ci:
        args.ci = 'none'
    top, mid, bot = ((r'\toprule', r'\midrule', r'\bottomrule')
                     if args.rules == 'booktabs'
                     else (r'\hline', r'\hline', r'\hline'))
    env = 'table*' if args.span == 'page' else 'table'

    def value(m, s):
        rec = pooled[m].get(s)
        if rec is None or math.isnan(rec[0]):
            return '--'
        d = args.fmd_digits if m == 'fmd' else args.digits
        txt = fmt(rec[0], d)
        if s in best.get(m, ()):
            txt = r'\textbf{' + txt + '}'
        if args.ci == 'inline' and not math.isnan(rec[1]):
            if args.per_sample:
                txt += r'$\pm$' + fmt(rec[2] - rec[0], d)
            else:
                txt += r' {\scriptsize[' + fmt(rec[1], d) + ', ' + fmt(rec[2], d) + ']}'
        return txt

    def interval(m, s):
        rec = pooled[m].get(s)
        if rec is None or math.isnan(rec[0]) or math.isnan(rec[1]):
            return ''
        d = args.fmd_digits if m == 'fmd' else args.digits
        return r'{\scriptsize[' + fmt(rec[1], d) + ', ' + fmt(rec[2], d) + ']}'

    ncol = len(metrics) + (2 if args.group_column else 1)
    L = []
    L.append(r'\begin{' + env + '}[t]')
    L.append(r'\centering')
    L.append(r'\caption{General quality: corpus-pooled divergence from the '
             r'ground-truth continuation ($\downarrow$, 0 is the '
             r'reference)'
             + ((r'; each sample index pooled over all songs against the '
                 r'pooled reference, mean$\pm$std over the samples')
                if args.per_sample else
                (r'; brackets: 95\% bootstrap interval over songs'
                 if args.ci != 'none' else ''))
             + ('' if args.no_bold else r'. Bold: best per column'
                + (' among the ranked systems' if args.ranked_only else '')
                + ((', ' + ', '.join(NAMES.get(x, x) for x in sorted(bold_excl))
                    + ' excluded') if bold_excl else ''))
             + r'. The last row is the split-half null of the reference itself, '
             r'the value a perfect system scores at this sample size.}')
    L.append(r'\label{' + args.label + '}')
    L.append(r'\small')
    if args.span == 'column':
        L.append(r'\setlength{\tabcolsep}{3.5pt}')
    lead = 'l' * (2 if args.group_column else 1)
    L.append(r'\begin{tabular}{' + lead + 'c' * len(metrics) + '}')
    L.append(top)
    L.append(('Group & ' if args.group_column else '') + 'System & '
             + ' & '.join(lab for _m, lab in metrics) + r' \\')
    L.append(mid)
    for i, (family, ranked, rows) in enumerate(blocks):
        if i:
            L.append(mid)
        if not args.group_column:
            L.append(r'\multicolumn{' + str(ncol) + r'}{l}{\textbf{'
                     + family + r'}} \\')
        prev_base = None
        for j, (s, _name) in enumerate(rows):
            lead_cell = ''
            if args.group_column:
                # the family on its block's first row only; no multirow
                # package needed, and the rule above the block does the rest
                lead_cell = (r'\textbf{' + family + '}' if j == 0 else '') + ' & '
            L.append(lead_cell + row_name(s, prev_base) + ' & '
                     + ' & '.join(value(m, s) for m, _ in metrics) + r' \\')
            prev_base = PARTS[s][0]
            if args.ci == 'stacked':
                L.append((' & ' if args.group_column else '') + ' & '
                         + ' & '.join(interval(m, s) for m, _ in metrics)
                         + r' \\')
    if null:
        L.append(mid)
        L.append((' & ' if args.group_column else '')
                 + r'\textit{reference split-half null} & '
                 + ' & '.join(fmt(null.get(m, float('nan')),
                                  args.fmd_digits if m == 'fmd' else args.digits)
                              for m, _ in metrics) + r' \\')
    L.append(bot)
    L.append(r'\end{tabular}')
    L.append(r'\end{' + env + '}')
    with open(args.out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'wrote {args.out}')
    print('\n'.join(L))
    n = {m: next(iter(pooled[m].values()))[3] for m, _ in metrics}
    print(f'\nsongs per column: {n}', file=sys.stderr)


if __name__ == '__main__':
    main()
