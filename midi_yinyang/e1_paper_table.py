"""Paper-ready E1 table (LaTeX, booktabs) and figure from an eval_metrics CSV.

Reads results/E1_<run>_metrics.csv (one row per system x song x sample),
averages samples within each (system, song) first -- so n = songs, as
aggregate_eval_results does -- and writes:

  <out>.tex   a booktabs table, systems as columns, one block per
              hypothesis in the pre-registered order H3 > H2 > H1. H3
              rows are JSDs against the reference continuation (target
              0). H2 rows are the raw harmonisation statistics with the
              REFERENCE as its own column, the way Yeh et al. (2021)
              report them, since 'closer to the reference' is the
              reading and the reader should see the reference. H1 rows
              are survival (target 1) and density ratios (target 1).
              Each cell is mean +- std over songs; the cell closest to
              the target in its row is bold; a superscript marks a
              system that is significantly closer (+) or farther (-)
              from the target than the baseline on a one-sided paired
              Wilcoxon over per-song |value - target| differences.
  <out>.pdf   signed distance from the reference for the H2 statistics
              (generated minus reference, per song; mean +- s.e.m.),
              one marker per system, reference at zero. Shows which
              side of the reference each design lands on, which the
              table's magnitudes hide.

Song counts and p-values are computed here, not copied from the
aggregator's console output, so the artefacts regenerate from the CSV.

Usage (via e1_paper_table.sbatch):
    python e1_paper_table.py --csv results/E1_mixed_matched_metrics.csv \\
        --out results/E1_paper --baseline A3
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

from aggregate_eval_results import wilcoxon_signed_rank, metric_target

# display order and paper names; ids not listed are appended as-is
SYSTEM_ORDER = ['A3', 'A1', 'S1', 'S-scratch', 'P-mc', 'P-cm', 'WS']
SYSTEM_NAME = {
    'A3': r'Duet (ours)',
    'A1': r'Duet, causal',
    'S1': r'Shared, fine-tuned',
    'S-scratch': r'Shared, scratch',
    'P-mc': r'Cascade m$\to$c',
    'P-cm': r'Cascade c$\to$m',
    'WS': r'Whole-song',
}

# (block, metric, paper label, primary?)  -- raw H2 stats carry a ref column
ROWS = [
    ('H3 stream grammar (JSD to reference, $\\downarrow$)', [
        ('harmonic_rhythm_jsd', 'Harmonic rhythm', True),
        ('onset_grid_jsd_a', 'Melody onset grid', False),
        ('onset_grid_jsd_b', 'Chord onset grid', False),
        ('duration_jsd_a', 'Melody duration', False),
        ('duration_jsd_b', 'Chord duration', False),
    ]),
    ('H2 inter-stream fit (closer to reference is better)', [
        ('chord_tone_cov', 'Chord-tone cov.', True),
        ('ctnctr', 'CTnCTR', False),
        ('pcs', 'PCS', False),
        ('mctd', 'MCTD', False),
        ('coupling', 'Coupling (cov. $-$ shifted cov.)', False),
    ]),
    ('H1 stream integrity', [
        ('survival_min', 'Survival, min over streams ($\\uparrow$)', True),
        ('survival_b', 'Chord survival ($\\uparrow$)', False),
        ('density_ratio_a', 'Melody density ratio ($\\to 1$)', False),
        ('density_ratio_b', 'Chord density ratio ($\\to 1$)', False),
    ]),
]
H2_RAW = {'chord_tone_cov', 'ctnctr', 'pcs', 'mctd', 'coupling'}


def _f(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def load(csv_path, mode='co'):
    """per_song[system][song][metric] = mean over samples."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if mode and r.get('mode', mode) != mode:
                continue
            for k, v in r.items():
                if k in ('system', 'mode', 'song', 'sample', 'run'):
                    continue
                x = _f(v)
                if x is not None:
                    acc[r['system']][r['song']][k].append(x)
    out = {}
    for s, songs in acc.items():
        out[s] = {sg: {k: float(np.mean(v)) for k, v in m.items()}
                  for sg, m in songs.items()}
    return out


def target_of(metric):
    """Target value per song: a constant for JSD/survival/ratio, the
    reference statistic (metric_ref) for the raw H2 rows."""
    if metric in H2_RAW:
        return None          # per-song, from <metric>_ref
    t = metric_target(metric)
    if t is None and metric.startswith('survival'):
        t = 1.0
    return t


def song_values(per_song, system, metric):
    """{song: (value, target)} for songs where both exist."""
    out = {}
    for sg, m in per_song.get(system, {}).items():
        v = m.get(metric)
        if v is None:
            continue
        if metric in H2_RAW:
            t = m.get(metric + '_ref')
            if t is None:
                continue
        else:
            t = target_of(metric)
        out[sg] = (v, t)
    return out


def fmt_cell(vals, bold, mark):
    if not vals:
        return '--'
    m, s = np.mean(vals), np.std(vals)
    txt = f'{m:.3f}$\\pm${s:.3f}'
    if bold:
        txt = r'\textbf{' + txt + '}'
    if mark:
        txt += f'$^{{{mark}}}$'
    return txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True, help='path stem; writes .tex and .pdf')
    p.add_argument('--baseline', default='A3')
    p.add_argument('--mode', default='co')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--systems', nargs='*', default=None)
    args = p.parse_args()

    per_song = load(args.csv, args.mode)
    systems = args.systems or [s for s in SYSTEM_ORDER if s in per_song] + \
        sorted(s for s in per_song if s not in SYSTEM_ORDER)
    if args.baseline not in per_song:
        raise SystemExit(f'baseline {args.baseline} not in CSV ({sorted(per_song)})')

    n_songs = sorted({sg for s in systems for sg in per_song[s]})
    print(f'[table] {len(n_songs)} songs: {" ".join(n_songs)}; systems: {" ".join(systems)}')

    lines = []
    ncol = len(systems) + 2          # metric + reference + systems
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{l' + 'c' * (ncol - 1) + '}')
    lines.append(r'\toprule')
    head = ['Metric', 'Reference'] + [SYSTEM_NAME.get(s, s) for s in systems]
    lines.append(' & '.join(head) + r' \\')
    lines.append(r'\midrule')

    fig_rows = []   # (metric label, system, mean delta, sem)
    for block, rows in ROWS:
        lines.append(r'\multicolumn{' + str(ncol) + r'}{l}{\textit{' + block + r'}} \\')
        for metric, label, primary in rows:
            base_sv = song_values(per_song, args.baseline, metric)
            cells, dists = [], {}
            ref_vals = []
            for s in systems:
                sv = song_values(per_song, s, metric)
                vals = [v for v, _ in sv.values()]
                dists[s] = {sg: abs(v - t) for sg, (v, t) in sv.items()}
                cells.append((s, vals, sv))
                if metric in H2_RAW and not ref_vals:
                    ref_vals = [t for _, t in sv.values()]
            # closest to target on the mean distance
            mean_dist = {s: np.mean(list(d.values())) if d else np.inf
                         for s, d in dists.items()}
            best = min(mean_dist, key=mean_dist.get)
            row = [(r'$^*$' if primary else '') + label]
            if metric in H2_RAW:
                row.append(fmt_cell(ref_vals, False, ''))
            else:
                row.append(f'{target_of(metric):g}')
            for s, vals, sv in cells:
                mark = ''
                if s != args.baseline and dists[s] and dists[args.baseline]:
                    common = sorted(set(dists[s]) & set(dists[args.baseline]))
                    diffs = [dists[s][sg] - dists[args.baseline][sg] for sg in common]
                    if len(diffs) >= 5:
                        _, p_less, _ = wilcoxon_signed_rank(diffs, 'less')
                        _, p_more, _ = wilcoxon_signed_rank(diffs, 'greater')
                        if p_less < args.alpha:
                            mark = '+'
                        elif p_more < args.alpha:
                            mark = '-'
                row.append(fmt_cell(vals, s == best and bool(vals), mark))
                if metric in H2_RAW and sv:
                    deltas = [v - t for v, t in sv.values()]
                    fig_rows.append((label, s, float(np.mean(deltas)),
                                     float(np.std(deltas) / math.sqrt(len(deltas)))))
            lines.append(' & '.join(row) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(
        r'\caption{Co-generation on ' + str(len(n_songs)) + ' held-out POP909 songs that no '
        r'compared system trained on (mean$\pm$std over songs, 3 samples per song averaged '
        r'first). $^*$ pre-registered primary endpoint. Bold: closest to the reference in '
        r'the row. $^{+}$/$^{-}$: significantly closer to / farther from the reference than '
        r'\emph{' + SYSTEM_NAME.get(args.baseline, args.baseline) + r'} '
        r'(one-sided paired Wilcoxon on per-song distances, $p<' + f'{args.alpha:g}' + r'$).}')
    lines.append(r'\label{tab:e1}')
    lines.append(r'\end{table*}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out + '.tex', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'[table] wrote {args.out}.tex')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[figure] matplotlib not available; skipped')
        return
    labels = [lab for _, lab, _ in ROWS[1][1]]
    labels = [l.split(' (')[0] for l in labels]
    fig_rows = [(l.split(' (')[0], s, m, e) for l, s, m, e in fig_rows]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X']
    width = 0.8
    k = len(systems)
    for i, s in enumerate(systems):
        xs, ys, es = [], [], []
        for j, lab in enumerate(labels):
            hit = [(m, e) for l, ss, m, e in fig_rows if l == lab and ss == s]
            if not hit:
                continue
            xs.append(j - width / 2 + width * (i + 0.5) / k)
            ys.append(hit[0][0]); es.append(hit[0][1])
        ax.errorbar(xs, ys, yerr=es, fmt=markers[i % len(markers)], ms=4.5,
                    capsize=2, lw=0.9, label=SYSTEM_NAME.get(s, s),
                    color='black' if s == args.baseline else None,
                    zorder=3 if s == args.baseline else 2)
    ax.axhline(0, color='0.3', lw=0.8, ls='--', zorder=1)
    ax.text(len(labels) - 0.5, 0, ' reference', va='center', ha='left',
            fontsize=7, color='0.3')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('generated $-$ reference', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xlim(-0.6, len(labels) - 0.4 + 0.9)
    fig.legend(fontsize=7, ncol=4, frameon=False, loc='lower center',
               bbox_to_anchor=(0.5, -0.02))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(args.out + '.pdf')
    fig.savefig(args.out + '.png', dpi=200)
    print(f'[figure] wrote {args.out}.pdf / .png')


if __name__ == '__main__':
    main()
