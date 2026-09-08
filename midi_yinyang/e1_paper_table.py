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
  <out>_consistency.pdf   melody-chord consistency read REFERENCE-FREE:
              one panel per H2 statistic, systems on a shared vertical
              axis, mean +- s.e.m. over songs with the per-song values
              behind, and the reference pair as a dashed line with its
              own s.e.m. band -- one row of the comparison, not a
              target.

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
SYSTEM_ORDER = ['A3', 'A3f', 'A1', 'S1', 'S-scratch', 'P-mc', 'P-cm', 'WS']
SYSTEM_NAME = {
    'A3': r'Duet (ours)',
    'A3f': r'Duet, slot sees $t{-}1$',
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

    consistency_figure(per_song, systems, args.baseline, args.out + '_consistency')
    consistency_table(per_song, systems, args.baseline, args.out + '_consistency',
                      len(n_songs), args.alpha)


def consistency_table(per_song, systems, baseline, out, n_songs, alpha):
    """Same content as the figure, as a two-column-spanning table in the
    IEEE conference style (ruled, caption above): reference pair as the
    first row, then the systems, the five statistics as columns, mean +-
    s.e.m. over songs. Superscripts mark a two-sided paired Wilcoxon
    difference from the baseline on the raw per-song values (a
    comparison among systems, not to a target): a = p < alpha,
    b = p < alpha/5 (Bonferroni over the five columns)."""
    cols = [(m, t) for m, t, _ in _CONSISTENCY]
    ncol = len(cols) + 1
    bl = SYSTEM_NAME.get(baseline, baseline)
    L = []
    L.append(r'\begin{table*}[htbp]')
    L.append(r'\caption{Melody--chord consistency of the generated pair on ' + str(n_songs)
             + r' held-out songs (mean$\pm$s.e.m. over songs; 3 samples per song averaged first).}')
    L.append(r'\begin{center}')
    L.append(r'\begin{tabular}{|l|' + 'c|' * len(cols) + '}')
    L.append(r'\hline')
    L.append(r'\textbf{System} & ' + ' & '.join(r'\textbf{' + t + '}' for _, t in cols) + r' \\')
    L.append(r'\hline')
    ref_cells = []
    for m, _ in cols:
        sv = song_values(per_song, baseline, m)
        rv = [t for _, t in sv.values()]
        ref_cells.append(f'{np.mean(rv):.3f}$\\pm${np.std(rv) / math.sqrt(len(rv)):.3f}'
                         if rv else '--')
    L.append(r'\textit{Reference pair} & ' + ' & '.join(ref_cells) + r' \\')
    L.append(r'\hline')
    for s_ in systems:
        cells = []
        for m, _ in cols:
            sv = song_values(per_song, s_, m)
            vals = [v for v, _ in sv.values()]
            if not vals:
                cells.append('--')
                continue
            txt = f'{np.mean(vals):.3f}$\\pm${np.std(vals) / math.sqrt(len(vals)):.3f}'
            if s_ != baseline:
                bv = song_values(per_song, baseline, m)
                common = sorted(set(sv) & set(bv))
                diffs = [sv[k][0] - bv[k][0] for k in common]
                if len(diffs) >= 5:
                    _, pv, _ = wilcoxon_signed_rank(diffs, 'two-sided')
                    if pv < alpha / len(cols):
                        txt += r'$^{\mathrm{b}}$'
                    elif pv < alpha:
                        txt += r'$^{\mathrm{a}}$'
            cells.append(txt)
        name = SYSTEM_NAME.get(s_, s_)
        if s_ == baseline:
            name = r'\textbf{' + name + '}'
        L.append(name + ' & ' + ' & '.join(cells) + r' \\')
        L.append(r'\hline')
    L.append(r'\multicolumn{' + str(ncol) + r'}{l}{Chord-tone coverage: fraction of melody '
             r'onsets on a chord tone. Coupling: coverage on the true pairing minus coverage '
             r'with the chords shifted by two bars.} \\')
    L.append(r'\multicolumn{' + str(ncol) + r'}{l}{$^{\mathrm{a}}$/$^{\mathrm{b}}$: differs '
             r'from \emph{' + bl + r'} at $p<' + f'{alpha:g}' + r'$ / $p<'
             + f'{alpha / len(cols):.2g}' + r'$ (two-sided paired Wilcoxon over songs).}')
    L.append(r'\end{tabular}')
    L.append(r'\label{tab:consistency}')
    L.append(r'\end{center}')
    L.append(r'\end{table*}')
    with open(out + '.tex', 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'[table] wrote {out}.tex')


# marker/colour per system id: ours black filled; families share a shape
_STYLE = {
    'A3': dict(marker='o', color='black', mfc='black'),
    'A3f': dict(marker='o', color='black', mfc='0.6'),
    'A1': dict(marker='o', color='black', mfc='white'),
    'S1': dict(marker='s', color='0.35', mfc='0.35'),
    'S-scratch': dict(marker='s', color='0.35', mfc='white'),
    'P-mc': dict(marker='^', color='0.5', mfc='0.5'),
    'P-cm': dict(marker='v', color='0.5', mfc='white'),
    'WS': dict(marker='D', color='0.6', mfc='white'),
}
_CONSISTENCY = [
    ('chord_tone_cov', 'Chord-tone coverage', 'onsets on chord tones'),
    ('ctnctr', 'CTnCTR', 'incl. resolved non-chord tones'),
    ('pcs', 'PCS', 'consonance ($-1$ to $1$)'),
    ('mctd', 'MCTD', 'centroid distance ($\\downarrow$ closer)'),
    ('coupling', 'Coupling', 'cov. $-$ shifted cov.'),
]


def consistency_figure(per_song, systems, baseline, out):
    """Melody-chord consistency, reference-free: one panel per statistic,
    systems on a shared vertical axis, mean +- s.e.m. over songs with the
    per-song values behind, and the reference pair as a dashed line with
    its own s.e.m. band -- a row of the comparison, not a target."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[figure] matplotlib not available; skipped')
        return
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    order = list(systems)[::-1]          # first system drawn on top
    ypos = {s: i for i, s in enumerate(order)}
    n = len(_CONSISTENCY)
    fig, axes = plt.subplots(1, n, figsize=(7.1, 0.3 * len(order) + 0.85),
                             sharey=True)
    for ax, (metric, title, sub) in zip(axes, _CONSISTENCY):
        ref = None
        for s in order:
            sv = song_values(per_song, s, metric)
            if not sv:
                continue
            vals = [v for v, _ in sv.values()]
            if ref is None:
                ref = [t for _, t in sv.values()]
            y = ypos[s]
            ax.plot(vals, [y] * len(vals), '.', color='0.75', ms=2.5,
                    zorder=1, clip_on=False)
            m, e = float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))
            st = _STYLE.get(s, dict(marker='o', color='0.5', mfc='0.5'))
            ax.errorbar(m, y, xerr=e, fmt=st['marker'], color=st['color'],
                        mfc=st['mfc'], ms=4 if s != baseline else 4.8,
                        mew=0.8, elinewidth=0.8, capsize=1.6, zorder=3)
        if ref:
            rm, re_ = float(np.mean(ref)), float(np.std(ref) / math.sqrt(len(ref)))
            ax.axvspan(rm - re_, rm + re_, color='0.9', zorder=0, lw=0)
            ax.axvline(rm, color='0.3', lw=0.7, ls='--', zorder=2)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel(sub, fontsize=5.8, color='0.35', labelpad=2)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis='x', labelsize=6, length=2, pad=1)
        ax.tick_params(axis='y', length=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color='0.92', lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.margins(x=0.08)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([SYSTEM_NAME.get(s, s) for s in order], fontsize=7)
    fig.text(0.15, 0.015, 'dashed: reference pair (mean, s.e.m. band);  '
             'markers: mean $\\pm$ s.e.m. over songs;  dots: individual songs',
             fontsize=5.8, color='0.35', ha='left', va='bottom')
    fig.subplots_adjust(left=0.15, right=0.985, top=0.9, bottom=0.24, wspace=0.22)
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png', dpi=220)
    print(f'[figure] wrote {out}.pdf / .png')


if __name__ == '__main__':
    main()
