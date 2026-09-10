"""One-page summary figure of E1: the paper's duet against the baselines
across the three hypothesis blocks and the coherence table, as small
multiples in the style of e1_paper_table.consistency_figure.

Nine panels, 3 x 3 (from results/E1_<run>_metrics.csv and
results/E1_mixed_<tag>_coherence.csv):
  H3 harmonic rhythm JSD (primary; target 0)
  H2 chord-tone coverage (primary; reference band) and coupling (reference band)
  H1 survival, min over streams (primary; target 1) and melody density ratio (target 1)
  coherence: chord copy-bars, CPI, PCE4 and tonal drift, each with the reference band.

Every panel: systems on a shared vertical axis, per-song values as dots,
mean +- s.e.m. over songs as the marker, the reference as a dashed line
with its own s.e.m. band (or a fixed target line). Read among systems.

Usage (via e1_summary_figure.sbatch):
    python e1_summary_figure.py --metrics results/E1_mixed_matched_metrics.csv \\
        --coherence results/E1_mixed_matched_coherence.csv \\
        --systems A3ctcaT A1 S1 S-scratch P-mc P-cm WS --baseline A3ctcaT \\
        --out results/E1_summary_matched
"""
import argparse
import math

import numpy as np

from e1_paper_table import load, SYSTEM_NAME, _STYLE

# shorter row labels for the figure (the tables keep the full names)
FIG_NAME = {'A3ctcaT': 'Duet (ours)', 'A3fctcaT': 'Duet (ours), slot sees $t{-}1$',
            'A3fcaT': 'Duet (ours), cp8 retrain'}

# (source, metric, title, subtitle, reference): reference is 'ref' (the
# per-song <metric>_ref column) or a constant target.
PANELS = [
    ('metrics', 'harmonic_rhythm_jsd', 'Harmonic rhythm', 'H3, JSD to reference ($\\downarrow$)', 0.0),
    ('metrics', 'chord_tone_cov', 'Chord-tone coverage', 'H2, melody onsets on chord tones', 'ref'),
    ('metrics', 'coupling', 'Coupling', 'H2, cov. $-$ shifted cov.', 'ref'),
    ('metrics', 'survival_min', 'Survival (min)', 'H1, active-bar fraction ($\\uparrow$)', 1.0),
    ('metrics', 'density_ratio_a', 'Melody density', 'H1, notes/bar over reference', 1.0),
    ('coherence', 'copy_bar_b', 'Chord copy-bars', 'exact repeats of an earlier bar', 'ref'),
    ('coherence', 'cpi', 'CPI', 'unique chord trigrams', 'ref'),
    ('coherence', 'pce4', 'PCE$_4$', 'pitch-class entropy, 4 bars', 'ref'),
    ('coherence', 'drift_slope', 'Tonal drift', 'similarity to prompt, per 16 bars', 'ref'),
]


def values(per_song, system, metric, reference):
    """(list of per-song values, list of per-song reference values or None)."""
    vals, refs = [], []
    for sg, m in per_song.get(system, {}).items():
        v = m.get(metric)
        if v is None:
            continue
        if reference == 'ref':
            r = m.get(metric + '_ref')
            if r is None:
                continue
            refs.append(r)
        vals.append(v)
    return vals, (refs if reference == 'ref' else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', required=True)
    p.add_argument('--coherence', required=True)
    p.add_argument('--systems', nargs='+', required=True)
    p.add_argument('--baseline', default=None)
    p.add_argument('--out', required=True, help='path stem; writes .pdf and .png')
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

    src = {'metrics': load(args.metrics), 'coherence': load(args.coherence, mode=None)}
    for name, per in src.items():
        missing = [s for s in args.systems if s not in per]
        if missing:
            raise SystemExit(f'{name}: systems not in CSV: {missing} (have {sorted(per)})')
    baseline = args.baseline or args.systems[0]
    order = list(args.systems)[::-1]
    ypos = {s: i for i, s in enumerate(order)}

    ncol = 3
    nrow = 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, nrow * (0.22 * len(order) + 0.75)),
                             sharey=True)
    axes = axes.ravel()
    for ax, (source, metric, title, sub, reference) in zip(axes, PANELS):
        per = src[source]
        ref_all = []
        for s in order:
            vals, refs = values(per, s, metric, reference)
            if not vals:
                continue
            if refs and not ref_all:
                ref_all = refs
            y = ypos[s]
            ax.plot(vals, [y] * len(vals), '.', color='0.75', ms=2.5, zorder=1, clip_on=False)
            m, e = float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))
            st = _STYLE.get(s, dict(marker='o', color='0.5', mfc='0.5'))
            ax.errorbar(m, y, xerr=e, fmt=st['marker'], color=st['color'], mfc=st['mfc'],
                        ms=4.8 if s == baseline else 4, mew=0.8, elinewidth=0.8,
                        capsize=1.6, zorder=3)
        if reference == 'ref' and ref_all:
            rm, re_ = float(np.mean(ref_all)), float(np.std(ref_all) / math.sqrt(len(ref_all)))
            ax.axvspan(rm - re_, rm + re_, color='0.9', zorder=0, lw=0)
            ax.axvline(rm, color='0.3', lw=0.7, ls='--', zorder=2)
        elif reference != 'ref':
            ax.axvline(float(reference), color='0.3', lw=0.7, ls='--', zorder=2)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel(sub, fontsize=5.6, color='0.35', labelpad=2)
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
    for ax in axes[len(PANELS):]:
        ax.axis('off')
    for r in range(nrow):
        axes[r * ncol].set_yticks(range(len(order)))
        axes[r * ncol].set_yticklabels([FIG_NAME.get(s, SYSTEM_NAME.get(s, s)) for s in order],
                                       fontsize=6.5)
    fig.text(0.17, 0.008, 'dashed: reference pair (s.e.m. band) or fixed target;  '
             'markers: mean $\\pm$ s.e.m. over songs;  dots: individual songs',
             fontsize=5.6, color='0.35', ha='left', va='bottom')
    fig.subplots_adjust(left=0.17, right=0.985, top=0.95, bottom=0.08, wspace=0.2, hspace=0.62)
    fig.savefig(args.out + '.pdf')
    fig.savefig(args.out + '.png', dpi=220)
    print(f'[figure] wrote {args.out}.pdf / .png  ({len(args.systems)} systems)')


if __name__ == '__main__':
    main()
