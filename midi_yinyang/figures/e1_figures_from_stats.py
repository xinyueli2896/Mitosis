"""Paper figures for E1 built from the SUMMARY statistics of the merged
tables (merge job 226545 for H1-H3, the matching score_coherence run
for the coherence columns; 8 held-out POP909 songs, 3 samples per song
averaged first). Use when the per-song CSVs are not at hand; the
per-song version with dots is e1_summary_figure.py.

H1-H3 rows are mean +- std over songs in the log; error bars here are
s.e.m. = std / sqrt(8). Coherence rows are mean +- s.e.m. in the log.

Writes:
  fig_e1_summary.{pdf,png}      3 x 3 small multiples: the three primary
                                endpoints, coupling, melody density, and
                                four coherence columns
  fig_e1_consistency.{pdf,png}  the five H2 harmonisation statistics
Usage:
  python figures/e1_figures_from_stats.py --out figures/
"""
import argparse
import math
import os

import numpy as np

N_SONGS = 8
SQ = math.sqrt(N_SONGS)

SYSTEMS = ['A3ctcaT', 'A1', 'S1', 'S-scratch', 'P-mc', 'P-cm', 'WS']
NAME = {
    'A3ctcaT': 'Duet (ours)', 'A1': 'Duet, causal', 'S1': 'Shared, fine-tuned',
    'S-scratch': 'Shared, scratch', 'P-mc': 'Cascade m$\\rightarrow$c',
    'P-cm': 'Cascade c$\\rightarrow$m', 'WS': 'Whole-song',
}
STYLE = {
    'A3ctcaT': dict(marker='o', color='black', mfc='black'),
    'A1': dict(marker='o', color='black', mfc='white'),
    'S1': dict(marker='s', color='0.35', mfc='0.35'),
    'S-scratch': dict(marker='s', color='0.35', mfc='white'),
    'P-mc': dict(marker='^', color='0.5', mfc='0.5'),
    'P-cm': dict(marker='v', color='0.5', mfc='white'),
    'WS': dict(marker='D', color='0.6', mfc='white'),
}

# metric -> {system: (mean, err)}; 'ref' -> (mean, err) of the reference pair
# H1-H3: (mean, std) from the log, converted to s.e.m. below.
H = {
    'harmonic_rhythm_jsd': {'A3ctcaT': (0.179, 0.129), 'A1': (0.151, 0.148), 'S1': (0.137, 0.092),
                            'S-scratch': (0.206, 0.235), 'P-mc': (0.161, 0.170), 'P-cm': (0.190, 0.140),
                            'WS': (0.205, 0.107), 'ref': (0.0, 0.0)},
    'chord_tone_cov': {'A3ctcaT': (0.609, 0.066), 'A1': (0.574, 0.059), 'S1': (0.631, 0.054),
                       'S-scratch': (0.576, 0.029), 'P-mc': (0.548, 0.047), 'P-cm': (0.602, 0.061),
                       'WS': (0.649, 0.044), 'ref': (0.595, 0.086)},
    'ctnctr': {'A3ctcaT': (0.868, 0.021), 'A1': (0.850, 0.019), 'S1': (0.879, 0.026),
               'S-scratch': (0.862, 0.029), 'P-mc': (0.843, 0.045), 'P-cm': (0.880, 0.029),
               'WS': (0.906, 0.016), 'ref': (0.859, 0.032)},
    'pcs': {'A3ctcaT': (0.495, 0.043), 'A1': (0.437, 0.052), 'S1': (0.545, 0.061),
            'S-scratch': (0.431, 0.036), 'P-mc': (0.414, 0.052), 'P-cm': (0.473, 0.070),
            'WS': (0.557, 0.056), 'ref': (0.548, 0.061)},
    'mctd': {'A3ctcaT': (1.364, 0.035), 'A1': (1.405, 0.051), 'S1': (1.328, 0.040),
             'S-scratch': (1.401, 0.028), 'P-mc': (1.416, 0.049), 'P-cm': (1.366, 0.054),
             'WS': (1.328, 0.041), 'ref': (1.325, 0.057)},
    'coupling': {'A3ctcaT': (0.133, 0.079), 'A1': (0.105, 0.035), 'S1': (0.147, 0.056),
                 'S-scratch': (0.096, 0.027), 'P-mc': (0.108, 0.042), 'P-cm': (0.102, 0.048),
                 'WS': (0.152, 0.075), 'ref': (0.088, 0.064)},
    'survival_min': {'A3ctcaT': (0.879, 0.065), 'A1': (0.887, 0.082), 'S1': (0.869, 0.107),
                     'S-scratch': (0.833, 0.160), 'P-mc': (0.912, 0.056), 'P-cm': (0.888, 0.042),
                     'WS': (0.885, 0.066), 'ref': (1.0, 0.0)},
    'density_ratio_a': {'A3ctcaT': (0.812, 0.273), 'A1': (0.883, 0.306), 'S1': (0.999, 0.379),
                        'S-scratch': (0.979, 0.313), 'P-mc': (1.066, 0.433), 'P-cm': (1.018, 0.384),
                        'WS': (0.875, 0.445), 'ref': (1.0, 0.0)},
}
for m in H.values():
    for k, (mu, sd) in m.items():
        m[k] = (mu, sd / SQ)

# coherence: (mean, s.e.m.) from the score_coherence table
C = {
    'copy_bar_b': {'ref': (0.39, 0.04), 'A3ctcaT': (0.41, 0.05), 'A1': (0.25, 0.04), 'S1': (0.30, 0.04),
                   'S-scratch': (0.23, 0.04), 'P-mc': (0.27, 0.04), 'P-cm': (0.25, 0.04), 'WS': (0.34, 0.03)},
    'cpi': {'ref': (0.48, 0.05), 'A3ctcaT': (0.45, 0.05), 'A1': (0.57, 0.03), 'S1': (0.55, 0.04),
            'S-scratch': (0.65, 0.04), 'P-mc': (0.59, 0.04), 'P-cm': (0.57, 0.06), 'WS': (0.53, 0.03)},
    'pce4': {'ref': (2.67, 0.04), 'A3ctcaT': (2.65, 0.03), 'A1': (2.65, 0.04), 'S1': (2.63, 0.03),
             'S-scratch': (2.67, 0.04), 'P-mc': (2.70, 0.03), 'P-cm': (2.59, 0.05), 'WS': (2.64, 0.02)},
    'drift_slope': {'ref': (-0.06, 0.05), 'A3ctcaT': (-0.14, 0.06), 'A1': (-0.16, 0.04), 'S1': (-0.12, 0.03),
                    'S-scratch': (-0.09, 0.02), 'P-mc': (-0.07, 0.03), 'P-cm': (-0.07, 0.01), 'WS': (-0.04, 0.02)},
    'copy_bar_a': {'ref': (0.16, 0.05), 'A3ctcaT': (0.14, 0.05), 'A1': (0.11, 0.02), 'S1': (0.13, 0.03),
                   'S-scratch': (0.05, 0.01), 'P-mc': (0.13, 0.03), 'P-cm': (0.14, 0.03), 'WS': (0.01, 0.00)},
    'gs': {'ref': (0.71, 0.03), 'A3ctcaT': (0.78, 0.02), 'A1': (0.76, 0.01), 'S1': (0.71, 0.02),
           'S-scratch': (0.75, 0.02), 'P-mc': (0.71, 0.02), 'P-cm': (0.72, 0.02), 'WS': (0.71, 0.01)},
}

SUMMARY = [
    (H, 'harmonic_rhythm_jsd', 'Harmonic rhythm', 'H3: JSD to reference ($\\downarrow$)'),
    (H, 'chord_tone_cov', 'Chord-tone coverage', 'H2: melody onsets on chord tones'),
    (H, 'coupling', 'Coupling', 'H2: coverage $-$ shifted coverage'),
    (H, 'survival_min', 'Stream survival', 'H1: min active-bar fraction ($\\uparrow$)'),
    (H, 'density_ratio_a', 'Melody density', 'H1: notes per bar over reference'),
    (C, 'copy_bar_b', 'Chord copy-bars', 'exact repeats of an earlier bar'),
    (C, 'cpi', 'Chord progression irregularity', 'unique chord trigrams'),
    (C, 'pce4', 'Pitch-class entropy', 'over 4 bars'),
    (C, 'drift_slope', 'Tonal drift', 'similarity to prompt, per 16 bars'),
]
CONSISTENCY = [
    (H, 'chord_tone_cov', 'Chord-tone coverage', 'onsets on chord tones'),
    (H, 'ctnctr', 'CTnCTR', 'incl. resolved non-chord tones'),
    (H, 'pcs', 'PCS', 'consonance ($-1$ to $1$)'),
    (H, 'mctd', 'MCTD', 'centroid distance ($\\downarrow$ closer)'),
    (H, 'coupling', 'Coupling', 'cov. $-$ shifted cov.'),
]


def panel(ax, table, metric, title, sub, order, ypos, matplotlib):
    d = table[metric]
    rm, re_ = d['ref']
    if re_ > 0:
        ax.axvspan(rm - re_, rm + re_, color='0.9', zorder=0, lw=0)
    ax.axvline(rm, color='0.3', lw=0.7, ls='--', zorder=2)
    for s in order:
        mu, e = d[s]
        st = STYLE[s]
        ax.errorbar(mu, ypos[s], xerr=e, fmt=st['marker'], color=st['color'], mfc=st['mfc'],
                    ms=5 if s == 'A3ctcaT' else 4.2, mew=0.8, elinewidth=0.8, capsize=1.8, zorder=3)
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel(sub, fontsize=5.8, color='0.35', labelpad=2)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.tick_params(axis='x', labelsize=6, length=2, pad=1)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='x', color='0.92', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.margins(x=0.12)


def render(panels, nrow, ncol, out, foot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5,
                         'ytick.major.width': 0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    order = SYSTEMS[::-1]
    ypos = {s: i for i, s in enumerate(order)}
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, nrow * (0.2 * len(order) + 0.72)), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (table, metric, title, sub) in zip(axes, panels):
        panel(ax, table, metric, title, sub, order, ypos, matplotlib)
    for ax in axes[len(panels):]:
        ax.axis('off')
    for r in range(nrow):
        axes[r * ncol].set_yticks(range(len(order)))
        axes[r * ncol].set_yticklabels([NAME[s] for s in order], fontsize=6.6)
    fig.text(0.17, 0.008, foot, fontsize=5.6, color='0.35', ha='left', va='bottom')
    fig.subplots_adjust(left=0.17, right=0.985, top=0.93 if nrow > 1 else 0.86,
                        bottom=0.09 if nrow > 1 else 0.3, wspace=0.2, hspace=0.62)
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png', dpi=220)
    print(f'[figure] wrote {out}.pdf / .png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='figures')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    foot = ('dashed line: reference pair (band: s.e.m.) or fixed target;  '
            'markers: mean $\\pm$ s.e.m. over 8 held-out songs')
    render(SUMMARY, 3, 3, os.path.join(a.out, 'fig_e1_summary'), foot)
    render(CONSISTENCY, 1, 5, os.path.join(a.out, 'fig_e1_consistency'), foot)


if __name__ == '__main__':
    main()
