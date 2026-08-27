"""Paper figure: DISTRIBUTION of per-expert top-1 loads, shared router
vs per-modality gates, in both regimes -- the companion view to the
expert_load heatmaps. The heatmaps answer "is any cell dead?"; this
figure shows the shape of the whole population of 48 loads (12 layers x
4 experts) per model.

One panel per regime (teacher-forced / free-running frontier), two
Gaussian-KDE curves per panel with a rug of the raw 48 values. NOTE ON
THE MEAN: per-layer loads sum to 1 across the E=4 experts, so every
distribution has mean exactly 1/E = 0.25 by construction -- the mean
carries no information and only the dispersion and tails differ. The
legend therefore reports sigma and the min (the deadliness-relevant left
tail). Teacher-forced: the two routers are near-identical (sigma .062
vs .056, same min .15). Free-running: the shared router's distribution
is wider on both tails (sigma .085 vs .071, min .096 vs .145) --
per-modality gates stay closer to balanced on self-generated context,
matching the deadliness table.

Data: the same load tables as the heatmap figures (imported from those
scripts): teacher-forced jobs 182681/182680, free-running frontier jobs
182747/182746.

Run: python figures/plot_load_distribution.py   (matplotlib only)
Writes figures/load_distribution.pdf / .png
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_expert_load import SHARED as TF_SHARED, MG as TF_MG
from plot_expert_load_freerun import SHARED as FR_SHARED, MG as FR_MG

C_SHARED = '0.45'
C_MG = '#0072B2'
E = 4


def kde(vals, grid):
    """Gaussian KDE, Silverman's rule-of-thumb bandwidth."""
    v = np.asarray(vals, dtype=float)
    n = v.size
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    bw = 0.9 * min(v.std(ddof=1), iqr / 1.34) * n ** (-1 / 5)
    z = (grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(1) / (n * bw * np.sqrt(2 * np.pi))


def panel(ax, shared, mg, title):
    grid = np.linspace(0.0, 0.5, 400)
    ymax = 8.6
    ax.set_ylim(0, ymax)
    # 1/E marker first, and stopped short of the legend area
    ax.axvline(1 / E, ymax=0.70, color='black', linewidth=0.7,
               linestyle=(0, (3, 2)), zorder=1)
    ax.text(1 / E + 0.006, 0.15, r'$1/E$', fontsize=6.5, color='0.3',
            va='bottom')
    for vals, color, name, yrug in ((shared, C_SHARED, 'shared', 0.22),
                                    (mg, C_MG, 'per-mod. gates', 0.50)):
        v = np.array(vals, dtype=float).ravel()
        d = kde(v, grid)
        label = (f'{name} ' r'($\sigma$=' f'{v.std(ddof=1):.3f}, '
                 f'min={v.min():.2f})')
        ax.plot(grid, d, color=color, linewidth=1.2, label=label,
                zorder=3)
        ax.fill_between(grid, d, color=color, alpha=0.12, linewidth=0,
                        zorder=2)
        # rug of the raw 48 values, inside the axes above the baseline
        ax.plot(v, np.full_like(v, -yrug), '|', color=color,
                markersize=3.2, markeredgewidth=0.6, alpha=0.8,
                clip_on=False, zorder=3)
    ax.set_title(title, pad=4)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel('per-expert top-1 load', labelpad=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis='x', pad=12)
    ax.legend(frameon=False, loc='upper right', borderpad=0.1,
              handlelength=1.1, handletextpad=0.5, labelspacing=0.3)


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 6.5, 'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3), sharey=True)
    panel(axes[0], TF_SHARED, TF_MG, '(a) teacher-forced')
    panel(axes[1], FR_SHARED, FR_MG, '(b) free-running (frontier)')
    axes[0].set_ylabel('density')

    # the means are pinned at 1/E by normalisation; say so on the figure
    fig.text(0.5, -0.04,
             'per-layer loads sum to 1, so both means are exactly '
             r'$1/E=0.25$; the curves differ only in spread and tails',
             ha='center', fontsize=6.5, color='0.35',
             style='italic')

    fig.tight_layout(pad=0.4, w_pad=1.4)
    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'load_distribution.{ext}'),
                    dpi=300, bbox_inches='tight')
    # print the summary stats the legend shows, for the record
    for name, tf, fr in (('shared', TF_SHARED, FR_SHARED),
                         ('mg', TF_MG, FR_MG)):
        for reg, d in (('tf', tf), ('fr', fr)):
            v = np.array(d, dtype=float).ravel()
            print(f'{name:6s} {reg}: mean={v.mean():.4f} '
                  f'std={v.std(ddof=1):.4f} min={v.min():.3f} '
                  f'max={v.max():.3f}')
    print('wrote figures/load_distribution.{pdf,png}')


if __name__ == '__main__':
    main()
