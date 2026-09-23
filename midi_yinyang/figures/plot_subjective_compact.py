"""Compact paper figure for the subjective evaluation: ONE panel,
the five rating axes plus the overall mean along x, one marker per
system per axis with its within-subject 95% CI, and a star under a
system where the Holm-corrected paired t-test against ours is
significant. Same data and statistics as plot_subjective_dots.py in a
third of the height.

Run: python figures/plot_subjective_compact.py --csv ratings.csv
         --out figures/subjective_compact [--exclude ...] [--drop-constant]
Writes <out>.pdf / .png
"""

import argparse
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subjective_anova import AXES, SYSTEMS, load, matrix, holm  # noqa: E402

XLAB = ['Prompt', 'Structure', 'Mel.\u2013chord', 'Musicality', 'Creativity', 'Overall']
SHORT = {'Duet (alt commit)': 'Duet (ours)', 'S-finetune': 'Single-stream, ft.',
         'AMT': 'Anticipatory MT', 'Whole-song': 'Whole-Song-Gen',
         'S-scratch': 'Single-stream, scr.', 'GT': 'Ground truth'}
# identity by colour AND marker (colour-blind safe: gold for ours, black for
# ground truth, a blue and three greys for the baselines)
STYLE = {'Duet (alt commit)': ('#C9A227', 'o', 'full'),
         'S-finetune':        ('#0072B2', 's', 'full'),
         'AMT':               ('#5A5A5A', '^', 'full'),
         'Whole-song':        ('#8C8C8C', 'D', 'full'),
         'S-scratch':         ('#B8B8B8', 'v', 'full'),
         'GT':                ('#303030', 'o', 'none')}
INK, INK_2, GRID = '#303030', '#737373', '#DADADA'


def within_ci(Y, alpha=0.05):
    n, k = Y.shape
    Yc = Y - Y.mean(1, keepdims=True) + Y.mean()
    se = Yc.std(0, ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))
    return stats.t.ppf(1 - alpha / 2, n - 1) * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--drop-constant', action='store_true')
    args = ap.parse_args()
    rows = load(args.csv, args.exclude, args.drop_constant)
    Ys, n = matrix(rows, 'block')

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 5.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.39, 2.05))
    k = len(SYSTEMS)
    off = (np.arange(k) - (k - 1) / 2) * 0.13
    for j, a in enumerate(AXES + ['overall']):
        Y = Ys[a]
        mean, ci = Y.mean(0), within_ci(Y)
        praw = np.array([stats.ttest_rel(Y[:, 0], Y[:, i]).pvalue for i in range(1, k)])
        sig = holm(praw) < 0.05
        for i, s in enumerate(SYSTEMS):
            col, mk, fill = STYLE[s]
            x = j + off[i]
            ax.errorbar(x, mean[i], yerr=ci[i], fmt='none', ecolor=col, elinewidth=0.7,
                        capsize=1.2, capthick=0.7, zorder=2)
            ax.plot(x, mean[i], marker=mk, ms=3.6, mfc=col if fill == 'full' else 'white',
                    mec=col, mew=0.8, ls='none', zorder=3, label=SHORT[s] if j == 0 else None)
            if i > 0 and sig[i - 1]:
                ax.text(x, mean[i] - ci[i] - 0.06, '*', ha='center', va='top', fontsize=7,
                        color=INK, zorder=4)
    ax.set_xticks(range(6))
    ax.set_xticklabels(XLAB)
    ax.set_ylim(2.35, 4.7)
    ax.set_yticks([3, 4])
    ax.set_ylabel('mean rating')
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    for j in range(5):
        ax.axvline(j + 0.5, color=GRID, lw=0.5, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False,
              handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300)
        print('wrote', f'{args.out}.{ext}')


if __name__ == '__main__':
    main()
