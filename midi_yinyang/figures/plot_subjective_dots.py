"""Paper figure for the subjective evaluation: mean rating per system
with a within-subject 95% confidence interval, one small panel per
axis (five axes + the mean over axes). The companion to the
repeated-measures ANOVA of subjective_anova.py: the panels show what
the F-tests test, and a marker beside a system flags a Holm-corrected
post-hoc difference from ours (p < 0.05).

Intervals are Cousineau-Morey within-subject CIs: each (rater, song)
block's six ratings are centred on the block mean before the standard
error is taken, then scaled by sqrt(k / (k - 1)), so the bars show the
precision of the WITHIN-block contrasts the ANOVA uses, not the spread
between raters.

One ink for every system except ours (gold, filled) and ground truth
(open marker): the system is named on the axis, so colour has no job
here beyond pointing at the row that matters.

Run: python figures/plot_subjective_dots.py --csv ratings.csv --out figures/subjective_dots
     [--exclude ...] [--drop-constant]
Writes <out>.pdf / .png.
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
from subjective_anova import AXES, LABEL, SYSTEMS, load, matrix, holm  # noqa: E402

SHORT = {'Duet (alt commit)': 'Duet (ours)', 'S-finetune': 'Single-stream, ft.',
         'AMT': 'Anticipatory MT', 'Whole-song': 'Whole-Song-Gen',
         'S-scratch': 'Single-stream, scr.', 'GT': 'Ground truth'}
INK, INK_2, GRID = '#303030', '#737373', '#DADADA'
OURS = '#C9A227'          # the figure family's gold for ours
BAND = '#F6EFD6'


def within_ci(Y, alpha=0.05):
    n, k = Y.shape
    Yc = Y - Y.mean(1, keepdims=True) + Y.mean()
    se = Yc.std(0, ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))
    return stats.t.ppf(1 - alpha / 2, n - 1) * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True, help='path without extension')
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--drop-constant', action='store_true')
    args = ap.parse_args()
    rows = load(args.csv, args.exclude, args.drop_constant)
    Ys, n = matrix(rows, 'block')

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
    })
    # one ICASSP column (3.39 in): three rows of two panels
    fig, axes = plt.subplots(3, 2, figsize=(3.39, 4.6), sharex=True, sharey=True)
    order = list(range(len(SYSTEMS)))[::-1]              # ours at the top
    ypos = {s: i for i, s in zip(order, SYSTEMS)}
    for ax, a in zip(axes.ravel(), AXES + ['overall']):
        Y = Ys[a]
        mean, ci = Y.mean(0), within_ci(Y)
        praw = np.array([stats.ttest_rel(Y[:, 0], Y[:, j]).pvalue for j in range(1, len(SYSTEMS))])
        sig = holm(praw) < 0.05
        ax.axhspan(ypos[SYSTEMS[0]] - 0.45, ypos[SYSTEMS[0]] + 0.45, color=BAND, lw=0, zorder=0)
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        for j, s in enumerate(SYSTEMS):
            y = ypos[s]
            col = OURS if s == SYSTEMS[0] else INK
            ax.errorbar(mean[j], y, xerr=ci[j], fmt='none', ecolor=col, elinewidth=0.9,
                        capsize=1.6, capthick=0.9, zorder=2)
            if s == 'GT':
                ax.plot(mean[j], y, 'o', ms=4.2, mfc='white', mec=INK, mew=0.9, zorder=3)
            else:
                ax.plot(mean[j], y, 'o', ms=4.2, mfc=col, mec=col, zorder=3)
            if j > 0 and sig[j - 1]:
                ax.text(mean[j] + ci[j] + 0.06, y, '*', ha='left', va='center',
                        fontsize=8, color=INK, zorder=4)
        ax.set_title(LABEL[a], pad=3)
        ax.set_xlim(2.5, 4.75)
        ax.set_xticks([3, 4])
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([SHORT[s] for s in ypos], fontsize=7)
        ax.tick_params(length=2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for ax in axes[2]:
        ax.set_xlabel('mean rating', labelpad=2)
    fig.text(0.5, 0.005,
             f'within-subject 95% CI over {n} rater–song pairs\n'
             '* differs from ours (paired t, Holm-corrected, p < 0.05)',
             ha='center', va='bottom', fontsize=6.5, color=INK_2, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.7, w_pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300)
        print('wrote', f'{args.out}.{ext}')


if __name__ == '__main__':
    main()
