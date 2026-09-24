"""One-column grouped BAR chart of the subjective evaluation from the
SAME statistics as the dot plot and the spider chart (subjective_anova):
per axis one bar per system, its height the mean rating over the
complete rater-song blocks, an error bar on the bar for the
within-subject 95% CI (Cousineau-Morey), and an asterisk above a bar
whose Holm-corrected paired t-test against ours is significant. Six
groups: the five rating axes and the overall mean, in the dot plot's
row order. One colour per system (the spider chart's palette); ground
truth is an open bar.

Run: python figures/plot_subjective_bars.py --csv ratings.csv
         --out figures/subjective_bars [--exclude ...] [--drop-constant]
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
from matplotlib.patches import Patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subjective_anova import SYSTEMS, load, matrix, holm  # noqa: E402
from plot_e1_box import INK, GRID, SURFACE, FS_TICK, FS_LABEL, FS_LEGEND  # noqa: E402
from plot_subjective_compact import SYS, colour, within_ci, ROW_AXES, ROWS  # noqa: E402
from plot_subjective_spider import SYSTEM_COLOR  # noqa: E402

# bars left to right within a group, and the legend order: ground truth
# first as the reference, then ours, then the baselines
ORDER = ['GT', 'Duet (alt commit)', 'S-finetune', 'S-scratch', 'Whole-song', 'AMT']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--drop-constant', action='store_true')
    ap.add_argument('--width', type=float, default=3.39, help='figure width in inches')
    ap.add_argument('--height', type=float, default=1.4, help='figure height in inches')
    ap.add_argument('--legend-cols', type=int, default=3)
    ap.add_argument('--ylabel', default='', help="e.g. 'mean rating'; empty (default) leaves the width to the metric names")
    ap.add_argument('--ymin', type=float, default=2.5, help='bars start here (say so in the caption)')
    ap.add_argument('--ymax', type=float, default=4.95)
    ap.add_argument('--fs', type=float, default=-1.0,
                    help='font size offset from the plot_e1_box sizes')
    args = ap.parse_args()
    rows = load(args.csv, args.exclude, args.drop_constant)
    Ys, n = matrix(rows, 'block')

    K, k = len(ROW_AXES), len(ORDER)
    mean = np.zeros((k, K)); ci = np.zeros((k, K)); sig = np.zeros((k, K), bool)
    ours = SYSTEMS.index('Duet (alt commit)')
    for j, a in enumerate(ROW_AXES):
        Y = Ys[a]
        mu, c = Y.mean(0), within_ci(Y)
        # Holm over the k-1 paired tests against ours, as in subjective_anova
        others = [i for i in range(len(SYSTEMS)) if i != ours]
        praw = np.array([stats.ttest_rel(Y[:, ours], Y[:, i]).pvalue for i in others])
        padj = dict(zip(others, holm(praw) < 0.05))
        for i, s in enumerate(ORDER):
            si = SYSTEMS.index(s)
            mean[i, j], ci[i, j] = mu[si], c[si]
            sig[i, j] = padj.get(si, False)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': FS_LABEL + args.fs, 'axes.labelsize': FS_LABEL + args.fs,
        # the six one-line metric names only fit a column at one point less
        'xtick.labelsize': FS_TICK + args.fs - 1, 'ytick.labelsize': FS_TICK + args.fs,
        'legend.fontsize': FS_LEGEND + args.fs, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.3,
    })
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    group_w = 0.8                        # share of the unit pitch given to the k bars
    bw = group_w / k
    x = np.arange(K)
    handles = []
    for i, s in enumerate(ORDER):
        disp, fam, _mk = SYS[s]
        col = SYSTEM_COLOR.get(s, colour(s))
        xi = x - group_w / 2 + bw * (i + 0.5)
        open_bar = fam == 'GT'           # ground truth: open bar, ink edge
        ax.bar(xi, mean[i] - args.ymin, bw, bottom=args.ymin,
               color=SURFACE if open_bar else col, edgecolor=col, linewidth=0.4,
               zorder=2)
        ax.errorbar(xi, mean[i], yerr=ci[i], fmt='none', ecolor=INK if open_bar else INK,
                    elinewidth=0.4, capsize=0.9, capthick=0.4, zorder=3)
        handles.append(Patch(facecolor=SURFACE if open_bar else col, edgecolor=col,
                             linewidth=0.5, label=disp))
        # asterisk above the error bar: differs from ours (Holm, p < 0.05)
        for xx, yy, sg in zip(xi, mean[i] + ci[i], sig[i]):
            if sg:
                ax.text(xx, yy + 0.02, '*', ha='center', va='bottom', color=INK,
                        fontsize=FS_LABEL + args.fs, zorder=4, clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels(ROWS, color=INK)
    # the long second name needs room: the first label is right-aligned to
    # its group's right edge instead of centred under it
    labs = ax.get_xticklabels()
    labs[0].set_ha('right'); labs[0].set_x(group_w / 2)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_yticks([3, 3.5, 4, 4.5])
    if args.ylabel:
        ax.set_ylabel(args.ylabel, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.3, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK)
    ax.tick_params(length=1.6, width=0.3, colors=INK)
    ax.tick_params(axis='x', length=0, pad=1.5)
    # figure-level legend on one row above the axes, as in the compact plot
    # matplotlib fills a legend column-first; permute so it READS row-first
    leg_rows = -(-len(handles) // args.legend_cols)
    hh = [handles[i] for i in sorted(range(len(handles)), key=lambda i: (i % args.legend_cols, i // args.legend_cols))]
    fig.legend(hh, [h.get_label() for h in hh], ncol=args.legend_cols,
               loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=FS_LEGEND + args.fs, handletextpad=0.3, columnspacing=0.6,
               borderaxespad=0.1, handlelength=1.1, handleheight=0.8, labelcolor=INK)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 1 - 0.11 * leg_rows])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print('wrote', f'{args.out}.{ext}')


if __name__ == '__main__':
    main()
