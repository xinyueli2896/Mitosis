"""One-column grouped BAR chart of the subjective evaluation from the
SAME statistics as the dot plot and the spider chart (subjective_anova):
per axis one bar per system, its height the mean rating over the
complete rater-song blocks, an error bar on the bar for the
within-subject 95% CI (Cousineau-Morey). Six groups: the five rating
axes and the overall mean, in the dot plot's row order. One colour per
system (the spider chart's palette); ground truth is an open bar.
Significance against ours is stated in the text, not on the plot.

Run: python figures/plot_subjective_bars.py --csv ratings.csv
         --out figures/subjective_bars [--exclude ...] [--drop-constant]
Writes <out>.pdf / .png
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subjective_anova import SYSTEMS, load, matrix  # noqa: E402
from plot_e1_box import INK, GRID, SURFACE, FS_TICK, FS_LABEL, FS_LEGEND  # noqa: E402
from plot_subjective_compact import SYS, colour, within_ci, ROW_AXES, ROWS  # noqa: E402
from plot_subjective_spider import SYSTEM_COLOR  # noqa: E402

# bars left to right within a group, and the legend order
ORDER = ['Duet (alt commit)', 'S-finetune', 'S-scratch', 'Whole-song', 'AMT', 'GT']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--drop-constant', action='store_true')
    ap.add_argument('--width', type=float, default=3.39, help='figure width in inches')
    ap.add_argument('--height', type=float, default=1.45, help='figure height in inches')
    ap.add_argument('--ymin', type=float, default=2.5, help='bars start here (say so in the caption)')
    ap.add_argument('--ymax', type=float, default=4.85)
    args = ap.parse_args()
    rows = load(args.csv, args.exclude, args.drop_constant)
    Ys, n = matrix(rows, 'block')

    K, k = len(ROW_AXES), len(ORDER)
    mean = np.zeros((k, K)); ci = np.zeros((k, K))
    for j, a in enumerate(ROW_AXES):
        Y = Ys[a]
        mu, c = Y.mean(0), within_ci(Y)
        for i, s in enumerate(ORDER):
            mean[i, j], ci[i, j] = mu[SYSTEMS.index(s)], c[SYSTEMS.index(s)]

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': FS_LABEL - 3, 'axes.labelsize': FS_LABEL - 3,
        'xtick.labelsize': FS_TICK - 3, 'ytick.labelsize': FS_TICK - 3,
        'legend.fontsize': FS_LEGEND - 3, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.3,
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
    ax.set_xticks(x)
    ax.set_xticklabels(ROWS, color=INK)
    ax.set_xlim(-0.55, K - 0.45)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_yticks([3, 3.5, 4, 4.5])
    ax.set_ylabel('mean rating', color=INK)
    ax.grid(axis='y', color=GRID, lw=0.3, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK)
    ax.tick_params(length=1.6, width=0.3, colors=INK)
    ax.tick_params(axis='x', length=0, pad=1.5)
    # figure-level legend on one row above the axes, as in the compact plot
    fig.legend(handles, [h.get_label() for h in handles], ncol=6,
               loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=FS_LEGEND - 3, handletextpad=0.3, columnspacing=0.7,
               borderaxespad=0.1, handlelength=1.1, handleheight=0.8, labelcolor=INK)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print('wrote', f'{args.out}.{ext}')


if __name__ == '__main__':
    main()
