"""Compact, one-column paper figure for the subjective evaluation: ONE
panel, vertical. The five rating axes plus the overall mean are rows;
the rating is the x axis; within a row one marker per system with its
within-subject 95% CI, and a star to the right of a system where the
Holm-corrected paired t-test against ours is significant. Colours are
the objective-evaluation family palette of plot_e1_box (ours gold,
internal baselines slate, external baselines maroon, not-ranked grey,
ground truth ink, open marker), with marker shapes telling systems of
one family apart.

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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from subjective_anova import AXES, SYSTEMS, load, matrix, holm  # noqa: E402
from plot_e1_box import FAMILY_COLOR, INK, GRID, SURFACE, FS_TICK, FS_LABEL, FS_LEGEND  # noqa: E402

# row order of the paper's figure (2026-09-24): the two consistency axes
# first, then structure, musicality, creativity, overall
ROW_AXES = ['consistency', 'fit', 'structure', 'musicality', 'creativity', 'overall']
ROWS = ['Prompt\nconsistency', 'Melody–chord\nconsistency', 'Structure', 'Musicality',
        'Creativity', 'Overall']
# system -> (display, family, marker); ground truth in ink with an open marker
SYS = {'Duet (alt commit)': ('Duet (ours)', 'Ours', 'o'),
       'S-finetune':        ('SS finetuned', 'Not ranked', 's'),
       'S-scratch':         ('SS scratch', 'Internal baselines', 'v'),
       'Whole-song':        ('Whole-Song Gen', 'External baselines', 'D'),
       'AMT':               ('Anticipatory MT', 'External baselines', '^'),
       'GT':                ('Ground truth', 'GT', 'o')}


def colour(s):
    fam = SYS[s][1]
    return INK if fam == 'GT' else FAMILY_COLOR[fam]


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
    ap.add_argument('--height', type=float, default=2.0, help='figure height in inches')
    ap.add_argument('--width', type=float, default=3.0, help='figure width in inches')
    args = ap.parse_args()
    rows = load(args.csv, args.exclude, args.drop_constant)
    Ys, n = matrix(rows, 'block')

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': FS_LABEL - 1, 'axes.labelsize': FS_LABEL - 1,
        'xtick.labelsize': FS_TICK - 1, 'ytick.labelsize': FS_TICK - 1, 'legend.fontsize': FS_LEGEND - 1,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.45,
    })
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    k = len(SYSTEMS)
    # rows top to bottom; systems stacked within a row, ours on top
    off = (np.arange(k) - (k - 1) / 2) * 0.11
    handles = {}
    for j, a in enumerate(ROW_AXES):
        Y = Ys[a]
        mean, ci = Y.mean(0), within_ci(Y)
        praw = np.array([stats.ttest_rel(Y[:, 0], Y[:, i]).pvalue for i in range(1, k)])
        sig = holm(praw) < 0.05
        ybase = len(ROWS) - 1 - j
        for i, s in enumerate(SYSTEMS):
            disp, fam, mk = SYS[s]
            col = colour(s)
            y = ybase - off[i]
            ax.errorbar(mean[i], y, xerr=ci[i], fmt='none', ecolor=col, elinewidth=0.55,
                        capsize=1.0, capthick=0.55, zorder=2)
            # significance is carried by the marker fill (2026-09-24): a
            # system whose ratings differ from ours (Holm-corrected paired
            # t, p < 0.05) is drawn filled, one that does not is drawn open;
            # ours is always filled. No star, so nothing sits on the plot.
            filled = (i == 0) or bool(sig[i - 1])
            h, = ax.plot(mean[i], y, marker=mk, ms=3.2, mfc=col if filled else SURFACE,
                         mec=col, mew=0.7, ls='none', zorder=3)
            if s not in handles:
                # legend handle: always the filled form of the marker
                hl, = ax.plot([], [], marker=mk, ms=3.2, mfc=col, mec=col, mew=0.7, ls='none')
                handles[s] = (hl, disp)
        if j < len(ROWS) - 1:
            ax.axhline(ybase - 0.5, color=GRID, lw=0.45, zorder=0)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS[::-1], color=INK)
    ax.set_ylim(-0.5, len(ROWS) - 0.5)
    ax.set_xlim(2.5, 4.75)
    ax.set_xticks([3, 4])
    ax.set_xlabel('mean rating', color=INK)
    ax.grid(axis='x', color=GRID, lw=0.45, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK)
    ax.tick_params(length=2.0, width=0.45, colors=INK)
    order = ['Duet (alt commit)', 'S-finetune', 'S-scratch', 'Whole-song', 'AMT', 'GT']
    # figure-level legend, centred on the full width above the axes so no
    # entry is clipped; the layout below leaves it the top 14% of the height
    fig.legend([handles[s][0] for s in order], [handles[s][1] for s in order], ncol=3,
               loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=FS_LEGEND - 1, handletextpad=0.3, columnspacing=0.7,
               borderaxespad=0.1, handlelength=1.2, labelcolor=INK)
    fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.86])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print('wrote', f'{args.out}.{ext}')


if __name__ == '__main__':
    main()
