"""One-column spider chart of the subjective evaluation from the SAME
statistics as the dot plot (subjective_anova): per axis, the mean
rating of every system over the complete rater-song blocks, a shaded
ring for its within-subject 95% CI (Cousineau-Morey), and a filled
vertex where the Holm-corrected paired t-test against ours is
significant, open where it is not -- markers dropped 2026-09-24, colour
alone tells the systems apart. Six axes: the five rating axes and
the overall mean. Same palette and markers as plot_subjective_compact.

Run: python figures/plot_subjective_spider.py --csv ratings.csv
         --out figures/subjective_spider_ci [--exclude ...] [--drop-constant]
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
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subjective_anova import SYSTEMS, load, matrix, holm  # noqa: E402
from plot_e1_box import INK, GRID, SURFACE, FS_TICK, FS_LABEL, FS_LEGEND  # noqa: E402
from plot_subjective_compact import SYS, colour, within_ci, ROW_AXES  # noqa: E402

# one colour per system (2026-09-24): with no markers the two external
# systems could not share the family maroon. Ours, the two single-stream
# models and ground truth keep the family palette; Whole-Song Gen keeps
# the external maroon and AMT takes a green.
SYSTEM_COLOR = {'AMT': '#2e8b57', 'Whole-song': '#b3243f'}

AXIS_LABEL = {'consistency': 'Prompt\ncons.', 'fit': 'Mel.–chord\ncons.',
              'structure': 'Structure', 'musicality': 'Musicality',
              'creativity': 'Creativity', 'overall': 'Overall'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--drop-constant', action='store_true')
    ap.add_argument('--rmin', type=float, default=2.5, help='centre of the chart')
    ap.add_argument('--width', type=float, default=3.39)
    ap.add_argument('--alpha', type=float, default=0.18, help='band opacity')
    ap.add_argument('--interval', choices=['fill', 'bars', 'both'], default='fill',
                    help='how the CI is drawn: a translucent ring (fill), a radial '
                         'bar with caps on every spoke (bars), or both')
    ap.add_argument('--knee', type=float, default=4.5,
                    help='rating above which the radius is squeezed')
    ap.add_argument('--squash', type=float, default=0.1,
                    help='share of the radius given to [knee, 5]; the rest goes '
                         'to [rmin, knee], where the ratings actually sit. Tick '
                         'labels keep the original scale (say so in the caption)')
    args = ap.parse_args()
    rows = load(args.csv, args.exclude, args.drop_constant)
    Ys, n = matrix(rows, 'block')

    axes = ROW_AXES                      # the dot plot's row order
    K = len(axes)
    k = len(SYSTEMS)
    mean = np.zeros((k, K)); ci = np.zeros((k, K)); sig = np.zeros((k, K), bool)
    for j, a in enumerate(axes):
        Y = Ys[a]
        mean[:, j], ci[:, j] = Y.mean(0), within_ci(Y)
        praw = np.array([stats.ttest_rel(Y[:, 0], Y[:, i]).pvalue for i in range(1, k)])
        sig[1:, j] = holm(praw) < 0.05
    sig[0, :] = True                     # ours: always drawn filled

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': FS_LABEL - 2, 'mathtext.fontset': 'stix',
    })
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(args.width, args.width * 0.82))
    ax = fig.add_axes([0.16, 0.19, 0.68, 0.70], polar=True)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)

    def r(v):                            # rating -> radius, piecewise linear
        v = np.asarray(v, float)
        lo = (v - args.rmin) / (args.knee - args.rmin) * (1 - args.squash)
        hi = (1 - args.squash) + (v - args.knee) / (5.0 - args.knee) * args.squash
        return np.where(v <= args.knee, lo, hi)
    ax.set_ylim(0, 1.0)
    ticks = [t for t in (3, 3.5, 4, 4.5, 5) if t > args.rmin]
    ax.set_yticks(list(r(ticks)))
    ax.set_yticklabels([f'{t:g}' for t in ticks], fontsize=FS_TICK - 3, color=INK)
    ax.set_rlabel_position(180 / K)
    ax.set_xticks(ang)
    ax.set_xticklabels([AXIS_LABEL[a] for a in axes], fontsize=FS_LABEL - 2, color=INK)
    ax.tick_params(axis='x', pad=-1)
    ax.grid(color=GRID, lw=0.4)
    ax.spines['polar'].set_color(GRID); ax.spines['polar'].set_linewidth(0.4)

    order = ['S-scratch', 'Whole-song', 'AMT', 'S-finetune', 'GT', 'Duet (alt commit)']
    handles = {}
    for s in order:
        i = SYSTEMS.index(s)
        disp, fam, mk = SYS[s]
        col = SYSTEM_COLOR.get(s, colour(s))
        m = r(np.concatenate([mean[i], mean[i][:1]]))
        lo = r(np.concatenate([mean[i] - ci[i], (mean[i] - ci[i])[:1]]))
        hi = r(np.concatenate([mean[i] + ci[i], (mean[i] + ci[i])[:1]]))
        # the within-subject CI: a ring between the two polygons and/or a
        # radial bar with caps on every spoke, which stays readable where
        # rings overlap
        if args.interval in ('fill', 'both'):
            ax.fill(np.concatenate([ang_c, ang_c[::-1]]), np.concatenate([hi, lo[::-1]]),
                    color=col, alpha=args.alpha if args.interval == 'fill' else args.alpha * 0.45,
                    lw=0, zorder=2)
        if args.interval in ('bars', 'both'):
            cap = 0.045                      # cap half-width, radians (fixed)
            for j in range(K):
                ax.plot([ang[j], ang[j]], [lo[j], hi[j]], color=col, lw=0.9,
                        solid_capstyle='butt', zorder=3)
                for rr in (lo[j], hi[j]):
                    # a cap is a short arc at fixed radius and fixed angle
                    th = np.linspace(ang[j] - cap, ang[j] + cap, 5)
                    ax.plot(th, np.full_like(th, rr), color=col, lw=0.9, zorder=3)
        # colour alone tells the systems apart (2026-09-24, by request):
        # no vertex markers; significance is stated in the text
        ax.plot(ang_c, m, color=col, lw=0.9, zorder=3)
        handles[s] = Line2D([0], [0], color=col, lw=1.4, label=disp)
    leg_order = ['Duet (alt commit)', 'S-finetune', 'S-scratch', 'Whole-song', 'AMT', 'GT']
    fig.legend([handles[s] for s in leg_order], [handles[s].get_label() for s in leg_order],
               ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.0), frameon=False,
               fontsize=FS_LEGEND - 2.5, handletextpad=0.4, columnspacing=0.9,
               handlelength=1.4, labelcolor=INK)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print('wrote', f'{args.out}.{ext}')


if __name__ == '__main__':
    main()
