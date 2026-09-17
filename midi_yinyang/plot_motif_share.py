"""Motif development: how the continuation's interval n-grams relate to
the prompt, as one stacked bar per system.

Reads the per-sample metrics CSV (eval_metrics, block P2) and draws,
per stream, a horizontal stacked bar per system whose segments are the
mean shares of the continuation's interval n-grams that are, by the
strictest relation each has to the prompt,

    exact      the same pitches occur in the prompt line (a copy)
    transp     the same intervals occur (a transposition)
    inv        the inversion or retrograde occurs
    contour    only the up/down shape occurs
    novel      none of the above

so the five sum to 1. The ground truth gets the top bar, recovered from
share minus delta (the delta is the paired difference to the ground
truth, so the reference share is exact). "Repeated" is exact + transp,
"developed" is inv + contour; read every share against the ground
truth's bar, since contour matches are cheap at n = 3.

Usage (via plot_e1_box.sbatch, MOTIF_SHARE=1):
  python plot_motif_share.py --csv results/E1_p80_v5b92_metrics.csv \
      --out results/E1_p80_v5b92_motif_share --n 3
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_e1_box import (GROUPS, SURFACE, INK, INK_2, GRID,  # noqa: E402
                         FS_TICK, FS_LABEL, FS_TITLE, FS_LETTER,
                         grouped_legend, _split_label)

CLASSES = [('exact', 'exact copy'), ('transp', 'transposed'),
           ('inv', 'inverted / retrograde'), ('contour', 'contour only'),
           ('novel', 'novel')]
# dark to light: the more literal the relation, the darker the segment
CLASS_COLOR = {'exact': '#2b2b2b', 'transp': '#4a5a6a', 'inv': '#8593a1',
               'contour': '#c0c7ce', 'novel': '#ededed'}
STREAMS = [('a', 'Melody'), ('b', 'Chord')]


def read(path, n):
    """{(stream, system): {class: [per-song mean share]}}, plus the
    ground truth per stream recovered from share - delta."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gt = defaultdict(lambda: defaultdict(list))
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for st, _ in STREAMS:
                for k, _l in CLASSES:
                    key = f'motif{n}_share_{k}_{st}'
                    try:
                        v, d = float(row[key]), float(row[key + '_delta'])
                    except (KeyError, ValueError):
                        continue
                    if math.isnan(v):
                        continue
                    acc[(st, row['system'])][k][row['song']].append(v)
                    if not math.isnan(d):
                        gt[(st, row['song'])][k].append(v - d)
    means = {}
    for key, per_cls in acc.items():
        means[key] = {k: float(np.mean([np.mean(v) for v in songs.values()]))
                      for k, songs in per_cls.items()}
    gt_means = {}
    for st, _ in STREAMS:
        per_song = [g for (s_, _song), g in gt.items() if s_ == st]
        if per_song:
            gt_means[st] = {k: float(np.mean([np.mean(g[k]) for g in per_song
                                              if k in g]))
                            for k, _l in CLASSES}
    return means, gt_means


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True, help='path WITHOUT extension')
    p.add_argument('--n', type=int, default=3, help='n-gram length (3 or 5)')
    p.add_argument('--width', type=float, default=3.4)
    p.add_argument('--exclude', default='')
    args = p.parse_args()

    means, gt = read(args.csv, args.n)
    excluded = {x.strip() for x in args.exclude.split(',') if x.strip()}
    order = [(s, d, g) for g, _r, members in GROUPS
             for s, d, _sh in members if s not in excluded]
    present = {s for _st, s in means}
    rows = [(s, d) for s, d, _g in order if s in present]
    if not rows:
        raise SystemExit('no P2 share columns in the CSV: rescore with the '
                         'current eval_metrics')

    fig, axes = plt.subplots(len(STREAMS), 1, squeeze=False,
                             figsize=(args.width, 0.28 * (len(rows) + 1)
                                      * len(STREAMS) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    print(f'motif{args.n} shares (mean over songs; GT recovered from '
          f'share - delta)')
    for r, (st, sname) in enumerate(STREAMS):
        ax = axes[r][0]
        bars = [('GT', 'Ground truth', gt.get(st))] + \
               [(s, d, means.get((st, s))) for s, d in rows]
        n = len(bars)
        print(f'  {sname:7s} ' + ' '.join(f'{k:>8s}' for k, _ in CLASSES))
        for i, (s, disp, shares) in enumerate(bars):
            y = n - 1 - i
            if not shares:
                continue
            left = 0.0
            for k, _l in CLASSES:
                w = shares.get(k, 0.0)
                ax.barh(y, w, left=left, height=0.62, color=CLASS_COLOR[k],
                        edgecolor=SURFACE, lw=0.5, zorder=3)
                left += w
            print(f'    {s:9s} ' + ' '.join(f'{shares.get(k, 0):8.3f}'
                                             for k, _ in CLASSES))
            name, qual = _split_label(disp)
            ax.annotate(name, xy=(0, y), xycoords=('axes fraction', 'data'),
                        xytext=(-4, 1.5 if qual else 0),
                        textcoords='offset points', ha='right',
                        va='bottom' if qual else 'center',
                        fontsize=FS_TICK, color=INK,
                        fontweight='bold' if s == 'GT' else 'normal')
            if qual:
                ax.annotate(qual, xy=(0, y), xycoords=('axes fraction', 'data'),
                            xytext=(-4, -1.5), textcoords='offset points',
                            ha='right', va='top', fontsize=FS_TICK - 1,
                            color=INK_2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([])
        ax.set_title(sname, fontsize=FS_TITLE, color=INK, pad=4)
        if r == len(STREAMS) - 1:
            ax.set_xlabel(f'share of the continuation\'s interval {args.n}-grams',
                          fontsize=FS_LABEL, color=INK)
        ax.tick_params(axis='x', labelsize=FS_TICK, colors=INK, length=2.5,
                       width=0.6, color=INK)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(INK)
        ax.set_facecolor(SURFACE)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.annotate(chr(ord('a') + r), xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-2, 4), textcoords='offset points', ha='right',
                    va='bottom', fontsize=FS_LETTER, weight='bold', color=INK)

    handles = [Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOR[k],
                         edgecolor=SURFACE, lw=0.5) for k, _ in CLASSES]
    grouped_legend(fig, [('Relation to prompt', handles,
                          [l for _k, l in CLASSES])],
                   y=0.0, xs=[0.02], ncol=3)
    leg = 0.55 / fig.get_figheight()
    fig.tight_layout(rect=(0, leg, 1, 1), h_pad=1.6)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')


if __name__ == '__main__':
    main()
