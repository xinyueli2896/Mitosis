"""Box plots of E1 per-song scores, grouped by system family.

Reads the per-sample metrics CSV that eval_metrics writes and draws one
panel per metric: a notched box per system, systems ordered and coloured
by family (ours / internal baselines / external baselines).

WHAT THE MARKS MEAN, because a box plot of a distance metric is easy to
misread:

  box      quartiles of the PER-SONG values. Samples of a song are
           averaged first -- three samples of one song share a prompt
           and a reference, so treating them as three observations
           understates the spread by about sqrt(3).
  notch    95% confidence interval of the MEDIAN, 1.58*IQR/sqrt(n).
           Two systems whose notches do not overlap differ at roughly
           the 95% level; this is the "significance interval" a reader
           should compare across boxes.
  dashed   the REFERENCE LEVEL: the value a system scores when it
  line     matches the ground-truth continuation. 0 for a divergence or
           a delta, 1 for a ratio. It is a target, not a maximum --
           these metrics are two-sided, and a system can miss by
           exceeding it.

Nothing here is a "higher is better" axis, so no direction arrows: each
panel says in its own label which way the reference lies.

Usage (via plot_e1_box.sbatch):
  python plot_e1_box.py --csv results/E1_p96_v5b93_metrics.csv \
      --out results/E1_p96_v5b93_box --metrics harmonic_rhythm_jsd,...
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# System registry. Display names are the paper's, not the repo's.
# ---------------------------------------------------------------------------
# Each entry is (family, ranked, members). `ranked` says whether the
# family takes part in the best-model comparison drawn on each panel.
#
# A3 and A3ctcaT are the SAME checkpoint under two decodes -- A3 is the
# default refinement schedule (K=4, T=0.9, top-p 0.95), A3ctcaT is
# ctc_alt with the follower sampled at T=1 and no nucleus cut -- so the
# labels name the decode, not two models.
GROUPS = [
    ('Ours', True, [
        ('A1',        'Duet\nw/o query'),
        ('A3',        'Duet\n(refine)'),
        ('A3ctcaT',   'Duet\n(alt. commit)'),
    ]),
    ('Internal baselines', True, [
        ('S-scratch', 'Single-stream\n(scratch)'),
        ('P-mc',      'Cascade\n(mel→chd)'),
        ('P-cm',      'Cascade\n(chd→mel)'),
    ]),
    ('External baselines', True, [
        ('WSf',       'Whole-Song\nGen'),
        ('AMT',       'Anticipatory\nMusic Transf.'),
    ]),
    # Held out of the ranking and placed last, by request. It is still
    # plotted, and the best-of-the-rest line runs the full width, so a
    # reader can see exactly where it lands relative to the field --
    # which is the only reason to separate it rather than drop it.
    ('Not ranked', False, [
        ('S1',        'Single-stream\n(finetuned)'),
    ]),
]

# Validated categorical slots 1-3 (see the dataviz palette reference).
# All-pairs CVD dE 9.2, normal-vision 24.0 on the light surface. Colour
# carries FAMILY, not system: the family is the comparison the figure is
# making, and every box is directly labelled besides.
FAMILY_COLOR = {
    'Ours':                '#2a78d6',
    'Internal baselines':  '#eb6834',
    'External baselines':  '#1baf7a',
    # Not a fourth categorical hue: the unranked slot is deliberately
    # achromatic so it reads as set apart rather than as a fourth family
    # competing on the same footing. Slot 4 would also put yellow beside
    # orange, which fails the all-pairs floors.
    'Not ranked':          '#8a8a85',
}

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8a85'

DEFAULT_METRICS = [
    'harmonic_rhythm_jsd',
    'chord_tone_cov_delta',
    'coupling_delta',
    'reuse_vs_prompt_a_delta',
]

# Axis labels. Keyed by metric; anything unlisted falls back to the raw
# name, so an unfamiliar metric still plots.
LABELS = {
    'harmonic_rhythm_jsd':     'Harmonic rhythm JSD',
    'onset_grid_jsd_a':        'Onset grid JSD (melody)',
    'onset_grid_jsd_b':        'Onset grid JSD (chord)',
    'duration_jsd_a':          'Duration JSD (melody)',
    'duration_jsd_b':          'Duration JSD (chord)',
    'chord_tone_cov_delta':    'Chord-tone coverage $-$ ref.',
    'ctnctr_delta':            'CTnCTR $-$ ref.',
    'pcs_delta':               'PCS $-$ ref.',
    'mctd_delta':              'MCTD $-$ ref.',
    'coupling_delta':          'Melody-chord coupling $-$ ref.',
    'reuse_vs_prompt_a_delta': 'Prompt reuse, melody $-$ ref.',
    'reuse_vs_prompt_b_delta': 'Prompt reuse, chord $-$ ref.',
    'grid_jsd_vs_prompt_a_delta': 'Metric-position JSD vs prompt $-$ ref.',
    'ioi_jsd_vs_prompt_a_delta':  'IOI JSD vs prompt $-$ ref.',
    'mel_interval_jsd':        'Melodic interval JSD',
    'voicing_jsd':             'Chord voicing JSD',
    'ubr2_onset_b_delta':      'Unique beat ratio, chord $-$ ref.',
}


def ref_level(metric):
    """Where the ground truth sits on this metric's axis.

    A ratio is 1 when the continuation matches the reference; a
    divergence or a signed difference is 0. Getting this wrong would
    draw the target line in the wrong place, so it is derived from the
    name rather than configured.
    """
    if metric.startswith('density_vs_prompt') and not metric.endswith('_delta'):
        return 1.0
    if metric.startswith('density_ratio'):
        return 1.0
    return 0.0


def read_per_song(path, metrics):
    """{metric: {system: {song: value}}}, samples averaged within song."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            sysname, song = row.get('system'), row.get('song')
            for m in metrics:
                v = row.get(m, '')
                if v in ('', None):
                    continue
                try:
                    x = float(v)
                except ValueError:
                    continue
                if math.isnan(x):
                    continue
                acc[m][sysname][song].append(x)
    return {m: {s: {c: float(np.mean(v)) for c, v in songs.items()}
                for s, songs in bysys.items()}
            for m, bysys in acc.items()}


def best_ranked(metric, per_song, order, present):
    """(system, display, family, mean) closest to the reference level.

    Only RANKED families compete. "Best" is least absolute distance from
    the reference on this metric, which is what these two-sided measures
    mean -- overshooting the target is as wrong as falling short, so a
    plain min or max would crown the wrong system on half of them.
    """
    ref = ref_level(metric)
    cands = []
    for sysname, disp, family, ranked in order:
        if not ranked or sysname not in present:
            continue
        vals = list(per_song.get(metric, {}).get(sysname, {}).values())
        if len(vals) >= 3:
            m = float(np.mean(vals))
            cands.append((abs(m - ref), sysname, disp, family, m))
    if not cands:
        return None
    _, sysname, disp, family, m = min(cands)
    return sysname, disp, family, m


def draw_panel(ax, metric, per_song, order, present):
    ref = ref_level(metric)
    positions, data, colors, labels = [], [], [], []
    missing = []
    for i, (sysname, disp, family, _ranked) in enumerate(order):
        labels.append(disp)
        if sysname in present:
            vals = list(per_song.get(metric, {}).get(sysname, {}).values())
        else:
            vals = []
        if len(vals) >= 3:
            positions.append(i)
            data.append(vals)
            colors.append(FAMILY_COLOR[family])
        else:
            missing.append(i)

    # Reference line first, so boxes sit on top of it.
    ax.axhline(ref, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)

    # Best of the ranked systems, drawn at its MEAN and running the full
    # width so the unranked column can be read against it. Coloured by
    # the winner's family -- the line IS that system's value -- and
    # named in ink, because a line colour alone would not say which.
    best = best_ranked(metric, per_song, order, present)
    if best is not None:
        _bs, bdisp, bfamily, bmean = best
        ax.axhline(bmean, color=FAMILY_COLOR[bfamily], lw=1.4, zorder=2)

    if data:
        bp = ax.boxplot(
            data, positions=positions, widths=0.62,
            notch=True, bootstrap=None, showfliers=False,
            patch_artist=True, zorder=3,
            # The best-of line is a MEAN, so every box carries its own
            # mean marker: comparing a mean line against a median bar
            # would mislead on these skewed distributions, where the
            # mean sits above the median.
            showmeans=True,
            meanprops=dict(marker='D', markersize=3.0,
                           markerfacecolor=SURFACE, markeredgecolor=INK,
                           markeredgewidth=0.8),
            medianprops=dict(color=SURFACE, lw=1.6),
            whiskerprops=dict(color=INK_2, lw=1.0),
            capprops=dict(color=INK_2, lw=1.0),
        )
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
            patch.set_alpha(0.9)
            # 2px surface ring so adjacent boxes never touch
            patch.set_linewidth(1.0)

    # Headroom for the caption below, claimed before anything is placed
    # against the limits: the placeholders span the full height, so they
    # have to be drawn against the FINAL ylim or they stop short.
    lo, hi = ax.get_ylim()
    hi = hi + 0.17 * (hi - lo)
    ax.set_ylim(lo, hi)

    if best is not None:
        # Named in the corner rather than on the line. On the line it
        # collided with whichever box sat at that height -- and the
        # unranked column, the one a reader most wants to compare
        # against this line, is exactly where the label landed.
        ax.annotate(
            'best ranked: ' + bdisp.replace('\n', ' '),
            xy=(0.012, 0.965), xycoords='axes fraction',
            ha='left', va='top', fontsize=6.4, color=INK)

    # Placeholder slots for systems not yet run: an empty hatched frame
    # is honest about the gap in a way a missing tick is not -- a reader
    # scanning the axis would otherwise not know the column was meant to
    # be there.
    for i in missing:
        ax.add_patch(Rectangle((i - 0.31, lo), 0.62, hi - lo,
                               facecolor='none', edgecolor=MUTED,
                               hatch='///', lw=0.8, alpha=0.55, zorder=2))
        ax.text(i, lo + 0.5 * (hi - lo), 'pending', rotation=90,
                ha='center', va='center', fontsize=6.5, color=MUTED)

    ax.set_ylabel(LABELS.get(metric, metric), fontsize=7.5, color=INK)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.tick_params(axis='y', labelsize=6.5, colors=INK_2, length=2)
    ax.tick_params(axis='x', length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=MUTED, alpha=0.22, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='eval_metrics --out CSV')
    p.add_argument('--out', required=True, help='output path WITHOUT extension')
    p.add_argument('--metrics', default=','.join(DEFAULT_METRICS))
    p.add_argument('--width', type=float, default=7.0, help='inches')
    p.add_argument('--panel-height', type=float, default=1.55, help='inches')
    p.add_argument('--title', default='')
    args = p.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    per_song = read_per_song(args.csv, metrics)

    present = set()
    for m in metrics:
        present |= set(per_song.get(m, {}))
    order = [(s, d, g, ranked)
             for g, ranked, members in GROUPS for s, d in members]
    known = {s for s, _, _, _ in order}
    extra = sorted(present - known)
    if extra:
        print(f'[warn] in the CSV but not in the figure: {" ".join(extra)}',
              file=sys.stderr)
    absent = [s for s, _, _, _ in order if s not in present]
    if absent:
        print(f'[warn] drawn as pending placeholders: {" ".join(absent)}',
              file=sys.stderr)

    n = len(metrics)
    fig, axes = plt.subplots(
        n, 1, figsize=(args.width, args.panel_height * n + 1.5),
        sharex=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    labels = None
    for ax, m in zip(axes, metrics):
        labels = draw_panel(ax, m, per_song, order, present)

    bottom = axes[-1]
    bottom.set_xticks(range(len(order)))
    bottom.set_xticklabels(labels, fontsize=6.5, color=INK)

    # Family separators and family labels, drawn across the whole stack
    # so the grouping reads as structure rather than as a legend entry.
    edges, start = [], 0
    for family, _ranked, members in GROUPS:
        edges.append((family, start, start + len(members) - 1))
        start += len(members)
    for _, _, right in edges[:-1]:
        for ax in axes:
            ax.axvline(right + 0.5, color=MUTED, lw=0.8, ls=':', alpha=0.8)
    for family, left, right in edges:
        # A coloured BRACKET carries the family; the name beside it stays
        # in ink. Painting the label itself in the series colour would
        # put 7.5pt text at the aqua slot's 2.7:1 against the surface,
        # and would make identity colour-alone for a reader who cannot
        # separate the hues.
        bottom.annotate(
            '', xy=(left - 0.34, -0.30), xytext=(right + 0.34, -0.30),
            xycoords=('data', 'axes fraction'),
            textcoords=('data', 'axes fraction'),
            arrowprops=dict(arrowstyle='-', color=FAMILY_COLOR[family],
                            lw=2.6, shrinkA=0, shrinkB=0),
            annotation_clip=False)
        bottom.annotate(
            family, xy=((left + right) / 2, -0.36),
            xycoords=('data', 'axes fraction'), ha='center', va='top',
            fontsize=7.5, color=INK, weight='bold',
            annotation_clip=False)

    handles = [
        Line2D([0], [0], color=INK, lw=1.2, ls=(0, (4, 3)),
               label='reference level (matches ground truth)'),
        Line2D([0], [0], color=INK_2, lw=1.4,
               label='mean of the best ranked system (named per panel)'),
        Line2D([0], [0], color=INK_2, lw=6, alpha=0.35,
               label='box: quartiles over songs   notch: 95% CI of median'),
        Line2D([0], [0], color='none', marker='D', markersize=3.0,
               markerfacecolor=SURFACE, markeredgecolor=INK,
               markeredgewidth=0.8, label='mean'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=6.8, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 0.004))
    if args.title:
        fig.suptitle(args.title, fontsize=9, color=INK, y=0.995)

    fig.tight_layout(rect=(0, 0.075, 1, 0.985))
    for ext in ('pdf', 'png'):
        path = f'{args.out}.{ext}'
        fig.savefig(path, dpi=300, facecolor=SURFACE)
        print(f'wrote {path}')

    # The numbers behind the picture, so a reader of the log can check a
    # box without opening the figure.
    print(f'\nper-song n and median, reference level in brackets')
    for m in metrics:
        b = best_ranked(m, per_song, order, present)
        won = f'  best(ranked): {b[0]} mean {b[3]:+.4f}' if b else ''
        print(f'  {m}  [ref {ref_level(m):.0f}]{won}')
        for sysname, disp, _, _ in order:
            vals = list(per_song.get(m, {}).get(sysname, {}).values())
            if not vals:
                print(f'    {sysname:11s} --')
                continue
            v = np.asarray(vals)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ci = 1.58 * (q3 - q1) / math.sqrt(len(v))
            print(f'    {sysname:11s} n={len(v):3d}  median {med:+.4f}'
                  f'  IQR [{q1:+.4f},{q3:+.4f}]  CI +-{ci:.4f}')


if __name__ == '__main__':
    main()
