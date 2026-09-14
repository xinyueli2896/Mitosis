"""Box plots of E1 per-song scores, grouped by system family.

Reads the per-sample metrics CSV that eval_metrics writes and draws one
panel per metric: one box per system, systems ordered and coloured by
family (ours / internal baselines / external baselines).

WHAT THE MARKS MEAN, because a box plot of a distance metric is easy to
misread:

  box      quartiles of the PER-SONG values, with the median as a pale
           line. Samples of a song are averaged first -- three samples
           of one song share a prompt and a reference, so treating them
           as three observations understates the spread by sqrt(3).
  stick    the MEAN and its 95% confidence interval, drawn dark over the
           box. This is the significance interval: two systems whose
           sticks do not overlap differ at about the 95% level. It sits
           on the mean rather than the median so that it is like for
           like with the best-of line, which is also a mean.
  dashed   the REFERENCE LEVEL: the value a system scores when it
  line     matches the ground-truth continuation. 0 for a divergence or
           a delta, 1 for a ratio. It is a target, not a maximum --
           these metrics are two-sided, and a system can miss by
           exceeding it.
  solid    the mean of the best RANKED system on that panel, named in
  line     the corner, running the full width so the unranked column
           can be read against it.

No notches: a notch deforms the box into an hourglass and folds inside
out when the CI exceeds the quartiles, which happens on the noisier
metrics here and reads as a broken box. The interval gets its own mark
instead, where it cannot distort the distribution it describes.

Nothing here is a "higher is better" axis, so no direction arrows: each
panel says in its own label which way the reference lies.

Usage (via plot_e1_box.sbatch):
  python plot_e1_box.py --csv results/E1_p96_v5b93_metrics.csv \
      --out results/E1_p96_v5b93_box_H3 --block H3
"""

import argparse
import csv
import math
import os
import re
import sys
import textwrap
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
#
# Each member is (internal name, long label, short label). The short
# label is used as soon as the figure has more than one column, where
# nine two-line names would overlap into illegibility; the key below
# then goes in the figure footnote, so nothing is left to guess.
GROUPS = [
    ('Ours', True, [
        ('A1',        'Duet\nw/o query',            'Duet–noQ'),
        ('A3',        'Duet\n(refine)',             'Duet–R'),
        ('A3ctcaT',   'Duet\n(alt. commit)',        'Duet–AC'),
    ]),
    ('Internal baselines', True, [
        ('S-scratch', 'Single-stream\n(scratch)',   'SS–scr'),
        ('P-mc',      'Cascade\n(mel→chd)',    'Casc–mc'),
        ('P-cm',      'Cascade\n(chd→mel)',    'Casc–cm'),
    ]),
    ('External baselines', True, [
        ('WSf',       'Whole-Song\nGen',            'WSG'),
        ('AMT',       'Anticipatory\nMusic Transf.', 'AMT'),
    ]),
    # Held out of the ranking and placed last, by request. It is still
    # plotted, and the best-of-the-rest line runs the full width, so a
    # reader can see exactly where it lands relative to the field --
    # which is the only reason to separate it rather than drop it.
    ('Not ranked', False, [
        ('S1',        'Single-stream\n(finetuned)', 'SS–ft'),
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

# Metrics with no reference level: raw per-stream rates whose row holds
# no matching ground-truth statistic. Their _delta or _ratio siblings do
# have one, and those are the panels to read for "how close to real".
NO_REFERENCE = {
    'survival_min', 'survival_a', 'survival_b', 'mel_poly_rate',
    'empty_rate_a', 'empty_rate_b', 'density_drift_a', 'density_drift_b',
    'chord_tone_cov', 'ctnctr', 'pcs', 'mctd', 'coupling', 'onset_sync',
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
    'mel_interval_jsd':        'Melodic interval JSD',
    'voicing_jsd':             'Chord voicing JSD',
    # The prompt-adherence and unique-beat-ratio families are labelled by
    # label_for() from the metric name, so that 48 panels do not depend
    # on 48 hand-kept entries staying consistent with one another.
    # H1 -- per-stream role and texture integrity
    'survival_min':            'Active-bar fraction (min of streams)',
    'survival_a':              'Active-bar fraction, melody',
    'survival_b':              'Active-bar fraction, chord',
    'mel_poly_rate':           'Melody frames with $\\geq$2 onsets',
    'density_ratio_a':         'Notes/bar over ref., melody',
    'density_ratio_b':         'Notes/bar over ref., chord',
    'density_drift_a':         'Per-bar density slope, melody',
    'density_drift_b':         'Per-bar density slope, chord',
    'empty_rate_a':            'Silent-frame fraction, melody',
    'empty_rate_b':            'Silent-frame fraction, chord',
    'register_overlap_delta':  'Register overlap $-$ ref.',
    # H2 -- raw forms, shown beside their deltas
    'chord_tone_cov':          'Chord-tone coverage',
    'ctnctr':                  'CTnCTR',
    'pcs':                     'PCS',
    'mctd':                    'MCTD',
    'coupling':                'Melody-chord coupling',
    'onset_sync':              'Onset synchrony',
    'onset_sync_delta':        'Onset synchrony $-$ ref.',
    'mel_stepwise_delta':      'Melody stepwise rate $-$ ref.',
    'dur_contrast_delta':      'Duration contrast $-$ ref.',
    'density_contrast_delta':  'Density contrast $-$ ref.',
}


# The four sheets the paper wants, named for what they ask rather than
# for the hypothesis letter they came from. Each is (metrics, default
# ncols, title); NCOLS on the command line overrides the default, which
# is set from the panel count -- a 32-panel sheet two columns wide is
# two feet tall.
#
# 'quality' is H3 minus mel_stepwise_delta: the three divergences named
# for the figure, per stream where the metric has one. Add
# mel_stepwise_delta to the list if the melody-shape check belongs here
# too.
def _block(name, drop_ref=True):
    from eval_metrics import H_GROUPS
    out = list(H_GROUPS[name])
    if drop_ref:
        out = [m for m in out
               if not m.endswith('_ref') and not m.startswith('prompt_onsets')]
    return out


def presets():
    return {
        'quality': (['harmonic_rhythm_jsd',
                     'onset_grid_jsd_a', 'onset_grid_jsd_b',
                     'duration_jsd_a', 'duration_jsd_b'],
                    # One column: five panels in two leaves a hole in the
                    # grid, and the column above it ends early, so its
                    # system names land halfway up the figure.
                    1, 'General quality: distribution match to the reference'),
        'fit': (_block('H2'), 2, 'Melody-chord fit'),
        'repetition': (_block('R'), 2, 'Repetition and structuredness'),
        'prompt': (_block('P'), 4, 'Prompt adherence'),
    }


# Generated labels for the two big families, so 48 panels do not need 48
# dictionary entries kept in sync by hand.
_STREAM = {'a': 'melody', 'b': 'chord'}
_PROMPT_STAT = {
    'reuse':        'Prompt reuse',
    'rhythm_reuse': 'Rhythm-figure reuse',
    'pc_jsd':       'Pitch-class JSD vs prompt',
    'grid_jsd':     'Metrical-position JSD vs prompt',
    'ioi_jsd':      'Inter-onset JSD vs prompt',
    'dur_jsd':      'Duration JSD vs prompt',
    'density':      'Onset-rate over prompt',
    'register':     'Register shift vs prompt (semitones)',
}


def label_for(metric):
    if metric in LABELS:
        return LABELS[metric]
    base, delta = metric, ''
    if base.endswith('_delta'):
        base, delta = base[:-6], ' $-$ ref.'
    m = re.match(r'ubr(\d)_(onset|state)_([ab])$', base)
    if m:
        n, mode, st = m.groups()
        return f'Unique beat ratio, {n}-beat {mode}, {_STREAM[st]}{delta}'
    m = re.match(r'(.+)_vs_prompt_([ab])$', base)
    if m and m.group(1) in _PROMPT_STAT:
        return f'{_PROMPT_STAT[m.group(1)]}, {_STREAM[m.group(2)]}{delta}'
    return metric


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
    # These are raw rates with no reference counterpart in the row: there
    # is no value that means "matches the ground truth", so no target
    # line is drawn. Returning 0 would put a dashed rule at the bottom of
    # the axis and invite it to be read as the target.
    if metric in NO_REFERENCE:
        return None
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
    if ref is None:
        return None      # no target, so no "best"
    cands = []
    for sysname, disp, _short, family, ranked in order:
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


def draw_panel(ax, metric, per_song, order, present, title_chars=0):
    ref = ref_level(metric)
    positions, data, colors, labels = [], [], [], []
    missing = []
    for i, (sysname, disp, short, family, _ranked) in enumerate(order):
        labels.append((disp, short))
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
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=1)

    # Best of the ranked systems, drawn at its MEAN and running the full
    # width so the unranked column can be read against it. Coloured by
    # the winner's family -- the line IS that system's value -- and
    # named in ink, because a line colour alone would not say which.
    best = best_ranked(metric, per_song, order, present)
    if best is not None:
        _bs, bdisp, bfamily, bmean = best
        ax.axhline(bmean, color=FAMILY_COLOR[bfamily], lw=1.4, zorder=2)

    if data:
        # NO NOTCHES. A notch deforms the box into an hourglass, and
        # when the CI is wider than the quartiles -- which happens on
        # the noisier metrics here -- matplotlib folds it inside out and
        # the box reads as broken. The interval is drawn instead as its
        # own mark, which cannot distort the distribution it sits on.
        bp = ax.boxplot(
            data, positions=positions, widths=0.52,
            notch=False, showfliers=False, showcaps=False,
            patch_artist=True, zorder=3,
            medianprops=dict(color=SURFACE, lw=1.4),
            whiskerprops=dict(color=INK_2, lw=0.8, alpha=0.7),
        )
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor(SURFACE)
            patch.set_alpha(0.78)
            patch.set_linewidth(0.8)

        # The mean and its 95% interval, as one mark: a dark stick
        # spanning the CI with the mean at its centre. The best-of line
        # is a mean, so this is what a reader compares against it --
        # putting the interval on the mean rather than on the median
        # keeps the two like for like.
        for pos, vals in zip(positions, data):
            v = np.asarray(vals, float)
            m = v.mean()
            half = 1.96 * v.std(ddof=1) / math.sqrt(len(v))
            ax.plot([pos, pos], [m - half, m + half], color=INK, lw=1.6,
                    solid_capstyle='butt', zorder=5)
            ax.plot([pos], [m], marker='o', markersize=3.2,
                    markerfacecolor=SURFACE, markeredgecolor=INK,
                    markeredgewidth=1.0, zorder=6)

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

    # In a narrow grid the metric name goes ABOVE the panel, wrapped:
    # rotated y-axis text cannot fit a 1.75in column and runs into the
    # neighbouring panel. Horizontal text has the whole panel width.
    if title_chars:
        ax.set_title('\n'.join(textwrap.wrap(label_for(metric), title_chars)),
                     fontsize=6.4, color=INK, pad=3)
    else:
        ax.set_ylabel(label_for(metric), fontsize=7.0, color=INK)
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
    p.add_argument('--block', default='',
                   help='what to plot, overriding --metrics: a preset '
                        '(quality/fit/repetition/prompt) or a raw '
                        'hypothesis block (H1/H2/H3/P/S/R)')
    p.add_argument('--ncols', type=int, default=None,
                   help='panels per row; default is the '
                        "preset's, else 2")
    p.add_argument('--deltas-only', action='store_true',
                   help='keep only the _delta panels of a block. P and R '
                        'carry a raw value and a delta for every '
                        'statistic, and only the delta has a reference '
                        'level, so this halves a 32-panel sheet without '
                        'losing a comparison that can be read.')
    p.add_argument('--width', type=float, default=7.0, help='inches')
    p.add_argument('--panel-height', type=float, default=1.45, help='inches')
    p.add_argument('--title', default='')
    args = p.parse_args()

    auto_ncols, auto_title = None, ''
    if args.block:
        from eval_metrics import H_GROUPS
        pre = presets()
        if args.block in pre:
            metrics, auto_ncols, auto_title = pre[args.block]
        elif args.block in H_GROUPS:
            # raw hypothesis block. `_ref` columns hold the reference
            # pair's own statistic, copied into every system's row:
            # identical across columns by construction, so a box plot of
            # one is the same box drawn nine times. prompt_onsets_* goes
            # too -- a guard on the INPUT, not a score.
            metrics = _block(args.block)
        else:
            raise SystemExit(
                f'unknown block {args.block}; presets: {" ".join(pre)}; '
                f'raw blocks: {" ".join(H_GROUPS)}')
        if args.deltas_only:
            metrics = [m for m in metrics if m.endswith('_delta')]
    else:
        metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    if args.ncols is None:
        args.ncols = auto_ncols or 2
    if not args.title and auto_title:
        args.title = auto_title

    per_song = read_per_song(args.csv, metrics)

    present = set()
    for m in metrics:
        present |= set(per_song.get(m, {}))
    order = [(s, d, sh, g, ranked)
             for g, ranked, members in GROUPS for s, d, sh in members]
    known = {s for s, _, _, _, _ in order}
    extra = sorted(present - known)
    if extra:
        print(f'[warn] in the CSV but not in the figure: {" ".join(extra)}',
              file=sys.stderr)
    absent = [s for s, _, _, _, _ in order if s not in present]
    if absent:
        print(f'[warn] drawn as pending placeholders: {" ".join(absent)}',
              file=sys.stderr)

    n = len(metrics)
    ncols = max(1, min(args.ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axgrid = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(args.width, args.panel_height * nrows
                 + (2.9 if ncols > 1 else 1.6)))
    fig.patch.set_facecolor(SURFACE)

    # ~12 characters per inch at 6.4pt; 0 means "use the y-axis label".
    title_chars = int(args.width / ncols * 12) if ncols > 1 else 0
    labels = None
    used = []
    for idx, m in enumerate(metrics):
        ax = axgrid[idx // ncols][idx % ncols]
        labels = draw_panel(ax, m, per_song, order, present, title_chars)
        used.append((idx // ncols, idx % ncols, ax))
    for r in range(nrows):
        for c in range(ncols):
            if r * ncols + c >= n:
                axgrid[r][c].set_visible(False)

    # x labels on the LAST USED panel of each column, not on every one:
    # repeating nine two-line system names down a grid is most of what
    # made the earlier figure feel busy.
    axes = [a for _, _, a in used]
    bottoms = []
    for c in range(ncols):
        col = [a for r, cc, a in used if cc == c]
        if col:
            bottoms.append(col[-1])
    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([])
    # Nine systems will not fit horizontally in a half-width panel at any
    # legible size -- short codes did not fix it, they only made the
    # overlap cryptic as well as unreadable. So a multi-column figure
    # turns the names vertical, where they cost height once per column
    # instead of width nine times, and keeps the FULL names: a key of
    # invented abbreviations is a second thing for a reader to hold.
    compact = ncols > 1
    if compact:
        shown = [lg.replace('\n', ' ') for lg, _sh in labels]
        for bottom in bottoms:
            bottom.set_xticklabels(shown, fontsize=6.0, color=INK,
                                   rotation=90, ha='center', va='top')
    else:
        for bottom in bottoms:
            bottom.set_xticklabels([lg for lg, _sh in labels],
                                   fontsize=6.2, color=INK)

    edges, start = [], 0
    for family, _ranked, members in GROUPS:
        edges.append((family, start, start + len(members) - 1))
        start += len(members)
    # Family brackets under the axis only in the single-column figure.
    # Under vertical names they would sit an inch and a half down, and
    # the family names themselves would overlap at half width -- so in
    # compact mode the families move into the legend instead, where the
    # swatch carries the colour and the text stays in ink.
    if not compact:
        for bottom in bottoms:
            for family, left, right in edges:
                bottom.annotate(
                    '', xy=(left - 0.34, -0.42), xytext=(right + 0.34, -0.42),
                    xycoords=('data', 'axes fraction'),
                    textcoords=('data', 'axes fraction'),
                    arrowprops=dict(arrowstyle='-',
                                    color=FAMILY_COLOR[family],
                                    lw=2.4, shrinkA=0, shrinkB=0),
                    annotation_clip=False)
                bottom.annotate(
                    family, xy=((left + right) / 2, -0.48),
                    xycoords=('data', 'axes fraction'), ha='center',
                    va='top', fontsize=6.8, color=INK, weight='bold',
                    annotation_clip=False)

    handles = [
        Line2D([0], [0], color=INK, lw=1.2, ls=(0, (4, 3)),
               label='reference level (matches ground truth)'),
        Line2D([0], [0], color=INK_2, lw=1.4,
               label='mean of the best ranked system (named per panel)'),
        Line2D([0], [0], color=INK_2, lw=6, alpha=0.35,
               label='box: quartiles over songs (line: median)'),
        Line2D([0], [0], color=INK, lw=1.6, marker='o', markersize=3.2,
               markerfacecolor=SURFACE, markeredgecolor=INK,
               markeredgewidth=1.0, label='mean and its 95% CI'),
    ]
    if compact:
        handles += [Line2D([0], [0], color=FAMILY_COLOR[f], lw=6,
                           alpha=0.78, label=f) for f, _l, _r in edges]
    # 3 columns, not 4: at four the widest row ran past the canvas and
    # clipped the last family name.
    fig.legend(handles=handles, loc='lower center',
               ncol=3 if compact else 2, frameon=False,
               fontsize=6.4, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 0.004))
    if args.title:
        fig.suptitle(args.title, fontsize=9, color=INK, y=0.995)

    fig.tight_layout(rect=(0, 0.075 if ncols == 1 else 0.10, 1, 0.985))
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
        r = ref_level(m)
        rtxt = 'no reference level' if r is None else f'ref {r:.0f}'
        print(f'  {m}  [{rtxt}]{won}')
        for sysname, disp, _short, _, _ in order:
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
