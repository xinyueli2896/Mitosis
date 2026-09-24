"""Distribution plots of E1 per-song scores, grouped by system family.

Reads the per-sample metrics CSV that eval_metrics writes and draws one
panel per metric: one box per system, systems ordered by family
(ours / internal baselines / external baselines / not ranked), the box
filled in its family's colour. Plain journal box plots. Default is the
WIDE layout (--orient h): four metrics side by side across both columns
of a two-column template (7.0 in), systems down the y-axis with names
on the leftmost panel; --orient v is the one-column layout with upright
boxes (3.4 in).

WHAT THE MARKS MEAN, because a plot of a distance metric is easy to
misread:

  box      the PER-SONG values: box from the first to the third
           quartile, median as a line across it, whiskers to the last
           value within 1.5 IQR; outliers beyond the whiskers are not
           drawn (the caption says so).
           Samples of a song are averaged first -- three samples of one
           song share a prompt and a reference, so treating them as
           three observations understates the spread by sqrt(3).
  diamond  the MEAN, white with an ink edge. The best-of line below is
           a mean, so this is the mark to read against it.
  dashed   the REFERENCE LEVEL: the value a system scores when it
  line     matches the ground-truth continuation. 0 for a divergence or
           a delta, 1 for a ratio. It is a target, not a maximum --
           these metrics are two-sided, and a system can miss by
           exceeding it.
  solid    the mean of the best RANKED system on that panel, named in
  line     the corner and drawn in that system's colour, running the
           full width so the unranked column can be read against it.

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
import matplotlib.colors
import matplotlib.ticker
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# System registry. Display names are the paper's, not the repo's.
# ---------------------------------------------------------------------------
# Each entry is (family, ranked, members). `ranked` says whether the
# family takes part in the best-model comparison drawn on each panel.
#
# Naming (2026-09-17, by request): "Duet" is the A3 checkpoint under
# the ALTERNATING-COMMIT decode (A3ctcaT: one stream's draft committed,
# the other predicted by its harmonizer, leader alternating per frame,
# follower at T=1, no nucleus cut). The refinement decode of the same
# checkpoint (A3, K=4, T=0.9, top-p 0.95) is NOT reported any more and
# is off the figures; it stays in the CSVs. "Duet w/o draft tokens" is
# A1, a separate checkpoint trained without the draft-token pathway and
# decoded autoregressively.
#
# Each member is (internal name, long label, short label). The short
# label is used as soon as the figure has more than one column, where
# nine two-line names would overlap into illegibility; the key below
# then goes in the figure footnote, so nothing is left to guess.
GROUPS = [
    # ORDER (2026-09-23, by request): ours first, then the single-stream
    # models scratch and finetuned, then the two external systems.
    # RANKING (2026-09-24, by request): the best system is marked again
    # (solid line at its mean, star at the row end), over every family
    # except the members of UNRANKED below -- the single-stream finetuned
    # model, which is not ranked in the tables either. "Duet w/o draft
    # tokens" is A1, the checkpoint trained without the draft-token
    # pathway and decoded autoregressively.
    ('Ours', True, [
        # a second line starting with "w/o", a minus or a bracket is a
        # QUALIFIER: the wide layout draws it a point smaller, in grey
        ('A3ctcaT',   'Duet',                     'Duet'),
        # the dense arm: one FFN of width 6144 in place of the routed
        # pool. The expert-adapter variants (A3L16E16, A3L16E16_nocross)
        # are scored in the CSVs but left off every figure and table.
        ('D1',        'Duet\nw/o MoE (dense)',   'Duet\ndense'),
        ('A1',        'Duet\nw/o draft tokens',  'Duet\nw/o DT'),
    ]),
    # The cascade arms (P-mc, P-cm) left the figures 2026-09-17, by
    # request; they are still scored and sit in the CSVs.
    ('Internal baselines', True, [
        ('S-scratch', 'Single-stream\n(scratch)',   'SS\nscr.'),
        ('S1',        'Single-stream\n(finetuned)', 'SS\nft.'),
    ]),
    ('External baselines', True, [
        # WSfv4: the whole-song baseline with its chord track re-voiced
        # into our rendering and cut to four voices (wholesong_chord_map
        # APPLY=WSfv4 MAX_VOICES=4). The raw WSf differs from every
        # other system in absolute pitch on every chord, and the
        # uncapped WSfv holds the fifth note of a seventh chord that the
        # cp4 duet arms never see.
        ('WSfv4',     'Whole-Song Gen',             'WSG'),
        ('AMT',       'AMT',                        'AMT'),
    ]),
]
# systems drawn but left out of the best-of comparison whatever their
# family says: the finetuned single-stream model starts from the same
# pretrained weights as ours but sees the task with no architectural
# change, so it is reported, not ranked (the tables star it likewise)
UNRANKED = {'S1'}

# Palette (2026-09-17, by request): four of the six swatches carry the
# FAMILIES -- the two darkest (near-black, navy) are kept off the boxes
# and used only for text and spines. Colour carries family, not system:
# the family is the comparison the figure is making, and every column
# is directly labelled besides. Ours in gold; internal baselines in
# slate; external baselines in maroon; the unranked column in warm
# grey. A per-system override can go in SYSTEM_COLOR; it is empty by
# request.
# Vibrant variants of the four family swatches (2026-09-17, by request:
# the originals read as vintage). Same hue roles -- warm gold for ours,
# a cool blue for the internal baselines, a red for the external ones,
# a neutral for the unranked column -- with the saturation raised.
PALETTE = {
    'black':  '#252323',
    'maroon': '#b3243f',    # was #462025
    'gold':   '#f2a900',    # was #e7af36
    'slate':  '#3b7dd8',    # was #70798c
    'grey':   '#9a9186',    # was #a39c8f
}
# Per-system shades within the family hue (2026-09-24, by request): the
# flagship keeps the family colour and its ablations two lighter shades
# of it, in the order of the sheet; each baseline family likewise runs
# from its darker to its lighter member.
# With --names none the legend names every system by its swatch and the
# x axis carries no names at all.
SYSTEM_COLOR = {
    # ours: darkest to lightest in the order of the sheet
    'A3ctcaT':   PALETTE['gold'],
    'D1':        '#f7c452',        # lighter gold
    'A1':        '#fbdc95',        # lighter still
    'S-scratch': PALETTE['slate'],
    'S1':        '#93b9ea',        # lighter slate
    'WSfv4':     PALETTE['maroon'],
    'AMT':       '#e0788c',        # lighter maroon
}
FAMILY_COLOR = {
    'Ours':                PALETTE['gold'],
    'Internal baselines':  PALETTE['slate'],
    'External baselines':  PALETTE['maroon'],
    'Not ranked':          PALETTE['grey'],
}


def color_of(sysname, family):
    return SYSTEM_COLOR.get(sysname, FAMILY_COLOR[family])


def lighten(color, amount=0.55):
    """Mix a colour toward white for a box fill the median still reads on."""
    r, g, b, _ = matplotlib.colors.to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def grouped_legend(fig, blocks, y=0.0, xs=None, ncol=None, loc='lower left',
                   **kw):
    """Several titled legend blocks side by side along the bottom.

    blocks: [(title, handles, labels)]. Each block is its own legend
    with a left-aligned bold title, so colour (model group) and mark or
    line type sit in separate, labelled blocks instead of one mixed
    list. xs: block anchors in figure fraction (left edges for the
    default loc='lower left'; default evenly spaced); ncol: columns per
    block, an int or one per block (default 1).
    """
    n = len(blocks)
    if xs is None:
        xs = [(i + 0.5) / n for i in range(n)]
    opts = dict(handlelength=1.6, handletextpad=0.5, columnspacing=1.0,
                labelspacing=0.3)
    opts.update(kw)
    if not isinstance(ncol, (list, tuple)):
        ncol = [ncol or 1] * n
    legs = []
    for (title, handles, labels), x, nc in zip(blocks, xs, ncol):
        leg = fig.legend(handles=handles, labels=labels, loc=loc,
                         bbox_to_anchor=(x, y), ncol=nc, frameon=False,
                         fontsize=FS_LEGEND, labelcolor=INK, title=title,
                         title_fontsize=FS_LEGEND, alignment='left',
                         handler_map={tuple: HandlerTuple(ndivide=None,
                                                          pad=0.25)}, **opts)
        leg.get_title().set_color(INK)
        leg.get_title().set_fontweight('bold')
        legs.append(leg)
    return legs

# Metrics with no reference level: raw per-stream rates whose row holds
# no matching ground-truth statistic. Their _delta or _ratio siblings do
# have one, and those are the panels to read for "how close to real".
NO_REFERENCE = {
    'survival_min', 'survival_a', 'survival_b', 'mel_poly_rate',
    'empty_rate_a', 'empty_rate_b', 'density_drift_a', 'density_drift_b',
    'chord_tone_cov', 'ctnctr', 'pcs', 'mctd', 'coupling', 'onset_sync',
    # exact repetition of the prompt scores 0 / 1 here, so neither is a
    # target; the delta carries the reference
    'pc_jsd_prompt_a', 'pc_jsd_prompt_b',
    'onset_sim_prompt_a', 'onset_sim_prompt_b',
    'gc_a', 'gc_b', 'sc_a', 'sc_b',
}

SURFACE = '#ffffff'          # white ground
INK = PALETTE['black']       # text, spines, medians, the reference line
INK_2 = '#5a5754'            # secondary text (slate now carries a family)
MUTED = PALETTE['grey']      # placeholders, the noise-floor line
GRID = '#e8e6e2'             # horizontal rules behind the boxes

# Type sizes, in points. 7 pt is the floor for a one-column figure in a
# 10 pt two-column template: anything smaller prints grey.
FS_TICK = 7.0
FS_LABEL = 7.5
FS_TITLE = 7.5
FS_LETTER = 8.5
FS_LEGEND = 7.0

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans',
                        'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.color': INK,
    'axes.labelcolor': INK,
    'axes.linewidth': 0.6,
})

DEFAULT_METRICS = [
    'harmonic_rhythm_jsd',
    'chord_tone_cov_delta',
    'coupling_delta',
    'reuse_vs_prompt_a_delta',
]

# Axis labels. Keyed by metric; anything unlisted falls back to the raw
# name, so an unfamiliar metric still plots.
LABELS = {
    'fmd':                     'Frechet Music Distance (CLaMP 2)',
    'js_gc_a':  'Groove consistency JSD (melody)',
    'js_gc_b':  'Groove consistency JSD (chord)',
    'js_sc_a':  'Scale consistency JSD (melody)',
    'js_sc_b':  'Scale consistency JSD (chord)',
    'gc_a': 'Groove consistency, melody', 'gc_b': 'Groove consistency, chord',
    'sc_a': 'Scale consistency, melody',  'sc_b': 'Scale consistency, chord',
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
    # prompt adherence, the reported pair
    'pc_jsd_prompt_a':    'Pitch-class JSD vs prompt, melody',
    'pc_jsd_prompt_b':    'Pitch-class JSD vs prompt, chord',
    'onset_sim_prompt_a': 'Onset-pattern similarity vs prompt, melody',
    'onset_sim_prompt_b': 'Onset-pattern similarity vs prompt, chord',
    # pattern coverage, above a shuffled-prompt control (P1, CSV only)
    'motif_cov3_a':  'Motif coverage, 3-gram, melody',
    'motif_cov5_a':  'Motif coverage, 5-gram, melody',
    'rhythm_cov3_a': 'Rhythm coverage, 3-gram, melody',
    'rhythm_cov5_a': 'Rhythm coverage, 5-gram, melody',
    'joint_cov3_a':  'Motif+rhythm coverage, 3-gram, melody',
    'joint_cov5_a':  'Motif+rhythm coverage, 5-gram, melody',
    'motif_cov3_b':  'Bass-motion coverage, 3-gram, chord',
    'motif_cov5_b':  'Bass-motion coverage, 5-gram, chord',
    'rhythm_cov3_b': 'Rhythm coverage, 3-gram, chord',
    'rhythm_cov5_b': 'Rhythm coverage, 5-gram, chord',
    'joint_cov3_b':  'Bass+rhythm coverage, 3-gram, chord',
    'joint_cov5_b':  'Bass+rhythm coverage, 5-gram, chord',
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
        # POOLED. Every metric here is a divergence from the reference
        # distribution, and at ~100 onsets a per-song JSD carries an
        # upward bias of ~0.06 that swamps the differences between
        # systems. Pooled over the corpus the bias drops below 0.001, so
        # this sheet reads the pooled value with a bootstrap over songs.
        # There is no distribution left to box -- one histogram per
        # system is one number -- so the mark is a point and its
        # interval.
        'quality': (['fmd',
                     # JSD between generated and real DISTRIBUTIONS of
                     # two per-piece regularity statistics (Dong et al.
                     # 2020): groove consistency and scale consistency.
                     # The event-level JSDs (harmonic rhythm, onset grid,
                     # duration) were dropped from the sheet 2026-09-14
                     # by request; they stay in the pooled table and CSV
                     # and --block H3 still draws them per song.
                     'js_gc_a', 'js_gc_b', 'js_sc_a', 'js_sc_b'],
                    2, 'General quality: corpus-level distance from the '
                       'reference', True),
        # DELTAS ONLY on the three per-song sheets (2026-09-14, by
        # request). A raw value here has no target -- it rewards copying
        # the prompt, or has no reference counterpart at all -- and it
        # sits on the estimator's noise floor; the delta against the
        # ground-truth continuation is the number with a meaningful
        # zero. The raw columns stay in the table and the CSV.
        # chord_tone_cov stays in the table and CSV but is off the sheet
        # by request; onset_sync is the drum-task coupling statistic and
        # is never scored on melchord, so it would only ever be a row of
        # placeholders here.
        'fit': ([m for m in _block('H2')
                 if m.endswith('_delta')
                 and not m.startswith(('chord_tone_cov', 'onset_sync'))],
                2, 'Melody-chord fit, relative to the reference', False),
        'repetition': ([m for m in _block('R') if m.endswith('_delta')],
                       2, 'Repetition and structuredness, relative to the '
                          'reference', False),
        # Raw value beside its delta. Both measures reward exact
        # repetition, so the raw value has no target line -- the delta
        # against the ground-truth continuation is where 0 means
        # "varies from the prompt as much as the real song did".
        'prompt': ([f'{k}_prompt_{st}_delta'
                    for st in ('a', 'b') for k in ('pc_jsd', 'onset_sim')],
                   2, 'Prompt adherence, relative to the reference', False),
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
    # a hand-written entry for the plain metric also labels its _delta
    if base in LABELS:
        return LABELS[base] + delta
    m = re.match(r'ubr(\d)_(onset|state)_([ab])$', base)
    if m:
        n, mode, st = m.groups()
        # UBR, spelled out in the caption: the full name is four lines
        # at one-column width
        return f'UBR {n}-beat {mode}, {_STREAM[st]}{delta}'
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


def system_mean(metric, sysname, per_song):
    vals = list(per_song.get(metric, {}).get(sysname, {}).values())
    return float(np.mean(vals)) if len(vals) >= 3 else None


def read_pooled(path):
    """{metric: {system: (jsd, ci_lo, ci_hi, n_songs, n_obs)}}."""
    out = defaultdict(dict)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            def num(k):
                try:
                    return float(row[k])
                except (KeyError, TypeError, ValueError):
                    return float('nan')
            out[row['metric']][row['system']] = (
                num('jsd'), num('ci_lo'), num('ci_hi'),
                int(float(row.get('n_songs') or 0)),
                int(float(row.get('n_obs') or 0)),
                row.get('boot_mode') or 'songs',
                row.get('weight') or 'note',
                # per-sample pooled divergence (absent in older CSVs)
                num('ps_mean'), num('ps_std'),
                int(float(row.get('ps_n') or 0)))
    return out


def _finish_axes(ax, metric, order, title_chars, letter, caption):
    """Axis furniture shared by both panel kinds.

    Journal conventions: white ground, left and bottom spines in ink,
    faint horizontal rules for reading values across the boxes, no top
    or right spine. The panel letter is bold in the corner, the caption
    after it.
    """
    if title_chars is None:
        pass                        # --no-titles: the caption names the panels
    elif title_chars:
        ax.set_title('\n'.join(textwrap.wrap(label_for(metric), title_chars)),
                     fontsize=FS_TITLE, color=INK, pad=3)
    else:
        ax.set_ylabel(label_for(metric), fontsize=FS_LABEL, color=INK)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.tick_params(axis='y', labelsize=FS_TICK, colors=INK, length=2.5,
                   width=0.6, color=INK)
    ax.tick_params(axis='x', length=2.0, width=0.6, color=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.6)
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    if letter and title_chars is None:
        # no title above the panel: the letter goes into that margin,
        # clear of a star in the first column
        ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(0, 2), textcoords='offset points',
                    ha='left', va='bottom', fontsize=FS_LETTER, weight='bold',
                    color=INK)
    elif letter:
        ax.annotate(letter, xy=(0.012, 0.97), xycoords='axes fraction',
                    ha='left', va='top', fontsize=FS_LETTER, weight='bold',
                    color=INK)
    if caption:
        ax.annotate(caption, xy=(0.012, 0.97), xycoords='axes fraction',
                    xytext=(13 if letter else 0, 0), textcoords='offset points',
                    ha='left', va='top', fontsize=FS_TICK, color=INK)


def _pending(ax, i, lo_y, hi_y):
    ax.add_patch(Rectangle((i - 0.31, lo_y), 0.62, hi_y - lo_y,
                           facecolor='none', edgecolor=MUTED,
                           hatch='///', lw=0.6, alpha=0.5, zorder=2))
    ax.text(i, lo_y + 0.5 * (hi_y - lo_y), 'pending', rotation=90,
            ha='center', va='center', fontsize=FS_TICK, color=MUTED)


def draw_panel_pooled(ax, metric, pooled, order, present, title_chars=0,
                      null=None, letter=''):
    """One point per system: the corpus-pooled JSD and its bootstrap CI."""
    vals = pooled.get(metric, {})
    labels, missing = [], []
    ax.axhline(0.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if null is not None and not math.isnan(null):
        # noise floor at this sample size: a system at or below it is
        # indistinguishable from perfect, and gaps below it are not gaps
        ax.axhline(null, color=MUTED, lw=1.0, ls=(0, (1, 1.5)), zorder=1)

    ranked_best = None
    for i, (sysname, disp, short, family, ranked) in enumerate(order):
        labels.append((disp, short))
        rec = vals.get(sysname)
        if rec is None or math.isnan(rec[0]):
            missing.append(i)
            continue
        v, lo, hi, _ns, _no, _bl, _w = rec
        col = color_of(sysname, family)
        if not math.isnan(lo):
            # a bare vertical line has no visible end: where the
            # interval stops was a guess. Caps make the endpoints marks.
            cap = 0.16
            ax.plot([i, i], [lo, hi], color=INK_2, lw=1.1,
                    solid_capstyle='butt', zorder=4)
            for y in (lo, hi):
                ax.plot([i - cap, i + cap], [y, y], color=INK_2, lw=1.1,
                        zorder=4)
        ax.plot([i], [v], marker='o', markersize=6.0,
                markerfacecolor=col, markeredgecolor=SURFACE,
                markeredgewidth=1.0, zorder=5)
        if ranked and (ranked_best is None or v < ranked_best[0]):
            ranked_best = (v, disp, col)

    # JSD is bounded below by 0 and 0 IS the reference, so best is simply
    # the smallest -- no absolute-distance rule needed here.
    # The unranked column gets no line of its own, by request: it is
    # read against the best-ranked line like any other column.
    if ranked_best is not None:
        ax.axhline(ranked_best[0], color=ranked_best[2], lw=1.4, zorder=2)

    lo_y, hi_y = ax.get_ylim()
    hi_y = hi_y + 0.12 * (hi_y - lo_y)
    lo_y = min(lo_y, -0.02 * (hi_y - lo_y))
    ax.set_ylim(lo_y, hi_y)
    for i in missing:
        _pending(ax, i, lo_y, hi_y)
    caption = ('best ranked: ' + ranked_best[1].replace('\n', ' ')
               if ranked_best is not None else '')
    _finish_axes(ax, metric, order, title_chars, letter, caption)
    return labels


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


def draw_panel(ax, metric, per_song, order, present, title_chars=0,
               letter='', overall=False):
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
            colors.append(color_of(sysname, family))
        else:
            missing.append(i)

    # Reference line first, so the boxes sit on top of it.
    if ref is not None:
        ax.axhline(ref, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)

    # Best of the ranked systems, drawn at its MEAN and running the full
    # width so the unranked column can be read against it. Coloured by
    # the winner's family -- the line IS that system's value -- and
    # named in ink, because a line colour alone would not say which.
    # No line for the unranked column's mean, by request.
    levels = [] if ref is None else [ref]

    best = best_ranked(metric, per_song, order, present)
    if best is not None:
        bs, bdisp, bfamily, bmean = best
        ax.axhline(bmean, color=color_of(bs, bfamily), lw=1.2, zorder=2)
        levels.append(bmean)

    # Standard box plot: quartile box, median line, whiskers to the last
    # value within 1.5 IQR. Outliers are NOT drawn (2026-09-17, by
    # request); the caption says so. The fill is the family colour
    # mixed toward white so the ink median stays readable on the two
    # dark families; the edge and whiskers carry the colour at full
    # strength.
    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.62,
                        patch_artist=True, whis=1.5, showfliers=False,
                        manage_ticks=False, zorder=3,
                        medianprops=dict(color=INK, lw=1.1),
                        whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8),
                        boxprops=dict(lw=0.8),
                        flierprops=dict(marker='o', markersize=1.8,
                                        markerfacecolor=INK_2,
                                        markeredgecolor='none', alpha=0.7))
        for k, c in enumerate(colors):
            bp['boxes'][k].set_facecolor(lighten(c))
            bp['boxes'][k].set_edgecolor(c)
            for part in ('whiskers', 'caps'):
                for art in bp[part][2 * k:2 * k + 2]:
                    art.set_color(c)
        # The mean as a small white diamond: the best-of line is a mean,
        # so this is the mark a reader compares against it.
        for pos, vals in zip(positions, data):
            ax.plot([pos], [float(np.mean(vals))], marker='D',
                    markersize=2.6, markerfacecolor=SURFACE,
                    markeredgecolor=INK, markeredgewidth=0.7, zorder=5)

    # The y-range is set by the whiskers: a single far-off song would
    # otherwise flatten every box to a line.
    ends = []
    for d in data:
        v = np.asarray(d, float)
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
        ends += [inside.min(), inside.max()]
    lo = min(ends + levels) if ends else 0.0
    hi = max(ends + levels) if ends else 1.0
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.06 * span, hi + 0.06 * span

    # Headroom for the panel letter and the best-of star, claimed before
    # anything is placed against the limits: the placeholders span the
    # full height, so they have to be drawn against the FINAL ylim or
    # they stop short.
    hi = hi + 0.14 * (hi - lo)
    ax.set_ylim(lo, hi)

    # Placeholder slots for systems not yet run: an empty hatched frame
    # is honest about the gap in a way a missing tick is not -- a reader
    # scanning the axis would otherwise not know the column was meant to
    # be there.
    for i in missing:
        _pending(ax, i, lo, hi)

    # The best ranked system is marked with a star above its box rather
    # than named in the corner: at one-column width the name overran the
    # panel. The star sits in the headroom, clear of the outliers.
    if best is not None:
        bpos = next(i for i, o in enumerate(order) if o[0] == bs)
        ax.plot([bpos], [hi - 0.05 * (hi - lo)], marker='*', markersize=6.5,
                markerfacecolor=color_of(bs, bfamily), markeredgecolor=INK,
                markeredgewidth=0.5, lw=0, clip_on=False, zorder=7)
    if overall:
        # --best-overall: the best of EVERY column, ranked or not, as a
        # hollow grey star; skipped when it is the ranked winner already
        bo = best_ranked(metric, per_song,
                         [(s_, d_, sh_, g_, True) for s_, d_, sh_, g_, _r in order],
                         present)
        if bo is not None and (best is None or bo[0] != best[0]):
            opos = next(i for i, o in enumerate(order) if o[0] == bo[0])
            ax.plot([opos], [hi - 0.05 * (hi - lo)], marker='*', markersize=6.5,
                    markerfacecolor=SURFACE, markeredgecolor=INK_2,
                    markeredgewidth=0.7, lw=0, clip_on=False, zorder=7)
    _finish_axes(ax, metric, order, title_chars, letter, '')
    return labels


def _split_label(disp):
    """(name, qualifier): the qualifier is a second line that starts
    with a minus sign or a bracket, drawn smaller and greyer; anything
    else is one line at full size."""
    parts = disp.split('\n', 1)
    if len(parts) == 2 and (parts[1][:1] in ('−', '-', '(')
                            or parts[1].startswith('w/o')):
        return parts[0], parts[1]
    return disp.replace('\n', ' '), ''


def draw_panel_h(ax, metric, per_song, order, present, letter=''):
    """draw_panel with the systems down the y-axis, values along x.

    Wide layout (2026-09-17): four metrics side by side across both
    columns, the first system at the top. Same marks as draw_panel --
    box, median, 1.5 IQR whiskers, no outliers, mean diamond, dashed
    reference, best ranked system's mean as a solid line with a star.
    """
    n = len(order)
    ypos = {i: n - 1 - i for i in range(n)}          # first system on top
    ref = ref_level(metric)
    positions, data, colors = [], [], []
    missing = []
    for i, (sysname, _disp, _short, family, _ranked) in enumerate(order):
        vals = (list(per_song.get(metric, {}).get(sysname, {}).values())
                if sysname in present else [])
        if len(vals) >= 3:
            positions.append(ypos[i]); data.append(vals)
            colors.append(color_of(sysname, family))
        else:
            missing.append(ypos[i])

    if ref is not None:
        ax.axvline(ref, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
    levels = [] if ref is None else [ref]
    best = best_ranked(metric, per_song, order, present)
    if best is not None:
        bs, _bdisp, bfamily, bmean = best
        ax.axvline(bmean, color=color_of(bs, bfamily), lw=1.2, zorder=2)
        levels.append(bmean)

    if data:
        kw = ({'orientation': 'horizontal'}
              if matplotlib.__version__ >= '3.10' else {'vert': False})
        bp = ax.boxplot(data, positions=positions, widths=0.62,
                        patch_artist=True, whis=1.5, showfliers=False,
                        manage_ticks=False, zorder=3,
                        medianprops=dict(color=INK, lw=1.0),
                        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                        boxprops=dict(lw=0.8), **kw)
        for k, c in enumerate(colors):
            bp['boxes'][k].set_facecolor(lighten(c))
            bp['boxes'][k].set_edgecolor(c)
            for part in ('whiskers', 'caps'):
                for art in bp[part][2 * k:2 * k + 2]:
                    art.set_color(c)
        for pos, vals in zip(positions, data):
            ax.plot([float(np.mean(vals))], [pos], marker='D',
                    markersize=2.6, markerfacecolor=SURFACE,
                    markeredgecolor=INK, markeredgewidth=0.7, zorder=5)

    ends = []
    for d in data:
        v = np.asarray(d, float)
        q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
        inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
        ends += [inside.min(), inside.max()]
    lo = min(ends + levels) if ends else 0.0
    hi = max(ends + levels) if ends else 1.0
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.06 * span, hi + 0.06 * span
    hi = hi + 0.12 * (hi - lo)                       # room for the star
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, n - 0.3)
    for y in missing:
        ax.add_patch(Rectangle((lo, y - 0.31), hi - lo, 0.62,
                               facecolor='none', edgecolor=MUTED,
                               hatch='///', lw=0.6, alpha=0.5, zorder=2))
        ax.text(lo + 0.5 * (hi - lo), y, 'pending', ha='center',
                va='center', fontsize=FS_TICK, color=MUTED)
    if best is not None:
        by = ypos[next(i for i, o in enumerate(order) if o[0] == bs)]
        ax.plot([hi - 0.06 * (hi - lo)], [by], marker='*', markersize=6.5,
                markerfacecolor=color_of(bs, bfamily), markeredgecolor=INK,
                markeredgewidth=0.5, lw=0, clip_on=False, zorder=7)

    # furniture: title on top, values along x, faint vertical rules
    ax.set_title('\n'.join(textwrap.wrap(label_for(metric), 22)),
                 fontsize=FS_TITLE, color=INK, pad=4)
    ax.tick_params(axis='x', labelsize=FS_TICK, colors=INK, length=2.5,
                   width=0.6, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK); ax.spines[side].set_linewidth(0.6)
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    if letter:
        ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-2, 4), textcoords='offset points',
                    ha='right', va='bottom', fontsize=FS_LETTER,
                    weight='bold', color=INK)
    return ypos


def _write_names_h(ax, order, ypos):
    """System names on the y-axis of the leftmost panel, as text: the
    model name at tick size, a qualifier line (e.g. "− harmonizers")
    a point smaller and in grey beneath it."""
    for i, (_s, disp, _sh, _g, _r) in enumerate(order):
        name, qual = _split_label(disp)
        y = ypos[i]
        if qual:
            ax.annotate(name, xy=(0, y), xycoords=('axes fraction', 'data'),
                        xytext=(-4, 0.5), textcoords='offset points',
                        ha='right', va='bottom', fontsize=FS_TICK, color=INK)
            ax.annotate(qual, xy=(0, y), xycoords=('axes fraction', 'data'),
                        xytext=(-4, -0.5), textcoords='offset points',
                        ha='right', va='top', fontsize=FS_TICK - 1.0,
                        color=INK_2)
        else:
            ax.annotate(name, xy=(0, y), xycoords=('axes fraction', 'data'),
                        xytext=(-4, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=FS_TICK, color=INK)


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
    p.add_argument('--pooled', action='store_true',
                   help='plot the CORPUS-POOLED value with its bootstrap '
                        'CI instead of the per-song distribution. One '
                        'histogram per system over every song, so there '
                        'is nothing to box: the mark is a point and an '
                        'interval. Only defined for the pooled metrics.')
    p.add_argument('--per-song', action='store_true',
                   help='force the per-song distribution even for a preset '
                        'that defaults to pooled.')
    p.add_argument('--pooled-csv', default=None,
                   help='pooled CSV; default is --csv with _metrics.csv '
                        'swapped for _pooled.csv')
    p.add_argument('--order', default='',
                   help='sheet-specific column order, internal names '
                        'space-separated; unnamed systems follow in '
                        'registry order')
    p.add_argument('--ranked', default=None,
                   help='sheet-specific ranked set (internal names, '
                        'space-separated): only these compete for the '
                        'best-of line and star; overrides the family flags')
    p.add_argument('--best-overall', action='store_true',
                   help='also mark the best of ALL columns, ranked or not, '
                        'with a hollow grey star (no line); omitted when '
                        'it is the ranked winner')
    p.add_argument('--divider-after', default='',
                   help='draw a dashed vertical rule after this system '
                        '(internal name) on every panel, e.g. between the '
                        'ranked and the unranked columns')
    p.add_argument('--exclude', default='',
                   help='comma-separated internal system names to leave '
                        'off the figure entirely (no column, no '
                        'placeholder), e.g. A3')
    p.add_argument('--deltas-only', action='store_true',
                   help='keep only the _delta panels of a block. P and R '
                        'carry a raw value and a delta for every '
                        'statistic, and only the delta has a reference '
                        'level, so this halves a 32-panel sheet without '
                        'losing a comparison that can be read.')
    p.add_argument('--width', type=float, default=3.4,
                   help='inches for --orient v (3.4 = one column). The '
                        'wide layout uses 7.0, the full page width, '
                        'unless a value above 5 is given')
    p.add_argument('--panel-height', type=float, default=1.35, help='inches')
    p.add_argument('--title', default='')
    p.add_argument('--no-titles', action='store_true',
                   help='--orient v: no metric name on the panels, only the '
                        'letters; the caption names them')
    p.add_argument('--names', choices=['long', 'short', 'none'], default='long',
                   help='--orient v only: long (default) writes the full '
                        'names under the bottom panels, vertical when there '
                        'is more than one column; short writes the two-line '
                        'short codes horizontally, which keeps the figure '
                        'one column wide without a name column; none writes '
                        'no names on the axis and lists every system by its '
                        'colour swatch in the legend instead')
    p.add_argument('--orient', choices=['h', 'v'], default='h',
                   help="h (default): wide layout, four metrics side by "
                        "side across both columns, systems down the "
                        "y-axis, names on the leftmost panel; v: the "
                        "one-column layout, boxes upright, names below")
    args = p.parse_args()

    auto_ncols, auto_title, pooled_mode = None, '', False
    if args.block:
        from eval_metrics import H_GROUPS
        pre = presets()
        if args.block in pre:
            metrics, auto_ncols, auto_title, pooled_mode = pre[args.block]
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
        if args.deltas_only and not pooled_mode:
            # a pooled sheet is divergence-from-reference already and
            # carries no _delta suffix; the flag would empty it
            metrics = [m for m in metrics if m.endswith('_delta')]
    else:
        metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    if args.ncols is None:
        args.ncols = auto_ncols or 2
    user_ncols = args.ncols if args.ncols != (auto_ncols or 2) else None
    if auto_title:
        # printed, not drawn: the LaTeX caption carries the sheet's
        # title, and at one-column width the height is the panels'.
        # --title still draws one on request.
        print(f'[sheet] {auto_title}')
    if args.per_song:
        pooled_mode = False
    if args.pooled:
        pooled_mode = True

    pooled, null_level = {}, {}
    if pooled_mode:
        pc = args.pooled_csv or re.sub(r'_metrics\.csv$', '_pooled.csv',
                                       args.csv)
        if not os.path.exists(pc):
            raise SystemExit(
                f'ERROR: pooled mode needs {pc}, which eval_metrics writes '
                'with --pooled-out (score_e1/eval_e1 do this by default). '
                'Rescore, or pass --per-song to plot the per-song values '
                'instead -- but do not describe those as unbiased.')
        pooled = read_pooled(pc)
        # FMD lives in its own CSV because it needs a separate venv and a
        # GPU; it has the same schema and the same shape of result -- one
        # value per system with a song bootstrap -- so it merges in as
        # another pooled metric rather than needing its own plot path.
        fc = re.sub(r'_pooled\.csv$', '_fmd.csv', pc)
        if fc != pc and os.path.exists(fc):
            for k, v in read_pooled(fc).items():
                pooled[k] = {s_: r for s_, r in v.items()
                             if not s_.startswith('_')}
            print(f'[fmd] merged {fc}')
        # The reference split-half null rides in the pooled CSV as a
        # pseudo-system. It is a floor -- what a perfect system scores
        # against half the corpus -- so it is drawn as a level on each
        # panel, not as a column, and stripped from the system set.
        null_level = {}
        for k, v in list(pooled.items()):
            for s_ in list(v):
                if s_.startswith('_'):
                    if s_ == '_null_ref_split':
                        null_level[k] = v[s_][0]
                    del v[s_]
        if 'fmd' in metrics and 'fmd' not in pooled:
            # Not scored yet. Drop the panel rather than draw a row of
            # placeholders: the sheet is complete without it, and a
            # whole row of hatching reads as a failure, not "not yet".
            print(f'[warn] FMD not scored yet (no {fc}; eval_fmd.sbatch) '
                  '-- quality sheet drawn without it', file=sys.stderr)
            metrics = [m for m in metrics if m != 'fmd']
        # 1 = songs resampled, decodes held fixed; 2 = decodes resampled
        # too; one-per-song = decodes only. Two figures that differ only
        # in whisker length are otherwise indistinguishable, so the
        # legend has to say which; likewise the song/note weighting.
        modes = {r[5] for v in pooled.values() for r in v.values()}
        boot_mode = modes.pop() if len(modes) == 1 else 'mixed'
        weights = {r[6] for v in pooled.values() for r in v.values()}
        pooled_weight = weights.pop() if len(weights) == 1 else 'mixed'
        unknown = [m for m in metrics if m not in pooled]
        if unknown:
            print(f'[warn] not in the pooled CSV, drawn as pending: '
                  f'{" ".join(unknown)}', file=sys.stderr)

    per_song = {} if pooled_mode else read_per_song(args.csv, metrics)

    present = set()
    src = pooled if pooled_mode else per_song
    for m in metrics:
        present |= set(src.get(m, {}))
    excluded = {x for x in re.split(r'[,\s]+', args.exclude) if x}   # commas OR spaces: sbatch --export splits on commas
    order = [(s, d, sh, g, ranked and s not in UNRANKED)
             for g, ranked, members in GROUPS for s, d, sh in members
             if s not in excluded]
    if excluded:
        print(f'[exclude] left off the figure: {" ".join(sorted(excluded))}',
              file=sys.stderr)
    _split = lambda v: [x for x in re.split(r'[,\s]+', v or '') if x]
    if args.order:
        # a sheet-specific column order (internal names); anything not
        # named keeps its registry order after the named ones
        want = _split(args.order)
        by_name = {o[0]: o for o in order}
        bad = [w for w in want if w not in by_name]
        if bad:
            raise SystemExit(f'--order: unknown or excluded system(s) {" ".join(bad)}')
        order = [by_name[w] for w in want] + [o for o in order if o[0] not in want]
    if args.ranked is not None:
        # a sheet-specific ranked set, overriding the family flags and
        # UNRANKED: only these compete for the best-of line and star
        rk = set(_split(args.ranked))
        bad = rk - {o[0] for o in order}
        if bad:
            raise SystemExit(f'--ranked: unknown or excluded system(s) {" ".join(sorted(bad))}')
        order = [(s, d, sh, g, s in rk) for s, d, sh, g, _r in order]
    divider_at = None
    if args.divider_after:
        names_ = [o[0] for o in order]
        if args.divider_after not in names_:
            raise SystemExit(f'--divider-after: {args.divider_after} is not on the figure')
        divider_at = names_.index(args.divider_after) + 0.5
    known = {s for s, _, _, _, _ in order} | excluded
    extra = sorted(present - known)
    if extra:
        print(f'[warn] in the CSV but not in the figure: {" ".join(extra)}',
              file=sys.stderr)
    absent = [s for s, _, _, _, _ in order if s not in present]
    if absent:
        print(f'[warn] drawn as pending placeholders: {" ".join(absent)}',
              file=sys.stderr)

    n = len(metrics)
    if args.orient == 'h' and not pooled_mode:
        # ---- wide layout: rows of four, systems down the y-axis ----
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)
        width = args.width if args.width > 5 else 7.0
        row_h = 0.30 * len(order) + 0.5
        fig, axgrid = plt.subplots(nrows, ncols, squeeze=False,
                                   figsize=(width, row_h * nrows + 0.6))
        fig.patch.set_facecolor(SURFACE)
        ypos = None
        for idx, m in enumerate(metrics):
            ax = axgrid[idx // ncols][idx % ncols]
            letter = chr(ord('a') + idx) if n > 1 and idx < 26 else ''
            ypos = draw_panel_h(ax, m, per_song, order, present, letter)
            if idx % ncols == 0:
                _write_names_h(ax, order, ypos)
        for r in range(nrows):
            for c in range(ncols):
                if r * ncols + c >= n:
                    axgrid[r][c].set_visible(False)
        edges = []
        for i, (_s, _d, _sh, family, _r) in enumerate(order):
            if edges and edges[-1][0] == family:
                edges[-1] = (family, edges[-1][1], i)
            else:
                edges.append((family, i, i))
        # legend in two titled blocks: model group (colour) and marks
        marks = [
            Line2D([0], [0], color=INK, lw=0.9, ls=(0, (4, 3)),
                   label='reference (ground truth)'),
            Line2D([0], [0], color=INK_2, lw=1.2, marker='*', markersize=6.5,
                   markerfacecolor=SURFACE, markeredgecolor=INK,
                   markeredgewidth=0.5, label='best ranked system (mean)'),
            Line2D([0], [0], color=INK, lw=0, marker='D', markersize=2.8,
                   markerfacecolor=SURFACE, markeredgecolor=INK,
                   markeredgewidth=0.7,
                   label='mean (box: quartiles; whiskers: 1.5 IQR; no outliers)'),
        ]
        groups_h, groups_l = [], []
        for f, left, right in edges:
            cols = []
            for s_, _d, _sh, _g, _r in order[left:right + 1]:
                if color_of(s_, f) not in cols:
                    cols.append(color_of(s_, f))
            sw = tuple(Rectangle((0, 0), 1, 1, facecolor=lighten(c),
                                 edgecolor=c, lw=0.8) for c in cols)
            groups_h.append(sw[0] if len(sw) == 1 else sw)
            groups_l.append(f)
        grouped_legend(fig, [('Model group', groups_h, groups_l),
                             ('Marks', marks, [h.get_label() for h in marks])],
                       y=-0.005, xs=[0.14, 0.52], ncol=[2, 1])
        if args.title:
            fig.suptitle(args.title, fontsize=FS_LETTER, color=INK, y=0.995)
        leg_frac = 0.52 / fig.get_figheight()
        fig.tight_layout(rect=(0, leg_frac, 1, 1.0), w_pad=1.2, h_pad=1.6)
        for ext in ('pdf', 'png'):
            path = f'{args.out}.{ext}'
            fig.savefig(path, dpi=300, facecolor=SURFACE)
            print(f'wrote {path}')
        _print_numbers(metrics, order, per_song, pooled, pooled_mode, present)
        return

    if user_ncols is None and n % args.ncols:
        # A ragged grid leaves a hole and strands one column's system
        # names mid-figure. Take the widest of 3 / 2 / 1 that fills it.
        args.ncols = next(c for c in (3, 2, 1) if n % c == 0)
    ncols = max(1, min(args.ncols, n))
    nrows = math.ceil(n / ncols)
    # Below the panels: vertical system names (~1.5 in at 7 pt for the
    # longest) and the legend.
    short_names = args.names == 'short'
    # below the panels: vertical full names need ~1.5 in; two-line short
    # codes ~0.35 in; none leaves the legend alone, which then also lists
    # the systems (~0.85 in)
    extra = (0.8 if args.names == 'none' else
             1.0 if short_names else (2.25 if ncols > 1 else 1.6))
    fig, axgrid = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(args.width, args.panel_height * nrows + extra))
    fig.patch.set_facecolor(SURFACE)

    # ~10 characters per inch at 7.5pt; 0 means "use the y-axis label";
    # None means no metric name on the panel at all (--no-titles)
    title_chars = (None if args.no_titles else
                   int(args.width / ncols * 10) if ncols > 1 else 0)
    labels = None
    used = []
    for idx, m in enumerate(metrics):
        ax = axgrid[idx // ncols][idx % ncols]
        # bold panel letters, as in a journal figure; only where there
        # is more than one panel to point at
        letter = chr(ord('a') + idx) if n > 1 and idx < 26 else ''
        if pooled_mode:
            labels = draw_panel_pooled(ax, m, pooled, order, present,
                                       title_chars, null=null_level.get(m),
                                       letter=letter)
        else:
            labels = draw_panel(ax, m, per_song, order, present, title_chars,
                                letter=letter, overall=args.best_overall)
        if ncols >= 3:
            # narrow panels: three y ticks at most, a point smaller, so
            # the tick labels do not take half the panel's width
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.tick_params(axis='y', labelsize=FS_TICK - 1.0, pad=1.5)
        elif args.names == 'none':
            # the one-column sheet: y tick numbers a point smaller and
            # three or four of them, not a ladder
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.tick_params(axis='y', labelsize=FS_TICK - 1.0)
        if divider_at is not None:
            # a dashed rule between the ranked systems and the rest
            ax.axvline(divider_at, color=INK_2, lw=0.6, ls=(0, (3, 2.5)),
                       zorder=1)
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
    if args.names == 'none':
        pass                                  # the legend names the systems
    elif short_names:
        # --names short (2026-09-24, by request): the two-line short
        # codes horizontal under each column, so the panels keep the
        # full column width and the figure loses the name column
        for bottom in bottoms:
            bottom.set_xticklabels([sh for _lg, sh in labels],
                                   fontsize=FS_TICK - (1.5 if ncols > 1 else 0.5),
                                   color=INK, linespacing=0.95)
            bottom.tick_params(axis='x', pad=1.5)
    elif compact:
        shown = [lg.replace('\n', ' ') for lg, _sh in labels]
        for bottom in bottoms:
            bottom.set_xticklabels(shown, fontsize=FS_TICK, color=INK,
                                   rotation=90, ha='center', va='top')
    else:
        for bottom in bottoms:
            bottom.set_xticklabels([lg for lg, _sh in labels],
                                   fontsize=FS_TICK, color=INK)

    # family spans over the filtered order, so an excluded member does
    # not leave its family's bracket one column too wide
    edges = []
    for i, (_s, _d, _sh, family, _r) in enumerate(order):
        if edges and edges[-1][0] == family:
            edges[-1] = (family, edges[-1][1], i)
        else:
            edges.append((family, i, i))
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
                    va='top', fontsize=FS_TICK, color=INK, weight='bold',
                    annotation_clip=False)

    word = 'pooled JSD' if pooled_mode else 'mean'

    handles = [
        Line2D([0], [0], color=INK, lw=0.9, ls=(0, (4, 3)),
               label='reference (ground truth)'),
        Line2D([0], [0], color=INK_2, lw=1.2, marker='*', markersize=6.5,
               markerfacecolor=SURFACE, markeredgecolor=INK,
               markeredgewidth=0.5,
               label=f'best ranked ({word})'),
    ]
    if pooled_mode:
        handles += [
            Line2D([0], [0], color='none', marker='o', markersize=6.0,
                   markerfacecolor=MUTED, markeredgecolor=SURFACE,
                   markeredgewidth=1.0,
                   label=f'corpus-pooled JSD, weighted per '
                         f'{pooled_weight}'),
            Line2D([0], [0], color=MUTED, lw=1.0, ls=(0, (1, 1.5)),
                   label='noise floor: reference split-half null'),
            Line2D([0], [0], color=INK_2, lw=1.1, marker='_', markersize=7,
                   markeredgewidth=1.1,
                   label='95% bootstrap CI over ' + {
                       'songs': 'songs',
                       'songs+samples': 'songs and samples',
                       'one-per-song': 'decodes only (NOT over songs)',
                   }.get(boot_mode, boot_mode)),
        ]
    else:
        handles += [
            Line2D([0], [0], color=INK, lw=0, marker='D', markersize=2.8,
                   markerfacecolor=SURFACE, markeredgecolor=INK,
                   markeredgewidth=0.7,
                   label='mean; whiskers 1.5 IQR'),
        ]
        if args.best_overall:
            handles.insert(2, Line2D([0], [0], color='none', marker='*',
                                     markersize=6.5, markerfacecolor=SURFACE,
                                     markeredgecolor=INK_2, markeredgewidth=0.7,
                                     label='best of all columns (mean)'))
    labels_ = [h.get_label() for h in handles]
    if args.names == 'none':
        # one swatch per system, in its own shade, full name on one line
        for s, d, _sh, f, _r in order:
            c = color_of(s, f)
            handles.append(Rectangle((0, 0), 1, 1, facecolor=lighten(c),
                                     edgecolor=c, lw=0.8))
            labels_.append(d.replace('\n', ' '))
    elif compact:
        # one legend entry per family. If members carry their own
        # colours (SYSTEM_COLOR) the swatch is those side by side;
        # otherwise a single swatch in the family colour.
        for f, left, right in edges:
            cols = []
            for s, _d, _sh, _g, _r in order[left:right + 1]:
                if color_of(s, f) not in cols:
                    cols.append(color_of(s, f))
            sw = tuple(Rectangle((0, 0), 1, 1, facecolor=lighten(c),
                                 edgecolor=c, lw=0.8) for c in cols)
            handles.append(sw[0] if len(sw) == 1 else sw)
            labels_.append(f)
    # Two legend columns fit a 3.4 in figure at 7 pt; three would not.
    fig.legend(handles=handles, labels=labels_, loc='lower center',
               ncol=2 if args.width < 5 else 3, frameon=False,
               # the system-swatch legend (--names none) one point smaller
               fontsize=FS_LEGEND - (1.0 if args.names == 'none' else 0.0),
               labelcolor=INK,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.2,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)},
               bbox_to_anchor=(0.5, 0.004))
    if args.title:
        fig.suptitle(args.title, fontsize=FS_LETTER, color=INK, y=0.995)

    # The legend's share of the height, so it never overlaps the names.
    n_leg = math.ceil(len(handles) / (2 if args.width < 5 else 3))
    row_h = 0.13 if args.names == 'none' else 0.15
    leg_frac = (row_h * n_leg + 0.06) / fig.get_figheight()
    fig.tight_layout(rect=(0, leg_frac, 1, 1.0), h_pad=1.0,
                     w_pad=0.4 if ncols >= 3 else 0.8)
    for ext in ('pdf', 'png'):
        path = f'{args.out}.{ext}'
        fig.savefig(path, dpi=300, facecolor=SURFACE)
        print(f'wrote {path}')

    _print_numbers(metrics, order, per_song, pooled, pooled_mode, present)


def _print_numbers(metrics, order, per_song, pooled, pooled_mode, present):
    # The numbers behind the picture, so a reader of the log can check a
    # mark without opening the figure.
    if pooled_mode:
        print('\ncorpus-pooled JSD [2.5, 97.5] bootstrap over songs')
        print('(a single value per system, so there is no std; boot_se is '
              'the spread of the replicates)')
        for m in metrics:
            print(f'  {m}')
            for sysname, disp, _sh, _g, ranked in order:
                rec = pooled.get(m, {}).get(sysname)
                if rec is None or math.isnan(rec[0]):
                    print(f'    {sysname:11s} --')
                    continue
                v, lo, hi, ns, no, _bl, _w = rec
                mark = '' if ranked else '   (not ranked)'
                print(f'    {sysname:11s} {v:.4f} [{lo:.4f},{hi:.4f}]'
                      f'  n={ns} songs / {no} obs{mark}')
    else:
        print('\nper-song n and median, reference level in brackets')
        for m in metrics:
            b = best_ranked(m, per_song, order, present)
            won = f'  best(ranked): {b[0]} mean {b[3]:+.4f}' if b else ''
            bo = best_ranked(m, per_song, [(s_, d_, sh_, g_, True) for s_, d_, sh_, g_, _r in order], present)
            if bo:
                won += f'  best(all): {bo[0]} mean {bo[3]:+.4f}'
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
