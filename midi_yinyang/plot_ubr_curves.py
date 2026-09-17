"""Unique beat ratio against phrase length, one curve per system.

The figure from BEAT (Qian et al.) sec. 4.5: cut the continuation into
beat intervals, and at each phrase length L report the fraction of the
first L intervals whose content had not appeared before. Real music
decays -- phrases come back -- and a model can miss on either side,
staying near 1 (never restates anything) or collapsing (loops). The
box plot in the repetition sheet shows only the endpoint at the last
beat; the SHAPE of the decay is the finding, so this draws the curve.

Layout, matching the paper: four columns -- 1-beat and 2-beat intervals,
each under the "full" criterion (everything sounding) and the "onset"
criterion (only what starts) -- and here one row per stream, since the
melody and the chord stream repeat on very different scales.

Marks:
  line     mean over SONGS of the per-song curve (a song's samples are
           averaged first), colour = family, dash = member within family
  band     +- 1.96 standard errors over songs, so two curves whose bands
           miss each other differ at about the 95% level at that L
  grey     the ground-truth continuations of the same songs: the target,
  dashed   not a ceiling -- above it is too scattered, below it too
           repetitive

Reads the same manifest and reference dirs as eval_metrics, so the
curves are over exactly the scored set.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_metrics as em                      # noqa: E402
from plot_e1_box import (GROUPS, SURFACE, INK, INK_2, MUTED, GRID,  # noqa
                         FS_TICK, FS_LABEL, FS_TITLE, FS_LETTER,
                         FS_LEGEND, color_of)

PANELS = [(1, 'state', '1-beat (Full)'), (2, 'state', '2-beat (Full)'),
          (1, 'onset', '1-beat (Onset)'), (2, 'onset', '2-beat (Onset)')]
STREAMS = [('a', 'Melody'), ('b', 'Chord')]
# colour is the FAMILY (plot_e1_box palette); the dash per member
# within a family tells its systems apart, in greyscale too
MEMBER_DASH = ['-', (0, (5, 2)), (0, (1, 1.2))]


def curve(s, beats, mode):
    """Cumulative unique ratio at every prefix length, in intervals."""
    sigs = em._beat_sigs(s, beats, mode)
    seen, out = set(), np.empty(len(sigs))
    for i, sig in enumerate(sigs):
        seen.add(sig)
        out[i] = len(seen) / (i + 1)
    return out


def collect(args):
    """{(stream, beats, mode): {system: {song: [curve, ...]}}}, plus ref."""
    lo, hi = args.prompt_frames, args.total_frames
    gen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ref = defaultdict(dict)
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            system, _mode, song, _sample, path = line.split('\t')[:5]
            try:
                ga, gb = em.load_streams(path.split(';'), args.task,
                                         args.mel_programs,
                                         args.chord_programs, hi)
            except Exception as e:
                print(f'  [fail] {path}: {e!r}', file=sys.stderr)
                continue
            streams = {'a': ga.slice(lo, hi), 'b': gb.slice(lo, hi)}
            for st, _ in STREAMS:
                for beats, mode, _t in PANELS:
                    gen[(st, beats, mode)][system][song].append(
                        curve(streams[st], beats, mode))
            if song not in ref[('a', 1, 'state')]:
                ra, rb = em.load_streams(
                    [os.path.join(args.ref_a_dir, f'{song}.mid'),
                     os.path.join(args.ref_b_dir, f'{song}.mid')],
                    args.task, args.mel_programs, args.chord_programs, hi)
                rs = {'a': ra.slice(lo, hi), 'b': rb.slice(lo, hi)}
                for st, _ in STREAMS:
                    for beats, mode, _t in PANELS:
                        ref[(st, beats, mode)][song] = curve(rs[st], beats,
                                                             mode)
    return gen, ref


def mean_se_delta(gen_per_song, ref_per_song):
    """Mean and SE over songs of (system - ground truth), PAIRED per song.

    Differencing within a song before averaging removes the song's own
    level from the comparison, so the band is the uncertainty of the
    deviation itself -- much tighter than differencing two marginal
    means, and the right quantity when the question is "how far from
    real music" rather than "how high".
    """
    rows = []
    for song, curves in gen_per_song.items():
        if song not in ref_per_song:
            continue
        r = ref_per_song[song]
        n = min(min(len(c) for c in curves), len(r))
        if n == 0:
            continue
        rows.append(np.mean([c[:n] - r[:n] for c in curves], axis=0))
    if not rows:
        return None, None, 0
    n = min(len(x) for x in rows)
    M = np.stack([x[:n] for x in rows])
    return M.mean(axis=0), M.std(axis=0, ddof=1) / np.sqrt(len(M)), len(M)


def mean_se(per_song):
    """Mean and SE over songs of per-song curves (samples averaged)."""
    rows = []
    for song, curves in per_song.items():
        cs = curves if isinstance(curves, list) else [curves]
        n = min(len(c) for c in cs)
        if n == 0:
            continue
        rows.append(np.mean([c[:n] for c in cs], axis=0))
    if not rows:
        return None, None, 0
    n = min(len(r) for r in rows)
    M = np.stack([r[:n] for r in rows])
    return M.mean(axis=0), M.std(axis=0, ddof=1) / np.sqrt(len(M)), len(M)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ref-a-dir', required=True)
    p.add_argument('--ref-b-dir', required=True)
    p.add_argument('--out', required=True, help='path WITHOUT extension')
    p.add_argument('--task', default='melchord')
    p.add_argument('--prompt-frames', type=int, default=96)
    p.add_argument('--total-frames', type=int, default=416)
    p.add_argument('--mel-programs', default='0,24')
    p.add_argument('--chord-programs', default='48')
    p.add_argument('--width', type=float, default=7.0)
    p.add_argument('--no-bands', action='store_true')
    p.add_argument('--exclude', default='',
                   help='comma-separated system names to leave off')
    p.add_argument('--y', choices=['ratio', 'delta'], default='ratio',
                   help="'ratio' (default, by request 2026-09-16): the "
                        "unique beat ratio itself, ground truth as a dashed "
                        "curve, each row's axis tight to its data. 'delta': "
                        "system minus ground truth, paired per song, so the "
                        "target is the zero line.")
    args = p.parse_args()
    args.mel_programs = {int(x) for x in args.mel_programs.split(',')}
    args.chord_programs = {int(x) for x in args.chord_programs.split(',')}

    gen, ref = collect(args)
    excluded = {x.strip() for x in args.exclude.split(',') if x.strip()}
    order = [(s, d, g, i)
             for g, _r, members in GROUPS
             for i, (s, d, _sh) in enumerate(members) if s not in excluded]
    present = {s for v in gen.values() for s in v}
    absent = [s for s, _, _, _ in order if s not in present]
    if absent:
        print(f'[warn] not in the manifest, not drawn: {" ".join(absent)}',
              file=sys.stderr)

    fig, axes = plt.subplots(len(STREAMS), len(PANELS), squeeze=False,
                             figsize=(args.width, 1.55 * len(STREAMS) + 1.35),
                             sharey='row')
    row_lims = {r: (np.inf, -np.inf) for r in range(len(STREAMS))}
    fig.patch.set_facecolor(SURFACE)
    handles = {}
    for r, (st, sname) in enumerate(STREAMS):
        for c, (beats, mode, title) in enumerate(PANELS):
            ax = axes[r][c]
            key = (st, beats, mode)
            ymin, ymax = np.inf, -np.inf
            if args.y == 'ratio':
                # reference first, underneath
                m, se, n = mean_se(ref[key])
                if m is not None:
                    x = np.arange(1, len(m) + 1) * beats
                    ax.plot(x, m, color=INK_2, lw=1.0, ls=(0, (4, 2)), zorder=4)
                    ymin, ymax = min(ymin, m.min()), max(ymax, m.max())
                    handles['_ref'] = Line2D(
                        [0], [0], color=INK_2, lw=1.0, ls=(0, (4, 2)),
                        label=f'ground truth (n={n} songs)')
            else:
                # the target IS the zero line
                ax.axhline(0, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=4)
                n_ref = len(ref[key])
                handles['_ref'] = Line2D(
                    [0], [0], color=INK_2, lw=0.9, ls=(0, (4, 2)),
                    label=f'ground truth (n={n_ref} songs)')
            for sysname, disp, family, i in order:
                if sysname not in gen[key]:
                    continue
                if args.y == 'ratio':
                    m, se, n = mean_se(gen[key][sysname])
                else:
                    m, se, n = mean_se_delta(gen[key][sysname], ref[key])
                if m is None:
                    continue
                x = np.arange(1, len(m) + 1) * beats
                col = color_of(sysname, family)
                ls = MEMBER_DASH[i % len(MEMBER_DASH)]
                ax.plot(x, m, color=col, lw=0.9, ls=ls, zorder=3)
                lo_b, hi_b = m - 1.96 * se, m + 1.96 * se
                if not args.no_bands:
                    ax.fill_between(x, lo_b, hi_b, color=col, alpha=0.16,
                                    lw=0, zorder=2)
                # the first few beats swing wildly (1/1, 1/2, 2/3 ...)
                # and would set the scale for the whole panel; scale to
                # the settled part
                k = max(4, len(m) // 10)
                ymin = min(ymin, lo_b[k:].min())
                ymax = max(ymax, hi_b[k:].max())
                handles[sysname] = Line2D(
                    [0], [0], color=col, lw=0.9, ls=ls,
                    label=disp.replace('\n', ' '))
            if np.isfinite(ymin) and np.isfinite(ymax):
                pad = 0.08 * (ymax - ymin) or 0.01
                lo_y, hi_y = ymin - pad, ymax + pad
                if args.y == 'delta':
                    lo_y, hi_y = min(lo_y, -pad), max(hi_y, pad)
                row_lims[r] = (min(row_lims[r][0], lo_y),
                               max(row_lims[r][1], hi_y))
            if r == 0:
                ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(f'{sname}\n' + ('UBR $-$ ground truth'
                                               if args.y == 'delta'
                                               else 'unique beat ratio'),
                              fontsize=FS_LABEL, color=INK)
            if r == len(STREAMS) - 1:
                ax.set_xlabel('phrase length (beats)', fontsize=FS_LABEL,
                              color=INK)

            ax.tick_params(labelsize=FS_TICK, colors=INK, length=2.5,
                           width=0.6, color=INK)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(INK)
                ax.spines[side].set_linewidth(0.6)
            ax.set_facecolor(SURFACE)
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.annotate(chr(ord('a') + r * len(PANELS) + c),
                        xy=(0.02, 0.96), xycoords='axes fraction',
                        ha='left', va='top', fontsize=FS_LETTER,
                        weight='bold', color=INK)

    # one y-range per row, tight to the data: melody and chord live on
    # different scales and a shared 0..1 axis hid the differences
    for r in range(len(STREAMS)):
        lo_y, hi_y = row_lims[r]
        if np.isfinite(lo_y) and np.isfinite(hi_y):
            axes[r][0].set_ylim(lo_y, hi_y)
    ordered = [handles[s] for s, _, _, _ in order if s in handles]
    if '_ref' in handles:
        ordered.append(handles['_ref'])
    fig.legend(handles=ordered, loc='lower center', ncol=4, frameon=False,
               fontsize=FS_LEGEND, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    what = ('system minus ground truth, paired per song' if args.y == 'delta'
            else 'mean over songs, samples averaged within song')
    fig.text(0.5, 0.115 if not args.no_bands else 0.10,
             'lines: ' + what
             + ('; bands: $\\pm$1.96 SE over songs' if not args.no_bands
                else ''),
             ha='center', fontsize=FS_TICK, color=INK_2)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')

    # The numbers a write-up quotes: each system's ratio at the last
    # beat, its deviation from ground truth there, and the range of that
    # deviation across the eight panels -- the BEAT paper's "within x-y%
    # of ground truth" framing, read off the log rather than the figure.
    print('\nunique beat ratio at the last beat: ratio (deviation from '
          'ground truth, % of ground truth)')
    span = {}
    for st, sname in STREAMS:
        for beats, mode, title in PANELS:
            key = (st, beats, mode)
            gm, _, n = mean_se(ref[key])
            if gm is None:
                continue
            gt = gm[-1]
            line = f'  {sname:6s} {title:15s} GT {gt:.3f} |'
            for sysname, _, _, _ in order:
                if sysname not in gen[key]:
                    continue
                mm, _, _ = mean_se(gen[key][sysname])
                if mm is None:
                    continue
                d = mm[-1] - gt
                rel = 100 * d / gt if gt else float('nan')
                line += f' {sysname} {mm[-1]:.3f} ({d:+.3f}, {rel:+.0f}%)'
                span.setdefault(sysname, []).append(rel)
            print(line)
    print('\nrange of relative deviation across all panels (min .. max, %):')
    for sysname, _, _, _ in order:
        if sysname in span:
            v = span[sysname]
            print(f'  {sysname:11s} {min(v):+6.1f} .. {max(v):+6.1f}'
                  f'   (|max| {max(abs(x) for x in v):.1f}%)')


if __name__ == '__main__':
    main()
