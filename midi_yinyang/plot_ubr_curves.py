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
  black    the ground-truth continuations of the same songs: the target,
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
from plot_e1_box import (GROUPS, FAMILY_COLOR, SURFACE, INK, INK_2,  # noqa
                         MUTED)

PANELS = [(1, 'state', '1-beat (Full)'), (2, 'state', '2-beat (Full)'),
          (1, 'onset', '1-beat (Onset)'), (2, 'onset', '2-beat (Onset)')]
STREAMS = [('a', 'Melody'), ('b', 'Chord')]
# dash per member within a family, so nine systems on three hues stay
# tellable apart without a fourth hue
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
                             figsize=(args.width, 1.55 * len(STREAMS) + 1.2),
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    handles = {}
    for r, (st, sname) in enumerate(STREAMS):
        for c, (beats, mode, title) in enumerate(PANELS):
            ax = axes[r][c]
            key = (st, beats, mode)
            # reference first, underneath
            m, se, n = mean_se(ref[key])
            if m is not None:
                x = np.arange(1, len(m) + 1) * beats
                ax.plot(x, m, color=INK, lw=1.6, ls=(0, (4, 2)), zorder=4)
                handles['_ref'] = Line2D([0], [0], color=INK, lw=1.6,
                                         ls=(0, (4, 2)),
                                         label=f'ground truth (n={n} songs)')
            for sysname, disp, family, i in order:
                if sysname not in gen[key]:
                    continue
                m, se, n = mean_se(gen[key][sysname])
                if m is None:
                    continue
                x = np.arange(1, len(m) + 1) * beats
                col = FAMILY_COLOR[family]
                ls = MEMBER_DASH[i % len(MEMBER_DASH)]
                ax.plot(x, m, color=col, lw=1.4, ls=ls, zorder=3)
                if not args.no_bands:
                    ax.fill_between(x, m - 1.96 * se, m + 1.96 * se,
                                    color=col, alpha=0.10, lw=0, zorder=2)
                handles[sysname] = Line2D(
                    [0], [0], color=col, lw=1.4, ls=ls,
                    label=disp.replace('\n', ' '))
            if r == 0:
                ax.set_title(title, fontsize=7.5, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(f'{sname}\nunique beat ratio', fontsize=7,
                              color=INK)
            if r == len(STREAMS) - 1:
                ax.set_xlabel('phrase length (beats)', fontsize=7,
                              color=INK_2)
            ax.set_ylim(0, 1.02)
            ax.tick_params(labelsize=6.3, colors=INK_2, length=2)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(MUTED)
                ax.spines[side].set_linewidth(0.8)
            ax.set_facecolor(SURFACE)
            ax.grid(color=MUTED, alpha=0.2, lw=0.6)
            ax.set_axisbelow(True)

    ordered = [handles[s] for s, _, _, _ in order if s in handles]
    if '_ref' in handles:
        ordered.append(handles['_ref'])
    fig.legend(handles=ordered, loc='lower center', ncol=5, frameon=False,
               fontsize=6.4, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.115 if not args.no_bands else 0.10,
             'lines: mean over songs, samples averaged within song'
             + ('; bands: $\\pm$1.96 SE over songs' if not args.no_bands
                else ''),
             ha='center', fontsize=6.0, color=INK_2)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
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
