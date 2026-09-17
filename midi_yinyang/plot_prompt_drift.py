"""Prompt drift across the continuation: pitch-class JSD against the
prompt, bar by bar, for every system and for the ground truth.

The prompt-adherence row of the E1 sheets is ONE number per song: the
duration-weighted pitch-class histogram of the whole continuation
against the prompt's, as a JSD, minus the same for the ground-truth
continuation. This figure spreads that number over TIME. For bar i of
the continuation (frames prompt + 16 i .. prompt + 16 (i + 1)) it
computes

    jsd_i   = JSD( pc(prompt), pc(bar i of the generation) )
    delta_i = jsd_i - JSD( pc(prompt), pc(bar i of the ground truth) )

with pc() the duration-weighted 12-bin pitch-class histogram used by
eval_metrics, so the two panels answer, per bar: how far has the
generation moved from the prompt's pitch material, and is that more or
less than the real song had moved by the same bar. --window cumulative
uses bars 0..i instead of bar i alone, i.e. the whole continuation so
far, which is smoother and reads as "drift of the piece to date".

Layout: one row per stream (melody, chord), two columns: the JSD itself
with the ground truth as a dashed grey curve, and the paired delta with
zero as the target. Lines are means over songs (samples averaged within
song), bands +-1.96 SE over songs. Colours are the E1 families, the
dash tells systems of one family apart, legend in titled blocks.

Reads the same manifest and reference dirs as eval_metrics, so the
curves are over exactly the scored set.

Usage (via plot_prompt_drift.sbatch):
  python plot_prompt_drift.py --manifest results/E1_p80_v5b92_manifest.tsv \
      --ref-a-dir temp/E1_v5b/prompts/mel --ref-b-dir temp/E1_v5b/prompts/chord \
      --prompt-frames 80 --total-frames 416 --out results/E1_p80_v5b92_drift
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
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_metrics as em                                   # noqa: E402
from plot_e1_box import (GROUPS, SURFACE, INK, INK_2, GRID,  # noqa: E402
                         FS_TICK, FS_LABEL, FS_TITLE, FS_LETTER,
                         color_of, grouped_legend)

STREAMS = [('a', 'Melody'), ('b', 'Chord')]
MEMBER_DASH = ['-', (0, (5, 2)), (0, (1, 1.2))]
BAR = em.FRAMES_PER_BAR


def pc_jsd(prompt_hist, seg):
    h = em._pc_profile_dw(seg)
    if h.sum() <= 0 or prompt_hist.sum() <= 0:
        return float('nan')
    return em.jsd(prompt_hist, h)


def curve(prompt, cont, window, bars):
    """One value per window of the continuation.

    window 'bar': window i = bars [i*bars, (i+1)*bars);
    window 'cumulative': window i = bars [0, (i+1)*bars)."""
    hp = em._pc_profile_dw(prompt)
    n_windows = cont.n_frames // (BAR * bars)
    out = np.full(n_windows, np.nan)
    for i in range(n_windows):
        lo = 0 if window == 'cumulative' else i * BAR * bars
        hi = (i + 1) * BAR * bars
        out[i] = pc_jsd(hp, cont.slice(lo, hi))
    return out


def collect(args):
    lo, hi = args.prompt_frames, args.total_frames
    gen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ref = defaultdict(dict)
    refs = {}          # song -> (a, b) reference streams, loaded once
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            system, _mode, song, _sample, path = line.split('\t')[:5]
            if song not in refs:
                refs[song] = em.load_streams(
                    [os.path.join(args.ref_a_dir, f'{song}.mid'),
                     os.path.join(args.ref_b_dir, f'{song}.mid')],
                    args.task, args.mel_programs, args.chord_programs, hi)
                for st, s_ in zip('ab', refs[song]):
                    ref[st][song] = curve(s_.slice(0, lo), s_.slice(lo, hi),
                                          args.window, args.bars)
            try:
                ga, gb = em.load_streams(path.split(';'), args.task,
                                         args.mel_programs,
                                         args.chord_programs, hi)
            except Exception as e:
                print(f'  [fail] {path}: {e!r}', file=sys.stderr)
                continue
            # The PROMPT is the reference's opening, identical for every
            # system by construction; it is read from the reference so a
            # system that re-renders its prompt cannot move the yardstick.
            for st, s_, r_ in zip('ab', (ga, gb), refs[song]):
                gen[st][system][song].append(
                    curve(r_.slice(0, lo), s_.slice(lo, hi),
                          args.window, args.bars))
    return gen, ref


def _stack(per_song, ref_per_song=None):
    """Rows = songs, samples averaged; optionally paired minus ref."""
    rows = []
    for song, curves in per_song.items():
        cs = np.array(curves if isinstance(curves, list) else [curves])
        m = np.nanmean(cs, axis=0)
        if ref_per_song is not None:
            if song not in ref_per_song:
                continue
            m = m - ref_per_song[song][:len(m)]
        rows.append(m)
    if not rows:
        return None, None, 0
    n = min(len(r) for r in rows)
    M = np.stack([r[:n] for r in rows])
    cnt = np.sum(~np.isnan(M), axis=0)
    mean = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(cnt, 1))
    return mean, se, len(M)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ref-a-dir', required=True)
    p.add_argument('--ref-b-dir', required=True)
    p.add_argument('--out', required=True, help='path WITHOUT extension')
    p.add_argument('--task', default='melchord')
    p.add_argument('--prompt-frames', type=int, default=80)
    p.add_argument('--total-frames', type=int, default=416)
    p.add_argument('--mel-programs', default='0,24')
    p.add_argument('--chord-programs', default='48')
    p.add_argument('--window', choices=['bar', 'cumulative'], default='bar')
    p.add_argument('--bars', type=int, default=1, help='bars per window')
    p.add_argument('--width', type=float, default=7.0)
    p.add_argument('--no-bands', action='store_true')
    p.add_argument('--exclude', default='', help='comma-separated systems')
    args = p.parse_args()
    args.mel_programs = {int(x) for x in args.mel_programs.split(',')}
    args.chord_programs = {int(x) for x in args.chord_programs.split(',')}

    gen, ref = collect(args)
    excluded = {x.strip() for x in args.exclude.split(',') if x.strip()}
    order = [(s, d, g, i) for g, _r, members in GROUPS
             for i, (s, d, _sh) in enumerate(members) if s not in excluded]
    present = {s for st in gen for s in gen[st]}
    absent = [s for s, _, _, _ in order if s not in present]
    if absent:
        print(f'[warn] not in the manifest, not drawn: {" ".join(absent)}',
              file=sys.stderr)

    unit = 'bar' if args.bars == 1 else f'{args.bars} bars'
    fig, axes = plt.subplots(len(STREAMS), 2, squeeze=False,
                             figsize=(args.width, 1.7 * len(STREAMS) + 1.1))
    fig.patch.set_facecolor(SURFACE)
    handles = {}
    table = {}
    for r, (st, sname) in enumerate(STREAMS):
        for c in range(2):
            ax = axes[r][c]
            if c == 0:
                m, se, n = _stack(ref[st])
                if m is not None:
                    x = np.arange(1, len(m) + 1) * args.bars
                    ax.plot(x, m, color=INK_2, lw=1.0, ls=(0, (4, 2)), zorder=4)
                    handles['_ref'] = Line2D([0], [0], color=INK_2, lw=1.0,
                                             ls=(0, (4, 2)),
                                             label=f'ground truth (n={n} songs)')
            else:
                ax.axhline(0, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=4)
            for sysname, disp, family, i in order:
                if sysname not in gen[st]:
                    continue
                m, se, n = (_stack(gen[st][sysname]) if c == 0
                            else _stack(gen[st][sysname], ref[st]))
                if m is None:
                    continue
                x = np.arange(1, len(m) + 1) * args.bars
                col = color_of(sysname, family)
                ls = MEMBER_DASH[i % len(MEMBER_DASH)]
                ax.plot(x, m, color=col, lw=0.9, ls=ls, zorder=3)
                if not args.no_bands:
                    ax.fill_between(x, m - 1.96 * se, m + 1.96 * se,
                                    color=col, alpha=0.14, lw=0, zorder=2)
                handles[sysname] = Line2D([0], [0], color=col, lw=0.9, ls=ls,
                                          label=disp.replace('\n', ' '))
                table[(st, c, sysname)] = m
            ax.set_title(('JSD to prompt' if c == 0 else
                          'JSD to prompt $-$ ground truth') + f', {sname.lower()}',
                         fontsize=FS_TITLE, color=INK, pad=4)
            if r == len(STREAMS) - 1:
                ax.set_xlabel(f'continuation position ({unit}s)' if args.bars == 1
                              else f'continuation position (bars)',
                              fontsize=FS_LABEL, color=INK)
            ax.tick_params(labelsize=FS_TICK, colors=INK, length=2.5,
                           width=0.6, color=INK)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(INK); ax.spines[side].set_linewidth(0.6)
            ax.set_facecolor(SURFACE)
            ax.yaxis.grid(True, color=GRID, lw=0.5); ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.annotate(chr(ord('a') + r * 2 + c), xy=(0.02, 0.96),
                        xycoords='axes fraction', ha='left', va='top',
                        fontsize=FS_LETTER, weight='bold', color=INK)

    blocks = []
    for title, fams in (('Ours', ('Ours',)),
                        ('Baselines', ('Internal baselines',
                                       'External baselines', 'Not ranked')),
                        ('Reference', ('_ref',))):
        if title == 'Reference':
            hs = [handles['_ref']] if '_ref' in handles else []
        else:
            hs = [handles[s] for s, _d, g, _i in order
                  if g in fams and s in handles]
        if hs:
            blocks.append((title, hs, [h.get_label() for h in hs]))
    grouped_legend(fig, blocks, y=0.0, xs=[0.08, 0.38, 0.78])
    fig.text(0.5, 0.135,
             f'lines: mean over songs, samples averaged within song; '
             f'window: {args.window}, {unit}'
             + ('' if args.no_bands else '; bands: $\\pm$1.96 SE over songs'),
             ha='center', fontsize=FS_TICK, color=INK_2)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.5, h_pad=1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')

    # numbers: first / middle / last third of the continuation
    print(f'\nJSD to prompt (window={args.window}, {unit}); thirds of the '
          f'continuation, then the delta against the ground truth')
    for st, sname in STREAMS:
        print(f'  {sname}')
        for sysname, disp, _g, _i in order:
            raw = table.get((st, 0, sysname)); dl = table.get((st, 1, sysname))
            if raw is None:
                continue
            k = len(raw) // 3 or 1
            fmt = lambda v: ' '.join(f'{np.nanmean(v[j*k:(j+1)*k]):.3f}'
                                     for j in range(3))
            print(f'    {sysname:11s} jsd {fmt(raw)}   delta {fmt(dl)}')
        m, _se, _n = _stack(ref[st])
        if m is not None:
            k = len(m) // 3 or 1
            print(f'    {"GT":11s} jsd ' + ' '.join(
                f'{np.nanmean(m[j*k:(j+1)*k]):.3f}' for j in range(3)))


if __name__ == '__main__':
    main()
