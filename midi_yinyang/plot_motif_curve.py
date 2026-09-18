"""Developedness against motif length, in three NESTED layers: for
motifs of 3..8 notes, how many of the melody continuation's motifs
relate to a prompt motif under an equivalence that loosens layer by
layer,

    layer 1  copied       the same pitches occur in the prompt
    layer 2  + transposed the same intervals occur (includes layer 1)
    layer 3  + transformed the inversion, the retrograde or only the
                          contour occurs (includes layers 1 and 2)

so the index can only grow from layer to layer, and the increment from
one layer to the next is what that device adds. The unit is ABSOLUTE:
motifs per continuation (--unit count, default), so a system that
writes more notes is not normalised down; --unit share divides by the
continuation's motif count. One panel per layer, motif length in NOTES
on x (a motif of m notes is m - 1 intervals), one curve per system, the
ground truth dashed: the yardstick for how much a real song restates,
transposes and transforms its opening. --y delta draws system minus
ground truth, paired per song, with zero as the target. Melody only.

--exclusive restores the earlier exclusive classification (each motif
in the strictest class only), which is the P2 share breakdown of
eval_metrics at every length.

Usage (via plot_motif_curve.sbatch):
  python plot_motif_curve.py --manifest results/E1_p80_v5b92_manifest.tsv \
      --ref-a-dir temp/E1_v5b/prompts/mel --ref-b-dir temp/E1_v5b/prompts/chord \
      --prompt-frames 80 --total-frames 416 --out results/E1_p80_v5b92_motif_curve
"""

import argparse
import re
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_metrics as em                                   # noqa: E402
from plot_e1_box import (GROUPS, SURFACE, INK, INK_2, GRID,  # noqa: E402
                         FS_TICK, FS_LABEL, FS_TITLE, FS_LETTER,
                         color_of, grouped_legend)

LAYERS_NESTED = [('exact', 'Layer 1: copied'),
                 ('transp', 'Layer 2: + transposed'),
                 ('transf', 'Layer 3: + transformed')]
LAYERS_EXCL = [('exact', 'Exact copy'), ('transp', 'Transposed'),
               ('transf', 'Transformed')]
CLASSES = ('exact', 'transp', 'inv', 'contour', 'novel')
MEMBER_DASH = ['-', (0, (5, 2)), (0, (1, 1.2))]
UNIT = 'count'        # set from --unit
NESTED = True         # set from --exclusive


def shares(prompt_line, line, n):
    """Class shares of the line's interval n-grams vs the prompt line."""
    p_iv = [b[1] - a[1] for a, b in zip(prompt_line, prompt_line[1:])]
    iv = [b[1] - a[1] for a, b in zip(line, line[1:])]
    if len(iv) < n or len(p_iv) < n:
        return None
    pitch_p = {tuple(x[1] for x in prompt_line[i:i + n + 1])
               for i in range(len(prompt_line) - n)}
    iv_p = set(em._grams(p_iv, n))
    inv_p = (set(em._grams(em._transformed(p_iv, 'inv'), n))
             | set(em._grams(em._transformed(p_iv, 'retro'), n)))
    ct_p = set(em._grams(em._transformed(p_iv, 'contour'), n))
    counts = dict.fromkeys(CLASSES, 0)
    total = 0
    for i in range(len(iv) - n + 1):
        total += 1
        pg = tuple(x[1] for x in line[i:i + n + 1])
        ig = tuple(iv[i:i + n])
        if pg in pitch_p:
            counts['exact'] += 1
        elif ig in iv_p:
            counts['transp'] += 1
        elif ig in inv_p:
            counts['inv'] += 1
        elif tuple(int(np.sign(v)) for v in ig) in ct_p:
            counts['contour'] += 1
        else:
            counts['novel'] += 1
    scale = 1.0 if UNIT == 'count' else 1.0 / total
    out = {k: c * scale for k, c in counts.items()}
    out['total'] = total
    if NESTED:
        # cumulative: each layer contains the ones before it
        out['transp'] = out['exact'] + out['transp']
        out['transf'] = out['transp'] + out['inv'] + out['contour']
    else:
        out['transf'] = out['inv'] + out['contour']
    return out


def collect(args):
    lo, hi = args.prompt_frames, args.total_frames
    lengths = list(range(args.min_notes, args.max_notes + 1))
    gen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ref = defaultdict(dict)                      # (layer) -> song -> curve
    refs = {}
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            system, _mode, song, _sample, path = line.split('\t')[:5]
            if song not in refs:
                ra, _rb = em.load_streams(
                    [os.path.join(args.ref_a_dir, f'{song}.mid'),
                     os.path.join(args.ref_b_dir, f'{song}.mid')],
                    args.task, args.mel_programs, args.chord_programs, hi)
                p_line = em._line(ra.slice(0, lo))
                refs[song] = (ra, p_line)
                r_line = em._line(ra.slice(lo, hi))
                for lay in CLASSES + ('transf', 'total'):
                    ref[lay][song] = np.array([
                        (shares(p_line, r_line, m - 1) or {}).get(lay, np.nan)
                        for m in lengths])
            try:
                ga, _gb = em.load_streams(path.split(';'), args.task,
                                          args.mel_programs,
                                          args.chord_programs, hi)
            except Exception as e:
                print(f'  [fail] {path}: {e!r}', file=sys.stderr)
                continue
            _ra, p_line = refs[song]
            g_line = em._line(ga.slice(lo, hi))
            for lay in CLASSES + ('transf', 'total'):
                gen[lay][system][song].append(np.array([
                    (shares(p_line, g_line, m - 1) or {}).get(lay, np.nan)
                    for m in lengths]))
    return gen, ref, lengths


def _stack(per_song, ref_per_song=None):
    rows = []
    for song, curves in per_song.items():
        cs = np.array(curves if isinstance(curves, list) else [curves])
        m = np.nanmean(cs, axis=0)
        if ref_per_song is not None:
            if song not in ref_per_song:
                continue
            m = m - ref_per_song[song]
        rows.append(m)
    if not rows:
        return None, None, 0
    M = np.stack(rows)
    cnt = np.sum(~np.isnan(M), axis=0)
    return (np.nanmean(M, axis=0),
            np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(cnt, 1)),
            len(M))


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
    p.add_argument('--min-notes', type=int, default=3)
    p.add_argument('--max-notes', type=int, default=8)
    p.add_argument('--y', choices=['raw', 'delta'], default='raw')
    p.add_argument('--unit', choices=['count', 'share'], default='count',
                   help='count: motifs per continuation (default); share: '
                        'fraction of the continuation\'s motifs')
    p.add_argument('--exclusive', action='store_true',
                   help='exclusive classes instead of nested layers')
    p.add_argument('--width', type=float, default=7.0)
    p.add_argument('--no-bands', action='store_true')
    p.add_argument('--exclude', default='')
    args = p.parse_args()
    args.mel_programs = {int(x) for x in args.mel_programs.split(',')}
    args.chord_programs = {int(x) for x in args.chord_programs.split(',')}

    global UNIT, NESTED
    UNIT, NESTED = args.unit, not args.exclusive
    LAYERS = LAYERS_NESTED if NESTED else LAYERS_EXCL
    gen, ref, lengths = collect(args)
    excluded = {x for x in re.split(r'[,\s]+', args.exclude) if x}   # commas OR spaces: sbatch --export splits on commas
    order = [(s, d, g, i) for g, _r, members in GROUPS
             for i, (s, d, _sh) in enumerate(members) if s not in excluded]
    present = {s for lay in gen for s in gen[lay]}

    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, len(LAYERS), squeeze=False,
                             figsize=(args.width, 2.1))
    fig.patch.set_facecolor(SURFACE)
    handles = {}
    x = np.array(lengths)
    for c, (lay, label) in enumerate(LAYERS):
        ax = axes[0][c]
        if args.y == 'raw':
            m, se, n = _stack(ref[lay])
            if m is not None:
                ax.plot(x, m, color=INK_2, lw=1.0, ls=(0, (4, 2)), zorder=4)
                handles['_ref'] = Line2D([0], [0], color=INK_2, lw=1.0,
                                         ls=(0, (4, 2)),
                                         label=f'ground truth (n={n} songs)')
        else:
            ax.axhline(0, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=4)
        for sysname, disp, family, i in order:
            if sysname not in gen[lay]:
                continue
            m, se, n = (_stack(gen[lay][sysname]) if args.y == 'raw'
                        else _stack(gen[lay][sysname], ref[lay]))
            if m is None:
                continue
            col = color_of(sysname, family)
            ls = MEMBER_DASH[i % len(MEMBER_DASH)]
            ax.plot(x, m, color=col, lw=0.9, ls=ls, marker='o', markersize=2.2,
                    zorder=3)
            if not args.no_bands:
                ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=col,
                                alpha=0.14, lw=0, zorder=2)
            handles[sysname] = Line2D([0], [0], color=col, lw=0.9, ls=ls,
                                      marker='o', markersize=2.2,
                                      label=disp.replace('\n', ' '))
        ax.set_title(label + ('' if args.y == 'raw' else ' $-$ GT'),
                     fontsize=FS_TITLE, color=INK, pad=4)
        ax.set_xlabel('motif length (notes)', fontsize=FS_LABEL, color=INK)
        if UNIT == 'share':
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=None))
        if c == 0:
            unit = ('motifs per continuation' if UNIT == 'count'
                    else 'share of melody motifs')
            ax.set_ylabel(unit + ('' if args.y == 'raw' else ' $-$ GT'),
                          fontsize=FS_LABEL, color=INK)
        ax.set_xticks(lengths)
        ax.tick_params(labelsize=FS_TICK, colors=INK, length=2.5, width=0.6,
                       color=INK)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(INK); ax.spines[side].set_linewidth(0.6)
        ax.set_facecolor(SURFACE)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.annotate(chr(ord('a') + c), xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-2, 4), textcoords='offset points', ha='right',
                    va='bottom', fontsize=FS_LETTER, weight='bold', color=INK)

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
    fig.tight_layout(rect=(0, 0.24, 1, 1), w_pad=1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')

    m_tot, _se, _n = _stack(ref['total'])
    print(f'\nmelody motifs by length (notes {lengths[0]}..{lengths[-1]}), '
          f'unit={UNIT}, {"nested layers" if NESTED else "exclusive classes"}; '
          'mean over songs, samples averaged within song')
    if m_tot is not None:
        print('    motifs per GT continuation: ' + ' '.join(f'{v:.0f}' for v in m_tot))
    for lay in ([l for l, _ in LAYERS] if NESTED else CLASSES):
        print(f'  == {lay}')
        for sysname, _d, _g, _i in order:
            if sysname not in gen[lay]:
                continue
            m, _se, _n = _stack(gen[lay][sysname])
            print(f'    {sysname:11s} ' + ' '.join(f'{v:.2f}' for v in m))
        m, _se, _n = _stack(ref[lay])
        if m is not None:
            print(f'    {"GT":11s} ' + ' '.join(f'{v:.2f}' for v in m))


if __name__ == '__main__':
    main()
