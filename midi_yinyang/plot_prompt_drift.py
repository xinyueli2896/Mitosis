"""Prompt drift across the continuation: the two prompt-adherence
statistics of the E1 sheet, bar by bar, for every system and for the
ground truth.

The prompt sheet reports, per song, ONE number per statistic over the
whole continuation. This figure spreads both over TIME, computed
EXACTLY as eval_metrics computes them, on bar i of the continuation
(frames prompt + 16 i .. prompt + 16 (i + 1)) instead of on the whole:

    pc_jsd_i     JSD (base 2) between the prompt's duration-weighted
                 12-bin pitch-class histogram and bar i's
    onset_sim_i  cosine between the prompt's mean per-bar binary onset
                 profile and bar i's onset vector
    groove_i     1 - mean Hamming distance between bar i's binary onset
                 vector and the prompt's bars (groove consistency's
                 form, referenced to the prompt instead of the previous
                 bar)
    scale_i      share of bar i's onset pitch classes inside the scale
                 that best fits the prompt (scale consistency's in-scale
                 fraction, with the scale fixed by the prompt)
    delta_i      each statistic minus the same statistic on bar i of the
                 ground-truth continuation (paired per song)

--window cumulative uses bars 0..i instead of bar i alone, i.e. the
whole continuation so far; its last point is the sheet's number.

Two figures, <out> and <out>_delta, each a 4 x 2 grid: rows are the
four statistics, columns the two streams. The raw figure carries the
ground truth as a dashed grey curve; the delta figure a zero line. Lines
are means over songs (samples averaged within song), bands +-1.96 SE
over songs. Colours are the E1 families, the dash tells systems of one
family apart, legend in titled blocks.

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


METRICS = [('pc', 'Pitch-class JSD to prompt'),
           ('on', 'Onset similarity to prompt'),
           ('gc', 'Groove consistency to prompt'),
           ('sc', 'Scale consistency to prompt')]


def prompt_scale(prompt):
    """The (root, scale) that covers most of the prompt's onset pitch
    classes -- scale_consistency's own fit, on the prompt alone."""
    pcs = [p % 12 for ps in prompt.onsets for p in ps]
    if not pcs:
        return None
    best, best_set = -1.0, None
    for root in range(12):
        for scale in (em._MAJOR, em._MINOR):
            inn = sum(1 for pc in pcs if (pc - root) % 12 in scale)
            if inn > best:
                best, best_set = inn, frozenset((root + d) % 12 for d in scale)
    return best_set


def curves(prompt, cont, window, bars):
    """{metric: one value per window of the continuation}.

    window 'bar': window i = bars [i*bars, (i+1)*bars);
    window 'cumulative': window i = bars [0, (i+1)*bars).

    pc, on: eval_metrics' prompt-adherence statistics, applied to the
            window instead of the whole continuation.
    gc:     groove consistency's Hamming form, PROMPT-REFERENCED: 1 minus
            the mean Hamming distance (over 16 grid positions) between
            each window bar's binary onset vector and each prompt bar's,
            averaged over all prompt-bar x window-bar pairs. groove
            consistency proper compares a bar with its predecessor; here
            the comparison is with the prompt's bars.
    sc:     scale consistency's in-scale fraction, PROMPT-REFERENCED:
            the share of the window's onset pitch classes that lie in
            the scale best fitting the PROMPT. scale consistency proper
            fits the scale to the excerpt itself; here the prompt fixes
            the scale, so the number reads as staying in the prompt's key.
    """
    hp = em._pc_profile_dw(prompt)
    rp = em._onset_profile(prompt, 0)
    Vp = em._bar_onset_vectors(prompt)
    scale = prompt_scale(prompt)
    n_windows = cont.n_frames // (BAR * bars)
    out = {m: np.full(n_windows, np.nan) for m, _ in METRICS}
    for i in range(n_windows):
        lo = 0 if window == 'cumulative' else i * BAR * bars
        hi = (i + 1) * BAR * bars
        seg = cont.slice(lo, hi)
        hc = em._pc_profile_dw(seg)
        if hp.sum() > 0 and hc.sum() > 0:
            out['pc'][i] = em.jsd(hp, hc)
        rc = em._onset_profile(seg, 0)
        if rp is not None and rc is not None:
            out['on'][i] = em._cosine(rp, rc)
        Vw = em._bar_onset_vectors(seg)
        if len(Vp) and len(Vw):
            ham = np.abs(Vw[:, None, :] - Vp[None, :, :]).sum(-1) / BAR
            out['gc'][i] = 1.0 - float(ham.mean())
        pcs = [p % 12 for ps in seg.onsets for p in ps]
        if scale is not None and pcs:
            out['sc'][i] = sum(1 for pc in pcs if pc in scale) / len(pcs)
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
                    cv = curves(s_.slice(0, lo), s_.slice(lo, hi),
                                args.window, args.bars)
                    for m, _ in METRICS:
                        ref[(st, m)][song] = cv[m]
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
                cv = curves(r_.slice(0, lo), s_.slice(lo, hi),
                            args.window, args.bars)
                for m, _ in METRICS:
                    gen[(st, m)][system][song].append(cv[m])
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
    present = {s for k in gen for s in gen[k]}
    absent = [s for s, _, _, _ in order if s not in present]
    if absent:
        print(f'[warn] not in the manifest, not drawn: {" ".join(absent)}',
              file=sys.stderr)

    unit = 'bar' if args.bars == 1 else f'{args.bars} bars'
    table = {}

    def draw(kind, out):
        """One figure: rows = statistic (pitch-class JSD, onset
        similarity), columns = stream (melody, chord). kind 'raw' draws
        the statistic with the ground truth dashed; 'delta' the paired
        difference with a zero line."""
        fig, axes = plt.subplots(len(METRICS), len(STREAMS), squeeze=False,
                                 figsize=(args.width, 1.45 * len(METRICS) + 1.1))
        fig.patch.set_facecolor(SURFACE)
        handles = {}
        for r, (met, label) in enumerate(METRICS):
            for c, (st, sname) in enumerate(STREAMS):
                ax = axes[r][c]
                key = (st, met)
                if kind == 'raw':
                    m, se, n = _stack(ref[key])
                    if m is not None:
                        x = np.arange(1, len(m) + 1) * args.bars
                        ax.plot(x, m, color=INK_2, lw=1.0, ls=(0, (4, 2)),
                                zorder=4)
                        handles['_ref'] = Line2D(
                            [0], [0], color=INK_2, lw=1.0, ls=(0, (4, 2)),
                            label=f'ground truth (n={n} songs)')
                else:
                    ax.axhline(0, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=4)
                for sysname, disp, family, i in order:
                    if sysname not in gen[key]:
                        continue
                    m, se, n = (_stack(gen[key][sysname]) if kind == 'raw'
                                else _stack(gen[key][sysname], ref[key]))
                    if m is None:
                        continue
                    x = np.arange(1, len(m) + 1) * args.bars
                    col = color_of(sysname, family)
                    ls = MEMBER_DASH[i % len(MEMBER_DASH)]
                    ax.plot(x, m, color=col, lw=0.9, ls=ls, zorder=3)
                    if not args.no_bands:
                        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se,
                                        color=col, alpha=0.14, lw=0, zorder=2)
                    handles[sysname] = Line2D([0], [0], color=col, lw=0.9,
                                              ls=ls, label=disp.replace('\n', ' '))
                    table[(st, met, kind, sysname)] = m
                ax.set_title((label if kind == 'raw' else label + ' $-$ GT')
                             + f', {sname.lower()}', fontsize=FS_TITLE,
                             color=INK, pad=4)
                if r == len(METRICS) - 1:
                    ax.set_xlabel('continuation position (bars)',
                                  fontsize=FS_LABEL, color=INK)
                ax.tick_params(labelsize=FS_TICK, colors=INK, length=2.5,
                               width=0.6, color=INK)
                for side in ('top', 'right'):
                    ax.spines[side].set_visible(False)
                for side in ('left', 'bottom'):
                    ax.spines[side].set_color(INK)
                    ax.spines[side].set_linewidth(0.6)
                ax.set_facecolor(SURFACE)
                ax.yaxis.grid(True, color=GRID, lw=0.5); ax.xaxis.grid(False)
                ax.set_axisbelow(True)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.annotate(chr(ord('a') + r * len(STREAMS) + c),
                            xy=(0.02, 0.96), xycoords='axes fraction',
                            ha='left', va='top', fontsize=FS_LETTER,
                            weight='bold', color=INK)
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
        what = ('the statistic on each window' if kind == 'raw'
                else 'system minus ground truth, paired per song')
        fig.text(0.5, 1.0 / fig.get_figheight(),
                 f'lines: {what}, mean over songs (samples averaged within '
                 f'song); window: {args.window}, {unit}'
                 + ('' if args.no_bands else '; bands: $\\pm$1.96 SE over songs'),
                 ha='center', fontsize=FS_TICK, color=INK_2)
        leg = 1.3 / fig.get_figheight()
        fig.tight_layout(rect=(0, leg, 1, 1), w_pad=1.2, h_pad=1.4)
        for ext in ('pdf', 'png'):
            fig.savefig(f'{out}.{ext}', dpi=300, facecolor=SURFACE)
            print(f'wrote {out}.{ext}')
        plt.close(fig)

    draw('raw', args.out)
    draw('delta', args.out + '_delta')

    # numbers: first / middle / last third of the continuation
    print(f'\nwindow={args.window}, {unit}; thirds of the continuation: '
          f'raw statistic, then its delta against the ground truth')
    for met, label in METRICS:
        print(f'  == {label}')
        for st, sname in STREAMS:
            print(f'  {sname}')
            for sysname, disp, _g, _i in order:
                raw = table.get((st, met, 'raw', sysname))
                dl = table.get((st, met, 'delta', sysname))
                if raw is None:
                    continue
                k = len(raw) // 3 or 1
                fmt = lambda v: ' '.join(f'{np.nanmean(v[j*k:(j+1)*k]):+.3f}'
                                         for j in range(3))
                print(f'    {sysname:11s} raw {fmt(raw)}   delta {fmt(dl)}')
            m, _se, _n = _stack(ref[(st, met)])
            if m is not None:
                k = len(m) // 3 or 1
                print(f'    {"GT":11s} raw ' + ' '.join(
                    f'{np.nanmean(m[j*k:(j+1)*k]):+.3f}' for j in range(3)))


if __name__ == '__main__':
    main()
