"""Subjective evaluation: per-system mean ratings as a spider (radar)
chart, plus a rater-demographics report.

Reads the ratings CSV the listening page writes (one row per rater x
clip: rater, age, gender, background, clip, prompt, system, and the
five 1-5 ratings consistency / structure / fit / musicality /
creativity). Drops the raters named by --exclude and any row whose
system is a bare letter (an earlier page version coded systems A..G
and the mapping is not recoverable), then:

  * one polygon per system over the five rating axes, the value being
    the MEAN over all kept ratings of that system, family colours as in
    plot_e1_box (ours gold, internal slate, external maroon, unranked
    grey, ground truth in ink dashed, only ours filled); the radial axis
    runs from --rmin (default 3) to 5, piecewise linear: the band above
    the ground-truth maximum gets --squash of the radius (default 0.12)
    so the systems spread out; tick labels are scale values
  * a per-system table of mean and 95% interval per axis, the interval
    from a bootstrap over RATERS (a rater's ratings are not
    independent), and an overall mean
  * demographics: raters kept, ratings per rater, age, gender and
    background counts, prompts covered

Usage (via plot_subjective.sbatch):
  python subjective_spider.py --csv ratings.csv --out results/subjective \
      --exclude azure-stoat-9403 violet-quail-2727
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_e1_box import (SURFACE, INK, INK_2, GRID, FAMILY_COLOR,  # noqa
                         FS_TICK, FS_LABEL, FS_LEGEND, grouped_legend)

AXES = [('consistency', 'Consistency'), ('structure', 'Structure'),
        ('fit', 'Melody-chord fit'), ('musicality', 'Musicality'),
        ('creativity', 'Creativity')]

# system id in the CSV -> (display name, family, line style)
SYSTEMS = [
    ('Duet (alt commit)', 'Duet w/o iterative refinement', 'Ours', '-'),
    ('Duet',              'Duet',                'Ours',               (0, (5, 2))),
    ('S-scratch',         'Single-stream (scratch)',   'Internal baselines', '-'),
    ('Whole-song',        'Whole-Song Gen',            'External baselines', '-'),
    ('AMT',               'Anticipatory Music Transf.', 'External baselines', (0, (5, 2))),
    ('S-finetune',        'Single-stream (finetuned)', 'Not ranked',         '-'),
    ('GT',                'Ground truth',              'GT',                 (0, (4, 3))),
]
GT_COLOR = INK

# --palette system: one colour per system from the six-swatch set
# (2026-09-17, by request), instead of the E1 family colours.
SYSTEM_COLORS = {
    'Duet (alt commit)': '#ffd700',   # gold
    'Duet':              '#ffd700',
    'S-scratch':         '#fa8775',   # light orange
    'S-finetune':        '#ea5f94',   # pink
    'Whole-song':        '#cd34b5',   # magenta
    'AMT':               '#9d02d7',   # purple
    'GT':                '#0000ff',   # blue
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True, help='path without extension')
    ap.add_argument('--exclude', nargs='*', default=[], help='rater ids')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--width', type=float, default=3.4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rmin', type=float, default=3.0,
                    help='inner edge of the radial axis (rating scale 1-5)')
    ap.add_argument('--squash', type=float, default=0.12,
                    help='share of the radius given to the range ABOVE the '
                         'ground-truth maximum, up to 5; the rest goes to '
                         '[rmin, GT max], so the systems spread out. 0 = '
                         'linear axis')
    ap.add_argument('--palette', choices=['family', 'system'], default='family',
                    help='family: the E1 family colours; system: one colour '
                         'per system (SYSTEM_COLORS)')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline='')))
    n_all = len(rows)
    excluded = set(args.exclude)
    dropped_excl = [r for r in rows if r['rater'] in excluded]
    rows = [r for r in rows if r['rater'] not in excluded]
    lettered = [r for r in rows if len(r['system']) == 1]
    rows = [r for r in rows if len(r['system']) > 1]
    known = {s for s, *_ in SYSTEMS}
    unknown = sorted({r['system'] for r in rows} - known)
    if unknown:
        print(f'[warn] systems not in the table, dropped: {unknown}',
              file=sys.stderr)
        rows = [r for r in rows if r['system'] in known]
    print(f'rows: {n_all} total; {len(dropped_excl)} from excluded raters '
          f'{sorted(excluded)}; {len(lettered)} lettered-system rows from '
          f'{len({r["rater"] for r in lettered})} rater(s) (unmappable, '
          f'dropped); {len(rows)} kept')

    # ---- demographics ------------------------------------------------
    raters = {}
    per_rater = Counter(r['rater'] for r in rows)
    for r in rows:
        raters.setdefault(r['rater'], (r['age'], r['gender'], r['background']))
    ages = [int(a) for a, _g, _b in raters.values() if a]
    print(f'\nRATERS KEPT: {len(raters)}  ratings per rater: '
          f'min {min(per_rater.values())} median '
          f'{int(np.median(list(per_rater.values())))} max {max(per_rater.values())}')
    print(f'  age: n={len(ages)} mean {np.mean(ages):.1f} median '
          f'{np.median(ages):.0f} range {min(ages)}-{max(ages)}; '
          f'bins: ' + ', '.join(f'{lo}-{hi}: {sum(lo <= a <= hi for a in ages)}'
                                for lo, hi in ((0, 17), (18, 29), (30, 44), (45, 64))))
    print('  gender: ' + ', '.join(f'{k or "n/a"}: {v}' for k, v in
                                    Counter(g for _a, g, _b in raters.values()).most_common()))
    print('  background: ' + ', '.join(f'{k or "n/a"}: {v}' for k, v in
                                        Counter(b for _a, _g, b in raters.values()).most_common()))
    print('  prompts covered: ' + ', '.join(
        f'p{p}: {n}' for p, n in sorted(Counter(r['prompt'] for r in rows).items())))
    print('  per rater: ' + ', '.join(f'{k}({v})' for k, v in per_rater.most_common()))

    # ---- means and rater bootstrap ----------------------------------
    by_sys = defaultdict(list)          # system -> [(rater, vec5)]
    for r in rows:
        by_sys[r['system']].append((r['rater'],
                                    np.array([float(r[a]) for a, _ in AXES])))
    rng = np.random.default_rng(args.seed)
    stats = {}
    rater_list = sorted(raters)
    for sysid, items in by_sys.items():
        M = np.array([v for _r, v in items])
        who = np.array([_r for _r, _v in items])
        mean = M.mean(0)
        boots = []
        for _ in range(args.n_boot):
            pick = rng.choice(rater_list, len(rater_list), replace=True)
            idx = np.concatenate([np.flatnonzero(who == p) for p in pick])
            if len(idx):
                boots.append(M[idx].mean(0))
        boots = np.array(boots)
        stats[sysid] = (mean, np.percentile(boots, 2.5, 0),
                        np.percentile(boots, 97.5, 0), len(M),
                        len(set(who)))

    print(f'\n{"system":40s} {"n":>4s} {"raters":>6s} ' +
          ' '.join(f'{lab[:11]:>11s}' for _a, lab in AXES) + '   overall')
    for sysid, disp, _fam, _ls in SYSTEMS:
        if sysid not in stats:
            print(f'{disp:40s}   -- not rated')
            continue
        mean, lo, hi, n, nr = stats[sysid]
        print(f'{disp:40s} {n:4d} {nr:6d} ' +
              ' '.join(f'{m:.2f}[{l:.2f},{h:.2f}]'[:11].rjust(11)
                       for m, l, h in zip(mean, lo, hi)) +
              f'   {mean.mean():.2f}')

    # ---- spider chart -----------------------------------------------
    K = len(AXES)
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    # Radial mapping: piecewise linear. [rmin, gt_max] fills (1 - squash)
    # of the radius, [gt_max, 5] the remaining share, so the empty band
    # above the ground truth stops eating the room the systems need.
    # Tick labels keep the ORIGINAL scale values. Say so in the caption.
    gt_max = float(stats['GT'][0].max()) if 'GT' in stats else 5.0
    knee = min(gt_max + 0.05, 5.0)
    if args.squash > 0 and knee < 5.0:
        def rmap(v):
            v = np.asarray(v, float)
            lo_part = (v - args.rmin) / (knee - args.rmin) * (1 - args.squash)
            hi_part = (1 - args.squash) + (v - knee) / (5.0 - knee) * args.squash
            return np.where(v <= knee, lo_part, hi_part)
    else:
        def rmap(v):
            return (np.asarray(v, float) - args.rmin) / (5.0 - args.rmin)
    fig = plt.figure(figsize=(args.width, args.width * 1.02))
    ax = fig.add_subplot(111, polar=True)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ticks = [t for t in (2, 2.5, 3, 3.5, 4, 4.5, 5) if t >= args.rmin]
    ax.set_yticks(list(rmap(ticks)))
    ax.set_yticklabels([f'{t:g}' for t in ticks], fontsize=FS_TICK - 1,
                       color=INK_2)
    ax.set_rlabel_position(90 / K)
    ax.set_xticks(ang)
    ax.set_xticklabels([lab for _a, lab in AXES], fontsize=FS_LABEL, color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.spines['polar'].set_color(GRID)
    handles = []
    for sysid, disp, fam, ls in SYSTEMS:
        if sysid not in stats:
            continue
        mean = stats[sysid][0]
        if args.palette == 'system':
            col = SYSTEM_COLORS[sysid]
        else:
            col = GT_COLOR if fam == 'GT' else FAMILY_COLOR[fam]
        v = rmap(np.concatenate([mean, mean[:1]]))
        ax.plot(ang_c, v, color=col, lw=1.1, ls=ls, zorder=3)
        if fam == 'Ours':
            # only our polygon is filled: five stacked fills greyed the
            # whole interior
            ax.fill(ang_c, v, color=col, alpha=0.12, lw=0, zorder=2)
        handles.append((fam, Line2D([0], [0], color=col, lw=1.1, ls=ls,
                                    label=disp)))
    # legend in titled blocks by model group; colour is the group, line
    # type tells systems of one group apart
    blocks = []
    for title, fams in (('Ours', ('Ours',)),
                        ('Baselines', ('Internal baselines',
                                       'External baselines', 'Not ranked')),
                        ('Reference', ('GT',))):
        hs = [h for f, h in handles if f in fams]
        if hs:
            blocks.append((title, hs, [h.get_label() for h in hs]))
    # one column is too narrow for three blocks in a row: Ours and
    # Reference stack on the left, the baselines sit on the right
    by_title = {t: (t, h, l) for t, h, l in blocks}
    if 'Ours' in by_title:
        grouped_legend(fig, [by_title['Ours']], y=0.10, xs=[0.02], handlelength=2.0)
    if 'Reference' in by_title:
        grouped_legend(fig, [by_title['Reference']], y=0.0, xs=[0.02], handlelength=2.0)
    if 'Baselines' in by_title:
        grouped_legend(fig, [by_title['Baselines']], y=0.0, xs=[0.5], handlelength=2.0)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')

    with open(f'{args.out}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'display', 'n_ratings', 'n_raters']
                   + [f'{a}_{k}' for a, _ in AXES for k in ('mean', 'lo', 'hi')]
                   + ['overall_mean'])
        for sysid, disp, _fam, _ls in SYSTEMS:
            if sysid in stats:
                mean, lo, hi, n, nr = stats[sysid]
                w.writerow([sysid, disp, n, nr]
                           + [f'{x:.3f}' for m, l, h in zip(mean, lo, hi) for x in (m, l, h)]
                           + [f'{mean.mean():.3f}'])
    print(f'wrote {args.out}.csv')


if __name__ == '__main__':
    main()
