"""Subjective evaluation: repeated-measures ANOVA over systems.

Design. Every rater hears all systems on a song, so the ratings form a
complete within-subject layout: subject = one (rater, song) block,
within factor = system (6 levels), one observation per cell. Per axis
(and for the mean over axes) this runs

  * one-way repeated-measures ANOVA on system, blocks as subjects:
    F(k-1, (k-1)(n-1)) with the block effect removed from the error;
    Greenhouse-Geisser epsilon from the sample covariance of the k
    columns, corrected df and p; partial eta squared = SS_sys / (SS_sys
    + SS_err);
  * post-hoc: paired t-tests of ours against every other system, Holm
    corrected across the k-1 comparisons of that axis;
  * robustness: the same ANOVA with RATER as the subject (a rater's
    songs averaged per system), which removes the dependence between
    blocks of one rater at the cost of n.

Reads the same ratings CSV as subjective_spider.py and applies the same
filters (--exclude, lettered pilot rows, --drop-constant). Writes
{out}_anova.csv (one row per analysis x axis) and {out}_posthoc.csv.

Usage (via plot_subjective.sbatch):
  python subjective_anova.py --csv ratings.csv --out results/subjective \\
      --drop-constant
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

AXES = ['consistency', 'structure', 'fit', 'musicality', 'creativity']
LABEL = {'consistency': 'Consistency', 'structure': 'Structure',
         'fit': 'Melody-chord fit', 'musicality': 'Musicality',
         'creativity': 'Creativity', 'overall': 'Overall'}
SYSTEMS = ['Duet (alt commit)', 'S-finetune', 'AMT', 'Whole-song', 'S-scratch', 'GT']
DISPLAY = {'Duet (alt commit)': 'Duet (ours)', 'S-finetune': 'Single-stream, finetuned',
           'AMT': 'Anticipatory Music Transformer', 'Whole-song': 'Whole-Song-Gen',
           'S-scratch': 'Single-stream, scratch', 'GT': 'Ground truth'}
OURS = SYSTEMS[0]


def load(path, exclude, drop_constant):
    rows = list(csv.DictReader(open(path, newline='')))
    rows = [r for r in rows if r['rater'] not in set(exclude) and len(r['system']) > 1
            and r['system'] in SYSTEMS]
    if drop_constant:
        vals = defaultdict(set)
        for r in rows:
            vals[r['rater']].update(r[a] for a in AXES)
        const = {k for k, v in vals.items() if len(v) == 1}
        rows = [r for r in rows if r['rater'] not in const]
        print(f'dropped {len(const)} constant rater(s): {sorted(const)}')
    return rows


def matrix(rows, subject):
    """subject: 'block' -> (rater, song); 'rater' -> rater, songs averaged.
    Returns dict axis -> array [n_subjects, k] in SYSTEMS order, complete
    cases only."""
    cell = defaultdict(lambda: defaultdict(list))   # subj -> system -> [vec]
    for r in rows:
        subj = (r['rater'], r['prompt']) if subject == 'block' else r['rater']
        cell[subj][r['system']].append([float(r[a]) for a in AXES])
    out = {a: [] for a in AXES + ['overall']}
    kept = 0
    for subj, d in cell.items():
        if any(s not in d for s in SYSTEMS):
            continue
        kept += 1
        M = np.array([np.mean(d[s], axis=0) for s in SYSTEMS])   # [k, 5]
        for j, a in enumerate(AXES):
            out[a].append(M[:, j])
        out['overall'].append(M.mean(1))
    return {a: np.array(v) for a, v in out.items()}, kept


def rm_anova(Y):
    """One-way repeated-measures ANOVA. Y: [n, k]."""
    n, k = Y.shape
    gm = Y.mean()
    ss_sys = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_subj = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_tot = ((Y - gm) ** 2).sum()
    ss_err = ss_tot - ss_sys - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (ss_sys / df1) / (ss_err / df2)
    p = stats.f.sf(F, df1, df2)
    # Greenhouse-Geisser epsilon
    S = np.cov(Y, rowvar=False, ddof=1)
    S_dc = S - S.mean(0, keepdims=True) - S.mean(1, keepdims=True) + S.mean()
    eps = np.trace(S_dc) ** 2 / ((k - 1) * (S_dc ** 2).sum())
    eps = float(min(max(eps, 1.0 / (k - 1)), 1.0))
    p_gg = stats.f.sf(F, df1 * eps, df2 * eps)
    eta_p = ss_sys / (ss_sys + ss_err)
    return dict(n=n, k=k, df1=df1, df2=df2, F=float(F), p=float(p), eps=eps,
                p_gg=float(p_gg), eta_p=float(eta_p))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True, help='path without extension')
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--drop-constant', action='store_true')
    args = ap.parse_args()
    rows = load(args.csv, args.exclude, args.drop_constant)

    anova_rows, post_rows = [], []
    for subject in ('block', 'rater'):
        Ys, n = matrix(rows, subject)
        print(f'\n=== subject = {subject}  (n = {n} complete subjects, k = {len(SYSTEMS)} systems)')
        print(f'{"axis":18s} {"F":>7s} {"df":>12s} {"p":>9s} {"eps_GG":>7s} {"p_GG":>9s} {"eta_p2":>7s}')
        for a in AXES + ['overall']:
            Y = Ys[a]
            r = rm_anova(Y)
            print(f'{LABEL[a]:18s} {r["F"]:7.2f} {r["df1"]:>4d},{r["df2"]:<7d} {r["p"]:9.2e} '
                  f'{r["eps"]:7.3f} {r["p_gg"]:9.2e} {r["eta_p"]:7.3f}')
            anova_rows.append(dict(subject=subject, axis=a, **r))
            # post-hoc against ours, Holm over the k-1 comparisons
            d = Y[:, 0:1] - Y[:, 1:]
            t = [stats.ttest_rel(Y[:, 0], Y[:, j]) for j in range(1, len(SYSTEMS))]
            praw = np.array([x.pvalue for x in t])
            padj = holm(praw)
            for j, s in enumerate(SYSTEMS[1:]):
                post_rows.append(dict(subject=subject, axis=a, system=s,
                                      mean_diff=float(d[:, j].mean()),
                                      sd_diff=float(d[:, j].std(ddof=1)),
                                      t=float(t[j].statistic), df=int(Y.shape[0] - 1),
                                      p_raw=float(praw[j]), p_holm=float(padj[j])))
        print(f'\n  post-hoc, ours minus system (paired t, Holm over {len(SYSTEMS) - 1}):')
        print(f'  {"axis":18s} ' + ' '.join(f'{DISPLAY[s][:14]:>22s}' for s in SYSTEMS[1:]))
        for a in AXES + ['overall']:
            cells = [x for x in post_rows if x['subject'] == subject and x['axis'] == a]
            print(f'  {LABEL[a]:18s} ' + ' '.join(
                f'{x["mean_diff"]:+.2f} (p={x["p_holm"]:.3f})'.rjust(22) for x in cells))

    with open(f'{args.out}_anova.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(anova_rows[0].keys()))
        w.writeheader(); w.writerows(anova_rows)
    with open(f'{args.out}_posthoc.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(post_rows[0].keys()))
        w.writeheader(); w.writerows(post_rows)
    print(f'\nwrote {args.out}_anova.csv, {args.out}_posthoc.csv')


if __name__ == '__main__':
    main()
