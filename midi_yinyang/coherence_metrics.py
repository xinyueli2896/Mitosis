"""Reference-free coherence, repetition and general symbolic-music
statistics for E1 outputs, read among systems with the reference pair as
one more row (the second results table; see EXPERIMENTS.md).

Inputs are the E1 manifest (system, mode, song, sample, path) and the
staged reference folders, exactly as eval_metrics takes them; every
statistic is computed on the TICK grid (eval_metrics.load_streams) over
the scored window [prompt, total), for the generated pair and for the
reference pair of the same song ('<name>_ref' columns). Bars are 16
frames.

Blocks (all per file; 'a' = melody, 'b' = chord, none = both streams):

COHERENCE -- does the piece hang together over bars?
  rep_chroma, rep_rhythm   mean over bars of the best cosine match to any
                           earlier non-adjacent bar (bar chroma / bar
                           onset-position histogram). structure_metrics'
                           definition, moved to the tick grid.
  period4_*, period8_*     mean cosine between bars 4 / 8 apart.
  si_short, si_mid, si_long  structureness indicators (Wu & Yang 2020)
                           in symbolic form: on the bar self-similarity
                           matrix (chroma + rhythm), the largest mean
                           diagonal similarity over lags 3-7, 8-14 and
                           15+ bars. The original uses an audio fitness
                           scape plot; this is the same question asked
                           of the bar features.
  drift_slope, drift_mean  per-bar chroma similarity to the PROMPT's
                           mean chroma: mean and linear trend per 16
                           bars (tonal anchoring / rollout drift).
  pce1, pce4               pitch-class histogram entropy over 1 and 4
                           bars, averaged (Wu & Yang H1/H4).
  cpi                      chord progression irregularity: fraction of
                           unique chord trigrams among all trigrams, the
                           chord being the sounding pc-set per beat.
  gs                       grooving pattern similarity: mean over bar
                           pairs of 1 - hamming/16 on the binary onset
                           pattern (both streams).

REPETITION -- degenerate looping the coherence rows cannot separate
              from motivic recurrence:
  copy_bar_a, copy_bar_b   fraction of non-empty bars whose exact
                           (frame, pitch) content equals an EARLIER bar.
  ngram4_a                 fraction of melody 4-grams of (interval,
                           duration) that already occurred earlier.
  ngram4_b                 same for the beat-level chord label sequence.
  loop_a, loop_b           longest run of consecutive identical bars,
                           as a fraction of the scored bars.

GENERAL -- MusPy-style per-stream statistics:
  pitch_range_a, n_pitches_a, pitch_entropy_a, scale_consistency_a
                           (best fit over the 24 major/minor scales)
  n_chord_types_b, chord_change_rate_b (changes per bar)

Usage (via score_coherence.sbatch):
    python coherence_metrics.py --manifest results/E1_p96_matched_manifest.tsv \\
        --ref-a-dir temp/E1_matched_p6/prompts/mel \\
        --ref-b-dir temp/E1_matched_p6/prompts/chord \\
        --prompt-frames 96 --total-frames 416 --out results/E1_p96_matched_coherence.csv
    python coherence_metrics.py --table results/E1_mixed_matched_coherence.csv \\
        --baseline A3ctcaT --out results/E1_paper_coherence
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter

import numpy as np

from eval_metrics import load_streams, FRAMES_PER_BAR

BAR = FRAMES_PER_BAR


# ---------------------------------------------------------------------------
# bar features
# ---------------------------------------------------------------------------

def _bars(stream, lo, hi):
    """list of (onset frames list per bar) -> bar index -> [(f, pitch)]"""
    n = (hi - lo) // BAR
    out = [[] for _ in range(n)]
    for f in range(lo, lo + n * BAR):
        for p in stream.onsets[f]:
            out[(f - lo) // BAR].append(((f - lo) % BAR, p))
    return out


def _chroma(bar_notes):
    v = np.zeros(12)
    for _, p in bar_notes:
        v[p % 12] += 1
    return v


def _rhythm(bar_notes):
    v = np.zeros(BAR)
    for f, _ in bar_notes:
        v[f] = 1
    return v


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _entropy(counts):
    c = np.asarray(counts, dtype=float)
    s = c.sum()
    if s == 0:
        return float('nan')
    p = c[c > 0] / s
    return float(-(p * np.log2(p)).sum())


MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def _scale_consistency(pitches):
    if not pitches:
        return float('nan')
    pcs = [p % 12 for p in pitches]
    best = 0.0
    for root in range(12):
        for scale in (MAJOR, MINOR):
            allowed = {(root + d) % 12 for d in scale}
            best = max(best, sum(1 for pc in pcs if pc in allowed) / len(pcs))
    return best


# ---------------------------------------------------------------------------
# metric blocks
# ---------------------------------------------------------------------------

def coherence(a, b, lo, hi, prompt_lo=0):
    out = {}
    ba, bb = _bars(a, lo, hi), _bars(b, lo, hi)
    n = len(ba)
    both = [x + y for x, y in zip(ba, bb)]
    chroma = [_chroma(x) for x in both]
    rhythm = [_rhythm(x) for x in both]

    def rep(feats):
        vals = []
        for i in range(2, len(feats)):
            vals.append(max(_cos(feats[i], feats[j]) for j in range(0, i - 1)))
        return float(np.mean(vals)) if vals else float('nan')

    def period(feats, k):
        vals = [_cos(feats[i], feats[i - k]) for i in range(k, len(feats))]
        return float(np.mean(vals)) if vals else float('nan')

    out['rep_chroma'] = rep(chroma)
    out['rep_rhythm'] = rep(rhythm)
    for k in (4, 8):
        out[f'period{k}_chroma'] = period(chroma, k)
        out[f'period{k}_rhythm'] = period(rhythm, k)

    # structureness indicators on the combined bar SSM
    feats = [np.concatenate([c / (np.linalg.norm(c) + 1e-9),
                             r / (np.linalg.norm(r) + 1e-9)]) for c, r in zip(chroma, rhythm)]
    S = np.array([[_cos(x, y) for y in feats] for x in feats]) if n else np.zeros((0, 0))

    def si(lo_lag, hi_lag):
        best = float('nan')
        for L in range(lo_lag, min(hi_lag, n)):
            d = float(np.mean([S[i, i + L] for i in range(n - L)]))
            best = d if math.isnan(best) else max(best, d)
        return best

    out['si_short'] = si(3, 8)
    out['si_mid'] = si(8, 15)
    out['si_long'] = si(15, n)

    # tonal drift relative to the prompt's chroma
    pa, pb = _bars(a, prompt_lo, lo), _bars(b, prompt_lo, lo)
    pc = np.sum([_chroma(x + y) for x, y in zip(pa, pb)], axis=0) if pa else np.zeros(12)
    sims = [_cos(c, pc) for c in chroma]
    out['drift_mean'] = float(np.mean(sims)) if sims else float('nan')
    if len(sims) >= 2:
        slope = np.polyfit(np.arange(len(sims)), sims, 1)[0]
        out['drift_slope'] = float(slope * 16)
    else:
        out['drift_slope'] = float('nan')

    # pitch-class entropy over 1 and 4 bars
    out['pce1'] = float(np.nanmean([_entropy(c) for c in chroma])) if chroma else float('nan')
    e4 = [_entropy(np.sum(chroma[i:i + 4], axis=0)) for i in range(0, n - 3)]
    out['pce4'] = float(np.nanmean(e4)) if e4 else float('nan')

    # chord progression irregularity over beat-level chord labels
    labels = _chord_labels(b, lo, hi)
    tri = [tuple(labels[i:i + 3]) for i in range(len(labels) - 2)]
    out['cpi'] = len(set(tri)) / len(tri) if tri else float('nan')

    # grooving pattern similarity
    pats = [_rhythm(x) for x in both]
    gs = [1 - np.abs(pats[i] - pats[j]).sum() / BAR
          for i in range(n) for j in range(i + 1, n)]
    out['gs'] = float(np.mean(gs)) if gs else float('nan')
    return out


def _chord_labels(b, lo, hi):
    """sounding pc-set of the chord stream per beat (4 frames)"""
    labels = []
    for f in range(lo, hi, BAR // 4):
        pcs = set()
        for g in range(f, min(f + BAR // 4, hi)):
            pcs |= set(b.sounding[g])
        labels.append(frozenset(pcs))
    return labels


def repetition(a, b, lo, hi):
    out = {}
    for tag, s in (('a', a), ('b', b)):
        bars = [tuple(sorted(x)) for x in _bars(s, lo, hi)]
        nonempty = [(i, x) for i, x in enumerate(bars) if x]
        copies = sum(1 for i, x in nonempty if x in bars[:i])
        out[f'copy_bar_{tag}'] = copies / len(nonempty) if nonempty else float('nan')
        best = run = 0
        for i in range(1, len(bars)):
            run = run + 1 if bars[i] == bars[i - 1] and bars[i] else 0
            best = max(best, run)
        out[f'loop_{tag}'] = best / len(bars) if bars else float('nan')

    # melody 4-grams of (interval, duration)
    line = [(f, max(ps)) for f in range(lo, hi) for ps in [a.onsets[f]] if ps]
    # eval_metrics keeps durations as a flat list, so the 4-grams use
    # inter-onset intervals in place of durations.
    grams = []
    for i in range(1, len(line) - 3):
        g = tuple((line[k + 1][1] - line[k][1], line[k + 1][0] - line[k][0])
                  for k in range(i - 1, i + 3))
        grams.append(g)
    seen, rep = set(), 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    out['ngram4_a'] = rep / len(grams) if grams else float('nan')

    labels = _chord_labels(b, lo, hi)
    grams = [tuple(labels[i:i + 4]) for i in range(len(labels) - 3)]
    seen, rep = set(), 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    out['ngram4_b'] = rep / len(grams) if grams else float('nan')
    return out


def general(a, b, lo, hi):
    out = {}
    pitches = [p for f in range(lo, hi) for p in a.onsets[f]]
    out['pitch_range_a'] = float(max(pitches) - min(pitches)) if pitches else float('nan')
    out['n_pitches_a'] = float(len(set(pitches))) if pitches else float('nan')
    out['pitch_entropy_a'] = _entropy(list(Counter(pitches).values())) if pitches else float('nan')
    out['scale_consistency_a'] = _scale_consistency(pitches)
    labels = _chord_labels(b, lo, hi)
    nonempty = [l for l in labels if l]
    out['n_chord_types_b'] = float(len(set(nonempty))) if nonempty else float('nan')
    changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1] and labels[i])
    n_bars = (hi - lo) / BAR
    out['chord_change_rate_b'] = changes / n_bars if n_bars else float('nan')
    return out


METRICS = (['rep_chroma', 'rep_rhythm', 'period4_chroma', 'period8_chroma',
            'period4_rhythm', 'period8_rhythm', 'si_short', 'si_mid', 'si_long',
            'drift_mean', 'drift_slope', 'pce1', 'pce4', 'cpi', 'gs']
           + ['copy_bar_a', 'copy_bar_b', 'loop_a', 'loop_b', 'ngram4_a', 'ngram4_b']
           + ['pitch_range_a', 'n_pitches_a', 'pitch_entropy_a', 'scale_consistency_a',
              'n_chord_types_b', 'chord_change_rate_b'])


def score_file(gen_paths, ref_paths, args):
    lo, hi = args.prompt_frames, args.total_frames
    ga, gb = load_streams(gen_paths, 'melchord', args.mel_programs, args.chord_programs, hi)
    ra, rb = load_streams(ref_paths, 'melchord', args.mel_programs, args.chord_programs, hi)
    row = {}
    for a, b, suf in ((ga, gb, ''), (ra, rb, '_ref')):
        d = {}
        d.update(coherence(a, b, lo, hi))
        d.update(repetition(a, b, lo, hi))
        d.update(general(a, b, lo, hi))
        row.update({k + suf: v for k, v in d.items()})
    return row


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

TABLE_COLS = [('si_mid', 'SI$_{\\mathrm{mid}}$'), ('rep_chroma', 'Rep.'),
              ('drift_slope', 'Drift'), ('pce4', 'PCE$_4$'), ('cpi', 'CPI'),
              ('gs', 'GS'), ('copy_bar_a', 'Copy$_m$'), ('copy_bar_b', 'Copy$_c$'),
              ('ngram4_a', '4-gram$_m$'), ('scale_consistency_a', 'Scale')]
ORDER = ['A11aT', 'A11K4', 'A3fctcaT', 'A3fcaT', 'A3fcaTar', 'A3fcK4', 'A3ctcaT', 'A3ctca', 'A3ctc', 'A3ctc2', 'A3', 'A3K0', 'A3ctcc', 'A3f', 'A3c', 'A3fc',
         'A10', 'A1', 'A1cp8', 'S1', 'S-scratch', 'P-mc', 'P-cm', 'WS']


def make_table(csv_path, baseline, out, alpha=0.05, systems=None):
    from aggregate_eval_results import wilcoxon_signed_rank
    from e1_paper_table import SYSTEM_NAME
    per = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            per.setdefault(r['system'], {}).setdefault(r['song'], []).append(r)
    if systems:
        missing = [s for s in systems if s not in per]
        if missing:
            raise SystemExit(f'--systems not in CSV: {missing} (have {sorted(per)})')
        systems = list(systems)
    else:
        systems = [s for s in ORDER if s in per] + sorted(s for s in per if s not in ORDER)

    def song_means(system, metric):
        d = {}
        for sg, rows in per[system].items():
            xs = [float(r[metric]) for r in rows if r.get(metric) not in ('', None)
                  and not math.isnan(float(r[metric]))]
            if xs:
                d[sg] = float(np.mean(xs))
        return d

    def cell(vals, mark=''):
        if not vals:
            return '--'
        return f'{np.mean(vals):.2f}$\\pm${np.std(vals) / math.sqrt(len(vals)):.2f}{mark}'

    ncol = len(TABLE_COLS) + 1
    n_songs = len({sg for s in systems for sg in per[s]})
    L = [r'\begin{table*}[htbp]',
         r'\caption{Coherence, repetition and general statistics of the generated pair on '
         + str(n_songs) + r' held-out songs (mean$\pm$s.e.m. over songs; 3 samples per song '
         r'averaged first). Reference-free: read among systems, with the reference pair as a row.}',
         r'\begin{center}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{|l|' + 'c|' * len(TABLE_COLS) + '}', r'\hline',
         r'\textbf{System} & ' + ' & '.join(r'\textbf{' + t + '}' for _, t in TABLE_COLS) + r' \\',
         r'\hline']
    ref_cells = [cell(list(song_means(baseline, m + '_ref').values())) for m, _ in TABLE_COLS]
    L.append(r'\textit{Reference pair} & ' + ' & '.join(ref_cells) + r' \\')
    L.append(r'\hline')
    for s in systems:
        cells = []
        for m, _ in TABLE_COLS:
            sv = song_means(s, m)
            mark = ''
            if s != baseline:
                bv = song_means(baseline, m)
                common = sorted(set(sv) & set(bv))
                diffs = [sv[k] - bv[k] for k in common]
                if len(diffs) >= 5:
                    _, pv, _ = wilcoxon_signed_rank(diffs, 'two-sided')
                    if pv < alpha / len(TABLE_COLS):
                        mark = r'$^{\mathrm{b}}$'
                    elif pv < alpha:
                        mark = r'$^{\mathrm{a}}$'
            cells.append(cell(list(sv.values()), mark))
        name = SYSTEM_NAME.get(s, s)
        if s == baseline:
            name = r'\textbf{' + name + '}'
        L.append(name + ' & ' + ' & '.join(cells) + r' \\')
        L.append(r'\hline')
    L.append(r'\multicolumn{' + str(ncol) + r'}{l}{SI$_{\mathrm{mid}}$: structureness at 8--14 bars; '
             r'Rep.: best chroma match to an earlier bar; Drift: trend of similarity to the prompt per 16 bars; '
             r'PCE$_4$: pitch-class entropy over 4 bars; CPI: unique chord trigrams; GS: groove similarity; '
             r'Copy: exact-copy bar fraction; 4-gram$_m$: repeated melodic 4-grams; Scale: best-scale fit.} \\')
    L.append(r'\multicolumn{' + str(ncol) + r'}{l}{$^{\mathrm{a}}$/$^{\mathrm{b}}$: differs from \emph{'
             + SYSTEM_NAME.get(baseline, baseline) + r'} at $p<' + f'{alpha:g}' + r'$ / $p<'
             + f'{alpha / len(TABLE_COLS):.3g}' + r'$ (two-sided paired Wilcoxon over songs).}')
    L += [r'\end{tabular}', r'\label{tab:coherence}', r'\end{center}', r'\end{table*}']
    with open(out + '.tex', 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'[table] wrote {out}.tex  ({len(systems)} systems, {n_songs} songs)')
    # console summary
    print(f'{"system":<12}' + ''.join(f'{t:>14}' for _, t in TABLE_COLS))
    print(f'{"reference":<12}' + ''.join(f'{c:>14}' for c in [x.split("$")[0] for x in ref_cells]))
    for s in systems:
        print(f'{s:<12}' + ''.join(f'{np.mean(list(song_means(s, m).values())) if song_means(s, m) else float("nan"):>14.3f}'
                                   for m, _ in TABLE_COLS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest')
    p.add_argument('--ref-a-dir')
    p.add_argument('--ref-b-dir')
    p.add_argument('--prompt-frames', type=int, default=96)
    p.add_argument('--total-frames', type=int, default=416)
    p.add_argument('--mel-programs', default='0,24')
    p.add_argument('--chord-programs', default='48')
    p.add_argument('--out', required=True)
    p.add_argument('--table', help='coherence CSV -> LaTeX table at --out')
    p.add_argument('--baseline', default='A3ctcaT')
    p.add_argument('--systems', nargs='*', default=None,
                   help='table only: restrict and order the rows (default: every system in the CSV)')
    args = p.parse_args()
    if args.table:
        make_table(args.table, args.baseline, args.out, systems=args.systems)
        return
    args.mel_programs = {int(x) for x in args.mel_programs.split(',')}
    args.chord_programs = {int(x) for x in args.chord_programs.split(',')}
    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            system, mode, song, sample, path = line.split('\t')[:5]
            gen_paths = path.split(';')
            ref_paths = [os.path.join(args.ref_a_dir, f'{song}.mid'),
                         os.path.join(args.ref_b_dir, f'{song}.mid')]
            try:
                row = score_file(gen_paths, ref_paths, args)
            except Exception as e:      # noqa: BLE001
                print(f'  [fail] {path}: {e!r}', file=sys.stderr)
                continue
            row.update(system=system, mode=mode, song=song, sample=sample)
            rows.append(row)
    keys = ['system', 'mode', 'song', 'sample'] + METRICS + [m + '_ref' for m in METRICS]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
