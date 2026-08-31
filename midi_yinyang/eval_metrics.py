"""Quantitative evaluation metrics for E1 co-generation (EXPERIMENTS.md).

Covers hypotheses H1-H3; H4 is the listening test and is out of scope.
Reporting follows the pre-registered priority ordering H3 > H2 > H1:
the summary prints hypothesis blocks in that order and each hypothesis
has ONE designated primary endpoint (marked *), the rest are
supporting diagnostics.

  H3 (stream-appropriate grammar)   [highest priority]
    * harmonic_rhythm_jsd   chord-change interval distribution vs ref
      mel_stepwise_delta    melody stepwise-motion rate minus ref
      onset_grid_jsd_<s>    within-bar onset-position histogram vs ref
      duration_jsd_<s>      note-duration distribution vs ref
  H2 (inter-stream fit)
    * chord_tone_cov_delta  melody-note chord-tone coverage minus the
                            SAME statistic on the reference pair
      ctnctr_delta          chord-tone/non-chord-tone ratio with the
                            proper-passing-tone allowance (Yeh et al.
                            2021), vs ref
      pcs_delta             duration-weighted pitch consonance score
                            (Yeh et al. 2021), vs ref
      mctd_delta            duration-weighted melody-chord tonal
                            distance in Harte centroid space (lower =
                            closer), vs ref
      coupling_delta        THE mechanism-sensitive one: coverage on
                            the true pairing minus coverage with the
                            chord track circularly shifted 2 bars.
                            Generic all-purpose harmony scores ~0
                            however consonant it is; streams that
                            genuinely track each other score positive.
                            Reported vs the reference pair's coupling.
      (drumnondrum: * onset_sync_delta -- nondrum onsets within +-1
       frame of a drum onset, minus reference rate)
  H1 (per-stream role & texture integrity)
    * survival_min          min over streams of active-bar fraction
      mel_poly_rate         melody frames with >=2 simultaneous onsets
      density_ratio_<s>     mean notes/bar over ref's mean
      density_drift_<s>     per-bar density slope (normalized)
      empty_rate_<s>        silent-frame fraction (collapse detector)
      register_overlap_delta  stream register overlap minus ref overlap

All metrics are computed on the CONTINUATION only (frames
--prompt-frames .. --total-frames, default 64..384) on a 16th-note
grid derived from the file's tempo (beat_div=4, 16 frames/bar).

Stream extraction from a midi file, in order of preference:
  1. track/instrument names MELODY / CHORD (combined files, duet outputs)
  2. --task drumnondrum: is_drum flag vs the rest
  3. programs via --mel-programs / --chord-programs (S* outputs, e.g. 0 vs 48)

Usage:
  # one generated file (combined) against the reference pair:
  python eval_metrics.py --task melchord \
      --gen temp/eval/pop909/A.2/co/001/sample0.mid \
      --ref-a ref/melody/001.mid --ref-b ref/chord/001.mid

  # batch over a manifest TSV (columns: system mode song sample path):
  python eval_metrics.py --task melchord --manifest runs.tsv \
      --ref-a-dir ref/melody --ref-b-dir ref/chord --out results.csv

The manifest path may also be "melody_path;chord_path" when a system
emits per-stream files.
"""

import argparse
import csv
import math
import os
import re
import sys

import numpy as np
import pretty_midi

BEAT_DIV = 4
FRAMES_PER_BAR = 16


# ---------------------------------------------------------------------------
# midi -> per-stream frame representation
# ---------------------------------------------------------------------------

def _file_tempo(pm, default=120.0):
    try:
        _, tempi = pm.get_tempo_changes()
        if len(tempi) > 0 and tempi[0] > 0:
            return float(tempi[0])
    except Exception:
        pass
    return default


class Stream:
    """Frame-quantized view of one stream: onsets and sounding notes."""

    def __init__(self, notes, step, n_frames):
        self.n_frames = n_frames
        self.onsets = [[] for _ in range(n_frames)]     # pitches starting here
        self.sounding = [set() for _ in range(n_frames)]  # pitch classes held
        self.durations = []                              # in frames
        for note in notes:
            f0 = int(round(note.start / step))
            f1 = max(f0 + 1, int(round(note.end / step)))
            if f0 < 0 or f0 >= n_frames:
                continue
            self.onsets[f0].append(note.pitch)
            self.durations.append(min(f1, n_frames) - f0)
            for f in range(f0, min(f1, n_frames)):
                self.sounding[f].add(note.pitch % 12)

    def slice(self, lo, hi):
        out = Stream([], 1.0, hi - lo)
        out.onsets = self.onsets[lo:hi]
        out.sounding = self.sounding[lo:hi]
        out.durations = self.durations   # durations kept whole-file; fine for dists
        return out

    def onset_frames(self):
        return [f for f, ps in enumerate(self.onsets) if ps]

    def n_onsets(self):
        return sum(len(ps) for ps in self.onsets)


def load_streams(paths, task, mel_programs, chord_programs, total_frames):
    """paths: [combined] or [stream_a, stream_b]. Returns (a, b) Streams
    (a = melody/drum, b = chord/nondrum) over frames [0, total_frames)."""
    if len(paths) == 2:
        out = []
        for p in paths:
            pm = pretty_midi.PrettyMIDI(p)
            step = 60.0 / _file_tempo(pm) / BEAT_DIV
            notes = [n for inst in pm.instruments for n in inst.notes]
            out.append(Stream(notes, step, total_frames))
        return out[0], out[1]

    pm = pretty_midi.PrettyMIDI(paths[0])
    step = 60.0 / _file_tempo(pm) / BEAT_DIV
    a_notes, b_notes, unmatched = [], [], []
    named = any((inst.name or '').strip().lower() in ('melody', 'chord')
                for inst in pm.instruments)
    for inst in pm.instruments:
        name = (inst.name or '').strip().lower()
        if named:
            if name == 'melody':
                a_notes.extend(inst.notes)
            elif name == 'chord':
                b_notes.extend(inst.notes)
            else:
                unmatched.append(inst)
        elif task == 'drumnondrum':
            (a_notes if inst.is_drum else b_notes).extend(inst.notes)
        else:
            if inst.program in mel_programs and not inst.is_drum:
                a_notes.extend(inst.notes)
            elif inst.program in chord_programs and not inst.is_drum:
                b_notes.extend(inst.notes)
            else:
                unmatched.append(inst)
    if unmatched:
        print(f'  [warn] {paths[0]}: {len(unmatched)} unmatched instrument(s) '
              f'dropped: {[(i.name, i.program, i.is_drum) for i in unmatched]}',
              file=sys.stderr)
    return (Stream(a_notes, step, total_frames),
            Stream(b_notes, step, total_frames))


# ---------------------------------------------------------------------------
# distribution helpers
# ---------------------------------------------------------------------------

def _norm(h):
    h = np.asarray(h, dtype=float)
    s = h.sum()
    return h / s if s > 0 else np.full_like(h, 1.0 / len(h))


def jsd(p, q):
    """Jensen-Shannon divergence (base 2, in [0,1])."""
    p, q = _norm(p), _norm(q)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hist(values, n_bins, cap=None):
    h = np.zeros(n_bins)
    for v in values:
        v = min(v, cap) if cap is not None else v
        if 0 <= v < n_bins:
            h[int(v)] += 1
    return h


# ---------------------------------------------------------------------------
# H3 -- stream-appropriate grammar (PRIMARY hypothesis block)
# ---------------------------------------------------------------------------

def chord_change_intervals(chord):
    """Frames between successive changes of the sounding chord pc-set."""
    changes = []
    prev = None
    for f in range(chord.n_frames):
        cur = frozenset(chord.sounding[f])
        if cur and cur != prev:
            changes.append(f)
            prev = cur
        elif cur:
            prev = cur
    return list(np.diff(changes)) if len(changes) >= 2 else []


def melody_line(mel):
    """(frame, pitch) of the top onset pitch per melody onset frame."""
    return [(f, max(ps)) for f, ps in enumerate(mel.onsets) if ps]


def h3_metrics(gen_a, gen_b, ref_a, ref_b, task):
    out = {}
    if task == 'melchord':
        g_int = chord_change_intervals(gen_b)
        r_int = chord_change_intervals(ref_b)
        out['harmonic_rhythm_jsd'] = (
            jsd(hist(g_int, 33, cap=32), hist(r_int, 33, cap=32))
            if g_int and r_int else float('nan'))

        def stepwise_rate(mel):
            line = melody_line(mel)
            if len(line) < 2:
                return float('nan')
            steps = [abs(b[1] - a[1]) for a, b in zip(line, line[1:])]
            return float(np.mean([s <= 2 for s in steps]))

        g_sw, r_sw = stepwise_rate(gen_a), stepwise_rate(ref_a)
        out['mel_stepwise_delta'] = g_sw - r_sw

    for label, g, r in (('a', gen_a, ref_a), ('b', gen_b, ref_b)):
        g_pos = [f % FRAMES_PER_BAR for f in g.onset_frames()]
        r_pos = [f % FRAMES_PER_BAR for f in r.onset_frames()]
        out[f'onset_grid_jsd_{label}'] = (
            jsd(hist(g_pos, FRAMES_PER_BAR), hist(r_pos, FRAMES_PER_BAR))
            if g_pos and r_pos else float('nan'))
        out[f'duration_jsd_{label}'] = (
            jsd(hist(g.durations, 33, cap=32), hist(r.durations, 33, cap=32))
            if g.durations and r.durations else float('nan'))
    return out


# ---------------------------------------------------------------------------
# H2 -- inter-stream fit
# ---------------------------------------------------------------------------

def chord_tone_coverage(mel, chord):
    """Fraction of melody onset notes whose pc is in the concurrently
    sounding chord pc-set (over frames where a chord sounds at all)."""
    hits = total = 0
    for f, ps in enumerate(mel.onsets):
        if not ps or not chord.sounding[f]:
            continue
        for p in ps:
            total += 1
            if p % 12 in chord.sounding[f]:
                hits += 1
    return hits / total if total else float('nan')


def _melody_seq(mel):
    """Ordered (frame, pitch) of the melody line: top note per onset frame."""
    return [(f, max(ps)) for f, ps in enumerate(mel.onsets) if ps]


def ctnctr(mel, chord):
    """Chord-tone / non-chord-tone ratio (Yeh et al. 2021).

    (n_chord_tones + n_proper_nonchord) / (n_chord_tones + n_nonchord),
    where a non-chord tone is PROPER when the next melody note resolves
    it by step (<= 2 semitones). Distinguishes passing tones from
    genuine harmonic clashes, which plain coverage cannot.
    """
    seq = _melody_seq(mel)
    n_c = n_n = n_p = 0
    for i, (f, p) in enumerate(seq):
        if not chord.sounding[f]:
            continue
        if p % 12 in chord.sounding[f]:
            n_c += 1
        else:
            n_n += 1
            if i + 1 < len(seq) and abs(seq[i + 1][1] - p) <= 2:
                n_p += 1
    denom = n_c + n_n
    return (n_c + n_p) / denom if denom else float('nan')


# Interval consonance (mod-12 distance melody-pc minus chord-pc):
# unison/3rds/P5/6ths +1, perfect 4th 0, everything else -1.
_PCS_SCORE = {0: 1.0, 3: 1.0, 4: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 5: 0.0}


def pcs(mel, chord):
    """Pitch consonance score (Yeh et al. 2021), duration-weighted.

    Every (sounding melody pc, sounding chord pc) pair on every frame
    contributes one consonance score; iterating frames IS the duration
    weighting. Range [-1, 1], higher = more consonant.
    """
    total = w = 0.0
    for f in range(mel.n_frames):
        if not mel.sounding[f] or not chord.sounding[f]:
            continue
        for mp in mel.sounding[f]:
            for cp in chord.sounding[f]:
                total += _PCS_SCORE.get((mp - cp) % 12, -1.0)
                w += 1.0
    return total / w if w else float('nan')


def _tonal_centroid(pcs_set):
    """Harte (2006) 6-D tonal centroid of a pitch-class set.

    Three circles -- fifths (r=1), minor thirds (r=1), major thirds
    (r=0.5) -- averaged over the set's chroma.
    """
    v = np.zeros(6)
    for pc in pcs_set:
        v += np.array([
            math.sin(pc * 7 * math.pi / 6), math.cos(pc * 7 * math.pi / 6),
            math.sin(pc * 3 * math.pi / 2), math.cos(pc * 3 * math.pi / 2),
            0.5 * math.sin(pc * 2 * math.pi / 3),
            0.5 * math.cos(pc * 2 * math.pi / 3),
        ])
    return v / len(pcs_set)


def mctd(mel, chord):
    """Melody-chord tonal distance (Yeh et al. 2021), duration-weighted.

    Per-frame Euclidean distance between the tonal centroids of the
    sounding melody pcs and the sounding chord pcs, averaged over
    frames where both sound. Lower = closer harmony; unlike coverage it
    grades HOW far a clash is, not just whether one exists.
    """
    dists = []
    for f in range(mel.n_frames):
        if mel.sounding[f] and chord.sounding[f]:
            dists.append(float(np.linalg.norm(
                _tonal_centroid(mel.sounding[f])
                - _tonal_centroid(chord.sounding[f]))))
    return float(np.mean(dists)) if dists else float('nan')


def _shift_stream(s, frames):
    """Circularly shift a stream in time (the mismatched-pair control)."""
    out = Stream([], 1.0, s.n_frames)
    k = frames % max(s.n_frames, 1)
    out.onsets = s.onsets[-k:] + s.onsets[:-k]
    out.sounding = s.sounding[-k:] + s.sounding[:-k]
    out.durations = s.durations
    return out


def harmonic_coupling(mel, chord, shift=None):
    """Coverage on the TRUE pairing minus coverage on a 2-bar-shifted one.

    Absolute consistency is gameable: a stream of all-purpose harmony
    (or a melody of endless chord tones) scores high coverage against
    ANY chord track. Coupling asks the question the query slots exist
    to answer -- does the chord track fit THIS melody at THIS moment
    better than the same music two bars away? Generic output scores
    ~0; genuinely co-ordinated streams score positive.
    """
    shift = 2 * FRAMES_PER_BAR if shift is None else shift
    true = chord_tone_coverage(mel, chord)
    ctrl = chord_tone_coverage(mel, _shift_stream(chord, shift))
    if math.isnan(true) or math.isnan(ctrl):
        return float('nan')
    return true - ctrl


def onset_synchrony(lead, follow, window=1):
    """Fraction of follower onset frames within +-window of a leader onset."""
    lead_frames = set(lead.onset_frames())
    fol = follow.onset_frames()
    if not fol:
        return float('nan')
    ok = sum(1 for f in fol
             if any((f + d) in lead_frames for d in range(-window, window + 1)))
    return ok / len(fol)


def h2_metrics(gen_a, gen_b, ref_a, ref_b, task):
    out = {}
    if task == 'melchord':
        for name, fn in (('chord_tone_cov', chord_tone_coverage),
                         ('ctnctr', ctnctr), ('pcs', pcs), ('mctd', mctd),
                         ('coupling', harmonic_coupling)):
            g = fn(gen_a, gen_b)
            r = fn(ref_a, ref_b)
            out[name] = g
            out[name + '_ref'] = r
            out[name + '_delta'] = g - r
    else:
        g = onset_synchrony(gen_a, gen_b)
        r = onset_synchrony(ref_a, ref_b)
        out['onset_sync'] = g
        out['onset_sync_ref'] = r
        out['onset_sync_delta'] = g - r
    return out


# ---------------------------------------------------------------------------
# H1 -- per-stream role & texture integrity
# ---------------------------------------------------------------------------

def h1_metrics(gen_a, gen_b, ref_a, ref_b, task):
    out = {}
    n_bars = gen_a.n_frames // FRAMES_PER_BAR

    def per_bar_density(s):
        d = np.zeros(n_bars)
        for f in s.onset_frames():
            if f // FRAMES_PER_BAR < n_bars:
                d[f // FRAMES_PER_BAR] += len(s.onsets[f])
        return d

    survivals = []
    for label, g, r in (('a', gen_a, ref_a), ('b', gen_b, ref_b)):
        gd, rd = per_bar_density(g), per_bar_density(r)
        out[f'density_ratio_{label}'] = (
            float(gd.mean() / rd.mean()) if rd.mean() > 0 else float('nan'))
        if gd.mean() > 0:
            slope = np.polyfit(np.arange(n_bars), gd, 1)[0]
            out[f'density_drift_{label}'] = float(slope / gd.mean())
        else:
            out[f'density_drift_{label}'] = float('nan')
        out[f'empty_rate_{label}'] = float(
            np.mean([len(ps) == 0 for ps in g.onsets]))
        active = np.flatnonzero(gd > 0)
        survivals.append(float(len(active) / n_bars) if n_bars else float('nan'))
        out[f'survival_{label}'] = survivals[-1]
    out['survival_min'] = min(survivals)

    # melody polyphony (role check; melchord only -- drums are polyphonic)
    if task == 'melchord':
        onset_frames = [ps for ps in gen_a.onsets if ps]
        out['mel_poly_rate'] = (
            float(np.mean([len(ps) >= 2 for ps in onset_frames]))
            if onset_frames else float('nan'))

    # register overlap between streams, vs the reference's overlap
    def register_overlap(sa, sb):
        pa = [p for ps in sa.onsets for p in ps]
        pb = [p for ps in sb.onsets for p in ps]
        if not pa or not pb:
            return float('nan')
        lo_a, hi_a = np.percentile(pa, [10, 90])
        lo_b, hi_b = np.percentile(pb, [10, 90])
        inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
        union = max(hi_a, hi_b) - min(lo_a, lo_b)
        return float(inter / union) if union > 0 else float('nan')

    g_ov = register_overlap(gen_a, gen_b)
    r_ov = register_overlap(ref_a, ref_b)
    out['register_overlap_delta'] = g_ov - r_ov
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

PRIMARY = {
    'melchord': {'H3': ['harmonic_rhythm_jsd'],
                 'H2': ['chord_tone_cov_delta'],
                 'H1': ['survival_min']},
    'drumnondrum': {'H3': ['onset_grid_jsd_b'],
                    'H2': ['onset_sync_delta'],
                    'H1': ['survival_min']},
}

H_GROUPS = {
    'H3': ['harmonic_rhythm_jsd', 'mel_stepwise_delta',
           'onset_grid_jsd_a', 'onset_grid_jsd_b',
           'duration_jsd_a', 'duration_jsd_b'],
    'H2': ['chord_tone_cov', 'chord_tone_cov_ref', 'chord_tone_cov_delta',
           'ctnctr', 'ctnctr_ref', 'ctnctr_delta',
           'pcs', 'pcs_ref', 'pcs_delta',
           'mctd', 'mctd_ref', 'mctd_delta',
           'coupling', 'coupling_ref', 'coupling_delta',
           'onset_sync', 'onset_sync_ref', 'onset_sync_delta'],
    'H1': ['survival_min', 'survival_a', 'survival_b', 'mel_poly_rate',
           'density_ratio_a', 'density_ratio_b',
           'density_drift_a', 'density_drift_b',
           'empty_rate_a', 'empty_rate_b', 'register_overlap_delta'],
}


# ---------------------------------------------------------------------------
# Which stream each metric describes.
#
# E3 needs this. In a conditional run the partner stream is GIVEN, and every
# conditional decoder copies it into the output verbatim -- A.2's
# make_actions_conditional returns ('given', condition[:, t, :]) for every
# frame, C.1/C.2 set m_tokens = drum_tokens[:, t, :] inside the AR loop. So a
# metric computed on the given stream measures the same copied ground truth
# for every system: those rows are IDENTICAL across systems by construction
# and say nothing about any model. Without this map the E3 table prints ~9 of
# its ~20 rows identical for every column, which reads as "the eval is
# broken" when it is the experiment's design.
#
#   'a'   melody / drum stream
#   'b'   chord / nondrum stream
#   'ref' reference-only statistic (identical for every system in any run)
#   None  cross-stream: depends on both, so the generated side moves it
# ---------------------------------------------------------------------------
STREAM_OF = {
    'harmonic_rhythm_jsd': 'b',
    'mel_stepwise_delta': 'a',
    'onset_grid_jsd_a': 'a', 'onset_grid_jsd_b': 'b',
    'duration_jsd_a': 'a', 'duration_jsd_b': 'b',
    'chord_tone_cov': None, 'chord_tone_cov_delta': None,
    'chord_tone_cov_ref': 'ref',
    'ctnctr': None, 'ctnctr_delta': None, 'ctnctr_ref': 'ref',
    'pcs': None, 'pcs_delta': None, 'pcs_ref': 'ref',
    'mctd': None, 'mctd_delta': None, 'mctd_ref': 'ref',
    'coupling': None, 'coupling_delta': None, 'coupling_ref': 'ref',
    'onset_sync': None, 'onset_sync_delta': None, 'onset_sync_ref': 'ref',
    'survival_min': None, 'survival_a': 'a', 'survival_b': 'b',
    'mel_poly_rate': 'a',
    'density_ratio_a': 'a', 'density_ratio_b': 'b',
    'density_drift_a': 'a', 'density_drift_b': 'b',
    'empty_rate_a': 'a', 'empty_rate_b': 'b',
    'register_overlap_delta': None,
}

# Which stream is GIVEN in each conditional mode. 'co' (E1) gives neither.
# drum2nondrum is B.1/C.1/C.2's own label for the mel2chord direction.
GIVEN_STREAM_BY_MODE = {
    'mel2chord': 'a',
    'drum2nondrum': 'a',
    'chord2mel': 'b',
}


def score_pair(gen_paths, ref_paths, args):
    lo, hi = args.prompt_frames, args.total_frames
    gen_a, gen_b = load_streams(gen_paths, args.task,
                                args.mel_programs, args.chord_programs, hi)
    ref_a, ref_b = load_streams(ref_paths, args.task,
                                args.mel_programs, args.chord_programs, hi)
    ga, gb = gen_a.slice(lo, hi), gen_b.slice(lo, hi)
    ra, rb = ref_a.slice(lo, hi), ref_b.slice(lo, hi)
    row = {}
    row.update(h3_metrics(ga, gb, ra, rb, args.task))
    row.update(h2_metrics(ga, gb, ra, rb, args.task))
    row.update(h1_metrics(ga, gb, ra, rb, args.task))
    return row


def summarize(rows, task):
    by_system = {}
    for r in rows:
        by_system.setdefault(r.get('system', '?'), []).append(r)
    print('\n================= SUMMARY (priority order: H3 > H2 > H1) '
          '=================')
    for h in ('H3', 'H2', 'H1'):
        print(f'\n--- {h} ---')
        keys = [k for k in H_GROUPS[h]
                if any(k in r and not _is_nan(r[k]) for r in rows)]
        header = 'metric'.ljust(26) + ''.join(
            s.ljust(14) for s in sorted(by_system))
        print(header)
        for k in keys:
            star = '*' if k in PRIMARY[task][h] else ' '
            line = (star + k).ljust(26)
            for s in sorted(by_system):
                vals = [r[k] for r in by_system[s]
                        if k in r and not _is_nan(r[k])]
                line += (f'{np.mean(vals):+.3f}'.ljust(14)
                         if vals else '--'.ljust(14))
            print(line)
    print('\n(* = pre-registered primary endpoint; deltas/JSD: closer to 0 '
          'is better; ratios: closer to 1 is better)')


def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['melchord', 'drumnondrum'])
    p.add_argument('--gen', help='generated midi (combined or "a;b" pair)')
    p.add_argument('--ref-a', help='reference stream-a midi (melody/drum)')
    p.add_argument('--ref-b', help='reference stream-b midi (chord/nondrum)')
    p.add_argument('--manifest',
                   help='TSV: system mode song sample path -- batch scoring')
    p.add_argument('--ref-a-dir', help='dir of <song>.mid stream-a refs')
    p.add_argument('--ref-b-dir', help='dir of <song>.mid stream-b refs')
    p.add_argument('--out', default=None, help='CSV output path')
    p.add_argument('--prompt-frames', type=int, default=64)
    p.add_argument('--total-frames', type=int, default=384)
    p.add_argument('--mel-programs', default='0,24',
                   help='program fallback for un-named melody tracks')
    p.add_argument('--chord-programs', default='48',
                   help='program fallback for un-named chord tracks')
    args = p.parse_args()
    args.mel_programs = {int(x) for x in args.mel_programs.split(',')}
    args.chord_programs = {int(x) for x in args.chord_programs.split(',')}

    rows = []
    if args.manifest:
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
                    row = score_pair(gen_paths, ref_paths, args)
                except Exception as e:
                    print(f'  [fail] {path}: {e!r}', file=sys.stderr)
                    continue
                row.update(system=system, mode=mode, song=song, sample=sample)
                rows.append(row)
    else:
        if not (args.gen and args.ref_a and args.ref_b):
            raise SystemExit('need --gen/--ref-a/--ref-b, or --manifest')
        row = score_pair(args.gen.split(';'), [args.ref_a, args.ref_b], args)
        row.update(system='single', mode='-', song='-', sample='0')
        rows.append(row)
        for k in ('H3', 'H2', 'H1'):
            pass

    if args.out:
        keys = ['system', 'mode', 'song', 'sample'] + [
            k for h in ('H3', 'H2', 'H1') for k in H_GROUPS[h]]
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {len(rows)} rows -> {args.out}')

    summarize(rows, args.task)


if __name__ == '__main__':
    main()
