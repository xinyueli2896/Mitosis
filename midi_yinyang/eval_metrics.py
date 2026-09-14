"""Quantitative evaluation metrics for E1 co-generation (EXPERIMENTS.md).

Covers hypotheses H1-H3 plus the S, R and P blocks (non-pre-registered,
added later); H4 is the listening test and is out of scope.
The summary prints blocks in the order H3 > H2 > P > S > R, and each
has ONE designated primary endpoint (marked *), the rest being
supporting diagnostics.

H1 IS WRITTEN TO THE CSV BUT NOT PRINTED (2026-09-14): P took its place
in the table. The CSV is the superset of everything computed; the table
is the reading order. Check the H1 columns before quoting P on any
system -- survival_* and empty_rate_* are what say a stream is there at
all, and a continuation that fell silent adheres to nothing, so its P
statistics are meaningless rather than bad.

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
  P (prompt adherence, per stream, each _delta against the ground-truth
     continuation of the SAME prompt)
    * reuse_vs_prompt_a_delta  beats of the continuation restating
                            prompt material exactly (same pitches, same
                            frames), minus the reference's rate
      pc_jsd                pitch-class distribution vs the prompt's
      density               onset rate over the prompt's
      register              mean onset pitch minus the prompt's
      rhythm_reuse          beats restating a prompt RHYTHM, pitch
                            discarded -- reuse without the pitch
                            requirement, so always at least as large
      grid_jsd              metrical-position histogram vs the prompt's:
                            pc_jsd is pitch mod the octave, this is time
                            mod the bar
      ioi_jsd               inter-onset-interval distribution vs prompt
      dur_jsd               note-duration distribution vs prompt
  S (stream expertise & role separation, added for E6)
  R (repetition/diversity)
      ubr<n>_<mode>_<s>     unique beat ratio over n-beat intervals,
                            'onset' or full 'state', with _delta vs ref.
                            Neither extreme is good, so read the delta.

  H1 (per-stream role & texture integrity)  CSV ONLY, not printed
    * survival_min          min over streams of active-bar fraction
      mel_poly_rate         melody frames with >=2 simultaneous onsets
      density_ratio_<s>     mean notes/bar over ref's mean
      density_drift_<s>     per-bar density slope (normalized)
      empty_rate_<s>        silent-frame fraction (collapse detector)
      register_overlap_delta  stream register overlap minus ref overlap

All metrics are computed on the CONTINUATION only (frames
--prompt-frames .. --total-frames, default 64..384) on a 16th-note
grid derived from the file's tempo (beat_div=4, 16 frames/bar).

Every JSD is reported twice. The per-song table averages a song's
samples, then takes mean +- std over songs; the CORPUS-POOLED table
(--pooled-out, --pooled-boot) sums the per-song histograms into one
histogram per system and brackets it with a percentile bootstrap over
songs. The two differ substantially and on purpose: a per-song
histogram holds ~100 onsets, where the JSD estimator carries an upward
bias of ~0.06 at 16 bins even between identical distributions, and the
bias grows as a stream thins, so a sparse stream is charged for being
sparse. Pooling ~10,000 onsets drops that bias below 0.001 but forfeits
per-song calibration: a system can match the corpus histogram while
missing every individual song. Quote both.

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
    """Main tempo (tick-weighted median), used only for reporting: frames
    are computed on the tick grid (frame_fn)."""
    from midi_tempo import main_tempo_pm
    try:
        return float(main_tempo_pm(pm, default=default))
    except Exception:
        pass
    return default


def frame_fn(pm):
    """seconds -> frames on the file's BEAT grid, via its tick map.

    The tokenizer (xf_midi.XFMidi with constant_tempo) reads note
    positions from ticks and ignores tempo events, so training data and
    prompts live on the tick grid whatever the tempo map says. Scoring
    used to convert seconds with the file's FIRST tempo event instead;
    POP909 files often carry a spurious initial tempo (1323 bpm on
    song 046), which put the whole scored window inside the first few
    seconds of the reference and produced density ratios of 5x, JSDs
    near their maximum and NaN rows for every system at once. Framing
    by ticks is exact for constant-tempo files (every generated file)
    and correct for real tempo maps (the references).
    """
    ticks_per_frame = pm.resolution / BEAT_DIV
    return lambda t: pm.time_to_tick(t) / ticks_per_frame


class Stream:
    """Frame-quantized view of one stream: onsets and sounding notes."""

    def __init__(self, notes, step, n_frames):
        """step: seconds per frame (float) or a callable seconds -> frames."""
        self.n_frames = n_frames
        self.onsets = [[] for _ in range(n_frames)]     # pitches starting here
        self.sounding = [set() for _ in range(n_frames)]  # pitch classes held
        # Absolute pitches held. `sounding` is deliberately pitch-CLASS
        # (the chord-tone metrics want that); the unique-beat-ratio's
        # "full state" criterion is a piano-roll state and needs real
        # pitches, so both are kept.
        self.sounding_abs = [set() for _ in range(n_frames)]
        self.durations = []                              # in frames
        # durations indexed by ONSET frame, so slice() can restrict them
        # to a window. Without this a sliced stream carries the whole
        # file's durations and any duration statistic silently scores
        # the prompt along with the continuation.
        self.dur_at = [[] for _ in range(n_frames)]
        to_frame = step if callable(step) else (lambda t: t / step)
        for note in notes:
            f0 = int(round(to_frame(note.start)))
            f1 = max(f0 + 1, int(round(to_frame(note.end))))
            if f0 < 0 or f0 >= n_frames:
                continue
            self.onsets[f0].append(note.pitch)
            self.durations.append(min(f1, n_frames) - f0)
            self.dur_at[f0].append(min(f1, n_frames) - f0)
            for f in range(f0, min(f1, n_frames)):
                self.sounding[f].add(note.pitch % 12)
                self.sounding_abs[f].add(note.pitch)

    def slice(self, lo, hi):
        out = Stream([], 1.0, hi - lo)
        out.onsets = self.onsets[lo:hi]
        out.sounding = self.sounding[lo:hi]
        out.sounding_abs = self.sounding_abs[lo:hi]
        # Durations of the notes that START in the window. These used to
        # be the whole file's, which put the PROMPT into every duration
        # statistic: the prompt is byte-identical between a system and
        # the reference, so it pulled duration_jsd toward 0 by the
        # prompt's share of the notes, equally for every system, and it
        # made any duration-vs-prompt comparison identically 0.
        out.dur_at = self.dur_at[lo:hi]
        out.durations = [d for ds in out.dur_at for d in ds]
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
            notes = [n for inst in pm.instruments for n in inst.notes]
            out.append(Stream(notes, frame_fn(pm), total_frames))
        return out[0], out[1]

    pm = pretty_midi.PrettyMIDI(paths[0])
    step = frame_fn(pm)
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
# Binning for every histogram-divergence metric, in ONE place.
#
# Both the per-song JSD and the corpus-pooled JSD read their bins from
# here, so the two cannot drift apart: a pooled number is then the same
# measurement as the per-song one, taken over a bigger sample, and not a
# second metric that happens to share a name.
#
#   metric -> (n_bins, cap)
# ---------------------------------------------------------------------------
POOLED_BINS = {
    'harmonic_rhythm_jsd': (33, 32),        # frames between chord changes
    'onset_grid_jsd_a': (FRAMES_PER_BAR, None),
    'onset_grid_jsd_b': (FRAMES_PER_BAR, None),
    'duration_jsd_a': (33, 32),
    'duration_jsd_b': (33, 32),
    'mel_interval_jsd': (25, None),         # melodic interval + 12, |i| <= 12
    'voicing_jsd': (8, None),               # chord size - 1, capped at 8
}


def _ph(metric, values):
    """Histogram of `values` under `metric`'s registered binning."""
    n_bins, cap = POOLED_BINS[metric]
    return hist(values, n_bins, cap=cap)


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


def _grid_values(s):
    """Within-bar position of every onset."""
    return [f % FRAMES_PER_BAR for f in s.onset_frames()]


def h3_metrics(gen_a, gen_b, ref_a, ref_b, task):
    out = {}
    if task == 'melchord':
        g_int = chord_change_intervals(gen_b)
        r_int = chord_change_intervals(ref_b)
        out['harmonic_rhythm_jsd'] = (
            jsd(_ph('harmonic_rhythm_jsd', g_int),
                _ph('harmonic_rhythm_jsd', r_int))
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
        k_pos, k_dur = f'onset_grid_jsd_{label}', f'duration_jsd_{label}'
        g_pos = _grid_values(g)
        r_pos = _grid_values(r)
        out[k_pos] = (jsd(_ph(k_pos, g_pos), _ph(k_pos, r_pos))
                      if g_pos and r_pos else float('nan'))
        out[k_dur] = (jsd(_ph(k_dur, g.durations), _ph(k_dur, r.durations))
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

FRAMES_PER_BEAT = FRAMES_PER_BAR // 4


def _beat_sigs(s, beats=1, mode='onset'):
    """One hashable signature per beat interval of a stream.

    mode 'onset' = pitches that start in each frame, 'state' = pitches
    sounding in each frame, 'rhythm' = whether anything starts in each
    frame, pitch discarded. The last one turns a beat into its rhythmic
    figure alone, so two beats carrying the same pattern on different
    notes count as the same signature.
    """
    w = beats * FRAMES_PER_BEAT
    if mode == 'rhythm':
        return [tuple(bool(s.onsets[f]) for f in range(i * w, i * w + w))
                for i in range(s.n_frames // w)]
    field = s.onsets if mode == 'onset' else s.sounding_abs
    return [tuple(tuple(sorted(field[f])) for f in range(i * w, i * w + w))
            for i in range(s.n_frames // w)]


def unique_beat_ratio(s, beats=1, mode='onset'):
    """Fraction of beat intervals whose content is seen for the FIRST time.

    From BEAT (Qian et al.), sec. 4.5: segment the piano roll into
    beat-wise intervals, call an interval unique if it has not appeared
    before, and divide the cumulative unique count by the number of
    intervals. Near 1.0 is high diversity, lower is more repetition.
    Neither extreme is good, which is why we report it against the
    reference rather than alone: real music sits in the middle, and a
    model can miss by being too repetitive OR too scattered.

    beats: interval width (1 or 2). mode: 'onset' compares only what
    STARTS in each frame, 'state' compares everything SOUNDING, so a
    held note makes two intervals differ that onset-only calls equal.

    An empty interval is a signature like any other, so a stream that
    falls silent scores as highly repetitive. That is not wrong -- it is
    repetitive -- but read it beside survival_*, which says whether the
    stream is there at all.
    """
    sigs = _beat_sigs(s, beats, mode)
    if not sigs:
        return float('nan')
    seen, uniq = set(), 0
    for sig in sigs:
        if sig not in seen:
            seen.add(sig)
            uniq += 1
    return uniq / len(sigs)


def repetition_metrics(gen_a, gen_b, ref_a, ref_b):
    """Unique beat ratio per stream, with the reference delta."""
    out = {}
    for label, g, r in (('a', gen_a, ref_a), ('b', gen_b, ref_b)):
        for beats in (1, 2):
            for mode in ('onset', 'state'):
                key = f'ubr{beats}_{mode}_{label}'
                gv = unique_beat_ratio(g, beats, mode)
                rv = unique_beat_ratio(r, beats, mode)
                out[key] = float(gv)
                out[f'{key}_delta'] = float(gv - rv)
    return out


def _pc_hist(s):
    return hist([p % 12 for ps in s.onsets for p in ps], 12)


def _mean_pitch(s):
    ps = [p for row in s.onsets for p in row]
    return float(np.mean(ps)) if ps else float('nan')


def _rate(s):
    return s.n_onsets() / s.n_frames if s.n_frames else float('nan')


def _iois(s):
    """Frames between successive onset frames."""
    fs = s.onset_frames()
    return [b - a for a, b in zip(fs, fs[1:])]


def _grid_hist(s, phase=0):
    """Metrical-position histogram: where in the bar the onsets fall.

    phase is the sliced stream's offset from the bar line, so a prompt
    and a continuation taken from different windows are compared on the
    same bar positions. It is 0 whenever the split is on a bar line,
    which every configured PROMPT_LENGTH is.
    """
    return hist([(f + phase) % FRAMES_PER_BAR for f in s.onset_frames()],
                FRAMES_PER_BAR)


def _prompt_stats(prompt, cont, phase=0):
    """How much a continuation follows on from its prompt.

    Four pitch-side statistics, each a different sense of "follows on":

      reuse       fraction of the continuation's beat intervals whose
                  content already appears in the prompt. Literal
                  restatement of prompt material, using the signatures
                  of unique_beat_ratio. Joint: a beat matches only if
                  the same pitches start on the same frames.
      pc_jsd      divergence between prompt and continuation pitch-class
                  distributions. Rises when the continuation leaves the
                  prompt's key.
      density     onsets per frame in the continuation over the same in
                  the prompt. 1.0 keeps the prompt's activity.
      register    mean onset pitch of the continuation minus the
                  prompt's, in semitones.

    and four rhythm-side ones, since a continuation can hold the
    prompt's key and register while abandoning its rhythm entirely, and
    density alone only counts onsets, never their placement:

      rhythm_reuse  reuse over the prompt's rhythmic FIGURES: the same
                    beat signatures with pitch discarded, so a beat
                    matches when the pattern of attacks recurs on other
                    notes. reuse is a lower bound on this by
                    construction.
      grid_jsd      metrical-position histogram vs the prompt's. The
                    rhythmic counterpart of pc_jsd -- that one is pitch
                    modulo the octave, this is time modulo the bar --
                    and it is what moves when a continuation drifts off
                    the prompt's beat or starts syncopating.
      ioi_jsd       inter-onset-interval distribution vs the prompt's:
                    pacing, independent of where in the bar the onsets
                    land.
      dur_jsd       note-duration distribution vs the prompt's. For a
                    legato monophonic line this tracks ioi_jsd; for the
                    chord stream, whose rhythm is mostly sustain, the
                    two part company.

    All eight are reported against the GROUND-TRUTH continuation of the
    same prompt, because none has a good absolute value: a real
    continuation neither copies its prompt nor ignores it, and only the
    reference says where between those the song actually sat.
    """
    out = {}
    sig_p = set(_beat_sigs(prompt, 1, 'onset'))
    sig_c = _beat_sigs(cont, 1, 'onset')
    out['reuse'] = (sum(1 for g in sig_c if g in sig_p) / len(sig_c)
                    if sig_c else float('nan'))
    out['pc_jsd'] = jsd(_pc_hist(cont), _pc_hist(prompt))
    rp, rc = _rate(prompt), _rate(cont)
    out['density'] = float(rc / rp) if rp else float('nan')
    out['register'] = _mean_pitch(cont) - _mean_pitch(prompt)

    rsig_p = set(_beat_sigs(prompt, 1, 'rhythm'))
    rsig_c = _beat_sigs(cont, 1, 'rhythm')
    out['rhythm_reuse'] = (sum(1 for g in rsig_c if g in rsig_p) / len(rsig_c)
                           if rsig_c else float('nan'))
    # Same binning as the h3 counterparts (POOLED_BINS), so a
    # prompt-adherence divergence is on the same scale as the
    # reference-divergence of the same quantity.
    out['grid_jsd'] = jsd(_grid_hist(cont, phase), _grid_hist(prompt))
    i_c, i_p = _iois(cont), _iois(prompt)
    out['ioi_jsd'] = (jsd(hist(i_c, 33, cap=32), hist(i_p, 33, cap=32))
                      if i_c and i_p else float('nan'))
    out['dur_jsd'] = (jsd(_ph('duration_jsd_a', cont.durations),
                          _ph('duration_jsd_a', prompt.durations))
                      if cont.durations and prompt.durations else float('nan'))
    return out


def prompt_match_metrics(prompt_a, prompt_b, gen_a, gen_b, ref_a, ref_b,
                         phase=0):
    """Prompt-to-continuation coherence, per stream, vs the reference."""
    out = {}
    for label, pr, g, r in (('a', prompt_a, gen_a, ref_a),
                            ('b', prompt_b, gen_b, ref_b)):
        # A system that writes only its continuation would give an empty
        # prompt here and silently score as following nothing; surface
        # the count so that reads as a bug rather than a result.
        out[f'prompt_onsets_{label}'] = pr.n_onsets()
        gs = _prompt_stats(pr, g, phase)
        rs = _prompt_stats(pr, r, phase)
        for k in gs:
            out[f'{k}_vs_prompt_{label}'] = float(gs[k])
            out[f'{k}_vs_prompt_{label}_delta'] = float(gs[k] - rs[k])
    return out


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

# ---------------------------------------------------------------------------
# S -- stream expertise & role separation (added 2026-09-03 for E6).
# (Named 'S', not 'H4': H4 is reserved for the listening test in
# EXPERIMENTS.md's pre-registered hypothesis list.)
#
# The E6 mechanism claim is not "the music is better overall" but "each
# stream gets better at ITS OWN idiom while the shared/coupling side is
# preserved". H4 measures that directly:
#   * own-idiom fidelity, one per stream, beyond the H3 timing rows:
#       mel_interval_jsd  melodic-interval distribution vs the reference
#                         melody (semitone steps of the top line, |i|<=12)
#       voicing_jsd       chord-stream voicing-size distribution vs the
#                         reference chords (simultaneously sounding
#                         pitch-classes per non-empty frame, capped at 8)
#   * role separation -- do the two streams KEEP their distinct roles,
#     to the same degree the reference pair does? Each is a log2 contrast
#     between the streams, reported as gen minus ref (target 0):
#       dur_contrast_delta      log2(median dur_b / median dur_a)
#       density_contrast_delta  log2(onsets_a / onsets_b)
# NOT pre-registered: this block was added after the first E6 scoring
# pass, chosen for the mechanism hypothesis, and must be reported as
# such.
# ---------------------------------------------------------------------------

def _interval_values(mel, span=12):
    """Melodic intervals, clipped to +-span and shifted to bin indices."""
    seq = [p for _, p in _melody_seq(mel)]
    return [max(-span, min(span, b - a)) + span for a, b in zip(seq, seq[1:])]


def _voicing_values(chord, cap=8):
    """Sounding chord sizes, capped, as bin indices."""
    return [min(len(s), cap) - 1 for s in chord.sounding if s]


def _interval_hist(mel, span=12):
    return _ph('mel_interval_jsd', _interval_values(mel, span))


def _voicing_hist(chord, cap=8):
    return _ph('voicing_jsd', _voicing_values(chord, cap))


def _log2_contrast(num, den):
    return math.log2(max(num, 1e-6) / max(den, 1e-6))


def s_metrics(gen_a, gen_b, ref_a, ref_b, task):
    out = {}
    out['mel_interval_jsd'] = jsd(_interval_hist(gen_a),
                                  _interval_hist(ref_a))
    out['voicing_jsd'] = jsd(_voicing_hist(gen_b), _voicing_hist(ref_b))

    def med_dur(s):
        return float(np.median(s.durations)) if s.durations else float('nan')

    def contrast_pair(g_val, r_val):
        if any(math.isnan(v) for v in (g_val, r_val)):
            return float('nan')
        return g_val - r_val

    g_dur = _log2_contrast(med_dur(gen_b), med_dur(gen_a)) \
        if gen_a.durations and gen_b.durations else float('nan')
    r_dur = _log2_contrast(med_dur(ref_b), med_dur(ref_a)) \
        if ref_a.durations and ref_b.durations else float('nan')
    out['dur_contrast_delta'] = contrast_pair(g_dur, r_dur)

    g_den = _log2_contrast(gen_a.n_onsets(), gen_b.n_onsets()) \
        if gen_a.n_onsets() and gen_b.n_onsets() else float('nan')
    r_den = _log2_contrast(ref_a.n_onsets(), ref_b.n_onsets()) \
        if ref_a.n_onsets() and ref_b.n_onsets() else float('nan')
    out['density_contrast_delta'] = contrast_pair(g_den, r_den)
    return out


PRIMARY = {
    'melchord': {'H3': ['harmonic_rhythm_jsd'],
                 'H2': ['chord_tone_cov_delta'],
                 'H1': ['survival_min'],
                 'P': ['reuse_vs_prompt_a_delta'],
                 'S': [], 'R': []},
    'drumnondrum': {'H3': ['onset_grid_jsd_b'],
                    'H2': ['onset_sync_delta'],
                    'H1': ['survival_min'],
                    'P': ['reuse_vs_prompt_a_delta'],
                    'S': [], 'R': []},
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
    'S': ['mel_interval_jsd', 'voicing_jsd',
           'dur_contrast_delta', 'density_contrast_delta'],
    # R -- repetition/diversity (unique beat ratio). Neither extreme is
    # good, so only the _delta columns carry a direction; the raw value
    # is kept beside them to show which side of the reference it fell.
    'R': [f'ubr{b}_{m}_{s}{d}'
          for s in ('a', 'b') for b in (1, 2) for m in ('onset', 'state')
          for d in ('', '_delta')],
    # P -- prompt adherence. Four pitch-side statistics then four
    # rhythm-side ones, per stream, each against the reference
    # continuation of the same prompt.
    'P': [f'{k}_vs_prompt_{s}{d}'
          for s in ('a', 'b')
          for k in ('reuse', 'pc_jsd', 'density', 'register',
                    'rhythm_reuse', 'grid_jsd', 'ioi_jsd', 'dur_jsd')
          for d in ('', '_delta')] + ['prompt_onsets_a', 'prompt_onsets_b'],
}

# Print/CSV order. H3 > H2 > H1 is the pre-registered priority; S, R and
# P are diagnostic blocks added later and follow it.
# Printed/aggregated blocks, in priority order. P took H1's slot
# (2026-09-14, by request): the table asks whether a continuation
# follows on from its prompt, not whether the streams stayed alive. Its
# primary endpoint is reuse_vs_prompt_a_delta -- the broadest single
# sense of "follows on", a beat counting as reused only when the same
# pitches start on the same frames -- read against the reference,
# because neither copying the prompt nor ignoring it is right.
GROUP_ORDER = ('H3', 'H2', 'P', 'S', 'R')

# H1 is back in the CSV (2026-09-14, by request: the H1 metrics are
# being plotted), but stays out of the printed table, where P holds its
# place. So the CSV is the superset -- everything computed is recorded
# -- and the table is the reading order. Add 'H1' to GROUP_ORDER to put
# it back in the table too.
CSV_GROUPS = GROUP_ORDER + ('H1',)


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
    'mel_interval_jsd': 'a', 'voicing_jsd': 'b',
    'dur_contrast_delta': None, 'density_contrast_delta': None,
}
# R and P are per-stream by construction: every key carries its stream
# in the suffix, and both families read one stream only. Registering
# them keeps E3 from presenting the GIVEN stream's copied ground truth
# as if it discriminated systems.
for _h in ('R', 'P'):
    for _k in H_GROUPS[_h]:
        _parts = _k.split('_')
        _s = _parts[-1] if _parts[-1] in ('a', 'b') else _parts[-2]
        assert _s in ('a', 'b'), f'no stream suffix on {_k}'
        STREAM_OF[_k] = _s

# Which stream is GIVEN in each conditional mode. 'co' (E1) gives neither.
# drum2nondrum is B.1/C.1/C.2's own label for the mel2chord direction.
GIVEN_STREAM_BY_MODE = {
    'mel2chord': 'a',
    'drum2nondrum': 'a',
    'chord2mel': 'b',
}


# ---------------------------------------------------------------------------
# Corpus-pooled JSD
#
# A per-song JSD compares two histograms built from one song's ~100
# onsets. At that sample size the estimator is badly biased upward --
# about 0.058 at 16 bins even when both sides are drawn from the SAME
# distribution -- and the bias grows as the stream gets sparser, so a
# sparse stream is penalised for being sparse. Pooling the corpus into
# one histogram per system (~10,000 onsets) drops the bias to ~0.0005.
#
# The cost is that pooling discards per-song calibration entirely: a
# system can match the corpus histogram while getting every individual
# song wrong. The two numbers answer different questions and are
# reported side by side, neither replacing the other.
#
# Pooling is a SUM of per-song count vectors, so we store those rather
# than raw values: the bootstrap over songs is then a matrix product
# instead of a re-tabulation.
# ---------------------------------------------------------------------------

def _pooled_hists(gen_a, gen_b, ref_a, ref_b, task):
    """Per-song count vectors for every pooled metric: {k: (gen, ref)}.

    Same extractors and same bins as the per-song JSDs above -- the
    only difference is that these are summed across songs before the
    divergence is taken, not after.

    prompt-match pc_jsd is deliberately absent: it measures a
    continuation against ITS OWN prompt, so pooling it would compare a
    corpus of continuations against a corpus of prompts, which is a
    different quantity and not a lower-variance estimate of this one.
    """
    out = {}
    if task == 'melchord':
        out['harmonic_rhythm_jsd'] = (
            _ph('harmonic_rhythm_jsd', chord_change_intervals(gen_b)),
            _ph('harmonic_rhythm_jsd', chord_change_intervals(ref_b)))
    for label, g, r in (('a', gen_a, ref_a), ('b', gen_b, ref_b)):
        k_pos, k_dur = f'onset_grid_jsd_{label}', f'duration_jsd_{label}'
        out[k_pos] = (_ph(k_pos, _grid_values(g)), _ph(k_pos, _grid_values(r)))
        out[k_dur] = (_ph(k_dur, g.durations), _ph(k_dur, r.durations))
    out['mel_interval_jsd'] = (_ph('mel_interval_jsd', _interval_values(gen_a)),
                               _ph('mel_interval_jsd', _interval_values(ref_a)))
    out['voicing_jsd'] = (_ph('voicing_jsd', _voicing_values(gen_b)),
                          _ph('voicing_jsd', _voicing_values(ref_b)))
    return out


def _norm_rows(H):
    s = H.sum(axis=1, keepdims=True)
    return np.where(s > 0, H / np.where(s > 0, s, 1.0), 1.0 / H.shape[1])


def _jsd_rows(P, Q):
    """jsd() applied row-wise to two stacks of histograms."""
    P, Q = _norm_rows(np.asarray(P, float)), _norm_rows(np.asarray(Q, float))
    M = 0.5 * (P + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        kp = np.where(P > 0, P * np.log2(P / M), 0.0).sum(axis=1)
        kq = np.where(Q > 0, Q * np.log2(Q / M), 0.0).sum(axis=1)
    return 0.5 * (kp + kq)


def _pooled_ci(G, R, n_boot=2000, seed=0, per_sample=None):
    """Percentile CI for the pooled JSD, resampling SONGS.

    G, R are (n_songs, n_bins) count matrices. The song is the unit of
    resampling, not the onset: onsets within a song are not independent,
    and an onset-level bootstrap would report an interval several times
    too narrow. Drawing multinomial weights over songs is the same
    thing as drawing songs with replacement, and lets every replicate
    be one matrix product.

    Near JSD 0 the interval is one-sided: every resample perturbs the
    two pooled histograms apart, and the divergence can only rise. It
    is a boundary effect and is negligible at the divergences these
    metrics actually take.
    """
    n = G.shape[0]
    if n < 2 or n_boot <= 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    if per_sample is None:
        vals = _jsd_rows(w @ G, w @ R)
    else:
        # Two-level (cluster) bootstrap: resample SONGS, then resample
        # each drawn song's SAMPLES with replacement. The one-level
        # version holds a song's three decodes fixed, so its interval
        # covers "would this hold on other songs" but not "would this
        # hold on another decode". This covers both, and is wider.
        # The reference is redrawn on the same song indices either way:
        # holding it at the full corpus would compare two different song
        # sets and add divergence from song composition alone.
        idx = rng.integers(0, n, size=(n_boot, n))
        Gb = np.empty((n_boot, G.shape[1]))
        for b in range(n_boot):
            acc = np.zeros(G.shape[1])
            for i in idx[b]:
                v = per_sample[i]
                k = len(v)
                pick = rng.integers(0, k, size=k)
                acc += np.sum([v[j] for j in pick], axis=0)
            Gb[b] = acc
        Rb = np.stack([R[idx[b]].sum(axis=0) for b in range(n_boot)])
        vals = _jsd_rows(Gb, Rb)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    # The bootstrap SE is the spread of the replicates. It is reported
    # for a table that wants a +- column, but the percentile interval is
    # the better summary: JSD is bounded at 0 and skewed near the floor,
    # so value +- SE can reach below 0, where the metric cannot go.
    return float(lo), float(hi), float(vals.std(ddof=1))


def pooled_summary(rows, task, n_boot=2000, out_csv=None, two_level=False):
    """One histogram per system over the whole corpus, with a song
    bootstrap. Samples of a song are summed into that song's counts."""
    keys = [k for k in POOLED_BINS
            if task == 'melchord' or k != 'harmonic_rhythm_jsd']
    # system -> key -> song -> [gen counts, ref counts]
    acc = {}
    for r in rows:
        pooled = r.get('_pooled')
        if not pooled:
            continue
        sysname, song = r.get('system', '?'), r.get('song', '?')
        per_sys = acc.setdefault(sysname, {})
        for k, (g, ref) in pooled.items():
            by_song = per_sys.setdefault(k, {})
            if song not in by_song:
                # the reference is the same file for every sample of a
                # song, so it is taken once; only the generated side
                # accumulates across samples.
                by_song[song] = [[], np.asarray(ref, float)]
            # per-SAMPLE vectors, not a running sum: a two-level
            # bootstrap has to be able to redraw the samples of a song,
            # and a sum has already thrown that away. The point estimate
            # sums them back, so it is unchanged.
            by_song[song][0].append(np.asarray(g, float))
    if not acc:
        return
    systems = sorted(acc)
    width = 26
    print('\n================= CORPUS-POOLED JSD '
          '=================')
    print('one histogram per system over all songs -- a single value, so '
          'it has no std;')
    _lvl = ('SONGS and, within each, its SAMPLES' if two_level else 'SONGS')
    print(f'the uncertainty is a [2.5, 97.5] percentile bootstrap over '
          f'{_lvl} (boot_se in the CSV)')
    print(f'({n_boot} replicates; pooling removes the per-song '
          'small-sample bias but also removes')
    print(' per-song calibration -- a system can match the corpus '
          'histogram and still get')
    print(' every song wrong, so read this beside the per-song table, '
          'not instead of it.')
    print(' A value at or below its own interval means the divergence '
          'is at the floor: JSD')
    print(' cannot go below 0, so resampling can only push it up.)')
    print('metric'.ljust(26) + ''.join(s.ljust(width) for s in systems))
    recs = []
    for k in keys:
        line = k.ljust(26)
        for sysname in systems:
            by_song = acc[sysname].get(k, {})
            songs = [s for s, (g, r) in by_song.items()
                     if sum(v.sum() for v in g) > 0 and r.sum() > 0]
            n_obs = int(sum(v.sum() for s in songs for v in by_song[s][0]))
            rec = dict(metric=k, system=sysname, jsd=float('nan'),
                       ci_lo=float('nan'), ci_hi=float('nan'),
                       boot_se=float('nan'),
                       n_songs=len(songs), n_obs=n_obs, n_boot=n_boot,
                       boot_levels=2 if two_level else 1)
            recs.append(rec)
            if not songs:
                line += '--'.ljust(width)
                continue
            Gs = [by_song[s][0] for s in songs]
            G = np.stack([np.sum(v, axis=0) for v in Gs])
            R = np.stack([by_song[s][1] for s in songs])
            rec['jsd'] = jsd(G.sum(axis=0), R.sum(axis=0))
            rec['ci_lo'], rec['ci_hi'], rec['boot_se'] = _pooled_ci(
                G, R, n_boot=n_boot, per_sample=Gs if two_level else None)
            cell = f'{rec["jsd"]:.3f}' if math.isnan(rec['ci_lo']) \
                else f'{rec["jsd"]:.3f} [{rec["ci_lo"]:.3f},{rec["ci_hi"]:.3f}]'
            line += cell.ljust(width)
        print(line)
    # What each pooled histogram rests on, per metric: the whole case
    # for pooling is a claim about these counts, and they differ by
    # metric -- a melody with few onsets contributes few intervals
    # however many chord changes the same song has.
    print('\nsongs / generated observations pooled')
    print('metric'.ljust(26) + ''.join(s.ljust(width) for s in systems))
    by_key = {(r['metric'], r['system']): r for r in recs}
    for k in keys:
        line = k.ljust(26)
        for sysname in systems:
            r = by_key[(k, sysname)]
            line += f'{r["n_songs"]} / {r["n_obs"]}'.ljust(width)
        print(line)
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['metric', 'system', 'jsd',
                                              'ci_lo', 'ci_hi', 'boot_se',
                                              'n_songs', 'n_obs', 'n_boot',
                                              'boot_levels'])
            w.writeheader()
            w.writerows(recs)
        print(f'\nwrote {len(recs)} pooled rows -> {out_csv}')


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
    row.update(repetition_metrics(ga, gb, ra, rb))
    row.update(prompt_match_metrics(gen_a.slice(0, lo), gen_b.slice(0, lo),
                                    ga, gb, ra, rb,
                                    phase=lo % FRAMES_PER_BAR))
    row.update(s_metrics(ga, gb, ra, rb, args.task))
    # count vectors for the corpus-pooled JSD. Underscored and never a
    # CSV column: the writer is extrasaction='ignore', so this rides
    # along in memory and is dropped on the way out.
    row['_pooled'] = _pooled_hists(ga, gb, ra, rb, args.task)
    return row


def _per_song(rows, key):
    """One value per SONG, averaging its samples first.

    The unit of analysis is the song, not the sample. Samples of one
    song share a prompt and a reference, so pooling 3 x N samples as
    3N independent observations understates the spread by about
    sqrt(3). EXPERIMENTS.md sec 5.3 registers this: average the samples
    within a song, then summarise over songs.
    """
    by_song = {}
    for r in rows:
        if key in r and not _is_nan(r[key]):
            by_song.setdefault(r.get('song', '?'), []).append(r[key])
    return {k: float(np.mean(v)) for k, v in by_song.items()}


def summarize(rows, task, baseline=None):
    by_system = {}
    for r in rows:
        by_system.setdefault(r.get('system', '?'), []).append(r)
    systems = sorted(by_system)
    if baseline not in by_system:
        baseline = None
    # widest metric name in any block, plus the * column and a gap
    NAMEW = max(len(k) for h in GROUP_ORDER for k in H_GROUPS[h]) + 3
    print('\n================= SUMMARY (priority order: '
          + ' > '.join(GROUP_ORDER) + ') =================')
    print('mean +- std over SONGS (samples averaged within song first); '
          'n = songs')
    if baseline:
        print(f'p: Wilcoxon signed-rank on per-song differences vs '
              f'{baseline}, paired by song')
    for h in GROUP_ORDER:
        print(f'\n--- {h} ---')
        keys = [k for k in H_GROUPS[h]
                if any(k in r and not _is_nan(r[k]) for r in rows)]
        width = 22 if baseline else 17
        header = 'metric'.ljust(NAMEW) + ''.join(s.ljust(width) for s in systems)
        print(header)
        for k in keys:
            star = '*' if k in PRIMARY[task][h] else ' '
            line = (star + k).ljust(NAMEW)
            base = _per_song(by_system[baseline], k) if baseline else {}
            for sysname in systems:
                song_vals = _per_song(by_system[sysname], k)
                if not song_vals:
                    line += '--'.ljust(width)
                    continue
                v = np.fromiter(song_vals.values(), float)
                cell = f'{v.mean():+.3f}+-{v.std(ddof=1):.3f}' if len(v) > 1 \
                    else f'{v.mean():+.3f}'
                if baseline and sysname != baseline:
                    shared = sorted(set(song_vals) & set(base))
                    if len(shared) > 1:
                        d = np.array([song_vals[c] - base[c] for c in shared])
                        if np.any(d != 0):
                            from scipy.stats import wilcoxon
                            pv = wilcoxon(d).pvalue
                            cell += f' p={pv:.3f}' if pv >= 1e-3 else ' p<.001'
                line += cell.ljust(width)
            print(line)
        if keys:
            ns = {s: len(_per_song(by_system[s], keys[0])) for s in systems}
            print('  n songs'.ljust(NAMEW)
                  + ''.join(str(ns[s]).ljust(width) for s in systems))
    print('\n(* = pre-registered primary endpoint; deltas/JSD: closer to 0 '
          'is better; ratios: closer to 1 is better)')
    print('NOTE std is over songs and JSD is bounded on [0,1] and skewed, so '
          'mean +- std can leave the range.\n     Each JSD here is also a '
          'small-sample estimate off ~100 onsets, biased up by ~0.06 and '
          'more\n     for a sparse stream; the corpus-pooled table below '
          'has the unbiased counterpart.')


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
    p.add_argument('--baseline', default=None,
                   help='system name to pair against: each other system '
                        'gets a Wilcoxon signed-rank p over PER-SONG '
                        'differences. Pairing by song removes the song '
                        'effect, which is far larger than the system '
                        'effect, so it is much more powerful than '
                        'comparing the two marginal means.')
    p.add_argument('--pooled-out', default=None,
                   help='CSV for the corpus-pooled JSD table: one row per '
                        '(metric, system) with the pooled value, its '
                        'bootstrap interval, and the counts behind it.')
    p.add_argument('--pooled-two-level', action='store_true',
                   help='bootstrap the SAMPLES within each drawn song as '
                        'well as the songs. The default one-level version '
                        'holds a song\'s decodes fixed, so its interval '
                        'covers other songs but not another decode; this '
                        'covers both and is wider. Slower: it cannot be '
                        'done as one matrix product.')
    p.add_argument('--pooled-boot', type=int, default=2000,
                   help='bootstrap replicates for the corpus-pooled JSD '
                        'table, resampling SONGS. 0 prints the pooled '
                        'point estimates without intervals.')
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
            k for h in CSV_GROUPS for k in H_GROUPS[h]]
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {len(rows)} rows -> {args.out}')

    summarize(rows, args.task, baseline=args.baseline)
    pooled_summary(rows, args.task, n_boot=args.pooled_boot,
                   out_csv=args.pooled_out, two_level=args.pooled_two_level)


if __name__ == '__main__':
    main()
