"""Quantify LONG-TERM STRUCTURE in generated midi continuations.

Motivation: listening says the duet models' output lacks long-term
structure relative to the single-stream baselines. This measures that
impression per system so hypotheses about the cause become testable:
which systems lack structure, in which stream, and does it decay with
bar index (rollout drift) or is it absent from the start (never
learned)?

Per file, on a 16-frames-per-bar grid (initial tempo, like
cp_capacity_check): a 12-dim chroma vector per bar (non-drum onsets)
and a 16-dim within-bar onset-position histogram per bar (all onsets,
drums included). From those, over the GENERATED region (bar index >=
--prompt-bars; outputs include their prompt):

  repetition   mean over bars of the best cosine match to any earlier
               non-adjacent bar. High = material recurs (motifs,
               sections); low = through-composed wandering.
  period@4/8   mean cosine similarity between bars 4 (resp. 8) apart:
               phrase- and section-level periodicity. Reported for
               chroma (harmony) and rhythm (groove) separately.
  drift_slope  linear trend, per 16 bars, of each generated bar's
               chroma similarity to the PROMPT's mean chroma.
               ~0 = tonally anchored; strongly negative = the piece
               walks away from its own key/material (rollout drift).
  drift_mean   mean of that same similarity (overall anchoredness).

Empty bars pair to similarity 0 with everything; a stream that dies
therefore drags repetition/periodicity down -- that is deliberate
(a dead stream has no structure), but check survival_b from the E1
tables to attribute cause.

Reference calibration: pass the ground-truth continuations as one more
system (e.g. REF:<folder of full songs>) and read every model's numbers
against REF's, not against 1.0 -- real music also modulates and varies.

Usage:
    python structure_metrics.py \
        --system A2:temp/e1_melchord_<id>/a2 \
        --system S1:temp/e1_melchord_<id>/s1 \
        --system REF:input/pop909_split_merged \
        --prompt-bars 4 [--split-streams] [--max-bars 24]

Folders are scanned recursively (duet_multi layouts nest song/mode/).
--split-streams additionally reports melody-only and chord-only rows
per system, attributing notes by program (24/0 = melody, 48 = chord,
drums excluded) -- melchord outputs only.
"""

import argparse
import os
import warnings
from collections import defaultdict
from glob import glob

import numpy as np
import pretty_midi

MEL_PROGRAMS = {0, 24}
CHORD_PROGRAMS = {48}


def bar_features(pm, max_bars=None, keep=None):
    """-> (chroma [n_bars, 12], rhythm [n_bars, 16]) on a 16th grid.
    keep: optional predicate(instrument) selecting which notes count
    toward chroma; rhythm always counts every kept instrument's onsets."""
    _, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    if bpm <= 0:
        bpm = 120.0
    step = 60.0 / bpm / 4.0
    end_frame = 0
    notes = []
    for inst in pm.instruments:
        if keep is not None and not keep(inst):
            continue
        for n in inst.notes:
            f = int(round(n.start / step))
            notes.append((f, n.pitch, inst.is_drum))
            end_frame = max(end_frame, f)
    n_bars = end_frame // 16 + 1
    if max_bars is not None:
        n_bars = min(n_bars, max_bars)
    chroma = np.zeros((n_bars, 12))
    rhythm = np.zeros((n_bars, 16))
    for f, pitch, is_drum in notes:
        bar = f // 16
        if bar >= n_bars:
            continue
        rhythm[bar, f % 16] += 1.0
        if not is_drum:
            chroma[bar, pitch % 12] += 1.0
    return chroma, rhythm


def _cos_rows(a, b):
    """Row-wise cosine; rows where either side is all-zero -> 0."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    dots = (a * b).sum(axis=1)
    out = np.zeros(len(a))
    ok = denom > 0
    out[ok] = dots[ok] / denom[ok]
    return out


def file_metrics(chroma, rhythm, prompt_bars):
    gen_c = chroma[prompt_bars:]
    gen_r = rhythm[prompt_bars:]
    n = len(gen_c)
    if n < 10:
        return None
    m = {}
    for name, mat in (('chroma', gen_c), ('rhythm', gen_r)):
        for lag in (4, 8):
            if n > lag:
                m[f'period{lag}_{name}'] = float(
                    _cos_rows(mat[:-lag], mat[lag:]).mean())
    # repetition: best match to any earlier bar at lag >= 2
    best = []
    for i in range(2, n):
        prev = gen_c[:i - 1]
        cur = np.repeat(gen_c[i:i + 1], len(prev), axis=0)
        best.append(_cos_rows(prev, cur).max())
    m['repetition'] = float(np.mean(best)) if best else 0.0
    # tonal drift vs the prompt's mean chroma
    anchor = chroma[:prompt_bars].sum(axis=0, keepdims=True)
    sims = _cos_rows(gen_c, np.repeat(anchor, n, axis=0))
    m['drift_mean'] = float(sims.mean())
    x = np.arange(n)
    m['drift_slope'] = float(np.polyfit(x, sims, 1)[0] * 16)
    m['empty_bar_rate'] = float((gen_c.sum(axis=1) == 0).mean())
    return m


STREAMS = {
    'both': None,
    'mel': lambda inst: (not inst.is_drum) and inst.program in MEL_PROGRAMS,
    'chord': lambda inst: (not inst.is_drum) and inst.program in CHORD_PROGRAMS,
}

COLS = ['repetition', 'period4_chroma', 'period8_chroma',
        'period4_rhythm', 'period8_rhythm', 'drift_mean', 'drift_slope',
        'empty_bar_rate']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--system', action='append', required=True,
                   metavar='NAME:FOLDER',
                   help='repeatable; folder scanned recursively for midis')
    p.add_argument('--prompt-bars', type=int, default=4)
    p.add_argument('--max-bars', type=int, default=None,
                   help='truncate long reference files to this many bars')
    p.add_argument('--split-streams', action='store_true',
                   help='also report melody-only / chord-only rows '
                        '(melchord program conventions: 0/24 mel, 48 chord)')
    args = p.parse_args()

    streams = list(STREAMS) if args.split_streams else ['both']
    rows = {}   # (system, stream) -> list of per-file metric dicts
    for spec in args.system:
        name, folder = spec.split(':', 1)
        files = sorted(glob(os.path.join(folder, '**', '*.mid'),
                            recursive=True)
                       + glob(os.path.join(folder, '**', '*.MID'),
                              recursive=True))
        if not files:
            print(f'[{name}] NO MIDIS under {folder}')
            continue
        for f in files:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pm = pretty_midi.PrettyMIDI(f)
            except Exception as e:
                print(f'[{name}] unreadable {f}: {e!r}')
                continue
            for stream in streams:
                chroma, rhythm = bar_features(pm, args.max_bars,
                                              STREAMS[stream])
                m = file_metrics(chroma, rhythm, args.prompt_bars)
                if m is not None:
                    rows.setdefault((name, stream), []).append(m)
        n_used = len(rows.get((name, 'both'), []))
        print(f'[{name}] {len(files)} midis, {n_used} usable')

    for stream in streams:
        print(f'\n=== stream: {stream} '
              f'(generated bars only, prompt={args.prompt_bars} bars) ===')
        header = f'{"system":10s}' + ''.join(f'{c:>16s}' for c in COLS) \
                 + f'{"n":>6s}'
        print(header)
        for (name, s), ms in rows.items():
            if s != stream:
                continue
            vals = {c: np.mean([m[c] for m in ms if c in m]) for c in COLS}
            print(f'{name:10s}'
                  + ''.join(f'{vals[c]:16.3f}' for c in COLS)
                  + f'{len(ms):6d}')

    print(
        '\nHow to read this (compare each model to the REF row, not to 1):\n'
        '  * low repetition/period4/period8 at SIMILAR drift_mean\n'
        '      -> structure never present (never learned / geometry),\n'
        '         not a rollout problem.\n'
        '  * drift_slope well below REF -> structure decays with bar\n'
        '      index: rollout/exposure-bias drift.\n'
        '  * chord-only row much worse than mel-only row (with high\n'
        '      empty_bar_rate) -> follower collapse is the carrier of\n'
        '      the deficit, not global structure per se.\n'
        '  * rhythm periods fine but chroma periods low -> groove is\n'
        '      stable, harmony wanders (tonal long-range failure).'
    )


if __name__ == '__main__':
    main()
