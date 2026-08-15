"""Diagnose a cascade (P-mc / P-cm) output: is the PIPELINE wrong, or is
the music just weak?

The decisive check is the PROMPT region. Bars before --prompt-frames are
copied real material in both streams, so if they do not match the source
song the plumbing is broken (misalignment, wrong track order, wrong
tempo, dropped stream) -- no listening judgement required. If the prompt
matches and only the generated region is unconvincing, the pipeline is
sound and what you hear is model quality.

Reports per file:
  tempo        output vs source (both streams render on the output's own
               grid, so a mismatch is a playback-speed issue only --
               eval_metrics frames by each file's tempo)
  streams      track names found, and note counts per stream
  prompt       per-stream agreement with the source over the prompt bars
               (fraction of source onset-frames whose pitch set matches)
  generated    notes, empty-bar rate, and chord-tone fit -- the share of
               generated melody onsets whose pitch class is in the chord
               sounding at that frame, measured the same way on the
               GROUND TRUTH so the number is calibrated, not absolute.

Usage:
    python check_cascade_output.py \
        --out temp/cascade_P-mc_<jobid>/4_final \
        --mel-folder input/pop909_split/melody \
        --chord-folder input/pop909_split/chord \
        --prompt-frames 96
"""

import argparse
import os
import warnings
from collections import defaultdict
from glob import glob

import pretty_midi

FRAMES_PER_BAR = 16


def _tempo(pm, default=120.0):
    _, tempi = pm.get_tempo_changes()
    t = float(tempi[0]) if len(tempi) else default
    return t if t > 0 else default


def _frames(pm, keep=None):
    """-> {frame: set(pitch)} on the file's own 16th grid."""
    step = 60.0 / _tempo(pm) / 4.0
    out = defaultdict(set)
    for inst in pm.instruments:
        if inst.is_drum or (keep is not None and not keep(inst)):
            continue
        for n in inst.notes:
            out[int(round(n.start / step))].add(n.pitch)
    return out


def _sounding(pm, keep=None):
    """-> {frame: set(pitch class)} of notes HELD at each frame."""
    step = 60.0 / _tempo(pm) / 4.0
    out = defaultdict(set)
    for inst in pm.instruments:
        if inst.is_drum or (keep is not None and not keep(inst)):
            continue
        for n in inst.notes:
            f0 = int(round(n.start / step))
            f1 = max(f0 + 1, int(round(n.end / step)))
            for f in range(f0, f1):
                out[f].add(n.pitch % 12)
    return out


def _by_name(name):
    return lambda inst: (inst.name or '').strip().upper() == name


def agreement(a, b, upto):
    """Fraction of b's frames below `upto` whose pitch set a reproduces."""
    ref = {f: p for f, p in b.items() if f < upto}
    if not ref:
        return float('nan'), 0
    hit = sum(1 for f, pitches in ref.items() if a.get(f) == pitches)
    return hit / len(ref), len(ref)


def chord_fit(mel, chord_sounding, lo, hi):
    """Share of melody onsets in [lo, hi) whose pitch class is in the
    chord sounding at that frame."""
    tot = fit = 0
    for f, pitches in mel.items():
        if not (lo <= f < hi):
            continue
        pcs = chord_sounding.get(f, set())
        for p in pitches:
            tot += 1
            if p % 12 in pcs:
                fit += 1
    return (fit / tot if tot else float('nan')), tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True,
                   help='cascade 4_final dir (or a single .mid)')
    p.add_argument('--mel-folder', required=True)
    p.add_argument('--chord-folder', required=True)
    p.add_argument('--prompt-frames', type=int, default=96)
    p.add_argument('--total-frames', type=int, default=416)
    args = p.parse_args()

    files = ([args.out] if args.out.endswith('.mid')
             else sorted(glob(os.path.join(args.out, '**', '*.mid'),
                              recursive=True)))
    if not files:
        raise SystemExit(f'no midis under {args.out}')

    print(f'{"file":38s}{"tempo out/src":>16s}{"mel prompt":>12s}'
          f'{"chd prompt":>12s}{"gen notes m/c":>15s}{"chordfit gen/ref":>18s}')
    print('-' * 111)
    for f in files:
        # song id = the dir two levels up (<song>/co/sample_i.mid)
        parts = f.split(os.sep)
        song = parts[-3] if len(parts) >= 3 else os.path.splitext(
            os.path.basename(f))[0]
        src_mel = src_chd = None
        for ext in ('.mid', '.MID'):
            if os.path.exists(os.path.join(args.mel_folder, song + ext)):
                src_mel = os.path.join(args.mel_folder, song + ext)
                src_chd = os.path.join(args.chord_folder, song + ext)
        if src_mel is None or not os.path.exists(src_chd):
            print(f'{song:38s}  [no source pair in the given folders]')
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out_pm = pretty_midi.PrettyMIDI(f)
            mel_pm = pretty_midi.PrettyMIDI(src_mel)
            chd_pm = pretty_midi.PrettyMIDI(src_chd)

        out_mel = _frames(out_pm, _by_name('MELODY'))
        out_chd = _frames(out_pm, _by_name('CHORD'))
        if not out_mel and not out_chd:
            names = sorted({(i.name or '<unnamed>') for i in out_pm.instruments})
            print(f'{song:38s}  [no MELODY/CHORD tracks; found {names}]')
            continue

        m_ok, m_n = agreement(out_mel, _frames(mel_pm), args.prompt_frames)
        c_ok, c_n = agreement(out_chd, _frames(chd_pm), args.prompt_frames)

        gen_m = sum(len(v) for k, v in out_mel.items()
                    if k >= args.prompt_frames)
        gen_c = sum(len(v) for k, v in out_chd.items()
                    if k >= args.prompt_frames)

        fit_gen, _ = chord_fit(out_mel, _sounding(out_pm, _by_name('CHORD')),
                               args.prompt_frames, args.total_frames)
        fit_ref, _ = chord_fit(_frames(mel_pm), _sounding(chd_pm),
                               args.prompt_frames, args.total_frames)

        label = f'{song}/{os.path.basename(f)[:14]}'
        print(f'{label:38s}{_tempo(out_pm):7.1f}/{_tempo(mel_pm):<8.1f}'
              f'{m_ok:11.2f}{c_ok:12.2f}'
              f'{gen_m:8d}/{gen_c:<6d}{fit_gen:9.2f}/{fit_ref:<8.2f}')

    print('\nHow to read:')
    print('  * prompt columns near 1.00 -> the plumbing is correct: the')
    print('    copied real material survives the round trip intact. Low')
    print('    values mean misalignment / wrong track order / a dropped')
    print('    stream -- a PIPELINE bug, not model quality.')
    print('  * tempo out != src -> playback speed only (scoring frames')
    print('    each file by its own tempo), but it is why output can')
    print('    sound rushed or dragging next to the original.')
    print('  * gen notes: 0 in a stream means that stage produced nothing.')
    print('  * chordfit gen vs ref: how often the melody note belongs to')
    print('    the sounding chord, generated vs ground truth. Well below')
    print('    the reference = the two streams do not fit each other,')
    print('    which is the cascade\'s expected weakness (no mutual')
    print('    conditioning) -- and is a RESULT, not a bug.')


if __name__ == '__main__':
    main()
