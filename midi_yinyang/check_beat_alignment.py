"""Is a MIDI folder representable on the CP 16th-note grid?

SCOPE CAVEAT (learned on Pop1K7, jobs 185995/186001): this check
measures onsets in SECONDS through the file's full tempo map against a
fixed step derived from the FIRST tempo event. That is the right
instrument for constant-tempo corpora (Nottingham), and the WRONG one
for beat-synced transcriptions (Pop1K7), whose note ticks sit cleanly
on the grid while ~one tempo event per beat carries the performance
timing -- such files flag TRIPLET/OFF-GRID here at 70-90% while
tokenizing perfectly. The tokenizer reads TICK space only
(XFMidi constant_tempo discards the tempo map), so when this check
flags a corpus, confirm with check_tokenizer_grid.py before excluding
anything; a high tempo-event count is the tell.

The tokenizer quantizes every onset to step = 60/tempo/4 (a 16th note).
Anything not on that grid is SNAPPED at tokenization -- the source file
sounds fine, but the tokenized version (training data, prompts, and
anything decoded back) lurches. Three failure modes, all present in the
Nottingham conversions:

  TRIPLET     onsets at 1/3 or 2/3 of a step (compound meter -- jigs in
              6/8, slip jigs in 9/8 -- encoded as triplet subdivisions).
              Deviation clusters at ~0.33: every offbeat note is snapped
              a 32nd note away. Sounds exactly like "not beat aligned".
  OFF-GRID    onsets at other fractions (swing, humanized timing, tempo
              mismatch between the tempo map and the note ticks).
  METER       time signature whose bar is not 16 steps (3/4 = 12,
              6/8 = 12, 9/8 = 18, 2/4 = 8, mixed signatures mid-tune).
              Notes may still align to the 16th grid, but every
              16-frames-per-bar assumption downstream (structure
              metrics, "N bars" prompt lengths) mis-slices these songs.

Per file: time signature(s), tempo events (Nottingham files typically
have NONE -- the 120 BPM default applies everywhere), off-grid onset
fraction, whether deviations cluster at thirds (triplet encoding), and
bar length in steps. Folder summary lists the offenders.

Usage (via check_beat_alignment.sbatch):
    python check_beat_alignment.py --folder input/not_split/melody \\
        [--folder input/not_split/chord] [--file jigs108.mid]
"""

import argparse
import glob
import os
import warnings

import mido
import pretty_midi


def analyze(path):
    m = mido.MidiFile(path)
    tempos, sigs = [], []
    t = 0
    for msg in mido.merge_tracks(m.tracks):
        t += msg.time
        if msg.type == 'set_tempo':
            tempos.append((t, round(mido.tempo2bpm(msg.tempo), 2)))
        elif msg.type == 'time_signature':
            sigs.append((t, msg.numerator, msg.denominator))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pm = pretty_midi.PrettyMIDI(path)
    bpm = tempos[0][1] if tempos else 120.0
    step = 60.0 / bpm / 4.0

    devs = []
    for inst in pm.instruments:
        for n in inst.notes:
            frac = (n.start / step) % 1.0
            devs.append(min(frac, 1.0 - frac))
    n = len(devs)
    off = [d for d in devs if d > 0.10]
    # triplet signature: deviations clustering at 1/3 of a step
    trip = [d for d in off if abs(d - 1.0 / 3.0) < 0.06]

    # bar length in 16th steps per signature: numerator * (16 / denominator)
    bars = sorted({int(num * 16 / den) for _, num, den in sigs}) or [16]
    uniq_sigs = sorted({f'{num}/{den}' for _, num, den in sigs}) or ['none']

    flags = []
    if n and len(trip) / n > 0.05:
        flags.append('TRIPLET')
    elif n and len(off) / n > 0.05:
        flags.append('OFF-GRID')
    if any(b != 16 for b in bars):
        flags.append(f'METER({",".join(uniq_sigs)}->{bars}steps/bar)')
    if len(uniq_sigs) > 1:
        flags.append('MIXED-METER')
    if not tempos:
        flags.append('no-tempo(120 default)')

    return {'file': os.path.basename(path), 'n': n,
            'sigs': uniq_sigs, 'bars': bars, 'tempos': len(tempos),
            'off_frac': len(off) / max(n, 1),
            'trip_frac': len(trip) / max(n, 1),
            'max_dev': max(devs) if devs else 0.0,
            'flags': flags}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', action='append', required=True)
    p.add_argument('--file', default=None,
                   help='basename to report in full detail (e.g. jigs108.mid)')
    args = p.parse_args()

    print('=' * 76)
    print('BEAT-ALIGNMENT / GRID-REPRESENTABILITY CHECK  (step = 16th note)')
    print('=' * 76)
    problems = []
    for folder in args.folder:
        files = sorted(glob.glob(os.path.join(folder, '*.mid'))
                       + glob.glob(os.path.join(folder, '*.MID')))
        print(f'\n--- {folder}  ({len(files)} files) ---')
        print(f'{"file":<28}{"sig":<12}{"steps/bar":<10}{"off-grid":<9}'
              f'{"triplet":<9}{"max dev":<8}flags')
        for f in files:
            try:
                r = analyze(f)
            except Exception as e:
                print(f'{os.path.basename(f):<28}READ FAILED: {e!r}')
                problems.append((os.path.basename(f), ['UNREADABLE']))
                continue
            verbose = args.file and r['file'] == args.file
            if r['flags'] and r['flags'] != ['no-tempo(120 default)'] or verbose:
                print(f'{r["file"]:<28}{",".join(r["sigs"]):<12}'
                      f'{",".join(map(str, r["bars"])):<10}'
                      f'{r["off_frac"]:<9.1%}{r["trip_frac"]:<9.1%}'
                      f'{r["max_dev"]:<8.3f}{" ".join(r["flags"])}')
            if [x for x in r['flags'] if not x.startswith('no-tempo')]:
                problems.append((r['file'], r['flags']))
        clean = len(files) - len({p for p, _ in problems})
    print('\n' + '=' * 76)
    print('SUMMARY')
    if not problems:
        print('  every file sits on the 16th grid with 16-step bars.')
    else:
        trip = [f for f, fl in problems if any('TRIPLET' in x for x in fl)]
        meter = [f for f, fl in problems if any(x.startswith('METER') for x in fl)]
        print(f'  {len(set(f for f, _ in problems))} flagged file(s).')
        if trip:
            print(f'  TRIPLET-ENCODED ({len(trip)}): {trip[:8]}'
                  f'{"..." if len(trip) > 8 else ""}')
            print('    Compound-meter tunes whose offbeat eighths sit a third')
            print('    of a step off the grid. Tokenization SNAPS each one a')
            print('    32nd away -- the tokenized song lurches even though the')
            print('    source file sounds fine. These songs are corrupted as')
            print('    training data, as prompts, and as references. Options:')
            print('    exclude them, or move to a grid divisible by 3')
            print('    (beat_div=6/12 -- a retokenization of everything).')
        if meter:
            print(f'  NON-16-STEP BARS ({len(meter)}): {meter[:8]}'
                  f'{"..." if len(meter) > 8 else ""}')
            print('    Notes may align to the grid, but every 16-frames-per-')
            print('    bar assumption (structure metrics, bar-counted prompt')
            print('    lengths) mis-slices these songs.')
    print('=' * 76)


if __name__ == '__main__':
    main()
