"""Tokenizer's-eye-view grid check: are note onsets quantized in TICK
space, the only domain the tokenizer reads?

Motivation: check_beat_alignment.py measures onsets in SECONDS through
the file's full tempo map against a fixed step from the FIRST tempo --
correct for constant-tempo corpora (Nottingham), but a beat-synced
transcription (Pop1K7: madmom-aligned, note ticks on the grid, a tempo
event per beat carrying the performance timing) looks massively
"off-grid" there while being perfectly clean where it matters. The
tokenizer (preprocess_large_midi_dataset.preprocess_midi) loads files
as XFMidi(constant_tempo=60/beat_div), which DISCARDS the tempo map and
maps raw ticks to subbeats -- so the only question for tokenization
quality is whether onsets sit on the subbeat tick grid.

Per file this reports, in subbeat units (beat_div per beat, default 4 =
16ths): off-grid onset fraction, whether deviations cluster at thirds
(genuine triplet subdivisions), the number of tempo events (many =
beat-synced layout, exactly the case the seconds-domain check
misjudges), and ticks/beat. Verdict per folder.

Reading the two checks together:
  seconds-check FLAGGED + tick-check CLEAN
      -> beat-synced layout; tokenization is SAFE; the seconds check
         was the wrong instrument. (Also means decoded outputs are on
         a straight metronomic grid, not the performance timing.)
  both CLEAN        -> quantized constant-tempo file, safe.
  tick-check DIRTY  -> genuinely unquantized for the tokenizer; these
         onsets get snapped and the tokenized stream jitters. Exclude,
         or quantize upstream, or use the corpus's quantized release.

Usage (via check_tokenizer_grid.sbatch):
    python check_tokenizer_grid.py --folder input/pop1k7_test_split/melody \\
        [--folder ...] [--beat-div 4]
"""

import argparse
import glob
import os
import warnings

import mido

from xf_midi import XFMidi


def analyze(path, beat_div):
    n_tempo = 0
    m = mido.MidiFile(path)
    for track in m.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                n_tempo += 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        midi = XFMidi(path, constant_tempo=60.0 / beat_div)
    devs = []
    for ins in midi.instruments:
        for note in ins.notes:
            frac = note.start % 1.0          # start is in SUBBEAT units
            devs.append(min(frac, 1.0 - frac))
    n = len(devs)
    off = [d for d in devs if d > 0.10]
    trip = [d for d in off if abs(d - 1.0 / 3.0) < 0.06]
    return {
        'file': os.path.basename(path),
        'n': n,
        'tpb': midi.resolution,
        'tempo_events': n_tempo,
        'off_frac': len(off) / max(n, 1),
        'trip_frac': len(trip) / max(n, 1),
        'max_dev': max(devs) if devs else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', action='append', required=True)
    p.add_argument('--beat-div', type=int, default=4,
                   help='subbeats per beat the tokenizer uses (default 4 '
                        '= 16th notes)')
    args = p.parse_args()

    print('=' * 76)
    print('TOKENIZER-VIEW GRID CHECK  (tick space, tempo map discarded, '
          f'beat_div={args.beat_div})')
    print('=' * 76)

    any_dirty = False
    for folder in args.folder:
        files = sorted(glob.glob(os.path.join(folder, '*.mid')) +
                       glob.glob(os.path.join(folder, '*.MID')))
        print(f'\n--- {folder}  ({len(files)} files) ---')
        hdr = (f'{"file":<26} {"notes":>6} {"tpb":>5} {"tempo-ev":>8} '
               f'{"off-grid":>8} {"triplet":>8} {"max dev":>8}')
        print(hdr)
        for f in files:
            r = analyze(f, args.beat_div)
            dirty = r['off_frac'] > 0.05
            any_dirty = any_dirty or dirty
            mark = '  <-- DIRTY' if dirty else ''
            print(f'{r["file"]:<26} {r["n"]:>6} {r["tpb"]:>5} '
                  f'{r["tempo_events"]:>8} {r["off_frac"]:>7.1%} '
                  f'{r["trip_frac"]:>7.1%} {r["max_dev"]:>8.3f}{mark}')

    print('\n' + '=' * 76)
    if any_dirty:
        print('DIRTY: some files are off-grid IN TICK SPACE -- the '
              'tokenizer will snap')
        print('these onsets and the tokenized stream jitters. Exclude '
              'them, quantize')
        print('upstream, or switch to the corpus\'s quantized release.')
    else:
        print('CLEAN: all onsets sit on the subbeat tick grid the '
              'tokenizer reads.')
        print('If the seconds-domain check (check_beat_alignment) '
              'flagged these same')
        print('files, that is the beat-synced-transcription layout: '
              'tick grid clean,')
        print('tempo map carrying performance timing. Tokenization is '
              'SAFE; note the')
        print('tempo map (and thus the performance feel) is discarded '
              'by design --')
        print('decoded outputs are on a straight metronomic grid.')
    print('=' * 76)


if __name__ == '__main__':
    main()
