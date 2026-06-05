"""Split a folder of MIDIs into drum-only and nondrum-only versions.

For each .mid in --input_dir we write:
    <output_dir>/drum/<same_relpath>     containing ONLY drum tracks (is_drum=True)
    <output_dir>/nondrum/<same_relpath>  containing ONLY pitched tracks

Filenames and subdirectory structure are preserved. Files with no notes on a
side are skipped on that side (a drum-only output isn't written if the input
has no drum tracks, and vice versa).

Run:
    python split_drum_nondrum.py \\
        --input_dir input/lamd_test_prompts \\
        --output_dir input/lamd_test_prompts_split
"""

import argparse
import os

import pretty_midi


def _split_one(in_path, drum_path, nondrum_path):
    midi = pretty_midi.PrettyMIDI(in_path)
    drum_ins = [i for i in midi.instruments if i.is_drum]
    nondrum_ins = [i for i in midi.instruments if not i.is_drum]
    wrote = []

    def _write(side_path, instruments):
        out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        out.instruments.extend(instruments)
        os.makedirs(os.path.dirname(side_path), exist_ok=True)
        out.write(side_path)

    if drum_ins:
        _write(drum_path, drum_ins)
        wrote.append('drum')
    if nondrum_ins:
        _write(nondrum_path, nondrum_ins)
        wrote.append('nondrum')
    return wrote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True,
                    help='Folder of .mid files (recursive). Subdirectory '
                         'structure is preserved in --output_dir.')
    ap.add_argument('--output_dir', required=True,
                    help='Two subdirs are created here: drum/ and nondrum/.')
    args = ap.parse_args()

    midi_paths = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                midi_paths.append(os.path.join(root, f))
    midi_paths.sort()
    print(f'[scan] {len(midi_paths)} MIDIs in {args.input_dir}')

    drum_root = os.path.join(args.output_dir, 'drum')
    nondrum_root = os.path.join(args.output_dir, 'nondrum')

    n_drum = n_nondrum = n_both = n_fail = 0
    for p in midi_paths:
        rel = os.path.relpath(p, args.input_dir)
        try:
            wrote = _split_one(p,
                               os.path.join(drum_root, rel),
                               os.path.join(nondrum_root, rel))
        except Exception as e:
            n_fail += 1
            print(f'[fail] {rel}: {e}')
            continue
        if 'drum' in wrote:
            n_drum += 1
        if 'nondrum' in wrote:
            n_nondrum += 1
        if len(wrote) == 2:
            n_both += 1

    print(f'[done] drum-only outputs:    {n_drum}')
    print(f'[done] nondrum-only outputs: {n_nondrum}')
    print(f'[done] songs with both:      {n_both}')
    if n_fail:
        print(f'[done] failed to read:       {n_fail}')


if __name__ == '__main__':
    main()
