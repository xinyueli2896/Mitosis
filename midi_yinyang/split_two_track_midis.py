"""Split two-track melchord midis (Nottingham convention: instrument 0 =
melody, instrument 1 = chord accompaniment) into per-stream folders, so
held-out two-track files can serve as eval_e3 / eval_e1 prompt+reference
folders (MEL_FOLDER / CHORD_FOLDER), which expect one stream per file.

Files with fewer than two note-carrying instruments (e.g. the ~1% of
melody-only 'morris' tunes) are skipped with a warning -- the paired
eval requires both streams.

Timing is preserved via pretty_midi round-trip with the source file's
initial tempo. Output filenames mirror the inputs.

Usage:
    python split_two_track_midis.py \
        --src ~/nottingham-heldout-yy \
        --dst-mel input/nottingham_heldout_split/melody \
        --dst-chord input/nottingham_heldout_split/chord
"""

import argparse
import os
import warnings
from glob import glob

import pretty_midi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst-mel', required=True)
    p.add_argument('--dst-chord', required=True)
    args = p.parse_args()

    files = sorted(glob(os.path.join(args.src, '*.mid'))
                   + glob(os.path.join(args.src, '*.MID')))
    if not files:
        raise SystemExit(f'no midis in {args.src}')
    os.makedirs(args.dst_mel, exist_ok=True)
    os.makedirs(args.dst_chord, exist_ok=True)

    n_ok = 0
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pm = pretty_midi.PrettyMIDI(f)
        insts = [i for i in pm.instruments if i.notes]
        if len(insts) < 2:
            print(f'[skip] {os.path.basename(f)}: {len(insts)} '
                  f'note-carrying instrument(s), need 2')
            continue
        if len(insts) > 2:
            print(f'[warn] {os.path.basename(f)}: {len(insts)} instruments; '
                  f'using the first as melody, the SECOND as chord, '
                  f'ignoring the rest')
        _, tempi = pm.get_tempo_changes()
        bpm = float(tempi[0]) if len(tempi) else 120.0
        base = os.path.basename(f)
        if base.lower().endswith('.mid'):
            base = base[:-4] + '.mid'
        for inst, dst in ((insts[0], args.dst_mel), (insts[1], args.dst_chord)):
            out = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            out.instruments.append(inst)
            out.write(os.path.join(dst, base))
        n_ok += 1
    print(f'split {n_ok}/{len(files)} files -> {args.dst_mel} , {args.dst_chord}')


if __name__ == '__main__':
    main()
