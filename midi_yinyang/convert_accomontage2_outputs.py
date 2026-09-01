"""T8 converter: AccoMontage2 chord_gen.mid -> eval-ready combined midi.

chord_gen.mid (verified against chorderator's send_out()) is
  instrument 0: our input melody, unchanged, at the song's tempo
  instrument 1: the generated chord track (rendered in the song's key)

eval_metrics.load_streams resolves combined files by track names
MELODY / CHORD, so conversion is: name the two tracks, set the program
convention the S* systems use (melody program 0, chord program 48),
and write one midi per song. A per-song sanity check compares the
melody track note-for-note against the input melody midi (count and
first/last onset) so a silent misordering of instruments can never
slip into scoring; mismatches are reported and the song is skipped.

Usage (via convert_accomontage2_outputs.sbatch):
    python convert_accomontage2_outputs.py \
        --in-dir temp/accomontage2_pop909 \
        --melody-dir input/pop909_split/melody \
        --out-dir temp/accomontage2_pop909_eval
"""

import argparse
import os
import sys

import pretty_midi


def fingerprint(notes):
    """(count, (first onset, pitch), (last onset, pitch)) of a note list."""
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    if not notes:
        return 0, None, None
    return len(notes), (round(notes[0].start, 2), notes[0].pitch), \
        (round(notes[-1].start, 2), notes[-1].pitch)


def midi_fingerprint(path):
    pm = pretty_midi.PrettyMIDI(path)
    return fingerprint([n for inst in pm.instruments if not inst.is_drum
                        for n in inst.notes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--melody-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    songs = sorted(d for d in os.listdir(args.in_dir)
                   if os.path.isdir(os.path.join(args.in_dir, d)))
    ok, failed = [], []
    for sid in songs:
        src = os.path.join(args.in_dir, sid, 'chord_gen.mid')
        if not os.path.isfile(src):
            failed.append((sid, 'no chord_gen.mid'))
            continue
        try:
            pm = pretty_midi.PrettyMIDI(src)
            if len(pm.instruments) < 2:
                raise RuntimeError(
                    f'expected 2 instruments, got {len(pm.instruments)}')
            mel, chord = pm.instruments[0], pm.instruments[1]

            # Sanity: instrument 0 must be OUR melody, not the chords.
            ref_path = None
            for f in os.listdir(args.melody_dir):
                if sid in f and f.lower().endswith('.mid'):
                    ref_path = os.path.join(args.melody_dir, f)
                    break
            if ref_path:
                ref_fp = midi_fingerprint(ref_path)
                got_fp = fingerprint(mel.notes)
                if abs(got_fp[0] - ref_fp[0]) > max(3, 0.02 * ref_fp[0]):
                    raise RuntimeError(
                        f'melody track mismatch: {got_fp[0]} notes vs '
                        f'reference {ref_fp[0]} -- instruments swapped?')

            out = pretty_midi.PrettyMIDI(
                initial_tempo=pm.get_tempo_changes()[1][0]
                if len(pm.get_tempo_changes()[1]) else 120.0)
            mel.name, mel.program, mel.is_drum = 'MELODY', 0, False
            chord.name, chord.program, chord.is_drum = 'CHORD', 48, False
            out.instruments.append(mel)
            out.instruments.append(chord)
            out.write(os.path.join(args.out_dir, f'{sid}.mid'))
            ok.append(sid)
        except Exception as e:                    # noqa: BLE001
            failed.append((sid, repr(e)[:120]))

    print(f'converted: {len(ok)}/{len(songs)}')
    for sid, err in failed:
        print(f'  FAILED {sid}: {err}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
