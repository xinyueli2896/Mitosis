"""Tag Nottingham midi tracks for single-stream (merged) tokenization.

Nottingham midis carry melody (first note-bearing track) and the
rendered chord accompaniment (second note-bearing track) BOTH on
program 0, so the single-stream CP tokenizer -- which distinguishes
streams only by program token -- would merge them into one
indistinguishable stream. This writes a copy of each song with the
chord track's program forced to a distinct value (default 48,
matching the POP909 tagged-combined convention and
cp_transformer_inference.py's auto-tagging), so single-stream melchord
finetuning (S1) keeps the two parts separable, just like the POP909
"-tagged" folder built for the same purpose.

Songs with fewer than 2 note-bearing instruments (no chord track at
all) are skipped -- the SAME population excluded from the duet
melody/chord pair (see preprocess_large_midi_dataset.py's
_nottingham_two_track_files), keeping the single-stream and duet
Nottingham data aligned to the same song set.

Usage:
    python tag_nottingham_melchord.py \
        --src /home/xinyue.li/nottingham-dataset/MIDI \
        --dst nottingham-melchord-tagged \
        --chord-program 48
"""

import argparse
import os
import warnings

import mido
import pretty_midi


def _force_track_program(track, program):
    out = mido.MidiTrack()
    had_pc = False
    for msg in track:
        if msg.type == 'program_change':
            out.append(msg.copy(program=program))
            had_pc = True
        else:
            out.append(msg.copy())
    if not had_pc:
        out.insert(0, mido.Message('program_change', program=program, time=0))
    return out


def has_two_note_tracks(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pm = pretty_midi.PrettyMIDI(path)
        return sum(1 for ins in pm.instruments if len(ins.notes) > 0) >= 2
    except Exception:
        return False


def tag_file(src_path, dst_path, chord_program):
    mid = mido.MidiFile(src_path)
    note_track_idx = [i for i, t in enumerate(mid.tracks)
                      if any(m.type == 'note_on' and m.velocity > 0 for m in t)]
    if len(note_track_idx) < 2:
        raise ValueError('fewer than 2 note-bearing tracks')
    chord_idx = note_track_idx[1]
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for i, t in enumerate(mid.tracks):
        out.tracks.append(_force_track_program(t, chord_program)
                          if i == chord_idx else t)
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    out.save(dst_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--chord-program', type=int, default=48)
    args = p.parse_args()

    names = sorted(f for f in os.listdir(args.src) if f.lower().endswith('.mid'))
    kept = failed = 0
    for name in names:
        src_path = os.path.join(args.src, name)
        if not has_two_note_tracks(src_path):
            continue
        try:
            tag_file(src_path, os.path.join(args.dst, name), args.chord_program)
            kept += 1
        except Exception as e:
            print(f'  {name}: {e!r}')
            failed += 1

    print(f'Done. {kept} tagged -> {args.dst}, {failed} failed, '
          f'{len(names) - kept - failed} skipped (melody-only, no chord track).')


if __name__ == '__main__':
    main()
