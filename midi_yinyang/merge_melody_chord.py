"""Merge paired melody/chord midi folders into combined midis.

The inverse of split_melody_chord.py: for each filename present in BOTH
folders, write <dst>/<id>.mid containing

    track 0   tempo/meta track (taken from the melody file; both streams
              were built on the same grid so the tempo tracks match)
    MELODY    every note-bearing track from the melody file
    CHORD     every note-bearing track from the chord file

Track names are forced to MELODY / CHORD so the combined files work with
split_melody_chord.py and cp_transformer_inference.py's track-name
separation.

Example:
    python merge_melody_chord.py \
        --melody POP909-Dataset/POP909-melody \
        --chord  POP909-Dataset/POP909-chord \
        --dst    POP909-Dataset/POP909-melody-chord
"""

import argparse
import os
from glob import glob

import mido


def _rename_track(track, name):
    out = mido.MidiTrack()
    out.append(mido.MetaMessage("track_name", name=name, time=0))
    for msg in track:
        if msg.type == "track_name":
            continue
        out.append(msg.copy())
    return out


def _note_tracks(mid):
    return [t for t in mid.tracks
            if any(m.type == "note_on" and m.velocity > 0 for m in t)]


def merge_pair(mel_path, chord_path, out_path):
    mel = mido.MidiFile(mel_path)
    cho = mido.MidiFile(chord_path)
    if mel.ticks_per_beat != cho.ticks_per_beat:
        raise ValueError(
            f"ppq mismatch: melody {mel.ticks_per_beat} vs "
            f"chord {cho.ticks_per_beat} -- the two files are not on the "
            "same grid"
        )
    mel_notes = _note_tracks(mel)
    cho_notes = _note_tracks(cho)
    if not mel_notes:
        raise ValueError(f"no note tracks in {mel_path}")
    if not cho_notes:
        raise ValueError(f"no note tracks in {chord_path}")

    out = mido.MidiFile(ticks_per_beat=mel.ticks_per_beat)
    out.tracks.append(mel.tracks[0])   # tempo/meta grid
    for t in mel_notes:
        out.tracks.append(_rename_track(t, "MELODY"))
    for t in cho_notes:
        out.tracks.append(_rename_track(t, "CHORD"))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--melody", required=True,
                        help="folder of melody midis")
    parser.add_argument("--chord", required=True,
                        help="folder of chord midis (same filenames)")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()

    mel_files = {os.path.basename(p): p
                 for p in glob(os.path.join(args.melody, "*.mid"))
                 + glob(os.path.join(args.melody, "*.MID"))}
    cho_files = {os.path.basename(p): p
                 for p in glob(os.path.join(args.chord, "*.mid"))
                 + glob(os.path.join(args.chord, "*.MID"))}

    names = sorted(set(mel_files) & set(cho_files))
    only_mel = sorted(set(mel_files) - set(cho_files))
    only_cho = sorted(set(cho_files) - set(mel_files))
    if only_mel:
        print(f"WARNING: {len(only_mel)} melody-only files skipped "
              f"(e.g. {only_mel[:5]})")
    if only_cho:
        print(f"WARNING: {len(only_cho)} chord-only files skipped "
              f"(e.g. {only_cho[:5]})")
    if args.ids:
        wanted = set(args.ids)
        names = [n for n in names if os.path.splitext(n)[0] in wanted]

    failed = []
    for n in names:
        try:
            merge_pair(mel_files[n], cho_files[n],
                       os.path.join(args.dst, n))
        except Exception as e:
            failed.append((n, repr(e)))
    print(f"Done. {len(names) - len(failed)} merged, {len(failed)} failed.")
    for n, err in failed[:20]:
        print(f"  {n}: {err}")


if __name__ == "__main__":
    main()
