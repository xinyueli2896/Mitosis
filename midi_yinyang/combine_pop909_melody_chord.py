"""Build combined melody+chord midis for beat-aligned POP909 songs.

For each song id:
  * read <aligned>/<id>.mid (output of preprocess_pop909_align.py; provides
    the tempo track and the MELODY track)
  * render <src>/<id>/chord_midi.txt as a CHORD track on the same tempo
    grid (same logic as build_pop909_chord_midi.py, reused directly)
  * write <dst>/<id>.mid with three tracks: tempo, MELODY, CHORD.

Useful when the separate chord midis were lost but the aligned midis and
the chord annotations survive, or when a single listenable melody+chord
file per song is wanted.

Example (rebuild just 001-005):
    python combine_pop909_melody_chord.py \
        --aligned /home/xinyue.li/POP909-Dataset/POP909-aligned \
        --src     /home/xinyue.li/POP909-Dataset/POP909 \
        --dst     /home/xinyue.li/POP909-Dataset/POP909-melody-chord \
        --ids 001 002 003 004 005
"""

import argparse
import os
from glob import glob

import mido

from build_pop909_chord_midi import build_chord_track


def combine_melody_chord(aligned_midi_path, chord_txt_path, out_path):
    src = mido.MidiFile(aligned_midi_path)
    ppq = src.ticks_per_beat

    melody_tracks = [t for t in src.tracks
                     if (t.name or "").strip().lower() == "melody"]
    if not melody_tracks:
        raise ValueError(
            f"no melody track in {aligned_midi_path} "
            f"(tracks: {[t.name for t in src.tracks]})"
        )

    out = mido.MidiFile(ticks_per_beat=ppq)
    out.tracks.append(src.tracks[0])          # tempo/meta track
    out.tracks.extend(melody_tracks)          # MELODY
    out.tracks.append(build_chord_track(src.tracks[0], ppq, chord_txt_path))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", default="POP909-Dataset/POP909-aligned")
    parser.add_argument("--src", default="POP909-Dataset/POP909")
    parser.add_argument("--dst", default="POP909-Dataset/POP909-melody-chord")
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()

    midi_paths = sorted(glob(os.path.join(args.aligned, "*.mid")))
    if args.ids:
        wanted = set(args.ids)
        midi_paths = [p for p in midi_paths
                      if os.path.splitext(os.path.basename(p))[0] in wanted]
        found = {os.path.splitext(os.path.basename(p))[0] for p in midi_paths}
        for sid in sorted(wanted - found):
            print(f"  WARNING: {sid}: no aligned midi found in {args.aligned}")

    failed = []
    for i, midi_path in enumerate(midi_paths):
        sid = os.path.splitext(os.path.basename(midi_path))[0]
        chord_txt = os.path.join(args.src, sid, "chord_midi.txt")
        out_path = os.path.join(args.dst, f"{sid}.mid")
        if not os.path.exists(chord_txt):
            failed.append((sid, "missing chord_midi.txt"))
            continue
        try:
            combine_melody_chord(midi_path, chord_txt, out_path)
            print(f"  {sid} -> {out_path}")
        except Exception as e:
            failed.append((sid, repr(e)))
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(midi_paths)}] processed")

    print(f"Done. {len(midi_paths) - len(failed)} succeeded, {len(failed)} failed.")
    for sid, err in failed[:20]:
        print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
