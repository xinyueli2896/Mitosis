"""Extract the MELODY track from each beat-aligned POP909 midi.

Input:  POP909-Dataset/POP909-aligned/<id>.mid (output of preprocess_pop909_align.py).
Output: POP909-Dataset/POP909-melody/<id>.mid containing only the tempo track
        and the track whose name matches "melody" (case-insensitive).
"""

import argparse
import os
from glob import glob

import mido


def extract_melody(in_path, out_path):
    mid = mido.MidiFile(in_path)
    melody_tracks = [t for t in mid.tracks if (t.name or "").strip().lower() == "melody"]
    if not melody_tracks:
        raise ValueError(f"no melody track in {in_path} (tracks: {[t.name for t in mid.tracks]})")

    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    # Track 0: tempo/meta from the original tempo track (assumed first track).
    out.tracks.append(mid.tracks[0])
    out.tracks.extend(melody_tracks)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="POP909-Dataset/POP909-aligned")
    parser.add_argument("--dst", default="POP909-Dataset/POP909-melody")
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()

    midi_paths = sorted(glob(os.path.join(args.src, "*.mid")))
    if args.ids:
        wanted = set(args.ids)
        midi_paths = [p for p in midi_paths if os.path.splitext(os.path.basename(p))[0] in wanted]

    failed = []
    for i, midi_path in enumerate(midi_paths):
        sid = os.path.splitext(os.path.basename(midi_path))[0]
        out_path = os.path.join(args.dst, f"{sid}.mid")
        try:
            extract_melody(midi_path, out_path)
        except Exception as e:
            failed.append((sid, repr(e)))
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(midi_paths)}] processed")

    print(f"Done. {len(midi_paths) - len(failed)} succeeded, {len(failed)} failed.")
    for sid, err in failed[:20]:
        print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
