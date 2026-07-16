"""Split combined melody+chord midis back into separate per-stream midis.

Works on two kinds of input:
  * the combined POP909 files from combine_pop909_melody_chord.py --
    tracks are matched by NAME ("MELODY" / "CHORD");
  * generated midis from the single-stream CP transformer (decode_output)
    -- tracks have no names, so they are matched by their program_change
    (--mel-program / --chord-program, matching what the prompt files were
    built with, e.g. 0 and 48).

Each output midi keeps the source's tempo/meta track (track 0) plus the
matched stream tracks, so timing and bar structure are identical to the
source.

Example:
    python split_melody_chord.py \
        --src input/POP909-melody-chord-score \
        --dst-melody temp/split/melody \
        --dst-chord  temp/split/chord \
        --mel-program 0 --chord-program 48
"""

import argparse
import os
from glob import glob

import mido


def _track_first_program(track):
    for msg in track:
        if msg.type == "program_change":
            return msg.program
    return None


def _track_has_notes(track):
    return any(msg.type == "note_on" and msg.velocity > 0 for msg in track)


def classify_tracks(mid, mel_program, chord_program):
    """Return (meta_tracks, melody_tracks, chord_tracks, unmatched)."""
    meta, melody, chord, unmatched = [], [], [], []
    for i, track in enumerate(mid.tracks):
        if not _track_has_notes(track):
            meta.append(track)
            continue
        name = (track.name or "").strip().lower()
        prog = _track_first_program(track)
        if name == "melody":
            melody.append(track)
        elif name == "chord":
            chord.append(track)
        elif prog is not None and prog == mel_program != chord_program:
            melody.append(track)
        elif prog is not None and prog == chord_program != mel_program:
            chord.append(track)
        else:
            unmatched.append((i, track.name, prog))
    return meta, melody, chord, unmatched


def write_stream(path, ppq, meta_tracks, stream_tracks):
    out = mido.MidiFile(ticks_per_beat=ppq)
    for t in meta_tracks:
        out.tracks.append(t)
    for t in stream_tracks:
        out.tracks.append(t)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.save(path)


def split_file(in_path, mel_out, chord_out, mel_program, chord_program):
    mid = mido.MidiFile(in_path)
    meta, melody, chord, unmatched = classify_tracks(
        mid, mel_program, chord_program)
    if unmatched:
        for i, name, prog in unmatched:
            print(f"    WARNING: track {i} (name={name!r}, program={prog}) "
                  f"matched neither stream; dropped")
    if not melody:
        raise ValueError("no melody track found (by name or program)")
    if not chord:
        raise ValueError("no chord track found (by name or program)")
    write_stream(mel_out, mid.ticks_per_beat, meta, melody)
    write_stream(chord_out, mid.ticks_per_beat, meta, chord)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="folder of combined midis (or a single .mid)")
    parser.add_argument("--dst-melody", required=True)
    parser.add_argument("--dst-chord", required=True)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--mel-program", type=int, default=0,
                        help="program fallback for unnamed melody tracks")
    parser.add_argument("--chord-program", type=int, default=48,
                        help="program fallback for unnamed chord tracks")
    args = parser.parse_args()

    if os.path.isfile(args.src):
        paths = [args.src]
    else:
        paths = sorted(glob(os.path.join(args.src, "*.mid"))
                       + glob(os.path.join(args.src, "*.MID")))
    if args.ids:
        wanted = set(args.ids)
        paths = [p for p in paths
                 if os.path.splitext(os.path.basename(p))[0] in wanted]

    failed = []
    for p in paths:
        base = os.path.basename(p)
        try:
            split_file(p,
                       os.path.join(args.dst_melody, base),
                       os.path.join(args.dst_chord, base),
                       args.mel_program, args.chord_program)
            print(f"  {base} -> melody + chord")
        except Exception as e:
            failed.append((base, repr(e)))

    print(f"Done. {len(paths) - len(failed)} succeeded, {len(failed)} failed.")
    for base, err in failed[:20]:
        print(f"  {base}: {err}")


if __name__ == "__main__":
    main()
