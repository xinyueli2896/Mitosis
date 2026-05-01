"""Realign POP909 MIDI files so their beat grid matches beat_midi.txt.

Each song folder POP909-Dataset/POP909/<id>/ contains <id>.mid and beat_midi.txt
(columns: time_sec, beat_in_measure, downbeat_flag). The midi's own tempo track
does not match the true beat positions. This script preserves all event ticks
(so musical beat positions stay intact) and replaces the tempo track with one
tempo event per beat such that beat i (tick i * PPQ) plays at beat_midi[i] -
beat_midi[0]. Within a beat segment the tempo is constant, which is the natural
linear interpolation in time for sub-beat events.

The absolute lead-in offset (beat_midi[0]) is dropped: in the output, audio
beat 0 plays at midi time 0. Re-add it by inserting a silent preroll if needed.

Output: <out_root>/<id>.mid (preserves original PPQ).
"""

import argparse
import os
from glob import glob

import mido
import numpy as np


def realign_midi(midi_path, beat_path, out_path):
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    new_beats = np.loadtxt(beat_path)[:, 0]
    if len(new_beats) < 2:
        raise ValueError(f"need >=2 beats, got {len(new_beats)}")
    # tempo per beat segment, in microseconds per quarter note
    tempos_us = np.maximum(1, np.round(np.diff(new_beats) * 1e6).astype(np.int64))

    out = mido.MidiFile(ticks_per_beat=ppq)

    tempo_track = mido.MidiTrack()
    last_tick = 0
    for i, tempo in enumerate(tempos_us):
        tick = i * ppq
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=int(tempo), time=tick - last_tick))
        last_tick = tick
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))
    out.tracks.append(tempo_track)

    # Copy all original tracks verbatim except: drop their set_tempo events
    # (replaced by the dedicated tempo track above). end_of_track stays.
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        abs_tick = 0
        last_emit_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                continue  # absorbed; the next emitted message gets the accumulated delta
            if msg.type == "end_of_track":
                continue  # we'll re-append at the end
            new_track.append(msg.copy(time=abs_tick - last_emit_tick))
            last_emit_tick = abs_tick
        new_track.append(mido.MetaMessage("end_of_track", time=max(0, abs_tick - last_emit_tick)))
        out.tracks.append(new_track)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="POP909-Dataset/POP909")
    parser.add_argument("--dst", default="POP909-Dataset/POP909-aligned")
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()

    song_dirs = sorted(d for d in glob(os.path.join(args.src, "*")) if os.path.isdir(d))
    if args.ids:
        wanted = set(args.ids)
        song_dirs = [d for d in song_dirs if os.path.basename(d) in wanted]

    failed = []
    for i, song_dir in enumerate(song_dirs):
        sid = os.path.basename(song_dir)
        midi_path = os.path.join(song_dir, f"{sid}.mid")
        beat_path = os.path.join(song_dir, "beat_midi.txt")
        out_path = os.path.join(args.dst, f"{sid}.mid")
        if not (os.path.exists(midi_path) and os.path.exists(beat_path)):
            failed.append((sid, "missing input"))
            continue
        try:
            realign_midi(midi_path, beat_path, out_path)
        except Exception as e:
            failed.append((sid, repr(e)))
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(song_dirs)}] processed")

    print(f"Done. {len(song_dirs) - len(failed)} succeeded, {len(failed)} failed.")
    for sid, err in failed[:20]:
        print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
