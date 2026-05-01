"""Realign POP909 MIDI files so their beat grid matches beat_midi.txt.

Each song folder POP909-Dataset/POP909/<id>/ contains <id>.mid and beat_midi.txt
(columns: time_sec, strong_beat_flag, downbeat_flag — each row is one quarter
note beat). The midi's own tempo track does not match the audio beat times;
this script rewrites the tempo so audio beat i lands at output midi tick
(i + 1) * PPQ at exactly beat_midi[i] seconds. Concretely:

  * Output tick 0 .. PPQ           -> tempo such that the segment takes beats[0]
                                       seconds (the audio lead-in).
  * Output tick i*PPQ .. (i+1)*PPQ -> tempo such that the segment takes
                                       beats[i] - beats[i-1] seconds, for i >= 1.
  * Every original event is shifted forward by PPQ ticks so the musical beat
    structure aligns: original tick i*PPQ -> output tick (i+1)*PPQ.

Within a beat segment the tempo is constant, which is the natural linear
interpolation in time for sub-beat events. After realignment, audio time t
corresponds exactly to output midi time t (no offset needed for sync).

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
    beats = np.loadtxt(beat_path)[:, 0]
    if len(beats) < 2:
        raise ValueError(f"need >=2 beats, got {len(beats)}")

    # Tempo for segment [i*PPQ, (i+1)*PPQ]:
    #   i == 0:  duration = beats[0]                  (lead-in before audio beat 0)
    #   i >= 1:  duration = beats[i] - beats[i-1]     (audio beat i-1 -> beat i)
    tempos_us = [int(round(max(beats[0], 1e-6) * 1e6))]
    for dt in np.diff(beats):
        tempos_us.append(int(round(max(dt, 1e-6) * 1e6)))

    out = mido.MidiFile(ticks_per_beat=ppq)

    tempo_track = mido.MidiTrack()
    last_tick = 0
    for i, tempo in enumerate(tempos_us):
        tick = i * ppq
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - last_tick))
        last_tick = tick
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))
    out.tracks.append(tempo_track)

    # Shift every other event forward by one beat (PPQ ticks) so the original
    # musical position at tick i*PPQ lands on output tick (i+1)*PPQ, which under
    # the tempo schedule above plays at beats[i].
    SHIFT = ppq
    for track in mid.tracks:
        events = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ("set_tempo", "end_of_track"):
                continue
            events.append((abs_tick + SHIFT, msg))
        new_track = mido.MidiTrack()
        prev = 0
        for at, msg in events:
            new_track.append(msg.copy(time=at - prev))
            prev = at
        new_track.append(mido.MetaMessage("end_of_track", time=0))
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
