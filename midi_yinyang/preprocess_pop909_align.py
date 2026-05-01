"""Realign POP909 MIDI files so the beat grid matches beat_midi.txt while
keeping every note at its original audio time.

Each song folder POP909-Dataset/POP909/<id>/ contains <id>.mid and beat_midi.txt
(columns: time_sec, strong_beat_flag, downbeat_flag — each row is one quarter
note beat). The midi's note times in seconds are correct; only its tempo track
is wrong. This script:

  * builds a new tempo schedule so audio beat i plays at output midi tick
    (i + 1) * PPQ at exactly beat_midi[i] seconds (with a one-beat lead-in
    segment for the audio time before beat 0);
  * re-positions every note/control/pitch event at the output tick that maps
    back to its original audio time under the new tempo schedule.

After realignment, output midi time t == audio time t for every event, AND the
midi's beat grid (visible in DAWs) lines up with the audio beats.

Output: <out_root>/<id>.mid (preserves original PPQ).
"""

import argparse
import bisect
import os
from glob import glob

import mido
import numpy as np


def build_tick_to_sec(mid):
    """Return tick -> seconds for the given MidiFile, using all set_tempo events."""
    ppq = mid.ticks_per_beat
    events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                events.append((abs_tick, msg.tempo))
    events.sort(key=lambda e: e[0])
    if not events or events[0][0] != 0:
        events.insert(0, (0, 500000))
    ticks = [e[0] for e in events]
    tempos = [e[1] for e in events]
    cum = [0.0]
    for i in range(1, len(events)):
        cum.append(cum[-1] + (ticks[i] - ticks[i - 1]) * tempos[i - 1] / ppq * 1e-6)

    def tick_to_sec(t):
        i = bisect.bisect_right(ticks, t) - 1
        return cum[i] + (t - ticks[i]) * tempos[i] / ppq * 1e-6

    return tick_to_sec


def realign_midi(midi_path, beat_path, out_path):
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    beats = np.loadtxt(beat_path)[:, 0]
    if len(beats) < 2:
        raise ValueError(f"need >=2 beats, got {len(beats)}")

    # New tempo schedule. tempo[0] = lead-in segment (ticks 0..PPQ taking beats[0] s);
    # tempo[i] for i>=1 = audio beat (i-1)->i segment (ticks i*PPQ..(i+1)*PPQ).
    new_tempos = [int(round(max(beats[0], 1e-6) * 1e6))]
    for dt in np.diff(beats):
        new_tempos.append(int(round(max(dt, 1e-6) * 1e6)))
    # cum[i] = output midi time at the start of segment i (= tick i * PPQ)
    cum = [0.0]
    for t in new_tempos[:-1]:
        cum.append(cum[-1] + t / 1e6)

    def time_to_tick(t):
        idx = max(0, min(bisect.bisect_right(cum, t) - 1, len(new_tempos) - 1))
        rel = t - cum[idx]
        return int(round(idx * ppq + rel * 1e6 * ppq / new_tempos[idx]))

    orig_tick_to_sec = build_tick_to_sec(mid)

    out = mido.MidiFile(ticks_per_beat=ppq)

    tempo_track = mido.MidiTrack()
    last_tick = 0
    for i, tempo in enumerate(new_tempos):
        tick = i * ppq
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - last_tick))
        last_tick = tick
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))
    out.tracks.append(tempo_track)

    for track in mid.tracks:
        events = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ("set_tempo", "end_of_track"):
                continue
            t_orig = orig_tick_to_sec(abs_tick)
            events.append((time_to_tick(t_orig), msg))
        events.sort(key=lambda e: e[0])
        new_track = mido.MidiTrack()
        prev = 0
        for at, msg in events:
            new_track.append(msg.copy(time=max(0, at - prev)))
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
