"""Realign POP909 MIDI files to the corrected beat times in beat_midi.txt.

Each song folder POP909-Dataset/POP909/<id>/ contains <id>.mid and beat_midi.txt
(columns: time_sec, beat_in_measure, downbeat_flag). The midi's own tempo track
does not match the true beat positions; this script piecewise-linearly time-warps
every event so beat i in the original midi (tick i * PPQ) lands at beat_midi.txt[i].

Output: <out_root>/<id>.mid, written under a fixed 120 BPM tempo (after warping
the absolute event times the choice of output tempo only affects tick resolution).
"""

import argparse
import bisect
import os
from glob import glob

import mido
import numpy as np


def build_tick_to_sec(mid):
    """Return a function tick -> seconds using all set_tempo events in the file."""
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
        events.insert(0, (0, 500000))  # default 120 BPM if missing at tick 0
    ticks = [e[0] for e in events]
    tempos = [e[1] for e in events]
    cum_time = [0.0]
    for i in range(1, len(events)):
        cum_time.append(cum_time[-1] + (ticks[i] - ticks[i - 1]) * tempos[i - 1] / ppq * 1e-6)

    def tick_to_sec(t):
        i = bisect.bisect_right(ticks, t) - 1
        return cum_time[i] + (t - ticks[i]) * tempos[i] / ppq * 1e-6

    return tick_to_sec


def remap_time(t_orig, orig_beats, new_beats):
    """Piecewise-linear remap from orig_beats grid to new_beats grid.
    Outside the grid, extrapolate using the slope of the nearest interval."""
    n = len(orig_beats)
    if t_orig <= orig_beats[0]:
        slope = (new_beats[1] - new_beats[0]) / (orig_beats[1] - orig_beats[0]) if n > 1 else 1.0
        return new_beats[0] + (t_orig - orig_beats[0]) * slope
    if t_orig >= orig_beats[-1]:
        slope = (new_beats[-1] - new_beats[-2]) / (orig_beats[-1] - orig_beats[-2]) if n > 1 else 1.0
        return new_beats[-1] + (t_orig - orig_beats[-1]) * slope
    idx = bisect.bisect_right(orig_beats, t_orig) - 1
    f = (t_orig - orig_beats[idx]) / (orig_beats[idx + 1] - orig_beats[idx])
    return new_beats[idx] + f * (new_beats[idx + 1] - new_beats[idx])


def realign_midi(midi_path, beat_path, out_path, out_tempo=500000):
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    new_beat_times = np.loadtxt(beat_path)[:, 0]
    n_beats = len(new_beat_times)

    tick_to_sec = build_tick_to_sec(mid)
    orig_beat_times = [tick_to_sec(i * ppq) for i in range(n_beats)]
    sec_per_tick = out_tempo / ppq * 1e-6

    out_mid = mido.MidiFile(ticks_per_beat=ppq)
    # Dedicated tempo/meta track at index 0.
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("set_tempo", tempo=out_tempo, time=0))
    out_mid.tracks.append(meta_track)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        abs_tick = 0
        events = []
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                continue  # rely on the fixed-tempo meta track
            events.append((abs_tick, msg))

        warped = []
        for at, msg in events:
            t_warp = remap_time(tick_to_sec(at), orig_beat_times, new_beat_times)
            warped.append((max(0, int(round(t_warp / sec_per_tick))), msg))
        warped.sort(key=lambda e: e[0])  # beat warp is monotone, but be defensive

        prev_tick = 0
        for new_tick, msg in warped:
            new_track.append(msg.copy(time=max(0, new_tick - prev_tick)))
            prev_tick = new_tick
        out_mid.tracks.append(new_track)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_mid.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="POP909-Dataset/POP909",
                        help="Path to POP909 song folders")
    parser.add_argument("--dst", default="POP909-Dataset/POP909-aligned",
                        help="Output directory for realigned midis")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Optional list of song ids to process (e.g. 001 042)")
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
