"""Realign POP909 MIDI files so the beat grid matches beat_midi.txt while
keeping every note at its original audio time.

Each song folder POP909-Dataset/POP909/<id>/ contains <id>.mid and beat_midi.txt
(columns: time_sec, strong_beat_flag, downbeat_flag — each row is one quarter
note beat). The midi's note times in seconds are correct; only its tempo track
is wrong. This script:

  * builds a new tempo schedule so audio beat i plays at output midi tick
    (i + pad) * PPQ at exactly beat_midi[i] seconds, where pad puts the
    song's FIRST DOWNBEAT (third column of beat_midi.txt) on a bar
    boundary of the 4-beat tick grid: pad = (-d0) mod 4 for first-downbeat
    index d0. The pad beats (0..3 of them) hold the audio lead-in before
    beat 0, so output midi time t == audio time t still holds for every
    event (the chord builder reuses this tempo map); with pad = 0 the
    lead-in is dropped and anything before beat 0 clamps to tick 0.
  * re-positions every note/control/pitch event at the output tick that maps
    back to its original audio time under the new tempo schedule.
  * verifies that beat d0 sits at a multiple of 4 * PPQ and refuses to
    write the file otherwise.

History (2026-09-10): the previous version wrote beat i at tick
(i + 1) * PPQ -- a one-beat lead-in for every song -- and never read
the downbeat column. Since only a third of POP909 songs start on a
downbeat, the bar phase of the aligned corpus was spread over all four
beats (14% on the grid, 34% one beat late, 31% half a bar, 21% one
beat early; check_downbeat_phase.py). Training is unaffected (random
crop windows, relative positions), but E1 prompts were cut at
arbitrary beats, bar-based metrics used wrong bars, the whole-song
baseline was fed mis-phased lead sheets, and the listening copies
played off the DAW grid. --legacy reproduces the old layout.

After realignment, output midi time t == audio time t for every event, AND the
midi's beat grid (visible in DAWs) lines up with the audio beats and bars.

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


def first_downbeat_index(rows):
    """Index of the first beat whose downbeat flag (last column) is set;
    None when the file has no downbeat column or no flag."""
    if rows.ndim < 2 or rows.shape[1] < 2:
        return None
    idx = np.where(rows[:, -1] >= 0.5)[0]
    return int(idx[0]) if len(idx) else None


def realign_midi(midi_path, beat_path, out_path, legacy=False):
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    rows = np.loadtxt(beat_path)
    if rows.ndim == 1:
        rows = rows[None, :]
    beats = rows[:, 0]
    if len(beats) < 2:
        raise ValueError(f"need >=2 beats, got {len(beats)}")

    if legacy:
        pad = 1                                  # the old one-beat lead-in
    else:
        d0 = first_downbeat_index(rows)
        if d0 is None:
            raise ValueError("no downbeat flag in beat_midi.txt")
        pad = (-d0) % 4                          # beat d0 -> tick multiple of 4*PPQ

    # New tempo schedule, one segment per beat of ticks:
    #   segments 0..pad-1   the pad beats, sharing the lead-in (beats[0] s);
    #   segment pad + i     audio beat i -> i+1  (ticks (pad+i)*PPQ ..).
    new_tempos = []
    if pad > 0:
        lead = max(beats[0], 1e-6) / pad
        new_tempos += [int(round(lead * 1e6))] * pad
    for dt in np.diff(beats):
        new_tempos.append(int(round(max(dt, 1e-6) * 1e6)))
    # cum[i] = output midi time at the start of segment i (= tick i * PPQ)
    cum = [0.0]
    for t in new_tempos[:-1]:
        cum.append(cum[-1] + t / 1e6)
    t0 = 0.0 if pad > 0 else beats[0]            # audio time at tick 0

    def time_to_tick(t):
        t = max(t - t0, 0.0)                     # pad == 0: lead-in clamps to tick 0
        idx = max(0, min(bisect.bisect_right(cum, t) - 1, len(new_tempos) - 1))
        rel = t - cum[idx]
        return int(round(idx * ppq + rel * 1e6 * ppq / new_tempos[idx]))

    if not legacy:
        db_tick = time_to_tick(beats[d0])
        if db_tick % (4 * ppq) != 0:
            raise AssertionError(
                f"first downbeat (beat {d0}) lands at tick {db_tick}, not on the 4-beat grid")

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
    parser.add_argument("--legacy", action="store_true",
                        help="reproduce the pre-2026-09-10 layout (beat i at tick (i+1)*PPQ, "
                             "downbeats ignored)")
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
            realign_midi(midi_path, beat_path, out_path, legacy=args.legacy)
        except Exception as e:
            failed.append((sid, repr(e)))
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(song_dirs)}] processed")

    print(f"Done. {len(song_dirs) - len(failed)} succeeded, {len(failed)} failed.")
    for sid, err in failed[:20]:
        print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
