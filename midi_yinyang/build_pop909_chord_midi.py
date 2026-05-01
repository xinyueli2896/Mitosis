"""Build a chord-track midi for each beat-aligned POP909 song from chord_midi.txt.

For each song:
  * read POP909-Dataset/POP909-aligned/<id>.mid (uses its tempo track so the
    chord midi shares the same beat grid and audio-time sync)
  * parse POP909-Dataset/POP909/<id>/chord_midi.txt (start, end, label) where
    labels follow Harte syntax (e.g. C:maj, F#:min7, Bb:sus4(b7), G:maj/5, N)
  * emit a single CHORD track with three to five simultaneous notes per chord
    (bass note around C2, chord tones around C4)
  * write POP909-Dataset/POP909-chord/<id>.mid

The aligned midi's tempo schedule is reused so chord tick positions land at the
correct audio times; the same time_to_tick logic is used as in
preprocess_pop909_align.py.
"""

import argparse
import bisect
import os
import re
from glob import glob

import mido


PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Harte interval (scale degree relative to a major scale root) -> semitones.
INTERVAL_SEMITONES = {
    "1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11,
    "9": 14, "11": 17, "13": 21,
}

QUALITY_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "min9": (0, 3, 7, 10, 14),
}


def parse_root(s):
    """C, C#, Cb, Bb, F##, etc. -> pitch class 0..11."""
    pc = PITCH_CLASS[s[0]]
    for ch in s[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
    return pc % 12


def parse_interval(token):
    """'5' -> 7, 'b7' -> 10, '#5' -> 8, etc."""
    m = re.match(r"^([#b]*)(\d+)$", token)
    if not m:
        raise ValueError(f"bad interval token: {token!r}")
    accidentals, deg = m.group(1), m.group(2)
    semi = INTERVAL_SEMITONES[deg]
    semi += accidentals.count("#") - accidentals.count("b")
    return semi


def parse_chord(label):
    """Return (root_pc, intervals_set, bass_interval) or None for 'N'."""
    if label == "N":
        return None
    bass_token = None
    main = label
    if "/" in main:
        main, bass_token = main.split("/", 1)
    if ":" in main:
        root_str, qual = main.split(":", 1)
    else:
        root_str, qual = main, "maj"
    root = parse_root(root_str)
    intervals = set(_intervals_for_quality(qual))
    bass_interval = parse_interval(bass_token) if bass_token else 0
    intervals.add(bass_interval)
    return root, sorted(intervals), bass_interval


def _intervals_for_quality(qual):
    """Resolve a quality string, possibly with a parenthesised extension list,
    e.g. 'sus4(b7)', 'maj7(9,11)'."""
    m = re.match(r"^([a-zA-Z0-9]+)(?:\(([^)]+)\))?$", qual)
    if not m:
        return QUALITY_INTERVALS.get(qual, (0, 4, 7))
    base, extras = m.group(1), m.group(2)
    intervals = list(QUALITY_INTERVALS.get(base, (0, 4, 7)))
    if extras:
        for tok in extras.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                intervals.append(parse_interval(tok))
            except (ValueError, KeyError):
                pass
    return tuple(intervals)


def chord_to_midi_pitches(root, intervals, bass_interval, bass_root_midi=36, chord_root_midi=60):
    """Pick concrete MIDI pitches: bass at C2-B2 (root_midi=36) + upper voicing at C4-B4 (=60)."""
    bass_pitch = bass_root_midi + (root + bass_interval) % 12
    chord_pitches = sorted({chord_root_midi + (root + i) % 12 for i in intervals})
    return [bass_pitch] + chord_pitches


def build_time_to_tick(tempo_track, ppq):
    """Walk a tempo MidiTrack and return a function mapping seconds -> ticks."""
    abs_tick = 0
    events = []
    for msg in tempo_track:
        abs_tick += msg.time
        if msg.type == "set_tempo":
            events.append((abs_tick, msg.tempo))
    if not events or events[0][0] != 0:
        events.insert(0, (0, 500000))
    cum = [0.0]
    for i in range(1, len(events)):
        cum.append(cum[-1] + (events[i][0] - events[i - 1][0]) * events[i - 1][1] / ppq * 1e-6)

    def time_to_tick(t):
        idx = max(0, bisect.bisect_right(cum, t) - 1)
        rel = t - cum[idx]
        tempo = events[idx][1]
        return int(round(events[idx][0] + rel * 1e6 * ppq / tempo))

    return time_to_tick


def build_chord_midi(aligned_midi_path, chord_txt_path, out_path):
    src = mido.MidiFile(aligned_midi_path)
    ppq = src.ticks_per_beat
    time_to_tick = build_time_to_tick(src.tracks[0], ppq)

    out = mido.MidiFile(ticks_per_beat=ppq)
    out.tracks.append(src.tracks[0])  # share the same tempo schedule

    chord_track = mido.MidiTrack()
    chord_track.append(mido.MetaMessage("track_name", name="CHORD", time=0))

    events = []  # (tick, kind, pitch, velocity)  kind: 0=on, 1=off
    with open(chord_txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            t_start, t_end, label = float(parts[0]), float(parts[1]), parts[2]
            parsed = parse_chord(label)
            if parsed is None:
                continue
            root, intervals, bass_interval = parsed
            pitches = chord_to_midi_pitches(root, intervals, bass_interval)
            tick_on = time_to_tick(t_start)
            tick_off = max(tick_on + 1, time_to_tick(t_end))
            for p in pitches:
                events.append((tick_on, 0, p))
                events.append((tick_off, 1, p))

    # Off-events first when ticks tie, so re-attacks don't get cancelled.
    events.sort(key=lambda e: (e[0], -e[1]))

    prev_tick = 0
    for tick, kind, pitch in events:
        delta = max(0, tick - prev_tick)
        if kind == 0:
            chord_track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            chord_track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
        prev_tick = tick

    chord_track.append(mido.MetaMessage("end_of_track", time=0))
    out.tracks.append(chord_track)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", default="POP909-Dataset/POP909-aligned")
    parser.add_argument("--src", default="POP909-Dataset/POP909")
    parser.add_argument("--dst", default="POP909-Dataset/POP909-chord")
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()

    midi_paths = sorted(glob(os.path.join(args.aligned, "*.mid")))
    if args.ids:
        wanted = set(args.ids)
        midi_paths = [p for p in midi_paths if os.path.splitext(os.path.basename(p))[0] in wanted]

    failed = []
    for i, midi_path in enumerate(midi_paths):
        sid = os.path.splitext(os.path.basename(midi_path))[0]
        chord_txt = os.path.join(args.src, sid, "chord_midi.txt")
        out_path = os.path.join(args.dst, f"{sid}.mid")
        if not os.path.exists(chord_txt):
            failed.append((sid, "missing chord_midi.txt"))
            continue
        try:
            build_chord_midi(midi_path, chord_txt, out_path)
        except Exception as e:
            failed.append((sid, repr(e)))
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(midi_paths)}] processed")

    print(f"Done. {len(midi_paths) - len(failed)} succeeded, {len(failed)} failed.")
    for sid, err in failed[:20]:
        print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
