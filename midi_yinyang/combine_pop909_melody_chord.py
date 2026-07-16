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

Two output modes:

  default      keeps the aligned grid EXACTLY as the rest of the dataset
               (per-beat set_tempo preserving audio timing, one-beat
               lead-in). Use this when the files feed the tokenizer.
  --score      DAW/listening view: constant tempo (the song's average),
               a time_signature derived from beat_midi.txt's downbeat
               flags, and the grid shifted so downbeats land on
               barlines. Note timing is preserved in BEAT space (rubato
               is flattened). Do NOT tokenize score-mode files alongside
               default-mode ones -- the lead-in differs.

Example (rebuild just 001-005):
    python combine_pop909_melody_chord.py \
        --aligned /home/xinyue.li/POP909-Dataset/POP909-aligned \
        --src     /home/xinyue.li/POP909-Dataset/POP909 \
        --dst     /home/xinyue.li/POP909-Dataset/POP909-melody-chord \
        --ids 001 002 003 004 005
"""

import argparse
import bisect
import os
from glob import glob

import mido
import numpy as np

from build_pop909_chord_midi import build_chord_track, parse_chord, \
    chord_to_midi_pitches


def read_beats(beat_txt_path):
    """beat_midi.txt rows: time_sec, beat_flag, downbeat_flag.
    Returns (times, meter, first_downbeat_index)."""
    arr = np.loadtxt(beat_txt_path)
    times = arr[:, 0]
    downbeat_idx = np.flatnonzero(arr[:, 2] > 0.5) if arr.shape[1] >= 3 else []
    if len(downbeat_idx) >= 2:
        meter = int(np.median(np.diff(downbeat_idx)))
        d0 = int(downbeat_idx[0])
    else:
        meter, d0 = 4, 0
        print("  WARNING: no downbeat annotations; assuming 4/4 with "
              "downbeat on the first beat")
    return times, max(meter, 1), d0


def sec_to_beat_fn(beat_times):
    """Piecewise-linear audio-seconds -> beat-position (beat i at position i),
    extrapolating with the edge intervals."""
    bt = list(beat_times)

    def sec_to_beat(t):
        if t <= bt[0]:
            return (t - bt[0]) / max(bt[1] - bt[0], 1e-6)
        if t >= bt[-1]:
            return (len(bt) - 1) + (t - bt[-1]) / max(bt[-1] - bt[-2], 1e-6)
        i = bisect.bisect_right(bt, t) - 1
        return i + (t - bt[i]) / max(bt[i + 1] - bt[i], 1e-6)

    return sec_to_beat


def _shift_track(track, tick_shift):
    """Copy a MidiTrack with every event's absolute tick shifted."""
    out = mido.MidiTrack()
    first = True
    for msg in track:
        if first and msg.time + tick_shift >= 0:
            out.append(msg.copy(time=msg.time + tick_shift))
            first = False
        else:
            out.append(msg.copy())
    return out


def _set_track_program(track, program):
    """Copy a MidiTrack with all program_change events forced to
    `program` (one inserted at tick 0 if the track had none)."""
    out = mido.MidiTrack()
    had_pc = False
    for msg in track:
        if msg.type == "program_change":
            out.append(msg.copy(program=program))
            had_pc = True
        else:
            out.append(msg.copy())
    if not had_pc:
        out.insert(0, mido.Message("program_change", program=program, time=0))
    return out


def _first_note_tick(track):
    at = 0
    for msg in track:
        at += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            return at
    return None


def _retime_track(track, cut_ticks):
    """Copy a MidiTrack with absolute ticks moved earlier by cut_ticks
    (clamped at 0; only tick-0 metas can precede the cut by construction)."""
    out = mido.MidiTrack()
    at = prev = 0
    for msg in track:
        at += msg.time
        new_at = max(0, at - cut_ticks)
        out.append(msg.copy(time=new_at - prev))
        prev = new_at
    return out


def combine_melody_chord(aligned_midi_path, chord_txt_path, out_path,
                         beat_txt_path=None, score=False, bpm=None,
                         trim=False, mel_program=None, chord_program=None):
    src = mido.MidiFile(aligned_midi_path)
    ppq = src.ticks_per_beat

    melody_tracks = [t for t in src.tracks
                     if (t.name or "").strip().lower() == "melody"]
    if not melody_tracks:
        raise ValueError(
            f"no melody track in {aligned_midi_path} "
            f"(tracks: {[t.name for t in src.tracks]})"
        )
    if mel_program is not None:
        melody_tracks = [_set_track_program(t, mel_program)
                         for t in melody_tracks]

    def finish_chord(track):
        return (_set_track_program(track, chord_program)
                if chord_program is not None else track)

    out = mido.MidiFile(ticks_per_beat=ppq)

    if not score:
        # Dataset grid: identical to every other aligned/melody/chord midi.
        out.tracks.append(src.tracks[0])          # tempo/meta track
        out.tracks.extend(melody_tracks)          # MELODY
        out.tracks.append(finish_chord(
            build_chord_track(src.tracks[0], ppq, chord_txt_path)))
    else:
        beat_times, meter, d0 = read_beats(beat_txt_path)
        sec_to_beat = sec_to_beat_fn(beat_times)
        # Bar phase: beat b sits at tick (b + P + meter) * ppq, with P
        # chosen so the first downbeat lands on a barline. The extra
        # `meter` is one leading empty bar that absorbs pickup notes
        # (the aligner's lead-in audio before beat 0).
        P = (-d0) % meter
        base = (P + meter) * ppq

        def beat_to_tick(b):
            return max(0, int(round((b + P + meter) * ppq)))

        # Constant tempo: --bpm if given, else the song's average.
        if bpm is not None:
            beat_sec = 60.0 / bpm
        else:
            beat_sec = ((beat_times[-1] - beat_times[0])
                        / max(len(beat_times) - 1, 1))
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("time_signature", numerator=meter,
                                     denominator=4, time=0))
        meta.append(mido.MetaMessage("set_tempo",
                                     tempo=int(round(beat_sec * 1e6)), time=0))
        meta.append(mido.MetaMessage("end_of_track", time=0))
        out.tracks.append(meta)

        # MELODY: aligned ticks are already beat-linear (beat i at tick
        # (i+1)*ppq), so the score grid is a constant shift of
        # (P + meter - 1) * ppq ticks -- expressive in-beat timing kept.
        note_tracks = [_shift_track(t, (P + meter - 1) * ppq)
                       for t in melody_tracks]

        # CHORD: render chord_midi.txt (audio seconds) straight onto the
        # score grid via the beat map.
        chord_track = mido.MidiTrack()
        chord_track.append(mido.MetaMessage("track_name", name="CHORD",
                                            time=0))
        events = []
        with open(chord_txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                t_start, t_end, label = (float(parts[0]), float(parts[1]),
                                         parts[2])
                parsed = parse_chord(label)
                if parsed is None:
                    continue
                root, intervals, bass_interval = parsed
                pitches = chord_to_midi_pitches(root, intervals,
                                                bass_interval)
                tick_on = beat_to_tick(sec_to_beat(t_start))
                tick_off = max(tick_on + 1, beat_to_tick(sec_to_beat(t_end)))
                for p in pitches:
                    events.append((tick_on, 0, p))
                    events.append((tick_off, 1, p))
        events.sort(key=lambda e: (e[0], -e[1]))
        prev_tick = 0
        for tick, kind, pitch in events:
            delta = max(0, tick - prev_tick)
            if kind == 0:
                chord_track.append(mido.Message("note_on", note=pitch,
                                                velocity=80, time=delta))
            else:
                chord_track.append(mido.Message("note_off", note=pitch,
                                                velocity=0, time=delta))
            prev_tick = tick
        chord_track.append(mido.MetaMessage("end_of_track", time=0))
        note_tracks.append(finish_chord(chord_track))

        if trim:
            # Cut leading silence in whole bars so downbeats stay on
            # barlines. The cut point is the first note in EITHER stream.
            firsts = [t for t in (_first_note_tick(tr) for tr in note_tracks)
                      if t is not None]
            if firsts:
                bar = meter * ppq
                cut = (min(firsts) // bar) * bar
                if cut > 0:
                    note_tracks = [_retime_track(tr, cut)
                                   for tr in note_tracks]
                    print(f"    trimmed {cut // bar} leading empty bar(s)")

        out.tracks.extend(note_tracks)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", default="POP909-Dataset/POP909-aligned")
    parser.add_argument("--src", default="POP909-Dataset/POP909")
    parser.add_argument("--dst", default="POP909-Dataset/POP909-melody-chord")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--score", action="store_true", default=False,
                        help="DAW/listening view: constant tempo, "
                             "time signature, downbeats on barlines "
                             "(needs <src>/<id>/beat_midi.txt). Not for "
                             "tokenization.")
    parser.add_argument("--bpm", type=float, default=None,
                        help="Constant tempo for --score mode (e.g. 120). "
                             "Default: the song's average tempo.")
    parser.add_argument("--trim", action="store_true", default=False,
                        help="--score mode: cut leading silence (whole "
                             "bars only, so downbeats stay on barlines).")
    parser.add_argument("--mel-program", type=int, default=None,
                        help="Force the MELODY track's midi program. Give "
                             "melody and chord DISTINCT programs when the "
                             "file feeds the single-stream CP transformer, "
                             "so the tokenizer keeps the streams apart and "
                             "generated output decodes into separate "
                             "tracks (e.g. --mel-program 0 "
                             "--chord-program 48).")
    parser.add_argument("--chord-program", type=int, default=None,
                        help="Force the CHORD track's midi program.")
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
        beat_txt = os.path.join(args.src, sid, "beat_midi.txt")
        out_path = os.path.join(args.dst, f"{sid}.mid")
        if not os.path.exists(chord_txt):
            failed.append((sid, "missing chord_midi.txt"))
            continue
        if args.score and not os.path.exists(beat_txt):
            failed.append((sid, "missing beat_midi.txt (needed for --score)"))
            continue
        try:
            combine_melody_chord(midi_path, chord_txt, out_path,
                                 beat_txt_path=beat_txt, score=args.score,
                                 bpm=args.bpm, trim=args.trim,
                                 mel_program=args.mel_program,
                                 chord_program=args.chord_program)
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
