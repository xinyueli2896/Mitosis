"""The tempo a MIDI file is 'at', robust to beat-synced tempo maps.

The time-aligned POP909 files (preprocess_pop909_align.py) carry one
set_tempo event per beat, and the FIRST event covers the lead-in from
tick 0 to the first annotated beat: about 0.06 s, which encodes as
~990 bpm. Every reader that took 'the tempo' to be the first event
(the output writers' get_input_tempo, the scorer's old framing, the
hybrid builder) therefore saw ~990 for every time-aligned song and
wrote generated outputs at that speed. Ticks are unaffected -- the
tokenizer and the scorer read ticks -- but the files are unplayable.

main_tempo() returns the tick-weighted MEDIAN tempo: sort the tempo
segments by bpm and take the bpm at half the file's total ticks. For a
constant-tempo file that is the tempo; for a beat-synced map it is the
typical beat, and a 0.06 s lead-in cannot move it.
"""
import warnings

import mido


def main_tempo(path, default=120.0):
    try:
        m = mido.MidiFile(path)
    except Exception:       # noqa: BLE001
        return float(default)
    events = []
    t = 0
    end = 0
    for msg in mido.merge_tracks(m.tracks):
        t += msg.time
        if msg.type == 'set_tempo':
            events.append((t, mido.tempo2bpm(msg.tempo)))
    end = t
    if not events:
        return float(default)
    events.sort()
    if events[0][0] != 0:
        events.insert(0, (0, default))
    segs = []
    for i, (tick, bpm) in enumerate(events):
        nxt = events[i + 1][0] if i + 1 < len(events) else max(end, tick)
        if nxt > tick and bpm > 0:
            segs.append((bpm, nxt - tick))
    if not segs:
        return float(events[0][1]) if events[0][1] > 0 else float(default)
    segs.sort()
    total = sum(d for _, d in segs)
    acc = 0
    for bpm, d in segs:
        acc += d
        if acc * 2 >= total:
            return float(bpm)
    return float(segs[-1][0])


def main_tempo_pm(pm, default=120.0):
    """Same, from a loaded pretty_midi object (tempo change list)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            times, tempi = pm.get_tempo_changes()
    except Exception:       # noqa: BLE001
        return float(default)
    if len(tempi) == 0:
        return float(default)
    end = max([pm.get_end_time()] + list(times))
    segs = []
    for i, (t0, bpm) in enumerate(zip(times, tempi)):
        t1 = times[i + 1] if i + 1 < len(times) else end
        if t1 > t0 and bpm > 0:
            # weight by TICKS (beats), not seconds: a 990 bpm lead-in is
            # short in seconds but its tick span is what the grid uses
            segs.append((float(bpm), (t1 - t0) * bpm))
    if not segs:
        return float(tempi[0]) if tempi[0] > 0 else float(default)
    segs.sort()
    total = sum(w for _, w in segs)
    acc = 0.0
    for bpm, w in segs:
        acc += w
        if acc * 2 >= total:
            return bpm
    return segs[-1][0]
