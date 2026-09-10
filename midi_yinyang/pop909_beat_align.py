"""Beat-align POP909 from the raw dataset: one mapping for everything.

Input, per song folder <raw>/<id>/:  <id>.mid, beat_midi.txt, chord_midi.txt
Output, under <dst>/:
  aligned/<id>.mid   every original track, re-timed onto the beat grid
  melody/<id>.mid    tempo track + the MELODY track only
  chord/<id>.mid     tempo track + a CHORD track rendered from chord_midi.txt
  align_report.tsv   one row per song (see below)

The mapping. beat_midi.txt gives the audio time t_i of every beat i (and
a downbeat flag). The raw midi's note times in seconds are right and its
tempo track is wrong, so each event is first converted to seconds with
the raw tempo track and then to a fractional BEAT POSITION by linear
interpolation between the annotated beat times:

    b(t) = i + (t - t_i) / (t_{i+1} - t_i)     for t_i <= t < t_{i+1}

(extrapolated with the first / last beat interval outside the annotated
range). The output tick is

    tick(t) = round((b(t) + pad) * PPQ)

with pad = (-d0) mod 4 for first-downbeat index d0, so that the song's
first downbeat sits on a bar boundary of the 4-beat tick grid (--no-pad
sets pad = 0). Melody notes, every other track, and the chord
annotations all go through this ONE function, so melody and chords
cannot disagree by construction. Nothing is added or removed: pad only
decides which tick is zero; the 0..3 empty beats it creates hold the
pickup as an incomplete first bar.

The output tempo track has one tempo per beat, equal to the annotated
beat interval, so a DAW plays beat i at audio time t_i plus a constant
(the pad beats take the first beat interval). It is not needed by the
pipeline, which works on ticks, but it keeps listening honest; there
are no absurd lead-in tempos any more.

What this cannot fix: bars that POP909 annotates as 2, 3, 5 or 6 beats
long. After such a bar the true downbeats leave the 4-beat grid; the
report counts them per song (n_irregular, first_irregular_bar) so the
evaluation can choose songs accordingly.

Chord rendering follows build_pop909_chord_midi.py (Harte labels ->
bass at C2 plus upper voices at C4, 'N' skipped), imported from there.

Usage (via align_pop909.sbatch, CPU):
    python pop909_beat_align.py --raw /home/xinyue.li/POP909-Dataset/POP909-git/POP909 \\
        --dst /home/xinyue.li/POP909-Dataset/POP909-v3
"""
import argparse
import bisect
import csv
import os
from glob import glob

import mido
import numpy as np

from build_pop909_chord_midi import parse_chord, chord_to_midi_pitches


# ---------------------------------------------------------------------------
# raw midi ticks -> seconds (the raw tempo track is wrong for the beat grid
# but right for absolute time, which is all we take from it)
# ---------------------------------------------------------------------------

def raw_tick_to_sec(mid):
    ppq = mid.ticks_per_beat
    events = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == 'set_tempo':
                events.append((t, msg.tempo))
    events.sort(key=lambda e: e[0])
    if not events or events[0][0] != 0:
        events.insert(0, (0, 500000))
    ticks = [e[0] for e in events]
    tempos = [e[1] for e in events]
    cum = [0.0]
    for i in range(1, len(events)):
        cum.append(cum[-1] + (ticks[i] - ticks[i - 1]) * tempos[i - 1] / ppq * 1e-6)

    def f(tick):
        i = bisect.bisect_right(ticks, tick) - 1
        return cum[i] + (tick - ticks[i]) * tempos[i] / ppq * 1e-6
    return f


# ---------------------------------------------------------------------------
# seconds -> beat position -> output tick
# ---------------------------------------------------------------------------

class BeatMap:
    def __init__(self, beat_times, pad, ppq):
        self.t = np.asarray(beat_times, dtype=float)
        if len(self.t) < 2 or np.any(np.diff(self.t) <= 0):
            raise ValueError('beat times must be strictly increasing, >= 2 of them')
        self.pad = int(pad)
        self.ppq = int(ppq)
        self.n = len(self.t)
        self.dt_first = float(self.t[1] - self.t[0])
        self.dt_last = float(self.t[-1] - self.t[-2])

    def beat(self, sec):
        """Fractional beat position, linear between annotated beats,
        linear extrapolation outside."""
        if sec <= self.t[0]:
            return (sec - self.t[0]) / self.dt_first
        if sec >= self.t[-1]:
            return (self.n - 1) + (sec - self.t[-1]) / self.dt_last
        return float(np.interp(sec, self.t, np.arange(self.n)))

    def tick(self, sec):
        return max(0, int(round((self.beat(sec) + self.pad) * self.ppq)))

    def tempo_track(self):
        """One tempo per beat of ticks: pad beats at the first interval,
        then the annotated intervals, so midi time tracks audio time."""
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        segs = [self.dt_first] * self.pad + [float(d) for d in np.diff(self.t)]
        prev = 0
        for i, dt in enumerate(segs):
            tick = i * self.ppq
            tr.append(mido.MetaMessage('set_tempo', tempo=int(round(max(dt, 1e-3) * 1e6)),
                                       time=tick - prev))
            prev = tick
        tr.append(mido.MetaMessage('end_of_track', time=0))
        return tr


def first_downbeat_index(rows):
    if rows.ndim < 2 or rows.shape[1] < 2:
        raise ValueError('beat_midi.txt has no downbeat column')
    idx = np.where(rows[:, -1] >= 0.5)[0]
    if not len(idx):
        raise ValueError('no downbeat flag set')
    return int(idx[0]), idx


# ---------------------------------------------------------------------------
# tracks
# ---------------------------------------------------------------------------

def retime_track(track, raw_to_sec, bm):
    """Copy a raw track with every event re-timed through the beat map;
    tempo / time-signature metas are dropped (the new tempo track owns them)."""
    events = []
    t = 0
    for msg in track:
        t += msg.time
        if msg.type in ('set_tempo', 'time_signature', 'end_of_track'):
            continue
        events.append((bm.tick(raw_to_sec(t)), msg))
    # stable sort by tick; note-offs before note-ons on ties so re-attacks survive
    events.sort(key=lambda e: (e[0], 0 if (e[1].type == 'note_off' or
                                           (e[1].type == 'note_on' and e[1].velocity == 0)) else 1))
    out = mido.MidiTrack()
    prev = 0
    for tick, msg in events:
        out.append(msg.copy(time=tick - prev))
        prev = tick
    out.append(mido.MetaMessage('end_of_track', time=0))
    return out


def chord_track(chord_txt, bm):
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage('track_name', name='CHORD', time=0))
    events = []
    n_rows = 0
    with open(chord_txt) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            n_rows += 1
            t0, t1, label = float(parts[0]), float(parts[1]), parts[2]
            parsed = parse_chord(label)
            if parsed is None:
                continue
            root, intervals, bass = parsed
            on, off = bm.tick(t0), bm.tick(t1)
            off = max(on + 1, off)
            for p in chord_to_midi_pitches(root, intervals, bass):
                events.append((on, 1, p))
                events.append((off, 0, p))
    events.sort(key=lambda e: (e[0], e[1]))        # offs (0) before ons (1) on ties
    prev = 0
    for tick, kind, p in events:
        if kind:
            tr.append(mido.Message('note_on', note=p, velocity=80, time=tick - prev))
        else:
            tr.append(mido.Message('note_off', note=p, velocity=0, time=tick - prev))
        prev = tick
    tr.append(mido.MetaMessage('end_of_track', time=0))
    return tr, n_rows


def melody_tracks(mid):
    return [t for t in mid.tracks if (t.name or '').strip().lower() == 'melody']


# ---------------------------------------------------------------------------
# one song
# ---------------------------------------------------------------------------

def align_song(song_dir, dst, no_pad=False):
    sid = os.path.basename(song_dir)
    mid = mido.MidiFile(os.path.join(song_dir, f'{sid}.mid'))
    rows = np.loadtxt(os.path.join(song_dir, 'beat_midi.txt'))
    if rows.ndim == 1:
        rows = rows[None, :]
    d0, downbeats = first_downbeat_index(rows)
    pad = 0 if no_pad else (-d0) % 4
    ppq = mid.ticks_per_beat
    bm = BeatMap(rows[:, 0], pad, ppq)
    raw_to_sec = raw_tick_to_sec(mid)

    # verify the grid before writing anything
    db_tick = bm.tick(float(rows[d0, 0]))
    if not no_pad and db_tick % (4 * ppq) != 0:
        raise AssertionError(f'first downbeat at tick {db_tick}, not on the 4-beat grid')
    gaps = np.diff(downbeats)
    irregular = [int(i) for i, g in enumerate(gaps) if g != 4]

    tempo = bm.tempo_track()
    mel = melody_tracks(mid)
    if not mel:
        raise ValueError(f'no MELODY track (tracks: {[t.name for t in mid.tracks]})')

    full = mido.MidiFile(ticks_per_beat=ppq)
    full.tracks.append(tempo)
    for tr in mid.tracks:
        rt = retime_track(tr, raw_to_sec, bm)
        if len(rt) > 1:
            full.tracks.append(rt)

    melody = mido.MidiFile(ticks_per_beat=ppq)
    melody.tracks.append(tempo)
    for tr in mel:
        melody.tracks.append(retime_track(tr, raw_to_sec, bm))

    ctr, n_rows = chord_track(os.path.join(song_dir, 'chord_midi.txt'), bm)
    chord = mido.MidiFile(ticks_per_beat=ppq)
    chord.tracks.append(tempo)
    chord.tracks.append(ctr)

    for sub, m in (('aligned', full), ('melody', melody), ('chord', chord)):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        m.save(os.path.join(dst, sub, f'{sid}.mid'))

    return dict(id=sid, ppq=ppq, n_beats=len(rows), d0=d0, pad=pad,
                lead_in_s=f'{rows[0, 0]:.3f}', n_bars=len(gaps),
                n_irregular=len(irregular),
                first_irregular_bar=(irregular[0] if irregular else ''),
                chord_rows=n_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True, help='POP909 folder of the dataset repo (song subfolders)')
    p.add_argument('--dst', required=True, help='output root: aligned/ melody/ chord/ align_report.tsv')
    p.add_argument('--ids', nargs='*', default=None)
    p.add_argument('--no-pad', action='store_true', help='beat 0 at tick 0 (downbeats not put on the bar grid)')
    a = p.parse_args()

    dirs = sorted(d for d in glob(os.path.join(a.raw, '*')) if os.path.isdir(d))
    if a.ids:
        want = set(a.ids)
        dirs = [d for d in dirs if os.path.basename(d) in want]
    os.makedirs(a.dst, exist_ok=True)
    rows, failed = [], []
    for i, d in enumerate(dirs):
        try:
            rows.append(align_song(d, a.dst, no_pad=a.no_pad))
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((os.path.basename(d), repr(e)))
        if (i + 1) % 100 == 0:
            print(f'[{i + 1}/{len(dirs)}] processed')
    if rows:
        with open(os.path.join(a.dst, 'align_report.tsv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
            w.writeheader()
            w.writerows(rows)
    n_reg = sum(1 for r in rows if r['n_irregular'] == 0)
    print(f'Done. {len(rows)} aligned, {len(failed)} failed -> {a.dst}')
    print(f'  pad distribution: { {k: sum(1 for r in rows if r["pad"] == k) for k in range(4)} }')
    print(f'  songs with every bar 4 beats: {n_reg}; with an irregular bar: {len(rows) - n_reg}')
    for sid, err in failed[:20]:
        print(f'  FAILED {sid}: {err}')


if __name__ == '__main__':
    main()
