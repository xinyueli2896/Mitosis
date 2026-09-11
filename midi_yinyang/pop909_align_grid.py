"""POP909 -> metrically exact MIDI: seconds -> fractional beats -> grid.

beat_midi.txt is the only yardstick. It is a lookup table from wall-clock
seconds to musical time: row i says "beat i happens at t_i seconds".
Column 1 is that time, column 2 flags the half-bar beats, column 3 flags
the downbeat. One row is one BEAT (median tempo across the corpus reads
74 BPM that way; reading a row as an eighth would give 37, which is not
a pop corpus).

What this does, and how it differs from pop909_beat_align.py:

  1. seconds -> fractional beats, by linear interpolation between
     neighbouring beat times. Onset and offset are mapped INDEPENDENTLY,
     never onset + duration, because the tempo can change mid-note.
     Outside the annotated range the first / last inter-beat interval is
     extrapolated -- np.interp clamps there, which would pile every
     lead-in note onto beat 0.
  2. SNAP to a subdivision grid (--sub, default 4 = 16ths; POP909's own
     processed release uses 24 so triplets survive). This is the step
     pop909_beat_align.py never did: it rounded to 1/480 of a beat,
     which is not quantization at all, so off-grid performance timing
     survived into the tokenizer and got snapped there instead.
  3. origin -- where the bar grid starts. POP909 songs often begin with
     a PICKUP, so the first downbeat flag in column 3 sits at index 1, 2
     or 3, not 0: only 308 of 909 songs start on a downbeat.

       bar       (DEFAULT) shift everything by (-first_down) mod 4 beats
                 so the first downbeat lands on a BAR LINE and the
                 pickup occupies the tail of an incomplete bar 1 -- what
                 a score does with an anacrusis. Nothing is dropped.
                 The shift is ONE CONSTANT applied to every event: no
                 time is inserted between notes, inter-onset distances
                 are untouched, the file simply starts with up to 3
                 empty beats.
       downbeat  shift so the first downbeat is bar 1 beat 1. Same bar
                 phase as `bar`, no leading empty beats, but the pickup
                 lands before 0 and is dropped and counted.
       beat0     annotation beat 0 stays at tick 0. Keeps the pickup and
                 adds nothing, but the bar grid is then wrong in the 601
                 songs that do not start on a downbeat: 286 of them put
                 beat 0 at bar position 3, so a DAW shows bar 1 beat 1
                 where the music has beat 4 of the pickup bar. Kept only
                 to reproduce earlier output.
  4. write at CONSTANT tempo (--tempo, default 120, so a beat is exactly
     0.5 s). The hand-labelled tempo curve is gone and the note grid is
     metrically exact. pop909_beat_align.py instead wrote one tempo per
     beat to preserve audio time, which keeps the curve alive.

Dropped beats. If the tracker missed a beat, one gap in column 1 is
about double its neighbours and every note inside it lands a beat late.
--fix-dropped (default on) inserts a synthetic beat at the midpoint of
any gap within --drop-tol of an exact multiple of the local median, and
the report counts them. Note this RENUMBERS every later beat, which is
the point -- musical time is restored -- but it also moves bar phase,
so the count is worth looking at before trusting a song.

Meter. Column 3 should be every fourth beat. Corpus-wide it is exactly
that in only 126 of 909 songs (gaps of 2, 3, 5, 6 and 8 all occur), so
the report carries n_irregular and frac_downbeats_on_bar and sections
that change meter are left for the caller to handle.

Outputs under <dst>/:  aligned/<id>.mid  melody/<id>.mid
                       chord/<id>.mid    align_grid_report.tsv

Usage (via align_pop909_grid.sbatch, CPU):
    python pop909_align_grid.py --raw <POP909 dir> --dst <out> --sub 4
"""
import argparse
import csv
import os
from glob import glob

import numpy as np
import pretty_midi as pm

from build_pop909_chord_midi import chord_to_midi_pitches, parse_chord

# Resolution must divide by the subdivision or the grid we just snapped
# to is re-rounded on write. 480 covers 4, 8, 12, 16 and 24 per beat.
RESOLUTION = 480


def load_beats(path, fix_dropped=True, tol=0.15):
    """Column 1 of beat_midi.txt, optionally with dropped beats restored.

    Returns (beat_times, downbeat_mask, n_inserted).
    """
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] < 3:
        raise ValueError(f'{path}: expected 3 columns, got {rows.shape[1]}')
    bt = rows[:, 0].astype(float)
    down = rows[:, 2] >= 0.5
    if len(bt) < 2:
        raise ValueError(f'{path}: needs at least 2 beats')
    if np.any(np.diff(bt) <= 0):
        raise ValueError(f'{path}: beat times are not strictly increasing')
    if not fix_dropped:
        return bt, down, 0

    d = np.diff(bt)
    med = float(np.median(d))
    out_t, out_d, n_ins = [bt[0]], [down[0]], 0
    for gap, t_next, d_next in zip(d, bt[1:], down[1:]):
        k = int(round(gap / med))
        # k>=2 and the gap within tol of exactly k median beats: the
        # tracker dropped k-1 beats here. Space them evenly.
        if k >= 2 and abs(gap - k * med) <= tol * med:
            for j in range(1, k):
                out_t.append(out_t[-1] + gap / k)
                out_d.append(False)     # a restored beat is not a downbeat
                n_ins += 1
        out_t.append(t_next)
        out_d.append(d_next)
    return np.asarray(out_t), np.asarray(out_d), n_ins


class GridMap:
    """seconds -> fractional beats -> snapped beats."""

    def __init__(self, beat_times, sub, origin_beat=0.0):
        self.bt = np.asarray(beat_times, dtype=float)
        self.idx = np.arange(len(self.bt), dtype=float)
        self.sub = int(sub)
        self.origin = float(origin_beat)
        self.d_first = float(self.bt[1] - self.bt[0])
        self.d_last = float(self.bt[-1] - self.bt[-2])

    def raw_beat(self, x):
        """Fractional beat, EXTRAPOLATED past the ends rather than
        clamped (np.interp clamps, which stacks lead-in notes on beat 0)."""
        x = np.asarray(x, dtype=float)
        b = np.interp(x, self.bt, self.idx)
        lo, hi = x < self.bt[0], x > self.bt[-1]
        if np.any(lo):
            b = np.where(lo, (x - self.bt[0]) / self.d_first, b)
        if np.any(hi):
            b = np.where(
                hi, self.idx[-1] + (x - self.bt[-1]) / self.d_last, b)
        return b

    def __call__(self, x):
        b = self.raw_beat(x) - self.origin
        return np.round(b * self.sub) / self.sub


def map_notes(notes, g, sec_per_beat):
    """Snap a list of (start, end, pitch, velocity) onto the grid.

    Onset and offset go through the map SEPARATELY. A note that rounds
    to zero length is given one grid unit. Anything landing before beat
    0 is dropped and counted (only reachable with --origin downbeat).
    """
    if not notes:
        return [], 0
    starts = g(np.array([n[0] for n in notes]))
    ends = g(np.array([n[1] for n in notes]))
    step = 1.0 / g.sub
    out, dropped = [], 0
    for (s, e, pitch, vel) in zip(starts, ends,
                                  [n[2] for n in notes],
                                  [n[3] for n in notes]):
        if e <= s:
            e = s + step
        if s < 0:
            dropped += 1
            continue
        out.append(pm.Note(velocity=int(vel), pitch=int(pitch),
                           start=float(s) * sec_per_beat,
                           end=float(e) * sec_per_beat))
    return out, dropped


def read_chords(path, g, sec_per_beat):
    notes, n_rows = [], 0
    with open(path) as fh:
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
            for p in chord_to_midi_pitches(root, intervals, bass):
                notes.append((t0, t1, p, 80))
    mapped, dropped = map_notes(notes, g, sec_per_beat)
    return mapped, n_rows, dropped


def snap_stats(onsets, g):
    """How far the melody actually moves, and whether a FINER grid holds it.

    The operative question is not "are these triplets" but "would 24 per
    beat have kept this note where it was, when 4 will not". A song on
    k/24 positions that are not multiples of 6 -- 1/12, 1/8, 1/6, 1/3 --
    is mangled by a 16th grid and exact on a 24 grid. Song 326 is one:
    median snap error 0.0617 beats at sub=4 and 0.0008 at sub=24, which
    is why its melody reads 73% "off-grid" under a 16th-note check while
    being perfectly regular music.

    Returns (median |err|, p95 |err|, fraction of onsets that sit on the
    24 grid but NOT the 4 grid).
    """
    if not len(onsets):
        return 0.0, 0.0, 0.0
    raw = g.raw_beat(np.asarray(onsets))
    err = np.abs(raw - np.round(raw * g.sub) / g.sub)
    frac = raw - np.floor(raw)
    tol = 1.0 / 96.0                       # half a 48th of a beat
    d24 = np.abs(frac[:, None] - np.arange(24) / 24.0).min(axis=1)
    d4 = np.abs(frac[:, None] - np.arange(4) / 4.0).min(axis=1)
    return (float(np.median(err)), float(np.percentile(err, 95)),
            float(np.mean((d24 < tol) & (d4 >= tol))))


def align_song(song_dir, dst, sub, tempo, origin, fix_dropped, drop_tol):
    sid = os.path.basename(song_dir)
    bt, down, n_ins = load_beats(os.path.join(song_dir, 'beat_midi.txt'),
                                 fix_dropped=fix_dropped, tol=drop_tol)
    db_idx = np.where(down)[0]
    if not len(db_idx):
        raise ValueError('no downbeat flag in column 3')
    first_down = int(db_idx[0])
    if origin == 'downbeat':
        origin_beat = float(first_down)
    elif origin == 'bar':
        # negative origin == a positive shift: raw_beat - origin moves
        # everything right by pad, so the first downbeat lands on a
        # multiple of 4 and the pickup fills the bar before it.
        origin_beat = -float((-first_down) % 4)
    else:
        origin_beat = 0.0

    g = GridMap(bt, sub, origin_beat)
    spb = 60.0 / float(tempo)

    src = pm.PrettyMIDI(os.path.join(song_dir, f'{sid}.mid'))
    out = pm.PrettyMIDI(resolution=RESOLUTION, initial_tempo=float(tempo))
    mel_out = pm.PrettyMIDI(resolution=RESOLUTION, initial_tempo=float(tempo))
    n_notes = n_drop = 0
    have_mel = False
    for inst in src.instruments:
        raw = [(n.start, n.end, n.pitch, n.velocity) for n in inst.notes]
        mapped, dropped = map_notes(raw, g, spb)
        n_notes += len(mapped)
        n_drop += dropped
        new = pm.Instrument(program=inst.program, is_drum=inst.is_drum,
                            name=inst.name)
        new.notes = mapped
        out.instruments.append(new)
        if (inst.name or '').strip().lower() == 'melody':
            m = pm.Instrument(program=inst.program, is_drum=inst.is_drum,
                              name=inst.name)
            m.notes = list(mapped)
            mel_out.instruments.append(m)
            have_mel = True
    if not have_mel:
        raise ValueError(
            f'no MELODY instrument (names: {[i.name for i in src.instruments]})')

    cnotes, n_rows, c_drop = read_chords(
        os.path.join(song_dir, 'chord_midi.txt'), g, spb)
    chd_out = pm.PrettyMIDI(resolution=RESOLUTION, initial_tempo=float(tempo))
    ci = pm.Instrument(program=48, is_drum=False, name='CHORD')
    ci.notes = cnotes
    chd_out.instruments.append(ci)

    for sub_dir, m in (('aligned', out), ('melody', mel_out),
                       ('chord', chd_out)):
        os.makedirs(os.path.join(dst, sub_dir), exist_ok=True)
        m.write(os.path.join(dst, sub_dir, f'{sid}.mid'))

    mel_on = [n.start for i in src.instruments
              for n in i.notes if (i.name or '').strip().lower() == 'melody']
    snap_med, snap_p95, trip = snap_stats(mel_on, g)

    gaps = np.diff(db_idx)
    irregular = [int(i) for i, gp in enumerate(gaps) if gp != 4]
    phase = (g.raw_beat(bt[db_idx]) - origin_beat) % 4
    on_bar = float(np.mean(np.minimum(phase, 4.0 - phase) < 1e-6))
    return dict(id=sid, n_beats=len(bt), inserted_beats=n_ins,
                first_downbeat=first_down, origin=origin, sub=sub,
                tempo=tempo, n_notes=n_notes, notes_dropped=n_drop,
                chord_rows=n_rows, chord_notes_dropped=c_drop,
                n_bars=len(gaps), n_irregular=len(irregular),
                first_irregular_bar=(irregular[0] if irregular else ''),
                n_downbeats=len(db_idx),
                frac_downbeats_on_bar=f'{on_bar:.4f}',
                mel_snap_median=f'{snap_med:.5f}',
                mel_snap_p95=f'{snap_p95:.5f}',
                mel_needs_sub24_frac=f'{trip:.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--ids', nargs='*', default=None)
    p.add_argument('--sub', type=int, default=4,
                   help='grid subdivisions per beat: 4 = 16ths (default), '
                        '24 = POP909\'s own release, keeps triplets')
    p.add_argument('--tempo', type=float, default=120.0,
                   help='constant output tempo; 120 makes a beat 0.5 s')
    p.add_argument('--origin', choices=('bar', 'downbeat', 'beat0'),
                   default='bar',
                   help='bar (default): shift by (-first_down) mod 4 so the '
                        'first downbeat is on a BAR LINE and the pickup '
                        'fills an incomplete first bar -- nothing dropped, '
                        'and the shift is one constant so no time is '
                        'inserted between notes. downbeat: first downbeat at '
                        'bar 1 beat 1, pickup dropped. beat0: annotation '
                        'beat 0 at tick 0 -- wrong bar phase in the 601 '
                        'songs that do not start on a downbeat.')
    p.add_argument('--no-fix-dropped', action='store_true',
                   help='do NOT restore beats the tracker missed')
    p.add_argument('--drop-tol', type=float, default=0.15,
                   help='a gap counts as k dropped beats when it is within '
                        'this fraction of k x the local median')
    a = p.parse_args()

    if RESOLUTION % a.sub:
        raise SystemExit(f'--sub {a.sub} must divide the {RESOLUTION} '
                         f'tick resolution, or the snap is re-rounded')

    dirs = sorted(d for d in glob(os.path.join(a.raw, '*')) if os.path.isdir(d))
    if a.ids:
        want = set(a.ids)
        dirs = [d for d in dirs if os.path.basename(d) in want]
    os.makedirs(a.dst, exist_ok=True)
    rows, failed = [], []
    for i, d in enumerate(dirs):
        try:
            rows.append(align_song(d, a.dst, a.sub, a.tempo, a.origin,
                                   not a.no_fix_dropped, a.drop_tol))
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((os.path.basename(d), repr(e)))
        if (i + 1) % 100 == 0:
            print(f'[{i + 1}/{len(dirs)}] processed', flush=True)
    if rows:
        with open(os.path.join(a.dst, 'align_grid_report.tsv'), 'w',
                  newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                               delimiter='\t')
            w.writeheader()
            w.writerows(rows)
    print(f'Done. {len(rows)} aligned, {len(failed)} failed -> {a.dst}')
    if rows:
        ins = np.array([r['inserted_beats'] for r in rows])
        dr = np.array([r['notes_dropped'] for r in rows])
        onb = np.array([float(r['frac_downbeats_on_bar']) for r in rows])
        print(f'  grid: {a.sub}/beat at {a.tempo} BPM, origin={a.origin}')
        print(f'  dropped beats restored: {int(ins.sum())} across '
              f'{int((ins > 0).sum())} songs (max {int(ins.max())} in one)')
        print(f'  notes dropped before beat 0: {int(dr.sum())}')
        print(f'  downbeats on the 4-beat grid: mean {onb.mean():.3f}; '
              f'{int((onb >= 1.0).sum())} songs at 100%')
        p95 = np.array([float(r['mel_snap_p95']) for r in rows])
        trp = np.array([float(r['mel_needs_sub24_frac']) for r in rows])
        print(f'  melody snap error (beats): median of p95 = '
              f'{np.median(p95):.4f}; {int((p95 > 0.02).sum())} songs move '
              f'a note by > 1/50 beat')
        print(f'  songs needing a FINER grid (>10% of melody onsets on '
              f'the 24 grid but off the 4 grid): {int((trp > 0.10).sum())}'
              f'  -- a 4/beat grid mangles these; --sub 24 or exclude')
    for sid, err in failed[:20]:
        print(f'  FAILED {sid}: {err}')


if __name__ == '__main__':
    main()
