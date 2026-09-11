"""Held-out inference set: crop every song so the prompt is the SAME
musical content everywhere.

The old E1 selection asked two things of a song: that it be in
whole-song-gen's split, and that both streams happen to be sounding in
the first 6 or 8 bars. The second is a property of where the song's
intro ends, so it threw away most of the corpus and still left the
prompts holding different amounts of music.

This inverts it. The prompt is CUT TO the melody instead of hoping the
melody arrives in time, and the anchor is the FIRST SOUNDING MELODY NOTE
-- not a search for a convenient window. Let F be the bar holding that
note:

    F is a pickup bar      melody enters partway through F, so the crop
    (anacrusis)            begins ONE bar in front of it:
                               bar 1     empty lead
                               bar 2     the pickup, F
                               bars 3..  --mel-bars sounding bars

    F is not a pickup      melody begins on F's bar line, so the crop
                           begins TWO bars in front of it:
                               bars 1-2  empty lead
                               bars 3..  --mel-bars sounding bars

Either way the prompt is --mel-bars + 2 bars, and the one invariant
every prompt shares is that THE MELODY'S FIRST FULL DOWNBEAT IS AT
FRAME 32 (bar 2). Whatever the song has in front of that is kept, and
where it has less than two bars' worth the difference is PADDED with
silence -- to the frame, not to the bar, so a pickup the aligner left
as a partial bar still ends exactly on its downbeat. The pickup bar in
the song stays a pickup bar in the prompt. Chord is cropped and padded
on exactly the same tick, so the two streams stay in register by
construction. Whether F is a pickup bar comes from beat_midi.txt: the
first note is on an annotated downbeat, or it is not.

Bars come from the file's OWN time signatures, so a song with a 6/4 bar
is measured in its real bars rather than a fixed four beats. Songs whose
metre changes inside the prompt or the scored window are reported
(metre_changes_in_prompt / _in_window) and, with --require-regular,
dropped: a metre change inside the window shifts the bar grid under
every bar-based metric.

Songs are restricted to a HELD-OUT split, read the same way
audit_wholesong_split.py reads it -- index %% split-ratio == 0 over the
dataset's own .txt / .length.pt, which is never trained on under either
of the two split conventions in this repo.

Outputs under <dst>/:  melody/<id>.mid  chord/<id>.mid
                       prompt_crops.tsv  (one row per candidate)

Usage (via build_prompt_crops.sbatch, CPU):
    python build_prompt_crops.py --mel-src <v5>/melody \\
        --chord-src <v5>/chord --dst input/heldout_v5 \\
        --dataset pop909_melody_cp8_v2
"""
import argparse
import collections
import csv
import os
import re
import warnings
from glob import glob

import mido
import numpy as np
import pretty_midi as pm

warnings.filterwarnings('ignore')
RESOLUTION = 480

# a note in TICKS -- see the block in main() on why nothing is measured
# in seconds
Note = collections.namedtuple('Note', 'start end pitch')


def song_id(name):
    m = re.search(r'(\d{3})', os.path.basename(str(name)))
    return m.group(1) if m else None


# FramedDataset target length. audit_wholesong_split filters to songs at
# least this long BEFORE taking the modulo, so matching it matters: a
# shorter song is not in the dataset at all and is neither train nor val.
TRAIN_LENGTH = 384


def heldout_ids(stem, split_ratio=10, train_length=TRAIN_LENGTH):
    """Song ids never trained on, by the FramedDataset rule.

    Mirrors audit_wholesong_split.our_val: a song is held out iff its
    index in the dataset is 0 mod split_ratio. Both split conventions in
    this repo (train = idx%10 != 0, and train = idx%10 > 1) leave
    idx%10 == 0 untrained, so this is the safe set under either.
    """
    import torch
    txt, lpt = f'data/{stem}.txt', f'data/{stem}.length.pt'
    if not (os.path.exists(txt) and os.path.exists(lpt)):
        raise SystemExit(f'missing {txt} or {lpt} -- build the dataset first, '
                         f'or pass --ids to skip the split filter')
    names = {}
    with open(txt) as fh:
        for line in fh:
            i, nm = line.rstrip('\n').split('\t', 1)
            names[int(i)] = nm
    lengths = torch.load(lpt)
    keep = (np.asarray(lengths) >= train_length) if train_length else \
        np.ones(len(lengths), dtype=bool)
    idx = np.arange(len(lengths))[keep]
    val = [i for i in idx if i % split_ratio == 0]
    trn = [i for i in idx if i % split_ratio != 0]
    ids = {song_id(names[i]) for i in val if song_id(names[i])}
    train_ids = {song_id(names[i]) for i in trn if song_id(names[i])}
    print(f'[split] {stem}: {len(lengths)} songs, {int(keep.sum())} at least '
          f'{train_length} frames, {len(val)} held out (idx %% {split_ratio} '
          f'== 0) -> {len(ids)} distinct song ids')
    return ids, train_ids


def bar_starts(mid, end_time):
    """Bar-start times in seconds, from the file's OWN metre and tempo map.

    Uses pretty_midi's get_downbeats(), which walks the full tempo map
    and the time-signature changes together. The hand-rolled walk this
    replaces took bpm from the FIRST tempo event and stepped by a
    constant -- fine for a file the grid aligner wrote (one tempo), and
    catastrophic for one carrying a lead-in artefact: song 004's v5
    melody opens with an absurd tempo, so every "bar" spanned a sliver
    of a second, the walk hit its 10000-bar cap before the first note,
    and EVERY bar test came back false. That is what made 004 impossible
    to keep even with --force-ids.

    Falls back to the constant-tempo walk only if get_downbeats returns
    nothing, and reports a degenerate grid rather than silently handing
    back 10000 meaningless bars.
    """
    ts = sorted(mid.time_signature_changes, key=lambda t: t.time) or \
        [pm.TimeSignature(4, 4, 0.0)]
    times, tempi = mid.get_tempo_changes()
    bpm = float(np.median(tempi)) if len(tempi) else 120.0
    spb = 60.0 / max(bpm, 1e-6)

    try:
        db = np.asarray(mid.get_downbeats(), dtype=float)
    except Exception:                                   # noqa: BLE001
        db = np.zeros(0)
    db = db[np.isfinite(db)]
    if len(db) >= 2:
        step = float(np.median(np.diff(db)))
        # extend past the last downbeat so a note in the final bar is
        # still inside a bar
        while len(db) and db[-1] <= end_time and step > 1e-6:
            db = np.append(db, db[-1] + step)
            if len(db) > 20000:
                break
        return db, ts, spb

    out, t, i = [], 0.0, 0
    while t <= end_time + 1e-6 and len(out) < 10000:
        while i + 1 < len(ts) and ts[i + 1].time <= t + 1e-9:
            i += 1
        out.append(t)
        t += ts[i].numerator * spb * (4.0 / ts[i].denominator)
    return np.array(out), ts, spb


def degenerate_grid(bars, end_time):
    """Is this bar grid unusable? A grid that covers a tiny fraction of
    the music, or has almost no bars, means the tempo map is broken --
    report that instead of 'no bar-aligned start', which blames the
    melody for a timing artefact."""
    if len(bars) < 2:
        return 'bar grid has < 2 bars'
    if bars[-1] < 0.5 * end_time:
        return (f'bar grid covers only {bars[-1]:.1f}s of {end_time:.1f}s '
                f'({len(bars)} bars) -- broken tempo map')
    return ''


def sounding_bars(inst_notes, bars):
    """Per bar, is the stream ACTIVE in it -- any note overlapping the bar,
    not merely starting in it.

    Onsets are the wrong test for the chord stream. A chord held across
    two bars has an onset only in the first, so a song with slow harmonic
    rhythm shows chord "activity" in every other bar and four consecutive
    both-stream bars never occur, even though the chord sounds without a
    gap. Measuring overlap asks the question actually meant: is there
    harmony under the melody here.
    """
    if not inst_notes:
        return np.zeros(len(bars), dtype=bool)
    on = np.array([n.start for n in inst_notes])
    off = np.array([n.end for n in inst_notes])
    ends = np.append(bars[1:], bars[-1] + (bars[-1] - bars[-2])
                     if len(bars) > 1 else bars[-1] + 1.0)
    hit = np.zeros(len(bars), dtype=bool)
    for b, (t0, t1) in enumerate(zip(bars, ends)):
        hit[b] = bool(np.any((on < t1 - 1e-9) & (off > t0 + 1e-9)))
    return hit


def complete_bars(inst_notes, bars, tol):
    """Per bar, does the stream start AT the bar line -- i.e. is this a
    bar the melody 'officially' begins on, rather than one it wanders
    into partway through?

    This is the distinction that makes the pickup bar work. A song with
    an anacrusis has melody onsets in the bar BEFORE its first real
    downbeat; treating any onset as 'sounding' would make that anacrusis
    the first sounding bar and leave the buffer empty, which is the
    opposite of what the buffer is for. A bar qualifies here only if a
    note begins within `tol` of its start, or a note is sustaining
    across it.
    """
    out = np.zeros(len(bars), dtype=bool)
    if not inst_notes:
        return out
    on = np.array([n.start for n in inst_notes])
    off = np.array([n.end for n in inst_notes])
    for b, t in enumerate(bars):
        if np.any((on >= t - tol) & (on <= t + tol)):
            out[b] = True
        elif np.any((on < t) & (off > t + tol)):
            out[b] = True
    return out


def crop_ticks(src_path, dst_path, tick0, pad_ticks=0, force_bpm=None):
    """Window a MIDI file in TICK space, changing nothing else.

    Cropping must be a pure windowing operation: every tempo and
    time-signature event survives, and a surviving note keeps the exact
    tick offsets it had relative to them. The previous version rebuilt
    the file through pretty_midi with ONE tempo taken from tempi[0] while
    keeping note times in SECONDS -- so any file carrying a tempo map came
    out with different tick positions, i.e. a different notated rhythm,
    and a different tempo per song depending on what its first event
    happened to be. That is editing the source, not cropping it.

    pad_ticks prepends that many EMPTY ticks, used when the song does not
    have enough bars in front of its first melody note and the missing
    ones have to be manufactured. It is a pure shift of everything that
    survives -- no note is created, and the relative timing inside the
    window is untouched.

    Notes are kept when their ONSET is at or after tick0 and shifted by
    -tick0 + pad_ticks; a note straddling the cut is dropped rather than truncated,
    so nothing appears from nowhere at bar 1 beat 1. Meta and control
    events before the cut are re-emitted at tick 0, because they set
    state the window needs (tempo, metre, program, key).

    force_bpm replaces the tempo map with a single set_tempo at tick 0,
    so every file in the set plays at one constant tempo. It touches no
    note: every event keeps the exact tick it had, because a tick is a
    musical position and the tempo map only says how fast those ticks go
    by. The aligner writes a fixed-tempo grid where a tick IS a fixed
    fraction of a beat, so this makes the rendered seconds agree with
    the notation -- and it means a stray tempo map in the source cannot
    reach the prompt set.
    """
    src = mido.MidiFile(src_path)
    out = mido.MidiFile(ticks_per_beat=src.ticks_per_beat, type=src.type)
    for ti, tr in enumerate(src.tracks):
        events, open_on, t = [], {}, 0
        for msg in tr:
            t += msg.time
            if msg.type == 'end_of_track':
                continue
            is_off = (msg.type == 'note_off'
                      or (msg.type == 'note_on' and msg.velocity == 0))
            if msg.type == 'note_on' and not is_off:
                open_on.setdefault((msg.channel, msg.note), []).append((t, msg))
            elif is_off:
                stack = open_on.get((msg.channel, msg.note))
                if not stack:
                    continue
                on_t, on_msg = stack.pop(0)
                if on_t < tick0:
                    continue
                events.append((on_t - tick0 + pad_ticks, 1, on_msg))
                events.append((max(t - tick0, on_t - tick0 + 1) + pad_ticks,
                               0, msg))
            elif not (force_bpm and msg.type == 'set_tempo'):
                # State set before the cut belongs at tick 0, NOT at
                # pad_ticks: a padded lead bar would otherwise play under
                # whatever the format defaults to until the first real
                # event, since the tempo and metre that govern it were
                # pushed past it.
                events.append((0 if t <= tick0 else t - tick0 + pad_ticks,
                               2, msg))
        for stack in open_on.values():                 # never closed
            for on_t, on_msg in stack:
                if on_t >= tick0:
                    events.append((on_t - tick0 + pad_ticks, 1, on_msg))
        events.sort(key=lambda e: (e[0], -e[1]))
        new_tr = mido.MidiTrack()
        if force_bpm and ti == 0:
            new_tr.append(mido.MetaMessage(
                'set_tempo', tempo=mido.bpm2tempo(float(force_bpm)), time=0))
        prev = 0
        for tick, _, msg in events:
            new_tr.append(msg.copy(time=tick - prev))
            prev = tick
        new_tr.append(mido.MetaMessage('end_of_track', time=0))
        out.tracks.append(new_tr)
    out.save(dst_path)


def tempo_summary(mid):
    """(n tempo events, first bpm, median bpm) -- so a source that is not
    the constant tempo you expect is visible per song rather than
    surfacing later as 'why are the tempos all different'."""
    _, tempi = mid.get_tempo_changes()
    if not len(tempi):
        return 0, 120.0, 120.0
    return len(tempi), float(tempi[0]), float(np.median(tempi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mel-src', required=True)
    p.add_argument('--chord-src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--dataset', default='pop909_melody_cp8_v2',
                   help='dataset stem whose index %% split-ratio == 0 songs '
                        'are the held-out set (see audit_wholesong_split)')
    p.add_argument('--split-ratio', type=int, default=10)
    p.add_argument('--align-report', default=None,
                   help="align_grid_report.tsv from the aligner. Supplies "
                        "each song's IRREGULAR BAR positions, which the midi "
                        "itself cannot carry once time signatures are off "
                        "(a flat 4/4 file has no record of them). Given it, "
                        "the crop point is chosen to put the scored window "
                        "in a metrically REGULAR stretch where one exists -- "
                        "so a song with a metre change elsewhere is still "
                        "usable, instead of being dropped or silently "
                        "scored across a bar-grid shift.")
    p.add_argument('--force-ids', nargs='*', default=[],
                   help='ids to keep even when a drop rule rejects them. '
                        'The crop point itself never needs forcing any '
                        'more -- it is the first sounding melody note, '
                        'which every non-empty song has -- so this now '
                        'only waives MIN_BARS and --require-regular. The '
                        'TSV records it in `forced`, so a forced song is '
                        'never mistaken for one that passed; use it when '
                        'you have LISTENED to the song.')
    p.add_argument('--extra-ids', nargs='*',
                   default=['001', '002', '003', '004', '005'],
                   help='ids to ADD to the held-out set. Default is the five '
                        'songs cut before preprocessing: they are absent '
                        'from the dataset entirely, so the index rule cannot '
                        'find them, but they were never trained on and are '
                        'therefore valid. Refused if any turns out to be in '
                        'the TRAIN split -- that would be contamination, not '
                        'a bonus song. Pass --extra-ids with no values for '
                        'none.')
    p.add_argument('--no-length-filter', action='store_true',
                   help='skip the lengths >= 384 filter. Off by default so '
                        'the split matches audit_wholesong_split exactly; a '
                        'shorter song is not in the dataset at all, so it is '
                        'untrained but is not what "held out" has meant here.')
    p.add_argument('--ids', nargs='*', default=None,
                   help='explicit ids; skips the held-out split filter')
    p.add_argument('--mel-bars', type=int, default=4,
                   help='sounding bars of melody the prompt carries after '
                        'the lead (the prompt is this + 2 bars: one empty '
                        'lead plus the pickup, or two empty lead bars '
                        'where the melody starts on a bar line)')
    p.add_argument('--head-beats', type=float, default=0.25,
                   help='how close to a bar line the first melody note must '
                        'begin for that bar to count as the melody '
                        'OFFICIALLY starting (in beats; 0.25 = a 16th). A '
                        'first note later than that makes its bar an '
                        'ANACRUSIS, and the crop then takes one bar in '
                        'front of it instead of two.')
    p.add_argument('--tempo', type=float, default=120.0,
                   help='tempo every written file gets, as a single '
                        'set_tempo at tick 0 (default 120). No note moves: '
                        'ticks are untouched, only the map that says how '
                        'fast they go by. 0 keeps the source tempo map.')
    p.add_argument('--min-bars', type=int, default=25,
                   help='bars that must remain from the crop point, so '
                        'there is room for the generated continuation')
    p.add_argument('--require-regular', action='store_true',
                   help='drop songs whose metre changes inside the scored '
                        'window (prompt + min-bars)')
    a = p.parse_args()

    # lead bar + pickup bar + the sounding bars
    prompt_bars = a.mel_bars + 2
    if a.ids:
        want = set(a.ids)
        print(f'[split] explicit --ids given ({len(want)}); the held-out '
              f'filter is NOT applied')
    else:
        want, train_ids = heldout_ids(
            a.dataset, a.split_ratio,
            train_length=None if a.no_length_filter else TRAIN_LENGTH)
        extra = {e for e in (a.extra_ids or []) if e}
        bad = extra & train_ids
        if bad:
            raise SystemExit(
                f'REFUSING: --extra-ids {sorted(bad)} are in the TRAIN split '
                f'of {a.dataset}. Adding them would put training songs in the '
                f'metrics. Drop them, or check you passed the dataset the '
                f'model actually trains on.')
        if extra:
            already = extra & want
            new = extra - want
            print(f'[split] + {len(new)} manually held-out id(s) {sorted(new)}'
                  + (f' ({sorted(already)} already in the split)'
                     if already else '')
                  + ' -- absent from the dataset, so never trained on')
            want = want | extra
    have = {song_id(f) for f in glob(os.path.join(a.mel_src, '*.mid'))}
    ids = sorted(i for i in want if i and i in have)
    missing = sorted(i for i in want if i and i not in have)
    print(f'held-out candidates: {len(want)}; present in {a.mel_src}: '
          f'{len(ids)}')
    if missing:
        print(f'  NOT IN THE SOURCE FOLDER ({len(missing)}): '
              f'{missing[:20]}{" ..." if len(missing) > 20 else ""}')
    print()

    os.makedirs(os.path.join(a.dst, 'melody'), exist_ok=True)
    os.makedirs(os.path.join(a.dst, 'chord'), exist_ok=True)
    print(f'[dst] writing to {os.path.abspath(a.dst)}')
    pre_existing = {
        sub: sorted(glob(os.path.join(a.dst, sub, '*.mid')))
        for sub in ('melody', 'chord')}
    n_pre = sum(len(v) for v in pre_existing.values())
    if n_pre:
        print(f'[dst] {n_pre} midi(s) already there from an earlier build; '
              f'kept songs are overwritten and anything no longer kept is '
              f'REMOVED at the end, so the folder is one build, never two')
    irregular, src_db = {}, {}
    if not a.align_report:
        print('\n' + '!' * 68)
        print('NO --align-report. Without it there are no annotated downbeats, '
              'so:\n'
              '  - the pickup test falls back to the midi bar lines and a\n'
              '    +-head-beats tolerance, which cannot tell a manufactured\n'
              '    pickup bar line from a real one;\n'
              '  - bars come from the file, which is a flat 4/4 once the\n'
              '    aligner runs with TIME_SIGS=0, so a metre change is\n'
              '    invisible and the crop can land mid-bar.\n'
              'Point ALIGN_REPORT at the aligner\'s align_grid_report.tsv.')
        print('!' * 68 + '\n')
    if a.align_report:
        with open(a.align_report) as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                raw = (row.get('irregular_bars') or '').strip()
                irregular[row['id']] = (
                    {int(x) for x in raw.split(',') if x != ''} if raw
                    else set())
                # TRUE downbeat frames. Essential, not decorative: with
                # TIME_SIGS=0 the midi is a flat 4/4, so pretty_midi's
                # get_downbeats returns multiples of 16 frames -- the
                # GRID, not the song's downbeats. Past an irregular bar
                # those differ, and cropping at a grid line then lands
                # the prompt mid-bar while every internal check agrees
                # with itself. audit_prompt_phase caught exactly that on
                # songs 010, 456 and 816.
                dbr = (row.get('downbeat_frames') or '').strip()
                if dbr:
                    src_db[row['id']] = [int(x) for x in dbr.split(',')
                                         if x != '']
        n_irr = sum(1 for v in irregular.values() if v)
        print(f'[metre] {a.align_report}: {n_irr} of {len(irregular)} songs '
              f'have an irregular bar; windows will avoid them where possible')

    force_ids = {i for i in (a.force_ids or []) if i}
    if force_ids:
        print(f'[force] keeping {sorted(force_ids)} even if the normal rule '
              f'rejects them; `forced` in the TSV says which fallback was '
              f'used')
    rows, kept, reasons = [], [], {}
    # Every row carries every column, so a song dropped early still gets
    # a line and the TSV never depends on which row happened to be first.
    COLS = ['id', 'grid_pre_pad', 'first_mel_frame', 'mel_offset_beats',
            'pickup_src', 'first_mel_bar', 'is_pickup', 'lead_bars',
            'crop_bar', 'crop_tick',
            'crop_sec', 'crop_frame', 'pad_frames', 'bars_remaining',
            'pad_bars', 'lead_has_melody', 'pickup_has_melody',
            'metre_changes_in_prompt', 'metre_changes_in_window',
            'irregular_bars', 'window_regular', 'src_tempo_events',
            'src_bpm_first', 'src_bpm_median', 'forced', 'dropped']

    def stub(sid, why):
        reasons[sid] = why
        rows.append(dict({c: '' for c in COLS}, id=sid, dropped=why))

    for sid in missing:
        stub(sid, 'not in the source folder')
    for sid in ids:
        mp = os.path.join(a.mel_src, f'{sid}.mid')
        cp_ = os.path.join(a.chord_src, f'{sid}.mid')
        if not os.path.exists(cp_):
            stub(sid, 'no chord file')
            continue
        mel, chd = pm.PrettyMIDI(mp), pm.PrettyMIDI(cp_)
        if mel.resolution != chd.resolution:
            stub(sid, f'melody ppq {mel.resolution} != chord ppq '
                      f'{chd.resolution}; the streams are not on one grid')
            continue
        # ---- everything below is in TICKS, not seconds ----------------
        # A tick is a musical position; seconds are a tick times whatever
        # tempo map the file happens to carry. Analysing in seconds and
        # converting the crop point back with time_to_tick meant a source
        # with a stray tempo map cropped at a DIFFERENT tick -- and
        # measured the melody entering in a different bar -- than the
        # same notes under a constant tempo. Each file's own time_to_tick
        # takes pretty_midi's seconds back to that file's ticks exactly,
        # and the aligner's grid is 4 frames to the beat, ppq ticks to
        # the beat, so frames and ticks are the same grid at a fixed
        # ratio and no tempo enters anywhere.
        ppq = mel.resolution
        tpf = ppq / 4.0                          # ticks per frame
        mel_notes = [Note(mel.time_to_tick(n.start), mel.time_to_tick(n.end),
                          n.pitch) for i in mel.instruments for n in i.notes]
        chd_notes = [Note(chd.time_to_tick(n.start), chd.time_to_tick(n.end),
                          n.pitch) for i in chd.instruments for n in i.notes]
        if not mel_notes or not chd_notes:
            stub(sid, 'empty stream')
            continue
        end = max(max(n.end for n in mel_notes),
                  max(n.end for n in chd_notes))
        n_tempo, bpm0, bpmm = tempo_summary(mel)
        bars_sec, ts_list, _ = bar_starts(mel, mel.tick_to_time(int(end)))
        bars = np.array([mel.time_to_tick(s) for s in bars_sec], dtype=float)
        spb = float(ppq)                         # ticks per beat
        db_frames, grid_pre_pad = None, 0
        if sid in src_db and len(src_db[sid]) >= 2:
            # prefer the SOURCE's true downbeats over the file's flat grid
            fps = tpf                            # ticks per frame
            db_frames = list(src_db[sid])
            # ---- the PICKUP BAR is missing from downbeat_frames --------
            # The aligner sets origin_beat = -((-first_down) % 4), so a
            # song with an anacrusis is shifted until its first ANNOTATED
            # downbeat lands on frame 16: the aligner has already made the
            # pickup a full bar 0. But downbeat_frames lists annotated
            # downbeats only, so its first entry is that frame 16 and bar
            # 0 -- the pickup bar, the one this whole crop is defined
            # relative to -- was not in the grid at all.
            #
            # Everything then went wrong in the same direction: the first
            # melody note sat BEFORE bars[0], searchsorted returned -1,
            # F clamped to 0 (the first FULL bar), the melody looked like
            # it started on a bar line, and the crop began at frame 16 --
            # throwing the anacrusis away and counting lead bars from the
            # wrong place. Songs 001 and 003 are exactly this case.
            while db_frames[0] > 0:
                db_frames.insert(0, max(db_frames[0] - 16, 0))
                grid_pre_pad += 1
            bars = np.array([f * fps for f in db_frames], dtype=float)
        bad = degenerate_grid(bars, end)
        if bad:
            stub(sid, bad)
            continue
        m_hit = sounding_bars(mel_notes, bars)
        c_hit = sounding_bars(chd_notes, bars)
        tol = a.head_beats * spb
        m_start = complete_bars(mel_notes, bars, tol)
        both = m_hit & c_hit
        win = prompt_bars + a.min_bars
        if db_frames is not None:
            # bar i is irregular when the gap to bar i+1 is not 16 frames.
            # Indexed against the SAME grid the crop uses, pickup bar
            # included -- reading it off src_db instead would put every
            # irregular bar one index early on a pickup song.
            irr = {i for i, (x, y) in enumerate(zip(db_frames, db_frames[1:]))
                   if y - x != 16}
        else:
            irr = irregular.get(sid, set())
        forced = sid in force_ids

        # ------------------------------------------------------------
        # F = the bar holding the FIRST SOUNDING MELODY NOTE.
        #
        # This is the anchor, and it is not negotiable -- no search, no
        # fallback ladder. The previous version looked for the first bar
        # with a bar-aligned melody start AND mel_bars of both-stream
        # activity AND a regular metre, and slid later whenever one of
        # those failed. Sliding moves the crop off the song's actual
        # opening: songs 010 and 816 ended up with a "lead bar" that was
        # really bar 2 of the tune, i.e. the model was handed melody in
        # the bar that is supposed to be empty head-room. Anchoring on
        # the first note cannot do that, and the metre/both-stream
        # conditions become things REPORTED about the window rather than
        # things that move it.
        # ------------------------------------------------------------
        first_on = float(min(n.start for n in mel_notes))
        if first_on < bars[0] - 1e-6:
            # the grid still does not cover the music it is supposed to
            # measure -- refuse rather than silently cut the opening off
            stub(sid, f'melody begins {bars[0] - first_on:.3f}s BEFORE the '
                      f'first bar line of the grid -- the crop would throw '
                      f'the opening away')
            continue
        F = int(np.searchsorted(bars, first_on + 1e-9, side='right')) - 1
        F = int(np.clip(F, 0, len(bars) - 1))
        # a note a hair BEFORE a bar line is that bar's downbeat, played
        # early -- not an anacrusis one bar earlier
        if F + 1 < len(bars) and bars[F + 1] - first_on <= tol:
            F += 1
        # ---- is F a PICKUP bar? Ask beat_midi.txt, not the midi -------
        # beat_midi.txt column 3 flags the downbeat of every bar, and the
        # aligner carries those straight through to downbeat_frames. So
        # the question "does the melody start on a bar line or partway
        # into one" has an exact answer in the annotation: is the first
        # onset's frame one of the ANNOTATED downbeats?
        #
        # Exact beats the midi-side test twice over. It needs no
        # --head-beats tolerance, so a note a 16th after the downbeat is
        # no longer read as being on it. And it is not fooled by the
        # pickup bar line the grid above had to manufacture: a song whose
        # anacrusis fills its whole pickup bar starts at frame 0, which
        # IS a bar line on that grid but is NOT an annotated downbeat --
        # correctly a pickup, where complete_bars would have called it a
        # bar-line start and given it two lead bars.
        annotated = set(src_db.get(sid, ()))
        if annotated:
            first_frame = int(round(first_on / (spb / 4.0)))
            is_pickup = first_frame not in annotated
            pickup_src = 'beat_midi downbeat flags'
        else:
            # no report: fall back to the midi's own bar lines and tol
            is_pickup = not bool(m_start[F])
            pickup_src = f'midi bar lines, +-{a.head_beats} beat'
        lead_bars = 1 if is_pickup else 2
        # ANCHOR = the bar whose downbeat the melody's phrase begins on:
        # the bar after the pickup, or F itself. Both rules put it two
        # bars into the prompt -- lead + pickup, or lead + lead -- so
        # the one invariant every prompt shares is
        #
        #     the anchor downbeat lands at output frame 32 (bar 2)
        #
        # and everything in front of it is whatever the song had there,
        # padded with silence where the song had less than two bars.
        anchor = F + 1 if is_pickup else F
        if anchor >= len(bars):
            stub(sid, 'melody enters in the last bar of the grid')
            continue
        start_bar = max(anchor - 2, 0)
        how = ''
        # The pad is measured in FRAMES, to what makes the invariant
        # hold -- not in whole bars. A whole-bar pad only keeps the
        # pickup a pickup when the bar it sits in is already a full 4/4
        # bar. When the aligner leaves the opening bar partial (a
        # two-beat bar 0), a whole bar in front puts the downbeat in the
        # middle of output bar 1, the anacrusis stops being an anacrusis,
        # and every bar-based metric is a half-bar out of phase. Padding
        # to the frame keeps the pickup exactly where it was relative to
        # its downbeat, whatever length the aligner gave the bar.
        pad_t = int(round(32 * tpf - (bars[anchor] - bars[start_bar])))
        if pad_t < 0:
            # a LONG bar in front (6/4 etc.) pushes the anchor past frame
            # 32; front padding cannot fix that, and the window-regular
            # column already says the bar is irregular
            how = 'a long bar in front puts the anchor past frame 32'
            pad_t = 0
        pad_frames = int(round(pad_t / tpf))
        pad_bars = pad_frames / 16.0
        # the mel_bars of melody the prompt is meant to carry begin on
        # the anchor
        s0 = anchor

        def clean(b0):
            """Is the window from the ANCHOR on free of irregular bars?
            Bars in front of the anchor are the lead, and a short one
            there (the aligner's partial opening bar) is exactly what the
            frame pad repairs; a long one is flagged above."""
            return not any(anchor <= x < b0 + win for x in irr)

        if irr and not clean(start_bar):
            how = (how + '; ' if how else '') + 'window crosses a metre change'
        if not both[s0:s0 + a.mel_bars].all():
            n_both = int(both[s0:s0 + a.mel_bars].sum())
            how = (how + '; ' if how else '') + \
                f'only {n_both}/{a.mel_bars} prompt bars have both streams'
        remaining = len(bars) - start_bar
        ts_ticks = [mel.time_to_tick(t.time) for t in ts_list]
        n_ts_prompt = sum(1 for tt in ts_ticks
                          if bars[start_bar] < tt <
                          bars[min(start_bar + prompt_bars, len(bars) - 1)])
        win_end = min(start_bar + prompt_bars + a.min_bars, len(bars) - 1)
        n_ts_win = sum(1 for tt in ts_ticks
                       if bars[start_bar] < tt < bars[win_end])
        tick0 = int(round(bars[start_bar]))
        row = dict(id=sid, grid_pre_pad=grid_pre_pad,
                   first_mel_frame=int(round(first_on / (spb / 4.0))),
                   # how far INTO its bar the melody enters. A real
                   # anacrusis sits near the end of the bar (001 enters
                   # on beat 3 of 4); a phrase that merely starts on the
                   # off-beat of its own first bar sits near the start
                   # (003 enters on beat 0.5). Both are "not on a
                   # downbeat", so the annotation calls both a pickup --
                   # this column is what lets you see the difference.
                   mel_offset_beats=f'{(first_on - bars[F]) / spb:.2f}',
                   pickup_src=pickup_src,
                   first_mel_bar=F, is_pickup=int(is_pickup),
                   lead_bars=lead_bars, crop_bar=start_bar,
                   crop_tick=tick0,
                   # seconds AT THE WRITTEN TEMPO -- the only tempo the
                   # prompt set has
                   crop_sec=f'{tick0 / ppq * 60.0 / (a.tempo or bpmm):.3f}',
                   # the crop offset in FRAMES. Recorded rather than left
                   # to be re-derived as crop_bar*16, which is only true
                   # when bars are a flat 4/4 grid -- exactly the
                   # assumption that made the first phase audit disagree
                   # with the builder.
                   crop_frame=int(round(bars[start_bar] / (spb / 4.0))),
                   pad_frames=pad_frames,
                   bars_remaining=remaining,
                   pad_bars=f'{pad_bars:g}',
                   # 0 by construction -- there is no melody anywhere
                   # before F -- and computed rather than asserted so a
                   # future change to the anchor shows up here instead of
                   # silently handing the model a head-room bar with a
                   # tune in it.
                   lead_has_melody=int(bool(m_hit[start_bar:F].any()))
                   if F > start_bar else 0,
                   pickup_has_melody=int(is_pickup and bool(m_hit[F])),
                   metre_changes_in_prompt=n_ts_prompt,
                   metre_changes_in_window=n_ts_win,
                   irregular_bars=len(irr),
                   window_regular=int(bool(clean(start_bar))),
                   src_tempo_events=n_tempo, src_bpm_first=f'{bpm0:.2f}',
                   src_bpm_median=f'{bpmm:.2f}',
                   forced=how)
        if remaining < prompt_bars + a.min_bars and not forced:
            reasons[sid] = f'only {remaining} bars left (need ' \
                           f'{prompt_bars + a.min_bars})'
            row['dropped'] = reasons[sid]
            rows.append(row)
            continue
        if a.require_regular and n_ts_win and not forced:
            reasons[sid] = f'{n_ts_win} metre change(s) in the window'
            row['dropped'] = reasons[sid]
            rows.append(row)
            continue
        row['dropped'] = ''
        # ONE tick for both streams. They share a ppq (checked above)
        # and the aligner's grid, so the same tick is the same musical
        # instant in both -- no tempo map is consulted on the way.
        # Both streams get the identical pad, in ticks.
        bpm = a.tempo if a.tempo > 0 else None
        crop_ticks(mp, os.path.join(a.dst, 'melody', f'{sid}.mid'),
                   tick0, pad_ticks=pad_t, force_bpm=bpm)
        crop_ticks(cp_, os.path.join(a.dst, 'chord', f'{sid}.mid'),
                   tick0, pad_ticks=pad_t, force_bpm=bpm)
        rows.append(row)
        kept.append(sid)

    # ---- one folder, one build -------------------------------------
    # Overwriting the kept songs is not enough: a song dropped by this
    # run keeps whatever an EARLIER run wrote, and anything that globs
    # the folder picks it up. That is how a set built by an older rule
    # goes on being used after the rule changes -- the files for the
    # songs that still pass look new, and the rest are fossils.
    keep_set = set(kept)
    stale = [p for sub in ('melody', 'chord')
             for p in glob(os.path.join(a.dst, sub, '*.mid'))
             if song_id(p) not in keep_set]
    for p in stale:
        os.remove(p)
    if stale:
        print(f'[dst] removed {len(stale)} midi(s) left by an earlier build: '
              f'{sorted({song_id(p) for p in stale})[:20]}')

    import datetime
    import subprocess
    try:
        head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        head = head.stdout.strip() or 'unknown'
    except Exception:                                       # noqa: BLE001
        head = 'unknown'
    with open(os.path.join(a.dst, 'build_stamp.txt'), 'w') as fh:
        fh.write(f'built   {datetime.datetime.now().isoformat(timespec="seconds")}\n'
                 f'commit  {head}\n'
                 f'rule    lead = 1 bar + the pickup, or 2 bars; prompt = '
                 f'{prompt_bars} bars = {prompt_bars * 16} frames\n'
                 f'mel_src {os.path.abspath(a.mel_src)}\n'
                 f'report  {a.align_report or "NONE -- midi-side pickup test"}\n'
                 f'kept    {len(kept)}\n')

    rows.sort(key=lambda r: r['id'])
    if rows:
        with open(os.path.join(a.dst, 'prompt_crops.tsv'), 'w',
                  newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, delimiter='\t')
            w.writeheader()
            w.writerows(rows)
    print(f'KEPT {len(kept)} songs -> {a.dst}/{{melody,chord}}')
    print(f'  prompt = {prompt_bars} bars ({prompt_bars * 16} frames) '
          f'everywhere: lead + {a.mel_bars} sounding bars, where the lead '
          f'is 1 bar + the pickup for a song with an anacrusis and 2 bars '
          f'for one that starts on a bar line')
    if kept:
        keep_rows = [r for r in rows if not r['dropped']]
        pk = np.array([int(r['pickup_has_melody']) for r in keep_rows])
        tw = np.array([int(r['metre_changes_in_window']) for r in keep_rows])
        cb = np.array([int(r['crop_bar']) for r in keep_rows])
        pb = np.array([float(r['pad_bars']) for r in keep_rows])
        lm = np.array([int(r['lead_has_melody']) for r in keep_rows])
        ip = np.array([int(r['is_pickup']) for r in keep_rows])
        srcs = {r['pickup_src'] for r in keep_rows}
        print(f'  anchor: {int(ip.sum())}/{len(ip)} songs open with a '
              f'PICKUP bar (1 bar in front), {int((ip == 0).sum())} start '
              f'on a bar line (2 bars in front)')
        print(f'  pickup decided by: {sorted(srcs)}')
        print(f'  lead bars: {int((pb == 0).sum())} entirely from the song, '
              f'{int((pb > 0).sum())} needed padding '
              f'({int(pb.sum())} bars padded in total)')
        if lm.sum():
            print(f'  !! {int(lm.sum())} song(s) have MELODY in a lead bar '
                  f'-- that should be impossible with a first-note anchor')
        print(f'  crop point: median bar {int(np.median(cb))}, '
              f'max {int(cb.max())}')
        gp = np.array([int(r['grid_pre_pad'] or 0) for r in keep_rows])
        if gp.sum():
            print(f'  {int((gp > 0).sum())} song(s) had their PICKUP BAR '
                  f'restored to the grid (downbeat_frames starts at 16 on '
                  f'a song the aligner shifted for an anacrusis)')
        bpms = {r['src_bpm_median'] for r in rows
                if not r['dropped'] and r['src_bpm_median']}
        nte = {r['src_tempo_events'] for r in rows
               if not r['dropped'] and r['src_tempo_events'] != ''}
        print(f'  SOURCE tempo: {len(bpms)} distinct median bpm, tempo-event '
              f'counts {sorted(nte)}'
              + ('' if len(bpms) == 1 and nte == {1}
                 else '  <- the source is NOT one constant tempo'))
        # verify the WRITTEN files, not the intent
        bad_t = []
        for sid in kept:
            for sub in ('melody', 'chord'):
                mf = mido.MidiFile(os.path.join(a.dst, sub, f'{sid}.mid'))
                tp = [round(mido.tempo2bpm(m.tempo), 3) for tr in mf.tracks
                      for m in tr if m.type == 'set_tempo']
                want = [round(a.tempo, 3)] if a.tempo > 0 else None
                if want is not None and tp != want:
                    bad_t.append((sid, sub, tp))
        if a.tempo > 0:
            print(f'  WRITTEN tempo: '
                  + (f'every file is exactly one set_tempo at {a.tempo:g} bpm'
                     if not bad_t else
                     f'{len(bad_t)} file(s) are NOT a single {a.tempo:g}: '
                     f'{bad_t[:6]}'))
            if bad_t:
                raise SystemExit('tempo forcing failed -- see above')
        else:
            print('  WRITTEN tempo: source map copied verbatim (--tempo 0)')
        if irregular:
            wr = [r for r in rows if not r['dropped'] and r['window_regular'] != '']
            nreg = sum(1 for r in wr if int(r['window_regular']))
            print(f'  scored window metrically REGULAR in {nreg}/{len(wr)} '
                  f'kept songs'
                  + ('' if nreg == len(wr) else
                     f'  ({len(wr) - nreg} cross a metre change -- see '
                     f'`forced`)'))
        fz = [(r['id'], r['forced']) for r in keep_rows if r.get('forced')]
        if fz:
            print(f'  {len(fz)} kept song(s) carry a caveat in `forced`:')
            for s, why in fz[:12]:
                print(f'      {s}: {why}')
        print(f'  metre changes inside the scored window: '
              f'{int((tw > 0).sum())} songs'
              + ('' if a.require_regular else
                 '  <- --require-regular drops these'))
    if reasons:
        from collections import Counter
        c = Counter(v.split('(')[0].strip() for v in reasons.values())
        print(f'  dropped {len(reasons)}: {dict(c)}')
    print(f'\nper-song TSV -> {a.dst}/prompt_crops.tsv')


if __name__ == '__main__':
    main()
