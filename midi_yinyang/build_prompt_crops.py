"""Held-out inference set: crop every song so the prompt is the SAME
musical content everywhere.

The old E1 selection asked two things of a song: that it be in
whole-song-gen's split, and that both streams happen to be sounding in
the first 6 or 8 bars. The second is a property of where the song's
intro ends, so it threw away most of the corpus and still left the
prompts holding different amounts of music.

This inverts it. The prompt is CUT TO the melody instead of hoping the
melody arrives in time:

    bar 1        pickup / buffer -- whatever melody precedes the first
                 full bar, or silence if the song has no pickup
    bars 2..5    the first --mel-bars (default 4) consecutive bars in
                 which BOTH streams sound

so every prompt carries four sounding bars of melody plus one bar of
head-room, regardless of how long the intro was. Chord is cropped on
exactly the same boundary -- everything before the pickup bar is
discarded -- so the two streams stay in register by construction.

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


def crop_ticks(src_path, dst_path, tick0, pad_ticks=0):
    """Window a MIDI file in TICK space, changing nothing else.

    Cropping must be a pure windowing operation: every tempo and
    time-signature event survives, and a surviving note keeps the exact
    tick offsets it had relative to them. The previous version rebuilt
    the file through pretty_midi with ONE tempo taken from tempi[0] while
    keeping note times in SECONDS -- so any file carrying a tempo map came
    out with different tick positions, i.e. a different notated rhythm,
    and a different tempo per song depending on what its first event
    happened to be. That is editing the source, not cropping it.

    pad_ticks prepends that many EMPTY ticks, used when the song has no
    bar before its pickup and one has to be manufactured. It is a pure
    shift of everything that survives -- no note is created, and the
    relative timing inside the window is untouched.

    Notes are kept when their ONSET is at or after tick0 and shifted by
    -tick0 + pad_ticks; a note straddling the cut is dropped rather than truncated,
    so nothing appears from nowhere at bar 1 beat 1. Meta and control
    events before the cut are re-emitted at tick 0, because they set
    state the window needs (tempo, metre, program, key).
    """
    src = mido.MidiFile(src_path)
    out = mido.MidiFile(ticks_per_beat=src.ticks_per_beat, type=src.type)
    for tr in src.tracks:
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
            else:
                events.append((max(t - tick0, 0) + pad_ticks, 2, msg))
        for stack in open_on.values():                 # never closed
            for on_t, on_msg in stack:
                if on_t >= tick0:
                    events.append((on_t - tick0 + pad_ticks, 1, on_msg))
        events.sort(key=lambda e: (e[0], -e[1]))
        new_tr = mido.MidiTrack()
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
                   help='ids to keep even when the normal rule rejects '
                        'them. The crop point is still chosen, by the '
                        'first fallback that works: melody-only bars, '
                        'then any bar-aligned melody start, then any '
                        'melody at all. MIN_BARS is waived too. The TSV '
                        'records which rule was used in `forced`, so a '
                        'forced song is never mistaken for one that '
                        'passed -- use it when you have LISTENED to the '
                        'song and know the crop is fine.')
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
                   help='consecutive sounding bars required in the prompt '
                        '(the prompt is this + 1 pickup bar)')
    p.add_argument('--head-beats', type=float, default=0.25,
                   help='how close to a bar line a melody note must begin '
                        'for that bar to count as the melody OFFICIALLY '
                        'starting (in beats; 0.25 = a 16th). Bars the '
                        'melody only wanders into partway through are '
                        'anacrusis, and belong in the buffer bar.')
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
    irregular, src_db = {}, {}
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
    for sid in missing:
        rows.append(dict(id=sid, first_both_bar='', crop_bar='',
                         crop_sec='', bars_remaining='',
                         pickup_has_melody='', metre_changes_in_prompt='',
                         metre_changes_in_window='', pad_bars='', lead_has_melody='', pad_frames='', crop_frame='', irregular_bars='', window_regular='', src_tempo_events='', src_bpm_first='', src_bpm_median='', forced='',
                         dropped='not in the source folder'))
        reasons[sid] = 'not in the source folder'

    def stub(sid, why):
        reasons[sid] = why
        rows.append(dict(id=sid, first_both_bar='', crop_bar='',
                         crop_sec='', bars_remaining='',
                         pickup_has_melody='', metre_changes_in_prompt='',
                         metre_changes_in_window='', pad_bars='', lead_has_melody='', pad_frames='', crop_frame='', irregular_bars='', window_regular='', src_tempo_events='', src_bpm_first='', src_bpm_median='', forced='',
                         dropped=why))
    for sid in ids:
        mp = os.path.join(a.mel_src, f'{sid}.mid')
        cp_ = os.path.join(a.chord_src, f'{sid}.mid')
        if not os.path.exists(cp_):
            stub(sid, 'no chord file')
            continue
        mel, chd = pm.PrettyMIDI(mp), pm.PrettyMIDI(cp_)
        mel_notes = [n for i in mel.instruments for n in i.notes]
        chd_notes = [n for i in chd.instruments for n in i.notes]
        if not mel_notes or not chd_notes:
            stub(sid, 'empty stream')
            continue
        end = max(max(n.end for n in mel_notes),
                  max(n.end for n in chd_notes))
        n_tempo, bpm0, bpmm = tempo_summary(mel)
        bars, ts_list, spb = bar_starts(mel, end)
        if sid in src_db and len(src_db[sid]) >= 2:
            # prefer the SOURCE's true downbeats over the file's flat grid
            fps = spb / 4.0                      # seconds per frame
            bars = np.array([f * fps for f in src_db[sid]], dtype=float)
        bad = degenerate_grid(bars, end)
        if bad:
            stub(sid, bad)
            continue
        m_hit = sounding_bars(mel_notes, bars)
        c_hit = sounding_bars(chd_notes, bars)
        tol = a.head_beats * spb
        m_start = complete_bars(mel_notes, bars, tol)
        both = m_hit & c_hit
        # B = the first bar the melody OFFICIALLY starts on (m_start) and
        # from which mel_bars consecutive bars have both streams. B >= 1
        # so the bar in front of it is free to hold the anacrusis.
        win = prompt_bars + a.min_bars
        if sid in src_db and len(src_db[sid]) >= 2:
            # bar i is irregular when the gap to bar i+1 is not 16 frames
            f = src_db[sid]
            irr = {i for i, (x, y) in enumerate(zip(f, f[1:])) if y - x != 16}
        else:
            irr = irregular.get(sid, set())

        def clean(b):
            """Is the whole scored window free of irregular bars? Bar b-2
            is the lead, b-1 the pickup, so the window runs
            [b-2, b-2+win) -- clamped at 0 where the lead is padded."""
            lo = max(b - 2, 0)
            return not any(lo <= x < lo + win for x in irr)

        B, how = None, ''
        # first pass: a window that is metrically regular
        for b in range(1, len(bars) - a.mel_bars):
            if m_start[b] and both[b:b + a.mel_bars].all() and clean(b):
                B = b
                break
        if B is None and irr:
            # fall back to the first valid start even if the window
            # crosses a metre change, and SAY so in the row
            for b in range(1, len(bars) - a.mel_bars):
                if m_start[b] and both[b:b + a.mel_bars].all():
                    B, how = b, 'window crosses a metre change'
                    break
        elif B is None:
            for b in range(1, len(bars) - a.mel_bars):
                if m_start[b] and both[b:b + a.mel_bars].all():
                    B = b
                    break
        forced = sid in force_ids
        if B is None and forced:
            # Fallbacks, most demanding first; `how` records which one.
            for label, test in (
                ('melody-only bars',
                 lambda b: m_start[b] and m_hit[b:b + a.mel_bars].all()),
                ('bar-aligned melody start', lambda b: m_start[b]),
                ('any melody', lambda b: m_hit[b]),
            ):
                for b in range(1, max(len(bars) - a.mel_bars, 2)):
                    if test(b):
                        B, how = b, label
                        break
                if B is not None:
                    break
        if B is None:
            # Say WHICH half of the condition failed, so this is
            # actionable instead of a shrug. Either the melody never
            # begins on a bar line (raise --head-beats) or the two
            # streams are never both active for mel_bars in a row.
            run = best = 0
            for v in both:
                run = run + 1 if v else 0
                best = max(best, run)
            stub(sid, f'no bar-aligned start with {a.mel_bars} consecutive '
                      f'both-stream bars (longest both-stream run {best}; '
                      f'{int(m_start.sum())} bar-aligned melody starts in '
                      f'{len(bars)} bars)')
            continue
        # One bar in front of the pickup: the song's own if it has one,
        # otherwise an empty bar manufactured by padding. The pickup bar
        # itself stays where it was, at B-1.
        lead = B - 2
        pad_bars = 1 if lead < 0 else 0
        start_bar = max(lead, 0)
        remaining = len(bars) - start_bar
        n_ts_prompt = sum(1 for t in ts_list
                          if bars[start_bar] < t.time <
                          bars[min(start_bar + prompt_bars, len(bars) - 1)])
        win_end = min(start_bar + prompt_bars + a.min_bars, len(bars) - 1)
        n_ts_win = sum(1 for t in ts_list
                       if bars[start_bar] < t.time < bars[win_end])
        row = dict(id=sid, first_both_bar=B, crop_bar=start_bar,
                   crop_sec=f'{bars[start_bar]:.3f}',
                   # the crop offset in FRAMES. Recorded rather than left
                   # to be re-derived as crop_bar*16, which is only true
                   # when bars are a flat 4/4 grid -- exactly the
                   # assumption that made the first phase audit disagree
                   # with the builder.
                   crop_frame=int(round(bars[start_bar] / (spb / 4.0))),
                   pad_frames=pad_bars * 16,
                   bars_remaining=remaining,
                   pad_bars=pad_bars,
                   lead_has_melody=int(bool(m_hit[start_bar]))
                   if not pad_bars else 0,
                   pickup_has_melody=int(bool(m_hit[B - 1])),
                   metre_changes_in_prompt=n_ts_prompt,
                   metre_changes_in_window=n_ts_win,
                   irregular_bars=len(irr),
                   window_regular=int(bool(B is not None and clean(B))),
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
        t0 = float(bars[start_bar])
        # Each file converts the crop TIME to its own ticks through its
        # own tempo map, so the two streams cut at the same musical
        # instant even if their maps ever differ.
        # one padded bar = 4 beats; both streams get the identical pad,
        # so they stay in register
        pad_m = pad_bars * 4 * mel.resolution
        pad_c = pad_bars * 4 * chd.resolution
        crop_ticks(mp, os.path.join(a.dst, 'melody', f'{sid}.mid'),
                   int(round(mel.time_to_tick(t0))), pad_ticks=pad_m)
        crop_ticks(cp_, os.path.join(a.dst, 'chord', f'{sid}.mid'),
                   int(round(chd.time_to_tick(t0))), pad_ticks=pad_c)
        rows.append(row)
        kept.append(sid)

    rows.sort(key=lambda r: r['id'])
    if rows:
        with open(os.path.join(a.dst, 'prompt_crops.tsv'), 'w',
                  newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                               delimiter='\t')
            w.writeheader()
            w.writerows(rows)
    print(f'KEPT {len(kept)} songs -> {a.dst}/{{melody,chord}}')
    print(f'  prompt = 1 lead bar + 1 pickup bar + {a.mel_bars} sounding '
          f'bars = {prompt_bars} bars ({prompt_bars * 16} frames)')
    if kept:
        pk = np.array([r['pickup_has_melody'] for r in rows
                       if not r['dropped']])
        tw = np.array([r['metre_changes_in_window'] for r in rows
                       if not r['dropped']])
        cb = np.array([r['crop_bar'] for r in rows if not r['dropped']])
        pb = np.array([int(r['pad_bars']) for r in rows if not r['dropped']])
        print(f'  lead bar: {int((pb == 0).sum())} taken from the song, '
              f'{int(pb.sum())} padded empty (no bar before the pickup)')
        print(f'  pickup bar actually holds melody in {int(pk.sum())}/'
              f'{len(pk)} (the rest start on a bar line, buffer is silent)')
        print(f'  crop point: median bar {int(np.median(cb))}, '
              f'max {int(cb.max())}')
        bpms = {r['src_bpm_median'] for r in rows
                if not r['dropped'] and r['src_bpm_median']}
        nte = {r['src_tempo_events'] for r in rows
               if not r['dropped'] and r['src_tempo_events'] != ''}
        print(f'  SOURCE tempo (carried through unchanged): '
              f'{len(bpms)} distinct median bpm, tempo-event counts {sorted(nte)}'
              + ('' if len(bpms) == 1 and nte == {1}
                 else '  <- v5 is NOT one constant tempo; the crop copies it verbatim'))
        if irregular:
            wr = [r for r in rows if not r['dropped'] and r['window_regular'] != '']
            nreg = sum(1 for r in wr if int(r['window_regular']))
            print(f'  scored window metrically REGULAR in {nreg}/{len(wr)} '
                  f'kept songs'
                  + ('' if nreg == len(wr) else
                     f'  ({len(wr) - nreg} cross a metre change -- see '
                     f'`forced`)'))
        fz = [r['id'] for r in rows if not r['dropped'] and r.get('forced')]
        if fz:
            print(f'  FORCED (normal rule rejected, kept anyway): {fz}')
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
