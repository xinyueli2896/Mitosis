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

import numpy as np
import pretty_midi as pm

warnings.filterwarnings('ignore')
RESOLUTION = 480


def song_id(name):
    m = re.search(r'(\d{3})', os.path.basename(str(name)))
    return m.group(1) if m else None


def heldout_ids(stem, split_ratio=10, train_length=None):
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
    keep = (lengths >= train_length) if train_length else \
        np.ones(len(lengths), dtype=bool)
    idx = np.arange(len(lengths))[np.asarray(keep)]
    val = [i for i in idx if i % split_ratio == 0]
    return {song_id(names[i]) for i in val if song_id(names[i])}


def bar_starts(mid, end_time):
    """Bar-start times in seconds, from the file's OWN metre map."""
    ts = sorted(mid.time_signature_changes, key=lambda t: t.time) or \
        [pm.TimeSignature(4, 4, 0.0)]
    tempo = mid.estimate_tempo() if False else None
    # constant tempo by construction (the aligner writes one); take it
    # from the tempo map rather than guessing
    times, tempi = mid.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    spb = 60.0 / bpm
    out, t, i = [], 0.0, 0
    while t <= end_time + 1e-6 and len(out) < 10000:
        while i + 1 < len(ts) and ts[i + 1].time <= t + 1e-9:
            i += 1
        out.append(t)
        t += ts[i].numerator * spb * (4.0 / ts[i].denominator)
    return np.array(out), ts, spb


def sounding_bars(inst_notes, bars):
    """Per bar, does any note ONSET fall in it."""
    if not inst_notes:
        return np.zeros(len(bars), dtype=bool)
    on = np.array([n.start for n in inst_notes])
    idx = np.searchsorted(bars, on, side='right') - 1
    hit = np.zeros(len(bars), dtype=bool)
    idx = idx[(idx >= 0) & (idx < len(bars))]
    hit[idx] = True
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


def crop(mid, t0, ts_list, spb):
    """Everything from t0 on, shifted to 0. Notes starting before t0 are
    discarded outright (the chord before the pickup bar goes with it);
    a note straddling t0 is dropped rather than truncated, so no
    half-note appears out of nowhere at bar 1 beat 1."""
    times, tempi = mid.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    out = pm.PrettyMIDI(resolution=RESOLUTION, initial_tempo=bpm)
    for inst in mid.instruments:
        new = pm.Instrument(program=inst.program, is_drum=inst.is_drum,
                            name=inst.name)
        for n in inst.notes:
            if n.start < t0 - 1e-9:
                continue
            new.notes.append(pm.Note(velocity=n.velocity, pitch=n.pitch,
                                     start=n.start - t0, end=n.end - t0))
        out.instruments.append(new)
    keep = []
    cur = None
    for t in ts_list:
        if t.time <= t0 + 1e-9:
            cur = t
        else:
            keep.append(pm.TimeSignature(t.numerator, t.denominator,
                                         t.time - t0))
    head = pm.TimeSignature(cur.numerator, cur.denominator, 0.0) if cur \
        else pm.TimeSignature(4, 4, 0.0)
    out.time_signature_changes = [head] + keep
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mel-src', required=True)
    p.add_argument('--chord-src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--dataset', default='pop909_melody_cp8_v2',
                   help='dataset stem whose index %% split-ratio == 0 songs '
                        'are the held-out set (see audit_wholesong_split)')
    p.add_argument('--split-ratio', type=int, default=10)
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

    prompt_bars = a.mel_bars + 1
    want = set(a.ids) if a.ids else heldout_ids(a.dataset, a.split_ratio)
    have = {song_id(f) for f in glob(os.path.join(a.mel_src, '*.mid'))}
    ids = sorted(i for i in want if i and i in have)
    print(f'held-out candidates: {len(want)}; present in {a.mel_src}: '
          f'{len(ids)}\n')

    os.makedirs(os.path.join(a.dst, 'melody'), exist_ok=True)
    os.makedirs(os.path.join(a.dst, 'chord'), exist_ok=True)
    rows, kept, reasons = [], [], {}
    for sid in ids:
        mp = os.path.join(a.mel_src, f'{sid}.mid')
        cp_ = os.path.join(a.chord_src, f'{sid}.mid')
        if not os.path.exists(cp_):
            reasons[sid] = 'no chord file'
            continue
        mel, chd = pm.PrettyMIDI(mp), pm.PrettyMIDI(cp_)
        mel_notes = [n for i in mel.instruments for n in i.notes]
        chd_notes = [n for i in chd.instruments for n in i.notes]
        if not mel_notes or not chd_notes:
            reasons[sid] = 'empty stream'
            continue
        end = max(max(n.end for n in mel_notes),
                  max(n.end for n in chd_notes))
        bars, ts_list, spb = bar_starts(mel, end)
        m_hit = sounding_bars(mel_notes, bars)
        c_hit = sounding_bars(chd_notes, bars)
        tol = a.head_beats * spb
        m_start = complete_bars(mel_notes, bars, tol)
        both = m_hit & c_hit
        # B = the first bar the melody OFFICIALLY starts on (m_start) and
        # from which mel_bars consecutive bars have both streams. B >= 1
        # so the bar in front of it is free to hold the anacrusis.
        B = None
        for b in range(1, len(bars) - a.mel_bars):
            if m_start[b] and both[b:b + a.mel_bars].all():
                B = b
                break
        if B is None:
            reasons[sid] = f'no bar-aligned start with {a.mel_bars} consecutive both-stream bars'
            continue
        start_bar = B - 1
        remaining = len(bars) - start_bar
        n_ts_prompt = sum(1 for t in ts_list
                          if bars[start_bar] < t.time <
                          bars[min(start_bar + prompt_bars, len(bars) - 1)])
        win_end = min(start_bar + prompt_bars + a.min_bars, len(bars) - 1)
        n_ts_win = sum(1 for t in ts_list
                       if bars[start_bar] < t.time < bars[win_end])
        row = dict(id=sid, first_both_bar=B, crop_bar=start_bar,
                   crop_sec=f'{bars[start_bar]:.3f}',
                   bars_remaining=remaining,
                   pickup_has_melody=int(bool(m_hit[start_bar])),
                   metre_changes_in_prompt=n_ts_prompt,
                   metre_changes_in_window=n_ts_win)
        if remaining < prompt_bars + a.min_bars:
            reasons[sid] = f'only {remaining} bars left (need ' \
                           f'{prompt_bars + a.min_bars})'
            row['dropped'] = reasons[sid]
            rows.append(row)
            continue
        if a.require_regular and n_ts_win:
            reasons[sid] = f'{n_ts_win} metre change(s) in the window'
            row['dropped'] = reasons[sid]
            rows.append(row)
            continue
        row['dropped'] = ''
        t0 = float(bars[start_bar])
        crop(mel, t0, ts_list, spb).write(
            os.path.join(a.dst, 'melody', f'{sid}.mid'))
        _, cts, _ = bar_starts(chd, end)
        crop(chd, t0, cts or ts_list, spb).write(
            os.path.join(a.dst, 'chord', f'{sid}.mid'))
        rows.append(row)
        kept.append(sid)

    if rows:
        with open(os.path.join(a.dst, 'prompt_crops.tsv'), 'w',
                  newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                               delimiter='\t')
            w.writeheader()
            w.writerows(rows)
    print(f'KEPT {len(kept)} songs -> {a.dst}/{{melody,chord}}')
    print(f'  prompt = 1 pickup bar + {a.mel_bars} sounding bars '
          f'= {prompt_bars} bars')
    if kept:
        pk = np.array([r['pickup_has_melody'] for r in rows
                       if not r['dropped']])
        tw = np.array([r['metre_changes_in_window'] for r in rows
                       if not r['dropped']])
        cb = np.array([r['crop_bar'] for r in rows if not r['dropped']])
        print(f'  pickup bar actually holds melody in {int(pk.sum())}/'
              f'{len(pk)} (the rest start on a bar line, buffer is silent)')
        print(f'  crop point: median bar {int(np.median(cb))}, '
              f'max {int(cb.max())}')
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
