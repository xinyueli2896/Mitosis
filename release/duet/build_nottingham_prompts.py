"""Held-out Nottingham prompts in the heldout_v5 format.

The Nottingham counterpart of build_prompt_crops.py. Every held-out tune
is split into its two streams and cropped so that, exactly as for
POP909, THE MELODY'S FIRST FULL DOWNBEAT IS AT FRAME 32 (two bars of
lead in front of it, padded with silence where the tune has less):

    tune opens with a pickup   the partial bar sits in bar 2 and ends on
                               frame 32; bar 1 is empty
    tune opens on a downbeat   bars 1-2 are empty

Which tunes are held out. The same rule build_prompt_crops.heldout_ids
uses: a tune is held out iff its index in the training manifest
(data/<dataset>.txt) is 0 mod --split-ratio and it is at least
TRAIN_LENGTH frames long. FramedDataset trains on index %% 10 > 1 and
validates on == 1, so these tunes are never seen in training.

Where the first downbeat is. A Nottingham MIDI is written by abc2midi:
there is no meta event for an anacrusis, the pickup is simply a partial
first bar starting at tick 0. Chord symbols sit on downbeats, so the
chord track gives the bar grid's phase (its first onset modulo the bar);
the first downbeat is the first grid line at or after the melody's first
onset. A tune without chords falls back to a grid starting at tick 0.

Metre. The scoring code (eval_metrics.FRAMES_PER_BAR) and the paper's
protocol (80-frame prompts, 416-frame continuations) assume 16 frames
per bar, so by default only tunes whose every time signature has a
16-frame bar (4/4, 2/2) are kept; --meters widens that if the scoring is
adapted. Training uses every metre.

Each written pair has one set_tempo (--tempo, 120 as in heldout_v5) and
programs as in the source (0 for both streams); stage_prompts.sh tags the
merged copy for the single-stream models.

Outputs under <dst>/:  melody/<tune>.mid  chord/<tune>.mid
                       prompt_crops.tsv  (one row per held-out tune)

Usage (scripts/prepare_data.sh eval with DATASET=nottingham):
    python build_nottingham_prompts.py --src /path/to/nottingham-dataset/MIDI \\
        --dst input/nottingham_heldout --dataset nottingham_melody_cp4_v2
"""
import argparse
import csv
import os
import shutil
import tempfile

import mido

from build_prompt_crops import TRAIN_LENGTH, crop_ticks

FRAMES_PER_BEAT = 4     # sixteenth-note frames per quarter note


def heldout_tunes(stem, split_ratio, train_length):
    """Tune stems at index %% split_ratio == 0 of data/<stem>.txt that are
    at least train_length frames long (build_prompt_crops.heldout_ids,
    without its three-digit POP909 song-id parsing)."""
    import torch
    txt, lpt = f'data/{stem}.txt', f'data/{stem}.length.pt'
    if not (os.path.exists(txt) and os.path.exists(lpt)):
        raise SystemExit(f'missing {txt} or {lpt}: run prepare_data.sh train first')
    names = {}
    with open(txt) as fh:
        for line in fh:
            i, nm = line.rstrip('\n').split('\t', 1)
            names[int(i)] = os.path.splitext(os.path.basename(nm))[0]
    lengths = torch.load(lpt).tolist()
    held = [names[i] for i in range(len(lengths))
            if i % split_ratio == 0 and lengths[i] >= train_length]
    print(f'[split] {stem}: {len(lengths)} tunes, {len(held)} held out '
          f'(idx % {split_ratio} == 0, >= {train_length} frames)')
    return held


def note_tracks(mid):
    return [i for i, tr in enumerate(mid.tracks)
            if any(m.type == 'note_on' and m.velocity > 0 for m in tr)]


def first_onset(track):
    t = 0
    for m in track:
        t += m.time
        if m.type == 'note_on' and m.velocity > 0:
            return t
    return None


def last_tick(mid):
    return max((sum(m.time for m in tr) for tr in mid.tracks), default=0)


def meters(mid):
    sigs = {(m.numerator, m.denominator) for tr in mid.tracks for m in tr
            if m.type == 'time_signature'}
    return sigs or {(4, 4)}


def keep_one_stream(mid, keep_idx):
    """A copy of `mid` whose only note-bearing track is keep_idx; the other
    note track is dropped, meta-only tracks are kept."""
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    notes = set(note_tracks(mid))
    for i, tr in enumerate(mid.tracks):
        if i == keep_idx or i not in notes:
            out.tracks.append(tr)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True,
                   help='folder that directly holds the Nottingham .mid files')
    p.add_argument('--dst', required=True)
    p.add_argument('--dataset', default='nottingham_melody_cp4_v2',
                   help='training manifest that defines the held-out split')
    p.add_argument('--split-ratio', type=int, default=10)
    p.add_argument('--train-length', type=int, default=TRAIN_LENGTH)
    p.add_argument('--lead-bars', type=int, default=2,
                   help='bars in front of the first full downbeat (2: frame 32)')
    p.add_argument('--min-frames', type=int, default=416,
                   help='keep a tune only if the cropped melody reaches this '
                        'many frames, so the reference covers the whole '
                        'generated span (GEN_LENGTH)')
    p.add_argument('--meters', nargs='+', default=['4/4', '2/2'],
                   help='keep tunes whose every time signature is listed')
    p.add_argument('--tempo', type=float, default=120.0)
    p.add_argument('--tunes', nargs='*', default=None,
                   help='use these tunes instead of the held-out split')
    a = p.parse_args()

    allowed = {tuple(int(x) for x in s.split('/')) for s in a.meters}
    tunes = a.tunes or heldout_tunes(a.dataset, a.split_ratio, a.train_length)
    mel_out, chd_out = os.path.join(a.dst, 'melody'), os.path.join(a.dst, 'chord')
    for d in (mel_out, chd_out):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    rows, kept = [], 0
    tmp = tempfile.mkdtemp()
    for tune in tunes:
        src = os.path.join(a.src, f'{tune}.mid')
        row = {'tune': tune}
        mid = mido.MidiFile(src)
        sig = meters(mid)
        row['meter'] = ' '.join(f'{n}/{d}' for n, d in sorted(sig))
        nt = note_tracks(mid)
        if len(nt) < 2:
            rows.append({**row, 'status': 'drop: no chord track'})
            continue
        if not sig <= allowed:
            rows.append({**row, 'status': 'drop: metre not in --meters'})
            continue
        tpb = mid.ticks_per_beat
        n, d = next(iter(sig))
        bar = tpb * 4 * n // d
        m0 = first_onset(mid.tracks[nt[0]])
        c0 = first_onset(mid.tracks[nt[1]])
        phase = (c0 % bar) if c0 is not None else 0
        # first grid line at or after the first melody note
        down = phase + -(-(m0 - phase) // bar) * bar
        lead = a.lead_bars * bar
        tick0, pad = max(0, down - lead), max(0, lead - down)
        frames = (last_tick(mid) - tick0 + pad) * FRAMES_PER_BEAT / tpb
        row.update(pickup=int(down > m0), mel_onset_beats=m0 / tpb,
                   chord_onset_beats=(c0 / tpb if c0 is not None else ''),
                   downbeat_beats=down / tpb, crop_tick=tick0, pad_tick=pad,
                   frames=round(frames))
        if frames < a.min_frames:
            rows.append({**row, 'status': f'drop: {frames:.0f} < {a.min_frames} frames'})
            continue
        for idx, out_dir in ((nt[0], mel_out), (nt[1], chd_out)):
            one = os.path.join(tmp, f'{tune}.mid')
            keep_one_stream(mid, idx).save(one)
            crop_ticks(one, os.path.join(out_dir, f'{tune}.mid'), tick0,
                       pad_ticks=pad, force_bpm=a.tempo or None)
        rows.append({**row, 'status': 'kept'})
        kept += 1
    shutil.rmtree(tmp, ignore_errors=True)

    cols = ['tune', 'status', 'meter', 'pickup', 'mel_onset_beats',
            'chord_onset_beats', 'downbeat_beats', 'crop_tick', 'pad_tick', 'frames']
    with open(os.path.join(a.dst, 'prompt_crops.tsv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    dropped = {}
    for r in rows:
        if r['status'] != 'kept':
            k = r['status'].split(':', 1)[1].strip()
            k = 'shorter than --min-frames' if '<' in k else k
            dropped[k] = dropped.get(k, 0) + 1
    print(f'[prompts] {kept} of {len(tunes)} tunes written to {a.dst}/{{melody,chord}}; '
          f'dropped: {dropped or "none"}')
    if not kept:
        raise SystemExit('no tune survived the filters')


if __name__ == '__main__':
    main()
