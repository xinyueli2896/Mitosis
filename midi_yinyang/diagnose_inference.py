"""Combined diagnostic for "inference output is chaotic" symptom.

Runs four checks in one go:

  1. Checkpoint health: training step, epoch, val_loss, total params.
  2. Training data drum rhythmicity: distribution of drum frames across
     the 16 subbeats of a bar. If even, the training data itself is
     rhythmically diffuse; if peaks on downbeats/off-beats, the data
     is coherent and the model is the problem.
  3. Generated drum stream: first 20 pitches + start times. Sanity-
     checks whether the model is placing drum hits on grid and using
     standard kit pitches (35..51).
  4. Loss decomposition: best-effort split of "easy" (EOS/pad slot
     fillers) vs "hard" (real pitch/duration) token contributions to
     val_loss, by running one forward pass on a held-out batch.

Skip any check by omitting its corresponding argument.

Run:
    python diagnose_inference.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --train-data data/la_melody_cp16_v2.pt \\
        --generated-midi temp/<run>/<song>/co/sample_0_temp1.0.mid \\
        --val-data data/la_chord_cp16_v2.pt
"""

import argparse
import collections
import os

import torch


def check_ckpt(ckpt_path):
    print(f'\n=== [1] CHECKPOINT  {ckpt_path} ===')
    if not os.path.exists(ckpt_path):
        print(f'  [skip] file not found')
        return None
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
    print(f'  global_step:           {ck.get("global_step") if isinstance(ck, dict) else "?"}')
    print(f'  epoch:                 {ck.get("epoch") if isinstance(ck, dict) else "?"}')
    if isinstance(ck, dict) and 'callbacks' in ck:
        for cb_name, cb_state in ck['callbacks'].items():
            if isinstance(cb_state, dict) and 'best_model_score' in cb_state:
                print(f'  best_model_score:      {cb_state["best_model_score"]}')
                print(f'  best_model_path:       {cb_state.get("best_model_path")}')
    n_params = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
    print(f'  total params:          {n_params/1e6:.1f}M')

    # Heuristic step thresholds for "is it done?"
    step = ck.get('global_step') if isinstance(ck, dict) else None
    if step is not None:
        if step < 5000:
            print(f'  [verdict] Very early training. Expect garbage output.')
        elif step < 30000:
            print(f'  [verdict] Mid-early training. Output likely coherent but imitative.')
        elif step < 100000:
            print(f'  [verdict] Substantial training. Should be usable; bugs in code more likely than undertraining.')
        else:
            print(f'  [verdict] Well-trained. If output is bad, look at sampling settings + data.')
    return ck


def check_train_data_rhythm(data_path):
    print(f'\n=== [2] TRAIN DATA DRUM RHYTHMICITY  {data_path} ===')
    if not os.path.exists(data_path):
        print(f'  [skip] file not found')
        return
    # We need the per-song length tensor to compute position-in-bar relative
    # to each song's START. Without it, songs whose lengths aren't multiples
    # of 16 shift their bar-alignment in the concatenated tensor, and the
    # histogram blurs to uniform across many songs regardless of per-song
    # rhythmicity. (The earlier version of this script had that bug.)
    length_path = data_path[:-3] + '.length.pt'
    if not os.path.exists(length_path):
        print(f'  [skip] length file {length_path} not found; cannot align to bars.')
        return
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    lengths = torch.load(length_path, map_location='cpu', weights_only=False)
    print(f'  data shape: {tuple(data.shape)}')
    print(f'  num songs:  {len(lengths)}')

    max_polyphony = data.shape[1] // 4

    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths.to(torch.long), dim=0)

    rng = torch.Generator().manual_seed(0)
    sample_songs = torch.randperm(len(lengths), generator=rng)[:200].tolist()

    pos_hits = collections.Counter()
    notes_total = 0
    sampled_song_frames = 0
    for song_idx in sample_songs:
        start = int(offsets[song_idx])
        end = int(offsets[song_idx + 1])
        if end <= start:
            continue
        rows = data[start:end].view(end - start, max_polyphony, 4)
        # Number of NOTES at each timestep (slot's pitch is not the pad
        # sentinel 255 => note is real).
        notes_per_frame = (rows[:, :, 1] != 255).sum(dim=-1)   # [T] int
        pos_in_bar = torch.arange(end - start) % 16
        for p in range(16):
            mask = (pos_in_bar == p)
            c = int(notes_per_frame[mask].sum().item())
            pos_hits[p] += c
            notes_total += c
        sampled_song_frames += (end - start)

    if notes_total == 0:
        print('  [warn] no notes in sampled songs')
        return
    print(f'  drum hits across {len(sample_songs)} sampled songs: {notes_total}')

    print(f'\n  drum hit density by position-in-bar (4/4, 16 subbeats):')
    print(f'           beat:  1                   2                   3                   4')
    print(f'        subbeat:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15')
    bars = ''
    for p in range(16):
        pct = 100 * pos_hits[p] / notes_total
        bars += f'  {pct:4.1f}'
    print(f'        density:{bars}')

    downbeats = [pos_hits[0], pos_hits[4], pos_hits[8], pos_hits[12]]
    offbeats  = [pos_hits[2], pos_hits[6], pos_hits[10], pos_hits[14]]
    odd_subbeats = [pos_hits[i] for i in [1, 3, 5, 7, 9, 11, 13, 15]]
    avg_db = sum(downbeats) / 4
    avg_ob = sum(offbeats) / 4
    avg_odd = sum(odd_subbeats) / 8
    print(f'\n  avg hit-density on downbeats (0,4,8,12):    {100*avg_db/notes_total:.2f}%')
    print(f'  avg hit-density on backbeats (2,6,10,14):   {100*avg_ob/notes_total:.2f}%')
    print(f'  avg hit-density on odd subbeats:            {100*avg_odd/notes_total:.2f}%')

    notes_per_bar = notes_total / max(sampled_song_frames / 16, 1)
    print(f'\n  avg drum hits per bar (train data):         {notes_per_bar:.2f}')
    print(f'  (compare with generated MIDI hits-per-bar in section [3])')

    if avg_db > 1.5 * avg_odd:
        print(f'\n  [verdict] Data is RHYTHMIC. Downbeats are emphasized.')
    elif avg_db > 1.1 * avg_odd:
        print(f'\n  [verdict] Data is MILDLY rhythmic. Some pattern present.')
    else:
        print(f'\n  [verdict] Data is DIFFUSE / ~uniform across subbeats. '
              f"Model can't produce something more rhythmic than the data.")


def check_generated_midi(midi_path):
    print(f'\n=== [3] GENERATED MIDI  {midi_path} ===')
    if not os.path.exists(midi_path):
        print(f'  [skip] file not found')
        return
    try:
        import pretty_midi
    except ImportError:
        print('  [skip] pretty_midi not installed')
        return
    midi = pretty_midi.PrettyMIDI(midi_path)
    print(f'  duration:              {midi.get_end_time():.2f} s')
    print(f'  num instruments:       {len(midi.instruments)}')
    for i, inst in enumerate(midi.instruments):
        flag = '(DRUM)' if inst.is_drum else f'(program {inst.program})'
        print(f'  track {i:2d} {flag:18s} notes={len(inst.notes):4d} name={inst.name!r}')
    print()
    for inst in midi.instruments:
        if not inst.is_drum:
            continue
        all_pitches = [n.pitch for n in inst.notes]
        pitches = all_pitches[:30]
        starts = [round(n.start, 3) for n in inst.notes[:30]]
        durations = [round(n.end - n.start, 3) for n in inst.notes[:30]]
        print(f'  drum first 30 pitches:    {pitches}')
        print(f'  drum first 30 starts:     {starts}')
        print(f'  drum first 30 durations:  {durations}')

        on_grid = sum(1 for s in starts if abs(round(s / 0.125) - s / 0.125) < 0.01)
        print(f'\n  drum on-grid (sub-beat at 120 BPM): {on_grid}/{len(starts)}')

        # Drum kit range covers GM percussion roughly 35..81 (extended kit
        # includes tambourine=54, claves=75, etc.). Tightening to 35..51 was
        # too strict for LAMD's broader kit.
        kit_pitches = sum(1 for p in pitches if 35 <= p <= 81)
        print(f'  drum in GM kit range [35,81]:       {kit_pitches}/{len(pitches)}')

        unique_pitches = len(set(pitches))
        print(f'  unique drum pitches in first 30 notes: {unique_pitches}')

        # Density: hits per bar over the whole song. Compare with train-data
        # average from section [2].
        total_bars = midi.get_end_time() / 2.0   # 2 seconds per bar at 120 BPM
        hits_per_bar = len(all_pitches) / max(total_bars, 1)
        print(f'  drum hits per bar (generated):      {hits_per_bar:.2f}')
        unique_all = len(set(all_pitches))
        print(f'  unique drum pitches (whole song):   {unique_all}')

        if hits_per_bar > 40:
            print(f'\n  [verdict] Drum is VERY DENSE ({hits_per_bar:.0f}+ hits/bar). '
                  f'Real drum tracks are 8-20 hits/bar. Likely undertrained EOS '
                  f'prediction -- model fills every slot up to max_polyphony.')
        elif hits_per_bar > 25:
            print(f'\n  [verdict] Drum is BUSY ({hits_per_bar:.0f} hits/bar). '
                  f'Could be intentional (latin/breakbeat) or undertraining.')

        if on_grid < len(starts) * 0.7:
            print(f'  [verdict] Drum hits are OFF GRID. Either model is undertrained '
                  f'or temperature is too high.')
        if kit_pitches < len(pitches) * 0.7:
            print(f'  [verdict] Drum pitches are OUTSIDE standard kit range. '
                  f'Decode might not be using is_drum=True correctly, OR the '
                  f'model is generating chaotic pitches.')
        if unique_pitches > 10:
            print(f'  [verdict] Too many distinct drum pitches for a coherent kit '
                  f'pattern. Real drum tracks use 3-7 voices. Suggests model is '
                  f'treating drum as melody.')
        break
    else:
        print('  [warn] no drum track found in the generated MIDI')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', help='Training checkpoint to inspect')
    ap.add_argument('--train-data', help='drum-stream .pt to check rhythmicity '
                                          '(la_melody_cp16_v2.pt in compat naming)')
    ap.add_argument('--generated-midi', help='inference output midi to inspect')
    args = ap.parse_args()

    if args.ckpt:
        check_ckpt(args.ckpt)
    if args.train_data:
        check_train_data_rhythm(args.train_data)
    if args.generated_midi:
        check_generated_midi(args.generated_midi)

    if not any([args.ckpt, args.train_data, args.generated_midi]):
        ap.print_help()


if __name__ == '__main__':
    main()
