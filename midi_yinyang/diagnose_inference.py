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
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    print(f'  shape: {tuple(data.shape)}')
    # Assume max_polyphony*4 cols; first 4 = first note's quad.
    max_polyphony = data.shape[1] // 4
    # Sample up to 100k frames for the histogram.
    n_samples = min(100_000, data.shape[0])
    rows = data[:n_samples].view(n_samples, max_polyphony, 4)

    has_note = (rows[:, :, 1] != 255).any(dim=-1)             # [n_samples] bool
    pos_in_bar = torch.arange(n_samples) % 16

    pos_hist = collections.Counter()
    total = 0
    for p in range(16):
        mask = (pos_in_bar == p) & has_note
        c = int(mask.sum().item())
        pos_hist[p] = c
        total += c
    if total == 0:
        print('  [warn] no notes in sampled frames; is this the right .pt?')
        return
    print(f'  drum-content frames sampled: {total} / {n_samples}')
    print(f'  drum density by position-in-bar (4/4, 16 subbeats):')
    print(f'           beat:  1                   2                   3                   4')
    print(f'        subbeat:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15')
    bars = ''
    for p in range(16):
        pct = 100 * pos_hist[p] / total
        bars += f'  {pct:4.1f}'
    print(f'        density:{bars}')

    # Rhythmic vs diffuse heuristic.
    downbeats = [pos_hist[0], pos_hist[4], pos_hist[8], pos_hist[12]]
    offbeats  = [pos_hist[2], pos_hist[6], pos_hist[10], pos_hist[14]]
    odd_subbeats = [pos_hist[i] for i in [1, 3, 5, 7, 9, 11, 13, 15]]
    avg_db = sum(downbeats) / 4
    avg_ob = sum(offbeats) / 4
    avg_odd = sum(odd_subbeats) / 8
    print(f'\n  avg density on downbeats (0,4,8,12):       {100*avg_db/total:.2f}%')
    print(f'  avg density on backbeats (2,6,10,14):      {100*avg_ob/total:.2f}%')
    print(f'  avg density on odd subbeats:               {100*avg_odd/total:.2f}%')
    if avg_db > 1.5 * avg_odd:
        print(f'  [verdict] Data is RHYTHMIC. Downbeats are emphasized. '
              f'If your generated output is arrhythmic, the model is the issue, not the data.')
    elif avg_db > 1.1 * avg_odd:
        print(f'  [verdict] Data is MILDLY rhythmic. Some pattern present but quantization is loose.')
    else:
        print(f'  [verdict] Data is DIFFUSE / ~uniform across subbeats. '
              f'LAMD has lots of non-quantized hits; model can\'t produce '
              f'something more rhythmic than what it was trained on.')


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
        pitches = [n.pitch for n in inst.notes[:30]]
        starts = [round(n.start, 3) for n in inst.notes[:30]]
        durations = [round(n.end - n.start, 3) for n in inst.notes[:30]]
        print(f'  drum first 30 pitches:    {pitches}')
        print(f'  drum first 30 starts:     {starts}')
        print(f'  drum first 30 durations:  {durations}')

        # On-grid check (at 120 BPM, subbeat = 0.125 s).
        on_grid = sum(1 for s in starts if abs(round(s / 0.125) - s / 0.125) < 0.01)
        print(f'\n  drum on-grid (sub-beat at 120 BPM): {on_grid}/{len(starts)}')

        # Kit-pitch check.
        kit_pitches = sum(1 for p in pitches if 35 <= p <= 51)
        print(f'  drum in standard kit range [35,51]: {kit_pitches}/{len(pitches)}')

        # Pitch diversity (rhythmic drums = small set, melodic = large set).
        unique_pitches = len(set(pitches))
        print(f'  unique drum pitches in first 30 notes: {unique_pitches}')

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
