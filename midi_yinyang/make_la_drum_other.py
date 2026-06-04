"""Preprocess the Los Angeles MIDI Dataset into paired drum + non-drum tensors.

Produces two .pt files following the same shape contract as
pop909_melody_cp4_v2.pt / pop909_chord_cp4_v2.pt, so the existing m2c
training scripts (cp_transformer_m2c_*.py) can consume them with zero or
minimal code changes.

Steps:
  1. Walk --midi_root recursively, collect all .mid / .MID files.
  2. Sort deterministically so song index i refers to the same MIDI in
     both output files (paired-stream training breaks otherwise).
  3. For each MIDI, call preprocess_midi(..., ins_ids=['drum', 'nondrum']).
     This returns a concatenated tensor of shape
     [T, 2 * max_polyphony * 4] (drum half + non-drum half). If EITHER
     side has no notes, preprocess_midi returns None and we skip the
     song -- ensuring both output files cover EXACTLY the same set of
     songs in the same order.
  4. Split each result into drum and non-drum halves and accumulate.
  5. Save:
       <prefix_m>.pt                       (drum stream)
       <prefix_m>.length.pt
       <prefix_m>.pitch_shift_range.pt
       <prefix_m>.txt                      (midi filename per song)
       <prefix_c>.pt                       (non-drum stream)
       <prefix_c>.length.pt
       <prefix_c>.pitch_shift_range.pt
       <prefix_c>.txt

Run:
    cd midi_yinyang
    python make_la_drum_other.py \\
        --midi_root /path/to/Los-Angeles-MIDI-Dataset/MIDIs \\
        --output_dir data/ \\
        --max_polyphony 4 \\
        --max_idx 50000 \\
        --dedup

After it finishes you can delete the raw LAMD MIDIs folder; only the
~hundreds of MB of .pt files in data/ are needed for training.
"""

import argparse
import os

import numpy as np
import torch
from joblib import Parallel, delayed

# Reuse the existing preprocess_midi pipeline -- it knows how to handle
# 'drum' and 'nondrum' as ins_ids, applies the duration template, returns
# (rolls, pitch_shift_range) per MIDI. Returns None if either side empty.
from preprocess_large_midi_dataset import preprocess_midi


def _process_one(midi_path, max_polyphony, dedup):
    """Returns (drum, other, pitch_shift_range) or None if the MIDI doesn't
    have both drum and non-drum content."""
    try:
        result = preprocess_midi(
            midi_path, max_polyphony,
            ins_ids=['drum', 'nondrum'],
            dedup=dedup,
            filter=True,
        )
    except Exception:
        return None
    if result is None:
        return None
    rolls, pitch_shift = result
    # rolls shape: [T, 2 * max_polyphony * 4]. Drum is the first
    # max_polyphony * 4 columns, non-drum is the next block.
    half = max_polyphony * 4
    drum = rolls[:, :half].contiguous()
    other = rolls[:, half:].contiguous()
    return drum, other, pitch_shift


def main():
    ap = argparse.ArgumentParser(
        description='Build paired drum / non-drum .pt tensors from the LA '
                    'MIDI Dataset.',
    )
    ap.add_argument('--midi_root', required=True,
                    help='Path to the folder containing the LAMD MIDIs '
                         '(typically <root>/MIDIs).')
    ap.add_argument('--output_dir', default='data',
                    help='Where the .pt files go. Default: data/')
    ap.add_argument('--max_polyphony', type=int, default=4,
                    help='Max simultaneous notes per timestep per stream. '
                         '4 matches the existing pop909_cp4_v2 setup.')
    ap.add_argument('--max_idx', type=int, default=None,
                    help='Cap on number of MIDIs to process (after sort). '
                         'Default: no cap (process all MIDIs in --midi_root). '
                         'Pass --max_idx 50000 to match the existing recipe '
                         'in preprocess_large_midi_dataset.create_la_drums().')
    ap.add_argument('--dedup', action='store_true', default=True,
                    help='Per-track note dedup (matches existing recipe).')
    ap.add_argument('--no_dedup', dest='dedup', action='store_false')
    ap.add_argument(
        '--naming', choices=['compat', 'clean'], default='compat',
        help='compat (default): la_melody_cp{P}_v2 = drum, '
             'la_chord_cp{P}_v2 = non-drum. Lets the existing FramedDataset '
             '"chord -> melody" filename rule pair them with zero code '
             'changes; semantically misleading but plug-and-play.\n'
             'clean: la_drum_cp{P}_v2 / la_other_cp{P}_v2. Honest names, '
             'requires a small FramedDataset edit (parameterize the '
             'partner-substring pair) before training.',
    )
    ap.add_argument('--n_jobs', type=int, default=-1,
                    help='Parallel workers. -1 = all cores.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect MIDIs deterministically.
    print(f'[scan] {args.midi_root}')
    midi_files = []
    for root, _, files in os.walk(args.midi_root):
        for f in files:
            if f.endswith(('.mid', '.MID', '.midi', '.MIDI')):
                midi_files.append(os.path.join(root, f))
    midi_files.sort()
    if args.max_idx is not None:
        midi_files = midi_files[:args.max_idx]
    print(f'[scan] {len(midi_files):,} MIDIs (after sort + cap)')

    # 2. Parallel preprocess.
    print(f'[preprocess] parallel with n_jobs={args.n_jobs}, '
          f'max_polyphony={args.max_polyphony}, dedup={args.dedup}')
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_process_one)(p, args.max_polyphony, args.dedup)
        for p in midi_files
    )

    # 3. Filter None (MIDIs missing either drum or non-drum content) and
    #    keep paired song-name list so song index i corresponds to the
    #    SAME source MIDI in both output streams.
    kept_pairs = [
        (os.path.relpath(midi_files[i], args.midi_root), r)
        for i, r in enumerate(results) if r is not None
    ]
    print(f'[filter] {len(kept_pairs):,} of {len(midi_files):,} MIDIs kept '
          f'(rest had empty drum or non-drum streams)')

    if len(kept_pairs) == 0:
        raise SystemExit('No MIDIs passed the filter; check --midi_root.')

    drum_tensors = [p[1][0] for p in kept_pairs]
    other_tensors = [p[1][1] for p in kept_pairs]
    pitch_shifts = [p[1][2] for p in kept_pairs]
    filenames = [p[0] for p in kept_pairs]

    # 4. Build outputs.
    drum_data = torch.cat(drum_tensors, dim=0)
    other_data = torch.cat(other_tensors, dim=0)
    pitch_shift_data = torch.stack(pitch_shifts, dim=0)
    drum_lengths = torch.tensor([t.shape[0] for t in drum_tensors],
                                 dtype=torch.long)
    # drum and non-drum have the same per-song length here (same T from the
    # single preprocess_midi call). Save both length files anyway for
    # symmetry with the pop909 layout that FramedDataset expects.
    other_lengths = drum_lengths.clone()

    P = args.max_polyphony
    if args.naming == 'compat':
        prefix_drum = os.path.join(args.output_dir, f'la_melody_cp{P}_v2')
        prefix_other = os.path.join(args.output_dir, f'la_chord_cp{P}_v2')
        readme_note = (
            'naming=compat: "melody" file = DRUM stream, "chord" file = '
            'NON-DRUM (other) stream. The misleading names let the existing '
            'FramedDataset "chord -> melody" filename rule pair them with '
            'zero code changes.'
        )
    else:
        prefix_drum = os.path.join(args.output_dir, f'la_drum_cp{P}_v2')
        prefix_other = os.path.join(args.output_dir, f'la_other_cp{P}_v2')
        readme_note = (
            'naming=clean: la_drum / la_other. Requires a small FramedDataset '
            'edit (parameterize the partner-substring pair, currently '
            'hard-coded as "chord" -> "melody") before training.'
        )

    print(f'[save] drum  stream -> {prefix_drum}.pt '
          f'({drum_data.shape[0]:,} rows x {drum_data.shape[1]} cols)')
    torch.save(drum_data, prefix_drum + '.pt')
    torch.save(drum_lengths, prefix_drum + '.length.pt')
    torch.save(pitch_shift_data, prefix_drum + '.pitch_shift_range.pt')
    with open(prefix_drum + '.txt', 'w') as f:
        for i, fn in enumerate(filenames):
            f.write(f'{i}\t{fn}\n')

    print(f'[save] other stream -> {prefix_other}.pt '
          f'({other_data.shape[0]:,} rows x {other_data.shape[1]} cols)')
    torch.save(other_data, prefix_other + '.pt')
    torch.save(other_lengths, prefix_other + '.length.pt')
    torch.save(pitch_shift_data, prefix_other + '.pitch_shift_range.pt')
    with open(prefix_other + '.txt', 'w') as f:
        for i, fn in enumerate(filenames):
            f.write(f'{i}\t{fn}\n')

    print(f'\n[done] {readme_note}')


if __name__ == '__main__':
    main()
