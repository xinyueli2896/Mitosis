"""Carve a test set out of an already-preprocessed paired tensor pack.

You have these in --data_dir (defaults match make_la_drum_other.py --naming compat):
    {prefix_drum}.pt                  e.g. la_melody_cp4_v2.pt
    {prefix_drum}.length.pt
    {prefix_drum}.pitch_shift_range.pt
    {prefix_drum}.txt                 lines like: "<idx>\\t<relpath_under_midi_root>"
    {prefix_other}.pt                 e.g. la_chord_cp4_v2.pt
    {prefix_other}.length.pt
    {prefix_other}.pitch_shift_range.pt
    {prefix_other}.txt

This script picks N songs to hold out, copies their raw .mid files from
--midi_root to --output_dir, and rewrites the four tensor pack files in
place (originals saved as .bak) WITHOUT those songs' rows.

Run:
    python holdout_test_midis.py \\
        --data_dir data/ \\
        --prefix_drum la_melody_cp4_v2 \\
        --prefix_other la_chord_cp4_v2 \\
        --midi_root /path/to/Los-Angeles-MIDI-Dataset/MIDIs \\
        --output_dir input/lamd_test_prompts \\
        --num_test 50 \\
        --seed 0
"""

import argparse
import os
import random
import shutil

import pretty_midi
import torch

from preprocess_large_midi_dataset import DURATION_TEMPLATES


def _render_tensor_to_midi(drum_rows, other_rows, save_path,
                            max_polyphony, tempo=120.0, beat_div=4):
    """Decode raw pack rows back into a .mid file.

    drum_rows, other_rows: uint8 tensors of shape [T, max_polyphony * 4],
    each row a sequence of (program, pitch, duration_idx, velocity) quads.
    program == 254 is the per-timestep EOS sentinel; 255 is the empty-slot
    fill. Drum half uses program == 127 for drums.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    step = 60.0 / tempo / beat_div
    instrument_map = {}

    def add_stream(rows):
        if rows.numel() == 0:
            return
        arr = rows.view(rows.shape[0], max_polyphony, 4).cpu().numpy()
        for t in range(arr.shape[0]):
            for k in range(max_polyphony):
                program = int(arr[t, k, 0])
                if program == 254 or program == 255:
                    break
                pitch = int(arr[t, k, 1])
                dur_idx = int(arr[t, k, 2])
                velocity = int(arr[t, k, 3]) or 100
                if dur_idx < 0 or dur_idx >= len(DURATION_TEMPLATES):
                    continue
                if pitch < 0 or pitch >= 128:
                    continue
                start = t * step
                end = start + float(DURATION_TEMPLATES[dur_idx]) * step
                if program == 127:
                    key = ('drum',)
                    if key not in instrument_map:
                        instrument_map[key] = pretty_midi.Instrument(0, is_drum=True)
                        midi.instruments.append(instrument_map[key])
                else:
                    key = ('inst', program)
                    if key not in instrument_map:
                        instrument_map[key] = pretty_midi.Instrument(program)
                        midi.instruments.append(instrument_map[key])
                instrument_map[key].notes.append(
                    pretty_midi.Note(velocity=velocity, pitch=pitch,
                                     start=start, end=end))

    add_stream(drum_rows)
    add_stream(other_rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def _load_pack(data_dir, prefix):
    p = os.path.join(data_dir, prefix)
    rolls = torch.load(p + '.pt')
    lengths = torch.load(p + '.length.pt')
    psr = torch.load(p + '.pitch_shift_range.pt')
    with open(p + '.txt') as f:
        manifest = [line.rstrip('\n').split('\t', 1)[1] for line in f if line.strip()]
    return rolls, lengths, psr, manifest


def _save_pack(data_dir, prefix, rolls, lengths, psr, manifest, backup=True):
    p = os.path.join(data_dir, prefix)
    for suffix in ('.pt', '.length.pt', '.pitch_shift_range.pt', '.txt'):
        src = p + suffix
        if backup and os.path.exists(src) and not os.path.exists(src + '.bak'):
            os.rename(src, src + '.bak')
    torch.save(rolls, p + '.pt')
    torch.save(lengths, p + '.length.pt')
    torch.save(psr, p + '.pitch_shift_range.pt')
    with open(p + '.txt', 'w') as f:
        for i, fn in enumerate(manifest):
            f.write(f'{i}\t{fn}\n')


def _split(rolls, lengths, psr, manifest, keep_idx, drop_idx):
    """Return (rolls_keep, lengths_keep, psr_keep, manifest_keep)."""
    # Per-song row offsets via cumsum.
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths.to(torch.long), dim=0)

    keep_rows = []
    for i in keep_idx:
        keep_rows.append(rolls[offsets[i]:offsets[i + 1]])
    rolls_keep = torch.cat(keep_rows, dim=0) if keep_rows else rolls.new_zeros((0, rolls.shape[1]))

    lengths_keep = lengths[torch.tensor(keep_idx, dtype=torch.long)]
    psr_keep = psr[torch.tensor(keep_idx, dtype=torch.long)]
    manifest_keep = [manifest[i] for i in keep_idx]
    return rolls_keep, lengths_keep, psr_keep, manifest_keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--prefix_drum', default='la_melody_cp4_v2',
                    help='Prefix of the drum-stream pack (compat naming uses "la_melody_cpN_v2").')
    ap.add_argument('--prefix_other', default='la_chord_cp4_v2',
                    help='Prefix of the non-drum pack (compat naming uses "la_chord_cpN_v2").')
    ap.add_argument('--midi_root', default=None,
                    help='Root the .txt manifest paths are relative to (same as '
                         '--midi_root passed to make_la_drum_other.py). If '
                         'given, held-out raw .mid files are copied from here. '
                         'If omitted, the script renders each held-out song '
                         'back to .mid directly from its tensor rows -- no '
                         'access to the original LAMD MIDIs required.')
    ap.add_argument('--render_from_tensor', action='store_true',
                    help='Force tensor rendering even when --midi_root is given.')
    ap.add_argument('--tempo', type=float, default=120.0,
                    help='Tempo to assume when rendering tensors back to MIDI. '
                         'Only the absolute timing changes; the score itself is '
                         'tempo-agnostic.')
    ap.add_argument('--beat_div', type=int, default=4,
                    help='Subbeats per beat used in preprocess_midi. Must match '
                         'whatever you ran make_la_drum_other.py with (default 4).')
    ap.add_argument('--output_dir', default='input/lamd_test_prompts',
                    help='Where to copy the held-out .mid files.')
    ap.add_argument('--num_test', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry_run', action='store_true',
                    help='Copy the held-out MIDIs to --output_dir and write '
                         '_heldout_manifest.txt, but do NOT rewrite the tensor '
                         'pack files. Lets you preview the test set before '
                         'committing to the trim.')
    ap.add_argument('--no_backup', dest='backup', action='store_false', default=True,
                    help='Skip writing .bak copies of the original pack files.')
    args = ap.parse_args()

    print(f'[load] {args.data_dir}/{args.prefix_drum}.*')
    rolls_d, lengths_d, psr_d, manifest_d = _load_pack(args.data_dir, args.prefix_drum)
    print(f'[load] {args.data_dir}/{args.prefix_other}.*')
    rolls_o, lengths_o, psr_o, manifest_o = _load_pack(args.data_dir, args.prefix_other)

    n = len(manifest_d)
    assert len(manifest_o) == n, (
        f'manifest mismatch: drum has {n} songs, other has {len(manifest_o)} -- '
        'the two packs are not paired.')
    assert manifest_d == manifest_o, (
        'manifests differ song-by-song; the two packs were not produced together. '
        'Refusing to proceed.')
    assert torch.equal(lengths_d, lengths_o), (
        'per-song lengths differ between drum and other packs; they are not paired.')
    print(f'[ok] paired pack with {n:,} songs, '
          f'drum rolls {tuple(rolls_d.shape)}, other rolls {tuple(rolls_o.shape)}')

    if args.num_test >= n:
        raise SystemExit(f'--num_test ({args.num_test}) >= total songs ({n}).')

    rng = random.Random(args.seed)
    test_idx = sorted(rng.sample(range(n), args.num_test))
    test_set = set(test_idx)
    keep_idx = [i for i in range(n) if i not in test_set]
    print(f'[pick] held out {len(test_idx)} songs, keeping {len(keep_idx)} for training')

    # Always materialize the output dir + manifest of held-out songs.
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, '_heldout_manifest.txt'), 'w') as f:
        for i in test_idx:
            f.write(f'{i}\t{manifest_d[i]}\n')

    # Per-song row offsets via cumsum (reused for both copy and render paths).
    offsets = torch.zeros(len(lengths_d) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths_d.to(torch.long), dim=0)
    P = rolls_d.shape[1] // 4

    if args.midi_root is not None and not args.render_from_tensor:
        missing = []
        for i in test_idx:
            rel = manifest_d[i]
            src = os.path.join(args.midi_root, rel)
            dst = os.path.join(args.output_dir, rel)
            if not os.path.exists(src):
                missing.append(rel)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        if missing:
            print(f'[warn] {len(missing)} test MIDIs were not found under {args.midi_root}; '
                  f'first few: {missing[:3]}')
        print(f'[copy] {len(test_idx) - len(missing)} raw MIDIs -> {args.output_dir}')
    else:
        # Render each held-out song straight from its tensor rows.
        for i in test_idx:
            drum_rows = rolls_d[offsets[i]:offsets[i + 1]]
            other_rows = rolls_o[offsets[i]:offsets[i + 1]]
            stem = os.path.splitext(os.path.basename(manifest_d[i]))[0]
            dst = os.path.join(args.output_dir,
                               f'song{i:06d}_{stem}.mid')
            _render_tensor_to_midi(drum_rows, other_rows, dst,
                                    max_polyphony=P, tempo=args.tempo,
                                    beat_div=args.beat_div)
        print(f'[render] {len(test_idx)} songs decoded from tensors -> '
              f'{args.output_dir} (tempo={args.tempo}, P={P})')

    # Rebuild train tensors.
    rolls_d2, lengths_d2, psr_d2, manifest_d2 = _split(rolls_d, lengths_d, psr_d, manifest_d, keep_idx, test_idx)
    rolls_o2, lengths_o2, psr_o2, manifest_o2 = _split(rolls_o, lengths_o, psr_o, manifest_o, keep_idx, test_idx)
    print(f'[trim] drum  rolls {tuple(rolls_d.shape)} -> {tuple(rolls_d2.shape)}')
    print(f'[trim] other rolls {tuple(rolls_o.shape)} -> {tuple(rolls_o2.shape)}')

    if args.dry_run:
        print(f'[dry_run] MIDIs copied to {args.output_dir}; tensor packs left '
              f'untouched. Re-run without --dry_run to rewrite them.')
        return

    _save_pack(args.data_dir, args.prefix_drum, rolls_d2, lengths_d2, psr_d2, manifest_d2, backup=args.backup)
    _save_pack(args.data_dir, args.prefix_other, rolls_o2, lengths_o2, psr_o2, manifest_o2, backup=args.backup)
    print('[done] pack files rewritten; originals saved as *.bak'
          if args.backup else '[done] pack files rewritten (no backup).')


if __name__ == '__main__':
    main()
