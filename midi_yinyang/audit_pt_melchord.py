"""Listening audit for paired melody/chord .pt tokenizations.

Decodes sampled songs from the tokenized tensors BACK to midi so the
tokenization can be verified by ear against the source files. Per
sampled song, writes into <out-dir>/<source-file-stem>/:

    melody.mid     the melody stream alone
    chord.mid      the chord stream alone
    combined.mid   both streams together (chord remapped to
                   --chord-program so the parts are audibly and
                   visually separate even when both were program 0)

and prints the per-stream notes-per-frame distribution — the
orientation check: the melody stream must be 0-1 notes/frame; 3-4-note
frames appearing in the melody file mean some source midi had its
tracks in reversed order.

Usage (Nottingham):
    python audit_pt_melchord.py \
        --mel-prefix data/nottingham_melody_cp4_v2 \
        --chord-prefix data/nottingham_chord_cp4_v2 \
        --out-dir temp/nottingham_audit --n-songs 5

Works for any paired dataset in this format (e.g. the POP909 pair).
"""

import argparse
import collections
import os

import torch

from dump_train_samples import render_quads_to_midi


def load_side(prefix):
    data = torch.load(prefix + '.pt', weights_only=False)
    lengths = torch.load(prefix + '.length.pt', weights_only=False)
    names = []
    with open(prefix + '.txt') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                names.append(line.split('\t', 1)[1])
    starts = torch.cumsum(lengths, dim=0) - lengths
    return data, lengths, starts, names


def notes_per_frame(rows, poly):
    arr = rows.view(rows.shape[0], poly, 4)
    counter = collections.Counter()
    for t in range(arr.shape[0]):
        n = 0
        for k in range(poly):
            if int(arr[t, k, 0]) in (254, 255):
                break
            n += 1
        counter[n] += 1
    return dict(sorted(counter.items()))


def remap_program(rows, poly, program):
    out = rows.clone().view(rows.shape[0], poly, 4)
    mask = out[:, :, 0] < 128          # real notes only (not EOS/pad/drum-127 stays)
    out[:, :, 0][mask] = program
    return out.view(rows.shape[0], poly * 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mel-prefix', required=True)
    p.add_argument('--chord-prefix', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-songs', type=int, default=5)
    p.add_argument('--song-indices', type=int, nargs='*', default=None,
                   help='explicit song indices; default: evenly spaced')
    p.add_argument('--max-polyphony', type=int, default=4)
    p.add_argument('--tempo', type=float, default=120.0)
    p.add_argument('--chord-program', type=int, default=48,
                   help='program the chord stream is remapped to in the '
                        'rendered midis (both alone and combined); '
                        'negative = keep original')
    args = p.parse_args()

    mel_data, mel_len, mel_start, mel_names = load_side(args.mel_prefix)
    cho_data, cho_len, cho_start, cho_names = load_side(args.chord_prefix)
    if mel_names != cho_names:
        raise SystemExit('manifests differ -- run check_paired_dataset.py')

    n = len(mel_names)
    if args.song_indices:
        indices = args.song_indices
    else:
        k = min(args.n_songs, n)
        indices = [round(i * (n - 1) / max(k - 1, 1)) for i in range(k)]

    poly = args.max_polyphony
    print(f'{n} songs in dataset; auditing indices {indices}')
    for idx in indices:
        stem = os.path.splitext(os.path.basename(mel_names[idx]))[0]
        out = os.path.join(args.out_dir, stem)
        mel = mel_data[mel_start[idx]:mel_start[idx] + mel_len[idx]]
        cho = cho_data[cho_start[idx]:cho_start[idx] + cho_len[idx]]
        if args.chord_program >= 0:
            cho_r = remap_program(cho, poly, args.chord_program)
        else:
            cho_r = cho
        render_quads_to_midi([mel], os.path.join(out, 'melody.mid'),
                             poly, tempo=args.tempo)
        render_quads_to_midi([cho_r], os.path.join(out, 'chord.mid'),
                             poly, tempo=args.tempo)
        render_quads_to_midi([mel, cho_r], os.path.join(out, 'combined.mid'),
                             poly, tempo=args.tempo)
        print(f'[{idx}] {stem}: frames mel={int(mel_len[idx])} '
              f'chord={int(cho_len[idx])}')
        print(f'      mel notes/frame:   {notes_per_frame(mel, poly)}')
        print(f'      chord notes/frame: {notes_per_frame(cho, poly)}')
    print(f'\nRendered to {args.out_dir}/<song>/{{melody,chord,combined}}.mid')
    print('Orientation check: the melody distribution must contain only '
          '0 and 1; 3-4-note melody frames indicate reversed tracks in '
          'that source midi.')


if __name__ == '__main__':
    main()
