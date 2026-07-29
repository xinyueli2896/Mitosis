"""Combine multiple paired melody/chord datasets (POP909, Nottingham,
...) into one training set for A.2 duet training.

Concatenates the four FramedDataset sidecar files (.pt, .length.pt,
.pitch_shift_range.pt, .txt) from each source pair, in the given
order, and re-indexes the manifest continuously across sources.

ORDER MATTERS for train/val split stability: FramedDataset's split is
by absolute song index mod 10. Putting a corpus FIRST, in its
existing internal order, keeps that corpus's songs at the SAME
absolute indices they had standalone -- so a held-out list defined
against the standalone dataset (e.g. EXPERIMENTS.md's POP909 ids
001,011,...,091) remains held out after combination, as long as that
corpus is listed first and no source's own row order is disturbed.
Corpora appended afterward only shift their OWN indices.

Manifest entries are prefixed `<tag>:` so songs from different
corpora can never collide by name, and so check_paired_dataset.py's
name-based diff still validates the combined pair (both streams get
the same prefixed names, in the same order).

Usage:
    python combine_melchord_datasets.py \
        --source pop909:data/pop909_melody_cp4_v2:data/pop909_chord_cp4_v2 \
        --source nottingham:data/nottingham_melody_cp4_v2:data/nottingham_chord_cp4_v2 \
        --out-mel data/melchord_pop909_nottingham_melody_cp4_v2 \
        --out-chord data/melchord_pop909_nottingham_chord_cp4_v2
"""

import argparse

import torch


def load_side(prefix):
    data = torch.load(prefix + '.pt', weights_only=False)
    length = torch.load(prefix + '.length.pt', weights_only=False)
    shift = torch.load(prefix + '.pitch_shift_range.pt', weights_only=False)
    names = []
    with open(prefix + '.txt') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                names.append(line.split('\t', 1)[1])
    return data, length, shift, names


def write_side(prefix, data_chunks, length_chunks, shift_chunks, names):
    torch.save(torch.cat(data_chunks, dim=0), prefix + '.pt')
    torch.save(torch.cat(length_chunks, dim=0), prefix + '.length.pt')
    torch.save(torch.cat(shift_chunks, dim=0), prefix + '.pitch_shift_range.pt')
    with open(prefix + '.txt', 'w') as f:
        for i, name in enumerate(names):
            f.write(f'{i}\t{name}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', action='append', required=True,
                   help='tag:mel_prefix:chord_prefix, repeatable. '
                        'ORDER MATTERS -- see module docstring.')
    p.add_argument('--out-mel', required=True)
    p.add_argument('--out-chord', required=True)
    args = p.parse_args()

    mel_data, mel_len, mel_shift, mel_names = [], [], [], []
    cho_data, cho_len, cho_shift, cho_names = [], [], [], []
    total_frames_mel = total_frames_cho = 0

    for src in args.source:
        try:
            tag, mel_prefix, cho_prefix = src.split(':', 2)
        except ValueError:
            raise SystemExit(f'--source must be tag:mel_prefix:chord_prefix, got {src!r}')

        md, ml, ms, mn = load_side(mel_prefix)
        cd, cl, cs, cn = load_side(cho_prefix)
        if mn != cn:
            raise SystemExit(
                f'[{tag}] melody/chord manifests differ -- run '
                f'check_paired_dataset.py on {mel_prefix} / {cho_prefix} '
                f'first (do not combine an unpaired source)')
        mel_data.append(md); mel_len.append(ml); mel_shift.append(ms)
        mel_names.extend(f'{tag}:{n}' for n in mn)
        cho_data.append(cd); cho_len.append(cl); cho_shift.append(cs)
        cho_names.extend(f'{tag}:{n}' for n in cn)
        total_frames_mel += md.shape[0]
        total_frames_cho += cd.shape[0]
        print(f'[{tag}] {len(mn)} songs, mel {md.shape[0]} frames, '
              f'chord {cd.shape[0]} frames, poly-width {md.shape[1]}/{cd.shape[1]}')

    mel_widths = {d.shape[1] for d in mel_data}
    cho_widths = {d.shape[1] for d in cho_data}
    if len(mel_widths) > 1 or len(cho_widths) > 1:
        raise SystemExit(
            f'sources have mismatched max_polyphony (mel widths='
            f'{mel_widths}, chord widths={cho_widths}) -- retokenize at '
            f'a common max_polyphony before combining')

    write_side(args.out_mel, mel_data, mel_len, mel_shift, mel_names)
    write_side(args.out_chord, cho_data, cho_len, cho_shift, cho_names)
    print(f'Combined {len(mel_names)} total songs, {total_frames_mel} '
          f'mel frames / {total_frames_cho} chord frames')
    print(f'  -> {args.out_mel}.pt')
    print(f'  -> {args.out_chord}.pt')


if __name__ == '__main__':
    main()
