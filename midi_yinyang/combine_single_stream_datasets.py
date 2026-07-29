"""Combine multiple single-stream (merged melody+chord) datasets for
S* single-stream finetuning.

Mirrors combine_melchord_datasets.py's role but for the single-stream
(cp_transformer.py) FramedDataset format, which has only ONE token
stream per song rather than a melody/chord pair.

ORDER MATTERS in the same way as the paired combiner: list a corpus
FIRST, in its existing internal order, to keep its songs at the SAME
absolute indices they had standalone -- preserving held-out ids under
FramedDataset's index-mod-10 split (EXPERIMENTS.md's POP909 ids
001,011,...,091 stay held out as long as POP909 is listed first).

Usage:
    python combine_single_stream_datasets.py \
        --source pop909:data/pop909_melchord_cp16_v2 \
        --source nottingham:data/nottingham_melchord_cp16_v2 \
        --out data/melchord_pop909_nottingham_cp16_v2
"""

import argparse

import torch


def load(prefix):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', action='append', required=True,
                   help='tag:prefix, repeatable. ORDER MATTERS -- see '
                        'module docstring.')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    all_data, all_len, all_shift, all_names = [], [], [], []
    widths = set()
    for src in args.source:
        try:
            tag, prefix = src.split(':', 1)
        except ValueError:
            raise SystemExit(f'--source must be tag:prefix, got {src!r}')
        d, l, s, n = load(prefix)
        widths.add(d.shape[1])
        all_data.append(d); all_len.append(l); all_shift.append(s)
        all_names.extend(f'{tag}:{x}' for x in n)
        print(f'[{tag}] {len(n)} songs, {d.shape[0]} frames, width {d.shape[1]}')

    if len(widths) > 1:
        raise SystemExit(
            f'sources have mismatched polyphony widths: {widths} -- '
            f'retokenize at a common max_polyphony before combining')

    torch.save(torch.cat(all_data, dim=0), args.out + '.pt')
    torch.save(torch.cat(all_len, dim=0), args.out + '.length.pt')
    torch.save(torch.cat(all_shift, dim=0), args.out + '.pitch_shift_range.pt')
    with open(args.out + '.txt', 'w') as f:
        for i, name in enumerate(all_names):
            f.write(f'{i}\t{name}\n')
    print(f'Combined {len(all_names)} songs -> {args.out}.pt')


if __name__ == '__main__':
    main()
