"""Verify two preprocessed CP streams are pairable song-by-song.

FramedDataset pairs song i of stream A with song i of stream B and only
checks that the song COUNTS match. If preprocessing dropped different
songs from each stream (a midi that fails to parse on one side only),
every pair after the first divergence is a wrong (melody, chord) match
and training silently learns garbage. This script compares the .txt
manifests written by create_npy_dataset_from_midi and fails loudly on
any mismatch.

Usage:
    python check_paired_dataset.py data/pop909_melody_cp4_v2 data/pop909_chord_cp4_v2
(prefixes, i.e. the dataset_name passed to create_npy_dataset_from_midi;
 reads <prefix>.txt, <prefix>.length.pt, <prefix>.pt)
"""

import sys

import torch


def read_manifest(prefix):
    names = []
    with open(prefix + '.txt') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            idx, name = line.split('\t', 1)
            names.append(name)
    return names


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    prefix_a, prefix_b = sys.argv[1], sys.argv[2]
    names_a = read_manifest(prefix_a)
    names_b = read_manifest(prefix_b)

    only_a = sorted(set(names_a) - set(names_b))
    only_b = sorted(set(names_b) - set(names_a))
    ok = True
    if only_a:
        ok = False
        print(f'[pairing] {len(only_a)} songs only in {prefix_a}: '
              f'{only_a[:10]}{" ..." if len(only_a) > 10 else ""}')
    if only_b:
        ok = False
        print(f'[pairing] {len(only_b)} songs only in {prefix_b}: '
              f'{only_b[:10]}{" ..." if len(only_b) > 10 else ""}')
    if ok and names_a != names_b:
        ok = False
        first = next(i for i, (a, b) in enumerate(zip(names_a, names_b))
                     if a != b)
        print(f'[pairing] same songs but different ORDER starting at index '
              f'{first}: {names_a[first]} vs {names_b[first]}')

    len_a = torch.load(prefix_a + '.length.pt', weights_only=False)
    len_b = torch.load(prefix_b + '.length.pt', weights_only=False)
    if len(len_a) != len(names_a) or len(len_b) != len(names_b):
        ok = False
        print(f'[pairing] length.pt entries do not match manifest counts: '
              f'{len(len_a)} vs {len(names_a)} / {len(len_b)} vs {len(names_b)}')

    if not ok:
        print('[pairing] FAIL -- do NOT train on this pair; fix the '
              'preprocessing drops first (both streams must keep exactly '
              'the same songs in the same order).')
        raise SystemExit(1)

    n = len(names_a)
    frames = torch.minimum(len_a, len_b)
    print(f'[pairing] OK: {n} songs paired 1:1, '
          f'{int(frames.sum())} usable frames '
          f'(min-per-song), median song length {int(frames.median())} frames.')


if __name__ == '__main__':
    main()
