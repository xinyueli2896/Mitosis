"""Dataset-order downbeat map, so training windows can start on real bars.

FramedDataset slices a training window at `offset -= offset % 16`, which
assumes sixteen frames to a bar for the whole song. POP909's own
annotation disagrees: 783 of 909 songs contain a bar that is not four
beats, and after the first one every later downbeat stops being a
multiple of 16 frames. Measured over the corpus, 46.5% of all beats sit
after their song's first irregular bar -- so nearly half of every
model's training windows start on a "bar line" that is not one, while
the loader's snapping teaches the model that frame 0 of a window always
is one.

This writes data/<stem>.downbeats.pt: a list indexed exactly like the
dataset, each entry a LongTensor of that song's TRUE downbeat frames,
taken from the aligner's align_grid_report.tsv. A song with no entry
gets an empty tensor and the loader falls back to the old rule for it,
so corpora with no such report are unaffected.

Usage (via build_downbeat_map.sbatch, CPU):
    python build_downbeat_map.py --dataset pop909_melody_cp8_v2_v5 \\
        --report <v5>/align_grid_report.tsv
"""
import argparse
import csv
import os
import re

import torch


def song_id(name):
    m = re.search(r'(\d{3})', os.path.basename(str(name)))
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   help='dataset stem under data/ (the .txt gives index -> '
                        'song, which is the order the map must follow)')
    p.add_argument('--report', required=True, action='append',
                   help='align_grid_report.tsv; repeatable, later files win')
    p.add_argument('--out', default=None,
                   help='default data/<dataset>.downbeats.pt')
    a = p.parse_args()

    txt = f'data/{a.dataset}.txt'
    if not os.path.exists(txt):
        raise SystemExit(f'missing {txt} -- build the dataset first')

    frames = {}
    for rp in a.report:
        if not os.path.exists(rp):
            raise SystemExit(f'missing report {rp}')
        with open(rp) as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                raw = (row.get('downbeat_frames') or '').strip()
                if raw:
                    frames[row['id']] = [int(x) for x in raw.split(',') if x != '']
        print(f'[report] {rp}: {len(frames)} songs with downbeat frames')

    names = {}
    with open(txt) as fh:
        for line in fh:
            i, nm = line.rstrip('\n').split('\t', 1)
            names[int(i)] = nm
    n = max(names) + 1 if names else 0

    out, hit, irregular = [], 0, 0
    for i in range(n):
        sid = song_id(names.get(i, ''))
        f = frames.get(sid) if sid else None
        if f:
            hit += 1
            if any(b - a_ != 16 for a_, b in zip(f, f[1:])):
                irregular += 1
            out.append(torch.tensor(f, dtype=torch.long))
        else:
            out.append(torch.zeros(0, dtype=torch.long))

    dst = a.out or f'data/{a.dataset}.downbeats.pt'
    torch.save(out, dst)
    print(f'{n} dataset entries; {hit} with a downbeat map '
          f'({irregular} of those contain an irregular bar); '
          f'{n - hit} fall back to the old offset % 16 rule')
    print(f'-> {dst}')


if __name__ == '__main__':
    main()
