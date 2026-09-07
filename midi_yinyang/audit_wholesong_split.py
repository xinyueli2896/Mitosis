"""Match our held-out POP909 songs with whole-song-gen's validation split.

whole-song-gen trains on POP909 with a 9:1 train/valid split
(data/pop909_split/split.npz in their data payload; valid index i is
song i+1). They have no separate test set -- valid IS their held-out
set. Our A-family, specialists and S1 train on the POP909 .pt datasets
with the FramedDataset rule: a song is held out (split='val') iff its
dataset index is a multiple of 10; plus the 5 songs cut out entirely
into input/pop909_split (the E1 test set so far).

A song is a fair test song for a table containing WS only if NEITHER
side trained on it. This script prints:

  * their valid ids, and whether split.npz equals the seed-1234
    regeneration of their create_train_valid_split (it should)
  * our val ids (from data/<dataset>.txt + .length.pt) and our test ids
  * where each of our current test songs sits in THEIR split
  * the SHARED-CLEAN set: their valid  ∩  (our val ∪ our test)

and, with --stage-dir, copies the shared-clean songs' melody/chord midis
into <stage-dir>/{melody,chord}/ so E1 can point MEL_FOLDER/CHORD_FOLDER
at them.

Usage (via audit_wholesong_split.sbatch):
    python audit_wholesong_split.py --split external/whole_song_gen/data/pop909_split/split.npz
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
import torch

TRAIN_LENGTH = 384      # FramedDataset target length (cp_transformer_m2c_moe)


def song_id(name):
    """'001.mid', '/x/y/001-v1.mid', '001' -> 1; None if no 3-digit id."""
    m = re.search(r'(?<!\d)(\d{3})(?!\d)', os.path.basename(name))
    return int(m.group(1)) if m else None


def their_split(fn):
    z = np.load(fn)
    return (set((z['train_inds'] + 1).tolist()),
            set((z['valid_inds'] + 1).tolist()))


def seed_regen(n=909, train_factor=9, seed=1234):
    np.random.seed(seed)
    ids = np.arange(n)
    n_train = int(n * train_factor / (train_factor + 1))
    tr = np.sort(np.random.choice(ids, n_train, replace=False))
    va = np.sort(np.setdiff1d(ids, tr))
    return set((tr + 1).tolist()), set((va + 1).tolist())


def our_val(dataset_txt, length_pt, split_ratio=10):
    names = {}
    with open(dataset_txt) as f:
        for line in f:
            i, name = line.rstrip('\n').split('\t', 1)
            names[int(i)] = name
    lengths = torch.load(length_pt)
    is_valid = (lengths >= TRAIN_LENGTH)
    idx = torch.arange(len(lengths))[is_valid]
    val_idx = idx[idx % split_ratio == 0].tolist()
    train_idx = idx[idx % split_ratio != 0].tolist()
    to_ids = lambda ix: {song_id(names[i]) for i in ix if song_id(names[i]) is not None}
    return to_ids(train_idx), to_ids(val_idx), names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, help='their split.npz')
    p.add_argument('--datasets', nargs='+',
                   default=['pop909_chord_cp4_v2', 'pop909_chord_cp8_v2',
                            'pop909_melchord_tagged_cp16_v2'],
                   help='dataset stems under data/ (the .txt and .length.pt '
                        'next to each give index -> song). Our systems train '
                        'on different files -- cp4 (A1/A3/specialists), cp8 '
                        '(YinYang), tagged cp16 (S1/S-scratch) -- and each '
                        'has its own index%%10 validation set; the audit '
                        'reports every one and intersects them.')
    p.add_argument('--test-folder', default='input/pop909_split/melody')
    p.add_argument('--mel-src', default='POP909-Dataset/POP909-melody')
    p.add_argument('--chord-src', default='POP909-Dataset/POP909-chord')
    p.add_argument('--stage-dir', default=None,
                   help='copy shared-clean songs here as {melody,chord}/<id>.mid')
    args = p.parse_args()

    fmt = lambda s: ' '.join(f'{i:03d}' for i in sorted(s))

    ws_train, ws_valid = their_split(args.split)
    rg_train, rg_valid = seed_regen()
    print(f'[theirs] n_train={len(ws_train)} n_valid={len(ws_valid)}')
    print(f'[theirs] valid ids: {fmt(ws_valid)}')
    print(f'[theirs] split.npz == seed-1234 regeneration: '
          f'{"YES" if ws_valid == rg_valid else "NO -- their payload was split differently"}')

    our_train, our_val_ids, names = set(), None, {}
    for stem in args.datasets:
        txt, lp = f'data/{stem}.txt', f'data/{stem}.length.pt'
        if not (os.path.isfile(txt) and os.path.isfile(lp)):
            print(f'[ours] {stem}: files missing under data/ -- skipped')
            continue
        tr, va, nm = our_val(txt, lp)
        print(f'[ours] {stem}: {len(nm)} songs; train={len(tr)} val={len(va)}'
              f' (index % 10 == 0)  val ids: {fmt(va)}')
        our_train |= tr
        our_val_ids = va if our_val_ids is None else (our_val_ids & va)
        names = names or nm
    if our_val_ids is None:
        sys.exit('no dataset files found under data/')
    print(f'[ours] validation songs common to ALL listed datasets: '
          f'{len(our_val_ids)}  ({fmt(our_val_ids)})')
    print(f'[ours] songs in ANY training split: {len(our_train)}')
    our_test = set()
    if os.path.isdir(args.test_folder):
        for f in os.listdir(args.test_folder):
            s = song_id(f)
            if s is not None:
                our_test.add(s)
    print(f'[ours] test ids (cut before preprocessing): {fmt(our_test)}  '
          f'({args.test_folder})')
    in_data = our_test & (our_train | our_val_ids)
    if in_data:
        print(f'[ours] WARNING: test songs present in a dataset file: {fmt(in_data)}')
    both = our_train & ws_valid
    print(f'[ours] our TRAIN songs inside their valid: {len(both)} '
          f'(these are fair for WS but not for us)')

    print('\n[current E1 test songs in THEIR split]')
    for s in sorted(our_test):
        where = ('their TRAIN -- WS trained on it' if s in ws_train
                 else 'their VALID -- clean' if s in ws_valid else 'absent')
        print(f'  {s:03d}: {where}')

    ours_heldout = our_val_ids | our_test
    shared = ours_heldout & ws_valid
    print(f'\n[SHARED-CLEAN] their valid ∩ our held-out (val ∪ test): '
          f'{len(shared)} songs')
    print(f'  {fmt(shared)}')
    print('  neither side trained on these; use them as the E1 test set '
          'for any table that contains WS.')

    if args.stage_dir:
        ok = 0
        for sub, src in (('melody', args.mel_src), ('chord', args.chord_src)):
            dst = os.path.join(args.stage_dir, sub)
            os.makedirs(dst, exist_ok=True)
            if not os.path.isdir(src):
                print(f'[stage] source folder missing: {src} -- not staged')
                continue
            for s in sorted(shared):
                cands = [f for f in os.listdir(src) if song_id(f) == s
                         and f.lower().endswith('.mid')]
                if not cands:
                    print(f'[stage] {sub}: no midi for song {s:03d} in {src}')
                    continue
                shutil.copyfile(os.path.join(src, cands[0]),
                                os.path.join(dst, f'{s:03d}.mid'))
                ok += 1
        print(f'[stage] {ok} files -> {args.stage_dir}/{{melody,chord}}')
        n_m = len(os.listdir(os.path.join(args.stage_dir, 'melody')))
        n_c = len(os.listdir(os.path.join(args.stage_dir, 'chord')))
        if n_m != n_c or n_m != len(shared):
            print(f'[stage] WARNING: melody={n_m} chord={n_c} expected={len(shared)}')
            sys.exit(1)


if __name__ == '__main__':
    main()
