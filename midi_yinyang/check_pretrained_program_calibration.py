"""Diagnose whether a pretrained CP-transformer ckpt's final_decoder
output projection has meaningfully-trained weights for *all* program
positions in the a-slot, or whether only a subset (e.g. just program 24
and 0 from the legacy POP909 hardcode) are calibrated.

The vocab layout in CPTokenizer (non-velocity mode) reserves the first
256 token IDs for the a-slot, of which 0..127 are real MIDI programs
(127 = drum bank). If the pretrained model was trained with
preserve_program=True (real programs in a-slot), every row of the
output projection at index 0..127 has been touched by gradients and
all program embeddings look "similar" in norm. If it was trained with
preserve_program=False (always emit only program 24 or 0), the only
rows that received gradient are 24 and 0; rows 1..23, 25..126, 127 are
at near-init norm and essentially random.

This is a yes/no health check on whether your warm-start carries any
drum-program knowledge.

Run:
    python check_pretrained_program_calibration.py \\
        --pretrained ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
"""

import argparse
import statistics

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--decoder_key', type=str, default='final_decoder.weight',
                    help='State-dict key for the output projection weight. '
                         'Usually final_decoder.weight; if your ckpt nests '
                         'it elsewhere, pass the full key path.')
    args = ap.parse_args()

    print(f'[load] {args.pretrained}')
    obj = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    sd = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    if args.decoder_key not in sd:
        candidates = [k for k in sd if 'final_decoder' in k or 'output' in k.lower()]
        print(f'[err] key {args.decoder_key!r} not in state dict')
        print(f'      candidates: {candidates[:8]}')
        return

    W = sd[args.decoder_key]                                       # [vocab, H]
    print(f'[ok] {args.decoder_key} shape = {tuple(W.shape)}')
    if W.dim() != 2:
        print('[warn] expected 2D weight matrix; got', W.dim(), 'dims')
        return
    vocab, H = W.shape
    if vocab < 128:
        print(f'[warn] vocab {vocab} < 128; not enough rows to span the full '
              'MIDI program range')
        return

    # Per-row L2 norms.
    norms = W.float().norm(dim=1).tolist()

    program_norms = norms[:128]
    legacy_set = {0, 24}                                           # the hardcoded programs
    legacy_norms = [program_norms[i] for i in legacy_set]
    other_program_norms = [n for i, n in enumerate(program_norms) if i not in legacy_set]
    eos_pad_norms = norms[128:256] if vocab >= 256 else []
    bslot_norms = norms[256:] if vocab >= 256 else norms[128:]

    def _stats(name, xs):
        if not xs:
            return
        if len(xs) > 1:
            print(f'  {name:36s}  n={len(xs):5d}  '
                  f'mean={statistics.mean(xs):.4f}  '
                  f'median={statistics.median(xs):.4f}  '
                  f'min={min(xs):.4f}  max={max(xs):.4f}')
        else:
            print(f'  {name:36s}  n={len(xs):5d}  val={xs[0]:.4f}')

    print('\n[norms]')
    _stats('legacy hardcoded {0, 24}', legacy_norms)
    _stats('other programs (1..127, except 0/24)', other_program_norms)
    _stats('a-slot reserve (128..255, EOS/PAD-ish)', eos_pad_norms)
    _stats('b-slot (pitch_duration tokens)', bslot_norms)

    # Spot-check selected programs.
    print('\n[selected per-program rows]')
    for p in [0, 24, 32, 40, 64, 127]:
        if p < len(program_norms):
            print(f'  program {p:3d}  ({_program_name(p):24s})  |row| = {program_norms[p]:.4f}')

    # Verdict.
    if not other_program_norms:
        return
    legacy_med = statistics.median(legacy_norms) if legacy_norms else 0.0
    other_med = statistics.median(other_program_norms)
    if other_med < 0.25 * legacy_med:
        print('\n[verdict] Other-program rows have NORMS ~0.25x the hardcoded '
              'rows -- looks like the pretrained model was trained with '
              "preserve_program=False and didn't update those positions. "
              'Your warm-start carries the 24/0 knowledge only; drum '
              '(program 127) etc. start essentially at random init. '
              'Expect ~10-30k training steps before drum-program prediction '
              'becomes coherent.')
    elif other_med < 0.75 * legacy_med:
        print('\n[verdict] Other-program rows are SOMEWHAT smaller than the '
              'hardcoded rows. The pretrained model may have seen some '
              'variety in a-slot programs but emphasized 24/0. Mixed '
              'warm-start quality; train should converge faster than from '
              'scratch but slower than from a true multi-program pretrain.')
    else:
        print('\n[verdict] All program rows have comparable norms. '
              'Your warm-start carries genuine multi-program knowledge '
              'including drum (127). The fastest convergence path.')


# General-MIDI bank for the most common programs we care about.
_PROGRAM_NAMES = {
    0: 'Acoustic Grand Piano',
    24: 'Acoustic Guitar (nylon)',
    32: 'Acoustic Bass',
    40: 'Violin',
    64: 'Soprano Sax',
    127: 'Gunshot / drum-bank marker',
}


def _program_name(p):
    return _PROGRAM_NAMES.get(p, f'GM{p}')


if __name__ == '__main__':
    main()
