"""Dump the keys / shapes of a pretrained CP-transformer checkpoint to confirm
the init scripts' regex patterns will match. Run with:

    python inspect_pretrained_ckpt.py [path_to_ckpt]

Default path is the one xinyue keeps the size-1 / 12-layer pretrained ckpt at.
"""

import argparse
import re
import sys

import torch


DEFAULT_CKPT = 'ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='?', default=DEFAULT_CKPT,
                    help='Path to the pretrained .ckpt file.')
    ap.add_argument('--inspect_layer', type=int, default=0,
                    help='Which global layer to dump keys for (default 0).')
    args = ap.parse_args()

    print(f'[load] {args.ckpt}')
    obj = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj:
        sd = obj['state_dict']
        print(f"[ok] checkpoint wrapped in dict; using ['state_dict']")
    else:
        sd = obj
        print('[ok] checkpoint is a raw state_dict')

    print(f'[ok] {len(sd):,} parameter tensors total')

    # Group keys by top-level prefix
    prefixes = {}
    for k in sd:
        top = k.split('.', 1)[0]
        prefixes[top] = prefixes.get(top, 0) + 1
    print('\n[summary] top-level module groups:')
    for p, n in sorted(prefixes.items(), key=lambda kv: -kv[1]):
        print(f'  {p:40s}  {n} tensors')

    # Find global layer count
    layer_ids = sorted({
        int(m.group(1)) for k in sd
        for m in [re.match(r'global_roformer\.layer\.(\d+)\.', k)] if m
    })
    n_layers = (max(layer_ids) + 1) if layer_ids else 0
    print(f'\n[layers] global_roformer has {n_layers} layers')

    if n_layers == 0:
        print('\n[warn] no keys matched global_roformer.layer.N.* -- the regex '
              'in init_pretrained_into_jointattn.py will not match. Check the '
              'naming above and tell me what you see.')
        return

    L = args.inspect_layer
    print(f'\n[keys in layer {L}]')
    for k in sorted(sd):
        if f'global_roformer.layer.{L}.' in k:
            shape = tuple(sd[k].shape)
            print(f'  {k:80s}  {shape}')

    # Check whether MoE-style keys already exist in the pretrained ckpt.
    moe_hits = [k for k in sd if 'experts' in k or 'moe' in k.lower()]
    if moe_hits:
        print(f'\n[moe] {len(moe_hits)} MoE-style keys found in the ckpt. '
              'Sample:')
        for k in moe_hits[:5]:
            print(f'  {k:80s}  {tuple(sd[k].shape)}')
        print('-> the pretrained ckpt is already MoE; the init script will '
              'need different logic. Tell me and I\'ll fix it.')
    else:
        print('\n[moe] no MoE keys in pretrained ckpt. Dense FFN -> replicate '
              'across all K experts is the right copy strategy.')


if __name__ == '__main__':
    main()
