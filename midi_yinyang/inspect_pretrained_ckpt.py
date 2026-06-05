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
    ap.add_argument('--list_prefix', type=str, default=None,
                    help='If set, dump all keys starting with this prefix '
                         "(e.g. 'model.' or 'model.layer.0.'). Overrides "
                         "the default global_roformer.layer.{N}. dump.")
    ap.add_argument('--head', type=int, default=80,
                    help='Truncate the listing to this many lines.')
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

    # If user gave an explicit prefix, dump that instead of the default search.
    if args.list_prefix is not None:
        prefix = args.list_prefix
        print(f'\n[list_prefix] keys starting with {prefix!r}:')
        count = 0
        for k in sorted(sd):
            if k.startswith(prefix):
                count += 1
                if count <= args.head:
                    print(f'  {k:80s}  {tuple(sd[k].shape)}')
        if count > args.head:
            print(f'  ... ({count - args.head} more keys omitted; '
                  f'pass --head {count} to see all)')
        print(f'[list_prefix] {count} keys matched')
        return

    # Default: search for a layer.N. pattern under any top-level prefix.
    layer_re = re.compile(r'^([^.]+(?:\.[^.]+)*?)\.layer\.(\d+)\.')
    by_prefix = {}
    for k in sd:
        m = layer_re.match(k)
        if m:
            base, idx = m.group(1), int(m.group(2))
            by_prefix.setdefault(base, set()).add(idx)

    if not by_prefix:
        print('\n[warn] no keys matched any <prefix>.layer.<N>. pattern. '
              'Dump all keys with: python inspect_pretrained_ckpt.py --list_prefix model.')
        return

    print('\n[layer-prefix candidates]')
    for base, ids in sorted(by_prefix.items(), key=lambda kv: -len(kv[1])):
        print(f'  {base:40s}  layers 0..{max(ids)}  ({len(ids)} unique)')

    # Use the prefix with the most layers as the canonical global stack.
    canonical_prefix = max(by_prefix, key=lambda b: len(by_prefix[b]))
    canonical_layers = max(by_prefix[canonical_prefix]) + 1
    print(f'\n[canonical] using prefix {canonical_prefix!r} with '
          f'{canonical_layers} layers as the global stack')

    L = args.inspect_layer
    layer_key_prefix = f'{canonical_prefix}.layer.{L}.'
    print(f'\n[keys in layer {L}] (prefix: {layer_key_prefix})')
    for k in sorted(sd):
        if k.startswith(layer_key_prefix):
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
