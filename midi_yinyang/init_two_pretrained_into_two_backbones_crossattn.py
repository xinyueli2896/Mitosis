"""Initialize M2CTwoBackbonesCrossAttn from two pretrained CP transformer
checkpoints (the cp_transformer.py / RoFormerSymbolicTransformer family).

Mapping:

  Pretrained ckpt for MEL backbone           ->  M2CTwoBackbonesCrossAttn
    local_embedding.weight                       local_embedding(.weight) or local_embedding_m
    local_encoder.layer.*                        local_encoder(.layer.*) or local_encoder_m.layer.*
    local_decoder.layer.*                        local_decoder(.layer.*) or local_decoder_m.layer.*
    model.layer.*                                global_roformer_m.layer.*
    final_decoder.{weight,bias}                  final_decoder_m.{weight,bias}
    global_sos                                   global_sos_m

  Pretrained ckpt for CHORD backbone (typically the SAME ckpt at start):
    local_embedding.weight                       local_embedding (if shared) / local_embedding_c (if untied)
    local_encoder.layer.*                        local_encoder (if shared) / local_encoder_c (if untied)
    local_decoder.layer.*                        local_decoder (if shared) / local_decoder_c (if untied)
    model.layer.*                                global_roformer_c.layer.*
    final_decoder.{weight,bias}                  final_decoder_c.{weight,bias}
    global_sos                                   global_sos_c

  Fresh init (no pretrained equivalent):
    cross_attn_m_reads_c                         (small / zero output proj init in module)
    cross_attn_c_reads_m
    gate_m / gate_c                              (bias = -10 in the model, so sigmoid ~= 0)
    token_type_embeddings

For "two duplicated ckpts" (today): pass --ckpt_pretrained <path>. Both
backbones load the same state dict. For "two different ckpts" (future):
pass --ckpt_pretrained_m <path1> --ckpt_pretrained_c <path2>.

For cross-modality later (e.g. audio + symbolic with different vocabs),
also pass --untie_local so each modality has its own local encoder/decoder/
embedding. Vocab-size mismatches between ckpt and model are reported and
the script exits.

Run:
    python init_two_pretrained_into_two_backbones_crossattn.py \\
        --ckpt_pretrained pretrained/cp_transformer_pretrained.pt \\
        --size 1 \\
        --out pretrained/two_backbones_crossattn_init.pt

Then start training from the init:
    python cp_transformer_m2c_two_backbones_crossattn.py \\
        --checkpoint_path pretrained/two_backbones_crossattn_init.pt \\
        --size 1 \\
        --path_to_dataset data/pop909_chord_cp4_v2.pt \\
        --wandb
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import re
from collections import OrderedDict

import torch

from cp_transformer_m2c_two_backbones_crossattn import M2CTwoBackbonesCrossAttn


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_state(path):
    """Open a .ckpt / .pt and strip common Lightning prefixes."""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj:
        sd = obj['state_dict']
    elif isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
        sd = obj['model']
    else:
        sd = obj
    # Strip a common wrapper prefix if present.
    keys = list(sd.keys())
    for prefix in ('net.', 'module.'):
        if all(k.startswith(prefix) for k in keys):
            sd = OrderedDict((k[len(prefix):], v) for k, v in sd.items())
            break
    return sd


def _inspect(sd, label):
    """Group keys by 2-segment prefix for a quick visual."""
    print(f'\n=== {label} ({len(sd)} tensors) ===')
    groups = {}
    for k, v in sd.items():
        head = '.'.join(k.split('.')[:2])
        groups.setdefault(head, []).append(
            (k, tuple(v.shape) if hasattr(v, 'shape') else None),
        )
    for head, items in groups.items():
        print(f'  {head}/  ({len(items)} tensors)')
        for k, s in items[:3]:
            print(f'    {k}  {s}')
        if len(items) > 3:
            print(f'    ... +{len(items) - 3} more')


# ---------------------------------------------------------------------------
# Per-modality mapping
# ---------------------------------------------------------------------------

# Rules: src key -> destination key transform fn that takes (key, modality).
# 'modality' is 'm' or 'c'. Returns the destination key the value should land
# at in M2CTwoBackbonesCrossAttn's state dict.

def _dest_key_for(src_key, modality, untie_local):
    """Compute the destination key for a single source key, or return None
    if the source key should not be mapped (no destination)."""
    suf_m = '_m' if untie_local else ''
    suf_c = '_c' if untie_local else ''

    if src_key.startswith('local_embedding.'):
        if untie_local:
            return src_key.replace(
                'local_embedding.', f'local_embedding_{modality}.',
            )
        # Shared: only mel ckpt fills it; chord ckpt is ignored for this key.
        return src_key if modality == 'm' else None

    if src_key.startswith('local_encoder.'):
        if untie_local:
            return src_key.replace(
                'local_encoder.', f'local_encoder_{modality}.',
            )
        return src_key if modality == 'm' else None

    if src_key.startswith('local_decoder.'):
        if untie_local:
            return src_key.replace(
                'local_decoder.', f'local_decoder_{modality}.',
            )
        return src_key if modality == 'm' else None

    if src_key.startswith('model.'):
        return src_key.replace('model.', f'global_roformer_{modality}.')

    if src_key.startswith('final_decoder.'):
        return src_key.replace(
            'final_decoder.', f'final_decoder_{modality}.',
        )

    if src_key == 'global_sos':
        return f'global_sos_{modality}'

    # Anything else (e.g. _future_mask buffers, optimizer state stragglers) skip.
    return None


# ---------------------------------------------------------------------------
# Build dest state dict
# ---------------------------------------------------------------------------

def build_dest(model, src_m, src_c, untie_local):
    """Construct a state dict for `model` from src_m (mel backbone pretrained)
    and src_c (chord backbone pretrained), keeping the model's existing values
    for any destination key not covered by either source."""
    dest_template = model.state_dict()
    out = OrderedDict()
    used = {'m': set(), 'c': set()}
    counts = {
        'copied_m': 0, 'copied_c': 0,
        'skipped_shared_dup': 0,
        'kept_dst_init': [], 'missing_in_src': [],
    }

    # First, route source keys to dest keys.
    src_to_dest_m = {}
    for src_key in src_m.keys():
        dk = _dest_key_for(src_key, 'm', untie_local)
        if dk is not None:
            src_to_dest_m[dk] = src_key

    src_to_dest_c = {}
    for src_key in src_c.keys():
        dk = _dest_key_for(src_key, 'c', untie_local)
        if dk is not None:
            src_to_dest_c[dk] = src_key

    # Build dest.
    for dest_key, dest_val in dest_template.items():
        # Check if available from mel source.
        if dest_key in src_to_dest_m:
            src_key = src_to_dest_m[dest_key]
            src_val = src_m[src_key]
            if src_val.shape != dest_val.shape:
                raise ValueError(
                    f'Shape mismatch [mel]: dest {dest_key} {tuple(dest_val.shape)} '
                    f'vs src {src_key} {tuple(src_val.shape)}'
                )
            out[dest_key] = src_val.clone()
            used['m'].add(src_key)
            counts['copied_m'] += 1
            # If shared and chord ckpt would also map to the same dest_key,
            # the chord pretrained's value is dropped (we prefer the mel ckpt).
            # That's intentional and only happens when untie_local=False.
            if dest_key in src_to_dest_c:
                counts['skipped_shared_dup'] += 1
            continue

        # Else from chord source.
        if dest_key in src_to_dest_c:
            src_key = src_to_dest_c[dest_key]
            src_val = src_c[src_key]
            if src_val.shape != dest_val.shape:
                raise ValueError(
                    f'Shape mismatch [chord]: dest {dest_key} {tuple(dest_val.shape)} '
                    f'vs src {src_key} {tuple(src_val.shape)}'
                )
            out[dest_key] = src_val.clone()
            used['c'].add(src_key)
            counts['copied_c'] += 1
            continue

        # Nothing pretrained for this dest key -- keep model's init.
        out[dest_key] = dest_val.clone()
        counts['kept_dst_init'].append(dest_key)

    return out, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_pretrained', help='Single pretrained CP transformer '
                    '.pt path; duplicated into both backbones.')
    ap.add_argument('--ckpt_pretrained_m', help='Pretrained ckpt for the MEL '
                    'backbone. Overrides --ckpt_pretrained for the mel side.')
    ap.add_argument('--ckpt_pretrained_c', help='Pretrained ckpt for the CHORD '
                    'backbone. Overrides --ckpt_pretrained for the chord side.')
    ap.add_argument('--size', type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument('--with_velocity', action='store_true', default=False)
    ap.add_argument('--untie_local', action='store_true', default=False)
    ap.add_argument('--crossattn_num_heads', type=int, default=8)
    ap.add_argument('--gate_init_bias', type=float, default=-10.0)
    ap.add_argument('--inspect', action='store_true',
                    help='Print src/dest key layouts and exit.')
    ap.add_argument('--out', required=False, default=None)
    args = ap.parse_args()

    ckpt_m_path = args.ckpt_pretrained_m or args.ckpt_pretrained
    ckpt_c_path = args.ckpt_pretrained_c or args.ckpt_pretrained
    if ckpt_m_path is None or ckpt_c_path is None:
        raise SystemExit('Need --ckpt_pretrained (duplicated) OR both '
                         '--ckpt_pretrained_m AND --ckpt_pretrained_c.')

    print(f'[load] mel backbone source:   {ckpt_m_path}')
    print(f'[load] chord backbone source: {ckpt_c_path}')
    src_m = _load_state(ckpt_m_path)
    if ckpt_c_path == ckpt_m_path:
        src_c = src_m
        print('[load] (same file, will be applied to both backbones)')
    else:
        src_c = _load_state(ckpt_c_path)
    _inspect(src_m, 'source: mel backbone')
    if src_c is not src_m:
        _inspect(src_c, 'source: chord backbone')

    print(f'\n[build] instantiating M2CTwoBackbonesCrossAttn '
          f'(size={args.size}, untie_local={args.untie_local})')
    model = M2CTwoBackbonesCrossAttn(
        size=args.size,
        with_velocity=args.with_velocity,
        untie_local=args.untie_local,
        crossattn_num_heads=args.crossattn_num_heads,
        gate_init_bias=args.gate_init_bias,
    )
    _inspect(model.state_dict(), 'destination: M2CTwoBackbonesCrossAttn')

    if args.inspect:
        print('\n[inspect] done. Exiting without writing.')
        return

    new_sd, counts = build_dest(model, src_m, src_c, args.untie_local)

    print('\n=== Mapping report ===')
    print(f'  copied_m: {counts["copied_m"]}')
    print(f'  copied_c: {counts["copied_c"]}')
    if counts['skipped_shared_dup'] > 0:
        print(f'  skipped_shared_dup: {counts["skipped_shared_dup"]} '
              f'(shared local components; chord ckpt overlap dropped)')
    print(f'  kept_dst_init: {len(counts["kept_dst_init"])} keys '
          '(no pretrained equivalent -- cross_attn_*, gate_*, token_type_embeddings)')
    for k in counts['kept_dst_init'][:10]:
        print(f'    - {k}')
    if len(counts['kept_dst_init']) > 10:
        print(f'    ... +{len(counts["kept_dst_init"]) - 10} more')

    # Strict load to confirm shapes match.
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    if missing or unexpected:
        print(f'[load] missing={missing} unexpected={unexpected}')
        raise SystemExit(1)
    print('\n[load] state_dict loaded into model with strict=True')

    out_path = args.out or (
        ckpt_m_path + '.two_backbones_crossattn_init.pt'
    )
    print(f'[save] writing {out_path}')
    torch.save(
        {
            'state_dict': new_sd,
            'hyper_parameters': {
                'size': args.size,
                'untie_local': args.untie_local,
                'crossattn_num_heads': args.crossattn_num_heads,
                'gate_init_bias': args.gate_init_bias,
                'init_from_pretrained_m': ckpt_m_path,
                'init_from_pretrained_c': ckpt_c_path,
                'variant': 'm2c_two_backbones_crossattn',
            },
        },
        out_path,
    )
    print('[save] done.')
    print('\nNext: train from this init:')
    print(f'  python cp_transformer_m2c_two_backbones_crossattn.py '
          f'--checkpoint_path {out_path} --size {args.size} '
          f'--path_to_dataset data/<your_dataset>.pt --wandb')


if __name__ == '__main__':
    main()
