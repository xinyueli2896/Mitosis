"""Initialize the per-layer M2CTwoBackbonesCrossAttn from one or two
pretrained CP transformer checkpoints.

Mapping for the per-layer global structure:

  Pretrained ckpt for MEL backbone               ->  destination key
    model.embed_positions.weight                     global_layers_m.{i}.embed_positions.weight  (for every i)
    model.layer.{i}.attention.self.{q,k,v}.*         global_layers_m.{i}.layer.0.attention.self.{q,k,v}.*
    model.layer.{i}.attention.output.dense.*         global_layers_m.{i}.layer.0.attention.output.dense.*
    model.layer.{i}.attention.output.LayerNorm.*     global_layers_m.{i}.layer.0.attention.output.LayerNorm.*
    model.layer.{i}.intermediate.dense.*             global_layers_m.{i}.layer.0.intermediate.dense.*
    model.layer.{i}.output.dense.*                   global_layers_m.{i}.layer.0.output.dense.*
    model.layer.{i}.output.LayerNorm.*               global_layers_m.{i}.layer.0.output.LayerNorm.*
    local_embedding.*                                local_embedding(_m).*
    local_encoder.*                                  local_encoder(_m).*
    local_decoder.*                                  local_decoder(_m).*
    final_decoder.*                                  final_decoder_m.*
    global_sos                                       global_sos_m

  Pretrained ckpt for CHORD backbone -- analogous, mapping to *_c slots.

  Fresh init (no pretrained equivalent):
    cross_attn_m_reads_c.{i}.*  / cross_attn_c_reads_m.{i}.*  (per-layer adapters)
    gates_m.{i}.*  / gates_c.{i}.*                            (per-layer gates)
    token_type_embeddings.*

The per-layer cross-attention adapter parameters are not in any pretrained
ckpt -- they're the only NEW trainable bridge between the two backbones.
At init the adapter output projection is small (set in the model's
__init__) and the gate biases are -10 (sigmoid -> ~0), so the gated
combination o = u_self + sigmoid(gate) * u_cross starts at o ~= u_self.

Run:
    python init_two_pretrained_into_two_backbones_crossattn.py \\
        --ckpt_pretrained pretrained/cp_transformer_pretrained.pt \\
        --size 1 \\
        --out pretrained/two_backbones_crossattn_perlayer_init.pt
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
# Helpers
# ---------------------------------------------------------------------------

def _load_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj:
        sd = obj['state_dict']
    elif isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
        sd = obj['model']
    else:
        sd = obj
    keys = list(sd.keys())
    for prefix in ('net.', 'module.'):
        if all(k.startswith(prefix) for k in keys):
            sd = OrderedDict((k[len(prefix):], v) for k, v in sd.items())
            break
    return sd


def _inspect(sd, label):
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
# Per-layer key mapping
# ---------------------------------------------------------------------------

_PER_LAYER_GLOBAL = re.compile(r'^model\.layer\.(\d+)\.(.+)$')
_GLOBAL_EMBED_POS = re.compile(r'^model\.embed_positions\.(.+)$')


def _dest_keys_for(src_key, modality, untie_local, num_layers):
    """Return list of (dest_key, value_transform) pairs for one source key.
    A single source key can fan out to multiple dest keys (e.g. embed_positions
    needs to be copied into every per-layer encoder's embed_positions).
    """
    # ----- Per-layer global -----
    m = _PER_LAYER_GLOBAL.match(src_key)
    if m:
        layer_idx = int(m.group(1))
        rest = m.group(2)
        return [(f'global_layers_{modality}.{layer_idx}.layer.0.{rest}', None)]

    m = _GLOBAL_EMBED_POS.match(src_key)
    if m:
        rest = m.group(1)
        # Copy into every per-layer encoder's embed_positions.
        return [
            (f'global_layers_{modality}.{i}.embed_positions.{rest}', None)
            for i in range(num_layers)
        ]

    # ----- Local components -----
    if src_key.startswith('local_embedding.'):
        if untie_local:
            return [(src_key.replace(
                'local_embedding.', f'local_embedding_{modality}.',
            ), None)]
        return [(src_key, None)] if modality == 'm' else []

    if src_key.startswith('local_encoder.'):
        if untie_local:
            return [(src_key.replace(
                'local_encoder.', f'local_encoder_{modality}.',
            ), None)]
        return [(src_key, None)] if modality == 'm' else []

    if src_key.startswith('local_decoder.'):
        if untie_local:
            return [(src_key.replace(
                'local_decoder.', f'local_decoder_{modality}.',
            ), None)]
        return [(src_key, None)] if modality == 'm' else []

    # ----- Per-modality output head and SOS -----
    if src_key.startswith('final_decoder.'):
        return [(src_key.replace(
            'final_decoder.', f'final_decoder_{modality}.',
        ), None)]

    if src_key == 'global_sos':
        return [(f'global_sos_{modality}', None)]

    return []


# ---------------------------------------------------------------------------
# Build dest
# ---------------------------------------------------------------------------

def build_dest(model, src_m, src_c, untie_local):
    dest_template = model.state_dict()
    out = OrderedDict(
        (k, v.clone()) for k, v in dest_template.items()
    )
    counts = {
        'copied_m': 0, 'copied_c': 0,
        'replicated_embed_positions_m': 0,
        'replicated_embed_positions_c': 0,
        'skipped_shared_dup': 0,
        'kept_dst_init': 0,
        'missing_in_dst': [],
    }
    num_layers = model.num_layers

    # Apply mel ckpt first.
    for src_key, src_val in src_m.items():
        for dest_key, _ in _dest_keys_for(src_key, 'm', untie_local, num_layers):
            if dest_key not in out:
                counts['missing_in_dst'].append((src_key, dest_key))
                continue
            if src_val.shape != out[dest_key].shape:
                raise ValueError(
                    f'Shape mismatch [mel]: dest {dest_key} {tuple(out[dest_key].shape)} '
                    f'vs src {src_key} {tuple(src_val.shape)}'
                )
            out[dest_key] = src_val.clone()
            if 'embed_positions' in dest_key:
                counts['replicated_embed_positions_m'] += 1
            else:
                counts['copied_m'] += 1

    # Apply chord ckpt second.
    for src_key, src_val in src_c.items():
        for dest_key, _ in _dest_keys_for(src_key, 'c', untie_local, num_layers):
            if dest_key not in out:
                counts['missing_in_dst'].append((src_key, dest_key))
                continue
            # Detect shared-local overlap (mel ckpt already wrote here).
            if (not untie_local) and (
                dest_key.startswith('local_embedding.')
                or dest_key.startswith('local_encoder.')
                or dest_key.startswith('local_decoder.')
            ):
                # mel ckpt already filled this; chord's value is dropped.
                counts['skipped_shared_dup'] += 1
                continue
            if src_val.shape != out[dest_key].shape:
                raise ValueError(
                    f'Shape mismatch [chord]: dest {dest_key} {tuple(out[dest_key].shape)} '
                    f'vs src {src_key} {tuple(src_val.shape)}'
                )
            out[dest_key] = src_val.clone()
            if 'embed_positions' in dest_key:
                counts['replicated_embed_positions_c'] += 1
            else:
                counts['copied_c'] += 1

    # Count "kept dst init": destination keys that no source rule wrote.
    written = set()
    for src_key in src_m.keys():
        for dk, _ in _dest_keys_for(src_key, 'm', untie_local, num_layers):
            written.add(dk)
    for src_key in src_c.keys():
        for dk, _ in _dest_keys_for(src_key, 'c', untie_local, num_layers):
            written.add(dk)
    counts['kept_dst_init'] = sum(1 for k in dest_template if k not in written)

    return out, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_pretrained', help='Single pretrained .pt; '
                    'duplicated into both backbones.')
    ap.add_argument('--ckpt_pretrained_m',
                    help='Overrides --ckpt_pretrained for the mel backbone.')
    ap.add_argument('--ckpt_pretrained_c',
                    help='Overrides --ckpt_pretrained for the chord backbone.')
    ap.add_argument('--size', type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument('--with_velocity', action='store_true', default=False)
    ap.add_argument('--untie_local', action='store_true', default=False)
    ap.add_argument('--crossattn_num_heads', type=int, default=8)
    ap.add_argument('--gate_init_bias', type=float, default=-10.0)
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--out', required=False, default=None)
    args = ap.parse_args()

    ckpt_m_path = args.ckpt_pretrained_m or args.ckpt_pretrained
    ckpt_c_path = args.ckpt_pretrained_c or args.ckpt_pretrained
    if ckpt_m_path is None or ckpt_c_path is None:
        raise SystemExit('Need --ckpt_pretrained OR both '
                         '--ckpt_pretrained_m and --ckpt_pretrained_c.')

    print(f'[load] mel backbone source:   {ckpt_m_path}')
    print(f'[load] chord backbone source: {ckpt_c_path}')
    src_m = _load_state(ckpt_m_path)
    src_c = src_m if ckpt_c_path == ckpt_m_path else _load_state(ckpt_c_path)
    if src_c is src_m:
        print('[load] (same file, will be applied to both backbones)')
    _inspect(src_m, 'source: mel backbone')
    if src_c is not src_m:
        _inspect(src_c, 'source: chord backbone')

    print(f'\n[build] instantiating M2CTwoBackbonesCrossAttn '
          f'(per-layer adapter, size={args.size}, untie_local={args.untie_local})')
    model = M2CTwoBackbonesCrossAttn(
        size=args.size,
        with_velocity=args.with_velocity,
        untie_local=args.untie_local,
        crossattn_num_heads=args.crossattn_num_heads,
        gate_init_bias=args.gate_init_bias,
    )
    _inspect(model.state_dict(), 'destination: M2CTwoBackbonesCrossAttn')

    if args.inspect:
        print('\n[inspect] done.')
        return

    new_sd, counts = build_dest(
        model, src_m, src_c, args.untie_local,
    )

    print('\n=== Mapping report ===')
    print(f'  per-layer global from mel ckpt:   {counts["copied_m"]} per-layer tensors')
    print(f'  per-layer global from chord ckpt: {counts["copied_c"]} per-layer tensors')
    print(f'  embed_positions replicated [m]:   {counts["replicated_embed_positions_m"]}')
    print(f'  embed_positions replicated [c]:   {counts["replicated_embed_positions_c"]}')
    if counts['skipped_shared_dup'] > 0:
        print(f'  shared-local overlap dropped:     {counts["skipped_shared_dup"]}')
    print(f'  kept dst init (cross-attn adapter, gates, type_emb): '
          f'{counts["kept_dst_init"]} keys')

    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    if missing or unexpected:
        print(f'[load] missing={missing} unexpected={unexpected}')
        raise SystemExit(1)
    print('\n[load] state_dict loaded with strict=True')

    out_path = args.out or (
        ckpt_m_path + '.two_backbones_crossattn_perlayer_init.pt'
    )
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
                'variant': 'm2c_two_backbones_crossattn_perlayer',
            },
        },
        out_path,
    )
    print(f'[save] {out_path}')
    print('\nNext: train from init:')
    print(f'  python cp_transformer_m2c_two_backbones_crossattn.py '
          f'--checkpoint_path {out_path} --size {args.size} '
          f'--path_to_dataset data/<your_dataset>.pt --wandb')


if __name__ == '__main__':
    main()
