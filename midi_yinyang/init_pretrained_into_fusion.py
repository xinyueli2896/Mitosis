"""Initialize the M2CFusion (two-backbone) model from a pretrained dense CP
transformer checkpoint.

What this does, key by key:

  1. Copy every per-layer parameter (attention QKV, attention output, attention
     LayerNorm, post-FFN LayerNorm) from the source into BOTH
     `global_roformer_m.layer.{i}.*` AND `global_roformer_c.layer.{i}.*`. The
     two backbones start bitwise-identical.

  2. Replicate the source's dense FFN `(intermediate.dense, output.dense)`
     into ALL N experts of EACH backbone (so 2 * N = 8 destinations for the
     default 4-expert setup). Each expert is identical to the source dense FFN
     at init -> MoE = dense FFN function at step 0.

  3. Initialize each backbone's router (RoFormerTopKRouter.gating) with small
     Gaussian weights and zero bias. Routing noise harmlessly differentiates
     expert gradients during training.

  4. Copy local_encoder, local_decoder, local_embedding, final_decoder,
     global_sos from source 1:1 (these are shared, not per-stream).

  5. Set gate_m.bias = gate_c.bias = -10 (sigmoid -> ~0) so the cross-attention
     adapter starts OFF at init: o_m = u_mm, o_c = u_cc. Together with (1),
     the fusion at init behaves as two parameter-shared copies of the
     pretrained model on the two streams.

  6. token_type_embeddings stays at the M2CFusion model's own init.

Run:
    python init_pretrained_into_fusion.py \\
        --pretrained pretrained/<name>.ckpt \\
        --size 1 \\
        --moe_num_experts 4 --moe_topk 2 \\
        --out pretrained/m2c_fusion_init_from_<name>.pt

Then phase-1 finetune (adapter only, backbones frozen):
    python cp_transformer_m2c_fusion.py \\
        --freeze_backbone \\
        --checkpoint_path pretrained/m2c_fusion_init_from_<name>.pt \\
        ...
"""

# Inject the vendored fork BEFORE anything else imports.
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import re
from collections import OrderedDict

import torch
import torch.nn as nn

from cp_transformer_m2c_fusion import M2CFusion


SRC_GLOBAL_PREFIX = 'model.'
# Two destinations per source global key in the fusion model.
DST_GLOBAL_PREFIXES = ('global_roformer_m.', 'global_roformer_c.')

_LAYER_RE = re.compile(r'^(encoder\.)?layer\.(\d+)\.(.+)$')


# ---------------------------------------------------------------------------
# State-dict utilities
# ---------------------------------------------------------------------------

def load_source_state_dict(path):
    """Open a Lightning .ckpt or plain torch save; return a flat state_dict
    with any 'net.' / 'model.' / 'module.' wrapping prefix stripped."""
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


def inspect(sd, label='state_dict'):
    print(f'\n=== {label} ({len(sd)} tensors) ===')
    groups = {}
    for k, v in sd.items():
        head = '.'.join(k.split('.')[:2])
        groups.setdefault(head, []).append((k, tuple(v.shape) if hasattr(v, 'shape') else None))
    for head, items in groups.items():
        print(f'  {head}/  ({len(items)} tensors)')
        for k, s in items[:3]:
            print(f'    {k}  {s}')
        if len(items) > 3:
            print(f'    ... +{len(items) - 3} more')


def infer_src_hyperparams(sd):
    info = {}
    if 'local_embedding.weight' in sd:
        v = sd['local_embedding.weight']
        info['vocab_size'] = int(v.shape[0])
        info['hidden_size'] = int(v.shape[1])
    max_layer = -1
    for k in sd:
        if not k.startswith(SRC_GLOBAL_PREFIX):
            continue
        m = _LAYER_RE.match(k[len(SRC_GLOBAL_PREFIX):])
        if m:
            max_layer = max(max_layer, int(m.group(2)))
    if max_layer >= 0:
        info['global_num_layers'] = max_layer + 1
    for k, v in sd.items():
        if k.startswith(SRC_GLOBAL_PREFIX) and k.endswith('.intermediate.dense.weight'):
            info['intermediate_size'] = int(v.shape[0])
            break
    return info


def split_global_key(k):
    """Parse a key under the global prefix into (layer_idx, inner) or
    (None, inner) if not a per-layer key. Tolerant of an optional 'encoder.'
    segment between the prefix and 'layer.'."""
    m = _LAYER_RE.match(k)
    if not m:
        return None, k
    return int(m.group(2)), m.group(3)


def _try_get_source(src_sd, layer_i, leaf_path):
    """Try both 'model.encoder.layer.i.<leaf>' and 'model.layer.i.<leaf>'."""
    for inner in (f'encoder.layer.{layer_i}.{leaf_path}',
                  f'layer.{layer_i}.{leaf_path}'):
        full = SRC_GLOBAL_PREFIX + inner
        if full in src_sd:
            return full, src_sd[full]
    return None, None


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def build_dest_state_dict(src_sd, dst_model, n_experts, gate_bias=-10.0,
                          router_init_std=0.02, verbose=True):
    dst_sd_template = dst_model.state_dict()
    out = OrderedDict()
    used_src = set()
    counts = {
        'copied_1to1': 0,
        'replicated_to_experts': 0,
        'copied_to_both_backbones': 0,
        'router_init': 0,
        'gate_init': 0,
        'kept_dst_init': 0,
        'missing_in_src': [],
    }

    for dst_key, dst_val in dst_sd_template.items():
        matched_global_prefix = None
        for pfx in DST_GLOBAL_PREFIXES:
            if dst_key.startswith(pfx):
                matched_global_prefix = pfx
                break

        if matched_global_prefix is not None:
            inner = dst_key[len(matched_global_prefix):]
            layer_idx, rest = split_global_key(inner)

            # Router: tiny Gaussian, zero bias
            if rest and 'moe_custome.RoFormerTopKRouter.gating.weight' in rest:
                out[dst_key] = torch.randn_like(dst_val) * router_init_std
                counts['router_init'] += 1
                continue
            if rest and 'moe_custome.RoFormerTopKRouter.gating.bias' in rest:
                out[dst_key] = torch.zeros_like(dst_val)
                counts['router_init'] += 1
                continue

            # Expert FFN: replicate from source dense FFN
            mexp = re.match(
                r'(encoder\.)?layer\.(\d+)\.moe_custome\.experts\.local_experts\.(\d+)\.(fc1|fc2)\.(weight|bias)$',
                inner,
            )
            if mexp:
                i = int(mexp.group(2))
                fc = mexp.group(4)
                wb = mexp.group(5)
                src_part = 'intermediate.dense' if fc == 'fc1' else 'output.dense'
                src_full, src_val = _try_get_source(src_sd, i, f'{src_part}.{wb}')
                if src_val is not None:
                    out[dst_key] = src_val.clone()
                    used_src.add(src_full)
                    counts['replicated_to_experts'] += 1
                else:
                    out[dst_key] = dst_val.clone()
                    counts['missing_in_src'].append(dst_key)
                continue

            # moe_output LayerNorm  <- source output.LayerNorm
            mln = re.match(
                r'(encoder\.)?layer\.(\d+)\.moe_output\.LayerNorm\.(weight|bias)$',
                inner,
            )
            if mln:
                i = int(mln.group(2))
                wb = mln.group(3)
                src_full, src_val = _try_get_source(src_sd, i, f'output.LayerNorm.{wb}')
                if src_val is not None:
                    out[dst_key] = src_val.clone()
                    used_src.add(src_full)
                    counts['copied_to_both_backbones'] += 1
                else:
                    out[dst_key] = dst_val.clone()
                    counts['missing_in_src'].append(dst_key)
                continue

            # Everything else under the per-layer global path (attention, attn LN)
            # tries to match the source 1:1 ignoring an optional 'encoder.' segment.
            candidates = []
            if inner.startswith('encoder.'):
                candidates.append(SRC_GLOBAL_PREFIX + inner)
                candidates.append(SRC_GLOBAL_PREFIX + inner[len('encoder.'):])
            else:
                candidates.append(SRC_GLOBAL_PREFIX + inner)
                candidates.append(SRC_GLOBAL_PREFIX + 'encoder.' + inner)
            picked = None
            for src_full in candidates:
                if src_full in src_sd:
                    picked = src_full
                    break
            if picked is not None:
                out[dst_key] = src_sd[picked].clone()
                used_src.add(picked)
                counts['copied_to_both_backbones'] += 1
            else:
                out[dst_key] = dst_val.clone()
                counts['missing_in_src'].append(dst_key)
            continue

        # Fusion-only: gates start OFF (sigmoid(-10) ~ 4.5e-5)
        if dst_key in ('gate_m.weight', 'gate_c.weight'):
            out[dst_key] = torch.zeros_like(dst_val)
            counts['gate_init'] += 1
            continue
        if dst_key in ('gate_m.bias', 'gate_c.bias'):
            out[dst_key] = torch.full_like(dst_val, gate_bias)
            counts['gate_init'] += 1
            continue

        if dst_key.startswith('token_type_embeddings.'):
            out[dst_key] = dst_val.clone()
            counts['kept_dst_init'] += 1
            continue

        # Shared params (local_encoder, local_decoder, local_embedding,
        # final_decoder, global_sos): direct key match against source.
        if dst_key in src_sd:
            src_val = src_sd[dst_key]
            if src_val.shape != dst_val.shape:
                raise ValueError(
                    f'Shape mismatch for {dst_key}: '
                    f'pretrained {tuple(src_val.shape)} vs ours {tuple(dst_val.shape)}.'
                )
            out[dst_key] = src_val.clone()
            used_src.add(dst_key)
            counts['copied_1to1'] += 1
        else:
            out[dst_key] = dst_val.clone()
            counts['missing_in_src'].append(dst_key)

    unused_src = [k for k in src_sd if k not in used_src]

    if verbose:
        print('\n=== Mapping report ===')
        for k, v in counts.items():
            if k == 'missing_in_src':
                print(f'  {k}: {len(v)} keys')
                for x in v[:15]:
                    print(f'    - {x}')
                if len(v) > 15:
                    print(f'    ... +{len(v) - 15} more')
            else:
                print(f'  {k}: {v}')
        print(f'  unused source keys: {len(unused_src)}')
        for x in unused_src[:15]:
            print(f'    - {x}')
        if len(unused_src) > 15:
            print(f'    ... +{len(unused_src) - 15} more')

    return out, counts, unused_src


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

@torch.no_grad()
def sanity_check(model, n_experts):
    model.eval()
    H = model.hidden_size
    x = torch.randn(2, 8, H)

    # 1) Both backbones start identical
    sd_m = model.global_roformer_m.state_dict()
    sd_c = model.global_roformer_c.state_dict()
    if set(sd_m.keys()) != set(sd_c.keys()):
        print('[sanity] WARNING: backbones have different state_dict keys')
    diffs = []
    for k in sd_m:
        d = (sd_m[k] - sd_c[k]).abs().max().item() if sd_m[k].dtype.is_floating_point else 0
        if d > 1e-6:
            diffs.append((k, d))
    if diffs:
        print(f'[sanity] WARNING: {len(diffs)} keys differ between backbones; e.g.')
        for k, d in diffs[:5]:
            print(f'    {k}: max-abs-diff {d:.2e}')
    else:
        print('[sanity] backbones bitwise-identical at init: OK')

    # 2) Within each backbone, all experts in layer 0 are identical
    for name, bb in (('m', model.global_roformer_m), ('c', model.global_roformer_c)):
        layer0 = bb.layer[0]
        experts = layer0.moe_custome.experts.local_experts
        for j in range(1, len(experts)):
            if not torch.equal(experts[j].fc1.weight, experts[0].fc1.weight):
                print(f'[sanity] WARNING: backbone {name} layer 0 expert {j} '
                      f'fc1 differs from expert 0')
                break
        else:
            moe_out = layer0.moe_custome(x)
            single = experts[0](x)
            diff = (moe_out - single).abs().max().item()
            print(f'[sanity] backbone {name}: MoE(x) - Expert_0(x) '
                  f'max-abs-diff {diff:.2e}')

    # 3) Gates approximately zero (so adapter is off)
    g_m = torch.sigmoid(model.gate_m.bias).item()
    g_c = torch.sigmoid(model.gate_c.bias).item()
    print(f'[sanity] sigmoid(gate_m.bias)={g_m:.2e}  '
          f'sigmoid(gate_c.bias)={g_c:.2e}  (should be tiny)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', required=True, help='Path to pretrained .ckpt or .pt')
    ap.add_argument('--size', type=int, default=1, choices=[0, 1],
                    help='0 = small (hidden=512), 1 = large (hidden=768). '
                         'The fusion model has only these two sizes.')
    ap.add_argument('--with_velocity', action='store_true')
    ap.add_argument('--moe_num_experts', type=int, default=4)
    ap.add_argument('--moe_topk', type=int, default=2)
    ap.add_argument('--moe_intermediate_size', type=int, default=None)
    ap.add_argument('--global_num_layers', type=int, default=None,
                    help='Default: match pretrained checkpoint.')
    ap.add_argument('--gate_bias', type=float, default=-10.0,
                    help='Bias for gate_m / gate_c. -10 -> sigmoid ~ 4.5e-5.')
    ap.add_argument('--inspect', action='store_true',
                    help='Dump source/dest state_dict layouts and exit.')
    ap.add_argument('--out', required=False, default=None)
    args = ap.parse_args()

    print(f'[load] reading {args.pretrained}')
    src_sd = load_source_state_dict(args.pretrained)
    inspect(src_sd, label='source (pretrained)')
    src_info = infer_src_hyperparams(src_sd)
    print(f'\n[infer] pretrained hyperparams (inferred from shapes): {src_info}')

    if args.global_num_layers is not None:
        gnl = args.global_num_layers
        if 'global_num_layers' in src_info and gnl != src_info['global_num_layers']:
            print(f'[warn] --global_num_layers={gnl} != pretrained '
                  f'{src_info["global_num_layers"]}. Mismatched depth will leave '
                  f'extra destination layers at random init or drop extra '
                  f'pretrained layers.')
    else:
        gnl = src_info.get('global_num_layers', 1)
        print(f'[auto] destination global_num_layers = {gnl} (matched to pretrained)')

    expected_H = 768 if args.size == 1 else 512
    if 'hidden_size' in src_info and src_info['hidden_size'] != expected_H:
        raise SystemExit(
            f'[fail] hidden_size mismatch: pretrained={src_info["hidden_size"]}, '
            f'destination (size={args.size}, large={args.size == 1})={expected_H}. '
            f'M2CFusion supports only the two sizes; extend cp_transformer_m2c_moe.py '
            f'if you need a larger hidden size.')

    print(f'[build] instantiating M2CFusion (size={args.size}, '
          f'with_velocity={args.with_velocity}, global_num_layers={gnl})')
    model = M2CFusion(
        large=(args.size == 1),
        with_velocity=args.with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
    )
    inspect(model.state_dict(), label='destination (M2CFusion)')

    if 'vocab_size' in src_info:
        dst_vocab = model.local_embedding.weight.shape[0]
        if src_info['vocab_size'] != dst_vocab:
            raise SystemExit(
                f'[fail] vocab mismatch: pretrained={src_info["vocab_size"]}, '
                f'destination={dst_vocab}. Tokenizer config differs '
                f'(--with_velocity / max_polyphony / chord vocab).')

    if args.inspect:
        print('\n[inspect] done. Exiting without writing.')
        return

    new_sd, counts, unused = build_dest_state_dict(
        src_sd, model,
        n_experts=args.moe_num_experts,
        gate_bias=args.gate_bias,
    )

    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    if missing or unexpected:
        print(f'[load] missing={missing} unexpected={unexpected}')
        raise SystemExit(1)
    print('[load] state_dict loaded into M2CFusion with strict=True')

    sanity_check(model, args.moe_num_experts)

    out_path = args.out or (args.pretrained + '.m2c_fusion_init.pt')
    print(f'[save] writing {out_path}')
    torch.save(
        {
            'state_dict': new_sd,
            'hyper_parameters': {
                'size': args.size,
                'with_velocity': args.with_velocity,
                'moe_num_experts': args.moe_num_experts,
                'moe_topk': args.moe_topk,
                'moe_intermediate_size': args.moe_intermediate_size,
                'global_num_layers': gnl,
                'gate_bias_init': args.gate_bias,
                'init_from_pretrained': args.pretrained,
                'variant': 'm2c_fusion_two_backbones',
            },
        },
        out_path,
    )
    print('[save] done.')
    print('\nNext: phase-1 finetune (adapter only, both backbones frozen):')
    print(f'  python cp_transformer_m2c_fusion.py --freeze_backbone '
          f'--checkpoint_path {out_path} ...')


if __name__ == '__main__':
    main()
