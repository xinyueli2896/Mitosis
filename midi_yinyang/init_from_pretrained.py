"""Initialize the m2c MoE fusion model from a pretrained dense CP transformer.

Strategy (matches the "two parameter-shared single-modality self-attention
models joined by a gated cross-attention adapter" view):

  1. Copy every layer's attention and LayerNorm 1:1 (same RoFormer architecture).
  2. Replicate the pretrained dense FFN -- (intermediate.dense, output.dense) --
     into ALL N MoE experts (fc1, fc2). At init every expert is identical, so
     the MoE block returns sum_j topk_weight_j * Expert_j(x) = Expert_0(x)
     because the top-k weights are softmax-normalized to sum to 1. The MoE
     block at step 0 is therefore mathematically equal to the dense FFN, no
     matter what the router decides.
  3. Initialize the router with small Gaussian weights and zero bias.
     Routing noise harmlessly differentiates expert gradients during training,
     letting them specialize.
  4. Copy local_encoder, local_decoder, local_embedding, final_decoder,
     global_sos 1:1 (these match the pretrained model directly).
  5. Set gate_m.bias = gate_c.bias = -10 (sigmoid -> ~0) so the cross-attention
     adapter starts OFF: o_m = u_mm, o_c = u_cc. The fusion at init behaves as
     two parameter-shared independent copies of the pretrained model on the
     two streams.
  6. token_type_embeddings stays at the m2c model's own init (this is the
     fusion's "which modality is this slot" signal -- no pretrained equivalent).

Run:
    python init_from_pretrained.py \\
        --pretrained pretrained/<name>.ckpt \\
        --size 1 \\
        --with_velocity \\
        --moe_num_experts 4 --moe_topk 2 \\
        --out pretrained/m2c_moe_init_from_<name>.pt

Then use --checkpoint_path pretrained/m2c_moe_init_from_<name>.pt in any of
the m2c training scripts (mask / samestep / original).

Caveats:
- Assumes the pretrained ckpt was trained with the same RoFormerSymbolicTransformer
  family from this repo (cp_transformer.py). If it's a different codebase, the
  --inspect option dumps source keys so you can adapt the rename map.
- Vocab/tokenizer must match (controlled by --with_velocity). The script will
  refuse to truncate or pad the local_embedding silently; if vocab sizes
  differ, it errors out.
- Hidden size and num_layers are inferred from the pretrained weights' shapes
  and must match --size. Mismatch -> error.
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

from cp_transformer_m2c_moe import RoFormerSymbolicTransformer


# ---------------------------------------------------------------------------
# State-dict utilities
# ---------------------------------------------------------------------------

def load_source_state_dict(path):
    """Open a Lightning .ckpt or plain torch save; return a flat state_dict
    with any 'model.' wrapping prefix stripped."""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    # Common containers
    if isinstance(obj, dict) and 'state_dict' in obj:
        sd = obj['state_dict']
    elif isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
        sd = obj['model']
    else:
        sd = obj
    # Strip a common Lightning wrapper prefix like 'net.' or 'model.' if every
    # key shares it. (Some training scripts wrap the model under self.net.)
    keys = list(sd.keys())
    for prefix in ('net.', 'model.', 'module.'):
        if all(k.startswith(prefix) for k in keys):
            sd = OrderedDict((k[len(prefix):], v) for k, v in sd.items())
            break
    return sd


def inspect(sd, label='state_dict'):
    """Print key layout grouped by prefix for human inspection."""
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
    """Look at shapes to guess hidden_size, num_layers, intermediate_size, vocab."""
    info = {}
    # hidden_size from local_embedding.weight: [vocab, H]
    for k in ('local_embedding.weight',):
        if k in sd:
            v = sd[k]
            info['vocab_size'] = int(v.shape[0])
            info['hidden_size'] = int(v.shape[1])
            break
    # num_layers from highest layer index in model.*
    max_layer = -1
    for k in sd:
        if not k.startswith('model.'):
            continue
        m = _LAYER_RE.match(k[len('model.'):])
        if m:
            max_layer = max(max_layer, int(m.group(2)))
    if max_layer >= 0:
        info['global_num_layers'] = max_layer + 1
    # intermediate_size from model.*.intermediate.dense.weight: [I, H]
    for k, v in sd.items():
        if k.startswith('model.') and k.endswith('.intermediate.dense.weight'):
            info['intermediate_size'] = int(v.shape[0])
            break
    return info


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

# Source attribute name for the global RoFormer in the pretrained ckpt.
# In cp_transformer.py this is `self.model` (RoFormerEncoder). The destination
# in cp_transformer_m2c_moe.py is `self.global_roformer`.
SRC_GLOBAL_PREFIX = 'model.'
DST_GLOBAL_PREFIX = 'global_roformer.'

# Per-layer dense FFN keys in the source -> per-expert MoE keys in dest.
# Source (HF RoFormer dense):
#   model.encoder.layer.{i}.intermediate.dense.weight   (H -> I)
#   model.encoder.layer.{i}.intermediate.dense.bias
#   model.encoder.layer.{i}.output.dense.weight         (I -> H)
#   model.encoder.layer.{i}.output.dense.bias
#   model.encoder.layer.{i}.output.LayerNorm.weight
#   model.encoder.layer.{i}.output.LayerNorm.bias
# Destination (our MoE fork):
#   global_roformer.encoder.layer.{i}.moe_custome.experts.local_experts.{j}.fc1.weight
#   global_roformer.encoder.layer.{i}.moe_custome.experts.local_experts.{j}.fc1.bias
#   global_roformer.encoder.layer.{i}.moe_custome.experts.local_experts.{j}.fc2.weight
#   global_roformer.encoder.layer.{i}.moe_custome.experts.local_experts.{j}.fc2.bias
#   global_roformer.encoder.layer.{i}.moe_output.LayerNorm.weight
#   global_roformer.encoder.layer.{i}.moe_output.LayerNorm.bias

_LAYER_RE = re.compile(r'^(encoder\.)?layer\.(\d+)\.(.+)$')


def split_global_key(k):
    """Parse a key under the global prefix into (layer_idx, inner) or
    (None, inner) if not a per-layer key. Handles both 'encoder.layer.i.X'
    and 'layer.i.X' (RoFormerEncoder bare vs wrapped under a model)."""
    m = _LAYER_RE.match(k)
    if not m:
        return None, k
    return int(m.group(2)), m.group(3)


def build_dest_state_dict(src_sd, dst_model, n_experts, gate_bias=-10.0,
                          router_init_std=0.02, verbose=True):
    """Construct a state_dict for dst_model (M2C MoE) populated from src_sd.

    Returns: (new_state_dict, report) where report is a dict of stats.
    """
    dst_sd_template = dst_model.state_dict()  # destination shapes / init values
    out = OrderedDict()
    used_src = set()
    counts = {'copied_1to1': 0, 'replicated_to_experts': 0,
              'router_init': 0, 'gate_init': 0, 'kept_dst_init': 0,
              'missing_in_src': []}

    # ---- 1. Pass through every destination key, decide its source ----
    for dst_key, dst_val in dst_sd_template.items():
        if dst_key.startswith(DST_GLOBAL_PREFIX):
            inner = dst_key[len(DST_GLOBAL_PREFIX):]
            layer_idx, rest = split_global_key(inner)

            # ----- Router: tiny init -----
            if rest and 'moe_custome.RoFormerTopKRouter.gating.weight' in rest:
                out[dst_key] = torch.randn_like(dst_val) * router_init_std
                counts['router_init'] += 1
                continue
            if rest and 'moe_custome.RoFormerTopKRouter.gating.bias' in rest:
                out[dst_key] = torch.zeros_like(dst_val)
                counts['router_init'] += 1
                continue

            # ----- Expert FFN: replicate from dense pretrained -----
            mexp = re.match(
                r'(encoder\.)?layer\.(\d+)\.moe_custome\.experts\.local_experts\.(\d+)\.(fc1|fc2)\.(weight|bias)$',
                inner,
            )
            if mexp:
                i = int(mexp.group(2))
                fc = mexp.group(4)   # fc1 or fc2
                wb = mexp.group(5)   # weight or bias
                src_part = 'intermediate.dense' if fc == 'fc1' else 'output.dense'
                # Try both possible source prefixes
                for src_inner in (f'encoder.layer.{i}.{src_part}.{wb}',
                                  f'layer.{i}.{src_part}.{wb}'):
                    src_full = SRC_GLOBAL_PREFIX + src_inner
                    if src_full in src_sd:
                        out[dst_key] = src_sd[src_full].clone()
                        used_src.add(src_full)
                        counts['replicated_to_experts'] += 1
                        break
                else:
                    out[dst_key] = dst_val.clone()
                    counts['missing_in_src'].append(dst_key)
                continue

            # ----- moe_output LayerNorm <- source output.LayerNorm -----
            mln = re.match(
                r'(encoder\.)?layer\.(\d+)\.moe_output\.LayerNorm\.(weight|bias)$',
                inner,
            )
            if mln:
                i = int(mln.group(2))
                wb = mln.group(3)
                for src_inner in (f'encoder.layer.{i}.output.LayerNorm.{wb}',
                                  f'layer.{i}.output.LayerNorm.{wb}'):
                    src_full = SRC_GLOBAL_PREFIX + src_inner
                    if src_full in src_sd:
                        out[dst_key] = src_sd[src_full].clone()
                        used_src.add(src_full)
                        counts['copied_1to1'] += 1
                        break
                else:
                    out[dst_key] = dst_val.clone()
                    counts['missing_in_src'].append(dst_key)
                continue

            # ----- Other per-layer keys (attention, attention LayerNorm) 1:1 ----
            # Try with and without an extra 'encoder.' segment on each side.
            candidates = []
            if inner.startswith('encoder.'):
                candidates.append(SRC_GLOBAL_PREFIX + inner)
                candidates.append(SRC_GLOBAL_PREFIX + inner[len('encoder.'):])
            else:
                candidates.append(SRC_GLOBAL_PREFIX + inner)
                candidates.append(SRC_GLOBAL_PREFIX + 'encoder.' + inner)
            for src_full in candidates:
                if src_full in src_sd:
                    out[dst_key] = src_sd[src_full].clone()
                    used_src.add(src_full)
                    counts['copied_1to1'] += 1
                    break
            else:
                out[dst_key] = dst_val.clone()
                counts['missing_in_src'].append(dst_key)
            continue

        # ----- Fusion-only: gates start OFF (sigmoid(-10) ~ 4.5e-5) -----
        if dst_key in ('gate_m.weight', 'gate_c.weight'):
            out[dst_key] = torch.zeros_like(dst_val)
            counts['gate_init'] += 1
            continue
        if dst_key in ('gate_m.bias', 'gate_c.bias'):
            out[dst_key] = torch.full_like(dst_val, gate_bias)
            counts['gate_init'] += 1
            continue

        # ----- token_type_embeddings: keep dst init (no pretrained match) ----
        if dst_key.startswith('token_type_embeddings.'):
            out[dst_key] = dst_val.clone()
            counts['kept_dst_init'] += 1
            continue

        # ----- Everything else (local_*, final_decoder, global_sos): 1:1 ----
        if dst_key in src_sd:
            src_val = src_sd[dst_key]
            if src_val.shape != dst_val.shape:
                raise ValueError(
                    f'Shape mismatch for {dst_key}: '
                    f'pretrained {tuple(src_val.shape)} vs ours {tuple(dst_val.shape)}. '
                    f'Check --size / --with_velocity / vocab compatibility.'
                )
            out[dst_key] = src_val.clone()
            used_src.add(dst_key)
            counts['copied_1to1'] += 1
        else:
            out[dst_key] = dst_val.clone()
            counts['missing_in_src'].append(dst_key)

    # ---- 2. Report unused source keys (potential dropped weights) ----
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
# Sanity check
# ---------------------------------------------------------------------------

@torch.no_grad()
def sanity_check_moe_equals_dense(model, n_experts):
    """With all experts identical and router weights tiny, the MoE block should
    produce the same output as if the dense FFN were applied directly.
    Verifies that the replication preserved the pretrained function."""
    model.eval()
    H = model.hidden_size
    x = torch.randn(2, 8, H)
    layer0 = model.global_roformer.encoder.layer[0]  # encoder wraps layer list

    # All experts identical?
    experts = layer0.moe_custome.experts.local_experts
    e0_w = experts[0].fc1.weight
    e0_b = experts[0].fc2.bias
    for j in range(1, len(experts)):
        assert torch.equal(experts[j].fc1.weight, e0_w), f'expert {j} fc1 differs from expert 0'
        assert torch.equal(experts[j].fc2.bias, e0_b), f'expert {j} fc2.bias differs from expert 0'
    print(f'[sanity] all {n_experts} experts are identical at init: OK')

    # MoE output == single-expert output (within float precision)
    moe_out = layer0.moe_custome(x)
    single = experts[0](x)
    diff = (moe_out - single).abs().max().item()
    print(f'[sanity] MoE(x) - Expert_0(x) max-abs-diff: {diff:.2e}')
    if diff > 1e-4:
        print('[sanity] WARNING: MoE != dense path. Did router/topk degenerate?')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', required=True, help='Path to pretrained .ckpt or .pt')
    ap.add_argument('--size', type=int, default=1, choices=[0, 1],
                    help='0 = small (hidden=512, heads=8), 1 = large (hidden=768, heads=12). '
                         'Our m2c MoE supports only these two; the original cp_transformer.py '
                         'pretrained models at sizes 2/3 (hidden=1024/1280) are not loadable.')
    ap.add_argument('--with_velocity', action='store_true')
    ap.add_argument('--moe_num_experts', type=int, default=4)
    ap.add_argument('--moe_topk', type=int, default=2)
    ap.add_argument('--moe_intermediate_size', type=int, default=None)
    ap.add_argument('--global_num_layers', type=int, default=None,
                    help='Override destination global layer count. Default: '
                         'match the pretrained checkpoint exactly (recommended). '
                         'Pass 1 to keep the m2c default (drops most pretrained layers).')
    ap.add_argument('--gate_bias', type=float, default=-10.0,
                    help='Bias for gate_m/gate_c. -10 -> sigmoid ~= 4.5e-5 (adapter ~off).')
    ap.add_argument('--inspect', action='store_true',
                    help='Just print source/dest state_dict keys and exit.')
    ap.add_argument('--out', required=False, default=None,
                    help='Output .pt path. Defaults to <pretrained>.m2c_moe_init.pt')
    args = ap.parse_args()

    print(f'[load] reading {args.pretrained}')
    src_sd = load_source_state_dict(args.pretrained)
    inspect(src_sd, label='source (pretrained)')
    src_info = infer_src_hyperparams(src_sd)
    print(f'\n[infer] pretrained hyperparams (inferred from shapes): {src_info}')

    # Decide destination global_num_layers
    if args.global_num_layers is not None:
        gnl = args.global_num_layers
        if 'global_num_layers' in src_info and gnl != src_info['global_num_layers']:
            print(f'[warn] --global_num_layers={gnl} != pretrained {src_info["global_num_layers"]}. '
                  f'Source layers beyond {gnl - 1} will be dropped; '
                  f'destination layers >= pretrained count will keep random init.')
    else:
        gnl = src_info.get('global_num_layers', 1)
        print(f'[auto] destination global_num_layers = {gnl} (matched to pretrained)')

    # Sanity check hidden size against the size flag's implied destination.
    # Our m2c MoE has exactly two sizes: large=False -> 512, large=True -> 768.
    expected_H = 768 if args.size == 1 else 512
    if 'hidden_size' in src_info and src_info['hidden_size'] != expected_H:
        raise SystemExit(
            f'[fail] hidden_size mismatch: pretrained={src_info["hidden_size"]}, '
            f'destination (size={args.size}, large={args.size == 1})={expected_H}. '
            f'If the pretrained ckpt is size 2/3 (hidden=1024/1280) from '
            f'cp_transformer.py, our m2c MoE does not support that hidden size; '
            f'either extend cp_transformer_m2c_moe.py to add the larger size, '
            f'or pick a smaller pretrained checkpoint.')

    print(f'[build] instantiating m2c MoE model (size={args.size}, '
          f'with_velocity={args.with_velocity}, global_num_layers={gnl})')
    model = RoFormerSymbolicTransformer(
        large=(args.size == 1),
        with_velocity=args.with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
    )
    # Vocab guard
    if 'vocab_size' in src_info:
        dst_vocab = model.local_embedding.weight.shape[0]
        if src_info['vocab_size'] != dst_vocab:
            raise SystemExit(
                f'[fail] vocab mismatch: pretrained={src_info["vocab_size"]}, '
                f'destination={dst_vocab}. Tokenizer config differs '
                f'(--with_velocity / max_polyphony / chord vocab). Aligning vocabs '
                f'requires explicit subset/extension and is not implicit.')
    inspect(model.state_dict(), label='destination (m2c MoE)')

    if args.inspect:
        print('\n[inspect] done. Exiting without writing.')
        return

    new_sd, counts, unused = build_dest_state_dict(
        src_sd, model,
        n_experts=args.moe_num_experts,
        gate_bias=args.gate_bias,
    )

    # Load into the model (strict=True now that every dest key is present)
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    if missing or unexpected:
        print(f'[load] missing={missing} unexpected={unexpected}')
        raise SystemExit(1)
    print('[load] state_dict loaded into model with strict=True')

    sanity_check_moe_equals_dense(model, args.moe_num_experts)

    out_path = args.out or (args.pretrained + '.m2c_moe_init.pt')
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
                'init_from_pretrained': args.pretrained,
                'gate_bias_init': args.gate_bias,
            },
        },
        out_path,
    )
    print('[save] done.')
    print('\nNext steps:')
    print(f'  python cp_transformer_m2c_moe.py     --checkpoint_path {out_path}  [other args]')
    print(f'  python cp_transformer_m2c_samestep.py --checkpoint_path {out_path}  [other args]')
    print(f'  python cp_transformer_m2c_moe_mask.py --checkpoint_path {out_path}  [other args]')


if __name__ == '__main__':
    main()
