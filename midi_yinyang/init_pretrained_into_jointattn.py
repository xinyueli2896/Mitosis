"""Warm-start an M2CJointAttn from a single pretrained CP-transformer
checkpoint.

For each pretrained global layer i (vendored RoFormer fork):
    new.global_layers[i].q_m = new.global_layers[i].q_c = pretrained.Wq
    same for k, v, o (per-head fan-out matches)
    new.global_layers[i].ffn.experts[j] = pretrained.FFN  for j=0..K-1
    new.global_layers[i].ln_attn_m, ln_attn_c, ln_ffn = pretrained.LN_*

Plus: copy local encoder/decoder, embeddings, tokenizer-side modules
verbatim from the pretrained ckpt (M2CMoE inherits these unchanged).

At step 0 this gives:
    M2CJointAttn(interleaved_input)  ==  pretrained_one_backbone(interleaved_input)

Hard-error if the pretrained's `global_num_layers` does not match the
target's `--global_num_layers` (otherwise we'd be silently slicing a
mismatched stack, which is what bit you on the cross-attn flow).

Run:
    python init_pretrained_into_jointattn.py \\
        --pretrained pretrained/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt \\
        --output     ckpt/m2c_jointattn_init.ckpt \\
        --model_size large \\
        --global_num_layers 12 \\
        --moe_num_experts 4 --moe_topk 2
"""

import argparse
import os
import re

import torch

from cp_transformer_m2c_jointattn import M2CJointAttn


# Pretrained CP-transformer stores the global RoFormer stack under
# `model.layer.{N}.*` (see inspect_pretrained_ckpt.py output). The standard
# HF RoFormer naming inside each layer:
#     attention.self.query/key/value.{weight,bias}
#     attention.output.dense.{weight,bias}            <- Wo
#     attention.output.LayerNorm.{weight,bias}        <- post-attn LN
#     intermediate.dense.{weight,bias}                <- FFN fc1
#     output.dense.{weight,bias}                      <- FFN fc2
#     output.LayerNorm.{weight,bias}                  <- post-FFN LN

_SRC_PREFIX = r'^model\.layer\.(\d+)\.'

_PER_LAYER_ATTN = re.compile(
    _SRC_PREFIX + r'attention\.(self|output)\.(.+)$'
)
_PER_LAYER_FFN_FC1 = re.compile(
    _SRC_PREFIX + r'intermediate\.dense\.(weight|bias)$'
)
_PER_LAYER_FFN_FC2 = re.compile(
    _SRC_PREFIX + r'output\.dense\.(weight|bias)$'
)
_PER_LAYER_LN_FFN = re.compile(
    _SRC_PREFIX + r'output\.LayerNorm\.(weight|bias)$'
)


def _count_pretrained_layers(sd):
    layer_ids = set()
    for k in sd.keys():
        m = re.match(_SRC_PREFIX, k)
        if m:
            layer_ids.add(int(m.group(1)))
    return max(layer_ids) + 1 if layer_ids else 0


def _map_global_key(src_key, src_val, moe_num_experts):
    """Return list of (target_key, transformed_tensor_or_None) pairs. None
    means: copy src_val directly. Returns [] if the key has no target.

    Source schema (pretrained CP-transformer one-backbone):
        model.layer.{N}.attention.self.{query,key,value}.{weight,bias}
        model.layer.{N}.attention.output.dense.{weight,bias}
        model.layer.{N}.attention.output.LayerNorm.{weight,bias}
        model.layer.{N}.intermediate.dense.{weight,bias}     (FFN fc1)
        model.layer.{N}.output.dense.{weight,bias}           (FFN fc2)
        model.layer.{N}.output.LayerNorm.{weight,bias}

    Target schema (M2CJointAttn):
        global_layers.{N}.{q,k,v,o}_{m,c}.{weight,bias}    (per-modality)
        global_layers.{N}.ln_attn.{weight,bias}             (shared)
        global_layers.{N}.ln_ffn.{weight,bias}              (shared)
        global_layers.{N}.ffn.fc1.{j}.{weight,bias}         (replicated across K experts)
        global_layers.{N}.ffn.fc2.{j}.{weight,bias}
    """
    m = _PER_LAYER_ATTN.match(src_key)
    if m:
        layer = int(m.group(1))
        which = m.group(2)         # 'self' or 'output'
        tail = m.group(3)          # e.g. 'query.weight'
        head, wb = tail.split('.', 1)
        if which == 'self':
            # self.{query,key,value}.{weight,bias} -> q/k/v projections,
            # replicated into both mel and chord branches.
            short = {'query': 'q', 'key': 'k', 'value': 'v'}[head]
            return [
                (f'global_layers.{layer}.{short}_m.{wb}', None),
                (f'global_layers.{layer}.{short}_c.{wb}', None),
            ]
        # which == 'output': either Wo (dense) or post-attn LN.
        if head == 'dense':
            return [
                (f'global_layers.{layer}.o_m.{wb}', None),
                (f'global_layers.{layer}.o_c.{wb}', None),
            ]
        if head == 'LayerNorm':
            return [(f'global_layers.{layer}.ln_attn.{wb}', None)]
        return []

    m = _PER_LAYER_FFN_FC1.match(src_key)
    if m:
        layer = int(m.group(1))
        wb = m.group(2)
        if moe_num_experts > 1:
            return [
                (f'global_layers.{layer}.ffn.fc1.{j}.{wb}', None)
                for j in range(moe_num_experts)
            ]
        # Dense fallback: nn.Sequential[Linear(0), GELU(1), Linear(2)]
        return [(f'global_layers.{layer}.ffn.0.{wb}', None)]

    m = _PER_LAYER_FFN_FC2.match(src_key)
    if m:
        layer = int(m.group(1))
        wb = m.group(2)
        if moe_num_experts > 1:
            return [
                (f'global_layers.{layer}.ffn.fc2.{j}.{wb}', None)
                for j in range(moe_num_experts)
            ]
        return [(f'global_layers.{layer}.ffn.2.{wb}', None)]

    m = _PER_LAYER_LN_FFN.match(src_key)
    if m:
        layer = int(m.group(1))
        wb = m.group(2)
        return [(f'global_layers.{layer}.ln_ffn.{wb}', None)]

    if src_key == 'global_sos':
        return [('global_sos', None)]

    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model_size', choices=['small', 'large'], default='large')
    ap.add_argument('--global_num_layers', type=int, default=None,
                    help='Must match pretrained layer count. Default: 12 if '
                         'large else 6.')
    ap.add_argument('--moe_num_experts', type=int, default=4)
    ap.add_argument('--moe_topk', type=int, default=2)
    ap.add_argument('--moe_intermediate_size', type=int, default=None)
    args = ap.parse_args()

    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    print(f'[load] pretrained: {args.pretrained}')
    sd_src = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    if isinstance(sd_src, dict) and 'state_dict' in sd_src:
        sd_src = sd_src['state_dict']

    n_pre = _count_pretrained_layers(sd_src)
    if n_pre != gnl:
        raise SystemExit(
            f'pretrained has {n_pre} global layers but --global_num_layers={gnl}. '
            'Refusing to silently slice / pad. Fix --global_num_layers or pick '
            'a matching pretrained ckpt.'
        )
    print(f'[ok] pretrained has {n_pre} global layers, matches target.')

    print(f'[build] M2CJointAttn(size={args.model_size}, gnl={gnl}, '
          f'K={args.moe_num_experts}, topk={args.moe_topk})')
    net = M2CJointAttn(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
    )
    target_sd = net.state_dict()

    # 1. Pass-through copy: anything that already matches by key gets copied.
    #    This covers local_encoder/local_decoder/local_embedding/
    #    token_type_embeddings/final_decoder/gate_m/gate_c, all of which are
    #    inherited from M2CMoE unchanged.
    n_passthru = 0
    for k in list(sd_src.keys()):
        if k in target_sd and target_sd[k].shape == sd_src[k].shape:
            target_sd[k] = sd_src[k]
            n_passthru += 1
    print(f'[init] passthrough copies (local enc/dec, embeds, etc.): {n_passthru}')

    # 2. Per-layer global remap.
    n_global = 0
    for src_key, src_val in sd_src.items():
        targets = _map_global_key(src_key, src_val, args.moe_num_experts)
        for tgt_key, transformed in targets:
            if tgt_key not in target_sd:
                continue
            v = transformed if transformed is not None else src_val
            if target_sd[tgt_key].shape != v.shape:
                print(f'[skip] shape mismatch for {tgt_key}: '
                      f'{tuple(target_sd[tgt_key].shape)} != {tuple(v.shape)}')
                continue
            target_sd[tgt_key] = v.clone()
            n_global += 1
    print(f'[init] global-stack copies (per-modality Q/K/V/O + MoE FFN + LN): '
          f'{n_global}')

    # 3. modality_type_emb is initialized to zero in the constructor so the
    #    very first forward pass exactly matches pretrained on each modality
    #    individually (since at step 0 Q_m = Q_c = pretrained.Wq etc.).
    #    Nothing to copy.

    # 4. Sanity: keys still at their nn.init defaults
    missing = []
    for k in target_sd:
        # Heuristic: anything in global_layers.* that wasn't set should warn.
        if k.startswith('global_layers.') and torch.equal(
            target_sd[k], net.state_dict()[k]
        ):
            missing.append(k)
    if missing:
        print(f'[warn] {len(missing)} target params left at fresh init '
              f'(first few: {missing[:5]})')

    net.load_state_dict(target_sd, strict=False)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f'[save] writing {args.output}')
    torch.save({'state_dict': net.state_dict()}, args.output)
    print('[done]')


if __name__ == '__main__':
    main()
