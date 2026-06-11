"""Warm-start an M2CIntraCrossAttn from a pretrained CP-transformer ckpt.

Same per-layer remap as init_pretrained_into_jointattn.py:
    new.global_layers[i].q_m = new.global_layers[i].q_c = pretrained.Wq
    (same for k, v, o)
    new.global_layers[i].ffn.fc{1,2}.{j} = pretrained.FFN  for j=0..K-1
    new.global_layers[i].ln_attn = pretrained.attn.LN
    new.global_layers[i].ln_ffn  = pretrained.ffn.LN

New parameters introduced by this variant (per-block gate_m, gate_c) are
NOT copied from the pretrained ckpt -- they don't exist in the source.
The M2CIntraCrossAttnLayer constructor bakes them in at:
    gate_m.weight = 0
    gate_m.bias   = -10           # sigmoid(-10) ~ 4.5e-5
    gate_c.weight = 0
    gate_c.bias   = -10
which means at step 0 the cross-stream contribution is ~off and each
modality reproduces what the pretrained block computes on its own
stream in isolation. The gates learn to open during training.

Hard-error if the pretrained's `global_num_layers` does not match the
target's `--global_num_layers`.

Run:
    python init_pretrained_into_intra_cross_attn.py \\
        --pretrained ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt \\
        --output     ckpt/m2c_intra_cross_attn_init/m2c_intra_cross_attn_init.ckpt \\
        --model_size large \\
        --global_num_layers 12 \\
        --moe_num_experts 4 --moe_topk 2
"""

import argparse
import os

import torch

from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn
from init_pretrained_into_jointattn import (
    _count_pretrained_layers, _map_global_key,
)


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
    ap.add_argument('--gate_init_bias', type=float, default=-10.0)
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
            'Refusing to silently slice / pad.'
        )
    print(f'[ok] pretrained has {n_pre} global layers, matches target.')

    print(f'[build] M2CIntraCrossAttn(size={args.model_size}, gnl={gnl}, '
          f'K={args.moe_num_experts}, topk={args.moe_topk}, '
          f'gate_init_bias={args.gate_init_bias})')
    net = M2CIntraCrossAttn(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        gate_init_bias=args.gate_init_bias,
    )
    target_sd = net.state_dict()

    # 1. Pass-through: local encoder/decoder, embeds, etc.
    n_passthru = 0
    for k in list(sd_src.keys()):
        if k in target_sd and target_sd[k].shape == sd_src[k].shape:
            target_sd[k] = sd_src[k]
            n_passthru += 1
    print(f'[init] passthrough copies: {n_passthru}')

    # 2. Per-layer global remap (same keys as jointattn -- per-block
    #    gate_m/gate_c are NOT in the source so they fall through to the
    #    constructor's bias=-10 init).
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
    print(f'[init] global-stack copies: {n_global}')

    # 3. Sanity: any global_layers.* params still at fresh init?
    #    Expected: per-block gate_m/gate_c (weight=0, bias=-10) by design.
    fresh = M2CIntraCrossAttn(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        gate_init_bias=args.gate_init_bias,
    ).state_dict()
    still_default = []
    for k in target_sd:
        if k.startswith('global_layers.') and torch.equal(target_sd[k], fresh[k]):
            still_default.append(k)
    expected_default_substrings = ('gate_m.', 'gate_c.')
    unexpected_default = [
        k for k in still_default
        if not any(s in k for s in expected_default_substrings)
    ]
    if unexpected_default:
        print(f'[warn] {len(unexpected_default)} non-gate global params left '
              f'at fresh init (first few: {unexpected_default[:5]})')
    else:
        print(f'[ok] only gate_m/gate_c are at fresh init '
              f'({len(still_default)} keys, as expected for warm-start)')

    # Sanity-print one gate's sigmoid value.
    g0 = torch.sigmoid(torch.tensor(args.gate_init_bias)).item()
    print(f'[sanity] sigmoid(gate.bias) = sigmoid({args.gate_init_bias}) '
          f'= {g0:.2e}   (cross-stream contribution at step 0)')

    net.load_state_dict(target_sd, strict=False)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f'[save] writing {args.output}')
    torch.save({'state_dict': net.state_dict()}, args.output)
    print('[done]')


if __name__ == '__main__':
    main()
