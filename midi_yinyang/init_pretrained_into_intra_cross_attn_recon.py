"""Warm-start an M2CIntraCrossAttnRecon from the pretrained CP-transformer.

Architecture is identical to M2CIntraCrossAttn (the recon variant only
adds an extra loss term, no new parameters), so the per-layer remap is
the same. Reuses init_pretrained_into_jointattn helpers.

Run:
    python init_pretrained_into_intra_cross_attn_recon.py \\
        --pretrained ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt \\
        --output     ckpt/m2c_intra_cross_attn_recon_init/m2c_intra_cross_attn_recon_init.ckpt \\
        --model_size large --global_num_layers 12 \\
        --moe_num_experts 4 --moe_topk 2
"""

import argparse
import os

import torch

from cp_transformer_m2c_intra_cross_attn_recon import M2CIntraCrossAttnRecon
from init_pretrained_into_jointattn import (
    _count_pretrained_layers, _map_global_key, assert_vocab_matches,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model_size', choices=['small', 'large'], default='large')
    ap.add_argument('--global_num_layers', type=int, default=None)
    ap.add_argument('--moe_num_experts', type=int, default=4)
    ap.add_argument('--moe_topk', type=int, default=2)
    ap.add_argument('--moe_intermediate_size', type=int, default=None)
    ap.add_argument('--gate_init_bias', type=float, default=-10.0)
    ap.add_argument('--recon_weight', type=float, default=1.0,
                    help='Stored on the model for downstream consistency; '
                         'the training CLI flag also sets this so this is '
                         'mostly cosmetic during init.')
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
            f'pretrained has {n_pre} global layers but --global_num_layers={gnl}.'
        )
    print(f'[ok] pretrained has {n_pre} global layers, matches target.')

    print(f'[build] M2CIntraCrossAttnRecon(size={args.model_size}, gnl={gnl}, '
          f'K={args.moe_num_experts}, topk={args.moe_topk}, '
          f'gate_init_bias={args.gate_init_bias}, '
          f'recon_weight={args.recon_weight})')
    net = M2CIntraCrossAttnRecon(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        gate_init_bias=args.gate_init_bias,
        recon_weight=args.recon_weight,
    )
    target_sd = net.state_dict()
    assert_vocab_matches(sd_src, target_sd)

    n_passthru = 0
    for k in list(sd_src.keys()):
        if k in target_sd and target_sd[k].shape == sd_src[k].shape:
            target_sd[k] = sd_src[k]
            n_passthru += 1
    print(f'[init] passthrough copies: {n_passthru}')

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

    fresh = M2CIntraCrossAttnRecon(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        gate_init_bias=args.gate_init_bias,
        recon_weight=args.recon_weight,
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
              f'({len(still_default)} keys, expected for warm-start)')

    g0 = torch.sigmoid(torch.tensor(args.gate_init_bias)).item()
    print(f'[sanity] sigmoid(gate.bias) = sigmoid({args.gate_init_bias}) '
          f'= {g0:.2e}')

    net.load_state_dict(target_sd, strict=False)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f'[save] writing {args.output}')
    torch.save({'state_dict': net.state_dict()}, args.output)
    print('[done]')


if __name__ == '__main__':
    main()
