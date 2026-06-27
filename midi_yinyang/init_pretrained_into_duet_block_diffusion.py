"""Warm-start an M2CDuetBlockDiffusion (A.3) from a pretrained single-stream
CP-transformer.

Same per-layer remap as init_pretrained_into_duet_block.py (A.2): per-modality
Q/K/V/O duplicated from the single-stream Q/K/V/O, MoE experts replicated
from the single dense FFN, etc. The A.3-specific extras land at fresh init:

  * gate_m, gate_c (cross gates):   bias = -10, weight = 0 -> silent.
  * gate_fm, gate_fc (frame gates): bias = -10, weight = 0 -> silent.
  * mask_m_emb, mask_c_emb (query-slot embeddings): zero-initialised.
  * k_emb_m.weight, k_emb_c.weight (noise-level embeddings):
        zero-initialised in M2CDuetBlockDiffusion.__init__.

At step 0 with gates silent, mask embeddings zero, and k-embeddings zero,
the AR forward through the clean stream reproduces the pretrained
single-stream model's behaviour exactly. The query slots and the
diffusion training signal are both cold-start.

Why warm-start from the pretrained base instead of an A.2 ckpt:
A.2 only ever saw k_m = k_c = K during training, so its query slots,
frame gate, and mask embeddings have been optimised for the
fully-masked regime exclusively. Warm-starting A.3 from A.2 would
inherit that single-regime bias (and the near-EOS query-slot collapse
that motivated A.3 in the first place). Starting from the neutral
single-stream pretrained ckpt lets the diffusion objective train the
query-slot pathway across the full noise spectrum from a clean init.

Run:
    python init_pretrained_into_duet_block_diffusion.py \\
        --pretrained ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt \\
        --output     ckpt/m2c_duet_block_diffusion_init/m2c_duet_block_diffusion_init.ckpt \\
        --model_size large --global_num_layers 12 \\
        --moe_num_experts 4 --moe_topk 2 \\
        --diffusion_K 4
"""

import argparse
import os

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion
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
    ap.add_argument('--query_loss_weight', type=float, default=1.0)
    ap.add_argument('--diffusion_K', type=int, default=4,
                    help='Number of noise-level bins for the discrete-diffusion '
                         'training at the query slots. Stored in the ckpt so '
                         'inference can recover it. Default 4.')
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

    print(f'[build] M2CDuetBlockDiffusion(size={args.model_size}, gnl={gnl}, '
          f'K={args.moe_num_experts}, topk={args.moe_topk}, '
          f'gate_init_bias={args.gate_init_bias}, '
          f'query_loss_weight={args.query_loss_weight}, '
          f'diffusion_K={args.diffusion_K})')
    net = M2CDuetBlockDiffusion(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        gate_init_bias=args.gate_init_bias,
        query_loss_weight=args.query_loss_weight,
        diffusion_K=args.diffusion_K,
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

    fresh = M2CDuetBlockDiffusion(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        gate_init_bias=args.gate_init_bias,
        query_loss_weight=args.query_loss_weight,
        diffusion_K=args.diffusion_K,
    ).state_dict()
    still_default = []
    for k in target_sd:
        if k.startswith('global_layers.') and torch.equal(target_sd[k], fresh[k]):
            still_default.append(k)
    expected_default_substrings = ('gate_m.', 'gate_c.', 'gate_fm.', 'gate_fc.')
    unexpected_default = [
        k for k in still_default
        if not any(s in k for s in expected_default_substrings)
    ]
    if unexpected_default:
        print(f'[warn] {len(unexpected_default)} non-gate global params left '
              f'at fresh init (first few: {unexpected_default[:5]})')
    else:
        print(f'[ok] only gate params left at fresh init '
              f'({len(still_default)} keys, expected for warm-start)')

    g0 = torch.sigmoid(torch.tensor(args.gate_init_bias)).item()
    print(f'[sanity] sigmoid(gate.bias) = sigmoid({args.gate_init_bias}) = {g0:.2e}')
    print(f'[sanity] mask_m_emb zero-init? '
          f'{torch.all(target_sd["mask_m_emb"] == 0).item()}')
    print(f'[sanity] mask_c_emb zero-init? '
          f'{torch.all(target_sd["mask_c_emb"] == 0).item()}')
    print(f'[sanity] k_emb_m zero-init? '
          f'{torch.all(target_sd["k_emb_m.weight"] == 0).item()}')
    print(f'[sanity] k_emb_c zero-init? '
          f'{torch.all(target_sd["k_emb_c.weight"] == 0).item()}')

    net.load_state_dict(target_sd, strict=False)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f'[save] writing {args.output}')
    torch.save({'state_dict': net.state_dict()}, args.output)
    print('[done]')


if __name__ == '__main__':
    main()
