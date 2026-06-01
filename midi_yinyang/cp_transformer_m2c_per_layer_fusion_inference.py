"""Inference script for the per-layer-fusion M2C MoE model.

Same 5 modes (co / mel2chord / chord2mel / mel_only / chord_only) and same
shift-by-2 sampling loop as cp_transformer_m2c_moe_inference.py -- the
per-layer-fusion model is compatible with that loop because the same-step
cross-stream attention it uses at training only reads ALREADY-SAMPLED tokens
at the same model-timestep (positions 2t and 2t+1 of the inference buffer
both hold the previously sampled m_{t-1} and c_{t-1}).

Only two things differ from the m2c MoE inference:

  1. load_model instantiates M2CPerLayerFusion (loads the stacked
     PerLayerFusionBlock state dict).
  2. gate_off walks model.fusion_blocks[i].gate_m / gate_c (per-block gates)
     instead of the top-level model.gate_m / gate_c that m2c MoE has.

Everything else (frame encoding, mode dispatch, MIDI rendering, folder batch
loop, mel_only / chord_only gate-zeroing) is reused via direct imports from
cp_transformer_m2c_moe_inference.

Run from midi_yinyang/:

    python cp_transformer_m2c_per_layer_fusion_inference.py \\
        --mode co \\
        --ckpt ckpt/<run>/last.ckpt \\
        --melody POP909-Dataset/POP909-melody/001.mid \\
        --chord  POP909-Dataset/POP909-chord/001.mid \\
        --prompt-length 100 --gen-length 384

    python cp_transformer_m2c_per_layer_fusion_inference.py \\
        --mode mel_only --ckpt ckpt/<run>/last.ckpt \\
        --melody POP909-Dataset/POP909-melody/001.mid \\
        --prompt-length 100 --gen-length 384
"""

# Inject the vendored transformers fork BEFORE anything else imports.
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import contextlib
import os
import re

import torch
import torch.nn as nn

from cp_transformer_m2c_per_layer_fusion import M2CPerLayerFusion

# Reuse the m2c MoE inference harness wholesale (modes, action factories,
# sampling loop, MIDI rendering, folder batch driver, single-song driver).
import cp_transformer_m2c_moe_inference as moe_inf
from cp_transformer_m2c_moe_inference import (
    MODES, _ConstNegGate,
    run_one, run_folder,
)


# ---------------------------------------------------------------------------
# gate_off override: per-block gates instead of top-level
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def gate_off(model, which):
    """Zero the cross-attention gate across every fusion block. For
    mel_only -> zero gate_m everywhere (drops u_mc -> o_m, leaves o_m = u_mm).
    Symmetric for chord_only. Restores originals on exit."""
    attr = 'gate_m' if which == 'mel' else 'gate_c'
    device = next(model.parameters()).device
    saved = []
    for block in model.fusion_blocks:
        orig = getattr(block, attr)
        setattr(block, attr, _ConstNegGate().to(device))
        saved.append((block, orig))
    try:
        yield
    finally:
        for block, orig in saved:
            setattr(block, attr, orig)


# Monkey-patch the moe_inf module so its run_one / run_folder see OUR gate_off
# when they call gate_off(model, 'mel' / 'chord') for single-stream modes.
moe_inf.gate_off = gate_off


# ---------------------------------------------------------------------------
# load_model override: instantiate M2CPerLayerFusion, auto-detect gnl
# ---------------------------------------------------------------------------

def _infer_global_num_layers(ckpt_path, ck, model_size):
    """Resolve global_num_layers for M2CPerLayerFusion, in priority order:
      1. ck['hyper_parameters']['global_num_layers']  (training persists this)
      2. _gnl(\\d+)_ substring in filename (default model_name template)
      3. count of fusion_blocks.<N>.* keys in state_dict
      4. size-based fallback (6 small / 12 large)
    """
    if isinstance(ck, dict):
        hp = ck.get('hyper_parameters') or {}
        if isinstance(hp, dict) and hp.get('global_num_layers') is not None:
            return int(hp['global_num_layers']), 'hyper_parameters'

    m = re.search(r'_gnl(\d+)_', os.path.basename(ckpt_path))
    if m:
        return int(m.group(1)), 'filename'

    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    if isinstance(state, dict):
        block_idxs = set()
        pat = re.compile(r'fusion_blocks\.(\d+)\.')
        for k in state.keys():
            mm = pat.search(k)
            if mm:
                block_idxs.add(int(mm.group(1)))
        if block_idxs:
            return max(block_idxs) + 1, 'state_dict_inference'

    return (12 if model_size == 'large' else 6), 'size_default'


def load_model(ckpt_path, model_size='small', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None,
               global_num_layers=None):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, source = _infer_global_num_layers(
            ckpt_path, ck, model_size,
        )
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto-detected from {source})')
    else:
        print(f'[load_model] global_num_layers={global_num_layers} (caller override)')

    net = M2CPerLayerFusion(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        global_num_layers=global_num_layers,
    )
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing keys: {missing[:5]}'
              f'{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected keys: {unexpected[:5]}'
              f'{"..." if len(unexpected) > 5 else ""}')
    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Inference for M2CPerLayerFusion (per-layer gated '
                    'fusion variant of m2c MoE).',
    )
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--melody', help='single melody midi')
    p.add_argument('--chord', help='single chord midi')
    p.add_argument('--mel-folder', help='folder of melody midis')
    p.add_argument('--chord-folder', help='folder of chord midis')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-samples', type=int, default=1)
    p.add_argument('--max-polyphony', type=int, default=4)
    p.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
    p.add_argument('--global-num-layers', type=int, default=None,
                   help='Override global transformer depth. Default: '
                        'auto-detect from checkpoint hyperparameters / '
                        'filename _gnlN_ tag / state_dict block count / '
                        'size-based fallback (6 or 12).')
    args = p.parse_args()

    if args.mel_folder or args.chord_folder:
        if args.melody or args.chord:
            p.error('--melody/--chord cannot be combined with folder args')

    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        with_velocity=args.with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=args.global_num_layers,
    )
    model.save_name = os.path.basename(args.ckpt)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if args.mel_folder or args.chord_folder:
        run_folder(model, args.mode, args.mel_folder, args.chord_folder, args)
    else:
        sid = os.path.splitext(os.path.basename(
            args.melody if args.melody else args.chord
        ))[0]
        run_one(model, args.mode, args.melody, args.chord, args, out_subdir=sid)


if __name__ == '__main__':
    main()
