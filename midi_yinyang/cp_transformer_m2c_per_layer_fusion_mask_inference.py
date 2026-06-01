"""Inference script for M2CPerLayerFusionMask.

Reuses cp_transformer_m2c_moe_mask_inference.main() wholesale by
monkey-patching the load_model binding in that module so it builds
M2CPerLayerFusionMask (per-layer fusion backbone + mask-predict head)
instead of M2CMaskMoE (SameStep backbone + mask-predict head). Everything
else -- the 5 modes, the per-block iterative refinement, MIDI rendering,
folder batching -- is reused as-is.

5 modes via masking patterns (cleaner than the AR variants because
single-stream just leaves the silenced modality permanently masked):

  co           : both streams masked from prompt_length onward, filled jointly
  mel2chord    : mel given, chord masked everywhere -- chord filled iteratively
  chord2mel    : symmetric
  mel_only     : chord stays [MASK] forever (never filled); mel filled iteratively
  chord_only   : symmetric

Within each block (timestep), n_refine_steps iterations: first iteration
samples both modalities in parallel; subsequent iterations sample the
more-confident slot first then refine the other given it. This gives true
bidirectional same-step coupling.

Run from midi_yinyang/:

    python cp_transformer_m2c_per_layer_fusion_mask_inference.py \\
        --mode co \\
        --ckpt ckpt/<run>/last.ckpt \\
        --melody POP909-Dataset/POP909-melody/001.mid \\
        --chord  POP909-Dataset/POP909-chord/001.mid \\
        --prompt-length 100 --gen-length 384 \\
        --n-refine-steps 2 \\
        --model-size large
"""

# Inject the vendored fork BEFORE anything else imports.
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import os
import re

import torch

from cp_transformer_m2c_per_layer_fusion_mask import M2CPerLayerFusionMask

# Reuse the existing mask-predict inference module (its main() function
# contains the full CLI + per-song mode dispatch + MIDI rendering pipeline).
import cp_transformer_m2c_moe_mask_inference as mask_inf


def _infer_global_num_layers(ckpt_path, ck, model_size):
    """Resolve global_num_layers for M2CPerLayerFusionMask, in priority order:
      1. ck['hyper_parameters']['global_num_layers']  (training persists this)
      2. _gnl(\\d+)_ in filename                       (default model_name template)
      3. count of fusion_blocks.<N>.* keys in state_dict
      4. size-based fallback (6 if small else 12)
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
               mel_loss_weight=1.0, acc_loss_weight=1.0,
               global_num_layers=None):
    """Instantiate M2CPerLayerFusionMask and load weights. global_num_layers
    auto-detected from the checkpoint if not provided."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, source = _infer_global_num_layers(
            ckpt_path, ck, model_size,
        )
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto-detected from {source})')
    else:
        print(f'[load_model] global_num_layers={global_num_layers} (caller override)')

    net = M2CPerLayerFusionMask(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        global_num_layers=global_num_layers,
        mel_loss_weight=mel_loss_weight,
        acc_loss_weight=acc_loss_weight,
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


# Monkey-patch: mask_inf.main() looks up `load_model` in its own module's
# namespace at call time. Overwriting it routes through OUR version (which
# builds M2CPerLayerFusionMask) without touching the original file.
mask_inf.load_model = load_model


if __name__ == '__main__':
    mask_inf.main()
