"""Inference for M2CIntraCrossAttn (per-modality Q/K/V/O + intra/cross
key-masked SDPA passes + per-block gates + shared MoE FFN).

This is a thin wrapper around cp_transformer_m2c_jointattn_inference: the
sampling loop, MIDI decode, prompt loading, run_one / run_folder helpers,
and CLI dispatch are all reused unchanged. The only thing that differs
is `load_model`, which instantiates M2CIntraCrossAttn instead of
M2CJointAttn (and passes `gate_init_bias`).

Use the same five modes as jointattn inference: co, mel2chord,
chord2mel, mel_only, chord_only.

Single-song CLI:

    python cp_transformer_m2c_intra_cross_attn_inference.py \\
        --mode co \\
        --ckpt ckpt/<run>/last.ckpt \\
        --melody input/some_drums.mid \\
        --chord  input/some_nondrum.mid \\
        --output-dir temp/intra_cross_attn_inference \\
        --prompt-length 100 --gen-length 384 \\
        --temperature 1.0 --max-polyphony 16 --model-size large
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import re
import torch

from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn
# Reuse everything except load_model. decode_m2c_frames and the
# run_one/run_folder/main helpers are model-class-agnostic because they
# only touch tokenizer + state-dict-loaded forward pass.
from cp_transformer_m2c_jointattn_inference import (  # noqa: F401
    decode_m2c_frames,
    _infer_global_num_layers,
    run_one,
    run_folder,
)


def load_model(ckpt_path, model_size='large', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None,
               global_num_layers=None, preserve_program=True,
               min_acc_tokens_before_eos=0, gate_init_bias=-10.0):
    """Build M2CIntraCrossAttn with the right depth/experts and load
    weights. `gate_init_bias` only matters for fresh-init -- when loading
    a trained ckpt, the saved gate weights override the init bias."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, source = _infer_global_num_layers(
            ckpt_path, ck, model_size,
        )
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto-detected from {source})')
    else:
        print(f'[load_model] global_num_layers={global_num_layers} (caller override)')

    net = M2CIntraCrossAttn(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        global_num_layers=global_num_layers,
        preserve_program=preserve_program,
        min_acc_tokens_before_eos=min_acc_tokens_before_eos,
        gate_init_bias=gate_init_bias,
    )
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing keys ({len(missing)}): {missing[:5]}'
              f'{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected keys ({len(unexpected)}): {unexpected[:5]}'
              f'{"..." if len(unexpected) > 5 else ""}')
    return net


if __name__ == '__main__':
    # Reuse the jointattn CLI dispatch. The only thing that matters is
    # which load_model gets called -- patch the jointattn module's
    # load_model symbol to point at ours so its main() picks up the right
    # model class.
    import cp_transformer_m2c_jointattn_inference as _ja_inf
    _ja_inf.load_model = load_model
    _ja_inf.main()
