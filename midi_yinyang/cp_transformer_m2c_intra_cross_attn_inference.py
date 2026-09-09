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
    resolve_best_ckpt,
    run_one,
    run_folder,
)
from cp_transformer_m2c_moe_inference import (  # noqa: F401
    general_inference,
    build_inference_sos,
    resolve_frame,
)


def load_model(ckpt_path, model_size='large', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None,
               global_num_layers=None, preserve_program=True,
               min_acc_tokens_before_eos=0, gate_init_bias=-10.0):
    """Build M2CIntraCrossAttn with the right depth/experts and load
    weights. `gate_init_bias` only matters for fresh-init -- when loading
    a trained ckpt, the saved gate weights override the init bias."""
    ckpt_path = resolve_best_ckpt(ckpt_path)
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
    # A.10 detection: the flag is a buffer VALUE restored from the ckpt.
    print(f'[load_model] same_frame={net.same_frame}'
          f'{" (A.10: two-forward decode, direction per frame)" if net.same_frame else ""}')
    return net


def general_inference_same_frame(model, gen_length, B, subseq_len, temperature,
                                 mel_action_fn, chord_action_fn):
    """A.10 decode. Per frame draw a direction (env A10_DIRECTION: alt
    (default; melody leads on even frames, chord on odd), m, c, random),
    then two forwards: the LEADER is read from the historical A.1
    position with the direction set (its row has no same-frame edge, so
    this is A.1's prediction); the leader's frame is encoded and placed
    at the row that holds it; the FOLLOWER is read from its row in a
    second forward, where the same-frame edge lets it attend that key.
    The second forward always has an even length (the stack asserts it):
    under direction 0 a zero filler stands at row 2t+3 (c_t, unknown),
    under direction 1 at row 2t+2 (m_t, unknown); the follower row reads
    neither filler (row 2t+1 reads keys <= 2t+1 and 2t+2; row 2t reads
    keys <= 2t and 2t+3), which audit_same_frame check 4 verifies. Falls back to the base loop when the
    checkpoint is not an A.10 model.
    """
    if not getattr(model, 'same_frame', False):
        return general_inference(model, gen_length, B, subseq_len, temperature,
                                 mel_action_fn, chord_action_fn)
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    H = model.hidden_size
    rule = _os.environ.get('A10_DIRECTION', 'alt')
    print(f'[gen] A.10 same-frame decode, direction rule = {rule}')

    def direction_for(t):
        if rule == 'm':
            return 0                  # chord follows melody
        if rule == 'c':
            return 1                  # melody follows chord
        if rule == 'random':
            return int(torch.randint(0, 2, (1,)).item())
        return t % 2                  # alt: even frames melody leads

    h_buffer = torch.zeros(B, 0, H, device=device, dtype=dtype)
    mel_frames, chord_frames = [], []
    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[gen] step {t}/{gen_length}')
        m_action, c_action = mel_action_fn(t), chord_action_fn(t)
        d = direction_for(t)
        dir_t = torch.full((B,), d, dtype=torch.long, device=device)
        m_tokens = c_tokens = None
        if m_action == 'sample' or c_action == 'sample':
            sos = build_inference_sos(model, B, device, dtype)
            h_in = torch.cat([sos, h_buffer], dim=1)          # 2 + 2t rows
            model._sf_dir = dir_t
            try:
                h_out, _ = model._global_interaction(h_in)
                if d == 0:
                    # leader melody from row 2t (h_out[-2]); then chord
                    if m_action == 'sample':
                        m_tokens = model.local_sampling(
                            h_out[:, -2], max_subseq_len=subseq_len,
                            temperature=temperature, token_type_id=0)
                    else:
                        m_tokens = resolve_frame(m_action, B, subseq_len, tokenizer, device)
                    if c_action == 'sample':
                        m_h = model._encode_frame(m_tokens, 0).to(dtype=dtype)
                        # rows 2t+2 = m_t, 2t+3 = zero filler (the stack
                        # needs an even length; row 2t+1 never reads 2t+3)
                        filler = torch.zeros(B, 1, H, device=device, dtype=dtype)
                        h_in2 = torch.cat([h_in, m_h, filler], dim=1)
                        h_out2, _ = model._global_interaction(h_in2)
                        c_tokens = model.local_sampling(
                            h_out2[:, -3], max_subseq_len=subseq_len,   # row 2t+1
                            temperature=temperature, token_type_id=1)
                else:
                    # leader chord from row 2t+1 (h_out[-1]); then melody
                    if c_action == 'sample':
                        c_tokens = model.local_sampling(
                            h_out[:, -1], max_subseq_len=subseq_len,
                            temperature=temperature, token_type_id=1)
                    else:
                        c_tokens = resolve_frame(c_action, B, subseq_len, tokenizer, device)
                    if m_action == 'sample':
                        c_h = model._encode_frame(c_tokens, 1).to(dtype=dtype)
                        filler = torch.zeros(B, 1, H, device=device, dtype=dtype)
                        h_in2 = torch.cat([h_in, filler, c_h], dim=1)  # rows 2t+2, 2t+3
                        h_out2, _ = model._global_interaction(h_in2)
                        m_tokens = model.local_sampling(
                            h_out2[:, -4], max_subseq_len=subseq_len,   # row 2t
                            temperature=temperature, token_type_id=0)
            finally:
                model._sf_dir = None
        if m_tokens is None:
            m_tokens = resolve_frame(m_action, B, subseq_len, tokenizer, device)
        if c_tokens is None:
            c_tokens = resolve_frame(c_action, B, subseq_len, tokenizer, device)
        mel_frames.append(m_tokens)
        chord_frames.append(c_tokens)
        m_h = model._encode_frame(m_tokens, 0).to(dtype=dtype)
        c_h = model._encode_frame(c_tokens, 1).to(dtype=dtype)
        h_buffer = torch.cat([h_buffer, m_h, c_h], dim=1)
    return mel_frames, chord_frames


if __name__ == '__main__':
    # Reuse the jointattn CLI dispatch. The only thing that matters is
    # which load_model gets called -- patch the jointattn module's
    # load_model symbol to point at ours so its main() picks up the right
    # model class.
    import cp_transformer_m2c_jointattn_inference as _ja_inf
    _ja_inf.load_model = load_model
    _ja_inf.general_inference = general_inference_same_frame   # A.10-aware
    _ja_inf.main()
