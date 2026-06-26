"""Inference for M2CDuetBlockAttn (Option B: denoising decode via query slots).

The training-time forward of M2CDuetBlockAttn appends 2 "query slots"
(mask_m_emb, mask_c_emb) to the global stack input and predicts a
random target frame T_query at those slots. At inference we use the
same mechanism: at each step t we append 2 query slots and read off
their hiddens as predictions for frame t.

Off-by-one limitation (consistent with training):
The query slot's intra/cross mask uses strict-less-than over the
prediction-frame axis (`frame(q) < T_query`). Under the shift trick,
clean position 2t contains the embedding of m_{t-1} (the most recent
committed frame) and has prediction-frame t. The strict mask therefore
EXCLUDES the most recent committed frame from the query slot's
attention. The model was trained with this exact behaviour, so
inference is consistent -- but the standard AR clean-stream path
(Option A) would actually see one more frame of past at inference.
Track val/test perplexity vs Option A to decide which to use in practice.

Single-song CLI:

    python cp_transformer_m2c_duet_block_inference.py \\
        --mode co \\
        --ckpt ckpt/<run>/last.ckpt \\
        --melody input/some_drums.mid \\
        --chord  input/some_nondrum.mid \\
        --output-dir temp/duet_block_inference \\
        --prompt-length 100 --gen-length 384 \\
        --temperature 1.0 --max-polyphony 16 --model-size large
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import torch

from cp_transformer_m2c_duet_block import M2CDuetBlockAttn
# Reuse decode_m2c_frames / run_one / run_folder / _infer_global_num_layers
# from the jointattn inference path -- they're model-class-agnostic.
from cp_transformer_m2c_jointattn_inference import (  # noqa: F401
    decode_m2c_frames,
    _infer_global_num_layers,
    resolve_best_ckpt,
    run_one,
    run_folder,
)
from cp_transformer_m2c_moe_inference import resolve_frame


def load_model(ckpt_path, model_size='large', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None,
               global_num_layers=None, preserve_program=True,
               min_acc_tokens_before_eos=0, gate_init_bias=-10.0):
    ckpt_path = resolve_best_ckpt(ckpt_path)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, source = _infer_global_num_layers(
            ckpt_path, ck, model_size,
        )
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto-detected from {source})')
    net = M2CDuetBlockAttn(
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


def general_inference_duet_block(model, gen_length, B, subseq_len, temperature,
                                  mel_action_fn, chord_action_fn):
    """DuetBlockAttn inference: at each step append 2 query slots, run the
    global stack with T_query = t, read the query slot hiddens, and decode
    them via local_sampling. The shifted clean stream is maintained as h_buffer.

    Mirrors the action-fn protocol used by the jointattn general_inference:
    each step's per-modality action is one of 'sample', ('given', tensor),
    or 'silence'. Returns (mel_frames, chord_frames).
    """
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    H = model.hidden_size

    h_buffer = torch.zeros(B, 0, H, device=device)
    mel_frames = []
    chord_frames = []

    # Precompute the static mask embeddings broadcast to batch.
    mask_m = model.mask_m_emb.view(1, 1, -1).expand(B, 1, -1)
    mask_c = model.mask_c_emb.view(1, 1, -1).expand(B, 1, -1)

    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[gen] step {t}/{gen_length}')

        m_action = mel_action_fn(t)
        c_action = chord_action_fn(t)
        need_model = m_action == 'sample' or c_action == 'sample'

        m_tokens = None
        c_tokens = None

        if need_model:
            # Build the prefix: sos + h_buffer of past committed frames.
            sos = model._assemble_sos(B, device, h_buffer.dtype)
            h_clean = torch.cat([sos, h_buffer], dim=1)   # [B, 2+2t, H]
            # Append the 2 query slots.
            h_in = torch.cat([h_clean,
                               mask_m.to(dtype=h_clean.dtype),
                               mask_c.to(dtype=h_clean.dtype)], dim=1)
            # T_query semantics: at training T_query in [1, T_full-1].
            # Use t for t>=1; for t==0 fall back to T_query=1 so the
            # mask isn't empty. Either way the query slot ignores the
            # most recent committed frame -- see docstring.
            T_query = max(t, 1)
            h_global, _ = model._run_global_stack(h_in, T_query=T_query)
            # Last 2 positions are the query slot outputs.
            h_m_pred = h_global[:, -2]
            h_c_pred = h_global[:, -1]

            if m_action == 'sample':
                m_tokens = model.local_sampling(
                    h_m_pred, max_subseq_len=subseq_len,
                    temperature=temperature, token_type_id=0,
                )
            if c_action == 'sample':
                c_tokens = model.local_sampling(
                    h_c_pred, max_subseq_len=subseq_len,
                    temperature=temperature, token_type_id=1,
                )

        if m_tokens is None:
            m_tokens = resolve_frame(m_action, B, subseq_len, tokenizer, device)
        if c_tokens is None:
            c_tokens = resolve_frame(c_action, B, subseq_len, tokenizer, device)

        mel_frames.append(m_tokens)
        chord_frames.append(c_tokens)

        # Encode the committed frames and append to h_buffer.
        m_h = model._encode_frame(m_tokens, 0)
        c_h = model._encode_frame(c_tokens, 1)
        h_buffer = torch.cat([h_buffer, m_h, c_h], dim=1)

    return mel_frames, chord_frames


def general_inference_duet_block_option_a(model, gen_length, B, subseq_len,
                                            temperature, mel_action_fn,
                                            chord_action_fn):
    """DuetBlock inference -- Option A: read predictions from the standard
    AR-shifted clean stream instead of the query slots.

    Why this exists (and is the default): the query slots receive
    drastically less gradient signal during training than the AR clean
    positions (~T_full x less density, since query loss touches 2 slots
    per batch while AR loss touches 2*T_full positions). Unless query
    loss has been upweighted heavily AND trained for many tens of
    thousands of steps, the query slots are under-trained and produce
    near-degenerate predictions at inference, often EOS-biased on both
    modalities. Option A sidesteps the query mechanism entirely and uses
    the AR clean stream, which is trained with the same density as
    DuetAttn (#A.1)'s baseline.

    The DuetBlock architecture's NEW machinery (3rd SDPA pass, frame
    gate, mask embeddings) still RUNS during this inference -- the
    forward is identical to Option B's. The difference is purely which
    two positions in h_global we read:

      Option B:  h_global[-2], h_global[-1]   (the appended query slots)
      Option A:  h_global[clean_len-2], h_global[clean_len-1]   (the last
                  two AR-shifted clean positions, which under the shift
                  predict m_t and c_t respectively)

    The two appended mask slots still exist in the sequence (so the
    layer's mask construction is consistent) but we ignore their outputs.
    """
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    H = model.hidden_size

    h_buffer = torch.zeros(B, 0, H, device=device)
    mel_frames = []
    chord_frames = []

    mask_m = model.mask_m_emb.view(1, 1, -1).expand(B, 1, -1)
    mask_c = model.mask_c_emb.view(1, 1, -1).expand(B, 1, -1)

    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[gen] step {t}/{gen_length}')

        m_action = mel_action_fn(t)
        c_action = chord_action_fn(t)
        need_model = m_action == 'sample' or c_action == 'sample'

        m_tokens = None
        c_tokens = None

        if need_model:
            sos = model._assemble_sos(B, device, h_buffer.dtype)
            h_clean = torch.cat([sos, h_buffer], dim=1)   # [B, 2+2t, H]
            clean_len = h_clean.shape[1]
            h_in = torch.cat([h_clean,
                               mask_m.to(dtype=h_clean.dtype),
                               mask_c.to(dtype=h_clean.dtype)], dim=1)
            # T_query is still passed to keep the layer's mask cache
            # consistent; we just don't use the query slot outputs.
            T_query = max(t, 1)
            h_global, _ = model._run_global_stack(h_in, T_query=T_query)
            # Read predictions from the standard shifted-AR positions:
            # h_global[clean_len-2] predicts m_t, [clean_len-1] predicts c_t.
            h_m_pred = h_global[:, clean_len - 2]
            h_c_pred = h_global[:, clean_len - 1]

            if m_action == 'sample':
                m_tokens = model.local_sampling(
                    h_m_pred, max_subseq_len=subseq_len,
                    temperature=temperature, token_type_id=0,
                )
            if c_action == 'sample':
                c_tokens = model.local_sampling(
                    h_c_pred, max_subseq_len=subseq_len,
                    temperature=temperature, token_type_id=1,
                )

        if m_tokens is None:
            m_tokens = resolve_frame(m_action, B, subseq_len, tokenizer, device)
        if c_tokens is None:
            c_tokens = resolve_frame(c_action, B, subseq_len, tokenizer, device)

        mel_frames.append(m_tokens)
        chord_frames.append(c_tokens)

        m_h = model._encode_frame(m_tokens, 0)
        c_h = model._encode_frame(c_tokens, 1)
        h_buffer = torch.cat([h_buffer, m_h, c_h], dim=1)

    return mel_frames, chord_frames


if __name__ == '__main__':
    # Patch the jointattn inference module. Default to Option A
    # (AR clean-stream readout); use Option B by replacing
    # `general_inference_duet_block_option_a` below with
    # `general_inference_duet_block` if you specifically want to test
    # query-slot decoding (typically not what you want unless the
    # query slots have been trained heavily).
    import cp_transformer_m2c_jointattn_inference as _ja_inf
    _ja_inf.load_model = load_model
    _ja_inf.general_inference = general_inference_duet_block_option_a
    _ja_inf.main()
