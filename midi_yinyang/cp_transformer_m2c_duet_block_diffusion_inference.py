"""Parallel-diffusion inference for M2CDuetBlockDiffusion (variant A.3).

Per frame t we run K+1 refinement passes through the global stack with
decreasing noise level k = K, K-1, ..., 0. At each round both query
slots are refined simultaneously and their estimates are fed back as
the next round's slot inputs:

  Round r = K  : both slots = mask_emb + k_emb_*(K)         -> (m_K, c_K)
  Round r = K-1: slot_* = encode(prev_est) + k_emb_*(K-1)   -> (m_{K-1}, c_{K-1})
  ...
  Round r = 0  : slot_* = encode(prev_est) + k_emb_*(0)     -> (m_0, c_0)

Only the round-0 outputs are committed. After K rounds both modalities
have iteratively conditioned on each other's previous-round estimate at
the same target frame -- the parallel-Gibbs approximation of "equalize
by adding" within frame t.

Mode support: same 5 modes as A.2 (co, mel2chord, chord2mel, mel_only,
chord_only). When one modality is `('given', frame)` or `'silence'`,
that slot is held constant across all rounds at the encoded action
frame with k_emb_*(0), and we only refine the slot of the sampled
modality. When both modalities are given/silence, no model call is
needed for that step.

Single-song CLI:

    python cp_transformer_m2c_duet_block_diffusion_inference.py \\
        --mode co \\
        --ckpt ckpt/<run>/last.ckpt \\
        --melody input/some_drums.mid \\
        --chord  input/some_nondrum.mid \\
        --output-dir temp/diffusion_inference \\
        --prompt-length 64 --gen-length 384 \\
        --temperature 1.0 --max-polyphony 16 --model-size large \\
        --refine-steps 4
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion
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
               min_acc_tokens_before_eos=0, gate_init_bias=-10.0,
               diffusion_K=None):
    ckpt_path = resolve_best_ckpt(ckpt_path)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, source = _infer_global_num_layers(
            ckpt_path, ck, model_size,
        )
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto-detected from {source})')

    # Try to recover K from the ckpt's k_emb shape if not given.
    state_dict_keys = (
        ck['state_dict'].keys() if isinstance(ck, dict) and 'state_dict' in ck
        else ck.keys() if isinstance(ck, dict) else []
    )
    if diffusion_K is None:
        for key, name in (('k_emb_m.weight', 'k_emb_m'),
                          ('k_emb_c.weight', 'k_emb_c')):
            if key in state_dict_keys:
                sd = ck['state_dict'] if 'state_dict' in ck else ck
                diffusion_K = sd[key].shape[0] - 1
                print(f'[load_model] diffusion_K={diffusion_K} (inferred from {name})')
                break
        if diffusion_K is None:
            diffusion_K = 4
            print(f'[load_model] diffusion_K={diffusion_K} (default; ckpt has '
                  f'no k_emb_* weights -- probably an A.2 ckpt being loaded '
                  f'into A.3 for compatibility testing)')

    net = M2CDuetBlockDiffusion(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        global_num_layers=global_num_layers,
        preserve_program=preserve_program,
        min_acc_tokens_before_eos=min_acc_tokens_before_eos,
        gate_init_bias=gate_init_bias,
        diffusion_K=diffusion_K,
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


def _build_slot(model, mode, prev_est_h, action_frame_h, r, K, slot_idx, B):
    """Construct a query slot input at refinement round r.

    Args:
        mode: 'sample' (refine across rounds) or 'committed' (held constant).
        prev_est_h: [B, 1, H] previous-round estimate's frame embedding, or
            None at round r==K when there is no previous estimate.
        action_frame_h: [B, 1, H] encoded given/silence frame, used when
            mode == 'committed'.
        r: current refinement round in [0, K].
        K: total noise bins.
        slot_idx: 0 for mod-a (mel/drum), 1 for mod-b (chord/nondrum).
        B: batch size. Must be passed explicitly -- at round K for a
            sampling modality BOTH prev_est_h and action_frame_h are None
            (mask emb only), so it cannot be inferred from the tensors.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if mode == 'sample':
        if r == K:
            base = (model.mask_m_emb if slot_idx == 0 else model.mask_c_emb)
            base = base.view(1, 1, -1).expand(B, 1, -1).to(dtype=dtype)
        else:
            base = prev_est_h.to(dtype=dtype)
        # k-embedding at the current noise level.
        k_emb = (model.k_emb_m if slot_idx == 0 else model.k_emb_c)
        k_e = k_emb(torch.tensor(r, device=device, dtype=torch.long))
        k_e = k_e.view(1, 1, -1).expand(B, 1, -1).to(dtype=dtype)
    else:
        # Committed: held at k=0 across all rounds.
        base = action_frame_h.to(dtype=dtype)
        k_emb = (model.k_emb_m if slot_idx == 0 else model.k_emb_c)
        k_e = k_emb(torch.tensor(0, device=device, dtype=torch.long))
        k_e = k_e.view(1, 1, -1).expand(B, 1, -1).to(dtype=dtype)

    return base + k_e


def general_inference_diffusion(model, gen_length, B, subseq_len, temperature,
                                  mel_action_fn, chord_action_fn,
                                  K_refine=None, seed_from_ar=True):
    """Parallel-diffusion AR decoding loop.

    For each step t in 0..gen_length-1, determines per-modality actions
    via mel_action_fn(t) / chord_action_fn(t). For sampling actions,
    runs K+1 refinement rounds at the query slots and commits the final
    estimate. For 'given' / 'silence', uses the action's frame directly
    and holds the corresponding slot constant across rounds.

    seed_from_ar (default True): in the FIRST round (r = K, query slots
    masked), read the initial drafts from the AR clean-stream positions
    (h_global[clean_len-2] / [clean_len-1], the shifted next-frame heads)
    instead of the query-slot outputs. Rationale: the query slots' mask
    only reaches committed content up to frame t-2 and, at round K, the
    partner slot is empty -- making the blind slot outputs the
    worst-informed predictions in the whole process. The AR heads see one
    more frame of committed history (through frame t-1 of both streams,
    incl. the frame pass), so they produce a strictly better-informed
    blind draft at zero extra cost (same forward). Total forward count is
    unchanged (K+1); only which positions are decoded in round one
    differs. Rounds r < K read the query slots as usual.

    Returns (mel_frames, chord_frames), each a list of [B, subseq_len].
    """
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    H = model.hidden_size

    K = K_refine if K_refine is not None else model.diffusion_K
    assert K >= 1

    h_buffer = torch.zeros(B, 0, H, device=device, dtype=dtype)
    mel_frames = []
    chord_frames = []

    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[gen] step {t}/{gen_length}  (K_refine={K})')

        m_action = mel_action_fn(t)
        c_action = chord_action_fn(t)
        m_sampling = (m_action == 'sample')
        c_sampling = (c_action == 'sample')
        need_model = m_sampling or c_sampling

        m_tokens = None
        c_tokens = None
        # action-frame encodings for non-sampling modalities (held constant
        # across all refinement rounds).
        m_action_h = None
        c_action_h = None
        if not m_sampling:
            m_frame = resolve_frame(m_action, B, subseq_len, tokenizer, device)
            m_tokens = m_frame
            m_action_h = model._encode_frame(m_frame, 0).to(dtype=dtype)  # [B, 1, H]
        if not c_sampling:
            c_frame = resolve_frame(c_action, B, subseq_len, tokenizer, device)
            c_tokens = c_frame
            c_action_h = model._encode_frame(c_frame, 1).to(dtype=dtype)

        if need_model:
            # Build the clean prefix once per step.
            sos = model._assemble_sos(B, device, dtype)
            h_clean = torch.cat([sos, h_buffer], dim=1)   # [B, 2+2t, H]
            T_query = max(t, 1)   # parent's convention; T_query>=1 keeps masks non-empty

            # Track previous-round estimates as encoded embeddings.
            prev_m_h = None
            prev_c_h = None
            last_m_tokens = None
            last_c_tokens = None

            for r in range(K, -1, -1):
                slot_m = _build_slot(
                    model,
                    mode=('sample' if m_sampling else 'committed'),
                    prev_est_h=prev_m_h,
                    action_frame_h=m_action_h,
                    r=r, K=K, slot_idx=0, B=B,
                )
                slot_c = _build_slot(
                    model,
                    mode=('sample' if c_sampling else 'committed'),
                    prev_est_h=prev_c_h,
                    action_frame_h=c_action_h,
                    r=r, K=K, slot_idx=1, B=B,
                )

                h_in = torch.cat([h_clean, slot_m, slot_c], dim=1)
                h_global, _ = model._run_global_stack(h_in, T_query=T_query)
                if r == K and seed_from_ar:
                    # Round-one seed: decode the AR clean-stream heads.
                    # Under the shift, clean position 2t predicts m_t and
                    # 2t+1 predicts c_t -- i.e. the last two clean
                    # positions. Better-informed than the blind masked
                    # slots (see docstring).
                    clean_len = h_clean.shape[1]
                    h_m_pred = h_global[:, clean_len - 2]
                    h_c_pred = h_global[:, clean_len - 1]
                else:
                    h_m_pred = h_global[:, -2]
                    h_c_pred = h_global[:, -1]

                if m_sampling:
                    m_tokens_r = model.local_sampling(
                        h_m_pred, max_subseq_len=subseq_len,
                        temperature=temperature, token_type_id=0,
                    )
                    last_m_tokens = m_tokens_r
                    prev_m_h = model._encode_frame(m_tokens_r, 0).to(dtype=dtype)
                if c_sampling:
                    c_tokens_r = model.local_sampling(
                        h_c_pred, max_subseq_len=subseq_len,
                        temperature=temperature, token_type_id=1,
                    )
                    last_c_tokens = c_tokens_r
                    prev_c_h = model._encode_frame(c_tokens_r, 1).to(dtype=dtype)

            if m_sampling:
                m_tokens = last_m_tokens
            if c_sampling:
                c_tokens = last_c_tokens

        if m_tokens is None:
            m_tokens = resolve_frame(m_action, B, subseq_len, tokenizer, device)
        if c_tokens is None:
            c_tokens = resolve_frame(c_action, B, subseq_len, tokenizer, device)

        mel_frames.append(m_tokens)
        chord_frames.append(c_tokens)

        # Commit: encode and append to h_buffer.
        m_h = model._encode_frame(m_tokens, 0).to(dtype=dtype)
        c_h = model._encode_frame(c_tokens, 1).to(dtype=dtype)
        h_buffer = torch.cat([h_buffer, m_h, c_h], dim=1)

    return mel_frames, chord_frames


if __name__ == '__main__':
    # Patch jointattn_inference to use the diffusion loader + refinement
    # decode loop, then defer to its main() which handles the 5-mode CLI.
    import cp_transformer_m2c_jointattn_inference as _ja_inf
    _ja_inf.load_model = load_model
    _ja_inf.general_inference = general_inference_diffusion
    _ja_inf.main()
