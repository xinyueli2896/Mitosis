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
from cp_transformer_m2c_moe import TRAIN_LENGTH
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
    # Scheme detection: v1.1 ckpts carry the slot_rope_aligned_flag
    # buffer (value 1 for the aligned scheme, 0 for a --legacy_slot_rope
    # ablation run); v1.0 ckpts lack the key entirely. Legacy-scheme
    # ckpts trained the slots at constant end-of-sequence phase, so
    # inference must zero-pad the clean stream (the decode loop reads
    # model.slot_rope_aligned).
    _sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    _flag_key = next(
        (k for k in state_dict_keys if k.endswith('slot_rope_aligned_flag')),
        None,
    )
    slot_rope_aligned = bool(_sd[_flag_key].item()) if _flag_key else False
    # v1.2 ckpts additionally carry time_rope_aligned_flag (rotary index
    # = physical // 2, m_t/c_t share position t). Absent in v1.0/v1.1.
    _tflag_key = next(
        (k for k in state_dict_keys if k.endswith('time_rope_aligned_flag')),
        None,
    )
    time_rope_aligned = bool(_sd[_tflag_key].item()) if _tflag_key else False
    scheme = ('v1.2 time-aligned scheme' if time_rope_aligned
              else 'v1.1 aligned scheme' if slot_rope_aligned
              else 'legacy v1.0 -> decode-time padding')
    print(f'[load_model] slot_rope_aligned={slot_rope_aligned} '
          f'time_rope_aligned={time_rope_aligned} ({scheme})')
    # A.2.moe_improved detection: the per-modality router bias is a real
    # parameter (ffn.modality_bias), so its presence in the state dict IS
    # the flag. Building the model without it would silently drop the
    # learned bias (strict=False) and decode with a router trained to
    # rely on it.
    moe_modality_bias = any(k.endswith('ffn.modality_bias')
                            for k in state_dict_keys)
    print(f'[load_model] moe_modality_bias={moe_modality_bias}'
          f'{" (A.2.moe_improved)" if moe_modality_bias else ""}')
    # A.2.moe_permod detection: per-modality router matrices. Same
    # principle -- the gate_m/gate_c keys' presence IS the flag; building
    # without them would leave the model with a fresh shared gate while
    # silently dropping both trained routers.
    moe_modality_gates = any(k.endswith('ffn.gate_m.weight')
                             for k in state_dict_keys)
    print(f'[load_model] moe_modality_gates={moe_modality_gates}'
          f'{" (A.2.moe_permod)" if moe_modality_gates else ""}')
    # A.2.moe_hardroute detection: hard routing adds no weights, so the
    # variant is carried by a registered buffer instead. Building without
    # it would decode a disjoint-pool model as if every expert were
    # reachable from either stream -- silently wrong, no error.
    moe_modality_hard_route = any(k.endswith('ffn.hard_route_flag')
                                  for k in state_dict_keys)
    print(f'[load_model] moe_modality_hard_route={moe_modality_hard_route}'
          f'{" (A.2.moe_hardroute)" if moe_modality_hard_route else ""}')
    # A.4 detection: the buffer travels in the ckpt with its VALUE (0/1),
    # unlike the presence-is-the-flag parameters, so read it.
    token_level_mask = False
    for k in state_dict_keys:
        if k.endswith('token_level_mask_flag'):
            sd_tmp = ck['state_dict'] if 'state_dict' in ck else ck
            token_level_mask = bool(int(sd_tmp[k].item()))
            break
    print(f'[load_model] token_level_mask={token_level_mask}'
          f'{" (A.4)" if token_level_mask else ""}')
    # A.8 detection: the block size travels in the ckpt as a buffer
    # VALUE. It must be restored, or the model would build A.3 masks and
    # the decode would take the single-frame path -- a silent
    # train/decode mismatch, which is exactly what A.8 exists to avoid.
    query_block = 1
    for k in state_dict_keys:
        if k.endswith('query_block_flag'):
            sd_tmp = ck['state_dict'] if 'state_dict' in ck else ck
            query_block = max(int(sd_tmp[k].item()), 1)
            break
    print(f'[load_model] query_block={query_block}'
          f'{" (A.8: BLOCK decode)" if query_block > 1 else ""}')
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
        slot_rope_aligned=slot_rope_aligned,
        time_rope_aligned=time_rope_aligned,
        moe_modality_bias=moe_modality_bias,
        moe_modality_gates=moe_modality_gates,
        moe_modality_hard_route=moe_modality_hard_route,
        token_level_mask=token_level_mask,
        query_block=query_block,
    )
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing keys ({len(missing)}): {missing[:5]}'
              f'{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected keys ({len(unexpected)}): {unexpected[:5]}'
              f'{"..." if len(unexpected) > 5 else ""}')
    # Free-running routing statistics (env-activated, like the A3_*
    # decode knobs, so every driver that loads through here gets it
    # without CLI plumbing): accumulate every MoE routing decision this
    # process makes and report + dump JSON at exit.
    _stats_out = _os.environ.get('MOE_ROUTING_STATS')
    if _stats_out:
        from moe_runtime_stats import attach as _attach_routing_stats
        _attach_routing_stats(net, _stats_out)
    return net


def _token_confidences(model, h_pred, tokens, type_id):
    """Teacher-forced per-token probabilities of a sampled frame.

    Mirrors the training query-logits path: embed [SOS, tokens][:-1]
    with the type embedding, local_decode conditioned on h_pred, gather
    each sampled token's probability. Returns [B, S] confidences (pad
    positions get +inf so they are never selected for re-masking).
    """
    B, S = tokens.shape
    device = tokens.device
    sos = torch.full((B, 1), model.tokenizer.sos_token, dtype=torch.long,
                     device=device)
    x = torch.cat([sos, tokens], dim=1)                      # [B, S+1]
    word = model.local_embedding(x)
    type_emb = model.token_type_embeddings(
        torch.full((B, S + 1), type_id, dtype=torch.long, device=device))
    emb = (word + type_emb)[:, :-1]                          # [B, S, H]
    logits = model.local_decode(h_pred, emb)                 # [B, S, V]
    probs = torch.softmax(logits.float(), dim=-1)
    conf = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    conf = torch.where(tokens == model.tokenizer.pad_token,
                       torch.full_like(conf, float('inf')), conf)
    return conf


def _remask_lowest(model, tokens, conf, frac):
    """A.4 decode schedule: re-mask the lowest-confidence non-pad tokens.

    frac in [0, 1] is the target masked fraction (k/K for the level the
    slot is about to enter). Per item, n = round(frac * n_maskable)
    lowest-confidence tokens are replaced by the frame mask id. frac=0
    returns tokens unchanged.
    """
    if frac <= 0:
        return tokens
    B, S = tokens.shape
    maskable = tokens != model.tokenizer.pad_token
    n_maskable = maskable.sum(dim=1)                          # [B]
    n_mask = (frac * n_maskable.float()).round().long()       # [B]
    order = conf.argsort(dim=1)               # ascending; pads (+inf) last
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(S, device=tokens.device)
                   .unsqueeze(0).expand(B, S))
    drop = ranks < n_mask.unsqueeze(1)
    return torch.where(
        drop, torch.full_like(tokens, model.frame_mask_token), tokens)


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


def general_inference_block(model, gen_length, B, subseq_len, temperature,
                            mel_action_fn, chord_action_fn,
                            K_refine=None, seed_from_ar=True,
                            final_temperature=None):
    """A.8 BLOCK decode: draft, refine and commit Bk frames at a time.

    The single-frame loop refines one frame against a fully committed
    past. A.8 trains slots as a CONTIGUOUS BLOCK -- every slot sees the
    prefix before the block and every other slot in it -- so its decode
    must present the same picture: Bk frame-pairs un-committed at once,
    mutually visible, refined together, committed together. That is the
    whole point of the variant (train/decode structural identity), and
    it also costs K+1 forwards per Bk frames instead of per frame.

    Differences from the single-frame loop, all forced by the block:
      * AR seeding applies to the FIRST frame of the block only -- the
        clean heads predict one frame ahead, and frames t0+1.. have no
        committed predecessor to be seeded from. They start masked.
      * the adaptive silent-frame early exit is skipped: silence is a
        per-frame property and the block commits jointly.
      * token re-masking (A.4) is not applied; A.8 builds on the
        frame-level kernels (A.3 / A.7).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tokenizer = model.tokenizer
    H = model.hidden_size
    K = K_refine if K_refine is not None else model.diffusion_K
    Bk = max(int(getattr(model, 'query_block_flag', 1)), 1)
    if final_temperature is None:
        final_temperature = temperature
    draft_temperature = None
    env_dt = _os.environ.get('A3_DRAFT_TEMP')
    if env_dt is not None:
        draft_temperature = float(env_dt)
    print(f'[gen] BLOCK decode: Bk={Bk} frames per cycle, K={K}')

    h_buffer = torch.zeros(B, 0, H, device=device, dtype=dtype)
    mel_frames, chord_frames = [], []

    t = 0
    while t < gen_length:
        blk = min(Bk, gen_length - t)
        frames = list(range(t, t + blk))
        if t % 10 < blk:
            print(f'[gen] frames {frames[0]}..{frames[-1]}/{gen_length}')

        m_actions = [mel_action_fn(f) for f in frames]
        c_actions = [chord_action_fn(f) for f in frames]
        m_samp = [a == 'sample' for a in m_actions]
        c_samp = [a == 'sample' for a in c_actions]

        # Given frames are resolved once and held at k_emb(0) all rounds.
        m_tok = [None] * blk
        c_tok = [None] * blk
        m_act_h = [None] * blk
        c_act_h = [None] * blk
        for j in range(blk):
            if not m_samp[j]:
                m_tok[j] = resolve_frame(m_actions[j], B, subseq_len,
                                          tokenizer, device)
                m_act_h[j] = model._encode_frame(m_tok[j], 0).to(dtype=dtype)
            if not c_samp[j]:
                c_tok[j] = resolve_frame(c_actions[j], B, subseq_len,
                                          tokenizer, device)
                c_act_h[j] = model._encode_frame(c_tok[j], 1).to(dtype=dtype)

        if any(m_samp) or any(c_samp):
            sos = model._assemble_sos(B, device, dtype)
            h_clean = torch.cat([sos, h_buffer], dim=1)
            clean_len = h_clean.shape[1]
            # Frame indices must be >= 1 and distinct (forward asserts
            # both); the very first block is shifted up by one, matching
            # the single-frame loop's T_query = max(t, 1) convention.
            base = max(frames[0], 1)
            tq = tuple(range(base, base + blk))

            prev_m = [None] * blk
            prev_c = [None] * blk
            last_m = [None] * blk
            last_c = [None] * blk

            for r in range(K, -1, -1):
                slots = []
                for j in range(blk):
                    slots.append(_build_slot(
                        model, 'sample' if m_samp[j] else 'committed',
                        prev_m[j], m_act_h[j], r, K, 0, B))
                    slots.append(_build_slot(
                        model, 'sample' if c_samp[j] else 'committed',
                        prev_c[j], c_act_h[j], r, K, 1, B))
                h_in = torch.cat([h_clean] + slots, dim=1)
                h_global, _ = model._run_global_stack(h_in, T_query=tq)

                temp_r = (final_temperature if r == 0
                          else (draft_temperature
                                if draft_temperature is not None
                                else temperature))
                temp_r = max(temp_r, 1e-4)

                for j in range(blk):
                    if r == K and seed_from_ar and j == 0:
                        # only the block's first frame has a committed
                        # predecessor to be AR-seeded from
                        h_m_pred = h_global[:, clean_len - 2]
                        h_c_pred = h_global[:, clean_len - 1]
                    else:
                        h_m_pred = h_global[:, clean_len + 2 * j]
                        h_c_pred = h_global[:, clean_len + 2 * j + 1]
                    if m_samp[j]:
                        last_m[j] = model.local_sampling(
                            h_m_pred, max_subseq_len=subseq_len,
                            temperature=temp_r, token_type_id=0)
                        prev_m[j] = model._encode_frame(
                            last_m[j], 0).to(dtype=dtype)
                    if c_samp[j]:
                        last_c[j] = model.local_sampling(
                            h_c_pred, max_subseq_len=subseq_len,
                            temperature=temp_r, token_type_id=1)
                        prev_c[j] = model._encode_frame(
                            last_c[j], 1).to(dtype=dtype)

            for j in range(blk):
                if m_samp[j]:
                    m_tok[j] = last_m[j]
                if c_samp[j]:
                    c_tok[j] = last_c[j]

        for j in range(blk):
            if m_tok[j] is None:
                m_tok[j] = resolve_frame(m_actions[j], B, subseq_len,
                                          tokenizer, device)
            if c_tok[j] is None:
                c_tok[j] = resolve_frame(c_actions[j], B, subseq_len,
                                          tokenizer, device)
            mel_frames.append(m_tok[j])
            chord_frames.append(c_tok[j])
            h_buffer = torch.cat(
                [h_buffer,
                 model._encode_frame(m_tok[j], 0).to(dtype=dtype),
                 model._encode_frame(c_tok[j], 1).to(dtype=dtype)], dim=1)
        t += blk

    return mel_frames, chord_frames


def general_inference_diffusion(model, gen_length, B, subseq_len, temperature,
                                  mel_action_fn, chord_action_fn,
                                  K_refine=None, seed_from_ar=True,
                                  final_temperature=None):
    """Parallel-diffusion AR decoding loop.

    Dispatches to general_inference_block for A.8 checkpoints (block
    size > 1), so every existing driver picks up the matching decode
    without new plumbing.

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

    K_refine: number of refinement rounds. None -> model.diffusion_K.
    K_refine=0 with seed_from_ar means: ONE forward, commit the AR-head
    drafts directly, never decode the slots -- i.e. A.1-style decoding
    on the A.3 checkpoint. This is the bisect knob for diagnosing
    slot-pathway collapse: if K_refine=0 sounds fine but K_refine=K goes
    silent after the prompt, the refinement rounds (under-trained slot
    denoisers / exposure gap on self-generated slot content) are what
    destroys the output, not the backbone.

    final_temperature: sampling temperature at the LAST (committed)
    round, with linear annealing from `temperature` at round K. Default
    None -> same as `temperature` (no annealing, backwards compatible).
    Recommended ~0.7 when refinement outputs sound chaotic: early rounds
    keep exploration, the committed round takes the denoiser's
    high-confidence mode instead of a fresh temperature-1 sample.

    Env overrides (read per call, so sbatch --export works):
      A3_REFINE_STEPS   int, same as K_refine.
      A3_SEED_FROM_AR   '0' disables the AR seed.
      A3_FINAL_TEMP     float, same as final_temperature.

    Returns (mel_frames, chord_frames), each a list of [B, subseq_len].
    """
    tokenizer = model.tokenizer
    if int(getattr(model, 'query_block_flag', 1)) > 1:
        return general_inference_block(
            model, gen_length, B, subseq_len, temperature,
            mel_action_fn, chord_action_fn,
            K_refine=K_refine, seed_from_ar=seed_from_ar,
            final_temperature=final_temperature)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    H = model.hidden_size

    env_k = _os.environ.get('A3_REFINE_STEPS')
    if K_refine is None and env_k is not None:
        K_refine = int(env_k)
        print(f'[gen] A3_REFINE_STEPS={K_refine} (env override)')
    if _os.environ.get('A3_SEED_FROM_AR') == '0':
        seed_from_ar = False
        print('[gen] A3_SEED_FROM_AR=0 (env override)')
    env_ft = _os.environ.get('A3_FINAL_TEMP')
    if final_temperature is None and env_ft is not None:
        final_temperature = float(env_ft)
        print(f'[gen] A3_FINAL_TEMP={final_temperature} (env override)')
    env_tp = _os.environ.get('A3_TOP_P')
    if env_tp is not None:
        model.sampling_top_p = float(env_tp)
        print(f'[gen] A3_TOP_P={model.sampling_top_p} (nucleus sampling)')
    # A3_DRAFT_TEMP: temperature for all NON-final rounds (r > 0),
    # overriding the linear anneal. Lets you invert the schedule --
    # sharp, stable drafts (e.g. 0.7) with a diverse final commit
    # (final_temperature 0.9-1.0) -- which fights repetition: the
    # committed token stream keeps entropy while the scaffold the
    # rounds condition on stays clean.
    if final_temperature is None:
        final_temperature = temperature
    env_dt = _os.environ.get('A3_DRAFT_TEMP')
    draft_temperature = float(env_dt) if env_dt is not None else None
    if draft_temperature is not None:
        print(f'[gen] A3_DRAFT_TEMP={draft_temperature} (piecewise schedule: '
              f'drafts@{draft_temperature}, commit@{final_temperature})')
    # A3_ADAPTIVE=1: skip refinement rounds on frames where either
    # stream's seed-round frame is SILENT. Refinement's value is mutual
    # negotiation between two active voices; with one voice silent there
    # is nothing to negotiate and extra rounds only inject drift
    # (listening tests: K=4 > K=0 when drums are present, K=4 <= K=0 on
    # no-drum prompts). Silence = frame whose first token is EOS.
    adaptive = _os.environ.get('A3_ADAPTIVE') == '1'
    if adaptive:
        print('[gen] A3_ADAPTIVE=1 (skip refinement on silent-frame steps)')
    # A.4 decode schedule: feed the next round a PARTIALLY re-masked
    # draft -- the (r-1)/K lowest-confidence tokens replaced by the
    # frame mask id -- matching the graded corruption the token-level
    # variant trained on. Default ON for A.4 ckpts; A3_TOKEN_REMASK=0
    # falls back to full-draft re-embedding (the plain schedule, which
    # A.4 also trained on via its k=0-adjacent draws).
    token_remask = (getattr(model, 'token_level_mask', False)
                    and _os.environ.get('A3_TOKEN_REMASK') != '0')
    if getattr(model, 'token_level_mask', False):
        print(f'[gen] A.4 ckpt: token-level re-masking '
              f'{"ON" if token_remask else "OFF (A3_TOKEN_REMASK=0)"}')

    def _is_silent(tokens):
        return tokens is not None and bool(
            (tokens[:, 0] == tokenizer.eos_token).all())

    K = K_refine if K_refine is not None else model.diffusion_K
    assert K >= 0
    assert K >= 1 or seed_from_ar, (
        'K_refine=0 requires seed_from_ar (there is no slot estimate to '
        'commit otherwise)'
    )

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
            clean_committed_len = h_clean.shape[1]
            T_query = max(t, 1)   # parent's convention; T_query>=1 keeps masks non-empty

            # --- RoPE-phase alignment with training -------------------
            # At training the clean stream is ALWAYS full length
            # (2*TRAIN_LENGTH positions), so the two query slots always
            # sit at rotary phase 2*TRAIN_LENGTH{, +1} regardless of
            # T_query. Appending the slots directly after 2+2t committed
            # positions puts them at a phase the slot pathway has NEVER
            # been trained at (relative distances to visible keys are
            # disjoint from training's for small t), which degrades slot
            # predictions into sparse/EOS-biased garbage. Fix: zero-pad
            # the clean stream out to the training length before
            # appending the slots. The pad rows are inert: slot rows
            # cannot attend them (pred-frame >= T_query is masked),
            # committed rows cannot (causal), and we never read pad
            # outputs. Env A3_PAD_FRAMES overrides the pad target
            # (frames per modality); '0' disables padding entirely.
            env_pad = _os.environ.get('A3_PAD_FRAMES')
            if env_pad is not None:
                pad_frames_target = int(env_pad)
            elif (getattr(model, 'slot_rope_aligned', False)
                  or getattr(model, 'time_rope_aligned', False)):
                # v1.1/v1.2 aligned schemes: the slots are rotary-indexed
                # from T_query inside _run_global_stack (v1.1: physical
                # 2*T_query+2/+3; v1.2: both at T_query+1), so their
                # phase already matches training at any t. No padding.
                pad_frames_target = 0
            else:
                pad_frames_target = TRAIN_LENGTH
            target_clean_len = max(2 * pad_frames_target,
                                    clean_committed_len)
            if target_clean_len > clean_committed_len:
                pad = torch.zeros(B, target_clean_len - clean_committed_len,
                                   H, device=device, dtype=dtype)
                h_clean_padded = torch.cat([h_clean, pad], dim=1)
            else:
                h_clean_padded = h_clean

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

                h_in = torch.cat([h_clean_padded, slot_m, slot_c], dim=1)
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

                # Temperature annealing across rounds: exploration early,
                # near-greedy at the committed round. Linear interpolation
                # from `temperature` at r=K down to final_temperature at
                # r=0. With final_temperature == temperature (default,
                # backwards compatible) this is a no-op.
                if draft_temperature is not None:
                    # Piecewise: every draft round at draft_temperature,
                    # only the committed round at final_temperature.
                    temp_r = final_temperature if r == 0 else draft_temperature
                else:
                    if K > 0:
                        frac = r / K
                    else:
                        frac = 1.0
                    temp_r = final_temperature + (temperature - final_temperature) * frac
                temp_r = max(temp_r, 1e-4)

                if m_sampling:
                    m_tokens_r = model.local_sampling(
                        h_m_pred, max_subseq_len=subseq_len,
                        temperature=temp_r, token_type_id=0,
                    )
                    last_m_tokens = m_tokens_r
                    feed_m = m_tokens_r
                    if token_remask and r > 0 and K > 0:
                        conf = _token_confidences(model, h_m_pred,
                                                   m_tokens_r, 0)
                        feed_m = _remask_lowest(model, m_tokens_r, conf,
                                                 (r - 1) / K)
                    prev_m_h = model._encode_frame(feed_m, 0).to(dtype=dtype)
                if c_sampling:
                    c_tokens_r = model.local_sampling(
                        h_c_pred, max_subseq_len=subseq_len,
                        temperature=temp_r, token_type_id=1,
                    )
                    last_c_tokens = c_tokens_r
                    feed_c = c_tokens_r
                    if token_remask and r > 0 and K > 0:
                        conf = _token_confidences(model, h_c_pred,
                                                   c_tokens_r, 1)
                        feed_c = _remask_lowest(model, c_tokens_r, conf,
                                                 (r - 1) / K)
                    prev_c_h = model._encode_frame(feed_c, 1).to(dtype=dtype)

                # Adaptive early-exit: after the seed round, if either
                # stream's current frame (sampled draft or given action)
                # is silent, commit the seed and skip the refinement.
                if adaptive and r == K:
                    m_probe = last_m_tokens if m_sampling else m_tokens
                    c_probe = last_c_tokens if c_sampling else c_tokens
                    if _is_silent(m_probe) or _is_silent(c_probe):
                        break

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
