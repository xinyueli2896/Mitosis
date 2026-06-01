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


# ---------------------------------------------------------------------------
# Within-block AR sampling replacement for mask_predict_with_modes
# ---------------------------------------------------------------------------

@torch.no_grad()
def mask_predict_with_modes_ar(model, mode, mel_prompt, chord_prompt,
                               prompt_length, gen_length, n_refine_steps,
                               temperature):
    """Drop-in replacement for cp_transformer_m2c_moe_mask_inference
    .mask_predict_with_modes that uses CP-structured AR sampling
    (model.local_sampling) within each masked block instead of the
    default parallel "softmax over the full vocab" sampling.

    The parallel sampler ignores the CP token-type alternation (program
    at even positions, pitch+dur at odd positions), routinely emitting
    invalid token-type combinations that produce broken-sounding output.
    local_sampling enforces the same per-position vocab constraints that
    the AR (non-mask) inference uses, sampling each within-block token
    given the previously-sampled tokens of that block.

    Block-level iterative refinement, mode dispatch, prompt/given/silence
    handling, and confidence-based "more-confident-first" picking are
    all unchanged from the original mask_predict_with_modes.
    """
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    subseq = mel_prompt.shape[2] if mel_prompt is not None else chord_prompt.shape[2]
    B = (mel_prompt.shape[0] if mel_prompt is not None
         else chord_prompt.shape[0])

    def silence_frame():
        f = torch.full((B, subseq), tokenizer.pad_token,
                       dtype=torch.long, device=device)
        f[:, 0] = tokenizer.eos_token
        return f

    if mode == 'co':
        assert mel_prompt is not None and chord_prompt is not None
        usable_mel = min(prompt_length, mel_prompt.shape[1])
        usable_chord = min(prompt_length, chord_prompt.shape[1])
        mel_truth = mel_prompt[:, :usable_mel]
        chord_truth = chord_prompt[:, :usable_chord]

        def mel_action(t):
            if t < usable_mel:
                return ('given', mel_truth[:, t, :])
            return 'sample'

        def chord_action(t):
            if t < usable_chord:
                return ('given', chord_truth[:, t, :])
            return 'sample'

    elif mode == 'mel2chord':
        assert mel_prompt is not None
        gen_length = min(gen_length, mel_prompt.shape[1])
        mel_truth = mel_prompt[:, :gen_length]

        def mel_action(t):
            return ('given', mel_truth[:, t, :])

        def chord_action(t):
            return 'sample'

    elif mode == 'chord2mel':
        assert chord_prompt is not None
        gen_length = min(gen_length, chord_prompt.shape[1])
        chord_truth = chord_prompt[:, :gen_length]

        def mel_action(t):
            return 'sample'

        def chord_action(t):
            return ('given', chord_truth[:, t, :])

    elif mode == 'mel_only':
        assert mel_prompt is not None
        usable_mel = min(prompt_length, mel_prompt.shape[1])
        mel_truth = mel_prompt[:, :usable_mel]

        def mel_action(t):
            if t < usable_mel:
                return ('given', mel_truth[:, t, :])
            return 'sample'

        def chord_action(t):
            return 'mask'

    elif mode == 'chord_only':
        assert chord_prompt is not None
        usable_chord = min(prompt_length, chord_prompt.shape[1])
        chord_truth = chord_prompt[:, :usable_chord]

        def mel_action(t):
            return 'mask'

        def chord_action(t):
            if t < usable_chord:
                return ('given', chord_truth[:, t, :])
            return 'sample'
    else:
        raise ValueError(mode)

    x_m_full = torch.empty(B, 0, subseq, dtype=torch.long, device=device)
    x_c_full = torch.empty(B, 0, subseq, dtype=torch.long, device=device)
    mask_m_full = torch.empty(B, 0, dtype=torch.bool, device=device)
    mask_c_full = torch.empty(B, 0, dtype=torch.bool, device=device)

    mel_frames = []
    chord_frames = []

    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[mask-predict-ar, mode={mode}] block {t}/{gen_length}')

        m_act = mel_action(t)
        c_act = chord_action(t)

        if isinstance(m_act, tuple) and m_act[0] == 'given':
            m_init = m_act[1]
            m_masked = False
            m_fill = False
        elif m_act == 'silence':
            m_init = silence_frame()
            m_masked = False
            m_fill = False
        elif m_act == 'mask':
            m_init = torch.full((B, subseq), model.mask_token_id,
                                dtype=torch.long, device=device)
            m_masked = True
            m_fill = False
        else:
            m_init = torch.full((B, subseq), model.mask_token_id,
                                dtype=torch.long, device=device)
            m_masked = True
            m_fill = True

        if isinstance(c_act, tuple) and c_act[0] == 'given':
            c_init = c_act[1]
            c_masked = False
            c_fill = False
        elif c_act == 'silence':
            c_init = silence_frame()
            c_masked = False
            c_fill = False
        elif c_act == 'mask':
            c_init = torch.full((B, subseq), model.mask_token_id,
                                dtype=torch.long, device=device)
            c_masked = True
            c_fill = False
        else:
            c_init = torch.full((B, subseq), model.mask_token_id,
                                dtype=torch.long, device=device)
            c_masked = True
            c_fill = True

        x_m_full = torch.cat([x_m_full, m_init.unsqueeze(1)], dim=1)
        x_c_full = torch.cat([x_c_full, c_init.unsqueeze(1)], dim=1)
        mask_m_full = torch.cat([
            mask_m_full, torch.full((B, 1), m_masked, dtype=torch.bool, device=device),
        ], dim=1)
        mask_c_full = torch.cat([
            mask_c_full, torch.full((B, 1), c_masked, dtype=torch.bool, device=device),
        ], dim=1)

        if not (m_fill or c_fill):
            mel_frames.append(x_m_full[:, t, :])
            chord_frames.append(x_c_full[:, t, :])
            continue

        m_fillable = (
            torch.full((B,), m_fill, dtype=torch.bool, device=device)
            & mask_m_full[:, t]
        )
        c_fillable = (
            torch.full((B,), c_fill, dtype=torch.bool, device=device)
            & mask_c_full[:, t]
        )

        for k in range(n_refine_steps):
            # Forward returns h_m_out / h_c_out (the global tower's per-block
            # hidden states) so we can drive local_sampling instead of the
            # broken parallel _sample_tokens path.
            logits_m, logits_c, _, h_m_out, h_c_out = model(
                x_m_full, x_c_full, mask_m_full, mask_c_full,
                return_h_out=True,
            )

            # CP-structured AR sampling within the current masked block.
            # token_type_id 0 = melody, 1 = chord. local_sampling enforces
            # program/pitch alternation and valid vocab ranges.
            m_sample = model._sample_block_within_block_ar(
                h_m_out[:, t], subseq, temperature, token_type_id=0,
            )
            c_sample = model._sample_block_within_block_ar(
                h_c_out[:, t], subseq, temperature, token_type_id=1,
            )

            last = (k == n_refine_steps - 1)
            if last:
                m_active = m_fillable.unsqueeze(-1)
                c_active = c_fillable.unsqueeze(-1)
                x_m_full[:, t] = torch.where(m_active, m_sample, x_m_full[:, t])
                x_c_full[:, t] = torch.where(c_active, c_sample, x_c_full[:, t])
                mask_m_full[:, t] = mask_m_full[:, t] & ~m_fillable
                mask_c_full[:, t] = mask_c_full[:, t] & ~c_fillable
                m_fillable = torch.zeros_like(m_fillable)
                c_fillable = torch.zeros_like(c_fillable)
            else:
                # Confidence over the WITHIN-block logits (still useful as
                # a per-modality "how sure am I?" signal, even though sampling
                # now goes through local_sampling).
                conf_m = torch.softmax(logits_m[:, t].float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                conf_c = torch.softmax(logits_c[:, t].float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                pick_m = (conf_m >= conf_c) & m_fillable
                pick_c = (conf_m < conf_c) & c_fillable
                only_m = m_fillable & ~c_fillable
                only_c = c_fillable & ~m_fillable
                pick_m = pick_m | only_m
                pick_c = pick_c | only_c
                x_m_full[:, t] = torch.where(pick_m.unsqueeze(-1), m_sample, x_m_full[:, t])
                x_c_full[:, t] = torch.where(pick_c.unsqueeze(-1), c_sample, x_c_full[:, t])
                mask_m_full[:, t] = mask_m_full[:, t] & ~pick_m
                mask_c_full[:, t] = mask_c_full[:, t] & ~pick_c
                m_fillable = m_fillable & ~pick_m
                c_fillable = c_fillable & ~pick_c

        mel_frames.append(x_m_full[:, t, :])
        chord_frames.append(x_c_full[:, t, :])

    return mel_frames, chord_frames


# Monkey-patch: route mask_inf.run_one (and indirectly main()) through the
# AR-within-block sampler.
mask_inf.mask_predict_with_modes = mask_predict_with_modes_ar


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
