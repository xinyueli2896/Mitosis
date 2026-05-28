"""Inference for the M2C MoE mask-predict model.

Matches training: both run the model with [MASK] at hidden positions through
the same pair-causal + symmetric-RoPE global attention. Inference adds
block-by-block extension with K within-block refinement passes that fill the
masked slots in confidence order, giving m_t and c_t at the SAME timestep
mutual dependency at sampling time.

Three modes (same surface as cp_transformer_m2c_moe_inference.py):
  co            both prompts; sample both for [prompt_length, gen_length)
  mel2chord     full mel given; chord generated for the same length
  chord2mel     symmetric
  mel_only      mel prompt + continuation, chord forced to a silence frame
  chord_only    symmetric

Run from midi_yinyang/:

    python cp_transformer_m2c_moe_mask_inference.py \
        --ckpt ckpt/<run>/last.ckpt \
        --mode co \
        --melody POP909-Dataset/POP909-melody/001.mid \
        --chord  POP909-Dataset/POP909-chord/001.mid \
        --prompt-length 100 --gen-length 384 \
        --n-refine-steps 2 \
        --temperature 1.0
"""

# Inject the vendored transformers fork BEFORE anything else imports
# (see cp_transformer_m2c_moe_mask.py for the full reasoning).
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import os
from glob import glob

import torch
import pretty_midi

from cp_transformer_m2c_moe_mask import M2CMaskMoE
from cp_transformer_m2c_moe_inference import (
    decode_m2c_frames,
    _load_raw,
)
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES


MODES = ['co', 'mel2chord', 'chord2mel', 'mel_only', 'chord_only']


def _load_prompt_tokens(model, midi_path, max_polyphony):
    raw = _load_raw(midi_path, max_polyphony).unsqueeze(0).cuda()
    pitch_shift = torch.zeros(1, dtype=torch.int8, device=raw.device)
    return model.preprocess(raw, pitch_shift)  # [1, T, subseq]


def load_model(ckpt_path, model_size='small', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None,
               mel_loss_weight=1.0, acc_loss_weight=1.0):
    net = M2CMaskMoE(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        mel_loss_weight=mel_loss_weight,
        acc_loss_weight=acc_loss_weight,
    )
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing keys: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected keys: {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
    return net


# ---------------------------------------------------------------------------
# Mode-aware mask-predict sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def mask_predict_with_modes(model, mode, mel_prompt, chord_prompt,
                            prompt_length, gen_length, n_refine_steps,
                            temperature):
    """Generalizes M2CMaskMoE.mask_predict_sampling to handle the same three
    modes as the parallel-decoding inference: co, mel2chord, chord2mel,
    mel_only, chord_only.

    For each block t in [0, gen_length):
      - decide per-modality whether the slot is:
          'given'    (use ground truth from a prompt / condition)
          'sample'   (let the model fill it via mask + iterative refinement)
          'silence'  (force a dummy frame: a-slot=eos, rest=pad)
      - if any slot is 'sample', run K refinement passes; otherwise just
        write the chosen frame.
    """
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    subseq = mel_prompt.shape[2] if mel_prompt is not None else chord_prompt.shape[2]
    B = (mel_prompt.shape[0] if mel_prompt is not None
         else chord_prompt.shape[0])

    # Per-timestep action plan
    def given_or_sample(prompt, t, modality_given_for_all):
        """('given', tensor) if t < usable prompt length OR modality is fully given,
        else 'sample'."""
        if prompt is None:
            return 'sample'
        if t < prompt.shape[1]:
            return ('given', prompt[:, t, :])
        return 'sample'

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
        # Keep the chord as [MASK] throughout, not silence frames. The mask-
        # predict model has seen "chord modality fully masked" during random-
        # mask training, so its mask_emb_c is an in-distribution signal for
        # the mel predictions to condition on. Silence frames, by contrast,
        # never appeared during training and pull mel generation off-manifold.
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

    # Build buffer incrementally. At each block t, decide actions, then either
    # (a) write the chosen frames directly (no sampling needed), or
    # (b) run K refinement passes for the slot(s) marked 'sample'.
    x_m_full = torch.empty(B, 0, subseq, dtype=torch.long, device=device)
    x_c_full = torch.empty(B, 0, subseq, dtype=torch.long, device=device)
    mask_m_full = torch.empty(B, 0, dtype=torch.bool, device=device)
    mask_c_full = torch.empty(B, 0, dtype=torch.bool, device=device)

    mel_frames = []
    chord_frames = []

    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[mask-predict, mode={mode}] block {t}/{gen_length}')

        m_act = mel_action(t)
        c_act = chord_action(t)

        # Track per-action intent. 'sample' slots get filled during
        # refinement; 'mask' slots STAY masked permanently (used by
        # mel_only / chord_only so the silenced modality looks like
        # the mask-predict training distribution rather than silence
        # tokens the model never saw).
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
        else:  # 'sample'
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

        # If nothing to fill, we're done for this block. Slots marked 'mask'
        # remain masked in mask_*_full so future blocks' forward passes still
        # see the mask_emb signal for the silenced modality.
        if not (m_fill or c_fill):
            mel_frames.append(x_m_full[:, t, :])
            chord_frames.append(x_c_full[:, t, :])
            continue

        # K refinement passes for the slot(s) marked _fill. Slots marked
        # 'mask' stay masked here -- the model's forward pass still sees
        # them as mask_emb, contributing in-distribution context to the
        # slot we're actually filling, but we never overwrite their tokens.
        # mask_full_fillable: still-masked AND was marked 'fill'.
        m_fillable = (
            torch.full((B,), m_fill, dtype=torch.bool, device=device)
            & mask_m_full[:, t]
        )
        c_fillable = (
            torch.full((B,), c_fill, dtype=torch.bool, device=device)
            & mask_c_full[:, t]
        )
        for k in range(n_refine_steps):
            logits_m, logits_c, _ = model(
                x_m_full, x_c_full, mask_m_full, mask_c_full,
            )
            m_block_logits = logits_m[:, t]
            c_block_logits = logits_c[:, t]

            m_sample = model._sample_tokens(m_block_logits, temperature)
            c_sample = model._sample_tokens(c_block_logits, temperature)

            last = (k == n_refine_steps - 1)
            if last:
                m_active = m_fillable.unsqueeze(-1)
                c_active = c_fillable.unsqueeze(-1)
                x_m_full[:, t] = torch.where(m_active, m_sample, x_m_full[:, t])
                x_c_full[:, t] = torch.where(c_active, c_sample, x_c_full[:, t])
                # Unmask filled slots; leave 'mask'-action slots masked.
                mask_m_full[:, t] = mask_m_full[:, t] & ~m_fillable
                mask_c_full[:, t] = mask_c_full[:, t] & ~c_fillable
                m_fillable = torch.zeros_like(m_fillable)
                c_fillable = torch.zeros_like(c_fillable)
            else:
                # Pick more-confident still-fillable slot.
                conf_m = torch.softmax(m_block_logits.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                conf_c = torch.softmax(c_block_logits.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
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


# ---------------------------------------------------------------------------
# Per-pair runner & CLI
# ---------------------------------------------------------------------------

def _list_midis(folder):
    if folder is None:
        return []
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


def run_one(model, mode, mel_path, chord_path, args, out_subdir):
    mel_prompt = (_load_prompt_tokens(model, mel_path, args.max_polyphony)
                  if mel_path else None)
    chord_prompt = (_load_prompt_tokens(model, chord_path, args.max_polyphony)
                    if chord_path else None)

    mel_frames, chord_frames = mask_predict_with_modes(
        model, mode, mel_prompt, chord_prompt,
        prompt_length=args.prompt_length,
        gen_length=args.gen_length,
        n_refine_steps=args.n_refine_steps,
        temperature=args.temperature,
    )

    save_name = getattr(model, 'save_name', 'm2c_mask_run')
    out_dir = os.path.join(f'temp/{save_name}', out_subdir, mode)
    write_mel = mode != 'chord_only'
    write_chord = mode != 'mel_only'
    # Each frame is [1, subseq]; write the single sample.
    decode_m2c_frames(
        mel_frames, chord_frames,
        save_path=os.path.join(
            out_dir,
            f'sample_temp{args.temperature}_K{args.n_refine_steps}.mid',
        ),
        tokenizer=model.tokenizer,
        with_velocity=model.with_velocity,
        write_mel=write_mel,
        write_chord=write_chord,
    )


def run_folder(model, mode, mel_folder, chord_folder, args):
    mel_files = _list_midis(mel_folder)
    chord_files = _list_midis(chord_folder)
    chord_index = {os.path.basename(p): p for p in chord_files}
    if mode in ('chord2mel', 'chord_only'):
        driver = chord_files
        items = [(({os.path.basename(m): m for m in mel_files}).get(os.path.basename(p)), p)
                 for p in driver]
    else:
        driver = mel_files
        items = [(p, chord_index.get(os.path.basename(p))) for p in driver]
    print(f'Found {len(items)} items in driver folder for mode={mode}.')
    for i, (mp, cp) in enumerate(items):
        base = mp if mp is not None else cp
        sid = os.path.splitext(os.path.basename(base))[0]
        print(f'[{i + 1}/{len(items)}] {sid}')
        try:
            run_one(model, mode, mp, cp, args, out_subdir=sid)
        except Exception as e:
            print(f'  failed: {e!r}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--melody', help='single melody midi')
    p.add_argument('--chord', help='single chord midi')
    p.add_argument('--mel-folder', help='folder of melody midis')
    p.add_argument('--chord-folder', help='folder of chord midis')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-refine-steps', type=int, default=2,
                   help='within-block refinement passes. 1 = parallel '
                        '(baseline, no intra-block dependency). 2 = MaskGIT/'
                        'Gibbs (bidirectional same-step coupling).')
    p.add_argument('--max-polyphony', type=int, default=4)
    p.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
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
    )
    model.save_name = os.path.basename(args.ckpt)
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
