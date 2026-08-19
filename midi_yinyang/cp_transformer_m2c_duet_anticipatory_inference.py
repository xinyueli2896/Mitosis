"""Inference for M2CDuetAnticipatory (variant B.1) -- drum -> nondrum only.

B.1 is architecturally identical to A.1 (M2CIntraCrossAttn); training
differs only in the input relabeling: the mod-a (drum) stream is shifted
ahead by k frames, so mod-a index t holds drum_{t+k} and the last k
mod-a slots are all-PAD frames. Nondrum predictions therefore condition
on k frames of FUTURE drum context under standard causal AR.

Inference must reproduce exactly that relabeling:

  * mod-a action at step t = ('given', drum[t+k])  for t + k < T_drum,
    else an all-PAD frame (matching the training pad-tail convention --
    note: all-PAD, NOT the eos+pad "silence" frame).
  * mod-b (nondrum) is teacher-forced from an optional prompt prefix,
    then sampled.

The generated nondrum stream lives on the REAL timeline (index t =
frame t), so no un-shifting is needed for it. The output MIDI contains
the ORIGINAL (unshifted) drum track + the generated nondrum.

The anticipation k is auto-detected from the ckpt path (the training
run dir embeds it as ..._k16_...); --anticipation-frames overrides.

Single-GPU. Example:

    python cp_transformer_m2c_duet_anticipatory_inference.py \\
        --ckpt ckpt/m2c_duet_anticipatory_v1.0_large_gnl12_drumnondrum_k16_batch_16_schedule/ \\
        --drum-folder input/rwc_test_prompts_split/drum \\
        --nondrum-folder input/rwc_test_prompts_split/nondrum \\
        --output-dir temp/duet_anticipatory_rwc \\
        --gen-length 384 --prompt-length 64 \\
        --temperature 1.0 --max-polyphony 16 --model-size large
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import os
import re

import torch

from cp_transformer_m2c_duet_anticipatory import M2CDuetAnticipatory
from cp_transformer_m2c_jointattn_inference import (
    _infer_global_num_layers,
    _get_input_tempo,
    decode_m2c_frames,
    resolve_best_ckpt,
)
from cp_transformer_m2c_moe_inference import (
    general_inference,
    _load_prompt_tokens,
)


def detect_anticipation_frames(ckpt_path, default=16):
    """Parse k from a path like .../m2c_duet_anticipatory_..._k16_batch...."""
    m = re.search(r'_k(\d+)_', ckpt_path)
    if m:
        return int(m.group(1))
    return default


def load_model(ckpt_path, model_size='large', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None,
               global_num_layers=None, preserve_program=True,
               min_acc_tokens_before_eos=0, gate_init_bias=-10.0,
               anticipation_frames=None):
    ckpt_path = resolve_best_ckpt(ckpt_path)
    if anticipation_frames is None:
        anticipation_frames = detect_anticipation_frames(ckpt_path)
        print(f'[load_model] anticipation_frames={anticipation_frames} '
              f'(auto-detected from ckpt path)')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, source = _infer_global_num_layers(
            ckpt_path, ck, model_size,
        )
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto-detected from {source})')
    net = M2CDuetAnticipatory(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        global_num_layers=global_num_layers,
        preserve_program=preserve_program,
        min_acc_tokens_before_eos=min_acc_tokens_before_eos,
        gate_init_bias=gate_init_bias,
        anticipation_frames=anticipation_frames,
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


def drum_to_nondrum(model, drum_tokens, nondrum_prompt_tokens, gen_length,
                    temperature):
    """Drum -> nondrum decoding with the anticipatory relabeling.

    Args:
        drum_tokens: [1, T_drum, subseq] full (processed) drum condition,
            on the REAL timeline.
        nondrum_prompt_tokens: [1, P, subseq] or None.
        gen_length: frames of nondrum to produce. Capped at T_drum.

    Returns (mel_frames, chord_frames): mel_frames is the ORIGINAL
    (unshifted) drum for output writing; chord_frames the generated
    nondrum. Both lists of [1, subseq], length gen_length.
    """
    tokenizer = model.tokenizer
    device = drum_tokens.device
    B, T_drum, S = drum_tokens.shape
    k = model.anticipation_frames
    gen_length = min(gen_length, T_drum)
    P = (nondrum_prompt_tokens.shape[1]
         if nondrum_prompt_tokens is not None else 0)

    # Training pad-tail convention: ALL pad tokens (not eos+pad silence).
    pad_frame = torch.full((B, S), tokenizer.pad_token,
                            dtype=drum_tokens.dtype, device=device)

    def mel_action(t):
        # mod-a index t holds drum_{t+k}; PAD tail past the end.
        if t + k < T_drum:
            return ('given', drum_tokens[:, t + k])
        return ('given', pad_frame)

    def chord_action(t):
        if t < P:
            return ('given', nondrum_prompt_tokens[:, t])
        return 'sample'

    _, chord_frames = general_inference(
        model, gen_length, B, S,
        temperature=temperature,
        mel_action_fn=mel_action,
        chord_action_fn=chord_action,
    )

    # Output drum = the real, unshifted condition.
    mel_frames = [drum_tokens[:, t] for t in range(gen_length)]
    return mel_frames, chord_frames


def co_generate(model, drum_prompt_tokens, nondrum_prompt_tokens,
                gen_length, temperature):
    """CO-GENERATION with the anticipatory relabeling: BOTH streams are
    generated past the prompt (nothing is given as ground truth).

    The layout the model was trained on puts drum_{s+k} at mod-a slot s
    and nondrum_s at mod-b slot s, so one decode step s emits a drum
    frame k ahead of the nondrum frame. Consequences, all handled here:

      * mod-a slot s is teacher-forced while s+k < P (still inside the
        prompt) and SAMPLED afterwards -- so the drum stream starts
        being generated at step P-k, k steps before the nondrum stream
        does. That asymmetry is the mechanism under test, not a bug.
      * the loop must run gen_length steps to emit nondrum_{L-1}; along
        the way it emits drum frames up to L-1+k, which are truncated.
      * drum frames 0..k-1 never appear in the model's sequence at all
        (their slot index would be negative), so they are taken from
        the prompt. This requires k <= P.

    Returns (drum_frames, nondrum_frames), each a list of gen_length
    [B, subseq] tensors on the REAL timeline: frames 0..P-1 are the
    prompt, P..L-1 are generated. Directly comparable to A.2's co-mode
    output.
    """
    device = drum_prompt_tokens.device
    B, P_drum, S = drum_prompt_tokens.shape
    k = model.anticipation_frames
    P_nd = (nondrum_prompt_tokens.shape[1]
            if nondrum_prompt_tokens is not None else 0)
    P = min(P_drum, P_nd) if P_nd else P_drum
    if k > P:
        raise ValueError(
            f'anticipation k={k} exceeds prompt length P={P}: drum frames '
            f'{P}..{k - 1} would be neither prompted nor generated (their '
            f'mod-a slot index is negative). Use a prompt of at least k '
            f'frames.'
        )

    def mel_action(s):
        # mod-a slot s holds drum_{s+k}.
        d = s + k
        if d < P:
            return ('given', drum_prompt_tokens[:, d])
        return 'sample'

    def chord_action(s):
        if s < P and nondrum_prompt_tokens is not None:
            return ('given', nondrum_prompt_tokens[:, s])
        return 'sample'

    shifted_drum, nondrum_frames = general_inference(
        model, gen_length, B, S,
        temperature=temperature,
        mel_action_fn=mel_action,
        chord_action_fn=chord_action,
    )

    # Un-shift the drum stream back onto the real timeline:
    # shifted_drum[s] is drum_{s+k}; drum_0..drum_{k-1} come from the prompt.
    drum_frames = [None] * gen_length
    for i in range(min(k, gen_length)):
        drum_frames[i] = drum_prompt_tokens[:, i]
    for s in range(gen_length):
        d = s + k
        if d < gen_length:
            drum_frames[d] = shifted_drum[s]
    # Safety net: any slot still unset (only possible if gen_length < k)
    # falls back to the prompt frame at that index.
    for i in range(gen_length):
        if drum_frames[i] is None:
            drum_frames[i] = drum_prompt_tokens[:, min(i, P_drum - 1)]
    return drum_frames, nondrum_frames


def _list_midis(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def run_folder(model, args):
    drum_files = _list_midis(args.drum_folder)
    if args.nondrum_folder:
        nondrum_by_base = {
            os.path.basename(p): p for p in _list_midis(args.nondrum_folder)
        }
    else:
        nondrum_by_base = {}

    if args.max_songs is not None:
        drum_files = drum_files[:args.max_songs]

    print(f'[infer] {len(drum_files)} drum prompts  mode={args.mode}  '
          f'(anticipation k={model.anticipation_frames})')
    if args.mode == 'co' and not args.nondrum_folder:
        raise SystemExit('--mode co requires --nondrum-folder '
                         '(both streams must be prompted)')
    os.makedirs(args.output_dir, exist_ok=True)

    for i, drum_path in enumerate(drum_files):
        base = os.path.basename(drum_path)
        sid = os.path.splitext(base)[0]
        nondrum_path = nondrum_by_base.get(base)
        print(f'[{i + 1}/{len(drum_files)}] {sid}'
              + (f'  (nondrum prompt: {nondrum_path})' if nondrum_path else ''))
        try:
            drum_tokens = _load_prompt_tokens(model, drum_path,
                                               args.max_polyphony)
            nondrum_prompt_tokens = None
            if nondrum_path is not None and args.prompt_length > 0:
                nondrum_prompt_tokens = _load_prompt_tokens(
                    model, nondrum_path, args.max_polyphony,
                )
                if nondrum_prompt_tokens.shape[1] > args.prompt_length:
                    nondrum_prompt_tokens = nondrum_prompt_tokens[
                        :, :args.prompt_length,
                    ]
                if nondrum_prompt_tokens.shape[1] > drum_tokens.shape[1]:
                    nondrum_prompt_tokens = nondrum_prompt_tokens[
                        :, :drum_tokens.shape[1],
                    ]

            # Prompt loading above is deterministic; only generation is
            # stochastic, so just the draw + write repeat. Matching the
            # baselines' sample count matters: the evaluation averages
            # samples per song before the paired test, so a system left at
            # one draw carries more per-song sampling variance than its
            # competitors and loses power in the comparison itself.
            for s in range(args.n_samples):
                if args.mode == 'co':
                    # Co-generation: prompt BOTH streams, generate both.
                    drum_prompt = drum_tokens
                    if args.prompt_length > 0:
                        drum_prompt = drum_tokens[:, :args.prompt_length]
                    if nondrum_prompt_tokens is None:
                        raise ValueError(
                            'co mode needs a nondrum prompt: pass '
                            '--nondrum-folder with a file matching this song')
                    mel_frames, chord_frames = co_generate(
                        model, drum_prompt, nondrum_prompt_tokens,
                        gen_length=args.gen_length,
                        temperature=args.temperature,
                    )
                else:
                    mel_frames, chord_frames = drum_to_nondrum(
                        model, drum_tokens, nondrum_prompt_tokens,
                        gen_length=args.gen_length,
                        temperature=args.temperature,
                    )

                out_dir = os.path.join(args.output_dir, sid)
                label = 'co' if args.mode == 'co' else args.mode_name
                if args.n_samples > 1:
                    # 'duet_multi': <song>/<mode-name>/sample_<i>_temp<T>.mid
                    out_dir = os.path.join(out_dir, label)
                    out_name = f'sample_{s}_temp{args.temperature}.mid'
                else:
                    # 'duet': 'co' writes <song>/co.mid so the output matches
                    # the layout the eval manifest builder scans.
                    out_name = ('co.mid' if args.mode == 'co'
                                else f'{label}_temp{args.temperature}.mid')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, out_name)
                output_tempo = _get_input_tempo(drum_path, default=120.0)
                decode_m2c_frames(
                    mel_frames, chord_frames,
                    save_path=out_path,
                    tokenizer=model.tokenizer,
                    with_velocity=model.with_velocity,
                    tempo=output_tempo,
                    write_mel=True,
                    write_chord=True,
                )
                print(f'  wrote {out_path}')
        except Exception as e:
            print(f'  failed: {e!r}')


def main():
    p = argparse.ArgumentParser(
        description='M2CDuetAnticipatory (B.1) inference: drum -> nondrum '
                    'with k-frame drum lookahead.',
    )
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', default='drum2nondrum',
                   choices=['drum2nondrum', 'co'],
                   help='drum2nondrum (default): the FULL drum stream is '
                        'given as ground truth and only nondrum is '
                        'generated (conditional, E3). co: both streams are '
                        'prompted for --prompt-length frames and both are '
                        'generated afterwards (co-generation, E1); requires '
                        '--nondrum-folder and writes <song>/co.mid.')
    p.add_argument('--drum-folder', required=True)
    p.add_argument('--nondrum-folder',
                   help='Optional nondrum prompt folder (paired by basename). '
                        'REQUIRED for --mode co.')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--prompt-length', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-samples', dest='n_samples', type=int, default=1,
                   help='independent continuations per song. >1 writes the '
                        "'duet_multi' manifest layout "
                        '(<song>/<mode>/sample_<i>_temp<T>.mid) instead of '
                        "'duet'; pass the SAME value as the other systems "
                        'so per-song sampling variance is comparable.')
    p.add_argument('--max-polyphony', type=int, default=16)
    p.add_argument('--model-size', type=str, default='large',
                   choices=['small', 'large'])
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--min-chord-tokens-before-eos', type=int, default=0)
    p.add_argument('--mode-name', dest='mode_name', default='drum2nondrum',
                   help='LABEL used for the output folder/file (the eval '
                        "manifest reads the mode from it). Purely cosmetic "
                        'to the model -- the conditioning is always mod_a '
                        'given -> mod_b generated -- but on the MELCHORD '
                        'task the drum/nondrum wording is wrong: pass '
                        'mel2chord (or chord2mel for a reversed ckpt) so the '
                        'outputs on disk name the direction they actually '
                        'contain. Default keeps the drumnondrum name.')
    p.add_argument('--max-songs', type=int, default=None)
    p.add_argument('--anticipation-frames', type=int, default=None,
                   help='Drum lookahead k used at TRAINING time. Default: '
                        'auto-detect from the ckpt path (_k<N>_), falling '
                        'back to 16. MUST match training or the model '
                        'conditions on the wrong drum frames.')
    args = p.parse_args()

    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        min_acc_tokens_before_eos=args.min_chord_tokens_before_eos,
        anticipation_frames=args.anticipation_frames,
    )
    model = model.cuda().eval()
    model.save_name = os.path.basename(os.path.dirname(args.ckpt))

    with torch.no_grad():
        run_folder(model, args)


if __name__ == '__main__':
    main()
