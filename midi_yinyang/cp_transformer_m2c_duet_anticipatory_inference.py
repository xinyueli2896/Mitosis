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

    print(f'[infer] {len(drum_files)} drum prompts  '
          f'(anticipation k={model.anticipation_frames})')
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

            mel_frames, chord_frames = drum_to_nondrum(
                model, drum_tokens, nondrum_prompt_tokens,
                gen_length=args.gen_length, temperature=args.temperature,
            )

            out_dir = os.path.join(args.output_dir, sid)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'drum2nondrum_temp{args.temperature}.mid')
            output_tempo = _get_input_tempo(drum_path, default=120.0)
            print(f'[tempo] source={drum_path} -> tempo={output_tempo:.2f} BPM')
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
    p.add_argument('--drum-folder', required=True)
    p.add_argument('--nondrum-folder',
                   help='Optional nondrum prompt folder (paired by basename).')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--prompt-length', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-polyphony', type=int, default=16)
    p.add_argument('--model-size', type=str, default='large',
                   choices=['small', 'large'])
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--min-chord-tokens-before-eos', type=int, default=0)
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
