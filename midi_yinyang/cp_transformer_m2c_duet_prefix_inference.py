"""Inference for M2CDuetPrefix (variant C.2) -- drum -> nondrum only.

Same drum->nondrum semantics as the rehearsal variant's inference,
but for the prefix-LM architecture:

  Sequence: [drum_0..drum_{T-1}] (bidirectional within prefix) +
            [sos_n, nondrum_0..nondrum_{T-2}] (causal within nondrum,
            reads all drum)
  Total:    2T

At inference we encode the drum prefix once, then iteratively AR-decode
the nondrum block.

Output: one .mid per drum-prompt file. Drum tracks are the original
prompt; nondrum tracks are model-generated.

Single-GPU. Example:

    python cp_transformer_m2c_duet_prefix_inference.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --drum-folder input/rwc_test_prompts_split/drum \\
        --output-dir temp/duet_prefix_rwc \\
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
from glob import glob

import torch

from cp_transformer_m2c_duet_prefix import M2CDuetPrefix
from cp_transformer_m2c_jointattn_inference import (
    _infer_global_num_layers,
    _get_input_tempo,
    decode_m2c_frames,
    resolve_best_ckpt,
)
from cp_transformer_m2c_moe_inference import _load_prompt_tokens
from cp_transformer_m2c_moe import TRAIN_LENGTH


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
    net = M2CDuetPrefix(
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


def drum_to_nondrum(model, drum_tokens, nondrum_prompt_tokens, gen_length,
                    temperature):
    """Run drum->nondrum AR decoding for the prefix-LM variant.

    Args:
        drum_tokens: [1, T_target, subseq] -- the full drum prefix.
        nondrum_prompt_tokens: [1, P, subseq] or None -- optional nondrum
            prompt for the first P frames.
        gen_length: total frames to produce. Capped at T_target.

    Returns:
        (mel_frames, chord_frames) -- lists of [1, subseq] tensors, length gen_length.
        mel_frames is just the prompt drum (the condition).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    H = model.hidden_size

    B = 1
    T_target = drum_tokens.shape[1]
    gen_length = min(gen_length, T_target)
    P_nondrum = (
        nondrum_prompt_tokens.shape[1]
        if nondrum_prompt_tokens is not None else 0
    )

    # 1. Encode the drum prompt as the bidirectional prefix.
    h_drum_frames = []
    for t in range(T_target):
        frame = drum_tokens[:, t, :]
        h_d = model._encode_frame(frame, 0)
        h_drum_frames.append(h_d)
    h_drum = torch.cat(h_drum_frames, dim=1).to(dtype=dtype)   # [1, T_target, H]
    print(f'[prefix] encoded drum prefix: T_target={T_target}')

    # 2. Iterative AR over the nondrum block. Suffix layout in training:
    #    [sos_n, nondrum_0, nondrum_1, …, nondrum_{T-2}] (length T, shifted).
    #    Prediction at position t targets nondrum_t.
    sos_n = (model.global_sos + model.sos_offset_c).view(1, 1, -1)
    sos_n = sos_n.expand(B, 1, -1).to(device=device, dtype=dtype)

    # nondrum buffer holds the encoded committed nondrum hiddens after
    # the SOS slot. At step 0, suffix = [sos_n]; at step t, suffix has
    # t+1 positions = [sos_n] + nondrum_buffer (length t).
    h_nondrum_buffer = torch.zeros(B, 0, H, device=device, dtype=dtype)
    chord_frames = []

    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[gen] step {t}/{gen_length}')

        # Build full input. Total length must be 2*T_target.
        suffix_committed = torch.cat([sos_n, h_nondrum_buffer], dim=1)
        # suffix_committed.shape[1] = 1 + t
        if suffix_committed.shape[1] < T_target:
            pad_len = T_target - suffix_committed.shape[1]
            pad = torch.zeros(B, pad_len, H, device=device, dtype=dtype)
            suffix_full = torch.cat([suffix_committed, pad], dim=1)
        else:
            suffix_full = suffix_committed[:, :T_target]

        h_in = torch.cat([h_drum, suffix_full], dim=1)   # [1, 2*T_target, H]
        h_global, _ = model._run_global_stack(h_in, T=T_target)

        # Prediction at position T_target + t targets nondrum_t.
        h_c_pred = h_global[:, T_target + t]   # [1, H]

        # Teacher-force from nondrum prompt if available; else sample.
        if t < P_nondrum:
            c_tokens = nondrum_prompt_tokens[:, t, :]
        else:
            c_tokens = model.local_sampling(
                h_c_pred, max_subseq_len=drum_tokens.shape[2],
                temperature=temperature, token_type_id=1,
            )

        chord_frames.append(c_tokens)
        c_h = model._encode_frame(c_tokens, 1).to(dtype=dtype)
        h_nondrum_buffer = torch.cat([h_nondrum_buffer, c_h], dim=1)

    # Drum side is just the prompt (the condition; not predicted by this variant).
    mel_frames = [drum_tokens[:, t, :] for t in range(gen_length)]
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

    print(f'[infer] {len(drum_files)} drum prompts')
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
            # Cap the drum CONDITION at gen_length. Without this, a
            # full-song drum file sets T = its full frame count -- far
            # beyond the trained T (TRAIN_LENGTH), pushing RoPE phases
            # and attention spans way out of the training distribution
            # and making each decode step O((k*T)^2) for whole-song T.
            # gen_length is already capped at T elsewhere; capping the
            # condition too keeps the sequence at trained geometry.
            if drum_tokens.shape[1] > args.gen_length:
                print(f'  [cap] drum condition {drum_tokens.shape[1]} '
                      f'-> {args.gen_length} frames')
                drum_tokens = drum_tokens[:, :args.gen_length]
            if args.gen_length > TRAIN_LENGTH:
                print(f'  [warn] gen_length {args.gen_length} > trained '
                      f'length {TRAIN_LENGTH}; geometry extrapolation.')
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
        description='M2CDuetPrefix (C.2) inference: drum -> nondrum only.',
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
    args = p.parse_args()

    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        min_acc_tokens_before_eos=args.min_chord_tokens_before_eos,
    )
    model = model.cuda().eval()
    model.save_name = os.path.basename(os.path.dirname(args.ckpt))

    with torch.no_grad():
        run_folder(model, args)


if __name__ == '__main__':
    main()
