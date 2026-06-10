"""Combined inference for M2CJointAttn: run all five modes on every paired
(melody, chord) midi from two folders and save EACH (song, mode) as its
own .mid file. Per-program track splitting (drums to is_drum=True, each
non-drum program as its own track) is preserved per file.

Output layout:

    <output-dir>/
        <song_id_0>/
            co.mid
            mel2chord.mid
            chord2mel.mid
            mel_only.mid
            chord_only.mid
        <song_id_1>/
            ...

Run from midi_yinyang/:

    python cp_transformer_m2c_jointattn_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder input/lamd_test_prompts_split/nondrum \\
        --chord-folder input/lamd_test_prompts_split/drum \\
        --output-dir temp/m2c_jointattn_combined \\
        --prompt-length 100 --gen-length 384 \\
        --max-songs 10 --temperature 1.0 \\
        --max-polyphony 16 --model-size large

Use --modes to restrict which modes run (default: all five).
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

from cp_transformer_m2c_moe_inference import (
    general_inference,
    make_actions_co,
    make_actions_conditional,
    make_actions_single,
    _load_prompt_tokens,
)
# Use jointattn's multi-program decoder (drum -> is_drum=True, each
# pitched program as its own Instrument).
from cp_transformer_m2c_jointattn_inference import (
    load_model,
    decode_m2c_frames,
)


MODES = ['co', 'mel2chord', 'chord2mel', 'mel_only', 'chord_only']


def _list_midis(folder):
    if folder is None:
        return []
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


def run_mode_for_song(model, mode, mel_path, chord_path, args):
    """Returns (mel_frames, chord_frames) or (None, None) if inputs missing."""
    if mode == 'co':
        if not mel_path or not chord_path:
            return None, None
        mel_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        chord_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        common = min(mel_prompt.shape[1], chord_prompt.shape[1], args.prompt_length)
        mel_prompt = mel_prompt[:, :common]
        chord_prompt = chord_prompt[:, :common]
        subseq_len = mel_prompt.shape[2]
        mel_action, chord_action = make_actions_co(mel_prompt, chord_prompt, common)
        gen_length = args.gen_length

    elif mode == 'mel2chord':
        if not mel_path:
            return None, None
        condition = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        b_prompt = None
        if chord_path and args.prompt_length > 0:
            b_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
            b_prompt = b_prompt[:, :args.prompt_length]
        gen_length = min(args.gen_length, condition.shape[1])
        condition = condition[:, :gen_length]
        subseq_len = condition.shape[2]
        mel_action, chord_action = make_actions_conditional(
            condition, 'mel', b_prompt=b_prompt,
        )

    elif mode == 'chord2mel':
        if not chord_path:
            return None, None
        condition = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        b_prompt = None
        if mel_path and args.prompt_length > 0:
            b_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
            b_prompt = b_prompt[:, :args.prompt_length]
        gen_length = min(args.gen_length, condition.shape[1])
        condition = condition[:, :gen_length]
        subseq_len = condition.shape[2]
        mel_action, chord_action = make_actions_conditional(
            condition, 'chord', b_prompt=b_prompt,
        )

    elif mode == 'mel_only':
        if not mel_path:
            return None, None
        prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        common = min(prompt.shape[1], args.prompt_length)
        prompt = prompt[:, :common]
        subseq_len = prompt.shape[2]
        mel_action, chord_action = make_actions_single(prompt, 'mel', common)
        gen_length = args.gen_length

    elif mode == 'chord_only':
        if not chord_path:
            return None, None
        prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        common = min(prompt.shape[1], args.prompt_length)
        prompt = prompt[:, :common]
        subseq_len = prompt.shape[2]
        mel_action, chord_action = make_actions_single(prompt, 'chord', common)
        gen_length = args.gen_length

    else:
        raise ValueError(f'unknown mode {mode}')

    mel_frames, chord_frames = general_inference(
        model, gen_length, B=1, subseq_len=subseq_len,
        temperature=args.temperature,
        mel_action_fn=mel_action,
        chord_action_fn=chord_action,
    )
    return mel_frames, chord_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mel-folder', required=True)
    p.add_argument('--chord-folder', required=True)
    p.add_argument('--output-dir', required=True,
                   help='Root directory; each song gets a subdirectory and '
                        'each mode its own .mid inside.')
    p.add_argument('--modes', nargs='+', default=MODES, choices=MODES,
                   help='Which modes to run. Default: all five.')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-polyphony', type=int, default=16)
    p.add_argument('--max-songs', type=int, default=None)
    p.add_argument('--model-size', type=str, default='large',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
    p.add_argument('--global-num-layers', type=int, default=None)
    p.add_argument('--preserve-program', dest='preserve_program',
                   action='store_true', default=True,
                   help='Preserve per-note program at inference (default '
                        'True for M2CJointAttn; must match training).')
    p.add_argument('--hardcode-program', dest='preserve_program',
                   action='store_false',
                   help='Squash program to 24 (mel) / 0 (chord), POP909-style.')
    p.add_argument('--min-chord-tokens-before-eos', dest='min_acc_tokens_before_eos',
                   type=int, default=0,
                   help='Force chord/non-drum to emit at least N tokens '
                        'before EOS within each frame. Rescue knob for '
                        'EOS-collapsed ckpts. Try 2 (>=1 full note) or 4.')
    args = p.parse_args()

    mel_files = _list_midis(args.mel_folder)
    chord_index = {os.path.basename(q): q for q in _list_midis(args.chord_folder)}
    pairs = [(m, chord_index[os.path.basename(m)]) for m in mel_files
             if os.path.basename(m) in chord_index]
    skipped = len(mel_files) - len(pairs)
    if args.max_songs is not None:
        pairs = pairs[:args.max_songs]
    print(f'Pairing: {len(pairs)} matched, {skipped} unmatched, '
          f'processing {len(pairs)} songs.')

    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        with_velocity=args.with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=args.global_num_layers,
        preserve_program=args.preserve_program,
        min_acc_tokens_before_eos=args.min_acc_tokens_before_eos,
    )
    print(f'[main] preserve_program={args.preserve_program}  '
          f'min_chord_tokens_before_eos={args.min_acc_tokens_before_eos}')
    model.cuda()
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    for song_idx, (mel_path, chord_path) in enumerate(pairs):
        sid = os.path.splitext(os.path.basename(mel_path))[0]
        song_dir = os.path.join(args.output_dir, sid)
        os.makedirs(song_dir, exist_ok=True)
        print(f'\n[{song_idx + 1}/{len(pairs)}] {sid}')

        for mode in args.modes:
            print(f'  mode={mode}')
            try:
                mel_frames, chord_frames = run_mode_for_song(
                    model, mode, mel_path, chord_path, args,
                )
                if mel_frames is None and chord_frames is None:
                    print('    skipped (missing input)')
                    continue
                # mel_only / chord_only: write only the generated side.
                write_mel = mode != 'chord_only'
                write_chord = mode != 'mel_only'
                save_path = os.path.join(song_dir, f'{mode}.mid')
                decode_m2c_frames(
                    mel_frames, chord_frames,
                    save_path=save_path,
                    tokenizer=model.tokenizer,
                    with_velocity=model.with_velocity,
                    write_mel=write_mel,
                    write_chord=write_chord,
                )
                print(f'    -> {save_path}')
            except Exception as e:
                print(f'    failed: {e!r}')


if __name__ == '__main__':
    main()
