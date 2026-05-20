"""Inference for the MaskGIT-style mask-predict joint model.

Reads a paired melody / chord prompt (single pair or two folders, same
flag layout as cp_transformer_shift2_dual_inference.py) and runs
MaskPredictDualRoFormer.mask_predict_sampling: block-by-block generation
with K within-block refinement passes. Output is a single midi with
separate MELODY and CHORD tracks.

Run from midi_yinyang/:

    # single pair
    python cp_transformer_shift2_mask_inference.py \
        --ckpt ckpt/<run>/<file>.ckpt \
        --melody POP909-Dataset/POP909-melody/001.mid \
        --chord  POP909-Dataset/POP909-chord/001.mid \
        --prompt-length 100 --gen-length 384 \
        --n-refine-steps 2 --temperature 1.0 --n-samples 2

    # whole folder
    python cp_transformer_shift2_mask_inference.py \
        --ckpt ckpt/<run>/<file>.ckpt \
        --mel-folder POP909-Dataset/POP909-melody \
        --chord-folder POP909-Dataset/POP909-chord
"""

import argparse
import os
import re
from glob import glob

import torch
import pretty_midi

from cp_transformer_shift2_mask import MaskPredictDualRoFormer
from cp_transformer_shift2_dual_inference import decode_output_dual
from preprocess_large_midi_dataset import preprocess_midi


def _load_prompt(model, midi_path, max_polyphony, common_len):
    out = preprocess_midi(midi_path, max_polyphony, filter=False)
    if out is None:
        raise RuntimeError(f'preprocess_midi returned None for {midi_path}')
    raw, _ = out
    raw = raw[:common_len].unsqueeze(0).cuda()
    pitch_shift = torch.zeros(1, dtype=torch.int8, device=raw.device)
    return model.preprocess(raw, pitch_shift)


def mask_predict_continuation(model, melody_path, chord_path,
                              prompt_length=100, generation_length=384,
                              n_refine_steps=2, temperature=1.0, n_samples=1,
                              max_polyphony_m=4, max_polyphony_c=8,
                              out_subdir=None):
    out_m = preprocess_midi(melody_path, max_polyphony_m, filter=False)
    out_c = preprocess_midi(chord_path, max_polyphony_c, filter=False)
    if out_m is None or out_c is None:
        raise RuntimeError(f'preprocess_midi failed on {melody_path} / {chord_path}')
    common_len = min(out_m[0].shape[0], out_c[0].shape[0])

    x_m = _load_prompt(model, melody_path, max_polyphony_m, common_len)
    x_c = _load_prompt(model, chord_path, max_polyphony_c, common_len)
    x_m = x_m[:, :prompt_length]
    x_c = x_c[:, :prompt_length]

    tag = out_subdir or os.path.splitext(os.path.basename(melody_path))[0]
    out_dir = os.path.join(f'temp/{model.save_name}', tag)
    with_velocity = getattr(model, 'with_velocity', False)
    decode_output_dual(
        [x_m[:, i, :] for i in range(x_m.shape[1])],
        [x_c[:, i, :] for i in range(x_c.shape[1])],
        save_path=os.path.join(out_dir, 'prompt.mid'),
        with_velocity=with_velocity,
    )

    with torch.no_grad():
        x_m_batch = x_m.repeat(n_samples, 1, 1)
        x_c_batch = x_c.repeat(n_samples, 1, 1)
        y_m, y_c = model.mask_predict_sampling(
            x_m_batch, x_c_batch,
            max_seq_len=generation_length,
            n_refine_steps=n_refine_steps,
            temperature=temperature,
        )

    for i in range(n_samples):
        m_i = [y_m[t][i:i + 1, :] for t in range(len(y_m))]
        c_i = [y_c[t][i:i + 1, :] for t in range(len(y_c))]
        decode_output_dual(
            m_i, c_i,
            save_path=os.path.join(
                out_dir,
                f'continuation_{i}_temp{temperature}_K{n_refine_steps}.mid',
            ),
            with_velocity=with_velocity,
        )


def _list_midis(folder):
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


def mask_predict_continuation_folder(model, mel_folder, chord_folder, **kwargs):
    mel_files = _list_midis(mel_folder)
    chord_index = {os.path.basename(p): p for p in _list_midis(chord_folder)}
    pairs = [(mp, chord_index[os.path.basename(mp)]) for mp in mel_files
             if os.path.basename(mp) in chord_index]
    skipped = len(mel_files) - len(pairs)
    print(f'Found {len(pairs)} paired midis (skipped {skipped} unmatched).')
    for i, (mp, cp) in enumerate(pairs):
        sid = os.path.splitext(os.path.basename(mp))[0]
        print(f'[{i + 1}/{len(pairs)}] {sid}')
        try:
            mask_predict_continuation(model, mp, cp, out_subdir=sid, **kwargs)
        except Exception as e:
            print(f'  failed: {e!r}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--melody', help='single melody midi (use with --chord)')
    src.add_argument('--mel-folder', help='folder of melody midis')
    p.add_argument('--chord', help='single chord midi')
    p.add_argument('--chord-folder', help='folder of chord midis')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-refine-steps', type=int, default=2,
                   help='Within-block refinement steps. 1 = pure parallel '
                        '(both slots sampled in one pass); 2 = MaskGIT/Gibbs '
                        '(more-confident slot first, then the other).')
    p.add_argument('--n-samples', type=int, default=1)
    p.add_argument('--max-poly-m', type=int, default=4)
    p.add_argument('--max-poly-c', type=int, default=8)
    p.add_argument('--size', type=int, default=0)
    p.add_argument('--with-velocity', action='store_true')
    args = p.parse_args()

    if args.melody and not args.chord:
        p.error('--melody requires --chord')
    if args.mel_folder and not args.chord_folder:
        p.error('--mel-folder requires --chord-folder')

    m = re.search(r'_size(\d+)_', os.path.basename(args.ckpt))
    if m:
        detected = int(m.group(1))
        if detected != args.size:
            print(f'Auto-detected size={detected} from checkpoint filename '
                  f'(overriding --size={args.size}).')
            args.size = detected

    model = MaskPredictDualRoFormer.load_from_checkpoint(
        args.ckpt, size=args.size, with_velocity=args.with_velocity,
    )
    model.save_name = os.path.basename(args.ckpt)
    model.cuda()
    model.eval()

    common_kwargs = dict(
        prompt_length=args.prompt_length,
        generation_length=args.gen_length,
        n_refine_steps=args.n_refine_steps,
        temperature=args.temperature,
        n_samples=args.n_samples,
        max_polyphony_m=args.max_poly_m,
        max_polyphony_c=args.max_poly_c,
    )
    if args.mel_folder:
        mask_predict_continuation_folder(
            model, args.mel_folder, args.chord_folder, **common_kwargs,
        )
    else:
        mask_predict_continuation(model, args.melody, args.chord, **common_kwargs)


if __name__ == '__main__':
    main()
