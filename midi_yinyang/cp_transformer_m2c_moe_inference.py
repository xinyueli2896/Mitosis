"""Inference for the melody+chord MoE model (cp_transformer_m2c_moe.py).

Takes paired melody / chord midi prompts (either a single pair or two folders
paired by filename) and runs global_sampling — joint continuation of both
streams under the dual-mask + gated MoE attention. Output is one midi per
sample with separate MELODY (program=24, Guitar) and CHORD (program=0, Piano)
tracks; matches the local-sampler's hardcoded slot constraints.

Run from midi_yinyang/:

    # single pair
    python cp_transformer_m2c_moe_inference.py \
        --ckpt ckpt/<run>/last.ckpt \
        --melody POP909-Dataset/POP909-melody/001.mid \
        --chord  POP909-Dataset/POP909-chord/001.mid \
        --prompt-length 100 --gen-length 384 \
        --temperature 1.0 --n-samples 2

    # whole folders, paired by basename
    python cp_transformer_m2c_moe_inference.py \
        --ckpt ckpt/<run>/last.ckpt \
        --mel-folder POP909-Dataset/POP909-melody \
        --chord-folder POP909-Dataset/POP909-chord \
        --prompt-length 100 --gen-length 384
"""

import argparse
import os
from glob import glob

import torch
import pretty_midi

# This import also bootstraps the vendored transformers_roformer_moe by adding
# its src/ to sys.path before any HF transformers import resolves.
from cp_transformer_m2c_moe import (
    RoFormerSymbolicTransformer,
    CPTokenizer,
)
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES


def decode_m2c_frames(frames, save_path, tokenizer, with_velocity=False,
                      tempo=120.0):
    """Render the alternating mel/chord frames list to a single midi with two
    named tracks. frames[i]: [1, subseq] long-tensor; even i = mel, odd = chord."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4

    mel_inst = pretty_midi.Instrument(program=24, name='MELODY')
    chord_inst = pretty_midi.Instrument(program=0, name='CHORD')
    midi.instruments.extend([mel_inst, chord_inst])

    for frame_idx, frame in enumerate(frames):
        is_mel = (frame_idx % 2 == 0)
        timestep = frame_idx // 2
        start_time = timestep * time_step_length
        inst = mel_inst if is_mel else chord_inst

        content = frame.squeeze(0)
        for i in range(0, len(content), 2):
            a_token = int(content[i].item())
            if a_token == tokenizer.eos_token:
                break
            if a_token == tokenizer.pad_token:
                continue
            if i + 1 >= len(content):
                break
            b_token = int(content[i + 1].item())
            if b_token == tokenizer.pad_token:
                continue

            if with_velocity:
                pitch_duration = b_token - 16 * 128
            else:
                pitch_duration = b_token - 128
            pitch = pitch_duration % 128
            duration = pitch_duration // 128

            if pitch < 0 or pitch >= 128:
                continue
            if duration < 0 or duration >= len(DURATION_TEMPLATES):
                continue

            end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
            inst.notes.append(pretty_midi.Note(
                velocity=100, pitch=pitch,
                start=start_time, end=end_time,
            ))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def _load_raw(midi_path, max_polyphony):
    out = preprocess_midi(midi_path, max_polyphony, filter=False)
    if out is None:
        raise RuntimeError(f'preprocess_midi returned None for {midi_path}')
    raw, _ = out
    return raw  # [T, max_polyphony * 4]


def m2c_continuation(model, melody_path, chord_path,
                     prompt_length=100, gen_length=384,
                     temperature=1.0, n_samples=1,
                     max_polyphony=4, out_subdir=None):
    """Run joint continuation for one (melody.mid, chord.mid) pair."""
    mel_raw = _load_raw(melody_path, max_polyphony)
    chord_raw = _load_raw(chord_path, max_polyphony)
    common_len = min(mel_raw.shape[0], chord_raw.shape[0])
    mel_raw = mel_raw[:common_len].unsqueeze(0).cuda()
    chord_raw = chord_raw[:common_len].unsqueeze(0).cuda()

    pitch_shift = torch.zeros(1, dtype=torch.int8, device=mel_raw.device)
    mel_x, chord_x = model.preprocess(mel_raw, pitch_shift, y=chord_raw)
    # mel_x, chord_x: [1, T, subseq]

    prompt_T = min(prompt_length, mel_x.shape[1], chord_x.shape[1])
    mel_x = mel_x[:, :prompt_T]
    chord_x = chord_x[:, :prompt_T]

    # Interleave to [1, 2*prompt_T, subseq] = [mel_0, chord_0, mel_1, chord_1, ...]
    prompt = torch.stack([mel_x, chord_x], dim=2).reshape(1, 2 * prompt_T, -1)

    # Output dir
    tag = out_subdir or os.path.splitext(os.path.basename(melody_path))[0]
    save_name = getattr(model, 'save_name', 'm2c_moe_run')
    out_dir = os.path.join(f'temp/{save_name}', tag)

    # Render the prompt itself for sanity-listening.
    prompt_frames = [prompt[:, k, :] for k in range(prompt.shape[1])]
    decode_m2c_frames(
        prompt_frames,
        save_path=os.path.join(out_dir, 'prompt.mid'),
        tokenizer=model.tokenizer,
        with_velocity=model.with_velocity,
    )

    # max_seq_len in global_sampling counts FRAMES added past the prompt and
    # advances 2 per loop step (one mel + one chord), so it should be even
    # and equal to 2 * (gen_length - prompt_length).
    add_frames = max(0, 2 * (gen_length - prompt_T))

    with torch.no_grad():
        prompt_batch = prompt.repeat(n_samples, 1, 1)
        frames = model.global_sampling(
            prompt_batch,
            max_seq_len=add_frames,
            temperature=temperature,
        )
        # frames: list of [n_samples, subseq] tensors, alternating mel/chord,
        # total len = 2*prompt_T + add_frames.

    for i in range(n_samples):
        sample_frames = [f[i:i + 1, :] for f in frames]
        decode_m2c_frames(
            sample_frames,
            save_path=os.path.join(
                out_dir,
                f'continuation_{i}_temp{temperature}.mid',
            ),
            tokenizer=model.tokenizer,
            with_velocity=model.with_velocity,
        )


def _list_midis(folder):
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


def m2c_continuation_folder(model, mel_folder, chord_folder, **kwargs):
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
            m2c_continuation(model, mp, cp, out_subdir=sid, **kwargs)
        except Exception as e:
            print(f'  failed: {e!r}')


def load_model(ckpt_path, model_size='small', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None):
    """Construct the model with given hyperparameters then load the state dict
    from a Lightning .ckpt or the bare .pt produced at the end of training."""
    net = RoFormerSymbolicTransformer(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
    )
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Accept both lightning checkpoints ({'state_dict': ..., 'hyper_parameters': ...})
    # and the trainer.fit-final torch.save dict ({'state_dict': ..., 'hyper_parameters': ...}).
    if isinstance(ck, dict) and 'state_dict' in ck:
        state = ck['state_dict']
    else:
        state = ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing keys: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected keys: {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--melody', help='single melody midi (use with --chord)')
    src.add_argument('--mel-folder', help='folder of melody midis (use with --chord-folder)')
    p.add_argument('--chord', help='single chord midi')
    p.add_argument('--chord-folder', help='folder of chord midis')
    p.add_argument('--prompt-length', type=int, default=100,
                   help='timesteps of each prompt fed in as conditioning')
    p.add_argument('--gen-length', type=int, default=384,
                   help='total output length in timesteps (prompt included)')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-samples', type=int, default=1)
    p.add_argument('--max-polyphony', type=int, default=4)
    p.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
    args = p.parse_args()

    if args.melody and not args.chord:
        p.error('--melody requires --chord')
    if args.mel_folder and not args.chord_folder:
        p.error('--mel-folder requires --chord-folder')

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

    common_kwargs = dict(
        prompt_length=args.prompt_length,
        gen_length=args.gen_length,
        temperature=args.temperature,
        n_samples=args.n_samples,
        max_polyphony=args.max_polyphony,
    )
    if args.mel_folder:
        m2c_continuation_folder(model, args.mel_folder, args.chord_folder, **common_kwargs)
    else:
        m2c_continuation(model, args.melody, args.chord, **common_kwargs)


if __name__ == '__main__':
    main()
