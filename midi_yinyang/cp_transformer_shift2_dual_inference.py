"""Inference / sampling for the dual-stream shift-by-2 cp_transformer.

Reads a melody.mid + chord.mid pair (or two folders thereof, paired by file
name), uses the first prompt_length sub-beats of each as a conditioning
prompt, and generates the rest by joint parallel decoding under
DualRoFormerShift2.dual_global_sampling. Writes each result (prompt +
continuation) as a single midi with a MELODY track and a CHORD track.

Run from the midi_yinyang/ directory with either argparse-style flags:

    # single pair
    python cp_transformer_shift2_dual_inference.py \
        --ckpt ckpt/<run>/<file>.ckpt \
        --melody POP909-Dataset/POP909-melody/001.mid \
        --chord  POP909-Dataset/POP909-chord/001.mid \
        --prompt-length 100 --gen-length 384 --temperature 1.0 --n-samples 2

    # whole folder (paired by file name)
    python cp_transformer_shift2_dual_inference.py \
        --ckpt ckpt/<run>/<file>.ckpt \
        --mel-folder POP909-Dataset/POP909-melody \
        --chord-folder POP909-Dataset/POP909-chord
"""

import argparse
import os
from glob import glob

import torch
import pretty_midi

from cp_transformer_shift2_dual import DualRoFormerShift2
from cp_transformer_inference import decode_output
from preprocess_large_midi_dataset import preprocess_midi


def decode_output_dual(melody_steps, chord_steps, save_path=None,
                       tempo=120.0, ratio=1.0, velocity=100):
    """Render a melody-stream + chord-stream sample as a single midi.

    Each stream is rendered with the existing decode_output (one Instrument per
    program), then merged into one PrettyMIDI; instruments are renamed
    "MELODY" / "CHORD" for clarity in DAWs."""
    midi_m = decode_output(melody_steps, save_path=None, tempo=tempo,
                           ratio=ratio, velocity=velocity)
    midi_c = decode_output(chord_steps, save_path=None, tempo=tempo,
                           ratio=ratio, velocity=velocity)
    combined = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for ins in midi_m.instruments:
        ins.name = 'MELODY'
        combined.instruments.append(ins)
    for ins in midi_c.instruments:
        ins.name = 'CHORD'
        combined.instruments.append(ins)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined.write(save_path)
    return combined


def _load_prompt(model, midi_path, max_polyphony, common_len):
    out = preprocess_midi(midi_path, max_polyphony, filter=False)
    if out is None:
        raise RuntimeError(f'preprocess_midi returned None for {midi_path}')
    raw, _ = out
    raw = raw[:common_len].unsqueeze(0).cuda()
    pitch_shift = torch.zeros(1, dtype=torch.int8, device=raw.device)
    return model.preprocess(raw, pitch_shift)  # [1, T, subseq]


def dual_continuation(model, melody_path, chord_path, prompt_length=100,
                      generation_length=384, temperature=1.0, n_samples=1,
                      max_polyphony_m=4, max_polyphony_c=8, out_subdir=None):
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
    decode_output_dual(
        [x_m[:, i, :] for i in range(x_m.shape[1])],
        [x_c[:, i, :] for i in range(x_c.shape[1])],
        save_path=os.path.join(out_dir, 'prompt.mid'),
    )

    with torch.no_grad():
        x_m_batch = x_m.repeat(n_samples, 1, 1)
        x_c_batch = x_c.repeat(n_samples, 1, 1)
        y_m, y_c = model.dual_global_sampling(
            x_m_batch, x_c_batch,
            max_seq_len=generation_length, temperature=temperature,
        )

    for i in range(n_samples):
        m_i = [y_m[t][i:i + 1, :] for t in range(len(y_m))]
        c_i = [y_c[t][i:i + 1, :] for t in range(len(y_c))]
        decode_output_dual(
            m_i, c_i,
            save_path=os.path.join(out_dir, f'continuation_{i}_temp{temperature}.mid'),
        )


def _list_midis(folder):
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


def dual_continuation_folder(model, mel_folder, chord_folder, **kwargs):
    """Run dual_continuation on every paired (melody, chord) midi found in the
    two folders. Pairing is by basename: a/001.mid <-> b/001.mid."""
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
            dual_continuation(model, mp, cp, out_subdir=sid, **kwargs)
        except Exception as e:
            print(f'  failed: {e!r}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--melody', help='single melody midi (use with --chord)')
    src.add_argument('--mel-folder', help='folder of melody midis (use with --chord-folder)')
    p.add_argument('--chord', help='single chord midi')
    p.add_argument('--chord-folder', help='folder of chord midis')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-samples', type=int, default=1)
    p.add_argument('--max-poly-m', type=int, default=4)
    p.add_argument('--max-poly-c', type=int, default=8)
    args = p.parse_args()

    if args.melody and not args.chord:
        p.error('--melody requires --chord')
    if args.mel_folder and not args.chord_folder:
        p.error('--mel-folder requires --chord-folder')

    model = DualRoFormerShift2.load_from_checkpoint(args.ckpt)
    model.save_name = os.path.basename(args.ckpt)
    model.cuda()
    model.eval()

    common_kwargs = dict(
        prompt_length=args.prompt_length,
        generation_length=args.gen_length,
        temperature=args.temperature,
        n_samples=args.n_samples,
        max_polyphony_m=args.max_poly_m,
        max_polyphony_c=args.max_poly_c,
    )
    if args.mel_folder:
        dual_continuation_folder(model, args.mel_folder, args.chord_folder, **common_kwargs)
    else:
        dual_continuation(model, args.melody, args.chord, **common_kwargs)


if __name__ == '__main__':
    main()
