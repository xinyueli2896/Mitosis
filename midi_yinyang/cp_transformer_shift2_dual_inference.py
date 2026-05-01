"""Inference / sampling for the dual-stream shift-by-2 cp_transformer.

Reads a melody.mid + chord.mid pair, uses the first prompt_length sub-beats of
each as a conditioning prompt, and generates the rest by joint parallel
decoding under DualRoFormerShift2.dual_global_sampling. Writes the result
(prompt + continuation) as a single midi with a MELODY track and a CHORD
track.

Run from the midi_yinyang/ directory:

    python cp_transformer_shift2_dual_inference.py \
        ckpt/<run>/<file>.ckpt prompt_melody.mid prompt_chord.mid \
        [prompt_length=100] [generation_length=384] [temperature=1.0] \
        [n_samples=1] [max_polyphony_m=4] [max_polyphony_c=8]
"""

import os
import sys

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
                      max_polyphony_m=4, max_polyphony_c=8):
    out_m = preprocess_midi(melody_path, max_polyphony_m, filter=False)
    out_c = preprocess_midi(chord_path, max_polyphony_c, filter=False)
    if out_m is None or out_c is None:
        raise RuntimeError('preprocess_midi failed on prompt input(s)')
    common_len = min(out_m[0].shape[0], out_c[0].shape[0])

    x_m = _load_prompt(model, melody_path, max_polyphony_m, common_len)
    x_c = _load_prompt(model, chord_path, max_polyphony_c, common_len)
    x_m = x_m[:, :prompt_length]
    x_c = x_c[:, :prompt_length]

    out_dir = f'temp/{model.save_name}'
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


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python cp_transformer_shift2_dual_inference.py <ckpt> '
              '<melody.mid> <chord.mid> [prompt_len=100] [gen_len=384] '
              '[temperature=1.0] [n_samples=1] [max_poly_m=4] [max_poly_c=8]')
        sys.exit(1)
    ckpt = sys.argv[1]
    melody_path = sys.argv[2]
    chord_path = sys.argv[3]
    prompt_length = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    generation_length = int(sys.argv[5]) if len(sys.argv) > 5 else 384
    temperature = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    n_samples = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    max_poly_m = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    max_poly_c = int(sys.argv[9]) if len(sys.argv) > 9 else 8

    model = DualRoFormerShift2.load_from_checkpoint(ckpt)
    model.save_name = os.path.basename(ckpt)
    model.cuda()
    model.eval()
    dual_continuation(
        model, melody_path, chord_path,
        prompt_length=prompt_length,
        generation_length=generation_length,
        temperature=temperature,
        n_samples=n_samples,
        max_polyphony_m=max_poly_m,
        max_polyphony_c=max_poly_c,
    )
