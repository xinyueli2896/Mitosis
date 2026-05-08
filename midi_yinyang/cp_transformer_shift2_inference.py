"""Inference / sampling for the single-modality shift-by-2 cp_transformer.

Mirrors cp_transformer_inference.py but loads RoFormerSymbolicTransformer from
cp_transformer_shift2 so global_sampling uses the pair-causal local decoder
(parallel decoding of (instrument, pitch+duration) per note).

Run from midi_yinyang/:

    python cp_transformer_shift2_inference.py <ckpt> <prompt.mid> \
        [prompt_length=75] [generation_length=384] [temperature=1.0] \
        [n_samples=2] [max_polyphony=16]
"""

import os
import re
import sys

import torch

from cp_transformer_shift2 import RoFormerSymbolicTransformer
from cp_transformer_inference import decode_output
from preprocess_large_midi_dataset import preprocess_midi


def continuation(model, midi_path, prompt_length=75, generation_length=384,
                 temperature=1.0, n_samples=1, max_polyphony=16):
    out = preprocess_midi(midi_path, max_polyphony, filter=False)
    if out is None:
        raise RuntimeError(f'preprocess_midi returned None for {midi_path}')
    raw, _ = out
    raw = raw.unsqueeze(0).cuda()
    pitch_shift = torch.zeros(1, dtype=torch.int8, device=raw.device)
    x = model.preprocess(raw, pitch_shift)
    x = x[:, :prompt_length]

    out_dir = f'temp/{model.save_name}'
    decode_output(
        [x[:, i, :] for i in range(x.shape[1])],
        save_path=os.path.join(out_dir, f'{os.path.basename(midi_path)}_prompt.mid'),
    )

    with torch.no_grad():
        x_batch = x.repeat(n_samples, 1, 1)
        outputs = model.global_sampling(
            x_batch, max_seq_len=generation_length, temperature=temperature,
        )

    for i in range(n_samples):
        steps = [outputs[t][i:i + 1, :] for t in range(len(outputs))]
        decode_output(
            steps,
            save_path=os.path.join(
                out_dir,
                f'{os.path.basename(midi_path)}_temp{temperature}_continuation_{i}.mid',
            ),
        )


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python cp_transformer_shift2_inference.py <ckpt> '
              '<prompt.mid> [prompt_len=75] [gen_len=384] [temperature=1.0] '
              '[n_samples=2] [max_polyphony=16]')
        sys.exit(1)
    ckpt = sys.argv[1]
    midi_path = sys.argv[2]
    prompt_length = int(sys.argv[3]) if len(sys.argv) > 3 else 75
    generation_length = int(sys.argv[4]) if len(sys.argv) > 4 else 384
    temperature = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    n_samples = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    max_polyphony = int(sys.argv[7]) if len(sys.argv) > 7 else 16

    # Constructor args weren't saved with the checkpoint, so we have to pass
    # the right size/with_velocity manually. Auto-detect size from the
    # filename pattern "_sizeN_" when present.
    size = 1
    m = re.search(r'_size(\d+)_', os.path.basename(ckpt))
    if m:
        size = int(m.group(1))
        print(f'Auto-detected size={size} from checkpoint filename.')
    with_velocity = 'vel' in os.path.basename(ckpt).split('_size')[0]
    model = RoFormerSymbolicTransformer.load_from_checkpoint(
        ckpt, size=size, with_velocity=with_velocity,
    )
    model.save_name = os.path.basename(ckpt)
    model.cuda()
    model.eval()
    continuation(
        model, midi_path,
        prompt_length=prompt_length,
        generation_length=generation_length,
        temperature=temperature,
        n_samples=n_samples,
        max_polyphony=max_polyphony,
    )
