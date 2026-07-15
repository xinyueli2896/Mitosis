"""Single-stream co-generation BASELINE on the pretrained CP transformer.

Runs the base (non-m2c) pretrained checkpoint on drum+nondrum material
treated as ONE merged stream -- no modality separation, no per-modality
projections, no interleaving. Each prompt MIDI is tokenized with
ins_ids='all' (drums and pitched instruments share the same
max_polyphony slots per frame) and continued with plain frame-level AR
via global_sampling.

This is the natural lower anchor for the co-generation comparison:
  baseline (this)   : P(frame_t | merged past)          -- 1 stream
  A.1 DuetAttn      : interleaved 2-stream joint AR
  A.3 BlockDiffusion: 2-stream + iterative same-step refinement

Semantics match the m2c combined scripts: --gen-length is the TOTAL
number of frames in the output (prompt included), so outputs are
directly comparable to the co-mode files those scripts produce.

Run from midi_yinyang/:

    python cp_transformer_baseline_singlestream.py \\
        --ckpt ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt \\
        --midi-folder input/rwc_test_prompts \\
        --output-dir temp/baseline_singlestream_rwc \\
        --prompt-length 64 --gen-length 384 \\
        --temperature 1.0 --max-polyphony 16
"""

import argparse
import os

import torch
import pretty_midi

from cp_transformer import RoFormerSymbolicTransformer
from cp_transformer_inference import decode_output
from preprocess_large_midi_dataset import preprocess_midi


def get_input_tempo(midi_path, default=120.0):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        _, tempi = pm.get_tempo_changes()
        if len(tempi) > 0 and tempi[0] > 0:
            return float(tempi[0])
    except Exception as e:
        print(f'[tempo] failed to read {midi_path}: {e!r}')
    return default


def load_prompt(model, midi_path, max_polyphony, prompt_length):
    """Merged-stream tokens [1, prompt_length, subseq] on the model device."""
    out = preprocess_midi(midi_path, max_polyphony, ins_ids='all', filter=False)
    if out is None:
        raise RuntimeError(f'preprocess_midi returned None for {midi_path}')
    raw = out[0]
    device = next(model.parameters()).device
    x = raw.unsqueeze(0).to(device)
    x = model.preprocess(
        x, pitch_shift=torch.zeros(1, dtype=torch.int8, device=device),
    )
    return x[:, :prompt_length]


def _list_midis(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def run_folder(model, args):
    midi_files = _list_midis(args.midi_folder)
    if args.max_songs is not None:
        midi_files = midi_files[:args.max_songs]
    print(f'[infer] {len(midi_files)} prompt MIDIs (merged single stream)')
    os.makedirs(args.output_dir, exist_ok=True)

    for i, path in enumerate(midi_files):
        sid = os.path.splitext(os.path.basename(path))[0]
        print(f'[{i + 1}/{len(midi_files)}] {sid}')
        try:
            prompt = load_prompt(model, path, args.max_polyphony,
                                  args.prompt_length)
            if prompt.shape[1] < args.prompt_length:
                print(f'  [skip] only {prompt.shape[1]} frames '
                      f'(< prompt_length {args.prompt_length})')
                continue
            prompt = prompt.repeat(args.n_samples, 1, 1)
            with torch.no_grad():
                frames = model.global_sampling(
                    prompt,
                    max_seq_len=args.gen_length,
                    temperature=args.temperature,
                )
            tempo = get_input_tempo(path)
            print(f'  [tempo] {tempo:.2f} BPM   frames={len(frames)}')
            out_dir = os.path.join(args.output_dir, sid)
            os.makedirs(out_dir, exist_ok=True)
            for s in range(args.n_samples):
                sample = [f[s:s + 1, :] for f in frames]
                out_path = os.path.join(
                    out_dir,
                    f'single_stream_temp{args.temperature}'
                    + (f'_{s}' if args.n_samples > 1 else '')
                    + '.mid',
                )
                decode_output(sample, save_path=out_path, tempo=tempo,
                              with_velocity=model.with_velocity)
                print(f'  wrote {out_path}')
        except Exception as e:
            print(f'  failed: {e!r}')


def main():
    p = argparse.ArgumentParser(
        description='Single-stream co-generation baseline on the '
                    'pretrained CP transformer (drum+nondrum merged).',
    )
    p.add_argument('--ckpt', required=True,
                   help='Path to the pretrained single-stream ckpt (a '
                        'Lightning ckpt with hparams, e.g. '
                        'cp_transformer_v0.42_..._fin.ckpt).')
    p.add_argument('--midi-folder', required=True,
                   help='Folder of UNSPLIT prompt MIDIs (drums + pitched '
                        'in one file), e.g. input/rwc_test_prompts.')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--prompt-length', type=int, default=64)
    p.add_argument('--gen-length', type=int, default=384,
                   help='TOTAL output frames incl. prompt (matches the '
                        'm2c combined scripts).')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-polyphony', type=int, default=16)
    p.add_argument('--n-samples', type=int, default=1)
    p.add_argument('--max-songs', type=int, default=None)
    args = p.parse_args()

    model = RoFormerSymbolicTransformer.load_from_checkpoint(args.ckpt)
    model.save_name = os.path.basename(args.ckpt)
    model = model.cuda().eval()

    run_folder(model, args)


if __name__ == '__main__':
    main()
