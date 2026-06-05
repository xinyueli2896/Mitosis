"""Render dataloader batches back to .mid for sanity-check listening.

Two ways to use it:

1. Standalone -- inspect a pack without running training:

    python dump_train_samples.py \\
        --path_to_dataset data/la_chord_cp16_v2.pt \\
        --out_dir temp/sanity \\
        --n_samples 4 --max_polyphony 16

2. From a Lightning training script, via DumpInputSamplesCallback. The
   callback grabs ONE batch at on_train_start (and optionally every N
   epochs), decodes the first n_samples (mel, chord) pairs to .mid in
   out_dir, and continues. Use it to confirm the model is seeing actual
   music, not garbage rows.
"""

import argparse
import os

import pretty_midi
import torch

from preprocess_large_midi_dataset import DURATION_TEMPLATES


def render_quads_to_midi(stream_rows_list, save_path, max_polyphony,
                          tempo=120.0, beat_div=4):
    """Decode one or more `[T, max_polyphony * 4]` uint8 streams to a .mid.

    Each row contains (program, pitch, duration_idx, velocity) quads packed
    side by side. program == 254 ends that timestep (per-step EOS); 255 is
    the empty-slot fill; 127 means drum.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    step = 60.0 / tempo / beat_div
    instrument_map = {}

    for rows in stream_rows_list:
        if rows is None or rows.numel() == 0:
            continue
        arr = rows.view(rows.shape[0], max_polyphony, 4).cpu().numpy()
        for t in range(arr.shape[0]):
            for k in range(max_polyphony):
                program = int(arr[t, k, 0])
                if program == 254 or program == 255:
                    break
                pitch = int(arr[t, k, 1])
                dur_idx = int(arr[t, k, 2])
                velocity = int(arr[t, k, 3]) or 100
                if dur_idx < 0 or dur_idx >= len(DURATION_TEMPLATES):
                    continue
                if pitch < 0 or pitch >= 128:
                    continue
                start = t * step
                end = start + float(DURATION_TEMPLATES[dur_idx]) * step
                if program == 127:
                    key = ('drum',)
                    if key not in instrument_map:
                        instrument_map[key] = pretty_midi.Instrument(0, is_drum=True)
                        midi.instruments.append(instrument_map[key])
                else:
                    key = ('inst', program)
                    if key not in instrument_map:
                        instrument_map[key] = pretty_midi.Instrument(program)
                        midi.instruments.append(instrument_map[key])
                instrument_map[key].notes.append(
                    pretty_midi.Note(velocity=velocity, pitch=pitch,
                                     start=start, end=end))

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    midi.write(save_path)


def dump_batch(batch, out_dir, n_samples, max_polyphony, tag='sample',
               tempo=120.0, beat_div=4):
    """Decode the first n_samples in `batch` (yielded as (x_mel, x_chord, ps))
    to <out_dir>/<tag>_<i>.mid (combined mel+chord), plus <tag>_<i>_mel.mid
    and <tag>_<i>_chord.mid for the per-stream versions.
    """
    x_mel, x_chord, _ = batch
    n_samples = min(n_samples, x_mel.shape[0])
    for i in range(n_samples):
        render_quads_to_midi(
            [x_mel[i], x_chord[i]],
            os.path.join(out_dir, f'{tag}_{i}.mid'),
            max_polyphony=max_polyphony, tempo=tempo, beat_div=beat_div,
        )
        render_quads_to_midi(
            [x_mel[i]],
            os.path.join(out_dir, f'{tag}_{i}_mel.mid'),
            max_polyphony=max_polyphony, tempo=tempo, beat_div=beat_div,
        )
        render_quads_to_midi(
            [x_chord[i]],
            os.path.join(out_dir, f'{tag}_{i}_chord.mid'),
            max_polyphony=max_polyphony, tempo=tempo, beat_div=beat_div,
        )
    print(f'[dump] wrote {n_samples} mel+chord pairs to {out_dir} (tag={tag})')


class DumpInputSamplesCallback:
    """Lightning callback. Lazy-imports lightning so importing this module
    doesn't require it.
    """

    def __init__(self, out_dir, n_samples=4, max_polyphony=16,
                 every_n_epochs=None, tempo=120.0, beat_div=4):
        try:
            from lightning.pytorch.callbacks import Callback as _Callback
        except ImportError:
            from pytorch_lightning.callbacks import Callback as _Callback
        self._Callback = _Callback
        self.out_dir = out_dir
        self.n_samples = n_samples
        self.max_polyphony = max_polyphony
        self.every_n_epochs = every_n_epochs
        self.tempo = tempo
        self.beat_div = beat_div
        # Synthesize a Callback subclass on the fly so we don't pollute the
        # module namespace with a non-importable class for users without L.
        outer = self

        class _Cb(self._Callback):
            def on_train_start(self, trainer, pl_module):
                try:
                    batch = next(iter(trainer.train_dataloader))
                except Exception as e:
                    print(f'[dump_input_samples] failed to fetch batch: {e}')
                    return
                dump_batch(batch, outer.out_dir, outer.n_samples,
                           outer.max_polyphony, tag='train_start',
                           tempo=outer.tempo, beat_div=outer.beat_div)

            def on_train_epoch_start(self, trainer, pl_module):
                if outer.every_n_epochs is None:
                    return
                if trainer.current_epoch == 0:
                    return  # already dumped at on_train_start
                if trainer.current_epoch % outer.every_n_epochs != 0:
                    return
                try:
                    batch = next(iter(trainer.train_dataloader))
                except Exception as e:
                    print(f'[dump_input_samples] failed to fetch batch: {e}')
                    return
                dump_batch(batch, outer.out_dir, outer.n_samples,
                           outer.max_polyphony,
                           tag=f'epoch{trainer.current_epoch:03d}',
                           tempo=outer.tempo, beat_div=outer.beat_div)

        self._cb = _Cb()

    def as_callback(self):
        return self._cb


def main():
    """Standalone CLI: build a FramedDataset, pull a couple of batches, decode."""
    from cp_transformer_m2c_moe import FramedDataset
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument('--path_to_dataset', required=True,
                    help='Path to the chord-stream .pt; melody-stream is '
                         'auto-resolved via chord->melody filename swap.')
    ap.add_argument('--out_dir', default='temp/sanity')
    ap.add_argument('--n_samples', type=int, default=4)
    ap.add_argument('--n_batches', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--target_length', type=int, default=512)
    ap.add_argument('--max_polyphony', type=int, default=16)
    ap.add_argument('--split', default='train')
    ap.add_argument('--tempo', type=float, default=120.0)
    ap.add_argument('--beat_div', type=int, default=4)
    args = ap.parse_args()

    ds = FramedDataset(args.path_to_dataset, args.target_length,
                       args.batch_size, split=args.split)
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    it = iter(loader)
    for b in range(args.n_batches):
        batch = next(it)
        dump_batch(batch, args.out_dir, args.n_samples, args.max_polyphony,
                   tag=f'batch{b}', tempo=args.tempo, beat_div=args.beat_div)


if __name__ == '__main__':
    main()
