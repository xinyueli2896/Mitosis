"""MAESTRO v3 as a paired corpus: performance MIDI + EnCodec 32 kHz codes.

The audio side is EnCodec 32 kHz (MusicGen's codec: 50 Hz, 4 codebooks of
2048), kept as one continuous stream. The symbolic side is a note corpus
in milliseconds in the format of preprocess_time_grid.py, tokenized at
load time exactly as the time-grid cp transformer was pretrained
(FRAME_CODEC = K codec frames per symbolic frame, onset offsets, 20 ms
durations), so symbolic frame s covers codec frames K*s .. K*s+K-1. Both
sides start at t = 0 of the recording (the Disklavier MIDI is aligned to
about 3 ms). MAESTRO has no tempo, and nothing here uses one.

Stages (each run under SLURM, see slurm/maestro_*.sbatch):

  encode  WAV -> codes/<idx>.npy   int16 [F, 4], one file per piece.
          Resumable, and shardable over GPUs with --shard i --num_shards n.
  pack    per official split, the MIDI as a note corpus cut to the audio's
          length, and the codes:
            data/<name>_<split>.notes.bin / .pieces.npz / .txt   (preprocess_time_grid.Corpus)
            data/<name>_<split>.codes.npy                        int16 [sum F, 4]
            data/<name>_<split>.code_start.npy                   first code row per piece

MaestroPairedDataset yields aligned random crops of both halves.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import IterableDataset

from preprocess_time_grid import CODEC_MS, Corpus, notes_from_midi, notes_to_roll, write_corpus

CODEC_HZ = 50
CODEC_SR = 32000
CODEC_HOP = CODEC_SR // CODEC_HZ  # 640 samples per code frame
N_CODEBOOKS = 4
SPLITS = ('train', 'validation', 'test')


def read_index(root):
    """Rows of maestro-v3.0.0.csv, in csv order; idx is the row number."""
    with open(os.path.join(root, 'maestro-v3.0.0.csv'), newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r['idx'] = i
    return rows


# ---------------------------------------------------------------- audio

def load_wav_32k_mono(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, x = wavfile.read(path, mmap=True)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != CODEC_SR:
        g = math.gcd(sr, CODEC_SR)
        x = resample_poly(x, CODEC_SR // g, sr // g).astype(np.float32)
    return x


@torch.no_grad()
def encode_codes(model, wav, device, window_s=60.0, context_s=2.0, batch=8):
    """EnCodec codes [F, 4] for a whole recording, F = ceil(len / 640).

    The recording is cut into windows of window_s seconds, each encoded
    with context_s seconds of real audio on both sides (the encoder has
    an LSTM, so cutting cold would change codes near every cut); only
    the codes of the window itself are kept.
    """
    n_frames = math.ceil(len(wav) / CODEC_HOP)
    win = int(window_s * CODEC_HZ)
    ctx = int(context_s * CODEC_HZ)
    padded = np.zeros((n_frames + 2 * ctx) * CODEC_HOP + win * CODEC_HOP, dtype=np.float32)
    padded[ctx * CODEC_HOP: ctx * CODEC_HOP + len(wav)] = wav
    starts = list(range(0, n_frames, win))
    span = (win + 2 * ctx) * CODEC_HOP
    out = np.zeros((n_frames, N_CODEBOOKS), dtype=np.int16)
    for b in range(0, len(starts), batch):
        chunk = starts[b:b + batch]
        x = np.stack([padded[s * CODEC_HOP: s * CODEC_HOP + span] for s in chunk])
        x = torch.from_numpy(x).to(device)[:, None, :]
        codes = model.encode(x).audio_codes  # [1, B, 4, win + 2*ctx]
        codes = codes[0, :, :, ctx:ctx + win].transpose(1, 2).cpu().numpy()  # [B, win, 4]
        for s, c in zip(chunk, codes):
            e = min(s + win, n_frames)
            out[s:e] = c[:e - s]
    return out


def stage_encode(args):
    from transformers import EncodecModel
    rows = read_index(args.root)
    rows = [r for r in rows if r['idx'] % args.num_shards == args.shard]
    os.makedirs(args.codes_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EncodecModel.from_pretrained(args.codec).to(device).eval()
    assert model.config.sampling_rate == CODEC_SR, model.config.sampling_rate
    todo = [r for r in rows if not os.path.isfile(os.path.join(args.codes_dir, f"{r['idx']}.npy"))]
    print(f'shard {args.shard}/{args.num_shards}: {len(rows)} pieces, {len(rows) - len(todo)} already encoded', flush=True)
    for k, r in enumerate(todo):
        wav = load_wav_32k_mono(os.path.join(args.root, r['audio_filename']))
        codes = encode_codes(model, wav, device, batch=args.batch)
        dst = os.path.join(args.codes_dir, f"{r['idx']}.npy")
        np.save(dst + '.tmp.npy', codes)
        os.replace(dst + '.tmp.npy', dst)  # a killed job never leaves a short file behind
        print(f'[{k + 1}/{len(todo)}] {r["idx"]} {len(wav) / CODEC_SR:7.1f}s -> {codes.shape}', flush=True)


# ---------------------------------------------------------------- pack

def _midi_piece(path, n_codec_frames, pedal):
    import pretty_midi
    notes, has_pedal = notes_from_midi(pretty_midi.PrettyMIDI(path), pedal=pedal)
    length_ms = n_codec_frames * CODEC_MS
    notes = notes[notes['onset_ms'] < length_ms]  # the MIDI never outruns its audio
    return notes, length_ms, (0, 0), has_pedal, len(np.unique(notes['velocity']))


def stage_pack(args):
    from joblib import Parallel, delayed
    rows = read_index(args.root)
    missing = [r['idx'] for r in rows if not os.path.isfile(os.path.join(args.codes_dir, f"{r['idx']}.npy"))]
    if missing:
        sys.exit(f'{len(missing)} pieces have no codes in {args.codes_dir} (first: {missing[:5]}); run the encode stage first')
    os.makedirs('data', exist_ok=True)
    summary = {'codec': args.codec, 'pedal': args.pedal, 'splits': {}}
    for split in SPLITS:
        part = [r for r in rows if r['split'] == split]
        codes = [np.load(os.path.join(args.codes_dir, f"{r['idx']}.npy")) for r in part]
        pieces = Parallel(n_jobs=args.jobs)(delayed(_midi_piece)(os.path.join(args.root, r['midi_filename']),
                                                                 len(c), bool(args.pedal)) for r, c in zip(part, codes))
        prefix = f'data/{args.name}_{split}'
        meta = write_corpus(prefix, (('maestro', r['midi_filename'], p) for r, p in zip(part, pieces)))
        np.save(prefix + '.codes.npy', np.concatenate(codes))
        np.save(prefix + '.code_start.npy', np.cumsum([0] + [len(c) for c in codes[:-1]]).astype(np.int64))
        summary['splits'][split] = meta
        print(f"{split}: {meta['pieces']} pieces, {meta['hours']} h, {meta['notes']} notes", flush=True)
    with open(f'data/{args.name}.json', 'w') as f:
        json.dump(summary, f, indent=1)


# ---------------------------------------------------------------- loader

class MaestroPairedDataset(IterableDataset):
    """Aligned random crops of tokenized MIDI and codes.

    Yields (rolls [B, L, P*5] uint8, codes [B, L*K, 4] long): L symbolic
    frames of K codec frames each, tokenized as the time-grid cp
    transformer was pretrained (pass its frame_codec and max_polyphony),
    and the L*K codec frames under them, so symbolic frame s owns
    codes[:, K*s:K*s+K]. Crops start on a symbolic-frame boundary.
    Splits are MAESTRO's own; no pitch shift or time stretch, since the
    audio is fixed.
    """

    def __init__(self, prefix, target_length, batch_size, frame_codec=6, max_polyphony=16, random_order=True,
                 repeat=True):
        self.prefix, self.L, self.K, self.P = prefix, target_length, frame_codec, max_polyphony
        self.batch_size, self.random_order, self.repeat = batch_size, random_order, repeat
        z = np.load(prefix + '.pieces.npz')
        self.n_frames = z['length_ms'] // (CODEC_MS * frame_codec)  # whole symbolic frames per piece
        self.valid_indices = np.flatnonzero(self.n_frames >= target_length)
        self.corpus = self.codes = None
        print('Metadata for dataset', prefix, 'loaded. Valid pieces:', len(self.valid_indices), 'of',
              len(self.n_frames), f'({frame_codec * CODEC_MS} ms frames, {target_length} per crop)')

    def __iter__(self):
        if self.corpus is None:
            self.corpus = Corpus(self.prefix)
            self.codes = np.load(self.prefix + '.codes.npy', mmap_mode='r')
            self.code_start = np.load(self.prefix + '.code_start.npy')
        L, K = self.L, self.K
        rng = np.random.default_rng()
        while True:
            order = rng.permutation(len(self.valid_indices)) if self.random_order else np.arange(len(self.valid_indices))
            for i in range(0, len(order), self.batch_size):
                ids = self.valid_indices[order[i:i + self.batch_size]]
                rolls, codes = [], []
                for sid in ids:
                    s0 = int(rng.integers(0, self.n_frames[sid] - L + 1)) if self.random_order else 0
                    rolls.append(notes_to_roll(self.corpus.piece(sid), s0 * K * CODEC_MS, L, K, self.P)[0])
                    a = self.code_start[sid] + s0 * K
                    codes.append(np.asarray(self.codes[a:a + L * K]))
                yield torch.from_numpy(np.stack(rolls)), torch.from_numpy(np.stack(codes).astype(np.int64))
            if not self.repeat:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['encode', 'pack'])
    ap.add_argument('--root', required=True, help='unzipped maestro-v3.0.0 folder (holds maestro-v3.0.0.csv)')
    ap.add_argument('--codes_dir', default='data/maestro_encodec32k')
    ap.add_argument('--codec', default='facebook/encodec_32khz')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8, help='encode: windows per forward pass')
    ap.add_argument('--pedal', type=int, default=1, help='pack: extend offsets through the sustain pedal')
    ap.add_argument('--name', default='maestro', help='pack: data/<name>_<split>.*')
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()
    if args.stage == 'encode':
        stage_encode(args)
    else:
        stage_pack(args)


if __name__ == '__main__':
    main()
