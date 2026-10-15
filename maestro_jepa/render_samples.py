#!/usr/bin/env python3
"""Render listening samples from the ZzWaang/music-jepa-maestro HDF5 files.

The dataset stores no waveforms. maestro_audio.h5 holds log-mel spectrograms
(229 mel bins in dB, floored at -80, plus 20 MFCC rows; 16 kHz, hop 160 =
100 frames/s) and maestro_midi.h5 holds aligned 88-key velocity piano rolls and
sustain pedal. So for each example this script writes:

  <id>_mel_griffinlim.wav  audio reconstructed from the log-mel (Griffin-Lim;
                           phase is estimated, so expect a "phasey" sound)
  <id>.mid                 MIDI rebuilt from the piano roll + pedal (CC64)
  <id>_midi_synth.wav      that MIDI rendered (FluidSynth if --sf2 is given,
                           otherwise pretty_midi's built-in sine synth)
  <id>_overview.png        mel spectrogram over piano roll, same time axis

Usage:
  python render_samples.py inspect   --data-dir data/maestro
  python render_samples.py calibrate --data-dir data/maestro
  python render_samples.py render    --data-dir data/maestro --out samples --n 5
"""
import argparse
import json
import os

import h5py
import numpy as np

SR = 16000
HOP = 160
FPS = SR // HOP  # 100
N_MELS = 229
LOWEST_MIDI = 21  # A0


def open_files(data_dir):
    audio = h5py.File(os.path.join(data_dir, "maestro_audio.h5"), "r")
    midi = h5py.File(os.path.join(data_dir, "maestro_midi.h5"), "r")
    return audio, midi


def load_examples(data_dir, split):
    path = os.path.join(data_dir, "splits", f"{split}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    audio, _ = open_files(data_dir)
    return [{"example_index": k} for k in sorted(audio["features"].keys(), key=int)]


def describe(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")


def cmd_inspect(args):
    audio, midi = open_files(args.data_dir)
    for label, f in [("maestro_audio.h5", audio), ("maestro_midi.h5", midi)]:
        print(f"== {label}: top-level keys {list(f.keys())}")
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Group):
                keys = list(obj.keys())
                print(f"  {key}/ : {len(keys)} entries, e.g. {keys[:5]}")
                describe(f"{key}/{keys[0]}", obj[keys[0]])
            else:
                describe(key, obj)
                if obj.dtype.names:
                    print(f"    fields: {obj.dtype.names}; first row: {obj[0]}")
    k = next(iter(audio["features"].keys()))
    feat = audio["features"][k][:, : 30 * FPS]
    mel, mfcc = feat[:N_MELS], feat[N_MELS:]
    print(f"== example {k}, first 30 s")
    print(f"  mel  min/mean/max: {mel.min():.1f} / {mel.mean():.1f} / {mel.max():.1f}")
    print(f"  mfcc min/max: {mfcc.min():.1f} / {mfcc.max():.1f}")
    pr = midi["midi_values"][k][:, : 30 * FPS]
    ped = midi["pedal"][k][:, : 30 * FPS]
    print(f"  piano roll frames {midi['midi_values'][k].shape[1]} vs audio frames "
          f"{audio['features'][k].shape[1]}; active keys {int((pr > 0).any(1).sum())}, "
          f"max vel {pr.max()}, pedal range {ped.min()}..{ped.max()}")
    for split in ("train", "val", "test"):
        p = os.path.join(args.data_dir, "splits", f"{split}.json")
        if os.path.exists(p):
            ex = json.load(open(p))
            print(f"== splits/{split}.json: {len(ex)} examples, first: {ex[0]}")


def mel_centers(fmin, fmax, htk):
    import librosa
    return librosa.mel_frequencies(N_MELS + 2, fmin=fmin, fmax=fmax, htk=htk)


def predicted_mel(pr, fmin, fmax, htk, n_harm=6):
    """Rough expected mel energy from the piano roll: harmonics through triangular filters."""
    edges = mel_centers(fmin, fmax, htk)
    lo, c, hi = edges[:-2], edges[1:-1], edges[2:]
    f0 = 440.0 * 2 ** ((np.arange(88) + LOWEST_MIDI - 69) / 12)
    resp = np.zeros((N_MELS, 88))
    for h in range(1, n_harm + 1):
        f = f0 * h
        tri = np.maximum(0, np.minimum((f[None] - lo[:, None]) / (c - lo)[:, None],
                                       (hi[:, None] - f[None]) / (hi - c)[:, None]))
        resp += tri / h
    return resp @ (pr > 0).astype(float)


def cmd_calibrate(args):
    """Guess the mel filterbank (fmin/fmax/htk) by matching the mel to the aligned piano roll."""
    audio, midi = open_files(args.data_dir)
    keys = [str(e["example_index"]) for e in load_examples(args.data_dir, args.split)[: args.n]]
    results = []
    for fmax in (8000.0, 7600.0, 6000.0):
        for fmin in (0.0, 20.0, 27.5, 30.0, 40.0):
            for htk in (False, True):
                scores = []
                for k in keys:
                    mel = audio["features"][k][:N_MELS, : 60 * FPS]
                    pr = midi["midi_values"][k][:, : 60 * FPS]
                    pred = predicted_mel(pr, fmin, fmax, htk)
                    a = mel - mel.mean(1, keepdims=True)
                    b = pred - pred.mean(1, keepdims=True)
                    scores.append((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                results.append((float(np.mean(scores)), fmin, fmax, htk))
    results.sort(reverse=True)
    print("score   fmin    fmax   htk")
    for s, fmin, fmax, htk in results[:8]:
        print(f"{s:.4f}  {fmin:5.1f}  {fmax:6.0f}  {htk}")
    s, fmin, fmax, htk = results[0]
    print(f"\nbest: --fmin {fmin} --fmax {fmax}" + (" --htk" if htk else ""))


def mel_to_wav(mel_db, args):
    import librosa
    power = librosa.db_to_power(mel_db)
    y = librosa.feature.inverse.mel_to_audio(
        power, sr=SR, n_fft=args.n_fft, hop_length=HOP, win_length=args.win_length,
        fmin=args.fmin, fmax=args.fmax, htk=args.htk, power=2.0, n_iter=args.gl_iters)
    return 0.9 * y / (np.abs(y).max() + 1e-9)


def roll_to_midi(pr, pedal, start_frame=0):
    """Velocity piano roll -> PrettyMIDI. A new note starts when a key goes on or its velocity changes."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="piano")
    t = lambda i: i / FPS
    padded = np.pad(pr.astype(int), ((0, 0), (1, 1)))
    for key in range(pr.shape[0]):
        row = padded[key]
        change = np.flatnonzero(row[1:] != row[:-1])
        for a, b in zip(change[:-1], change[1:]):
            v = row[a + 1]
            if v > 0:
                inst.notes.append(pretty_midi.Note(int(min(v, 127)), key + LOWEST_MIDI, t(a), t(b)))
    ped = pedal.reshape(-1).astype(int)
    prev = -1
    for i, v in enumerate(ped):
        if v != prev:
            inst.control_changes.append(pretty_midi.ControlChange(64, int(min(v, 127)), t(i)))
            prev = v
    pm.instruments.append(inst)
    return pm


def save_overview(path, mel, pr, ped, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                           gridspec_kw={"height_ratios": [3, 3, 0.6]})
    ext = [0, mel.shape[1] / FPS]
    ax[0].imshow(mel, origin="lower", aspect="auto", extent=ext + [0, N_MELS], cmap="magma")
    ax[0].set_ylabel("mel bin")
    ax[0].set_title(title)
    ax[1].imshow(pr, origin="lower", aspect="auto", extent=ext + [LOWEST_MIDI, LOWEST_MIDI + 88],
                 cmap="Greys", vmin=0, vmax=127)
    ax[1].set_ylabel("MIDI pitch")
    ax[2].plot(np.arange(ped.size) / FPS, ped.reshape(-1))
    ax[2].set_ylabel("pedal")
    ax[2].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def cmd_render(args):
    import soundfile as sf
    audio, midi = open_files(args.data_dir)
    examples = load_examples(args.data_dir, args.split)
    if args.ids:
        wanted = set(args.ids.split(","))
        examples = [e for e in examples if str(e["example_index"]) in wanted]
    else:
        rng = np.random.default_rng(args.seed)
        examples = [examples[i] for i in rng.choice(len(examples), min(args.n, len(examples)), replace=False)]
    os.makedirs(args.out, exist_ok=True)
    s0, s1 = int(args.start * FPS), int((args.start + args.dur) * FPS)
    for ex in examples:
        k = str(ex["example_index"])
        name = f"{args.split}_{k}" + (f"_midi{ex['midi_id']}" if "midi_id" in ex else "")
        mel = audio["features"][k][:N_MELS, s0:s1]
        pr = midi["midi_values"][k][:, s0:s1]
        ped = midi["pedal"][k][:, s0:s1]
        base = os.path.join(args.out, name)
        print(f"[{name}] {mel.shape[1] / FPS:.1f}s  meta={ex}")
        save_overview(base + "_overview.png", mel, pr, ped, name)
        pm = roll_to_midi(pr, ped)
        pm.write(base + ".mid")
        if args.sf2:
            y = pm.fluidsynth(fs=SR, sf2_path=args.sf2)
        else:
            y = pm.synthesize(fs=SR)
        sf.write(base + "_midi_synth.wav", 0.9 * y / (np.abs(y).max() + 1e-9), SR)
        if not args.skip_audio:
            sf.write(base + "_mel_griffinlim.wav", mel_to_wav(mel, args), SR)
    print(f"done: {len(examples)} examples in {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["inspect", "calibrate", "render"])
    p.add_argument("--data-dir", default="data/maestro")
    p.add_argument("--split", default="test", help="train | val | test (uses splits/<split>.json)")
    p.add_argument("--n", type=int, default=5, help="examples to render (or to calibrate on)")
    p.add_argument("--ids", default="", help="comma-separated example_index values instead of random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=float, default=30.0, help="clip start, seconds")
    p.add_argument("--dur", type=float, default=20.0, help="clip length, seconds")
    p.add_argument("--out", default="samples")
    p.add_argument("--sf2", default="", help="piano SoundFont for FluidSynth (optional)")
    p.add_argument("--skip-audio", action="store_true", help="only MIDI outputs, no Griffin-Lim")
    # Mel filterbank settings are not documented by the dataset card; check with `calibrate`.
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--win-length", type=int, default=None)
    p.add_argument("--fmin", type=float, default=30.0)
    p.add_argument("--fmax", type=float, default=8000.0)
    p.add_argument("--htk", action="store_true")
    p.add_argument("--gl-iters", type=int, default=64)
    args = p.parse_args()
    {"inspect": cmd_inspect, "calibrate": cmd_calibrate, "render": cmd_render}[args.command](args)


if __name__ == "__main__":
    main()
