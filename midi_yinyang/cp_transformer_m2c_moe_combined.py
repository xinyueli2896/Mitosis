"""Run all three M2C inference modes on every paired (melody, chord) midi
from two folders and assemble ONE midi file with the results organized
for GarageBand:

  - 6 named tracks (mode x modality): one row per (co, mel2chord, chord2mel)
    times (mel, chord). Each track uses a distinct General MIDI program so
    you can tell them apart by ear if you just hit play.
  - Songs placed sequentially in time, separated by `--gap-bars` of silence.
  - A sibling .txt file lists each song's start time so you can scrub to it.

Modes (each contributes both mel and chord output frames):
  - co          : both prompts; sample both for [prompt_length, gen_length)
  - mel2chord   : melody full ground truth; chord generated from scratch
  - chord2mel   : chord full ground truth; melody generated from scratch

Run from midi_yinyang/:

    python cp_transformer_m2c_moe_combined.py \
        --ckpt ckpt/<run>/last.ckpt \
        --mel-folder POP909-Dataset/POP909-melody \
        --chord-folder POP909-Dataset/POP909-chord \
        --output temp/m2c_combined/all_modes.mid \
        --prompt-length 100 --gen-length 384 \
        --max-songs 10 --temperature 1.0 \
        --model-size large
"""

import argparse
import math
import os
from glob import glob

import torch
import pretty_midi

from cp_transformer_m2c_moe import RoFormerSymbolicTransformer
from cp_transformer_m2c_moe_inference import (
    general_inference,
    make_actions_co,
    make_actions_conditional,
    make_actions_single,
    _load_prompt_tokens,
    load_model,
)
from preprocess_large_midi_dataset import DURATION_TEMPLATES


MODES = ['co', 'mel2chord', 'chord2mel', 'mel_only', 'chord_only']

# (program, track name). Programs picked so each (mode, modality) sounds
# different in GarageBand's default GM playback — easy to A/B by ear.
# Single-stream modes only populate one of the two tracks; the other one
# still exists in the file but stays empty for those songs.
TRACK_SPECS = {
    ('co',         'mel'):   (24, 'co-mel  (Nylon Guitar)'),
    ('co',         'chord'): ( 0, 'co-chord  (Acoustic Piano)'),
    ('mel2chord',  'mel'):   (25, 'mel2chord-mel  GIVEN  (Steel Guitar)'),
    ('mel2chord',  'chord'): ( 4, 'mel2chord-chord  GEN  (Electric Piano 1)'),
    ('chord2mel',  'mel'):   (26, 'chord2mel-mel  GEN  (Jazz Guitar)'),
    ('chord2mel',  'chord'): ( 5, 'chord2mel-chord  GIVEN  (Electric Piano 2)'),
    ('mel_only',   'mel'):   (27, 'mel_only-mel  GEN  (Clean Guitar)'),
    ('chord_only', 'chord'): ( 6, 'chord_only-chord  GEN  (Harpsichord)'),
}


def _snap_up_to_bar(t_sec, bar_sec):
    """Round t_sec UP to the next bar boundary (in seconds)."""
    return math.ceil(t_sec / bar_sec - 1e-9) * bar_sec


def add_frames_to_track(inst, frames, time_offset, tokenizer,
                        with_velocity=False, tempo=120.0):
    """Decode a list of frames into notes appended to `inst`, with every
    note's start/end shifted by time_offset (seconds)."""
    if frames is None:
        return
    time_step_length = 60.0 / tempo / 4
    for t, frame in enumerate(frames):
        start_time = t * time_step_length + time_offset
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


def run_mode_for_song(model, mode, mel_path, chord_path, args):
    """Run one of the three modes on one (mel, chord) pair and return the
    per-frame token lists for melody and chord."""
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

        # Optional chord prefix: use the first prompt_length timesteps of the
        # chord midi so chord generation has a seed instead of starting from
        # scratch. mel is still given throughout.
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


def _list_midis(folder):
    if folder is None:
        return []
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mel-folder', required=True)
    p.add_argument('--chord-folder', required=True)
    p.add_argument('--output', required=True, help='output midi path')
    p.add_argument('--prompt-length', type=int, default=100,
                   help='timesteps of prompt used for co mode')
    p.add_argument('--gen-length', type=int, default=384,
                   help='total output length in timesteps')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-polyphony', type=int, default=4)
    p.add_argument('--gap-bars', type=float, default=2.0,
                   help='bars of silence between consecutive songs')
    p.add_argument('--max-songs', type=int, default=None,
                   help='cap on number of songs (default: all paired)')
    p.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
    args = p.parse_args()

    # Build paired list
    mel_files = _list_midis(args.mel_folder)
    chord_index = {os.path.basename(q): q for q in _list_midis(args.chord_folder)}
    pairs = [(m, chord_index[os.path.basename(m)]) for m in mel_files
             if os.path.basename(m) in chord_index]
    skipped = len(mel_files) - len(pairs)
    if args.max_songs is not None:
        pairs = pairs[:args.max_songs]
    print(f'Pairing: {len(pairs)} matched, {skipped} unmatched, '
          f'processing {len(pairs)} songs.')

    # Load model once
    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        with_velocity=args.with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
    )
    model.cuda()
    model.eval()

    # Allocate six tracks up front.
    instruments = {
        key: pretty_midi.Instrument(program=program, name=name)
        for key, (program, name) in TRACK_SPECS.items()
    }

    # Timing
    tempo = 120.0
    time_step_sec = 60.0 / tempo / 4
    song_duration_sec = args.gen_length * time_step_sec
    bar_duration_sec = 4 * 60.0 / tempo
    gap_sec = args.gap_bars * bar_duration_sec

    # Process every song; place its outputs at current_offset across the tracks.
    # After each song advance to (song_duration + gap) and snap UP to the next
    # bar boundary so every song starts on a whole-bar line in GarageBand.
    current_offset = 0.0
    song_offsets = []
    for song_idx, (mel_path, chord_path) in enumerate(pairs):
        sid = os.path.splitext(os.path.basename(mel_path))[0]
        start_bar = current_offset / bar_duration_sec
        print(f'\n[{song_idx + 1}/{len(pairs)}] {sid}  '
              f'start={current_offset:.2f}s ({start_bar:.0f} bars)')
        song_offsets.append((sid, current_offset))
        for mode in MODES:
            print(f'  mode={mode}')
            try:
                mel_frames, chord_frames = run_mode_for_song(
                    model, mode, mel_path, chord_path, args,
                )
                # Single-stream modes populate only one of the two tracks;
                # don't write the silenced stream into the combined midi.
                if (mode, 'mel') in instruments:
                    add_frames_to_track(
                        instruments[(mode, 'mel')], mel_frames, current_offset,
                        model.tokenizer, model.with_velocity, tempo,
                    )
                if (mode, 'chord') in instruments:
                    add_frames_to_track(
                        instruments[(mode, 'chord')], chord_frames, current_offset,
                        model.tokenizer, model.with_velocity, tempo,
                    )
            except Exception as e:
                print(f'    failed: {e!r}')
        # Advance and snap to the next whole-bar line.
        current_offset = _snap_up_to_bar(
            current_offset + song_duration_sec + gap_sec,
            bar_duration_sec,
        )

    # Assemble midi in a deterministic, viewer-friendly track order matching
    # MODES x ('mel', 'chord'), skipping (mode, modality) pairs that don't
    # exist (e.g. single-stream modes only define one of the two slots).
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for mode in MODES:
        for modality in ('mel', 'chord'):
            if (mode, modality) in instruments:
                midi.instruments.append(instruments[(mode, modality)])

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    midi.write(args.output)
    print(f'\nWrote {args.output}')

    # Write the song offsets next to the midi for navigation.
    offsets_path = (args.output[:-4] if args.output.lower().endswith('.mid')
                    else args.output) + '_offsets.txt'
    with open(offsets_path, 'w') as f:
        f.write('# m2c combined output -- song offsets\n')
        f.write(f'# tempo={tempo} BPM, song_duration={song_duration_sec:.2f}s, '
                f'gap={gap_sec:.2f}s\n')
        f.write('# song_id\tstart_seconds\tstart_bars\n')
        for sid, offset in song_offsets:
            f.write(f'{sid}\t{offset:.2f}\t{offset / bar_duration_sec:.2f}\n')
    print(f'Wrote {offsets_path}')


if __name__ == '__main__':
    main()
