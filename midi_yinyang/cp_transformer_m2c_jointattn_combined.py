"""Combined inference for M2CJointAttn: run all five modes on every paired
(melody, chord) midi from two folders and assemble ONE midi where each
(mode, modality) slot expands to as many tracks as there are unique
programs in the generated/given notes. The drum stream becomes a single
is_drum=True track. The non-drum stream splits into one track per
encountered program (bass, piano, lead, strings, etc.).

This is a full custom implementation rather than a wrapper around
cp_transformer_m2c_moe_combined: that script pre-allocates exactly ONE
pretty_midi.Instrument per (mode, modality) and squashes every note into
it, which collapses LAMD's multi-instrument streams into a single
Classical-Guitar / Grand-Piano track per slot. We need lazy per-program
allocation, which is incompatible with the pre-allocation pattern.

Run from midi_yinyang/:

    python cp_transformer_m2c_jointattn_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder input/lamd_test_prompts_split/nondrum \\
        --chord-folder input/lamd_test_prompts_split/drum \\
        --output temp/m2c_jointattn_combined/all_modes.mid \\
        --prompt-length 100 --gen-length 384 \\
        --max-songs 10 --temperature 1.0 \\
        --max-polyphony 16 --model-size large

Track naming format: "<mode>-<modality>-PROG{NNN}" for pitched
instruments and "<mode>-<modality>-DRUM" for drums. The (mode, modality)
prefix lets you mute/solo by mode in GarageBand even though the per-mode
slot is now a track GROUP rather than a single track.
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import contextlib
import math
import os
from glob import glob

import pretty_midi
import torch

from cp_transformer_m2c_moe_inference import (
    general_inference,
    make_actions_co,
    make_actions_conditional,
    make_actions_single,
    _load_prompt_tokens,
)
from cp_transformer_m2c_jointattn_inference import load_model
from preprocess_large_midi_dataset import DURATION_TEMPLATES


MODES = ['co', 'mel2chord', 'chord2mel', 'mel_only', 'chord_only']


def _snap_up_to_bar(t_sec, bar_sec):
    """Round t_sec UP to the next bar boundary (in seconds)."""
    return math.ceil(t_sec / bar_sec) * bar_sec


def _list_midis(folder):
    if folder is None:
        return []
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


# ---------------------------------------------------------------------------
# Per-mode dispatch (mirrors m2c_moe_combined.run_mode_for_song without
# the gate_off / cross-stream gate logic).
# ---------------------------------------------------------------------------

def run_mode_for_song(model, mode, mel_path, chord_path, args):
    """Returns (mel_frames, chord_frames, mel_prompt_end, chord_prompt_end)."""
    used_b_prompt = False
    if mode == 'co':
        if not mel_path or not chord_path:
            return None, None, 0, 0
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
            return None, None, 0, 0
        condition = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        b_prompt = None
        if chord_path and args.prompt_length > 0:
            b_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
            b_prompt = b_prompt[:, :args.prompt_length]
            used_b_prompt = True
        gen_length = min(args.gen_length, condition.shape[1])
        condition = condition[:, :gen_length]
        subseq_len = condition.shape[2]
        mel_action, chord_action = make_actions_conditional(
            condition, 'mel', b_prompt=b_prompt,
        )

    elif mode == 'chord2mel':
        if not chord_path:
            return None, None, 0, 0
        condition = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        b_prompt = None
        if mel_path and args.prompt_length > 0:
            b_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
            b_prompt = b_prompt[:, :args.prompt_length]
            used_b_prompt = True
        gen_length = min(args.gen_length, condition.shape[1])
        condition = condition[:, :gen_length]
        subseq_len = condition.shape[2]
        mel_action, chord_action = make_actions_conditional(
            condition, 'chord', b_prompt=b_prompt,
        )

    elif mode == 'mel_only':
        if not mel_path:
            return None, None, 0, 0
        prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        common = min(prompt.shape[1], args.prompt_length)
        prompt = prompt[:, :common]
        subseq_len = prompt.shape[2]
        mel_action, chord_action = make_actions_single(prompt, 'mel', common)
        gen_length = args.gen_length

    elif mode == 'chord_only':
        if not chord_path:
            return None, None, 0, 0
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

    # Prompt-end per modality (where given truth stops and sampling begins).
    if mode == 'co':
        mel_end = chord_end = args.prompt_length
    elif mode == 'mel2chord':
        mel_end = gen_length
        chord_end = args.prompt_length if used_b_prompt else 0
    elif mode == 'chord2mel':
        mel_end = args.prompt_length if used_b_prompt else 0
        chord_end = gen_length
    elif mode == 'mel_only':
        mel_end, chord_end = args.prompt_length, 0
    else:  # chord_only
        mel_end, chord_end = 0, args.prompt_length

    return mel_frames, chord_frames, mel_end, chord_end


# ---------------------------------------------------------------------------
# Lazy per-(mode, modality, program) instrument map
# ---------------------------------------------------------------------------

def add_frames_lazy(inst_map, frames, mode, modality, time_offset,
                     tokenizer, with_velocity=False, tempo=120.0):
    """Decode frames into notes, routing each note to an instrument keyed by
    (mode, modality, program). Drums (program==127) collapse into a single
    is_drum=True track per (mode, modality). Pitched notes get one track per
    encountered program. New instruments are appended to inst_map; the
    caller is responsible for adding them to the final PrettyMIDI in order.
    """
    if frames is None:
        return
    time_step_length = 60.0 / tempo / 4

    def get_inst(program):
        if program == 127:
            key = (mode, modality, 'drum')
            if key not in inst_map:
                inst_map[key] = pretty_midi.Instrument(
                    program=0, is_drum=True,
                    name=f'{mode}-{modality}-DRUM',
                )
            return inst_map[key]
        key = (mode, modality, program)
        if key not in inst_map:
            inst_map[key] = pretty_midi.Instrument(
                program=program,
                name=f'{mode}-{modality}-PROG{program:03d}',
            )
        return inst_map[key]

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
                program = a_token % 128
                velocity_bin = a_token // 128
                note_velocity = velocity_bin * 8 + 4
                pitch_duration = b_token - 16 * 128
            else:
                program = a_token
                note_velocity = 100
                pitch_duration = b_token - 128
            pitch = pitch_duration % 128
            duration = pitch_duration // 128
            if program < 0 or program >= 128:
                continue
            if pitch < 0 or pitch >= 128:
                continue
            if duration < 0 or duration >= len(DURATION_TEMPLATES):
                continue
            end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
            inst = get_inst(program)
            inst.notes.append(pretty_midi.Note(
                velocity=note_velocity, pitch=pitch,
                start=start_time, end=end_time,
            ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mel-folder', required=True)
    p.add_argument('--chord-folder', required=True)
    p.add_argument('--output', required=True, help='output midi path')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-polyphony', type=int, default=16)
    p.add_argument('--gap-bars', type=float, default=2.0)
    p.add_argument('--max-songs', type=int, default=None)
    p.add_argument('--model-size', type=str, default='large',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
    p.add_argument('--global-num-layers', type=int, default=None)
    args = p.parse_args()

    mel_files = _list_midis(args.mel_folder)
    chord_index = {os.path.basename(q): q for q in _list_midis(args.chord_folder)}
    pairs = [(m, chord_index[os.path.basename(m)]) for m in mel_files
             if os.path.basename(m) in chord_index]
    skipped = len(mel_files) - len(pairs)
    if args.max_songs is not None:
        pairs = pairs[:args.max_songs]
    print(f'Pairing: {len(pairs)} matched, {skipped} unmatched, '
          f'processing {len(pairs)} songs.')

    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        with_velocity=args.with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=args.global_num_layers,
    )
    model.cuda()
    model.eval()

    tempo = 120.0
    time_step_sec = 60.0 / tempo / 4
    song_duration_sec = args.gen_length * time_step_sec
    bar_duration_sec = 4 * 60.0 / tempo
    gap_sec = args.gap_bars * bar_duration_sec

    inst_map = {}                                                # (mode, modality, program_or_drum) -> Instrument
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
                mel_frames, chord_frames, _, _ = run_mode_for_song(
                    model, mode, mel_path, chord_path, args,
                )
                # Single-stream modes silence the other side; don't render it.
                write_mel = mode != 'chord_only'
                write_chord = mode != 'mel_only'
                if write_mel:
                    add_frames_lazy(
                        inst_map, mel_frames, mode, 'mel', current_offset,
                        model.tokenizer, model.with_velocity, tempo,
                    )
                if write_chord:
                    add_frames_lazy(
                        inst_map, chord_frames, mode, 'chord', current_offset,
                        model.tokenizer, model.with_velocity, tempo,
                    )
            except Exception as e:
                print(f'    failed: {e!r}')

        current_offset = _snap_up_to_bar(
            current_offset + song_duration_sec + gap_sec,
            bar_duration_sec,
        )

    # Order tracks: MODES x ('mel', 'chord') x sorted(program). Drum first
    # within a (mode, modality) group so it's easy to find.
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for mode in MODES:
        for modality in ('mel', 'chord'):
            keys = [k for k in inst_map if k[0] == mode and k[1] == modality]
            # Drum first, then ascending program number.
            keys.sort(key=lambda k: (0 if k[2] == 'drum' else 1,
                                       0 if k[2] == 'drum' else k[2]))
            for k in keys:
                midi.instruments.append(inst_map[k])

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    midi.write(args.output)
    print(f'\nWrote {args.output} ({len(midi.instruments)} tracks)')

    # Sibling .txt listing song start times.
    txt_path = os.path.splitext(args.output)[0] + '_song_offsets.txt'
    with open(txt_path, 'w') as f:
        for sid, off in song_offsets:
            bar = off / bar_duration_sec
            f.write(f'{off:8.2f}s  bar {bar:6.1f}  {sid}\n')
    print(f'Wrote {txt_path}')


if __name__ == '__main__':
    main()
