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

# Inject the vendored transformers fork BEFORE anything else imports
# (see cp_transformer_m2c_moe_mask.py for the full reasoning).
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

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
    gate_off,
)
import contextlib
from preprocess_large_midi_dataset import DURATION_TEMPLATES


MODES = ['co', 'mel2chord', 'chord2mel', 'mel_only', 'chord_only']

# (program, track name). All melody tracks use General MIDI Acoustic Guitar
# (Nylon = Classical Guitar, program 24); all chord tracks use Acoustic
# Grand Piano (program 0). Track names still encode mode + which side is
# given vs generated, so you can identify them in GarageBand's track list
# without relying on instrument timbre.
MEL_PROGRAM = 24
CHORD_PROGRAM = 0
TRACK_SPECS = {
    ('co',         'mel'):   (MEL_PROGRAM,   'co-mel  (Classical Guitar)'),
    ('co',         'chord'): (CHORD_PROGRAM, 'co-chord  (Grand Piano)'),
    ('mel2chord',  'mel'):   (MEL_PROGRAM,   'mel2chord-mel  GIVEN  (Classical Guitar)'),
    ('mel2chord',  'chord'): (CHORD_PROGRAM, 'mel2chord-chord  GEN  (Grand Piano)'),
    ('chord2mel',  'mel'):   (MEL_PROGRAM,   'chord2mel-mel  GEN  (Classical Guitar)'),
    ('chord2mel',  'chord'): (CHORD_PROGRAM, 'chord2mel-chord  GIVEN  (Grand Piano)'),
    ('mel_only',   'mel'):   (MEL_PROGRAM,   'mel_only-mel  GEN  (Classical Guitar)'),
    ('chord_only', 'chord'): (CHORD_PROGRAM, 'chord_only-chord  GEN  (Grand Piano)'),
}


def _snap_up_to_bar(t_sec, bar_sec):
    """Round t_sec UP to the next bar boundary (in seconds)."""
    return math.ceil(t_sec / bar_sec - 1e-9) * bar_sec


def add_frames_to_track(inst, frames, time_offset, tokenizer,
                        with_velocity=False, tempo=120.0,
                        prompt_end_timestep=0):
    """Decode a list of frames into notes appended to `inst`, with every
    note's start/end shifted by time_offset (seconds).

    prompt_end_timestep is accepted but unused for note styling -- prompt
    vs generated is signalled only by markers (added globally to the midi
    file). All notes use the same velocity so the piano roll renders
    consistently."""
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


def _prompt_ends(mode, prompt_length, gen_length, used_b_prompt):
    """Per-modality timestep at which 'prompt' (given truth) ends and
    'generated' (sampled) begins. Returns (mel_end, chord_end).

    A modality fully given for the whole song (e.g., mel in mel2chord)
    returns gen_length, meaning every note in that track is treated as
    prompt (lower velocity, since it came from the source midi).
    A modality with no prompt at all returns 0."""
    if mode == 'co':
        return prompt_length, prompt_length
    if mode == 'mel2chord':
        return gen_length, (prompt_length if used_b_prompt else 0)
    if mode == 'chord2mel':
        return (prompt_length if used_b_prompt else 0), gen_length
    if mode == 'mel_only':
        return prompt_length, 0
    if mode == 'chord_only':
        return 0, prompt_length
    return 0, 0


def run_mode_for_song(model, mode, mel_path, chord_path, args):
    """Run one of the modes on one (mel, chord) pair. Returns
    (mel_frames, chord_frames, mel_prompt_end, chord_prompt_end) where the
    *_prompt_end values are the timestep at which 'prompt' transitions to
    'generated' for each modality (used downstream for note velocity and
    marker placement)."""
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

    # Single-stream modes: drop the cross-attention adapter so the silenced
    # modality's silence frames don't leak in (o_m = u_mm / o_c = u_cc).
    if mode == 'mel_only':
        gate_ctx = gate_off(model, 'mel')
    elif mode == 'chord_only':
        gate_ctx = gate_off(model, 'chord')
    else:
        gate_ctx = contextlib.nullcontext()

    with gate_ctx:
        mel_frames, chord_frames = general_inference(
            model, gen_length, B=1, subseq_len=subseq_len,
            temperature=args.temperature,
            mel_action_fn=mel_action,
            chord_action_fn=chord_action,
        )
    mel_end, chord_end = _prompt_ends(
        mode, args.prompt_length, gen_length, used_b_prompt,
    )
    return mel_frames, chord_frames, mel_end, chord_end


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
    # Markers to inject into the midi after pretty_midi writes it. Each entry
    # is (time_sec, text). GarageBand surfaces these as labelled lines in its
    # arrangement-view ruler so you can scroll to a specific song / boundary.
    markers = []
    for song_idx, (mel_path, chord_path) in enumerate(pairs):
        sid = os.path.splitext(os.path.basename(mel_path))[0]
        start_bar = current_offset / bar_duration_sec
        print(f'\n[{song_idx + 1}/{len(pairs)}] {sid}  '
              f'start={current_offset:.2f}s ({start_bar:.0f} bars)')
        song_offsets.append((sid, current_offset))
        markers.append((current_offset, f'[{sid}] song start'))
        for mode in MODES:
            print(f'  mode={mode}')
            try:
                mel_frames, chord_frames, mel_end, chord_end = run_mode_for_song(
                    model, mode, mel_path, chord_path, args,
                )
                # Single-stream modes populate only one of the two tracks;
                # don't write the silenced stream into the combined midi.
                if (mode, 'mel') in instruments:
                    add_frames_to_track(
                        instruments[(mode, 'mel')], mel_frames, current_offset,
                        model.tokenizer, model.with_velocity, tempo,
                        prompt_end_timestep=mel_end,
                    )
                if (mode, 'chord') in instruments:
                    add_frames_to_track(
                        instruments[(mode, 'chord')], chord_frames, current_offset,
                        model.tokenizer, model.with_velocity, tempo,
                        prompt_end_timestep=chord_end,
                    )

                # If this mode has a finite prompt boundary on either
                # modality (i.e. there is generated content somewhere),
                # add a marker so the user can jump to where generation
                # starts. We add a separate marker per modality only when
                # the two boundaries differ.
                song_gen_len = args.gen_length
                if mel_end > 0 and mel_end < song_gen_len:
                    markers.append((
                        current_offset + mel_end * time_step_sec,
                        f'[{sid}] {mode} mel prompt -> gen',
                    ))
                if chord_end > 0 and chord_end < song_gen_len and chord_end != mel_end:
                    markers.append((
                        current_offset + chord_end * time_step_sec,
                        f'[{sid}] {mode} chord prompt -> gen',
                    ))
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

    # Inject MIDI marker meta-events. pretty_midi doesn't expose markers, so
    # we open the just-written file with mido and append a dedicated marker
    # track. GarageBand surfaces these as labelled lines in its arrangement
    # ruler, letting you jump directly to a song or to where prompt -> gen
    # for any (song, mode, modality).
    if markers:
        import mido
        mid = mido.MidiFile(args.output)
        ticks_per_beat = mid.ticks_per_beat
        sec_per_tick = 60.0 / (tempo * ticks_per_beat)
        events = sorted(
            ((max(0, int(round(t / sec_per_tick))), text) for t, text in markers),
            key=lambda e: e[0],
        )
        marker_track = mido.MidiTrack()
        marker_track.append(mido.MetaMessage('track_name', name='MARKERS', time=0))
        last_tick = 0
        for tick, text in events:
            delta = max(0, tick - last_tick)
            marker_track.append(mido.MetaMessage('marker', text=text, time=delta))
            last_tick = tick
        marker_track.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(marker_track)
        mid.save(args.output)
        print(f'  + injected {len(events)} markers (song-start + prompt-end)')

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
