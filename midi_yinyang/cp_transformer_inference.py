import numpy as np

from cp_transformer import RoFormerSymbolicTransformer, CPTokenizer
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES
from settings import RWC_DATASET_PATH
from ckpt_utils import resolve_best_ckpt
import torch
import pretty_midi
import os
def decode_output(outputs, save_path=None, tempo=120.0, ratio=1.0, velocity=100, with_velocity=False, extra_instruments=None, fixed_program=None):
    tokenizer = CPTokenizer(with_velocity=with_velocity)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    if not isinstance(ratio, tuple):
        ratio = (ratio,) * len(outputs)
    for r, output in zip(ratio, outputs):
        instrument_map = {}
        for time_step, data in enumerate(output):
            content = data.squeeze(0)
            start_time = time_step * time_step_length
            for i in range(0, len(content), 2):
                a_token = int(content[i].item())
                if a_token == tokenizer.eos_token:
                    break
                if i + 1 >= len(content):
                    print('Incomplete note @', time_step, i)
                    break
                b_token = int(content[i + 1].item())
                if with_velocity:
                    # a-slot: program + 128 * velocity_bin in [0, 128*16)
                    program = a_token % 128
                    velocity_bin = a_token // 128
                    note_velocity = velocity_bin * 8 + 4  # midpoint of 8-wide bin
                    # b-slot: pitch + (duration + 16) * 128 in [128*16, 128*16 + 24*128)
                    pitch_duration = b_token - 16 * 128
                else:
                    program = a_token
                    note_velocity = velocity
                    pitch_duration = b_token - 128
                pitch = pitch_duration % 128
                duration = pitch_duration // 128
                if program < 0 or program >= 128:
                    print('Invalid program:', program, '@', time_step, i)
                    break
                if with_velocity and (velocity_bin < 0 or velocity_bin >= 16):
                    print('Invalid velocity bin:', velocity_bin, '@', time_step, i)
                    break
                if pitch < 0 or pitch >= 128:
                    print('Invalid pitch:', pitch, '@', time_step, i)
                    break
                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print('Invalid duration:', duration, '@', time_step, i)
                    break
                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
                if program not in instrument_map:
                    if program == 127: # placeholder for drums
                        instrument_map[program] = pretty_midi.Instrument(0, is_drum=True)
                    else:
                        instrument_map[program] = pretty_midi.Instrument(fixed_program if fixed_program is not None else program)
                    midi.instruments.append(instrument_map[program])
                instrument = instrument_map[program]
                instrument.notes.append(pretty_midi.Note(velocity=note_velocity, pitch=pitch, start=start_time * r, end=end_time * r))
    if extra_instruments is not None:
        for instrument in extra_instruments:
            midi.instruments.append(instrument)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        midi.write(save_path)
    return midi


def decompress(model, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0)
    x = x.cuda()
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda())[:2]


def get_input_tempo(midi_path, default=120.0):
    """First tempo of the prompt midi, so outputs render at the same speed."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        _, tempi = pm.get_tempo_changes()
        if len(tempi) > 0 and tempi[0] > 0:
            return float(tempi[0])
    except Exception as e:
        print(f'[tempo] failed to read {midi_path}: {e!r}')
    return default


# ---------------------------------------------------------------------------
# Track-name -> program tagging.
#
# The CP tokenizer identifies streams ONLY by program; midi track identity is
# discarded. When a prompt has melody and chord on the SAME program but on
# tracks named "MELODY"/"CHORD", we can still keep the streams separate:
# make a temp copy for tokenization with the CHORD track's program forced to
# a spare tag program, generate, then map the tag back to the original
# program in the written outputs. Result: identical sound, but melody and
# chord end up as separate tracks in the output midis.
# ---------------------------------------------------------------------------

CHORD_TAG_PROGRAM = 48   # spare program used only inside tokenization


def _force_track_program(track, program):
    import mido
    out = mido.MidiTrack()
    had_pc = False
    for msg in track:
        if msg.type == 'program_change':
            out.append(msg.copy(program=program))
            had_pc = True
        else:
            out.append(msg.copy())
    if not had_pc:
        out.insert(0, mido.Message('program_change', program=program, time=0))
    return out


def tag_chord_track(midi_path, tmp_dir, tag_program=CHORD_TAG_PROGRAM):
    """If midi_path has a track named 'chord', write a temp copy with that
    track's program forced to tag_program (for tokenization only).

    Returns (path_to_tokenize, restore_map). restore_map maps the tag
    program back to the chord track's original program for the OUTPUT
    midis ({} when no tagging happened)."""
    import mido
    mid = mido.MidiFile(midi_path)
    chord_idx = [i for i, t in enumerate(mid.tracks)
                 if (t.name or '').strip().lower() == 'chord']
    if not chord_idx:
        return midi_path, {}
    orig_prog = 0
    for i in chord_idx:
        for msg in mid.tracks[i]:
            if msg.type == 'program_change':
                orig_prog = msg.program
                break
    if orig_prog == tag_program:
        return midi_path, {}   # already distinct enough; nothing to do
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for i, t in enumerate(mid.tracks):
        out.tracks.append(_force_track_program(t, tag_program)
                          if i in chord_idx else t)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, os.path.basename(midi_path))
    out.save(tmp_path)
    print(f'  [tag] CHORD track: program {orig_prog} -> {tag_program} '
          f'(tokenization only; outputs restored to {orig_prog})')
    return tmp_path, {tag_program: orig_prog}


def _decode_and_save(frames, save_path, tempo, restore_map):
    """decode_output + map tag programs back, keeping tracks separate."""
    midi = decode_output(frames, save_path=None, tempo=tempo)
    if restore_map:
        for inst in midi.instruments:
            if not inst.is_drum and inst.program in restore_map:
                inst.program = restore_map[inst.program]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def continuation(model, midi_path, prompt_length=100, generation_length=384,
                 temperature=1.0, n_samples=1, tempo=120.0,
                 split_named_tracks=True, max_polyphony=16):
    name = os.path.basename(midi_path)
    restore_map = {}
    tokenize_path = midi_path
    if split_named_tracks:
        tokenize_path, restore_map = tag_chord_track(
            midi_path, f'temp/{model.save_name}/_tagged')
    # filter=False: the LA-quantization heuristic is for noisy scraped data;
    # it can reject curated inputs (RWC, POP909) by returning None.
    # max_polyphony MUST match the ckpt's training data: a model
    # finetuned on cp4 single-stream data sees 8-token subsequences,
    # so tokenizing prompts at cp16 (32 tokens) would be a
    # train/inference mismatch.
    out = preprocess_midi(tokenize_path, max_polyphony, filter=False)
    if out is None:
        print(f'  [skip] preprocess_midi failed for {midi_path}')
        return
    x = decompress(model, out[0])
    print(x.shape)
    if x.shape[1] < prompt_length:
        print(f'  [skip] only {x.shape[1]} frames < prompt_length {prompt_length}')
        return
    x = x[:, :prompt_length]
    _decode_and_save([x[:, i, :] for i in range(x.shape[1])],
                     f'temp/{model.save_name}/{name}_prompt.mid',
                     tempo, restore_map)
    with torch.no_grad():
        x = x.repeat(n_samples, 1, 1)
        output = model.global_sampling(x, temperature=temperature, max_seq_len=generation_length)
    for i in range(n_samples):
        output_i = [output[j][i:i + 1, :] for j in range(len(output))]
        _decode_and_save(output_i,
                         f'temp/{model.save_name}/{name}_temp{temperature}_continuation_{i}.mid',
                         tempo, restore_map)


def inference_perplexity(midi_files, max_polyphony=16, seq_length=384):
    model = RoFormerSymbolicTransformer.load_from_checkpoint('ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt')
    model.cuda()
    model.eval()
    # Collect the files, convert to tensor
    data = []
    for file in midi_files:
        x = preprocess_midi(file, max_polyphony, fixed_length=seq_length)[0]
        x = x[:, :seq_length]
        data.append(x)
    # Form a batch
    x = torch.stack(data, dim=0).cuda()
    # Calculate perplexity
    with torch.no_grad():
        perplexity_mean, perplexity_std = model.inference_perplexity(x)
    return perplexity_mean, perplexity_std


if __name__ == '__main__':
    import argparse
    from glob import glob

    p = argparse.ArgumentParser(
        description='Single-stream continuation on a folder of prompt midis '
                    '(merged stream, program token = stream identity).')
    p.add_argument('--ckpt', default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt',
                   help='ckpt file OR run directory; a directory (or a '
                        'last.ckpt with val_loss-tagged siblings) '
                        'auto-selects the smallest-val_loss ckpt')
    p.add_argument('--midi-folder', required=True,
                   help='folder of prompt midis (searched recursively)')
    p.add_argument('--prompt-length', type=int, default=64,
                   help='prompt frames; 64 = 4 bars at 16 frames/bar')
    p.add_argument('--gen-length', type=int, default=384,
                   help='TOTAL frames incl. prompt (matches the m2c scripts)')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-samples', type=int, default=2)
    p.add_argument('--max-polyphony', type=int, default=16,
                   help='polyphony slots used to tokenize prompts. MUST match '
                        'the training data of --ckpt: 16 for the LA-pretrained '
                        'and merged-stream finetunes, 4 for single-modality '
                        'specialists trained on cp4 streams.')
    p.add_argument('--max-songs', type=int, default=None)
    p.add_argument('--save-name', default=None,
                   help='output subdir under temp/; default: ckpt basename')
    p.add_argument('--no-split-tracks', action='store_true', default=False,
                   help='disable the MELODY/CHORD track-name separation '
                        '(by default a track named CHORD is program-tagged '
                        'during tokenization so the two streams stay on '
                        'separate tracks in the outputs, then restored)')
    args = p.parse_args()

    # Resolve to the BEST-val ckpt when given a run directory (or a
    # last.ckpt that has val_loss-tagged siblings), matching how the
    # duet inference scripts pick checkpoints -- otherwise an eval
    # would compare best-val duet models against a LAST baseline.
    ckpt_path = resolve_best_ckpt(args.ckpt)
    print(f'[main] ckpt = {ckpt_path}')
    model = RoFormerSymbolicTransformer.load_from_checkpoint(ckpt_path)
    model.save_name = args.save_name or os.path.basename(args.ckpt)
    model.cuda()
    model.eval()

    files = sorted(
        glob(os.path.join(args.midi_folder, '**', '*.mid'), recursive=True)
        + glob(os.path.join(args.midi_folder, '**', '*.MID'), recursive=True))
    if args.max_songs is not None:
        files = files[:args.max_songs]
    print(f'{len(files)} prompt MIDIs in {args.midi_folder}')
    for i, f in enumerate(files):
        print(f'=== [{i + 1}/{len(files)}] {f}')
        try:
            continuation(model, f,
                         prompt_length=args.prompt_length,
                         generation_length=args.gen_length,
                         temperature=args.temperature,
                         n_samples=args.n_samples,
                         tempo=get_input_tempo(f),
                         split_named_tracks=not args.no_split_tracks,
                         max_polyphony=args.max_polyphony)
        except Exception as e:
            print(f'  failed: {e!r}')
