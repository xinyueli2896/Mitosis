"""Inference for M2CJointAttn (per-modality Q/K/V/O + joint self-attn +
shared MoE FFN).

The sampling loop, MIDI decode, and folder iteration are shared with the
m2c_moe inference script; this file only swaps in the M2CJointAttn class
for model construction. Five modes (same as the other m2c variants):

  --mode co
      Co-generation. Both prompts given for --prompt-length timesteps,
      then sample both jointly. Requires --melody and --chord.

  --mode mel2chord
      Melody fully provided; sample chord for each timestep. Optionally
      seed the chord stream with the first --prompt-length frames of
      --chord (otherwise chord is sampled from t=0).

  --mode chord2mel
      Symmetric: chord is the condition, melody is generated.

  --mode mel_only
      Melody prompt for --prompt-length timesteps, then continue
      autoregressively. Chord stream gets silence frames every step
      and is NOT written to the output midi. Note: joint attention
      still sees the silence frames, so there is mild leakage compared
      to a true single-stream model -- but the silence carries almost
      no information so the practical impact is small.

  --mode chord_only
      Symmetric: chord prompt + chord continuation; melody silenced.

Run from midi_yinyang/:

    python cp_transformer_m2c_jointattn_inference.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mode co \\
        --melody input/lamd_test_prompts_split/nondrum/<song>.mid \\
        --chord  input/lamd_test_prompts_split/drum/<song>.mid \\
        --prompt-length 100 --gen-length 384 --temperature 1.0 \\
        --max-polyphony 16

    python cp_transformer_m2c_jointattn_inference.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mode chord2mel \\
        --chord  input/lamd_test_prompts_split/drum/<song>.mid \\
        --gen-length 384 --max-polyphony 16
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import contextlib
import os
import re
from glob import glob

import pretty_midi
import torch

from cp_transformer_m2c_moe_inference import (
    MODES,
    general_inference,
    make_actions_co,
    make_actions_conditional,
    make_actions_single,
    _load_prompt_tokens,
    _list_midis,
)
from cp_transformer_m2c_jointattn import M2CJointAttn
from preprocess_large_midi_dataset import DURATION_TEMPLATES


# ---------------------------------------------------------------------------
# Multi-program decode (replaces the single-instrument decode_m2c_frames from
# the m2c_moe inference module).
#
# The m2c_moe version hard-codes program=24 for "mel" and program=0 for
# "chord", losing per-note program information. For LAMD that means the
# drum stream loses is_drum=True and the non-drum stream loses its
# instrument variety (bass, piano, lead, strings, etc. all become one
# Classical Guitar). This version routes each note to a unique instrument
# keyed by program number, with program=127 -> Instrument(0, is_drum=True).
# ---------------------------------------------------------------------------

def decode_m2c_frames(mel_frames, chord_frames, save_path, tokenizer,
                      with_velocity=False, tempo=120.0,
                      write_mel=True, write_chord=True):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4

    inst_map = {}

    def get_inst(program):
        # program 127 = drum (preprocess_large_midi_dataset assigns this for
        # any is_drum=True instrument). Use a single drum track per file.
        if program == 127:
            key = ('drum',)
            if key not in inst_map:
                inst_map[key] = pretty_midi.Instrument(
                    program=0, is_drum=True, name='DRUM',
                )
                midi.instruments.append(inst_map[key])
            return inst_map[key]
        key = ('inst', program)
        if key not in inst_map:
            inst_map[key] = pretty_midi.Instrument(
                program=program, name=f'PROG{program:03d}',
            )
            midi.instruments.append(inst_map[key])
        return inst_map[key]

    streams = []
    if write_mel:
        streams.append(mel_frames)
    if write_chord:
        streams.append(chord_frames)

    for frames in streams:
        if frames is None:
            continue
        for t, frame in enumerate(frames):
            start_time = t * time_step_length
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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _infer_global_num_layers(ckpt_path, ck, model_size):
    """Same priority order as the moe inference helper, but matches our
    target schema 'global_layers.N.*' rather than 'global_roformer.layer.N.*'."""
    if isinstance(ck, dict):
        hp = ck.get('hyper_parameters') or {}
        if isinstance(hp, dict) and hp.get('global_num_layers') is not None:
            return int(hp['global_num_layers']), 'hyper_parameters'

    m = re.search(r'_gnl(\d+)_', os.path.basename(ckpt_path))
    if m:
        return int(m.group(1)), 'filename'

    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    if isinstance(state, dict):
        layer_idxs = set()
        pat = re.compile(r'^global_layers\.(\d+)\.')
        for k in state.keys():
            mm = pat.search(k)
            if mm:
                layer_idxs.add(int(mm.group(1)))
        if layer_idxs:
            return max(layer_idxs) + 1, 'state_dict_inference'

    return (12 if model_size == 'large' else 6), 'size_default'


def load_model(ckpt_path, model_size='large', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None,
               global_num_layers=None, preserve_program=True,
               min_acc_tokens_before_eos=0):
    """Build M2CJointAttn with the right depth/experts and load weights.

    preserve_program defaults to True (the M2CJointAttn primary use case).
    This is what enables both preprocess() to keep the actual a-slot program
    in the input tokens AND local_sampling() to permit the full 0..127
    program range in generated tokens. Pass False to fall back to the
    POP909 hardcoded behavior.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, source = _infer_global_num_layers(ckpt_path, ck, model_size)
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto-detected from {source})')
    else:
        print(f'[load_model] global_num_layers={global_num_layers} (caller override)')

    net = M2CJointAttn(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        global_num_layers=global_num_layers,
        preserve_program=preserve_program,
        min_acc_tokens_before_eos=min_acc_tokens_before_eos,
    )
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing keys ({len(missing)}): {missing[:5]}'
              f'{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected keys ({len(unexpected)}): {unexpected[:5]}'
              f'{"..." if len(unexpected) > 5 else ""}')
    return net


# ---------------------------------------------------------------------------
# Per-mode dispatch (no gate_off -- M2CJointAttn has no cross-stream gate)
# ---------------------------------------------------------------------------

def run_one(model, mode, mel_path, chord_path, args, out_subdir):
    tokenizer = model.tokenizer
    subseq_m = subseq_c = None
    mel_prompt = chord_prompt = None

    if mode == 'co':
        if mel_path is None or chord_path is None:
            raise ValueError('mode=co requires both --melody and --chord')
        mel_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        chord_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        common = min(mel_prompt.shape[1], chord_prompt.shape[1], args.prompt_length)
        mel_prompt = mel_prompt[:, :common]
        chord_prompt = chord_prompt[:, :common]
        subseq_m = mel_prompt.shape[2]
        subseq_c = chord_prompt.shape[2]
        mel_action, chord_action = make_actions_co(mel_prompt, chord_prompt, common)
        gen_length = args.gen_length

    elif mode in ('mel2chord', 'chord2mel'):
        cond_modality = 'mel' if mode == 'mel2chord' else 'chord'
        cond_path = mel_path if cond_modality == 'mel' else chord_path
        other_path = chord_path if cond_modality == 'mel' else mel_path
        if cond_path is None:
            raise ValueError(f'mode={mode} requires --{cond_modality}')
        condition = _load_prompt_tokens(model, cond_path, args.max_polyphony)

        b_prompt = None
        if other_path is not None and args.prompt_length > 0:
            b_prompt = _load_prompt_tokens(model, other_path, args.max_polyphony)
            b_prompt = b_prompt[:, :args.prompt_length]

        gen_length = min(args.gen_length, condition.shape[1])
        condition = condition[:, :gen_length]
        subseq_m = subseq_c = condition.shape[2]
        mel_action, chord_action = make_actions_conditional(
            condition, cond_modality, b_prompt=b_prompt,
        )

    elif mode in ('mel_only', 'chord_only'):
        prompt_modality = 'mel' if mode == 'mel_only' else 'chord'
        source_path = mel_path if prompt_modality == 'mel' else chord_path
        if source_path is None:
            raise ValueError(f'mode={mode} requires --{prompt_modality}')
        prompt = _load_prompt_tokens(model, source_path, args.max_polyphony)
        common = min(prompt.shape[1], args.prompt_length)
        prompt = prompt[:, :common]
        subseq_m = subseq_c = prompt.shape[2]
        mel_action, chord_action = make_actions_single(
            prompt, prompt_modality, common,
        )
        gen_length = args.gen_length

    else:
        raise ValueError(f'unknown mode {mode}')

    assert subseq_m == subseq_c, (subseq_m, subseq_c)
    subseq_len = subseq_m
    B = 1

    if args.n_samples > 1:
        n = args.n_samples

        def _broadcast(fn):
            def wrapped(t):
                a = fn(t)
                if isinstance(a, tuple) and a[0] == 'given':
                    return ('given', a[1].repeat(n, *([1] * (a[1].dim() - 1))))
                return a
            return wrapped
        mel_action = _broadcast(mel_action)
        chord_action = _broadcast(chord_action)
        B = n

    mel_frames, chord_frames = general_inference(
        model, gen_length, B, subseq_len,
        temperature=args.temperature,
        mel_action_fn=mel_action,
        chord_action_fn=chord_action,
    )

    save_name = getattr(model, 'save_name', 'm2c_jointattn_run')
    out_dir = os.path.join(f'temp/{save_name}', out_subdir, mode)
    write_mel = mode != 'chord_only'
    write_chord = mode != 'mel_only'
    for i in range(B):
        m_i = [f[i:i + 1, :] for f in mel_frames]
        c_i = [f[i:i + 1, :] for f in chord_frames]
        decode_m2c_frames(
            m_i, c_i,
            save_path=os.path.join(
                out_dir,
                f'sample_{i}_temp{args.temperature}.mid',
            ),
            tokenizer=tokenizer,
            with_velocity=model.with_velocity,
            write_mel=write_mel,
            write_chord=write_chord,
        )


def run_folder(model, mode, mel_folder, chord_folder, args):
    """Same iteration pattern as m2c_moe's run_folder but dispatches to our
    run_one. The pairing logic is identical so we just adapt it."""
    mel_files = _list_midis(mel_folder)
    chord_files = _list_midis(chord_folder)
    chord_index = {os.path.basename(p): p for p in chord_files}

    if mode in ('chord2mel', 'chord_only'):
        driver = chord_files
        def pair(driver_path):
            base = os.path.basename(driver_path)
            mp = None
            for m in mel_files:
                if os.path.basename(m) == base:
                    mp = m
                    break
            return mp, driver_path
    else:
        driver = mel_files
        def pair(driver_path):
            base = os.path.basename(driver_path)
            return driver_path, chord_index.get(base)

    items = [pair(p) for p in driver]
    print(f'Found {len(items)} items in driver folder for mode={mode}.')
    for i, (mp, cp) in enumerate(items):
        base = mp if mp is not None else cp
        sid = os.path.splitext(os.path.basename(base))[0]
        print(f'[{i + 1}/{len(items)}] {sid}')
        try:
            run_one(model, mode, mp, cp, args, out_subdir=sid)
        except Exception as e:
            print(f'  failed: {e!r}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--melody', help='single melody midi')
    p.add_argument('--chord', help='single chord midi')
    p.add_argument('--mel-folder', help='folder of melody midis')
    p.add_argument('--chord-folder', help='folder of chord midis')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-samples', type=int, default=1)
    p.add_argument('--max-polyphony', type=int, default=16,
                   help='Default 16 for LAMD cp16. Use 4 for POP909 cp4.')
    p.add_argument('--model-size', type=str, default='large',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
    p.add_argument('--global-num-layers', type=int, default=None,
                   help='Override global transformer depth. Default: '
                        'auto-detect from checkpoint hyperparameters / '
                        'filename _gnlN_ tag / state_dict layer count / '
                        'size-based fallback (6 or 12).')
    p.add_argument('--preserve-program', dest='preserve_program',
                   action='store_true', default=True,
                   help='Preserve per-note program at inference. Default '
                        'True for M2CJointAttn -- must match how the ckpt '
                        'was trained.')
    p.add_argument('--hardcode-program', dest='preserve_program',
                   action='store_false',
                   help='Squash program to 24 (mel) / 0 (chord) like the '
                        'legacy POP909 path. Only use if the ckpt was '
                        'trained without --preserve_program.')
    p.add_argument('--min-chord-tokens-before-eos', dest='min_acc_tokens_before_eos',
                   type=int, default=0,
                   help='Force the chord/non-drum side to emit at least N '
                        'tokens within each frame before EOS is allowed. '
                        'Rescue knob for ckpts trained with too-aggressive '
                        'EOS up-weighting that collapse to silence on the '
                        'non-drum stream. Try 2 (= forces at least 1 full '
                        'note: program + pitch_duration) or 4.')
    args = p.parse_args()

    if args.mel_folder or args.chord_folder:
        if args.melody or args.chord:
            p.error('--melody/--chord cannot be combined with folder args')

    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        with_velocity=args.with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=args.global_num_layers,
        preserve_program=args.preserve_program,
        min_acc_tokens_before_eos=args.min_acc_tokens_before_eos,
    )
    print(f'[main] preserve_program={args.preserve_program}  '
          f'min_chord_tokens_before_eos={args.min_acc_tokens_before_eos}')
    model.save_name = os.path.basename(args.ckpt)
    model.cuda()
    model.eval()

    if args.mel_folder or args.chord_folder:
        run_folder(model, args.mode, args.mel_folder, args.chord_folder, args)
    else:
        sid = os.path.splitext(os.path.basename(
            args.melody if args.melody else args.chord
        ))[0]
        run_one(model, args.mode, args.melody, args.chord, args, out_subdir=sid)


if __name__ == '__main__':
    main()
