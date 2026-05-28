"""Inference for the melody+chord MoE model with three generation modes.

Modes:

  --mode co
      Co-generation. Both prompts given for prompt_length timesteps;
      sample both jointly for [prompt_length, gen_length).
      Requires --melody and --chord (or --mel-folder + --chord-folder).

  --mode mel2chord
      Conditional. Melody fully provided as ground truth for gen_length
      timesteps; chord generated from scratch, same length. Requires
      --melody (or --mel-folder); --chord can be supplied to seed but
      is not used after timestep 0 -- chord is overwritten step by step.

  --mode chord2mel
      Symmetric: chord is the condition, melody is generated.

  --mode mel_only
      Single stream. Melody prompt for prompt_length timesteps, then
      melody is continued autoregressively. The chord stream is forced
      to a silence frame at every step (a-slot=eos, rest=pad). Only the
      melody track is written to the output midi.

  --mode chord_only
      Symmetric: chord prompt, chord continuation; melody silenced.

Run from midi_yinyang/:

    # co-generation (default)
    python cp_transformer_m2c_moe_inference.py \
        --mode co \
        --ckpt ckpt/<run>/last.ckpt \
        --melody POP909-Dataset/POP909-melody/001.mid \
        --chord  POP909-Dataset/POP909-chord/001.mid \
        --prompt-length 100 --gen-length 384 --temperature 1.0

    # full mel -> generate chord
    python cp_transformer_m2c_moe_inference.py \
        --mode mel2chord \
        --ckpt ckpt/<run>/last.ckpt \
        --melody POP909-Dataset/POP909-melody/001.mid \
        --gen-length 384

    # melody-only continuation
    python cp_transformer_m2c_moe_inference.py \
        --mode mel_only \
        --ckpt ckpt/<run>/last.ckpt \
        --melody POP909-Dataset/POP909-melody/001.mid \
        --prompt-length 100 --gen-length 384
"""

import argparse
import os
from glob import glob

import torch
import pretty_midi

from cp_transformer_m2c_moe import RoFormerSymbolicTransformer
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES


MODES = ['co', 'mel2chord', 'chord2mel', 'mel_only', 'chord_only']


def make_silence_frame(B, subseq_len, tokenizer, device):
    """A frame meaning 'no notes at this timestep'.
    a-slot at position 0 holds eos; everything else pad."""
    frame = torch.full(
        (B, subseq_len), tokenizer.pad_token,
        dtype=torch.long, device=device,
    )
    frame[:, 0] = tokenizer.eos_token
    return frame


def resolve_frame(action, B, subseq_len, tokenizer, device):
    """Convert a non-'sample' action into an actual frame tensor."""
    if isinstance(action, tuple) and action[0] == 'given':
        return action[1]
    if action == 'silence':
        return make_silence_frame(B, subseq_len, tokenizer, device)
    raise ValueError(f'unexpected action {action!r}')


# ---------------------------------------------------------------------------
# Per-mode action factories
# ---------------------------------------------------------------------------

def make_actions_co(mel_prompt, chord_prompt, prompt_length):
    def mel_action(t):
        if t < prompt_length:
            return ('given', mel_prompt[:, t, :])
        return 'sample'

    def chord_action(t):
        if t < prompt_length:
            return ('given', chord_prompt[:, t, :])
        return 'sample'
    return mel_action, chord_action


def make_actions_conditional(condition_data, condition_modality, b_prompt=None):
    """condition_data: [B, T, subseq] -- the fully-given stream.
    condition_modality: 'mel' or 'chord' -- which one is the condition.
    b_prompt: optional [B, P, subseq] -- a prefix of the GENERATED stream
        to seed it with, so it doesn't have to start from scratch. The
        generated stream is given for the first P timesteps and sampled
        afterwards. If None, the generated stream is sampled from t=0."""
    p_b = 0 if b_prompt is None else b_prompt.shape[1]
    if condition_modality == 'mel':
        def mel_action(t):
            return ('given', condition_data[:, t, :])

        def chord_action(t):
            if t < p_b:
                return ('given', b_prompt[:, t, :])
            return 'sample'
    else:
        def mel_action(t):
            if t < p_b:
                return ('given', b_prompt[:, t, :])
            return 'sample'

        def chord_action(t):
            return ('given', condition_data[:, t, :])
    return mel_action, chord_action


def make_actions_single(prompt, prompt_modality, prompt_length):
    """One modality has a prompt + autoregressive continuation; the other is
    silenced at every timestep."""
    if prompt_modality == 'mel':
        def mel_action(t):
            if t < prompt_length:
                return ('given', prompt[:, t, :])
            return 'sample'
        chord_action = lambda t: 'silence'
    else:
        mel_action = lambda t: 'silence'
        def chord_action(t):
            if t < prompt_length:
                return ('given', prompt[:, t, :])
            return 'sample'
    return mel_action, chord_action


# ---------------------------------------------------------------------------
# Unified sampling loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def general_inference(model, gen_length, B, subseq_len, temperature,
                      mel_action_fn, chord_action_fn):
    """Run gen_length timesteps; at each step decide per-modality whether to
    use a given tensor, sample from the model, or use a silence frame.
    Returns (mel_frames, chord_frames), each a list of [B, subseq] tensors."""
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    H = model.hidden_size

    h_buffer = torch.zeros(B, 0, H, device=device)
    mel_frames = []
    chord_frames = []

    for t in range(gen_length):
        if t % 10 == 0:
            print(f'[gen] step {t}/{gen_length}')

        m_action = mel_action_fn(t)
        c_action = chord_action_fn(t)
        need_model = m_action == 'sample' or c_action == 'sample'

        m_tokens = None
        c_tokens = None

        if need_model:
            # Build the shift-by-2 input: prepend the SOS pair to the buffer
            # of t past (mel, chord) block summaries. h_in length = 2 + 2t;
            # output positions 2t and 2t+1 predict m_t and c_t.
            sos = model.global_sos.view(1, 1, -1).expand(B, 2, H)
            h_in = torch.cat([sos, h_buffer], dim=1)
            h_out, _ = model._global_interaction(h_in)
            h_m_pred = h_out[:, -2]
            h_c_pred = h_out[:, -1]

            if m_action == 'sample':
                m_tokens = model.local_sampling(
                    h_m_pred, max_subseq_len=subseq_len,
                    temperature=temperature, token_type_id=0,
                )
            if c_action == 'sample':
                c_tokens = model.local_sampling(
                    h_c_pred, max_subseq_len=subseq_len,
                    temperature=temperature, token_type_id=1,
                )

        if m_tokens is None:
            m_tokens = resolve_frame(m_action, B, subseq_len, tokenizer, device)
        if c_tokens is None:
            c_tokens = resolve_frame(c_action, B, subseq_len, tokenizer, device)

        mel_frames.append(m_tokens)
        chord_frames.append(c_tokens)

        # Encode the chosen frames back to single block summaries and append
        # so the next step's _global_interaction sees them as past context.
        m_h = model._encode_frame(m_tokens, 0)
        c_h = model._encode_frame(c_tokens, 1)
        h_buffer = torch.cat([h_buffer, m_h, c_h], dim=1)

    return mel_frames, chord_frames


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_raw(midi_path, max_polyphony):
    out = preprocess_midi(midi_path, max_polyphony, filter=False)
    if out is None:
        raise RuntimeError(f'preprocess_midi returned None for {midi_path}')
    raw, _ = out
    return raw  # [T, max_polyphony * 4]


def _load_prompt_tokens(model, midi_path, max_polyphony):
    """preprocess_midi -> model.preprocess -> [1, T, subseq] tokens."""
    raw = _load_raw(midi_path, max_polyphony).unsqueeze(0).cuda()
    pitch_shift = torch.zeros(1, dtype=torch.int8, device=raw.device)
    return model.preprocess(raw, pitch_shift)  # [1, T, subseq]


def decode_m2c_frames(mel_frames, chord_frames, save_path, tokenizer,
                      with_velocity=False, tempo=120.0,
                      write_mel=True, write_chord=True):
    """Render alternating mel/chord frames to one midi with two named tracks."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4

    mel_inst = pretty_midi.Instrument(program=24, name='MELODY')
    chord_inst = pretty_midi.Instrument(program=0, name='CHORD')
    if write_mel:
        midi.instruments.append(mel_inst)
    if write_chord:
        midi.instruments.append(chord_inst)

    for stream_label, frames in (('mel', mel_frames), ('chord', chord_frames)):
        if stream_label == 'mel' and not write_mel:
            continue
        if stream_label == 'chord' and not write_chord:
            continue
        inst = mel_inst if stream_label == 'mel' else chord_inst
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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


# ---------------------------------------------------------------------------
# Mode dispatch
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

        # Optional prefix of the GENERATED stream so it doesn't have to
        # start from scratch. If the partner midi is supplied and
        # --prompt-length > 0, use its first prompt_length timesteps.
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

    # Use the same subseq_len for both streams (they should match after
    # uniform preprocessing: max_polyphony*2). If they differ, pad/truncate
    # would be needed; we just assume equal here.
    assert subseq_m == subseq_c, (subseq_m, subseq_c)
    subseq_len = subseq_m
    B = 1

    if args.n_samples > 1:
        # Replicate prompts and condition across the sample dimension by
        # wrapping the action functions.
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

    save_name = getattr(model, 'save_name', 'm2c_moe_run')
    tag = out_subdir
    out_dir = os.path.join(f'temp/{save_name}', tag, mode)
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


def _list_midis(folder):
    if folder is None:
        return []
    return sorted(
        p for p in glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.mid', '.midi'))
    )


def run_folder(model, mode, mel_folder, chord_folder, args):
    mel_files = _list_midis(mel_folder)
    chord_files = _list_midis(chord_folder)
    chord_index = {os.path.basename(p): p for p in chord_files}

    # Pair driver depends on which modality is required for the mode.
    if mode in ('chord2mel', 'chord_only'):
        driver = chord_files

        def pair(driver_path):
            base = os.path.basename(driver_path)
            mp = None
            # mel optional in these modes
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
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, model_size='small', with_velocity=False,
               moe_num_experts=4, moe_topk=2, moe_intermediate_size=None):
    net = RoFormerSymbolicTransformer(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
    )
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing keys: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected keys: {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
    return net


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
    p.add_argument('--max-polyphony', type=int, default=4)
    p.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'large'])
    p.add_argument('--with-velocity', action='store_true')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
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
    )
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
