"""Combined-output inference for M2CMixtureHead: runs ALL 5 modes on every
paired (melody, chord) midi from two folders and assembles ONE midi with the
results organized for GarageBand. Same output layout as the m2c MoE combined
script -- 8 named tracks per song (Classical Guitar + Grand Piano timbres),
bar-aligned song boundaries, MIDI markers at prompt-end / song-start.

Thin wrapper: imports cp_transformer_m2c_moe_combined.main() unchanged and
monkey-patches two of its module-level names:

  combined.load_model         -> mixture-head load_model (auto-detects
                                  global_num_layers AND mixture_K from
                                  the checkpoint).
  combined.run_mode_for_song  -> mixture-head sampling for all 5 modes
                                  (uses block_logp_per_k + sample_block_given_k
                                  from the mixture head's inference module).

Run:
    python cp_transformer_m2c_mixture_head_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder POP909-Dataset/POP909-melody \\
        --chord-folder POP909-Dataset/POP909-chord \\
        --output temp/mixture_head_combined/all_modes.mid \\
        --prompt-length 100 --gen-length 384 \\
        --max-songs 10 --model-size large
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse

import cp_transformer_m2c_moe_combined as combined
from cp_transformer_m2c_moe_combined import _load_prompt_tokens, _prompt_ends
from cp_transformer_m2c_mixture_head_inference import (
    load_model as mh_load_model,
    generate_modes as mh_generate_modes,
)


def run_mode_for_song(model, mode, mel_path, chord_path, args):
    """Mixture-head version of combined.run_mode_for_song.

    Same return contract: (mel_frames, chord_frames, mel_end, chord_end).
    Routes through generate_modes (joint mixture sampling) instead of
    general_inference (AR sampling).
    """
    used_b_prompt = False

    if mode == 'co':
        if not mel_path or not chord_path:
            return None, None, 0, 0
        mel_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        chord_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        common = min(mel_prompt.shape[1], chord_prompt.shape[1], args.prompt_length)
        mel_prompt = mel_prompt[:, :common]
        chord_prompt = chord_prompt[:, :common]
        gen_length = args.gen_length
        _prompt_length = common

    elif mode == 'mel2chord':
        if not mel_path:
            return None, None, 0, 0
        mel_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        chord_prompt = None
        if chord_path and args.prompt_length > 0:
            chord_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
            chord_prompt = chord_prompt[:, :args.prompt_length]
            used_b_prompt = True
        total_T = min(args.prompt_length + args.gen_length, mel_prompt.shape[1])
        mel_prompt = mel_prompt[:, :total_T]
        gen_length = args.gen_length
        _prompt_length = args.prompt_length

    elif mode == 'chord2mel':
        if not chord_path:
            return None, None, 0, 0
        chord_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        mel_prompt = None
        if mel_path and args.prompt_length > 0:
            mel_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
            mel_prompt = mel_prompt[:, :args.prompt_length]
            used_b_prompt = True
        total_T = min(args.prompt_length + args.gen_length, chord_prompt.shape[1])
        chord_prompt = chord_prompt[:, :total_T]
        gen_length = args.gen_length
        _prompt_length = args.prompt_length

    elif mode == 'mel_only':
        if not mel_path:
            return None, None, 0, 0
        mel_prompt = _load_prompt_tokens(model, mel_path, args.max_polyphony)
        common = min(mel_prompt.shape[1], args.prompt_length)
        mel_prompt = mel_prompt[:, :common]
        chord_prompt = None
        gen_length = args.gen_length
        _prompt_length = common

    elif mode == 'chord_only':
        if not chord_path:
            return None, None, 0, 0
        chord_prompt = _load_prompt_tokens(model, chord_path, args.max_polyphony)
        common = min(chord_prompt.shape[1], args.prompt_length)
        chord_prompt = chord_prompt[:, :common]
        mel_prompt = None
        gen_length = args.gen_length
        _prompt_length = common

    else:
        raise ValueError(f'unknown mode {mode}')

    mel_frames, chord_frames = mh_generate_modes(
        model, mode,
        mel_prompt=mel_prompt,
        chord_prompt=chord_prompt,
        prompt_length=_prompt_length,
        gen_length=gen_length,
        temperature=args.temperature,
    )

    # Marker placement: condition track given throughout in conditional modes.
    if mode == 'mel2chord':
        mel_end = args.prompt_length + args.gen_length
        chord_end = args.prompt_length if used_b_prompt else 0
    elif mode == 'chord2mel':
        mel_end = args.prompt_length if used_b_prompt else 0
        chord_end = args.prompt_length + args.gen_length
    else:
        mel_end, chord_end = _prompt_ends(
            mode, args.prompt_length, gen_length, used_b_prompt,
        )
    return mel_frames, chord_frames, mel_end, chord_end


# Monkey-patch: combined.main() looks up run_mode_for_song and load_model in
# the combined module's namespace at call time. Overwriting both routes the
# pipeline through OUR mixture-head versions without touching the original
# combined script source.
combined.run_mode_for_song = run_mode_for_song
combined.load_model = mh_load_model


if __name__ == '__main__':
    combined.main()
