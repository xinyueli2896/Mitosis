"""Combined-output inference for M2CPerLayerFusionMask: runs ALL 5 modes
(co / mel2chord / chord2mel / mel_only / chord_only) on every paired
(melody, chord) midi from two folders and assembles ONE midi file with all
the results organized for GarageBand -- same output layout as
cp_transformer_m2c_moe_combined.py.

Differs from cp_transformer_m2c_per_layer_fusion_combined.py in that it
uses the mask-predict variant of the per-layer fusion model:

  - Model: M2CPerLayerFusionMask (per-block dual-pass + mask-predict training)
  - Sampling: mask_predict_with_modes (block-by-block AR + K iterative
              refinement passes per block; gives true bidirectional same-
              step coupling)

Thin wrapper: imports cp_transformer_m2c_moe_combined.main() and monkey-
patches its run_mode_for_song + load_model bindings so the combined output
pipeline (per-song mode loop, bar-aligned song concatenation, named-track
layout, MIDI markers at prompt-end boundaries) runs unchanged.

Run from midi_yinyang/:

    python cp_transformer_m2c_per_layer_fusion_mask_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder POP909-Dataset/POP909-melody \\
        --chord-folder POP909-Dataset/POP909-chord \\
        --output temp/per_layer_fusion_mask_combined/all_modes.mid \\
        --prompt-length 100 --gen-length 384 \\
        --n-refine-steps 2 \\
        --max-songs 10 --temperature 1.0 \\
        --model-size large
"""

# Inject the vendored fork BEFORE anything else imports.
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse

# Import the existing combined pipeline (full CLI + per-song mode loop +
# bar-aligned song concatenation + MIDI marker injection).
import cp_transformer_m2c_moe_combined as combined
import cp_transformer_m2c_moe_mask_inference as mask_inf
from cp_transformer_m2c_moe_combined import _load_prompt_tokens, _prompt_ends
from cp_transformer_m2c_per_layer_fusion_mask_inference import load_model as plf_mask_load_model


def run_mode_for_song(model, mode, mel_path, chord_path, args):
    """Mask-predict version of combined.run_mode_for_song.

    Same return contract: (mel_frames, chord_frames, mel_end, chord_end).
    Uses mask_predict_with_modes instead of general_inference (the AR
    sampling loop), so single-stream modes keep the silenced modality
    permanently masked rather than feeding silence frames.

    n_refine_steps comes from args.n_refine_steps (defaults to 2 = MaskGIT
    / Gibbs coupling).
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
        # Override prompt_length in args.* so the called fn uses the per-song
        # bound. mask_predict_with_modes reads prompt_length from its kwarg.
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
        # Total output length = prompt + gen. Load enough mel for the whole
        # thing. gen_length passed to mask_predict_with_modes_ar is the
        # SAMPLING portion only; the function adds prompt_length internally.
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
        chord_prompt = None  # never used in mel_only; chord stays permanently masked
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

    mel_frames, chord_frames = mask_inf.mask_predict_with_modes(
        model, mode,
        mel_prompt=mel_prompt,
        chord_prompt=chord_prompt,
        prompt_length=_prompt_length,
        gen_length=gen_length,
        n_refine_steps=args.n_refine_steps,
        temperature=args.temperature,
    )
    # Marker placement: where does the "given" portion end for each modality?
    # In conditional modes the condition is given for the WHOLE output
    # (prompt + sampled), and the generated track is given only for the
    # prompt portion. Override the AR _prompt_ends defaults accordingly.
    if mode == 'mel2chord':
        mel_end = args.prompt_length + args.gen_length     # mel given throughout
        chord_end = args.prompt_length if used_b_prompt else 0
    elif mode == 'chord2mel':
        mel_end = args.prompt_length if used_b_prompt else 0
        chord_end = args.prompt_length + args.gen_length   # chord given throughout
    else:
        mel_end, chord_end = _prompt_ends(
            mode, args.prompt_length, gen_length, used_b_prompt,
        )
    return mel_frames, chord_frames, mel_end, chord_end


# Monkey-patch: combined.main() looks up run_mode_for_song and load_model in
# the combined module's namespace at call time. Overwriting both routes the
# pipeline through OUR mask-predict + M2CPerLayerFusionMask versions without
# touching the original combined script source.
combined.run_mode_for_song = run_mode_for_song
combined.load_model = plf_mask_load_model


def main():
    # Wrap combined.main() to add --n-refine-steps (used by the mask-predict
    # sampler, doesn't exist on the AR combined CLI). Parsed by combined.main
    # via its argparser would fail without this addition.
    #
    # Approach: monkey-patch argparse.ArgumentParser temporarily so it adds
    # --n-refine-steps when called inside combined.main().
    import argparse as _argparse
    _orig_parse_args = _argparse.ArgumentParser.parse_args

    def _patched_parse_args(self, *a, **kw):
        # Add the flag if it doesn't exist yet (idempotent against re-entry)
        if not any('--n-refine-steps' in opt.option_strings
                   for opt in self._actions):
            self.add_argument('--n-refine-steps', type=int, default=2,
                              help='Within-block refinement passes. 1 = '
                                   'parallel (no intra-block coupling). 2 = '
                                   'MaskGIT/Gibbs (bidirectional same-step '
                                   'coupling).')
        return _orig_parse_args(self, *a, **kw)

    _argparse.ArgumentParser.parse_args = _patched_parse_args
    try:
        combined.main()
    finally:
        _argparse.ArgumentParser.parse_args = _orig_parse_args


if __name__ == '__main__':
    main()
