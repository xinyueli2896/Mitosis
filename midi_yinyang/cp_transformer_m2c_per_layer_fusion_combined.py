"""Combined-output inference for M2CPerLayerFusion: runs ALL 5 modes
(co / mel2chord / chord2mel / mel_only / chord_only) on every paired
(melody, chord) midi from two folders and assembles ONE midi file with all
the results organized for GarageBand -- same output layout as
cp_transformer_m2c_moe_combined.py.

This is a thin wrapper that reuses cp_transformer_m2c_moe_combined.main()
unchanged. Two names get monkey-patched in the combined module before main
runs:

  combined.load_model -> per-layer-fusion's load_model (instantiates
                         M2CPerLayerFusion, auto-detects gnl from
                         fusion_blocks count in the checkpoint state_dict)
  combined.gate_off  -> per-layer-fusion's gate_off (walks
                         model.fusion_blocks[i].gate_m / gate_c instead of
                         the top-level model.gate_m / gate_c that m2c MoE has)

Everything else -- per-mode action factories, sampling loop, frame -> midi
rendering, bar-aligned song concatenation, MIDI markers at prompt-end
boundaries, named-track layout (8 tracks per song, Classical Guitar +
Grand Piano timbres) -- is reused as-is.

Run from midi_yinyang/:

    python cp_transformer_m2c_per_layer_fusion_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder POP909-Dataset/POP909-melody \\
        --chord-folder POP909-Dataset/POP909-chord \\
        --output temp/per_layer_fusion_combined/all_modes.mid \\
        --prompt-length 100 --gen-length 384 \\
        --max-songs 10 --temperature 1.0 \\
        --model-size large
"""

# Inject the vendored fork BEFORE anything else imports.
import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

# Import the existing combined script (its main() function holds the full
# pipeline: arg parsing, model load, per-song mode loop, midi assembly).
import cp_transformer_m2c_moe_combined as combined

# Per-layer-fusion's load_model / gate_off, which differ from m2c MoE's only
# in the model class instantiated (M2CPerLayerFusion) and where the gates
# live (per-block under model.fusion_blocks[i].* vs. top-level).
from cp_transformer_m2c_per_layer_fusion_inference import (
    load_model as plf_load_model,
    gate_off as plf_gate_off,
)


# Monkey-patch: combined.main() looks up `load_model` and `gate_off` in the
# combined module's namespace at call time. Overwriting those module-level
# names here makes combined.main() use OUR versions without touching the
# combined script's source.
combined.load_model = plf_load_model
combined.gate_off = plf_gate_off


if __name__ == '__main__':
    combined.main()
