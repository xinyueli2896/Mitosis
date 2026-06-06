"""Combined inference for M2CJointAttn: run all five modes on every paired
(melody, chord) midi from two folders and assemble ONE midi with the
results organized for GarageBand. Output structure (named tracks +
markers) is identical to cp_transformer_m2c_moe_combined.py -- only the
model class differs.

This is a thin wrapper: it imports the moe_combined main() and patches
two symbols:
  - load_model: swapped to the M2CJointAttn loader so the model is built
    from cp_transformer_m2c_jointattn.M2CJointAttn instead of the moe
    backbone.
  - gate_off: replaced by a no-op context manager because M2CJointAttn
    has no cross-stream gate to disable. mel_only/chord_only modes still
    run, but the silenced stream participates in joint attention (mild
    leakage, see the inference script's note).

Run from midi_yinyang/:

    python cp_transformer_m2c_jointattn_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder input/lamd_test_prompts_split/nondrum \\
        --chord-folder input/lamd_test_prompts_split/drum \\
        --output temp/m2c_jointattn_combined/all_modes.mid \\
        --prompt-length 100 --gen-length 384 \\
        --max-songs 10 --temperature 1.0 \\
        --max-polyphony 16 --model-size large

Output MIDI lays out 8 named tracks (mode x modality, skipping the
silenced side of single-stream modes), one row of notes per song,
separated by --gap-bars of silence and a marker labelled "[<song_id>]
song start" at each boundary.
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import contextlib

import cp_transformer_m2c_moe_combined as _moe_combined
from cp_transformer_m2c_jointattn_inference import load_model as _jointattn_load_model


# 1. Swap load_model so the combined script's main() builds M2CJointAttn
#    instead of the moe RoFormerSymbolicTransformer.
_moe_combined.load_model = _jointattn_load_model

# 2. M2CJointAttn has no cross-stream gate. Replace gate_off with a no-op
#    context manager so mel_only/chord_only modes don't crash trying to
#    look up model.gate_m / model.gate_c (which don't exist on this model).
_moe_combined.gate_off = lambda *_args, **_kwargs: contextlib.nullcontext()


if __name__ == '__main__':
    _moe_combined.main()
