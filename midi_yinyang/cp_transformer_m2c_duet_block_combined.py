"""Combined inference for M2CDuetBlockAttn: run all 5 modes on every
paired (mod_a, mod_b) midi from two folders and write one .mid per
(song, mode) pair under <output-dir>/<song>/<mode>.mid.

Same orchestration as cp_transformer_m2c_intra_cross_attn_combined.py;
only the model class + generation loop differ. We import the jointattn
combined's main() and patch BOTH the load_model symbol AND the
general_inference symbol (DuetBlock needs a custom decode loop that
appends query slots and reads off their hiddens).

Run from midi_yinyang/:

    python cp_transformer_m2c_duet_block_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder input/rwc_test_prompts_split/drum \\
        --chord-folder input/rwc_test_prompts_split/nondrum \\
        --output-dir temp/m2c_duet_block_rwc \\
        --prompt-length 64 --gen-length 384 \\
        --temperature 0.9 --max-polyphony 16 --model-size large
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)


if __name__ == '__main__':
    # Patch the jointattn inference module FIRST -- the combined module
    # imports run_one/run_folder from it, and run_one calls
    # general_inference at module scope of jointattn_inference. We need
    # the patched symbols visible there before the combined main starts.
    import cp_transformer_m2c_jointattn_inference as _ja_inf
    from cp_transformer_m2c_duet_block_inference import (
        load_model as _db_load_model,
        general_inference_duet_block as _db_general_inference,
    )
    _ja_inf.load_model = _db_load_model
    _ja_inf.general_inference = _db_general_inference

    # Then dispatch via jointattn_combined's CLI/main.
    import cp_transformer_m2c_jointattn_combined as _ja_comb
    _ja_comb.load_model = _db_load_model
    _ja_comb.main()
