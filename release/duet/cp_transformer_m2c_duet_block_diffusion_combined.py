"""Combined 5-mode inference for M2CDuetBlockDiffusion (variant A.3):
runs all 5 modes (co, mel2chord, chord2mel, mel_only, chord_only) on
every paired (mod_a, mod_b) midi from two folders and writes one .mid
per (song, mode) pair under <output-dir>/<song>/<mode>.mid.

Same orchestration as cp_transformer_m2c_duet_block_combined.py, but
the load_model + general_inference symbols are patched to the diffusion
versions so each step runs the parallel-diffusion refinement loop.

Run from midi_yinyang/:

    python cp_transformer_m2c_duet_block_diffusion_combined.py \\
        --ckpt ckpt/<run>/ \\
        --mel-folder input/rwc_test_prompts_split/drum \\
        --chord-folder input/rwc_test_prompts_split/nondrum \\
        --output-dir temp/m2c_duet_block_diffusion_rwc \\
        --prompt-length 64 --gen-length 384 \\
        --temperature 1.0 --max-polyphony 16 --model-size large
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)


if __name__ == '__main__':
    # Patch jointattn_inference's symbols (run_one/run_folder use
    # general_inference via that module's namespace).
    import cp_transformer_m2c_jointattn_inference as _ja_inf
    from cp_transformer_m2c_duet_block_diffusion_inference import (
        load_model as _diff_load_model,
        general_inference_diffusion as _diff_general_inference,
    )
    _ja_inf.load_model = _diff_load_model
    _ja_inf.general_inference = _diff_general_inference

    # ALSO patch jointattn_combined's own symbols (it imports
    # general_inference directly from moe_inference, not via
    # jointattn_inference).
    import cp_transformer_m2c_jointattn_combined as _ja_comb
    _ja_comb.load_model = _diff_load_model
    _ja_comb.general_inference = _diff_general_inference

    # Belt-and-suspenders: patch the source module too.
    import cp_transformer_m2c_moe_inference as _moe_inf
    _moe_inf.general_inference = _diff_general_inference

    _ja_comb.main()
