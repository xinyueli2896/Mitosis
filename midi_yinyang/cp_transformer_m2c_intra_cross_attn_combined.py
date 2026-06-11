"""Combined inference for M2CIntraCrossAttn: run all five modes on every
paired (mod_a, mod_b) midi from two folders and save EACH (song, mode)
as its own .mid file.

Same orchestration as cp_transformer_m2c_jointattn_combined.py; only the
model class differs. We import the jointattn combined's main() and
monkey-patch its `load_model` symbol to instantiate M2CIntraCrossAttn.

Output layout:

    <output-dir>/
        <song_id_0>/
            co.mid
            mel2chord.mid
            chord2mel.mid
            mel_only.mid
            chord_only.mid
        <song_id_1>/
            ...

Run from midi_yinyang/:

    python cp_transformer_m2c_intra_cross_attn_combined.py \\
        --ckpt ckpt/<run>/last.ckpt \\
        --mel-folder input/lamd_test_prompts_split/drum \\
        --chord-folder input/lamd_test_prompts_split/nondrum \\
        --output-dir temp/m2c_intra_cross_attn_combined \\
        --prompt-length 100 --gen-length 384 \\
        --max-songs 10 --temperature 1.0 \\
        --max-polyphony 16 --model-size large

For melchord task, swap --mel-folder / --chord-folder accordingly.

Use --modes to restrict which modes run (default: all five).
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__),
                           "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)


if __name__ == '__main__':
    # Patch the jointattn combined module's load_model to point at the
    # intra-cross-attn inference helper, then call its main(). Avoids
    # duplicating ~150 lines of CLI / folder / mode dispatch.
    import cp_transformer_m2c_jointattn_combined as _ja_comb
    from cp_transformer_m2c_intra_cross_attn_inference import (
        load_model as _icx_load_model,
    )
    _ja_comb.load_model = _icx_load_model
    _ja_comb.main()
