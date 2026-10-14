# Sourced by every stage script. Not meant to be run on its own.
#
# Every command in this repository runs from duet/ (the model code uses
# paths relative to it: data/, ckpt/, temp/, results/), with the patched
# transformers package first on PYTHONPATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUET="$ROOT/duet"

if [[ ! -f "$DUET/transformers_roformer_moe/src/transformers/models/roformer/grouped_gemm_util.py" ]]; then
    bash "$ROOT/scripts/setup_transformers.sh"
fi
export PYTHONPATH="$DUET/transformers_roformer_moe/src:$DUET${PYTHONPATH:+:$PYTHONPATH}"

cd "$DUET"
mkdir -p data ckpt temp results figures

# ---- dataset ------------------------------------------------------------
# DATASET picks the corpus every stage reads and writes: pop909 (the
# paper) or nottingham (the same melody-chord task on the Nottingham folk
# tunes). Each gets its own training tensors, held-out prompts, run
# directories, samples and result files, so the two never mix.
DATASET="${DATASET:-pop909}"
case "$DATASET" in
pop909)
    TASK=melchord                                   # tasks.py entry for the duet models
    DUET_DATA=pop909                                # data/${DUET_DATA}_{melody,chord}_cp4_v2.pt
    SS_DATA=data/pop909_melchord_tagged_cp16_v2.pt  # single-stream corpus
    SS_RUN_TAG=tagged
    DS_SUFFIX=""                                    # appended to scratch folders
    _EVAL_SRC=input/heldout_v5; _OUT_ROOT=temp/E1; RESULTS_TAG=E1
    # Held-out songs dropped from generation and scoring. 506 and 786 have
    # no chord note inside the prompt window, so no system is given
    # anything to continue; 456 has no Whole-Song Gen output. Leaves 92.
    _EXCLUDE="456 506 786"
    ;;
nottingham)
    TASK=melchord_nottingham
    DUET_DATA=nottingham
    SS_DATA=data/nottingham_melchord_tagged_cp16_v2.pt
    SS_RUN_TAG=nottingham_tagged
    DS_SUFFIX=_nottingham
    _EVAL_SRC=input/nottingham_heldout; _OUT_ROOT=temp/E1_nottingham; RESULTS_TAG=E1_nottingham
    _EXCLUDE=""
    ;;
*) echo "unknown DATASET '$DATASET' (pop909 nottingham)" >&2; exit 1 ;;
esac

# ---- the paper's evaluation protocol -----------------------------------
# Every system continues the same prompts: the first PROMPT_LENGTH frames
# (sixteenth notes; 80 = five bars) of each held-out song, to GEN_LENGTH
# frames in total, N_SAMPLES times.
PROMPT_LENGTH="${PROMPT_LENGTH:-80}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
EXCLUDE_SONGS="${EXCLUDE_SONGS-$_EXCLUDE}"
EVAL_SRC="${EVAL_SRC:-$_EVAL_SRC}"         # built by prepare_data.sh eval
OUT_ROOT="${OUT_ROOT:-$_OUT_ROOT}"         # generated samples, one folder per system
PROMPTS="$OUT_ROOT/prompts"

# The pretrained single-stream backbone (Jiang et al., ISMIR 2025).
BACKBONE="${BACKBONE:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"

# ---- system registry ----------------------------------------------------
# Friendly name -> the internal id every output folder, table and figure
# uses. Keep these ids: the scoring and plotting code keys on them.
system_id() {
    case "$1" in
        duet)          echo A3ctcaT ;;
        dense)         echo D1 ;;
        no_draft)      echo A1 ;;
        ss_finetuned)  echo S1 ;;
        ss_scratch)    echo S-scratch ;;
        wholesong)     echo WSfv4 ;;
        amt)           echo AMT ;;
        *) echo "unknown system '$1' (duet dense no_draft ss_finetuned ss_scratch wholesong amt)" >&2; return 1 ;;
    esac
}
ALL_SYSTEMS="duet dense no_draft ss_finetuned ss_scratch wholesong amt"

# Number of visible GPUs (CUDA_VISIBLE_DEVICES, else what torch sees).
n_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .
    else
        python -c "import torch; print(max(torch.cuda.device_count(), 1))"
    fi
}

# need <path> <hint>: stop with a hint when a required input is missing.
need() { [[ -e "$1" ]] || { echo "ERROR: $1 not found. $2" >&2; exit 1; }; }
