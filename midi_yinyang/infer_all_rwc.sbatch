#!/bin/bash
# SLURM batch script: combined inference on RWC prompts for variants
# A.1 (DuetAttn), A.2 (DuetBlock), C.1 (DuetRehearsal), and C.2 (DuetPrefix).
#
#   A.1  M2CIntraCrossAttn  -- 5 modes (co, mel2chord, chord2mel, mel_only, chord_only)
#   A.2  M2CDuetBlockAttn   -- 5 modes via Option A AR-clean-stream decoding
#   C.1  M2CDuetRehearsal   -- drum -> nondrum only (conditional-generation variant)
#   C.2  M2CDuetPrefix      -- drum -> nondrum only (prefix-LM conditional variant)
#
# Single-GPU, runs the inferences sequentially in one job.
#
# Required env (override on the sbatch line via --export=ALL,KEY=val):
#   CKPT_BASE       ckpt for A.1 (M2CIntraCrossAttn).      default below.
#   CKPT_BLOCK      ckpt for A.2 (M2CDuetBlockAttn).       default below.
#   CKPT_REHEARSAL  ckpt for C.1 (M2CDuetRehearsal).       default below.
#   CKPT_PREFIX     ckpt for C.2 (M2CDuetPrefix).          default below.
#   CKPT_* may be either a specific .ckpt file or a directory; if a
#   directory, the inference script auto-selects the val_loss-tagged
#   ckpt with the smallest val_loss (falling back to last.ckpt).
#
#   SKIP_BASE / SKIP_BLOCK / SKIP_REHEARSAL / SKIP_PREFIX
#       set to 1 to skip a variant.
#
# Common optional env:
#   MEL_FOLDER     drum prompts.    default: input/rwc_test_prompts_split/drum
#   CHORD_FOLDER   nondrum prompts. default: input/rwc_test_prompts_split/nondrum
#   OUTPUT_ROOT    where all outputs go. default: temp/rwc_infer_<JOBID>
#   MAX_SONGS      cap songs per variant. default: unset (all).
#   PROMPT_LENGTH  default 64.
#   GEN_LENGTH     default 384.
#   TEMPERATURE    default 1.0.
#   MAX_POLYPHONY  default 16.
#   MIN_CHORD_TOKENS_BEFORE_EOS  rescue knob for EOS-collapsed ckpts. default 0.
#   MODES          space-separated subset for A.1/A.2; default: all five.
#                  (C.1 ignores this -- only drum2nondrum is supported.)
#
# Output layout:
#   <OUTPUT_ROOT>/
#       intra_cross_attn/<song>/<mode>.mid     (A.1, 5 modes)
#       duet_block/<song>/<mode>.mid           (A.2, 5 modes)
#       duet_rehearsal/<song>/drum2nondrum_temp<T>.mid   (C.1)
#       duet_prefix/<song>/drum2nondrum_temp<T>.mid      (C.2)

#SBATCH --job-name=mitosis_infer_rwc_all
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=08:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

# Default to the run DIRECTORY; resolve_best_ckpt in the Python loaders
# automatically scans for *.val_loss=*.ckpt and picks the smallest val_loss,
# falling back to last.ckpt if no val_loss-tagged ckpts exist.
CKPT_BASE="${CKPT_BASE:-ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_drumnondrum_batch_16_schedule/}"
CKPT_BLOCK="${CKPT_BLOCK:-ckpt/m2c_duet_block_v1.0_large_gnl12_drumnondrum_batch_16_schedule/}"
CKPT_REHEARSAL="${CKPT_REHEARSAL:-ckpt/m2c_duet_rehearsal_v1.0_large_gnl12_drumnondrum_batch_16_schedule/}"
CKPT_PREFIX="${CKPT_PREFIX:-ckpt/m2c_duet_prefix_v1.0_large_gnl12_drumnondrum_batch_16_schedule/}"

SKIP_BASE="${SKIP_BASE:-0}"
SKIP_BLOCK="${SKIP_BLOCK:-0}"
SKIP_REHEARSAL="${SKIP_REHEARSAL:-0}"
SKIP_PREFIX="${SKIP_PREFIX:-0}"

MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"

OUTPUT_ROOT="${OUTPUT_ROOT:-temp/rwc_infer_${SLURM_JOB_ID}}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
MIN_CHORD_TOKENS_BEFORE_EOS="${MIN_CHORD_TOKENS_BEFORE_EOS:-0}"
MODES="${MODES:-co mel2chord chord2mel mel_only chord_only}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "RWC inference for variants A.1 (DuetAttn), A.2 (DuetBlock), C.1 (DuetRehearsal)"
echo "MEL_FOLDER (drum)    = $MEL_FOLDER"
echo "CHORD_FOLDER(nondrum)= $CHORD_FOLDER"
echo "OUTPUT_ROOT          = $OUTPUT_ROOT"
echo "PROMPT_LENGTH=$PROMPT_LENGTH  GEN_LENGTH=$GEN_LENGTH  TEMPERATURE=$TEMPERATURE"
echo "MAX_POLYPHONY=$MAX_POLYPHONY  MIN_CHORD_TOKENS_BEFORE_EOS=$MIN_CHORD_TOKENS_BEFORE_EOS"
echo "MODES=$MODES"
if [[ -n "${MAX_SONGS:-}" ]]; then echo "MAX_SONGS=$MAX_SONGS"; fi
echo "Skips: SKIP_BASE=$SKIP_BASE SKIP_BLOCK=$SKIP_BLOCK SKIP_REHEARSAL=$SKIP_REHEARSAL SKIP_PREFIX=$SKIP_PREFIX"
echo "----------------------------------------------------------------"
nvidia-smi | head -n 20 || true
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "${MAX_SONGS:-}" ]]; then
    EXTRA_ARGS+=(--max-songs "$MAX_SONGS")
fi

run_one_variant() {
    # 5-mode CLI: A.1 (intra_cross_attn) and A.2 (duet_block).
    local label="$1"
    local script="$2"
    local ckpt="$3"
    local out_subdir="$4"

    if [[ ! -e "$ckpt" ]]; then
        echo "[$label] ERROR: ckpt path $ckpt not found (file or dir); skipping."
        return
    fi
    local out_dir="${OUTPUT_ROOT}/${out_subdir}"
    echo "================================================================"
    echo "[$label] ckpt=$ckpt"
    echo "[$label] output=$out_dir"
    echo "================================================================"
    python "$script" \
        --ckpt "$ckpt" \
        --mel-folder "$MEL_FOLDER" \
        --chord-folder "$CHORD_FOLDER" \
        --output-dir "$out_dir" \
        --modes $MODES \
        --prompt-length "$PROMPT_LENGTH" \
        --gen-length "$GEN_LENGTH" \
        --temperature "$TEMPERATURE" \
        --max-polyphony "$MAX_POLYPHONY" \
        --model-size large \
        --moe-num-experts 4 --moe-topk 2 \
        --min-chord-tokens-before-eos "$MIN_CHORD_TOKENS_BEFORE_EOS" \
        "${EXTRA_ARGS[@]}"
    echo "[$label] done."
}

run_drum2nondrum_variant() {
    # drum->nondrum-only CLI shared by C.1 (duet_rehearsal) and C.2 (duet_prefix).
    local label="$1"
    local script="$2"
    local ckpt="$3"
    local out_subdir="$4"

    if [[ ! -e "$ckpt" ]]; then
        echo "[$label] ERROR: ckpt path $ckpt not found (file or dir); skipping."
        return
    fi
    local out_dir="${OUTPUT_ROOT}/${out_subdir}"
    echo "================================================================"
    echo "[$label] ckpt=$ckpt"
    echo "[$label] output=$out_dir"
    echo "================================================================"
    python "$script" \
        --ckpt "$ckpt" \
        --drum-folder "$MEL_FOLDER" \
        --nondrum-folder "$CHORD_FOLDER" \
        --output-dir "$out_dir" \
        --prompt-length "$PROMPT_LENGTH" \
        --gen-length "$GEN_LENGTH" \
        --temperature "$TEMPERATURE" \
        --max-polyphony "$MAX_POLYPHONY" \
        --model-size large \
        --moe-num-experts 4 --moe-topk 2 \
        --min-chord-tokens-before-eos "$MIN_CHORD_TOKENS_BEFORE_EOS" \
        "${EXTRA_ARGS[@]}"
    echo "[$label] done."
}

# A.1 M2CIntraCrossAttn
if [[ "$SKIP_BASE" != "1" ]]; then
    run_one_variant \
        "intra_cross_attn" \
        "cp_transformer_m2c_intra_cross_attn_combined.py" \
        "$CKPT_BASE" \
        "intra_cross_attn"
else
    echo "[intra_cross_attn] skipped (SKIP_BASE=1)"
fi

# A.2 M2CDuetBlockAttn -- Option B inference (query-slot decoding).
if [[ "$SKIP_BLOCK" != "1" ]]; then
    run_one_variant \
        "duet_block" \
        "cp_transformer_m2c_duet_block_combined.py" \
        "$CKPT_BLOCK" \
        "duet_block"
else
    echo "[duet_block] skipped (SKIP_BLOCK=1)"
fi

# C.1 M2CDuetRehearsal -- drum -> nondrum only.
if [[ "$SKIP_REHEARSAL" != "1" ]]; then
    run_drum2nondrum_variant \
        "duet_rehearsal" \
        "cp_transformer_m2c_duet_rehearsal_inference.py" \
        "$CKPT_REHEARSAL" \
        "duet_rehearsal"
else
    echo "[duet_rehearsal] skipped (SKIP_REHEARSAL=1)"
fi

# C.2 M2CDuetPrefix -- drum -> nondrum only (prefix-LM architecture).
if [[ "$SKIP_PREFIX" != "1" ]]; then
    run_drum2nondrum_variant \
        "duet_prefix" \
        "cp_transformer_m2c_duet_prefix_inference.py" \
        "$CKPT_PREFIX" \
        "duet_prefix"
else
    echo "[duet_prefix] skipped (SKIP_PREFIX=1)"
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_ROOT"
