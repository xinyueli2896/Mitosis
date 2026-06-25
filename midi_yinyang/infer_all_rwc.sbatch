#!/bin/bash
# SLURM batch script: combined inference on RWC prompts for variants #2, #3, #4.
#   #2 M2CIntraCrossAttn (DuetAttn base)
#   #3 M2CIntraCrossAttnRecon (loss-side conditioning baseline)
#   #4 M2CDuetBlockAttn (fair-looking fix, Option B query-slot decoding)
#
# Single-GPU, runs the three inferences sequentially in one job.
#
# Required env (override on the sbatch line via --export=ALL,KEY=val):
#   CKPT_BASE      ckpt path for #2 (M2CIntraCrossAttn).         optional, default below.
#   CKPT_RECON     ckpt path for #3 (M2CIntraCrossAttnRecon).    optional, default below.
#   CKPT_BLOCK     ckpt path for #4 (M2CDuetBlockAttn).          optional, default below.
#   SKIP_BASE / SKIP_RECON / SKIP_BLOCK   set to 1 to skip a variant.
#
# Common optional env:
#   MEL_FOLDER     drum prompts.  default: input/rwc_test_prompts_split/drum
#   CHORD_FOLDER   nondrum prompts. default: input/rwc_test_prompts_split/nondrum
#   OUTPUT_ROOT    where all outputs go. default: temp/rwc_infer_<JOBID>
#   MAX_SONGS      cap songs per variant. default: unset (all).
#   PROMPT_LENGTH  default 64.
#   GEN_LENGTH     default 384.
#   TEMPERATURE    default 0.9.
#   MAX_POLYPHONY  default 16.
#   MIN_CHORD_TOKENS_BEFORE_EOS  rescue knob for EOS-collapsed ckpts. default 0.
#   MODES          space-separated subset; default: all five.
#
# Output layout:
#   <OUTPUT_ROOT>/
#       intra_cross_attn/<song>/<mode>.mid
#       intra_cross_attn_recon/<song>/<mode>.mid
#       duet_block/<song>/<mode>.mid

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

CKPT_BASE="${CKPT_BASE:-ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_drumnondrum_batch_16_schedule/last.ckpt}"
CKPT_RECON="${CKPT_RECON:-ckpt/m2c_intra_cross_attn_recon_v1.0_large_gnl12_drumnondrum_batch_16_schedule/last.ckpt}"
CKPT_BLOCK="${CKPT_BLOCK:-ckpt/m2c_duet_block_v1.0_large_gnl12_drumnondrum_batch_16_schedule/last.ckpt}"

SKIP_BASE="${SKIP_BASE:-0}"
SKIP_RECON="${SKIP_RECON:-0}"
SKIP_BLOCK="${SKIP_BLOCK:-0}"

MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"

OUTPUT_ROOT="${OUTPUT_ROOT:-temp/rwc_infer_${SLURM_JOB_ID}}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
MIN_CHORD_TOKENS_BEFORE_EOS="${MIN_CHORD_TOKENS_BEFORE_EOS:-0}"
MODES="${MODES:-co mel2chord chord2mel mel_only chord_only}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "RWC inference for variants #2, #3, #4"
echo "MEL_FOLDER (drum)    = $MEL_FOLDER"
echo "CHORD_FOLDER(nondrum)= $CHORD_FOLDER"
echo "OUTPUT_ROOT          = $OUTPUT_ROOT"
echo "PROMPT_LENGTH=$PROMPT_LENGTH  GEN_LENGTH=$GEN_LENGTH  TEMPERATURE=$TEMPERATURE"
echo "MAX_POLYPHONY=$MAX_POLYPHONY  MIN_CHORD_TOKENS_BEFORE_EOS=$MIN_CHORD_TOKENS_BEFORE_EOS"
echo "MODES=$MODES"
if [[ -n "${MAX_SONGS:-}" ]]; then echo "MAX_SONGS=$MAX_SONGS"; fi
echo "Skips: SKIP_BASE=$SKIP_BASE SKIP_RECON=$SKIP_RECON SKIP_BLOCK=$SKIP_BLOCK"
echo "----------------------------------------------------------------"
nvidia-smi | head -n 20 || true
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "${MAX_SONGS:-}" ]]; then
    EXTRA_ARGS+=(--max-songs "$MAX_SONGS")
fi

run_one_variant() {
    local label="$1"
    local script="$2"
    local ckpt="$3"
    local out_subdir="$4"

    if [[ ! -f "$ckpt" ]]; then
        echo "[$label] ERROR: ckpt not found at $ckpt; skipping."
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

# #2 M2CIntraCrossAttn
if [[ "$SKIP_BASE" != "1" ]]; then
    run_one_variant \
        "intra_cross_attn" \
        "cp_transformer_m2c_intra_cross_attn_combined.py" \
        "$CKPT_BASE" \
        "intra_cross_attn"
else
    echo "[intra_cross_attn] skipped (SKIP_BASE=1)"
fi

# #3 M2CIntraCrossAttnRecon -- same script as #2, recon ckpts load directly.
if [[ "$SKIP_RECON" != "1" ]]; then
    run_one_variant \
        "intra_cross_attn_recon" \
        "cp_transformer_m2c_intra_cross_attn_combined.py" \
        "$CKPT_RECON" \
        "intra_cross_attn_recon"
else
    echo "[intra_cross_attn_recon] skipped (SKIP_RECON=1)"
fi

# #4 M2CDuetBlockAttn -- Option B inference (query-slot decoding).
if [[ "$SKIP_BLOCK" != "1" ]]; then
    run_one_variant \
        "duet_block" \
        "cp_transformer_m2c_duet_block_combined.py" \
        "$CKPT_BLOCK" \
        "duet_block"
else
    echo "[duet_block] skipped (SKIP_BLOCK=1)"
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_ROOT"
