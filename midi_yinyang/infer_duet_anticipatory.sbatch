#!/bin/bash
# SLURM batch script: M2CDuetAnticipatory (B.1) inference --
# drum -> nondrum with k-frame drum lookahead.
# Same shape as infer_duet_prefix.sbatch; the anticipation k is
# auto-detected from the ckpt path (_k<N>_), override via
# ANTICIPATION_FRAMES.

#SBATCH --job-name=mitosis_duet_anticipatory_infer
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=04:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

# Directory default: resolve_best_ckpt picks the smallest-val_loss ckpt.
CKPT="${CKPT:-ckpt/m2c_duet_anticipatory_v1.0_large_gnl12_drumnondrum_k16_batch_16_schedule/}"

DRUM_FOLDER="${DRUM_FOLDER:-input/rwc_test_prompts_split/drum}"
NONDRUM_FOLDER="${NONDRUM_FOLDER:-input/rwc_test_prompts_split/nondrum}"
OUTPUT_DIR="${OUTPUT_DIR:-temp/duet_anticipatory_${SLURM_JOB_ID}}"
GEN_LENGTH="${GEN_LENGTH:-384}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
ANTICIPATION_FRAMES="${ANTICIPATION_FRAMES:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

if [[ ! -e "$CKPT" ]]; then
    echo "ERROR: ckpt path $CKPT not found (file or dir)."
    exit 1
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Variant: M2CDuetAnticipatory (B.1) -- drum -> nondrum, k-frame lookahead"
echo "CKPT=$CKPT"
echo "DRUM_FOLDER=$DRUM_FOLDER"
echo "NONDRUM_FOLDER=$NONDRUM_FOLDER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GEN_LENGTH=$GEN_LENGTH  PROMPT_LENGTH=$PROMPT_LENGTH"
echo "TEMPERATURE=$TEMPERATURE  MAX_POLYPHONY=$MAX_POLYPHONY"
if [[ -n "$ANTICIPATION_FRAMES" ]]; then echo "ANTICIPATION_FRAMES=$ANTICIPATION_FRAMES (override)"; fi
if [[ -n "${MAX_SONGS:-}" ]]; then echo "MAX_SONGS=$MAX_SONGS"; fi
echo "----------------------------------------------------------------"
nvidia-smi | head -n 20 || true
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "${MAX_SONGS:-}" ]]; then
    EXTRA_ARGS+=(--max-songs "$MAX_SONGS")
fi
if [[ -n "$NONDRUM_FOLDER" ]]; then
    EXTRA_ARGS+=(--nondrum-folder "$NONDRUM_FOLDER")
fi
if [[ -n "$ANTICIPATION_FRAMES" ]]; then
    EXTRA_ARGS+=(--anticipation-frames "$ANTICIPATION_FRAMES")
fi

python cp_transformer_m2c_duet_anticipatory_inference.py \
    --ckpt "$CKPT" \
    --drum-folder "$DRUM_FOLDER" \
    --output-dir "$OUTPUT_DIR" \
    --gen-length "$GEN_LENGTH" \
    --prompt-length "$PROMPT_LENGTH" \
    --temperature "$TEMPERATURE" \
    --max-polyphony "$MAX_POLYPHONY" \
    --model-size large \
    --moe-num-experts 4 --moe-topk 2 \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_DIR"
