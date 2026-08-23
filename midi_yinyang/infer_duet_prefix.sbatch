#!/bin/bash
# SLURM batch script: M2CDuetPrefix (C.2) inference -- drum -> nondrum only.
# Same shape as infer_duet_rehearsal.sbatch; only the inference script
# (and therefore the model class loaded) differs.

#SBATCH --job-name=mitosis_duet_prefix_infer
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

CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
    echo "ERROR: CKPT not set. Pass via --export=ALL,CKPT=path/to/ckpt"
    exit 1
fi

DRUM_FOLDER="${DRUM_FOLDER:-input/rwc_test_prompts_split/drum}"
NONDRUM_FOLDER="${NONDRUM_FOLDER:-input/rwc_test_prompts_split/nondrum}"
OUTPUT_DIR="${OUTPUT_DIR:-temp/duet_prefix_${SLURM_JOB_ID}}"
GEN_LENGTH="${GEN_LENGTH:-384}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
# See infer_duet_rehearsal.sbatch for what these mean; --drum/--nondrum
# are mod_a/mod_b SLOTS, so MODE_NAME=mel2chord labels a melchord run's
# output tree with the direction it actually holds.
N_SAMPLES="${N_SAMPLES:-1}"
MODE_NAME="${MODE_NAME:-drum2nondrum}"
SWAP_STREAM_NAMES="${SWAP_STREAM_NAMES:-0}"
ALLOW_PARTIAL_LOAD="${ALLOW_PARTIAL_LOAD:-0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Variant: M2CDuetPrefix (C.2) -- drum -> nondrum"
echo "CKPT=$CKPT"
echo "DRUM_FOLDER=$DRUM_FOLDER"
echo "NONDRUM_FOLDER=$NONDRUM_FOLDER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GEN_LENGTH=$GEN_LENGTH  PROMPT_LENGTH=$PROMPT_LENGTH"
echo "TEMPERATURE=$TEMPERATURE  MAX_POLYPHONY=$MAX_POLYPHONY"
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
[[ "$SWAP_STREAM_NAMES" == "1" ]] && EXTRA_ARGS+=(--swap-stream-names)
[[ "$ALLOW_PARTIAL_LOAD" == "1" ]] && EXTRA_ARGS+=(--allow-partial-load)

python cp_transformer_m2c_duet_prefix_inference.py \
    --ckpt "$CKPT" \
    --drum-folder "$DRUM_FOLDER" \
    --output-dir "$OUTPUT_DIR" \
    --gen-length "$GEN_LENGTH" \
    --prompt-length "$PROMPT_LENGTH" \
    --temperature "$TEMPERATURE" \
    --max-polyphony "$MAX_POLYPHONY" \
    --n-samples "$N_SAMPLES" \
    --mode-name "$MODE_NAME" \
    --model-size large \
    --moe-num-experts 4 --moe-topk 2 \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_DIR"
