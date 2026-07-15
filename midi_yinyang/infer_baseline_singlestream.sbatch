#!/bin/bash
# SLURM batch script: single-stream co-generation BASELINE.
# Runs the base pretrained CP transformer on UNSPLIT drum+nondrum prompt
# MIDIs, treating everything as one merged stream (no modalities).
# The lower anchor for the A.1 / A.3 co-generation comparison.
#
# Env knobs (override via --export=ALL,KEY=val):
#   CKPT           pretrained single-stream ckpt. Default below.
#   MIDI_FOLDER    UNSPLIT prompt midis. Default: input/rwc_test_prompts
#   OUTPUT_DIR     default: temp/baseline_singlestream_<JOBID>
#   PROMPT_LENGTH  default 64.
#   GEN_LENGTH     default 384 (TOTAL frames incl. prompt -- matches m2c).
#   TEMPERATURE    default 1.0.
#   MAX_POLYPHONY  default 16.
#   N_SAMPLES      default 1.
#   MAX_SONGS      default unset (all).

#SBATCH --job-name=mitosis_baseline_singlestream
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

CKPT="${CKPT:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
MIDI_FOLDER="${MIDI_FOLDER:-input/rwc_test_prompts}"
OUTPUT_DIR="${OUTPUT_DIR:-temp/baseline_singlestream_${SLURM_JOB_ID}}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
N_SAMPLES="${N_SAMPLES:-1}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: ckpt not found at $CKPT"
    exit 1
fi
if [[ ! -d "$MIDI_FOLDER" ]]; then
    echo "ERROR: MIDI_FOLDER $MIDI_FOLDER not found. This baseline needs"
    echo "the UNSPLIT prompt midis (drums + pitched in one file), e.g."
    echo "input/rwc_test_prompts -- NOT the _split/drum|nondrum folders."
    exit 1
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Single-stream baseline (pretrained CP transformer, merged stream)"
echo "CKPT=$CKPT"
echo "MIDI_FOLDER=$MIDI_FOLDER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PROMPT_LENGTH=$PROMPT_LENGTH  GEN_LENGTH=$GEN_LENGTH (total)"
echo "TEMPERATURE=$TEMPERATURE  MAX_POLYPHONY=$MAX_POLYPHONY  N_SAMPLES=$N_SAMPLES"
if [[ -n "${MAX_SONGS:-}" ]]; then echo "MAX_SONGS=$MAX_SONGS"; fi
echo "----------------------------------------------------------------"
nvidia-smi | head -n 20 || true
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "${MAX_SONGS:-}" ]]; then
    EXTRA_ARGS+=(--max-songs "$MAX_SONGS")
fi

python cp_transformer_baseline_singlestream.py \
    --ckpt "$CKPT" \
    --midi-folder "$MIDI_FOLDER" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-length "$PROMPT_LENGTH" \
    --gen-length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --max-polyphony "$MAX_POLYPHONY" \
    --n-samples "$N_SAMPLES" \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_DIR"
