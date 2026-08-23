#!/bin/bash
# SLURM batch script: M2CDuetRehearsal (C.1) inference -- drum -> nondrum only.
#
# Required env (override via --export=ALL,KEY=val):
#   CKPT           ckpt path. Required.
#
# Optional env:
#   DRUM_FOLDER     default: input/rwc_test_prompts_split/drum
#   NONDRUM_FOLDER  default: input/rwc_test_prompts_split/nondrum (used for
#                   optional nondrum-prompt teacher-forcing; pass an empty
#                   string to disable).
#   OUTPUT_DIR      default: temp/duet_rehearsal_<JOBID>
#   GEN_LENGTH      default 384
#   PROMPT_LENGTH   default 64 (nondrum-prompt frames before sampling)
#   TEMPERATURE     default 0.9
#   MAX_POLYPHONY   default 16 (pass 8 for the melchord cp8 ckpts)
#   MAX_SONGS       default unset (process all)
#   N_SAMPLES       default 1. Each sample is a full AR decode over a 3T
#                   sequence with no KV cache, so cost is linear in this.
#   MODE_NAME       output folder/file label; default drum2nondrum. On the
#                   MELCHORD task pass mel2chord (or chord2mel for a
#                   reversed ckpt) so the tree names the direction it holds
#                   -- the --drum/--nondrum args are mod_a/mod_b SLOTS, not
#                   drum semantics.
#   SWAP_STREAM_NAMES  1 to name the mod_a track CHORD and mod_b MELODY
#                   (required when scoring a reversed melchord ckpt).
#   ALLOW_PARTIAL_LOAD  1 to proceed when the ckpt is missing parameters.
#                   OFF by default: a missing key stays at random init and
#                   the run still writes midi.
#
# Example (melchord, the shift-1 conditional ckpt, Nottingham prompts):
#   sbatch --export=ALL,\
# CKPT=ckpt/m2c_duet_rehearsal_v1.0_large_gnl12_melchord_nottingham_ps2_sh1_tgt_yyho_batch_8_schedule/,\
# DRUM_FOLDER=input/not_split/melody,NONDRUM_FOLDER=input/not_split/chord,\
# MODE_NAME=mel2chord,MAX_POLYPHONY=8,MAX_SONGS=3 \
#       midi_yinyang/infer_duet_rehearsal.sbatch

#SBATCH --job-name=mitosis_duet_rehearsal_infer
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
OUTPUT_DIR="${OUTPUT_DIR:-temp/duet_rehearsal_${SLURM_JOB_ID}}"
GEN_LENGTH="${GEN_LENGTH:-384}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
N_SAMPLES="${N_SAMPLES:-1}"
MODE_NAME="${MODE_NAME:-drum2nondrum}"
SWAP_STREAM_NAMES="${SWAP_STREAM_NAMES:-0}"
ALLOW_PARTIAL_LOAD="${ALLOW_PARTIAL_LOAD:-0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Variant: M2CDuetRehearsal (C.1) -- drum -> nondrum"
echo "CKPT=$CKPT"
echo "DRUM_FOLDER=$DRUM_FOLDER"
echo "NONDRUM_FOLDER=$NONDRUM_FOLDER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GEN_LENGTH=$GEN_LENGTH  PROMPT_LENGTH=$PROMPT_LENGTH"
echo "TEMPERATURE=$TEMPERATURE  MAX_POLYPHONY=$MAX_POLYPHONY  N_SAMPLES=$N_SAMPLES"
echo "MODE_NAME=$MODE_NAME  SWAP_STREAM_NAMES=$SWAP_STREAM_NAMES"
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

python cp_transformer_m2c_duet_rehearsal_inference.py \
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
