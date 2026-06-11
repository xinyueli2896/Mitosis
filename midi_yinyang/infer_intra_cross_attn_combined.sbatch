#!/bin/bash
# SLURM batch script: combined inference for M2CIntraCrossAttn. Runs all
# five generation modes (co, mel2chord, chord2mel, mel_only, chord_only)
# on every paired (mod_a, mod_b) midi from two input folders and writes
# one .mid per (song, mode) under OUTPUT_DIR/<song_id>/<mode>.mid.
#
# Single-GPU inference -- no DDP needed for generation.
#
# Required env (override on the sbatch line via --export=ALL,KEY=val):
#   CKPT           path to the trained ckpt (relative to REPO_DIR/midi_yinyang
#                  or absolute). Required.
#
# Common optional env:
#   TASK           drumnondrum (default) or melchord. Used only to pick
#                  default MEL_FOLDER and CHORD_FOLDER if those are unset;
#                  the inference itself is task-agnostic.
#   MEL_FOLDER     folder of mod_a prompt .mid files.
#   CHORD_FOLDER   folder of mod_b prompt .mid files.
#   OUTPUT_DIR     where (song, mode) midis get written.
#                  Default: temp/intra_cross_attn_combined_<JOBID>.
#   MAX_SONGS      cap the number of songs processed. Default: unset (all).
#   PROMPT_LENGTH  default 100.
#   GEN_LENGTH     default 384.
#   TEMPERATURE    default 1.0.
#   MAX_POLYPHONY  default 16 (must match training preprocess).
#   MIN_CHORD_TOKENS_BEFORE_EOS  rescue knob for EOS-collapsed ckpts. 0=off.
#   MODES          space-separated subset of modes; default: all five.
#
# Examples:
#   sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt midi_yinyang/infer_intra_cross_attn_combined.sbatch
#   sbatch --export=ALL,CKPT=...,TASK=melchord midi_yinyang/infer_intra_cross_attn_combined.sbatch
#   sbatch --export=ALL,CKPT=...,MAX_SONGS=20,MODES="co mel2chord" \
#          midi_yinyang/infer_intra_cross_attn_combined.sbatch

#SBATCH --job-name=mitosis_icx_infer
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
REPO_DIR=/l/users/xinyue.li/Mitosis

if [[ -z "${CKPT:-}" ]]; then
    echo "ERROR: CKPT not set. Pass via --export=ALL,CKPT=path/to/ckpt"
    exit 1
fi

TASK="${TASK:-drumnondrum}"
case "$TASK" in
    drumnondrum)
        MEL_FOLDER="${MEL_FOLDER:-input/lamd_test_prompts_split/drum}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/lamd_test_prompts_split/nondrum}"
        ;;
    melchord)
        MEL_FOLDER="${MEL_FOLDER:-input/pop909_test_prompts_split/melody}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_test_prompts_split/chord}"
        ;;
    *)
        echo "ERROR: unknown TASK=$TASK (expected drumnondrum or melchord)"
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-temp/intra_cross_attn_combined_${SLURM_JOB_ID}}"
PROMPT_LENGTH="${PROMPT_LENGTH:-100}"
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
echo "TASK=$TASK"
echo "CKPT=$CKPT"
echo "MEL_FOLDER  (mod_a) =$MEL_FOLDER"
echo "CHORD_FOLDER(mod_b) =$CHORD_FOLDER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PROMPT_LENGTH=$PROMPT_LENGTH  GEN_LENGTH=$GEN_LENGTH"
echo "TEMPERATURE=$TEMPERATURE  MAX_POLYPHONY=$MAX_POLYPHONY"
echo "MIN_CHORD_TOKENS_BEFORE_EOS=$MIN_CHORD_TOKENS_BEFORE_EOS"
echo "MODES=$MODES"
if [[ -n "${MAX_SONGS:-}" ]]; then echo "MAX_SONGS=$MAX_SONGS"; fi
echo "----------------------------------------------------------------"
nvidia-smi | head -n 20
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "${MAX_SONGS:-}" ]]; then
    EXTRA_ARGS+=(--max-songs "$MAX_SONGS")
fi

python cp_transformer_m2c_intra_cross_attn_combined.py \
    --ckpt "$CKPT" \
    --mel-folder "$MEL_FOLDER" \
    --chord-folder "$CHORD_FOLDER" \
    --output-dir "$OUTPUT_DIR" \
    --modes $MODES \
    --prompt-length "$PROMPT_LENGTH" \
    --gen-length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --max-polyphony "$MAX_POLYPHONY" \
    --model-size large \
    --moe-num-experts 4 --moe-topk 2 \
    --min-chord-tokens-before-eos "$MIN_CHORD_TOKENS_BEFORE_EOS" \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_DIR"
