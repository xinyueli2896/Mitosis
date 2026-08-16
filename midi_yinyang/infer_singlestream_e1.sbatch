#!/bin/bash
# SLURM: single-stream continuation in the layout E1 SCORES.
#
# eval_e1 runs its S1/S0/S-scratch rows through cp_transformer_inference
# with --save-name, producing temp/<SAVE_NAME>/ in the 'single' manifest
# layout. This script does exactly that one step, so a single-stream
# system can be generated on its own and folded into an EXISTING E1 run
# without re-running every other system:
#
#   1. this script            -> temp/<SAVE_NAME>/
#   2. eval_e1 with SKIP_INFER=1, OUT_ROOT=<the old run's root>,
#      S1_OUT_NAME=<that run's S1 dir name>, SSCRATCH_OUT_NAME=<SAVE_NAME>
#      -> rescoring only, no GPU generation
#
# NOTE the prompt folder: single-stream systems read MERGED,
# program-tagged midis (both streams in one file, chord on program 48).
# Every E1 run stages them at <OUT_ROOT>/prompts/merged_tagged -- reuse
# that folder so this system sees byte-identical prompts to the rest.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT           REQUIRED. Run dir (best-val auto-resolved) or file.
#   MIDI_FOLDER    REQUIRED. Merged tagged prompts (see note above).
#   SAVE_NAME      REQUIRED. Output lands in temp/<SAVE_NAME>/.
#   PROMPT_LENGTH  default 96 (6 bars).
#   GEN_LENGTH     default 416 (TOTAL frames incl. prompt).
#   N_SAMPLES      default 3.
#   MAX_POLYPHONY  default 16 (S1/S0/S-scratch are cp16 merged models).
#   MAX_SONGS      default unset (all).
#
# Example (S-scratch, reusing an earlier E1 run's prompts):
#   sbatch --export=ALL,\
# CKPT=ckpt/cp_transformer_v0.42_size1_pop909scratch_cp16_batch_12_schedule/,\
# MIDI_FOLDER=temp/e1_melchord_165497/prompts/merged_tagged,\
# SAVE_NAME=e1_melchord_Sscratch_p96 \
#       midi_yinyang/infer_singlestream_e1.sbatch

#SBATCH --job-name=infer_singlestream_e1
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
MIDI_FOLDER="${MIDI_FOLDER:-}"
SAVE_NAME="${SAVE_NAME:-}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

for v in CKPT MIDI_FOLDER SAVE_NAME; do
    if [[ -z "${!v}" ]]; then
        echo "ERROR: $v is required."; exit 1
    fi
done
[[ -e "$CKPT" ]] || { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
[[ -d "$MIDI_FOLDER" ]] || { echo "ERROR: prompt folder not found: $MIDI_FOLDER"; exit 1; }
N_PROMPTS=$(find "$MIDI_FOLDER" -maxdepth 1 \( -name '*.mid' -o -name '*.MID' \) | wc -l)
[[ "$N_PROMPTS" -gt 0 ]] || { echo "ERROR: no midis in $MIDI_FOLDER"; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "SINGLE-STREAM CONTINUATION (E1 'single' layout)"
echo "CKPT=$CKPT"
echo "MIDI_FOLDER=$MIDI_FOLDER  ($N_PROMPTS prompts)"
echo "SAVE_NAME=$SAVE_NAME  -> temp/$SAVE_NAME/"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "MAX_POLYPHONY=$MAX_POLYPHONY"
echo "================================================================"

python check_ckpts.py "$CKPT" || {
    echo "ERROR: unreadable checkpoint."; exit 1; }

EXTRA_ARGS=()
[[ -n "${MAX_SONGS:-}" ]] && EXTRA_ARGS+=(--max-songs "$MAX_SONGS")

python cp_transformer_inference.py \
    --ckpt "$CKPT" \
    --midi-folder "$MIDI_FOLDER" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples "$N_SAMPLES" \
    --max-polyphony "$MAX_POLYPHONY" \
    --save-name "$SAVE_NAME" \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Done -> temp/$SAVE_NAME"
echo "Score it into an existing E1 run WITHOUT regenerating anything:"
echo "  sbatch --export=ALL,TASK=melchord,SKIP_INFER=1,\\"
echo "OUT_ROOT=<that run's temp/e1_melchord_JOBID>,\\"
echo "S1_OUT_NAME=<that run's S1 dir name>,\\"
echo "SSCRATCH_OUT_NAME=$SAVE_NAME,\\"
echo "PROMPT_LENGTH=$PROMPT_LENGTH,GEN_LENGTH=$GEN_LENGTH,RESULT_TAG=<tag> \\"
echo "      midi_yinyang/eval_e1.sbatch"
echo "================================================================"
