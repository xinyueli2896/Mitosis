#!/bin/bash
# SLURM batch script: standalone inference for A.3 M2CDuetBlockDiffusion.
# Runs all 5 generation modes (co, mel2chord, chord2mel, mel_only,
# chord_only) on every paired (drum, nondrum) midi from two folders via
# parallel-diffusion refinement at the query slots (K+1 forwards per
# generated frame; round one seeds from the AR clean-stream heads).
#
# Single GPU. Expect roughly 5x A.1's wall time per song.
#
# Env knobs (override via --export=ALL,KEY=val):
#   CKPT           ckpt file OR run directory; a directory auto-selects
#                  the smallest-val_loss ckpt (fallback last.ckpt).
#                  Default below points at the standard K=4 run dir.
#   MEL_FOLDER     drum prompts.    default: input/rwc_test_prompts_split/drum
#   CHORD_FOLDER   nondrum prompts. default: input/rwc_test_prompts_split/nondrum
#   OUTPUT_DIR     default: temp/duet_block_diffusion_<JOBID>
#   MAX_SONGS      cap songs. default: unset (all).
#   PROMPT_LENGTH  default 64.
#   GEN_LENGTH     default 384.
#   TEMPERATURE    default 1.0.
#   MAX_POLYPHONY  default 16.
#   MIN_CHORD_TOKENS_BEFORE_EOS  rescue knob for EOS-collapsed ckpts. default 0.
#   MODES          space-separated subset; default: all five.
#
# Examples:
#   sbatch midi_yinyang/infer_duet_block_diffusion.sbatch
#   sbatch --export=ALL,CKPT=ckpt/<run>/,MAX_SONGS=5,MODES="co" \
#          midi_yinyang/infer_duet_block_diffusion.sbatch
#
# Output layout: <OUTPUT_DIR>/<song>/<mode>.mid

#SBATCH --job-name=mitosis_duet_block_diffusion_infer
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

# Prefer the v1.1 (aligned slot-RoPE + self-conditioning) run when it
# exists; fall back to the legacy v1.0 run (decoded with the padding
# rescue, auto-detected from the ckpt).
_V11_DIR="ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_drumnondrum_batch_16_schedule"
_V10_DIR="ckpt/m2c_duet_block_diffusion_v1.0_large_gnl12_K4_drumnondrum_batch_16_schedule"
if [[ -z "${CKPT:-}" ]]; then
    if [[ -d "$_V11_DIR" ]]; then CKPT="$_V11_DIR/"; else CKPT="$_V10_DIR/"; fi
    echo "[ckpt-default] using $CKPT"
fi

MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"
OUTPUT_DIR="${OUTPUT_DIR:-temp/duet_block_diffusion_${SLURM_JOB_ID}}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
MIN_CHORD_TOKENS_BEFORE_EOS="${MIN_CHORD_TOKENS_BEFORE_EOS:-0}"
MODES="${MODES:-co mel2chord chord2mel mel_only chord_only}"
# Samples per (song, mode). 1 (default) keeps the 'duet' layout
# (<song>/<mode>.mid); >1 switches to 'duet_multi'
# (<song>/<mode>/sample_<i>_temp<T>.mid) -- tell build_eval_manifest
# which when scoring.
N_SAMPLES="${N_SAMPLES:-1}"

# Decode-schedule defaults (listening-test winner on the 98k v1.1 ckpt:
# one refinement round, commit annealed to 0.9, 0.95 nucleus). Override
# any of these on the sbatch line; A3_REFINE_STEPS=0 gives pure AR decode.
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-1}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
export A3_TOP_P="${A3_TOP_P:-0.95}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

if [[ ! -e "$CKPT" ]]; then
    echo "ERROR: ckpt path $CKPT not found (file or dir)."
    exit 1
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Variant: A.3 M2CDuetBlockDiffusion (parallel-diffusion refinement)"
echo "CKPT=$CKPT"
echo "MEL_FOLDER (drum)    = $MEL_FOLDER"
echo "CHORD_FOLDER(nondrum)= $CHORD_FOLDER"
echo "OUTPUT_DIR           = $OUTPUT_DIR"
echo "PROMPT_LENGTH=$PROMPT_LENGTH  GEN_LENGTH=$GEN_LENGTH  TEMPERATURE=$TEMPERATURE"
echo "MAX_POLYPHONY=$MAX_POLYPHONY  MIN_CHORD_TOKENS_BEFORE_EOS=$MIN_CHORD_TOKENS_BEFORE_EOS"
echo "MODES=$MODES"
echo "A3_REFINE_STEPS=$A3_REFINE_STEPS  A3_FINAL_TEMP=$A3_FINAL_TEMP  A3_TOP_P=$A3_TOP_P"
if [[ -n "${MAX_SONGS:-}" ]]; then echo "MAX_SONGS=$MAX_SONGS"; fi
echo "----------------------------------------------------------------"
nvidia-smi | head -n 20 || true
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "${MAX_SONGS:-}" ]]; then
    EXTRA_ARGS+=(--max-songs "$MAX_SONGS")
fi

python cp_transformer_m2c_duet_block_diffusion_combined.py \
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
    --n-samples "$N_SAMPLES" \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_DIR"
