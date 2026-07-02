#!/bin/bash
# SLURM batch script: combined inference on RWC prompts for variants
# A.1 (DuetAttn), A.3 (DuetBlockDiffusion), C.1 (DuetRehearsal), and
# C.2 (DuetPrefix).
#
#   A.1  M2CIntraCrossAttn       -- 5 modes (co, mel2chord, chord2mel, mel_only, chord_only)
#   A.3  M2CDuetBlockDiffusion   -- 5 modes via parallel-diffusion refinement at the query slots
#                                   (supersedes A.2 M2CDuetBlockAttn; A.2's query slots were
#                                   trained only in the fully-masked regime and collapsed at
#                                   inference. A.3 trains across all noise levels and runs
#                                   K+1 refinement passes per frame for mutual within-frame
#                                   conditioning.)
#   C.1  M2CDuetRehearsal        -- drum -> nondrum only (conditional-generation variant)
#   C.2  M2CDuetPrefix           -- drum -> nondrum only (prefix-LM conditional variant)
#
# Two GPUs, two parallel lanes:
#   GPU 0: A.1 then A.3   (the 5-mode co-generation variants)
#   GPU 1: C.1 then C.2   (the drum->nondrum conditional variants)
# Both lanes run concurrently; the job ends when the slower lane finishes.
# Lane logs are written to <OUTPUT_ROOT>/lane_A.log and lane_C.log (and
# echoed into the main job log at the end).
#
# Required env (override on the sbatch line via --export=ALL,KEY=val):
#   CKPT_BASE        ckpt for A.1 (M2CIntraCrossAttn).         default below.
#   CKPT_DIFFUSION   ckpt for A.3 (M2CDuetBlockDiffusion).     default below.
#   CKPT_REHEARSAL   ckpt for C.1 (M2CDuetRehearsal).          default below.
#   CKPT_PREFIX      ckpt for C.2 (M2CDuetPrefix).             default below.
#   CKPT_* may be either a specific .ckpt file or a directory; if a
#   directory, the inference script auto-selects the val_loss-tagged
#   ckpt with the smallest val_loss (falling back to last.ckpt).
#
#   SKIP_BASE / SKIP_DIFFUSION / SKIP_REHEARSAL / SKIP_PREFIX
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
#SBATCH --gres=gpu:2
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
CKPT_DIFFUSION="${CKPT_DIFFUSION:-ckpt/m2c_duet_block_diffusion_v1.0_large_gnl12_K4_drumnondrum_batch_16_schedule/}"
CKPT_REHEARSAL="${CKPT_REHEARSAL:-ckpt/m2c_duet_rehearsal_v1.0_large_gnl12_drumnondrum_batch_16_schedule/}"
CKPT_PREFIX="${CKPT_PREFIX:-ckpt/m2c_duet_prefix_v1.0_large_gnl12_drumnondrum_batch_16_schedule/}"

SKIP_BASE="${SKIP_BASE:-0}"
SKIP_DIFFUSION="${SKIP_DIFFUSION:-0}"
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
echo "RWC inference for variants A.1 (DuetAttn), A.3 (DuetBlockDiffusion), C.1 (DuetRehearsal), C.2 (DuetPrefix)"
echo "MEL_FOLDER (drum)    = $MEL_FOLDER"
echo "CHORD_FOLDER(nondrum)= $CHORD_FOLDER"
echo "OUTPUT_ROOT          = $OUTPUT_ROOT"
echo "PROMPT_LENGTH=$PROMPT_LENGTH  GEN_LENGTH=$GEN_LENGTH  TEMPERATURE=$TEMPERATURE"
echo "MAX_POLYPHONY=$MAX_POLYPHONY  MIN_CHORD_TOKENS_BEFORE_EOS=$MIN_CHORD_TOKENS_BEFORE_EOS"
echo "MODES=$MODES"
if [[ -n "${MAX_SONGS:-}" ]]; then echo "MAX_SONGS=$MAX_SONGS"; fi
echo "Skips: SKIP_BASE=$SKIP_BASE SKIP_DIFFUSION=$SKIP_DIFFUSION SKIP_REHEARSAL=$SKIP_REHEARSAL SKIP_PREFIX=$SKIP_PREFIX"
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

# --- two parallel lanes, one GPU each ---------------------------------
# Lane A (GPU 0): the 5-mode co-generation variants, sequentially.
lane_A() {
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

    # A.3 M2CDuetBlockDiffusion -- parallel-diffusion refinement at the
    # query slots. Supersedes A.2 (whose query slots were trained only in
    # the fully-masked regime and collapsed at inference). The combined
    # script runs K+1 refinement passes per generated frame.
    if [[ "$SKIP_DIFFUSION" != "1" ]]; then
        run_one_variant \
            "duet_block_diffusion" \
            "cp_transformer_m2c_duet_block_diffusion_combined.py" \
            "$CKPT_DIFFUSION" \
            "duet_block_diffusion"
    else
        echo "[duet_block_diffusion] skipped (SKIP_DIFFUSION=1)"
    fi
}

# Lane C (GPU 1): the drum->nondrum conditional variants, sequentially.
lane_C() {
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
}

mkdir -p "$OUTPUT_ROOT"
LOG_A="$OUTPUT_ROOT/lane_A.log"
LOG_C="$OUTPUT_ROOT/lane_C.log"

# CUDA_VISIBLE_DEVICES indices are relative to the job's 2-GPU allocation.
echo "[lanes] launching lane A (A.1, A.3) on GPU 0 -> $LOG_A"
echo "[lanes] launching lane C (C.1, C.2) on GPU 1 -> $LOG_C"
( export CUDA_VISIBLE_DEVICES=0; lane_A ) >"$LOG_A" 2>&1 &
PID_A=$!
( export CUDA_VISIBLE_DEVICES=1; lane_C ) >"$LOG_C" 2>&1 &
PID_C=$!

# Wait for both; don't let set -e kill the script before both finish.
RC_A=0; RC_C=0
wait "$PID_A" || RC_A=$?
wait "$PID_C" || RC_C=$?

echo "================================================================"
echo "----- lane A log (A.1, A.3) -----"
cat "$LOG_A" || true
echo "----- lane C log (C.1, C.2) -----"
cat "$LOG_C" || true
echo "================================================================"
echo "Lane exit codes: A=$RC_A  C=$RC_C"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Outputs under $OUTPUT_ROOT"
if [[ "$RC_A" -ne 0 || "$RC_C" -ne 0 ]]; then
    exit 1
fi
