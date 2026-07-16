#!/bin/bash
# SLURM batch script: finetune the pretrained single-stream CP transformer
# on POP909 melody+chord. Runs the whole pipeline, skipping stages whose
# output already exists (FORCE=1 to redo):
#
#   1. merge   merge_melody_chord.py --chord-program 48:
#              POP909-melody + POP909-chord -> POP909-melody-chord-tagged
#              (distinct programs so the single-stream tokenizer keeps the
#               two streams apart -- REQUIRED, same-program streams fuse)
#   2. tokenize preprocess_large_midi_dataset.py pop909_melchord
#              -> data/pop909_melchord_cp16_v2.pt (+ .length/.pitch_shift/.txt)
#   3. finetune finetune_cp_transformer_pop909.py: pretrained weights,
#              fresh short OneCycle (LR_TOTAL_STEPS at MAX_LR), val on the
#              held-out POP909 split, ckpts under
#              ckpt/cp_transformer_v0.42_size1_pop909ft_batch_<B>_schedule/
#
# Env knobs (override via --export=ALL,KEY=val):
#   MELODY_DIR      default: POP909-Dataset/POP909-melody
#   CHORD_DIR       default: POP909-Dataset/POP909-chord
#   TAGGED_DIR      default: POP909-Dataset/POP909-melody-chord-tagged
#   CHORD_PROGRAM   default 48 (the tag; matches cp_transformer_inference's
#                   default track-name separation, so finetuned ckpts and
#                   the inference tagging agree on what program 48 means)
#   PRETRAINED      default: the v0.42 size1 pretrained ckpt
#   RESUME          full Lightning resume ckpt of a PRIOR finetune run;
#                   auto-detected from the run dir's last.ckpt if unset
#   MAX_LR          default 2e-5
#   LR_TOTAL_STEPS  default 20000
#   BATCH_SIZE      per-GPU, default 12 (x4 GPUs = pretraining's global 48)
#   RUN_TAG         optional run-name suffix
#   FORCE=1         re-run merge+tokenize even if outputs exist
#
# Example:
#   sbatch midi_yinyang/finetune_cp_transformer_pop909.sbatch

#SBATCH --job-name=mitosis_cpft_pop909
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

MELODY_DIR="${MELODY_DIR:-POP909-Dataset/POP909-melody}"
CHORD_DIR="${CHORD_DIR:-POP909-Dataset/POP909-chord}"
TAGGED_DIR="${TAGGED_DIR:-POP909-Dataset/POP909-melody-chord-tagged}"
CHORD_PROGRAM="${CHORD_PROGRAM:-48}"
PRETRAINED="${PRETRAINED:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
RESUME="${RESUME:-}"
MAX_LR="${MAX_LR:-2e-5}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
RUN_TAG="${RUN_TAG:-}"
FORCE="${FORCE:-0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs data

# --- detect GPU count / auto-resume ------------------------------------
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NGPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
NGPUS="${NGPUS:-1}"
TAG_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then TAG_SUFFIX="_${RUN_TAG}"; fi
RUN_DIR="ckpt/cp_transformer_v0.42_size1_pop909ft${TAG_SUFFIX}_batch_$((BATCH_SIZE * NGPUS))_schedule"
if [[ -z "$RESUME" && -f "$RUN_DIR/last.ckpt" ]]; then
    RESUME="$RUN_DIR/last.ckpt"
    echo "[auto-resume] found $RESUME"
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)  GPUs=$NGPUS"
echo "Finetune single-stream CP transformer on POP909 melody+chord"
echo "MELODY_DIR=$MELODY_DIR ($(ls "$MELODY_DIR" 2>/dev/null | wc -l) files)"
echo "CHORD_DIR =$CHORD_DIR ($(ls "$CHORD_DIR" 2>/dev/null | wc -l) files)"
echo "TAGGED_DIR=$TAGGED_DIR  CHORD_PROGRAM=$CHORD_PROGRAM"
echo "PRETRAINED=$PRETRAINED"
echo "RESUME=${RESUME:-<cold start from pretrained>}"
echo "MAX_LR=$MAX_LR  LR_TOTAL_STEPS=$LR_TOTAL_STEPS  BATCH=$BATCH_SIZE x $NGPUS"
echo "RUN_DIR=$RUN_DIR"
echo "================================================================"
nvidia-smi | head -n 12 || true

# --- 1. merge (tagged) --------------------------------------------------
if [[ "$FORCE" != "1" && "$(find "$TAGGED_DIR" -maxdepth 1 -name '*.mid' 2>/dev/null | wc -l)" -gt 0 ]]; then
    echo "[merge] $TAGGED_DIR already populated; skipping."
else
    python merge_melody_chord.py \
        --melody "$MELODY_DIR" --chord "$CHORD_DIR" \
        --dst "$TAGGED_DIR" --chord-program "$CHORD_PROGRAM"
fi

# --- 2. tokenize ---------------------------------------------------------
if [[ "$FORCE" != "1" && -f data/pop909_melchord_cp16_v2.pt ]]; then
    echo "[tokenize] data/pop909_melchord_cp16_v2.pt exists; skipping."
else
    POP909_COMBINED_PATH="$TAGGED_DIR" \
        python preprocess_large_midi_dataset.py pop909_melchord
fi

# --- 3. finetune ---------------------------------------------------------
export MASTER_PORT=$((20000 + RANDOM % 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS=()
if [[ -n "$RESUME" ]]; then
    EXTRA_ARGS+=(--resume "$RESUME")
fi
if [[ -n "$RUN_TAG" ]]; then
    EXTRA_ARGS+=(--run_tag "$RUN_TAG")
fi

if [[ "$NGPUS" -le 1 ]]; then
    python finetune_cp_transformer_pop909.py \
        --pretrained "$PRETRAINED" \
        --batch_size "$BATCH_SIZE" \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        "${EXTRA_ARGS[@]}"
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        finetune_cp_transformer_pop909.py \
        --pretrained "$PRETRAINED" \
        --batch_size "$BATCH_SIZE" \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        "${EXTRA_ARGS[@]}"
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local} finished at $(date)"
echo "Checkpoints under $RUN_DIR"
