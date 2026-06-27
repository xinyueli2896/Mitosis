#!/bin/bash
# SLURM batch script: multi-GPU training of M2CDuetBlockDiffusion (A.3).
# DuetBlock with discrete-diffusion-style training at the two query
# slots: per item, per slot, a noise level k in {0..K} controls whether
# the slot is fed mask_*_emb (k=K) or the ground-truth frame embedding
# (k=0), with a learned k-embedding added on top. Trains the model to
# handle every combination of (k_m, k_c), so at inference either parallel
# diffusion or MaskGIT-style iterative refinement is valid.
#
# WARM-START
#   By default this resumes weights from the best A.2 DuetBlock ckpt
#   under WARMSTART_DIR (the val_loss-tagged ckpt with smallest val_loss,
#   falling back to last.ckpt). The diffusion-specific params (k_emb_m,
#   k_emb_c) are zero-initialised, so at step 0 the model is equivalent
#   to the A.2 ckpt at k_m = k_c = K (fully masked) -- the regime A.2
#   was actually trained in. Training then teaches the lower-k regimes.
#
#   To start from scratch (no A.2 warm-start) pass SKIP_WARMSTART=1.
#   To resume an in-progress diffusion run, pass CKPT=path/to/last.ckpt.
#
# Knobs (all optional):
#   --gres=gpu:N            how many GPUs to allocate (1..4).
#   --export=ALL,TASK=...   drumnondrum (default) or melchord.
#   --export=ALL,CKPT=...   if set, skip warmstart + resume from that ckpt.
#   --export=ALL,WARMSTART_DIR=...   override the A.2 ckpt directory.
#   --export=ALL,WARMSTART_CKPT=...  override with a specific ckpt file.
#   --export=ALL,SKIP_WARMSTART=1    skip the A.2 warmstart entirely.
#   --export=ALL,DIFFUSION_K=8       number of noise-level bins (default 4).
#   --export=ALL,MAX_LR=...          override LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=...      per-GPU batch size (default 4).
#   --export=ALL,GATE_INIT_BIAS=-4   raise if gates stay at init.
#   --export=ALL,QUERY_LOSS_WEIGHT=2.0  bump query-slot CE weight.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first batch to .mid.
#
# Examples:
#   sbatch midi_yinyang/train_duet_block_diffusion.sbatch
#   sbatch --export=ALL,DIFFUSION_K=8 midi_yinyang/train_duet_block_diffusion.sbatch
#   sbatch --export=ALL,CKPT=ckpt/m2c_duet_block_diffusion_v1.0_large_gnl12_K4_drumnondrum_batch_16_schedule/last.ckpt \
#          midi_yinyang/train_duet_block_diffusion.sbatch

#SBATCH --job-name=mitosis_duet_block_diffusion
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

# Where to look for the A.2 warmstart ckpt. Defaults to the standard
# A.2 training run; override if you trained A.2 with a different tag.
WARMSTART_DIR="${WARMSTART_DIR:-ckpt/m2c_duet_block_v1.0_large_gnl12_drumnondrum_batch_16_schedule}"
WARMSTART_CKPT="${WARMSTART_CKPT:-}"
SKIP_WARMSTART="${SKIP_WARMSTART:-0}"

CKPT="${CKPT:-}"
MAX_LR="${MAX_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:--10.0}"
QUERY_LOSS_WEIGHT="${QUERY_LOSS_WEIGHT:-1.0}"
DIFFUSION_K="${DIFFUSION_K:-4}"
TASK="${TASK:-drumnondrum}"
DUMP_SAMPLES_DIR="${DUMP_SAMPLES_DIR:-temp/train_samples_duet_block_diffusion_${SLURM_JOB_ID}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

# --- detect GPU count --------------------------------------------------
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NGPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
NGPUS="${NGPUS:-1}"

if [[ -z "$MAX_LR" ]]; then
    case "$NGPUS" in
        1) MAX_LR=5e-5 ;;
        2) MAX_LR=7e-5 ;;
        4) MAX_LR=1e-4 ;;
        8) MAX_LR=1.5e-4 ;;
        *) MAX_LR=1e-4 ;;
    esac
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Variant: m2c_duet_block_diffusion (A.3)  K=$DIFFUSION_K"
echo "Task:    $TASK"
echo "Detected GPUs: $NGPUS   Per-GPU batch: $BATCH_SIZE   Global: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR   gate_init_bias: $GATE_INIT_BIAS   query_loss_weight: $QUERY_LOSS_WEIGHT"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip warmstart)"
elif [[ "$SKIP_WARMSTART" == "1" ]]; then
    echo "Mode: FRESH (SKIP_WARMSTART=1; no A.2 warmstart, no resume)"
else
    echo "Mode: WARMSTART from A.2 ckpt under $WARMSTART_DIR"
fi
echo "----------------------------------------------------------------"
nvidia-smi || true
echo "================================================================"

# --- step 1: resolve warmstart ckpt -----------------------------------
# Precedence: explicit CKPT (resume) > explicit WARMSTART_CKPT > best
# val_loss ckpt under WARMSTART_DIR > last.ckpt under WARMSTART_DIR.
TRAIN_CKPT=""
EXTRA_TRAIN_ARGS=()
if [[ -n "$CKPT" ]]; then
    if [[ ! -f "$CKPT" ]]; then
        echo "[resume] ERROR: CKPT not found at $CKPT"
        exit 1
    fi
    TRAIN_CKPT="$CKPT"
elif [[ "$SKIP_WARMSTART" != "1" ]]; then
    if [[ -n "$WARMSTART_CKPT" ]]; then
        if [[ ! -f "$WARMSTART_CKPT" ]]; then
            echo "[warmstart] ERROR: WARMSTART_CKPT not found at $WARMSTART_CKPT"
            exit 1
        fi
        TRAIN_CKPT="$WARMSTART_CKPT"
    elif [[ -d "$WARMSTART_DIR" ]]; then
        # Pick the val_loss-tagged ckpt with the smallest val_loss.
        best=""
        best_loss=""
        for f in "$WARMSTART_DIR"/*val_loss=*.ckpt; do
            [[ -f "$f" ]] || continue
            # strip up to val_loss=, then up to .ckpt
            v="${f##*val_loss=}"
            v="${v%.ckpt}"
            if [[ -z "$best" ]] || awk "BEGIN { exit !($v < $best_loss) }"; then
                best="$f"
                best_loss="$v"
            fi
        done
        if [[ -n "$best" ]]; then
            TRAIN_CKPT="$best"
            echo "[warmstart] picked $TRAIN_CKPT  (val_loss=$best_loss)"
        elif [[ -f "$WARMSTART_DIR/last.ckpt" ]]; then
            TRAIN_CKPT="$WARMSTART_DIR/last.ckpt"
            echo "[warmstart] no val_loss-tagged ckpt; falling back to last.ckpt"
        else
            echo "[warmstart] ERROR: no ckpt found under $WARMSTART_DIR"
            exit 1
        fi
    else
        echo "[warmstart] ERROR: WARMSTART_DIR $WARMSTART_DIR not found and no WARMSTART_CKPT/CKPT given"
        exit 1
    fi
    # Warmstart loads weights only, then runs the fresh diffusion
    # schedule (we want a clean LR schedule for the new objective).
    EXTRA_TRAIN_ARGS+=(--fresh_schedule)
fi

if [[ -n "$TRAIN_CKPT" ]]; then
    EXTRA_TRAIN_ARGS+=(--checkpoint_path "$TRAIN_CKPT")
    echo "================================================================"
    echo "[train] starting from $TRAIN_CKPT  (fresh schedule: $([[ " ${EXTRA_TRAIN_ARGS[*]} " == *" --fresh_schedule "* ]] && echo yes || echo no))"
    echo "================================================================"
else
    echo "================================================================"
    echo "[train] starting from scratch (no warmstart)"
    echo "================================================================"
fi

# --- step 2: training -------------------------------------------------
export MASTER_PORT=$((20000 + RANDOM % 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
    --task "$TASK"
    --batch_size "$BATCH_SIZE"
    --model_size large
    --moe_num_experts 4 --moe_topk 2
    --max_lr "$MAX_LR"
    --lr_total_steps 50000
    --gradient_clip_val 1.0
    --aux_loss_weight 0.01
    --moe_monitor_every_n_steps 500
    --dump_samples_dir "$DUMP_SAMPLES_DIR"
    --gate_init_bias "$GATE_INIT_BIAS"
    --query_loss_weight "$QUERY_LOSS_WEIGHT"
    --diffusion_K "$DIFFUSION_K"
    --wandb
)

if [[ "$NGPUS" -le 1 ]]; then
    python cp_transformer_m2c_duet_block_diffusion.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_duet_block_diffusion.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
