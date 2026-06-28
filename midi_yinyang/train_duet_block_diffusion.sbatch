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
#   By default this initialises from the base single-stream CP-transformer
#   pretrained ckpt via init_pretrained_into_duet_block_diffusion.py
#   (same pattern as A.1 / A.2 / C.1 / C.2). The init copies the
#   pretrained Q/K/V/O into the per-modality slots and zero-inits the
#   A.3-specific extras (mask_m_emb, mask_c_emb, k_emb_m, k_emb_c, gates).
#
#   We deliberately do NOT warm-start from an A.2 ckpt: A.2 was trained
#   only in the fully-masked regime, and warm-starting from it would
#   inherit the query-slot collapse that motivated A.3 in the first
#   place. The neutral pretrained init lets the diffusion objective
#   train the query-slot pathway across the full noise spectrum.
#
#   To resume an in-progress diffusion run, pass CKPT=path/to/last.ckpt.
#
# Knobs (all optional):
#   --gres=gpu:N            how many GPUs to allocate (1..4).
#   --export=ALL,TASK=...   drumnondrum (default) or melchord.
#   --export=ALL,CKPT=...   if set, skip init + full Lightning resume from
#                           that ckpt. PRESERVES optimizer momentum and
#                           scheduler state. To continue past the original
#                           50k cap, also set LR_TOTAL_STEPS to the new
#                           TOTAL (e.g. 100000 for another 50k steps).
#   --export=ALL,LR_TOTAL_STEPS=N    total steps the OneCycle schedule
#                           spans. Default 50000. When continuing, set
#                           this LARGER than the prior run's value --
#                           Lightning re-creates the scheduler with the
#                           new total_steps and loads the old last_epoch
#                           on top, so the LR keeps annealing smoothly
#                           instead of restarting.
#   --export=ALL,DIFFUSION_K=8       number of noise-level bins (default 4).
#   --export=ALL,FRESH_SCHEDULE=1    opt-in: load weights only, drop
#                           optimizer + scheduler + step counter, run a
#                           brand-new OneCycle from step 0. NOT the
#                           default -- only use if you want to change K,
#                           query_loss_weight, or other things that the
#                           optimizer state has baked in assumptions about.
#   --export=ALL,MAX_LR=...          override LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=...      per-GPU batch size (default 4).
#   --export=ALL,GATE_INIT_BIAS=-4   raise if gates stay at init.
#   --export=ALL,QUERY_LOSS_WEIGHT=2.0  bump query-slot CE weight.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first batch to .mid.
#
# Examples:
#   sbatch midi_yinyang/train_duet_block_diffusion.sbatch
#   sbatch --export=ALL,DIFFUSION_K=8 midi_yinyang/train_duet_block_diffusion.sbatch
#   sbatch --export=ALL,CKPT=ckpt/m2c_duet_block_diffusion_.../last.ckpt \
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

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
# Init ckpt path is K-dependent so different K values don't overwrite
# each other (the k_emb_* tables have different shapes).
DIFFUSION_K="${DIFFUSION_K:-4}"
INIT_CKPT="ckpt/m2c_duet_block_diffusion_init_K${DIFFUSION_K}/m2c_duet_block_diffusion_init_K${DIFFUSION_K}.ckpt"

CKPT="${CKPT:-}"
FRESH_SCHEDULE="${FRESH_SCHEDULE:-0}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-50000}"
MAX_LR="${MAX_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:--10.0}"
QUERY_LOSS_WEIGHT="${QUERY_LOSS_WEIGHT:-1.0}"
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
    if [[ "$FRESH_SCHEDULE" == "1" ]]; then
        echo "Mode: CONTINUE (load weights from CKPT=$CKPT, fresh LR schedule)"
    else
        echo "Mode: RESUME (full Lightning resume from CKPT=$CKPT)"
    fi
else
    echo "Mode: FRESH (init from PRETRAINED -> INIT_CKPT=$INIT_CKPT)"
fi
echo "LR_TOTAL_STEPS=$LR_TOTAL_STEPS"
echo "----------------------------------------------------------------"
nvidia-smi || true
echo "================================================================"

# --- step 1: warm-start init -----------------------------------------
if [[ -z "$CKPT" ]]; then
    if [[ -f "$INIT_CKPT" ]]; then
        echo "[init] $INIT_CKPT already exists; skipping warm-start step."
    else
        if [[ ! -f "$PRETRAINED" ]]; then
            echo "[init] ERROR: pretrained ckpt not found at $PRETRAINED"
            exit 1
        fi
        echo "[init] running init_pretrained_into_duet_block_diffusion.py"
        python init_pretrained_into_duet_block_diffusion.py \
            --pretrained "$PRETRAINED" \
            --output     "$INIT_CKPT" \
            --model_size large --global_num_layers 12 \
            --moe_num_experts 4 --moe_topk 2 \
            --gate_init_bias "$GATE_INIT_BIAS" \
            --query_loss_weight "$QUERY_LOSS_WEIGHT" \
            --diffusion_K "$DIFFUSION_K"
        echo "[init] done -> $INIT_CKPT"
    fi
    TRAIN_CKPT="$INIT_CKPT"
else
    if [[ ! -f "$CKPT" ]]; then
        echo "[resume] ERROR: CKPT not found at $CKPT"
        exit 1
    fi
    TRAIN_CKPT="$CKPT"
fi
echo "================================================================"
echo "[train] --checkpoint_path = $TRAIN_CKPT"
echo "================================================================"

# --- step 2: training -------------------------------------------------
export MASTER_PORT=$((20000 + RANDOM % 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_TRAIN_ARGS=()
if [[ "$FRESH_SCHEDULE" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--fresh_schedule)
fi

COMMON_ARGS=(
    --task "$TASK"
    --batch_size "$BATCH_SIZE"
    --model_size large
    --moe_num_experts 4 --moe_topk 2
    --max_lr "$MAX_LR"
    --lr_total_steps "$LR_TOTAL_STEPS"
    --gradient_clip_val 1.0
    --aux_loss_weight 0.01
    --moe_monitor_every_n_steps 500
    --dump_samples_dir "$DUMP_SAMPLES_DIR"
    --gate_init_bias "$GATE_INIT_BIAS"
    --query_loss_weight "$QUERY_LOSS_WEIGHT"
    --diffusion_K "$DIFFUSION_K"
    --checkpoint_path "$TRAIN_CKPT"
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
