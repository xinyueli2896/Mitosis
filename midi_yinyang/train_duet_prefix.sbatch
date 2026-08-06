#!/bin/bash
# SLURM batch script: multi-GPU training of M2CDuetPrefix on cscc-gpu.
# Prefix-LM variant: all drum first (bidirectional within drum), then all
# nondrum (causal within nondrum, full attention back to drum). One-way
# drum->nondrum conditional generation.
#
# Knobs (all optional):
#   --gres=gpu:N            how many GPUs to allocate (1..4).
#   --export=ALL,TASK=...   drumnondrum (default) or melchord.
#   --export=ALL,CKPT=...   if set, skip init + resume from that ckpt.
#   --export=ALL,MAX_LR=..  override LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=..  per-GPU batch size (default 4).
#   --export=ALL,GATE_INIT_BIAS=-4  raise if gates stay at init.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first batch to .mid.
#
# Examples:
#   sbatch midi_yinyang/train_duet_prefix.sbatch
#   sbatch --export=ALL,GATE_INIT_BIAS=-4 midi_yinyang/train_duet_prefix.sbatch

#SBATCH --job-name=mitosis_duet_prefix
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
INIT_CKPT=ckpt/m2c_duet_prefix_init/m2c_duet_prefix_init.ckpt

CKPT="${CKPT:-}"
MAX_LR="${MAX_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:--10.0}"
TASK="${TASK:-drumnondrum}"
# Data max_polyphony, auto-set from TASK (override with MAX_POLYPHONY).
# The melchord datasets are cp4 (8-token frames); the trainer defaults to
# 16 and the --dump_samples_dir callback decodes real frames with it, so
# an unset value silently mis-decodes the sample dump on cp4 tasks --
# the same bug fixed in the other duet training sbatches.
if [[ -z "${MAX_POLYPHONY:-}" ]]; then
    case "$TASK" in
        melchord|melchord_nottingham|melchord_pop909_nottingham) MAX_POLYPHONY=4 ;;
        *)                                                       MAX_POLYPHONY=16 ;;
    esac
fi
# OneCycle horizon (was hardcoded 50000; same default kept).
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-50000}"
# Steps between val evals; lower on small melchord corpora (e.g. 100) so
# the early minimum is resolved by more than one or two points.
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
# Optional run-name suffix: keeps a rerun out of an older run's ckpt dir,
# where resolve_best_ckpt would otherwise pick across both runs.
RUN_TAG="${RUN_TAG:-}"
DUMP_SAMPLES_DIR="${DUMP_SAMPLES_DIR:-temp/train_samples_duet_prefix_${SLURM_JOB_ID}}"

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
echo "Variant: m2c_duet_prefix  (drum prefix + causal nondrum reading drum)"
echo "Task:    $TASK"
echo "Detected GPUs: $NGPUS   Per-GPU batch: $BATCH_SIZE   Global: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR   gate_init_bias: $GATE_INIT_BIAS"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip init)"
else
    echo "Mode: FRESH (init from PRETRAINED -> INIT_CKPT=$INIT_CKPT)"
fi
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
        echo "[init] running init_pretrained_into_duet_prefix.py"
        python init_pretrained_into_duet_prefix.py \
            --pretrained "$PRETRAINED" \
            --output     "$INIT_CKPT" \
            --model_size large --global_num_layers 12 \
            --moe_num_experts 4 --moe_topk 2 \
            --gate_init_bias "$GATE_INIT_BIAS"
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
# Pre-flight: fail fast with a clear message if the allocated GPUs
# are visible but unusable (bad node) -- otherwise the job dies deep
# inside Lightning device setup with a cryptic INTERNAL ASSERT.
python check_gpu_health.py --expect "$NGPUS" || exit 1

export MASTER_PORT=$((20000 + RANDOM % 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$NGPUS" -le 1 ]]; then
    python cp_transformer_m2c_duet_prefix.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --max_polyphony "$MAX_POLYPHONY" \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 \
        --dump_samples_dir "$DUMP_SAMPLES_DIR" \
        --gate_init_bias "$GATE_INIT_BIAS" \
        --checkpoint_path "$TRAIN_CKPT" \
        --wandb
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_duet_prefix.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --max_polyphony "$MAX_POLYPHONY" \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 \
        --dump_samples_dir "$DUMP_SAMPLES_DIR" \
        --gate_init_bias "$GATE_INIT_BIAS" \
        --checkpoint_path "$TRAIN_CKPT" \
        --wandb
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
