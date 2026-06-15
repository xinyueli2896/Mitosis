#!/bin/bash
# SLURM batch script: multi-GPU training of M2CIntraCrossAttn on cscc-gpu.
# Mirrors train_slurm.sbatch but targets the intra-cross-attn variant
# (per-modality Q/K/V/O + two-pass intra/cross attention + per-block gates +
# shared MoE FFN). ~15-25% slower per step than jointattn due to the second
# SDPA per block; matched LR/aux/silence-aug defaults.
#
# Knobs (all optional):
#   --gres=gpu:N            how many GPUs to allocate (1..4).
#   --export=ALL,TASK=...   task name: drumnondrum (default) or melchord.
#                           Resolves mod_a/mod_b dataset paths + labels.
#   --export=ALL,CKPT=...   if set, skip init + resume from that ckpt.
#   --export=ALL,MAX_LR=..  override the LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=12 per-GPU batch size (default 8).
#   --export=ALL,GATE_INIT_BIAS=-4  raise if gates stay pinned at init.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first training batch as .mid
#                           into this dir for sanity-check listening.
#                           Default: temp/train_samples_<JOBID>.
#
# Examples:
#   sbatch midi_yinyang/train_intra_cross_attn.sbatch
#   sbatch --export=ALL,TASK=melchord midi_yinyang/train_intra_cross_attn.sbatch
#   sbatch --gres=gpu:2 midi_yinyang/train_intra_cross_attn.sbatch
#   sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt midi_yinyang/train_intra_cross_attn.sbatch

#SBATCH --job-name=mitosis_intra_cross_attn
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
# Node max is 256 GB but SLURM/kernel reserve ~16 GB, so 256G is
# rejected with "Memory specification can not be satisfied". 240G is
# safely under the cap. With mmap'd data tensors (see FramedDataset
# _robust_load mmap=True path), aggregate RAM is dominated by model
# state + activations rather than data duplication, so 240G is more
# than enough now.
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

# --- environment ---------------------------------------------------------
set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
INIT_CKPT=ckpt/m2c_intra_cross_attn_init/m2c_intra_cross_attn_init.ckpt

# Default CKPT for resume. If set (and the file exists), the script
# skips the warm-start init and resumes training from this Lightning
# ckpt -- optimizer state, LR scheduler step, global_step all restored.
# Leave empty to do a fresh warm-start from INIT_CKPT instead.
# Override at submit time via --export=ALL,CKPT=path/to/other.ckpt.
CKPT="${CKPT:-ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_drumnondrum_batch_16_schedule/last.ckpt}"
MAX_LR="${MAX_LR:-}"
# Default per-GPU batch size 4. intra-cross-attn's two SDPA passes per
# block roughly double attention activation memory vs jointattn, so 8
# OOMs on 40GB A100s with the size-large + 12-layer config. Override
# via --export=ALL,BATCH_SIZE=2 if you still OOM, or BATCH_SIZE=6 to
# push if you have memory headroom.
BATCH_SIZE="${BATCH_SIZE:-4}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:--10.0}"
# Task config: melchord (mod_a=mel, mod_b=chord) or drumnondrum
# (mod_a=drum, mod_b=nondrum). Dataset paths and display labels are
# resolved from tasks.py based on this name. Override with
# --export=ALL,TASK=melchord on the sbatch line.
TASK="${TASK:-drumnondrum}"
# Optional: dump the first training batch as .mid files into this dir at
# train start. Useful for sanity-check listening before letting training
# run for hours. Default empty = off.
DUMP_SAMPLES_DIR="${DUMP_SAMPLES_DIR:-temp/train_samples_${SLURM_JOB_ID}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

# --- detect GPU count --------------------------------------------------
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NGPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
NGPUS="${NGPUS:-1}"

# sqrt-scaled LR by NGPUS (base 5e-5 @ 1 GPU). Override via MAX_LR=...
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
echo "Variant: m2c_intra_cross_attn  (per-block gated intra+cross attn)"
echo "Task:    $TASK"
echo "Detected GPUs: $NGPUS   Per-GPU batch: $BATCH_SIZE   Global: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR   gate_init_bias: $GATE_INIT_BIAS"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip init)"
else
    echo "Mode: FRESH (init from PRETRAINED=$PRETRAINED -> INIT_CKPT=$INIT_CKPT)"
fi
echo "----------------------------------------------------------------"
nvidia-smi || true
echo "================================================================"

# --- step 1: warm-start init (only when no CKPT given) -----------------
if [[ -z "$CKPT" ]]; then
    if [[ -f "$INIT_CKPT" ]]; then
        echo "[init] $INIT_CKPT already exists; skipping warm-start step."
    else
        if [[ ! -f "$PRETRAINED" ]]; then
            echo "[init] ERROR: pretrained ckpt not found at $PRETRAINED"
            exit 1
        fi
        echo "[init] running init_pretrained_into_intra_cross_attn.py"
        python init_pretrained_into_intra_cross_attn.py \
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

# --- step 2: training ---------------------------------------------------
export MASTER_PORT=$((20000 + RANDOM % 10000))
# Allocator anti-fragmentation. intra-cross-attn does 2 SDPA passes per
# block, so attention activations can fragment the heap more than usual.
# Cheap to enable; helps avoid spurious OOMs near the memory ceiling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Loss weights (mel/acc/eos) all default to 1.0 in the training script,
# i.e. unweighted CE -- val_loss is comparable across runs.
if [[ "$NGPUS" -le 1 ]]; then
    python cp_transformer_m2c_intra_cross_attn.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps 50000 \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 \
        --dump_samples_dir "$DUMP_SAMPLES_DIR" \
        --gate_init_bias "$GATE_INIT_BIAS" \
        --checkpoint_path "$TRAIN_CKPT" \
        --wandb
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_intra_cross_attn.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps 50000 \
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
