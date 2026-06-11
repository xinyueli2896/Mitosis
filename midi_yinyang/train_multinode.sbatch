#!/bin/bash
# SLURM batch script: multi-NODE DDP training of M2CJointAttn.
#
# Default config: 2 nodes x 2 GPUs/node = 4 GPUs total.
# Override the geometry on the sbatch command line:
#     --nodes=N            number of nodes
#     --gres=gpu:G         GPUs PER NODE (not total!)
#
# Examples:
#   # 2 nodes x 2 GPUs = 4 GPUs total (default)
#   sbatch midi_yinyang/train_multinode.sbatch
#
#   # 4 nodes x 2 GPUs = 8 GPUs total
#   sbatch --nodes=4 midi_yinyang/train_multinode.sbatch
#
#   # 2 nodes x 4 GPUs = 8 GPUs total
#   sbatch --nodes=2 --gres=gpu:4 midi_yinyang/train_multinode.sbatch
#
#   # Resume from a ckpt:
#   sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt \
#          midi_yinyang/train_multinode.sbatch
#
# Watch with:
#     squeue -u $USER
#     tail -f $HOME/logs/mitosis_jointattn_multinode_<JOBID>.out
#
# IMPORTANT edits before first submission (same as the single-node script):
#   1. CONDA_ENV  -- conda env with torch + pytorch_lightning installed.
#   2. REPO_DIR   -- where this repo is cloned. Must be on a shared
#                    filesystem (e.g. /l/users/...) visible to all nodes.
#   3. PRETRAINED -- path (relative to REPO_DIR/midi_yinyang) to the
#                    pretrained CP-transformer ckpt for warm-start. Only
#                    used in fresh-init mode.
#   4. INIT_CKPT  -- where the warm-started init ckpt gets written/read.

#SBATCH --job-name=mitosis_jointattn_multinode
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

# --- environment ---------------------------------------------------------
set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
INIT_CKPT=ckpt/m2c_jointattn_init/m2c_jointattn_init.ckpt

CKPT="${CKPT:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

# --- determine geometry from SLURM env ---------------------------------
NNODES="$SLURM_JOB_NUM_NODES"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$GPUS_PER_NODE" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# Master node for the rendezvous (first node in the allocation).
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + RANDOM % 10000))
export MASTER_ADDR MASTER_PORT

# Per-GPU batch_size. Global batch = BATCH_SIZE * TOTAL_GPUS.
# Override via --export=ALL,BATCH_SIZE=12 on the sbatch line.
# Rule of thumb on 80GB GPUs with the size-large model (~310M params,
# K=4 MoE, gnl=12, cp16 subseq=32, seq=384):
#   batch=4   -> ~10-15 GB peak per GPU  (very safe)
#   batch=8   -> ~20-30 GB peak          (recommended default)
#   batch=16  -> ~50-60 GB peak          (push if you can; watch nvidia-smi)
BATCH_SIZE="${BATCH_SIZE:-8}"
GLOBAL_BATCH=$((BATCH_SIZE * TOTAL_GPUS))

# sqrt-scaled LR from a 5e-5 @ global-batch-8 baseline. Override with
# MAX_LR env var.
if [[ -z "${MAX_LR:-}" ]]; then
    case "$GLOBAL_BATCH" in
        8)   MAX_LR=5e-5 ;;
        16)  MAX_LR=7e-5 ;;
        32)  MAX_LR=1e-4 ;;
        64)  MAX_LR=1.5e-4 ;;
        128) MAX_LR=2e-4 ;;
        *)   MAX_LR=1e-4 ;;
    esac
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID  master_node=$(hostname)"
echo "Allocated nodes:   $SLURM_JOB_NODELIST"
echo "Nodes:             $NNODES"
echo "GPUs/node:         $GPUS_PER_NODE"
echo "Total GPUs:        $TOTAL_GPUS"
echo "Per-GPU batch:     $BATCH_SIZE   Global batch: $GLOBAL_BATCH"
echo "Effective max_lr:  $MAX_LR"
echo "MASTER_ADDR:       $MASTER_ADDR"
echo "MASTER_PORT:       $MASTER_PORT"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip init)"
else
    echo "Mode: FRESH (init from PRETRAINED=$PRETRAINED, write INIT_CKPT=$INIT_CKPT)"
fi
echo "----------------------------------------------------------------"
nvidia-smi || true
echo "================================================================"

# --- step 1: warm-start init (runs ONCE on the master node) ------------
# Lives outside srun so it only runs once across the whole job, on the
# master node only. INIT_CKPT must be on a shared filesystem so all
# worker nodes can read it during training.
if [[ -z "$CKPT" ]]; then
    if [[ -f "$INIT_CKPT" ]]; then
        echo "[init] $INIT_CKPT already exists; skipping warm-start step."
    else
        if [[ ! -f "$PRETRAINED" ]]; then
            echo "[init] ERROR: pretrained ckpt not found at $PRETRAINED"
            exit 1
        fi
        echo "[init] running init_pretrained_into_jointattn.py on master node"
        python init_pretrained_into_jointattn.py \
            --pretrained "$PRETRAINED" \
            --output     "$INIT_CKPT" \
            --model_size large --global_num_layers 12 \
            --moe_num_experts 4 --moe_topk 2
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

# --- step 2: distributed training across all nodes ---------------------
# srun launches one task on each allocated node. Inside each task,
# torchrun spawns one process per local GPU. The c10d rendezvous via
# MASTER_ADDR:MASTER_PORT joins all NNODES * GPUS_PER_NODE workers into
# a single process group. Lightning auto-detects this via the standard
# torchrun env vars (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_*).
srun --nodes="$NNODES" --ntasks-per-node=1 bash -c "
    cd '$REPO_DIR/midi_yinyang' && \
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=\$SLURM_NODEID \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        cp_transformer_m2c_jointattn.py \
            --batch_size $BATCH_SIZE \
            --model_size large \
            --path_to_dataset data/la_chord_cp16_v2.pt \
            --moe_num_experts 4 --moe_topk 2 \
            --acc_loss_weight 1.0 \
            --max_lr $MAX_LR \
            --lr_total_steps 50000 \
            --gradient_clip_val 1.0 \
            --aux_loss_weight 0.01 \
            --silence_augment_prob 0.1 \
            --eos_loss_weight 1.0 \
            --moe_monitor_every_n_steps 500 \
            --checkpoint_path '$TRAIN_CKPT' \
            --wandb
"

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
