#!/bin/bash
# SLURM batch script: 8-GPU training of M2CJointAttn.
#
# *** WARNING: the cscc-gpu-p partition limits any account to 4 GPUs total,
#     so requesting 8 here will be REJECTED by SLURM. This file is kept as
#     a template for clusters with looser limits. For the current CSCC
#     cluster use train_slurm.sbatch with --gres=gpu:4 (single-node) or
#     train_multinode.sbatch (2 nodes x 2 GPUs).
#
# Two modes:
#   (a) Train from a fresh warm-start init (default):
#         sbatch midi_yinyang/train_8gpu.sbatch
#       Runs init_pretrained_into_jointattn.py (single-process, one GPU) if
#       INIT_CKPT doesn't already exist, then launches 8-GPU DDP training
#       with --checkpoint_path $INIT_CKPT.
#
#   (b) Continue training from a specific ckpt:
#         sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt \
#                midi_yinyang/train_8gpu.sbatch
#       Skips the init step entirely and resumes from $CKPT. Path is
#       relative to REPO_DIR/midi_yinyang.
#
# Watch with:
#     squeue -u $USER
#     tail -f $HOME/logs/mitosis_jointattn_8gpu_<JOBID>.out
#
# IMPORTANT edits before first submission:
#   1. CONDA_ENV  -- conda env that has torch + pytorch_lightning installed.
#   2. REPO_DIR   -- where this repo is cloned on the cluster.
#   3. PRETRAINED -- path (relative to REPO_DIR/midi_yinyang) to the
#                    pretrained CP-transformer ckpt used for warm-start.
#                    Only used in mode (a).
#   4. INIT_CKPT  -- where the warm-started init ckpt gets written and read.
#                    Only used in mode (a).

#SBATCH --job-name=mitosis_jointattn_8gpu
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --gres=gpu:8
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

# --- environment ---------------------------------------------------------
set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/l/users/xinyue.li/Mitosis

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
INIT_CKPT=ckpt/m2c_jointattn_init/m2c_jointattn_init.ckpt

# Optional override: pass via `sbatch --export=ALL,CKPT=path/to/ckpt ...`.
# If unset, mode (a) runs init + trains from $INIT_CKPT.
# If set, mode (b) skips init + trains from $CKPT.
CKPT="${CKPT:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "REPO_DIR=$REPO_DIR  CONDA_ENV=$CONDA_ENV"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip init)"
else
    echo "Mode: FRESH (init from PRETRAINED=$PRETRAINED, write INIT_CKPT=$INIT_CKPT)"
fi
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------------------------"
nvidia-smi
echo "================================================================"

# --- step 1: warm-start init (only in mode a) --------------------------
if [[ -z "$CKPT" ]]; then
    if [[ -f "$INIT_CKPT" ]]; then
        echo "[init] $INIT_CKPT already exists; skipping warm-start step."
    else
        if [[ ! -f "$PRETRAINED" ]]; then
            echo "[init] ERROR: pretrained ckpt not found at $PRETRAINED"
            echo "       Copy it over (or download) and resubmit."
            exit 1
        fi
        echo "[init] running init_pretrained_into_jointattn.py"
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

# --- step 2: training (8 GPUs via torchrun + Lightning DDP) -------------
# Lightning auto-detects torchrun env vars (LOCAL_RANK, RANK, WORLD_SIZE)
# and wires up DDP for 8 processes x 1 GPU each on this node.
# Random MASTER_PORT avoids "address already in use" on back-to-back jobs.
export MASTER_PORT=$((20000 + RANDOM % 10000))

# Per-GPU batch_size 4 -> global batch 32 (4x your 1-GPU batch 8).
# max_lr scaled by sqrt(4) = 2x (5e-5 -> 1.5e-4) -- conservative.
# lr_total_steps halved to ~50k since each step covers 4x more data.

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    cp_transformer_m2c_jointattn.py \
    --batch_size 4 \
    --model_size large \
    --path_to_dataset data/la_chord_cp16_v2.pt \
    --moe_num_experts 4 --moe_topk 2 \
    --acc_loss_weight 1.0 \
    --max_lr 1.5e-4 \
    --lr_total_steps 50000 \
    --gradient_clip_val 1.0 \
    --aux_loss_weight 0.01 \
    --silence_augment_prob 0.1 \
    --moe_monitor_every_n_steps 500 \
    --checkpoint_path "$TRAIN_CKPT" \
    --wandb

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
