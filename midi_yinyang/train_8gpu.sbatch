#!/bin/bash
# SLURM batch script: 8-GPU training of M2CJointAttn on the cscc-gpu cluster.
# Submit with:
#     sbatch midi_yinyang/train_8gpu.sbatch
# Watch with:
#     squeue -u $USER
#     tail -f $HOME/logs/mitosis_jointattn_8gpu_<JOBID>.out
#
# IMPORTANT edits before first submission:
#   1. CONDA_ENV  -- conda env that has torch + pytorch_lightning installed.
#   2. REPO_DIR   -- where this repo is cloned on the cluster.
#   3. --checkpoint_path -- relative to REPO_DIR/midi_yinyang.
#      Remove the flag entirely to train from a fresh warm-start init.

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

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "REPO_DIR=$REPO_DIR  CONDA_ENV=$CONDA_ENV"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------------------------"
nvidia-smi
echo "================================================================"

# Lightning auto-detects torchrun env vars (LOCAL_RANK, RANK, WORLD_SIZE)
# and wires up DDP for 8 processes x 1 GPU each on this node.
# Random MASTER_PORT avoids "address already in use" when resubmitting
# back-to-back.
export MASTER_PORT=$((20000 + RANDOM % 10000))

# --- training ------------------------------------------------------------
# Per-GPU batch_size 4 -> global batch 32 (4x your 1-GPU batch 8).
# max_lr scaled by sqrt(4) = 2x (5e-5 -> 1.5e-4) -- conservative.
# lr_total_steps halved to ~50k since each step covers 4x more data.
#
# Drop --checkpoint_path if you want to train from a fresh warm-start
# init ckpt instead of resuming. Update the path if your ckpt dir name
# differs.

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
    --checkpoint_path ckpt/m2c_jointattn_v1.0_large_gnl12_batch_8_schedule/last.ckpt \
    --wandb

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
