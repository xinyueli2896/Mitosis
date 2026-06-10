#!/bin/bash
# SLURM batch script: multi-GPU training of M2CJointAttn on the cscc-gpu
# cluster. GPU count is auto-detected from SLURM's allocation, so a single
# script handles 1, 2, 4, 8 GPUs by overriding --gres on the sbatch line.
#
# Three knobs (all optional):
#   --gres=gpu:N           how many GPUs to allocate (overrides #SBATCH).
#                          The script auto-detects N at runtime.
#   --export=ALL,CKPT=...  if set, skip init and resume from that ckpt
#                          (path relative to REPO_DIR/midi_yinyang).
#                          If unset, run init first then train.
#   --export=ALL,MAX_LR=.. override the default LR (else 1.5e-4).
#
# Examples:
#   # 8 GPUs, fresh init + train
#   sbatch midi_yinyang/train_slurm.sbatch
#
#   # 4 GPUs, fresh init + train (script auto-uses 4 procs)
#   sbatch --gres=gpu:4 midi_yinyang/train_slurm.sbatch
#
#   # 2 GPUs, resume from a ckpt
#   sbatch --gres=gpu:2 \
#          --export=ALL,CKPT=ckpt/<run>/last.ckpt \
#          midi_yinyang/train_slurm.sbatch
#
#   # 8 GPUs, resume + custom LR
#   sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt,MAX_LR=1e-4 \
#          midi_yinyang/train_slurm.sbatch
#
# Watch with:
#   squeue -u $USER
#   tail -f $HOME/logs/mitosis_jointattn_*_<JOBID>.out

#SBATCH --job-name=mitosis_jointattn
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --gres=gpu:8
#SBATCH -p gpu
#SBATCH --time=12:00:00

# --- environment ---------------------------------------------------------
set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/l/users/xinyue.li/Mitosis

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
INIT_CKPT=ckpt/m2c_jointattn_init/m2c_jointattn_init.ckpt

# Optional sbatch --export args (read with bash :- default).
CKPT="${CKPT:-}"
MAX_LR="${MAX_LR:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

# --- detect GPU count from SLURM allocation -----------------------------
# SLURM_GPUS_ON_NODE is set by SLURM after --gres=gpu:N. Fall back to
# counting CUDA_VISIBLE_DEVICES, and finally to 1.
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NGPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
NGPUS="${NGPUS:-1}"

# --- LR defaults scaled by NGPUS (sqrt rule, base 5e-5 @ 1 GPU) ---------
# Override by exporting MAX_LR=... on the sbatch line.
if [[ -z "$MAX_LR" ]]; then
    case "$NGPUS" in
        1) MAX_LR=5e-5 ;;
        2) MAX_LR=7e-5 ;;
        4) MAX_LR=1e-4 ;;
        8) MAX_LR=1.5e-4 ;;
        *) MAX_LR=1e-4 ;;
    esac
fi

# Per-GPU batch_size: 4 keeps the model comfortable at large depth/MoE.
BATCH_SIZE=4

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "REPO_DIR=$REPO_DIR  CONDA_ENV=$CONDA_ENV"
echo "Detected GPUs: $NGPUS   (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Per-GPU batch_size: $BATCH_SIZE   Global batch: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip init)"
else
    echo "Mode: FRESH (init from PRETRAINED=$PRETRAINED, write INIT_CKPT=$INIT_CKPT)"
fi
echo "----------------------------------------------------------------"
nvidia-smi
echo "================================================================"

# --- step 1: warm-start init (only when no CKPT given) -----------------
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

# --- step 2: training ---------------------------------------------------
# Lightning auto-detects torchrun env vars (LOCAL_RANK, RANK, WORLD_SIZE)
# and wires up DDP for $NGPUS processes x 1 GPU each on this node.
# Random MASTER_PORT avoids "address already in use" on back-to-back jobs.
export MASTER_PORT=$((20000 + RANDOM % 10000))

if [[ "$NGPUS" -le 1 ]]; then
    # Single-GPU path: skip torchrun entirely.
    python cp_transformer_m2c_jointattn.py \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --path_to_dataset data/la_chord_cp16_v2.pt \
        --moe_num_experts 4 --moe_topk 2 \
        --acc_loss_weight 1.0 \
        --max_lr "$MAX_LR" \
        --lr_total_steps 50000 \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --silence_augment_prob 0.1 \
        --eos_loss_weight 3.0 \
        --moe_monitor_every_n_steps 500 \
        --checkpoint_path "$TRAIN_CKPT" \
        --wandb
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_jointattn.py \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --path_to_dataset data/la_chord_cp16_v2.pt \
        --moe_num_experts 4 --moe_topk 2 \
        --acc_loss_weight 1.0 \
        --max_lr "$MAX_LR" \
        --lr_total_steps 50000 \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --silence_augment_prob 0.1 \
        --eos_loss_weight 3.0 \
        --moe_monitor_every_n_steps 500 \
        --checkpoint_path "$TRAIN_CKPT" \
        --wandb
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
