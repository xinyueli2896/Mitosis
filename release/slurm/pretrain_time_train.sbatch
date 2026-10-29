#!/bin/bash
# Time-grid pretraining 3/3: train the cp transformer on time frames
# (duet/cp_transformer_time.py) on all 8 GPUs. Checkpoints in
# duet/ckpt/cp_time_<corpus>_<ms>ms_cp<P>_size<S>_batch_<B>/, selected on
# val_note_loss; a resubmitted job resumes from last.ckpt. Knobs: CORPUS,
# FRAME_CODEC, MAX_POLYPHONY, MODEL_SIZE, BATCH_SIZE, TRAIN_LENGTH,
# MAX_STEPS, STRETCH, WANDB (scripts/pretrain_time.sh).
#SBATCH --job-name=pretrain_time_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --account=deploy
#SBATCH --qos=deploy
set -eo pipefail
source ~/.bashrc || true
# many ~/.bashrc files return early in batch shells, before conda init
command -v conda >/dev/null && eval "$(conda shell.bash hook)"
conda activate mitosis
set -u
ROOT="${SLURM_SUBMIT_DIR:?submit from release/}"
cd "$ROOT"; mkdir -p logs
nvidia-smi --query-gpu=index,name,memory.used --format=csv
bash scripts/pretrain_time.sh train
