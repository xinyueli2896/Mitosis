#!/bin/bash
# Time-grid pretraining, stage 2 of 2: train the cp transformer (same
# architecture as the beat-time backbone, velocity tokenizer by default)
# on duet/data/$CORPUS.notes.pt, 30 s crops (1500 frames), on all 8 GPUs.
# A resubmitted job resumes from the run's last.ckpt. See
# pretrain_time_prep.sbatch for the chain.
#
# Knobs (see scripts/pretrain_time.sh): CORPUS, MODEL_SIZE, VELOCITY, BATCH_SIZE, TRAIN_LENGTH.
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
