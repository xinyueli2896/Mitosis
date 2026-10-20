#!/bin/bash
# The Duet retrains of README "Improving Duet's co-generation" on
# Nottingham, side by side on one 6-GPU node via scripts/train_all.sh
# (2 GPUs each, the paper recipe's count, so LR and global batch match
# the paper's duet run).
#
#   sbatch slurm/train_duet_variants_nottingham.sbatch
#
#   duet_v2    slot reads t-1 + conditional slots (0.8) + teacher-forced
#              leader + follower-state validation
#   duet_v2c48 duet_v2 with the chord stream on program 48, as the
#              single-stream models see it (tagged vs same-program input)
#   duet_prev  the paper recipe + slot reads t-1 only (attributes that change)
#
# Existing run directories are left alone (no FRESH here): a resubmitted
# job resumes each run from its last.ckpt. SYSTEMS=duet_v2 trains one;
# then pass --gpus=2. The program-48 chord tensor is built first when
# missing (prepare_data.sh chord48, CPU, a few minutes). Per-run logs: logs/train_nottingham_<system>.log.
#SBATCH --job-name=train_duet_variants
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=6
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

export DATASET=nottingham WANDB="${WANDB:-1}"
export SYSTEMS="${SYSTEMS:-duet_v2 duet_v2c48 duet_prev}"
export NOTTINGHAM_RAW="${NOTTINGHAM_RAW:-$HOME/xy/nottingham-dataset/MIDI}"

if [[ " $SYSTEMS " == *" duet_v2c48 "* && ! -f duet/data/nottingham_chord48_cp4_v2.pt ]]; then
    bash scripts/prepare_data.sh chord48
fi

nvidia-smi --query-gpu=index,name,memory.used --format=csv
bash scripts/train_all.sh
