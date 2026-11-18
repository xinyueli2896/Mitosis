#!/bin/bash
# The Duet retrains of README "Improving Duet's co-generation", side by
# side on one 8-GPU node via scripts/train_all.sh (2 GPUs each, the
# paper recipe's count, so LR and global batch match the paper's duet).
#
#   sbatch slurm/train_duet_variants.sbatch                              # Nottingham
#   sbatch --export=ALL,DATASET=pop909 slurm/train_duet_variants.sbatch  # POP909
#
#   duet_v2    trained for the leaderless joint decode: slot reads t-1,
#              both draft tokens hold AR-sampled drafts at k=0, both scored
#   duet_v2c48 duet_v2 with the chord stream on program 48, as the
#              single-stream models see it (Nottingham only)
#   duet_prev  the paper recipe + slot reads t-1 only (attributes that change)
#
# Defaults: all three on Nottingham (6 GPUs), duet_v2 and duet_prev on
# POP909 (4 GPUs); SYSTEMS=... picks others. Existing run directories
# are left alone: a resubmitted job resumes each run from its last.ckpt.
# The program-48 chord tensor is built first when missing
# (prepare_data.sh chord48, CPU, a few minutes).
# Per-run logs: logs/train_<dataset>_<system>.log.
#SBATCH --job-name=train_duet_variants
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
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

export DATASET="${DATASET:-nottingham}" WANDB="${WANDB:-1}"
if [[ "$DATASET" == nottingham ]]; then
    export SYSTEMS="${SYSTEMS:-duet_v2 duet_v2c48 duet_prev}"
    export NOTTINGHAM_RAW="${NOTTINGHAM_RAW:-$HOME/xy/nottingham-dataset/MIDI}"
else
    export SYSTEMS="${SYSTEMS:-duet_v2 duet_prev}"
fi

if [[ "$DATASET" == nottingham && ( " $SYSTEMS " == *" duet_v2c48 "* || " $SYSTEMS " == *" duet_jf"* ) \
      && ! -f duet/data/nottingham_chord48_cp4_v2.pt ]]; then
    bash scripts/prepare_data.sh chord48
fi

nvidia-smi --query-gpu=index,name,memory.used --format=csv
bash scripts/train_all.sh
