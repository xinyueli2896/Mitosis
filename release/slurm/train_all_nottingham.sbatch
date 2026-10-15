#!/bin/bash
# Every in-house system on Nottingham, packed onto one 8-GPU node by
# scripts/train_all.sh (dense, duet, duet_stale, no_draft on 2 GPUs each,
# then ss_finetuned on 3 and ss_scratch on 1 as GPUs free up).
#
#   sbatch slurm/train_all_nottingham.sbatch
#
# FRESH=1 (default) starts every run from scratch: earlier Nottingham run
# directories are moved, not deleted, to ckpt/old_<jobid>/. FRESH=0 lets
# the Duet-family runs resume from their last.ckpt instead.
# Knobs travel with --export=ALL,KEY=val (SYSTEMS, WANDB, STALE_EVERY, ...).
#SBATCH --job-name=train_nottingham
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

export DATASET=nottingham WANDB="${WANDB:-1}"
export NOTTINGHAM_RAW="${NOTTINGHAM_RAW:-$HOME/xy/nottingham-dataset/MIDI}"

if [[ "${FRESH:-1}" == 1 ]]; then
    old="duet/ckpt/old_${SLURM_JOB_ID}"
    shopt -s nullglob
    runs=(duet/ckpt/*_nottingham*_schedule)
    shopt -u nullglob
    if (( ${#runs[@]} )); then
        mkdir -p "$old"
        mv "${runs[@]}" "$old/"
        echo "[fresh] moved ${#runs[@]} earlier Nottingham runs to $old"
    fi
fi

nvidia-smi --query-gpu=index,name,memory.used --format=csv
bash scripts/train_all.sh
