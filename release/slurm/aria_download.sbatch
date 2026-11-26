#!/bin/bash
# Aria-MIDI: download and unpack one subset (default deduped). CPU only.
# The chain to a MIDI RVQ codec trained on Aria alone, from release/:
#
#   A=/scratch/$USER/aria-midi
#   J1=$(sbatch --parsable --export=ALL,ARIA_ROOT=$A slurm/aria_download.sbatch)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,CORPUS=aria_dedup,MIDI_DIRS=$A/deduped,MAESTRO_ROOT= slurm/prep_data.sbatch)
#   sbatch --dependency=afterok:$J2 --export=ALL,CORPUS=aria_dedup slurm/midi_rvq_train.sbatch
#
# A resubmitted download resumes the partial archive; a finished one is a no-op.
#SBATCH --job-name=sym2aud_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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
bash scripts/aria.sh download
