#!/bin/bash
# MAESTRO v3 stage 1 of 3: download and unzip the official WAV+MIDI release
# (108 GB zip, ~120 GB unzipped). CPU only. Set MAESTRO_ROOT to scratch.
# The full chain, from release/:
#
#   J1=$(sbatch --parsable --export=ALL,MAESTRO_ROOT=/scratch/$USER/maestro-v3.0.0 slurm/maestro_download.sbatch)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,MAESTRO_ROOT=/scratch/$USER/maestro-v3.0.0 slurm/maestro_encode.sbatch)
#   J3=$(sbatch --parsable --dependency=afterok:$J2 --export=ALL,MAESTRO_ROOT=/scratch/$USER/maestro-v3.0.0 slurm/maestro_pack.sbatch)
#
# A resubmitted download resumes the partial zip; a finished one is a no-op.
#SBATCH --job-name=maestro_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
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
bash scripts/maestro.sh download
