#!/bin/bash
# MIDI RVQ-VAE 3/3: MIDI codes for every MAESTRO split (one GPU),
# data/maestro_<split>.midi_codes.npy, row i the same 20 ms as row i of
# data/maestro_<split>.codes.npy (EnCodec).
#
#   sbatch --export=ALL,CKPT=ckpt/<run>/ slurm/midi_rvq_export.sbatch
#SBATCH --job-name=midi_rvq_export
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
bash scripts/midi_rvq.sh export
