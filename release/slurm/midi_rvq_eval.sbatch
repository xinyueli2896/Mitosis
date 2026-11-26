#!/bin/bash
# MIDI RVQ-VAE 2/3: whole-piece reconstruction F1 on MAESTRO test (one
# GPU). Writes <ckpt>.eval_<data>.json. The bar for using the codes: note
# F1 well above 0.99, since a wrong MIDI code is a wrong note.
#
#   sbatch --export=ALL,CKPT=ckpt/<run>/ slurm/midi_rvq_eval.sbatch
#SBATCH --job-name=sym2aud_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -p lowprio
set -eo pipefail
source ~/.bashrc || true
# many ~/.bashrc files return early in batch shells, before conda init
command -v conda >/dev/null && eval "$(conda shell.bash hook)"
conda activate mitosis
set -u
ROOT="${SLURM_SUBMIT_DIR:?submit from release/}"
cd "$ROOT"; mkdir -p logs
nvidia-smi --query-gpu=index,name,memory.used --format=csv
bash scripts/midi_rvq.sh eval
