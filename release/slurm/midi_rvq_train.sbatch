#!/bin/bash
# MIDI RVQ-VAE 1/3: train on all 8 GPUs (duet/midi_rvq.py). Checkpoints in
# duet/ckpt/midi_rvq_<corpus>_q<N_Q>x<N_CODES>_d<WIDTH>_batch<B>/, selected
# on val_note_f1 (MAESTRO validation when packed); a resubmitted job
# resumes from last.ckpt. Knobs: scripts/midi_rvq.sh.
#
#   sbatch --export=ALL,CORPUS=time_piano slurm/midi_rvq_train.sbatch
#SBATCH --job-name=midi_rvq_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --account=deploy
#SBATCH --qos=deploy
#SBATCH -p lowprio
#SBATCH --requeue
set -eo pipefail
source ~/.bashrc || true
# many ~/.bashrc files return early in batch shells, before conda init
command -v conda >/dev/null && eval "$(conda shell.bash hook)"
conda activate mitosis
set -u
ROOT="${SLURM_SUBMIT_DIR:?submit from release/}"
cd "$ROOT"; mkdir -p logs
nvidia-smi --query-gpu=index,name,memory.used --format=csv
bash scripts/midi_rvq.sh train
