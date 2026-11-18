#!/bin/bash
# Continue held-out prompts with a time-grid checkpoint and write MIDI
# (sample<i>.mid next to reference<i>.mid) to temp/time_samples/<run>/,
# or OUT. One GPU.
#
#   sbatch --export=ALL,CORPUS=time_piano,CKPT=ckpt/<run>/last.ckpt slurm/pretrain_time_sample.sbatch
#SBATCH --job-name=pretrain_time_sample
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
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
bash scripts/pretrain_time.sh sample
