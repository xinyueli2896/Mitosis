#!/bin/bash
# SLURM (CPU): why is the best SAVED checkpoint worse than the best
# LOGGED val_loss?
#
# Opens every .ckpt in a run directory and prints the trainer's
# global_step/epoch plus the ModelCheckpoint callback's own record
# (best_model_score, best_model_path, kth_value, best_k_models) as it
# stood when each file was written. That is the ground truth for a
# wandb-vs-disk disagreement -- see inspect_ckpt_callback.py's docstring
# for how to read each mismatch pattern.
#
# Env knobs (--export=ALL,KEY=val):
#   DIR   run directory to inspect (REQUIRED), e.g.
#         ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mgtk_melchord_long_batch_8_schedule
#
# Example:
#   sbatch --export=ALL,DIR=ckpt/<run_dir> \
#       midi_yinyang/inspect_ckpt_callback.sbatch

#SBATCH --job-name=inspect_ckpt_callback
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

if [[ -z "${DIR:-}" ]]; then
    echo "ERROR: set DIR=ckpt/<run_dir>"
    exit 1
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "INSPECT CHECKPOINT CALLBACK STATE   DIR=$DIR"
echo "================================================================"

python inspect_ckpt_callback.py --dir "$DIR"

echo "================================================================"
echo "Also: any OTHER job currently writing to this directory?"
echo "================================================================"
squeue -u "$USER" -o "%.10i %.30j %.10M %.6D %R" || true
