#!/bin/bash
# SLURM: does ModelCheckpoint(save_last=True) stop writing last.ckpt
# once the monitored metric stops improving?
#
# Runs diagnose_last_ckpt.py -- a toy LightningModule (nothing from the
# repo) with a CONSTANT monitored metric and the training script's
# exact ModelCheckpoint arguments. After fit it prints the checkpoint
# directory with mtimes and the global_step stored inside last.ckpt.
#
# Expected: last.ckpt global_step == 600 (max_steps).
# Reproduced: it equals the step of the last top-k save (100).
#
# Env knobs (--export=ALL,KEY=val):
#   NGPUS   0 (cpu, default), 1, or 2 (DDP -- what the A.8 run used).
#           For NGPUS>0 pass the GPU partition and gres on the command
#           line; flags before the script path override the directives
#           below:
#             sbatch -p cscc-gpu-p -q cscc-gpu-qos --gres=gpu:2 \
#                 --export=ALL,NGPUS=2 midi_yinyang/diagnose_last_ckpt.sbatch
#
# Takes about a minute.

#SBATCH --job-name=diagnose_last_ckpt
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:20:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

NGPUS="${NGPUS:-0}"
OUT="temp/diagnose_last_ckpt/ngpus${NGPUS}_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)  NGPUS=$NGPUS  out=$OUT"
echo "================================================================"

python diagnose_last_ckpt.py "$OUT" "$NGPUS"

echo "================================================================"
echo "The verdict line is the last one above. Both NGPUS=0 and NGPUS=2"
echo "matter: a DDP-only reproduction points at the strategy/barrier"
echo "path, a CPU reproduction at the callback itself."
echo "================================================================"
