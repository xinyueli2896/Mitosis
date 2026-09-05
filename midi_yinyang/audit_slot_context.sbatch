#!/bin/bash
# SLURM (CPU): audit for A.9 / A.3f (slot_sees_prev_frame, query_pairs=all).
#
# Runs audit_slot_context.py -- asks the attention mask directly what a
# query slot can read. The clean stream is shifted one frame right, so
# the historical "prediction frame < t" test admits content up to t-2:
# the slot for frame t has never seen frame t-1, while the AR head at
# the same rotary phase has. The fix admits "<= t", i.e. rows 2t and
# 2t+1 (content t-1, both streams) and nothing else -- checked, along
# with: no clean row reads a slot, a slot reads only its own partner
# (at Q=all as at Q=1), a Q=all training step with gradients, the flag
# round-tripping through the checkpoint, and the vectorised rotary
# indices matching the old loop.
#
# Takes seconds. Run before the first SLOT_SEES_PREV_FRAME=1 training.
#
# Example:
#   sbatch midi_yinyang/audit_slot_context.sbatch

#SBATCH --job-name=audit_slot_context
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:15:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "A.9 / A.3f slot-context audit"
echo "================================================================"

python audit_slot_context.py

echo "================================================================"
echo "Exit 0 = ALL CHECKS PASSED; the A.9 / A.3f training commands are unblocked."
echo "================================================================"
