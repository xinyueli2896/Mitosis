#!/bin/bash
# SLURM (CPU): gate for A.10 (same-frame edge) -- runs audit_same_frame.py
# on a tiny random model and checks the masks do exactly what the model
# file documents (leader/follower rows, causality, decode consistency,
# batched directions). Run it before the first SAME_FRAME=1 training.
#
# Example:
#   sbatch midi_yinyang/audit_same_frame.sbatch

#SBATCH --job-name=audit_same_frame
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:20:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs

python audit_same_frame.py
