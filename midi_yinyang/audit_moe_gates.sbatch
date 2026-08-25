#!/bin/bash
# SLURM (CPU): audit the A.2.moe_permod per-modality router gates.
#
# Runs audit_moe_gates.py -- warm-start contract (shared gate seeds both
# gates, bit-identical forward), right-gate-right-token (routing equals
# a manual per-parity recomputation; the layer derives the modality
# layout itself, query slots included), gate independence (changing one
# stream's gate leaves the other stream's routing bit-identical), pool
# freedom (either gate can reach every expert -- nothing is
# pre-assigned), and loud misuse. Takes seconds; run once after any
# change to SimpleMoEFFN or M2CDuetBlockLayer, and before launching the
# first MOE_MODALITY_GATES=1 training run.
#
# Example:
#   sbatch --exclude=cn-20 midi_yinyang/audit_moe_gates.sbatch

#SBATCH --job-name=audit_moe_gates
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
echo "A.2.moe_permod ROUTER GATES AUDIT"
echo "================================================================"

python audit_moe_gates.py

echo "================================================================"
echo "Also re-running the bias audit (shared code paths changed):"
echo "================================================================"
python audit_moe_bias.py
