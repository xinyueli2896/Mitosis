#!/bin/bash
# SLURM (CPU): audit the A.2.moe_hardroute disjoint-pool control.
#
# Runs audit_moe_hard_route.py -- pool disjointness (zero probability on
# the other modality's experts, at FFN level and through the layer that
# derives the modality layout itself), distribution validity (rows sum
# to 1, finite outputs, gradients still reach the gate through the -inf
# mask), FAIRNESS of the within-pool load-balancing loss (the arm is not
# penalised for an allocation its architecture forbids), parameter
# parity with the shared arm (differences are routing, not capacity),
# checkpoint detectability of hard_route_flag, loud misuse, and the
# unrepresentability of integrator experts. Then re-runs the gates and
# bias audits, since all three share SimpleMoEFFN.
#
# Takes seconds. Run after any change to SimpleMoEFFN or
# M2CDuetBlockLayer, and before launching the first
# MOE_MODALITY_HARD_ROUTE=1 training run.
#
# Example:
#   sbatch midi_yinyang/audit_moe_hard_route.sbatch

#SBATCH --job-name=audit_moe_hard_route
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
echo "A.2.moe_hardroute DISJOINT-POOL AUDIT"
echo "================================================================"

python audit_moe_hard_route.py

echo "================================================================"
echo "Re-running the gates and bias audits (shared SimpleMoEFFN paths):"
echo "================================================================"
python audit_moe_gates.py
python audit_moe_bias.py
