#!/bin/bash
# SLURM (CPU): audit the A.2.moe_improved per-modality router bias.
#
# Runs audit_moe_bias.py -- warm-start equivalence (zero bias ==
# bias-off, bit-identical), content-pathway isolation (the probe cache
# ignores the bias), routing steerability (a saturating bias moves every
# position's top-1 to its modality), and the modality layout the layer
# feeds the FFN (matches _build_masks, query slots included). Takes
# seconds; run it once after any change to SimpleMoEFFN or
# M2CDuetBlockLayer, and before launching the first MOE_MODALITY_BIAS=1
# training run.
#
# Example:
#   sbatch --exclude=cn-20 midi_yinyang/audit_moe_bias.sbatch

#SBATCH --job-name=audit_moe_bias
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
echo "A.2.moe_improved ROUTER BIAS AUDIT"
echo "================================================================"

python audit_moe_bias.py
