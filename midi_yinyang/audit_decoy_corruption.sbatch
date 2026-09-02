#!/bin/bash
# SLURM (CPU): audit the A.7 decoy-corruption variant.
#
# Runs audit_decoy_corruption.py -- eval-mode bit-equivalence with the
# plain model at k=K (val_loss stays comparable across the A-family),
# k=0 returning the exact true frame, lag draws respecting the
# calibrated bins with both signs, circular parity-correct gathers (a
# melody slot can never receive a chord frame), the k=K-only residual
# mask, self-conditioning overrides, revealed bookkeeping, untouched
# inputs + finite loss + flowing gradients, the mutual-exclusion
# guards (decoy vs token_level_mask / mask_revealed_query_loss, bin
# count), and value-carried flag detection.
#
# Takes seconds. Run after any change to
# cp_transformer_m2c_duet_block_diffusion.py, and before launching the
# first DECOY_CORRUPTION=1 training run.
#
# Example:
#   sbatch midi_yinyang/audit_decoy_corruption.sbatch

#SBATCH --job-name=audit_decoy_corruption
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
echo "A.7 decoy-corruption audit"
echo "================================================================"

python audit_decoy_corruption.py

echo "================================================================"
echo "Exit 0 = ALL CHECKS PASSED; the A7 training command is unblocked."
echo "================================================================"
