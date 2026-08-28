#!/bin/bash
# SLURM (CPU): audit the A.4 token-level-mask variant.
#
# Runs audit_a4_token_mask.py -- mask-id safety (the free id in the
# instrument-padding dead zone), bit-exact endpoint equivalence with the
# plain variant at k=K and k=0, the graded middle (masked fraction
# tracks k/K, pads untouched, mask id used), token-level
# self-conditioning, value-carried flag detection, the with_velocity
# guard, and the decode-time confidence re-masking helpers.
#
# Takes seconds. Run after any change to
# cp_transformer_m2c_duet_block_diffusion{.py,_inference.py}, and before
# launching the first TOKEN_LEVEL_MASK=1 training run.
#
# Example:
#   sbatch midi_yinyang/audit_a4_token_mask.sbatch

#SBATCH --job-name=audit_a4_token_mask
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
echo "A.4 TOKEN-LEVEL MASK AUDIT"
echo "================================================================"

python audit_a4_token_mask.py

echo "================================================================"
echo "Regression: the MoE audits still pass (shared model file changed):"
echo "================================================================"
python audit_moe_hard_route.py
python audit_moe_gates.py
python audit_moe_bias.py
