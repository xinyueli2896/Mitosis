#!/bin/bash
# SLURM (CPU): audit the A.8 query-block variant.
#
# Runs audit_query_block.py -- block geometry (contiguous draws in
# range, every legal start reachable, eval still single-frame), the
# mask structure that defines A.8 (every slot reads clean frames before
# the BLOCK START only, never its own target; every slot reads every
# other slot in the block; clean rows never see slots; B=1 reproduces
# A.3 bit-for-bit), decode dispatch to the block loop, the
# query_block_flag ckpt round-trip that keeps training and decoding in
# the same mode, the query_block/query_pairs exclusion, and a real
# block training step with gradients.
#
# Takes seconds. Run after any change to
# cp_transformer_m2c_duet_block_diffusion.py, and before launching the
# first QUERY_BLOCK>1 training run.
#
# Example:
#   sbatch midi_yinyang/audit_query_block.sbatch

#SBATCH --job-name=audit_query_block
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
echo "A.8 query-block audit"
echo "================================================================"

python audit_query_block.py

echo "================================================================"
echo "Exit 0 = ALL CHECKS PASSED; the A7 training command is unblocked."
echo "================================================================"
