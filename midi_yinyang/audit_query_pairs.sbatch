#!/bin/bash
# SLURM (CPU): audit the Q query-pair training knob (--query_pairs).
#
# Training supervises the query slots on ONE frame per forward, while
# the AR loss scores all 2T positions -- so the frame pass, the only
# mechanism the decode loop actually uses, gets roughly 1/T of the
# gradient. QUERY_PAIRS=Q appends Q query pairs for Q distinct frames
# instead, each with its own visibility window, its own (k_m, k_c) draw
# and its own loss.
#
# The property that makes this safe is that the added pairs are INERT:
# pairs are blind to each other, and the clean stream is blind to all of
# them. So pair j of a Q>1 forward must produce BIT-IDENTICAL query
# logits to a Q=1 forward at that same frame, and the AR logits must not
# move at all. audit_query_pairs.py checks exactly that, plus the mask
# regression at Q=1 (every existing checkpoint's geometry is untouched),
# pair isolation in the frame pass, per-pair rotary phases, target
# alignment, and the k / self-conditioning shape coercions.
#
# Takes seconds. Run after any change to
# cp_transformer_m2c_duet_block{,_diffusion}.py, and before launching
# the first QUERY_PAIRS>1 training run.
#
# Example:
#   sbatch midi_yinyang/audit_query_pairs.sbatch

#SBATCH --job-name=audit_query_pairs
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
echo "AUDIT: query-pair count (Q)"
echo "================================================================"

python audit_query_pairs.py

echo "================================================================"
echo "Done."
echo "================================================================"
