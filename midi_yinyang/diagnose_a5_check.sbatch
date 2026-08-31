#!/bin/bash
# SLURM (1 GPU): why is A.5 worse than A.3 even at REFINE_STEPS=0?
#
# Runs diagnose_a5_check.py -- three measurements on the two ckpts over
# the SAME deterministic validation batches, no training, no decoding:
#
#   CHECK 1  keep-mask sign: A.5's query_kept_frac (~0.75 correct;
#            ~0.25 = inverted mask = the run trained on the copy
#            positions only and the A.5 result is a bug).
#   CHECK 2  effective re-weighting: query CE over all positions vs
#            corrupted-only, same draws -- the ratio is the factor by
#            which A.5 silently scaled the query term's gradient share.
#   CHECK 3  eval-mode ar_loss / query_loss for both ckpts on identical
#            batches (directly comparable): ar_loss higher for A.5
#            means the query gradient dragged the shared backbone.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_A5    default: the A.5 run's last.ckpt (K4mgqm legacy dir)
#   CKPT_A3    default: the A.3 long run's last.ckpt (K4mg legacy dir)
#   N_BATCHES  default 40
#
# Example:
#   sbatch midi_yinyang/diagnose_a5_check.sbatch

#SBATCH --job-name=diagnose_a5_check
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

# The two runs are cp4-family melchord ckpts.
export MELCHORD_CP=4

CKPT_A5="${CKPT_A5:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mgqm_melchord_long_batch_8_schedule/last.ckpt}"
CKPT_A3="${CKPT_A3:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mg_melchord_long_batch_8_schedule/last.ckpt}"
N_BATCHES="${N_BATCHES:-40}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "A.5 POST-MORTEM CHECKS"
echo "CKPT_A5=$CKPT_A5"
echo "CKPT_A3=$CKPT_A3"
echo "================================================================"

python diagnose_a5_check.py \
    --ckpt_a5 "$CKPT_A5" \
    --ckpt_a3 "$CKPT_A3" \
    --n_batches "$N_BATCHES"

echo "================================================================"
echo "Done. Read CHECK 1 first: ~0.25 means bug, and the other checks"
echo "are then moot until A.5 is retrained."
echo "================================================================"
