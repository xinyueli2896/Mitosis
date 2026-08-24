#!/bin/bash
# SLURM (CPU): position-resolved reconstruction analysis for C.1.
#
# Tests the rehearsal model's core promise: that the suffix can
# RECONSTRUCT the conditioning stream (mod_a) from the prefix. The slot
# predicting a_k never holds a_k (a_{k-1} under shift-2, b_{k-1} under
# shift-1), so exact reconstruction requires retrieving it out of the
# prefix. Standard inference never exercises this -- the suffix's mod_a
# slots are teacher-forced and the model's own reconstruction discarded
# -- and val_recon_loss shows only the aggregate.
#
# This reports mod_a reconstruction accuracy AS A FUNCTION OF FRAME
# INDEX k, which doubles as the empirical test of the geometry fix:
#   legacy geometry  -> query-to-prefix rotary distance grows with k
#                       -> accuracy should DECAY with k
#   prefix_stride2   -> constant distance -> curve should be FLAT
# The scheme is read from the checkpoint's buffers, so pointing this at
# an old ckpt and a new one gives the before/after directly.
#
# C.1-only (C.2 emits no mod_a logits). CPU, a few minutes per batch.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT      REQUIRED. C.1 run dir (best-val auto-resolved) or file.
#   TASK      REQUIRED. Must match the ckpt's training task
#             (melchord_nottingham, melchord_nottingham_rev, ...).
#   BATCHES   default 4 val batches averaged.
#   MAX_POLYPHONY  auto: melchord* -> 8, else 16. Override for cp4 ckpts.
#
# Example (the C1A / C1B pair; submit one job per ckpt):
#   sbatch --export=ALL,TASK=melchord_nottingham,\
# CKPT=ckpt/m2c_duet_rehearsal_C1A_large_gnl12_melchord_nottingham_yyho_batch_8_schedule/ \
#       midi_yinyang/analyze_rehearsal_recon.sbatch

#SBATCH --job-name=rehearsal_recon
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

CKPT="${CKPT:-}"
TASK="${TASK:-}"
BATCHES="${BATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs results

for v in CKPT TASK; do
    if [[ -z "${!v}" ]]; then
        echo "ERROR: $v is required."
        echo "  sbatch --export=ALL,TASK=melchord_nottingham,\\"
        echo "CKPT=ckpt/<C1 run>/ midi_yinyang/analyze_rehearsal_recon.sbatch"
        exit 1
    fi
done
[[ -e "$CKPT" ]] || { echo "ERROR: ckpt not found: $CKPT"; exit 1; }

if [[ -z "${MAX_POLYPHONY:-}" ]]; then
    case "$TASK" in
        melchord*) MAX_POLYPHONY=8 ;;
        *)         MAX_POLYPHONY=16 ;;
    esac
fi
case "$TASK" in melchord*) export MELCHORD_CP="$MAX_POLYPHONY" ;; esac

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "C.1 RECONSTRUCTION ANALYSIS (teacher-forced, position-resolved)"
echo "TASK=$TASK  MAX_POLYPHONY=$MAX_POLYPHONY  BATCHES=$BATCHES"
echo "CKPT=$CKPT"
echo "================================================================"

python analyze_rehearsal_recon.py \
    --ckpt "$CKPT" \
    --task "$TASK" \
    --batches "$BATCHES" \
    --batch-size "$BATCH_SIZE"

echo "================================================================"
echo "Done. The per-frame curve CSV is under results/recon_curve_*.csv;"
echo "compare C1A vs C1B (and an old legacy-geometry ckpt) by overlaying"
echo "their curves. Free-running counterpart: infer_duet_rehearsal.sbatch"
echo "with MODE=reconstruct."
echo "================================================================"
