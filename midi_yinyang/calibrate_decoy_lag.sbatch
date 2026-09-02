#!/bin/bash
# SLURM (CPU): A7 pre-training calibration -- harmonic agreement of
# (mel_t, chord_{t+lag}) as a function of lag over the POP909 TRAINING
# songs (test ids excluded). The printed table + verdict place the A7
# corruption's lag bins for k=1..3 at equal spacing in MEASURED
# decoherence, and confirm the decay is monotone before any GPU is
# spent. Uses the eval harness's own Stream/coverage/shift machinery,
# so 'agreement' here is the same quantity harmonic_coupling scores.
#
# Env knobs (--export=ALL,KEY=val):
#   MELODY_DIR default /home/xinyue.li/POP909-Dataset/POP909-melody
#   CHORD_DIR  default /home/xinyue.li/POP909-Dataset/POP909-chord
#   MAX_SONGS  default 0 (all train songs); >0 for a quick pass
#
# Example:
#   sbatch midi_yinyang/calibrate_decoy_lag.sbatch

#SBATCH --job-name=calibrate_decoy_lag
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=04:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs results

MELODY_DIR="${MELODY_DIR:-/home/xinyue.li/POP909-Dataset/POP909-melody}"
CHORD_DIR="${CHORD_DIR:-/home/xinyue.li/POP909-Dataset/POP909-chord}"
MAX_SONGS="${MAX_SONGS:-0}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "A7 CALIBRATION: coverage vs lag   MAX_SONGS=$MAX_SONGS"
echo "MELODY_DIR=$MELODY_DIR"
echo "CHORD_DIR=$CHORD_DIR"
echo "================================================================"

python calibrate_decoy_lag.py \
    --melody-dir "$MELODY_DIR" \
    --chord-dir "$CHORD_DIR" \
    --exclude 001 002 003 004 005 \
    --max-songs "$MAX_SONGS" \
    --out results/decoy_lag_curve.csv

echo "================================================================"
echo "Done. The [bins] block above gives the A7 lag defaults; the"
echo "[VERDICT] line gates the training-code constants."
echo "================================================================"
