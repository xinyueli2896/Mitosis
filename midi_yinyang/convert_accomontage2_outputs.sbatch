#!/bin/bash
# SLURM (CPU): T8 conversion -- AccoMontage2 chord_gen.mid outputs into
# the eval-ready combined layout (tracks named MELODY / CHORD, melody
# program 0 / chord program 48), one midi per song. Includes a per-song
# melody-fingerprint check against the input melody so swapped
# instruments can never slip into scoring.
#
# Env knobs (--export=ALL,KEY=val):
#   IN_DIR      default temp/accomontage2_pop909
#   MELODY_DIR  default input/pop909_split/melody
#   OUT_DIR     default temp/accomontage2_pop909_eval
#
# Example:
#   sbatch midi_yinyang/convert_accomontage2_outputs.sbatch

#SBATCH --job-name=convert_accomontage2
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

IN_DIR="${IN_DIR:-temp/accomontage2_pop909}"
MELODY_DIR="${MELODY_DIR:-input/pop909_split/melody}"
OUT_DIR="${OUT_DIR:-temp/accomontage2_pop909_eval}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "T8 CONVERT: $IN_DIR -> $OUT_DIR"
echo "================================================================"

python convert_accomontage2_outputs.py \
    --in-dir "$IN_DIR" \
    --melody-dir "$MELODY_DIR" \
    --out-dir "$OUT_DIR"

echo "================================================================"
echo "Done. Next: the harmonization-row scoring via eval_metrics with"
echo "mode mel2chord semantics (melody is GIVEN; chord-side and"
echo "inter-stream metrics only)."
echo "================================================================"
