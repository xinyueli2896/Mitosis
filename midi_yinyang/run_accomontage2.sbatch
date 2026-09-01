#!/bin/bash
# SLURM (CPU): T8 -- AccoMontage2 harmonization of our POP909 test
# melodies (the melody-given block of the E1 external comparison).
#
# Runs run_accomontage2.py inside the AccoMontage2 venv: per test
# melody, phrase string from the hierarchical-structure-analysis labels
# + tonic key (POP909 annotation if present, else Krumhansl-Schmuckler
# estimate) -> chorderator -> <out>/<songid>/chord_gen.mid.
#
# DEFAULT IS A SMOKE TEST (LIMIT=3): three songs, loud per-song
# reporting, nonzero exit on any failure -- the label-alphabet and
# bar-alignment risks surface here before the full sweep. Then run
# LIMIT=0 for all test songs.
#
# Env knobs (--export=ALL,KEY=val):
#   MELODY_DIR  default input/pop909_split/melody
#   OUT_DIR     default temp/accomontage2_pop909
#   LIMIT       default 3 (smoke). 0 = all songs.
#
# Examples:
#   sbatch midi_yinyang/run_accomontage2.sbatch
#   sbatch --export=ALL,LIMIT=0 midi_yinyang/run_accomontage2.sbatch

#SBATCH --job-name=run_accomontage2
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"
AM="$MY/external/accomontage2"

conda activate "$CONDA_ENV"
mkdir -p /home/xinyue.li/logs

MELODY_DIR="${MELODY_DIR:-$MY/input/pop909_split/melody}"
OUT_DIR="${OUT_DIR:-$MY/temp/accomontage2_pop909}"
LIMIT="${LIMIT:-3}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "T8: AccoMontage2 harmonization   LIMIT=$LIMIT"
echo "MELODY_DIR=$MELODY_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "================================================================"

[[ -d "$AM/.venv" ]] || { echo "ERROR: run setup_accomontage2.sbatch first"; exit 1; }
# shellcheck disable=SC1091
source "$AM/.venv/bin/activate"

# chorderator resolves its bundled resources relative to the repo root.
cd "$AM"
python "$MY/run_accomontage2.py" \
    --melody-dir "$MELODY_DIR" \
    --labels-dir "$MY/external/hierarchical-structure-analysis/POP909" \
    --out-dir "$OUT_DIR" \
    --limit "$LIMIT"

echo "================================================================"
echo "Done. Outputs: $OUT_DIR/<songid>/chord_gen.mid"
echo "Next: converter into the split melody/chord eval layout, then the"
echo "harmonization-row scoring (EXPERIMENTS.md 2.6 T8)."
echo "================================================================"
