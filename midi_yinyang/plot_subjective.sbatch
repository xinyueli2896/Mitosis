#!/bin/bash
# SLURM (CPU): the subjective-evaluation spider chart and the rater
# demographics, from the listening page's ratings CSV.
#
# Env knobs (--export=ALL,KEY=val):
#   RATINGS   the CSV (required)
#   OUT       path without extension; default results/subjective
#   EXCLUDE   rater ids to drop, space-separated (default: none)
#   RMIN      inner edge of the radial axis, default 2
#
# Example:
#   sbatch --export=ALL,RATINGS=input/ratings.csv,EXCLUDE="azure-stoat-9403 violet-quail-2727" \
#       midi_yinyang/plot_subjective.sbatch

#SBATCH --job-name=plot_subjective
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:20:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR="${REPO_DIR:-/home/xinyue.li/Mitosis}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs results

RATINGS="${RATINGS:?set RATINGS=<csv>}"
OUT="${OUT:-results/subjective}"
EXCLUDE="${EXCLUDE:-}"
RMIN="${RMIN:-2}"

python subjective_spider.py --csv "$RATINGS" --out "$OUT" --rmin "$RMIN" \
    ${EXCLUDE:+--exclude $EXCLUDE}
