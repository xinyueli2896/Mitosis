#!/bin/bash
# SLURM (CPU): one-page E1 summary figure -- the paper's duet against the
# baselines across H3 / H2 / H1 and the coherence columns (small
# multiples; see e1_summary_figure.py). Needs the merged metrics CSV and
# the merged coherence CSV (merge_e1.sbatch, score_coherence.sbatch).
#
# Env knobs (--export=ALL,KEY=val):
#   RUN        metrics suffix (default mixed_matched)
#   TAG        coherence tag (default matched -> results/E1_mixed_matched_coherence.csv)
#   SYSTEMS    rows, first drawn largest (default the presentation line-up)
#   BASELINE   default the first of SYSTEMS
#
# Example:
#   sbatch --export=ALL,SYSTEMS="A3ctcaT A1 S1 S-scratch P-mc P-cm WS" midi_yinyang/e1_summary_figure.sbatch

#SBATCH --job-name=e1_summary_figure
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:10:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs results

RUN="${RUN:-mixed_matched}"
TAG="${TAG:-matched}"
SYSTEMS="${SYSTEMS:-A3ctcaT A1 S1 S-scratch P-mc P-cm WS}"
BASELINE="${BASELINE:-${SYSTEMS%% *}}"
METRICS="results/E1_${RUN}_metrics.csv"
COH="results/E1_mixed_${TAG}_coherence.csv"
[[ -f "$METRICS" ]] || { echo "ERROR: $METRICS missing"; exit 1; }
[[ -f "$COH" ]] || { echo "ERROR: $COH missing (run score_coherence.sbatch)"; exit 1; }

python e1_summary_figure.py --metrics "$METRICS" --coherence "$COH" \
    --systems $SYSTEMS --baseline "$BASELINE" --out "results/E1_summary_${TAG}"
echo "figure: results/E1_summary_${TAG}.pdf  (.png alongside)"
