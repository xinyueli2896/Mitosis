#!/bin/bash
# SLURM (CPU): paper-ready E1 table (LaTeX) and H2 figure from a metrics CSV.
#
# Runs e1_paper_table.py on results/E1_<RUN>_metrics.csv and writes
# results/E1_paper_<RUN>.tex / .pdf / .png. Song counts and p-values
# are recomputed from the CSV; nothing is copied from console output.
#
# Env knobs (--export=ALL,KEY=val):
#   RUN        metrics suffix (default mixed_matched ->
#              results/E1_mixed_matched_metrics.csv, the merged 8-song set)
#   BASELINE   default A3ctcaT (alternating commit, follower at T=1 / no
#              nucleus cut: the paper's symmetric decode; A3ctca = the
#              same at T=0.9 / top-p 0.95, A3 = refine K=4 ablation,
#              A3ctc/A3ctcc = one-directional diagnostics)
#
# Example:
#   sbatch midi_yinyang/e1_paper_table.sbatch
#   sbatch --export=ALL,RUN=p96_matched midi_yinyang/e1_paper_table.sbatch

#SBATCH --job-name=e1_paper_table
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
BASELINE="${BASELINE:-A3ctcaT}"
CSV="results/E1_${RUN}_metrics.csv"
[[ -f "$CSV" ]] || { echo "ERROR: $CSV missing"; exit 1; }

python e1_paper_table.py --csv "$CSV" --out "results/E1_paper_${RUN}" --baseline "$BASELINE"
echo "----- results/E1_paper_${RUN}.tex -----"
cat "results/E1_paper_${RUN}.tex"
