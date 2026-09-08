#!/bin/bash
# SLURM (CPU): are the E1 reference files what the scorer thinks?
#
# The 14-song E1 table shows every system at harmonic_rhythm_jsd ~0.8,
# duration_jsd_b ~0.83 and density_ratio_a ~5-7 against the reference
# (5-song set: 0.08, 0.04, ~1). Six models cannot all be five times
# denser than the truth, so the reference files are the suspects: 13
# were copied from POP909-Dataset/POP909-{melody,chord}, 004 from the
# old test split.
#
# Runs audit_e1_refs.py on both reference folders (tempo, resolution,
# notes inside the scored window on the eval grid, where the file
# ends), check_beat_alignment.py on the same folders (METER / OFF-GRID
# flags), and dumps per-song A3 and S1 rows from the metrics CSV so
# 004 can be compared with the other 13.
#
# Env knobs (--export=ALL,KEY=val):
#   WS_DIR    default input/pop909_split_ws      (the 14-song refs)
#   OLD_DIR   default input/pop909_split         (the 5-song refs)
#   METRICS   default results/E1_p96_ws_metrics.csv
#
# Example:
#   sbatch midi_yinyang/audit_e1_refs.sbatch

#SBATCH --job-name=audit_e1_refs
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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

WS_DIR="${WS_DIR:-input/pop909_split_ws}"
OLD_DIR="${OLD_DIR:-input/pop909_split}"
METRICS="${METRICS:-results/E1_p96_ws_metrics.csv}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E1 reference audit: $WS_DIR vs $OLD_DIR ; metrics $METRICS"
echo "================================================================"

python audit_e1_refs.py \
    --folder "$WS_DIR/melody" --folder "$WS_DIR/chord" \
    --folder "$OLD_DIR/melody" --folder "$OLD_DIR/chord" \
    --metrics "$METRICS" --systems A3 S1 WS

echo "================================================================"
echo "--- beat alignment / grid representability ---"
python check_beat_alignment.py \
    --folder "$WS_DIR/melody" --folder "$WS_DIR/chord" \
    --folder "$OLD_DIR/melody" --folder "$OLD_DIR/chord"

echo "================================================================"
echo "Read: a reference with window=0 or end_fr < 416 is scored against"
echo "nothing; a tempo far from the 5-song files' tempo, or a METER /"
echo "OFF-GRID flag, means the file is on a different grid than the"
echo "models were prompted on. Song 004 is the control (old-split file)."
echo "================================================================"
