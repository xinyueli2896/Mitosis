#!/bin/bash
# SLURM (CPU): build the dataset-order downbeat map, so training windows
# start on REAL bar lines.
#
# FramedDataset snaps a window offset with `offset -= offset % 16`, which
# assumes sixteen frames to a bar for a whole song. POP909's annotation
# disagrees: 783 of 909 songs contain a bar that is not four beats, and
# after the first one the real downbeats stop being multiples of 16.
# Measured over the corpus, 46.5% of all beats sit after their song's
# first irregular bar -- so nearly half of every model's training windows
# begin on a "bar line" that is not one, while the snapping itself
# teaches the model that frame 0 of a window always is.
#
# Writes data/<DATASET>.downbeats.pt. The trainer picks it up
# automatically from beside the dataset, so building it is enough to turn
# the behaviour on, and its absence changes nothing.
#
# Env knobs (--export=ALL,KEY=val):
#   DATASET   dataset stem under data/ (default pop909_melody_cp8_v2_v5)
#   REPORT    align_grid_report.tsv (default the v5 aligner output)
#
#   sbatch midi_yinyang/build_downbeat_map.sbatch

#SBATCH --job-name=build_downbeat_map
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
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs

POP=/home/xinyue.li/POP909-Dataset
DATASET="${DATASET:-pop909_melody_cp8_v2_v5}"
REPORT="${REPORT:-$POP/POP909-v5/align_grid_report.tsv}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "DATASET=$DATASET   REPORT=$REPORT"
echo "================================================================"

python build_downbeat_map.py --dataset "$DATASET" --report "$REPORT"

# The duet reads the map beside its mod_b (chord) dataset, so give the
# chord stem the same file. Both streams share one downbeat map by
# construction -- the aligner cuts them on the same grid.
CHORD_STEM="${DATASET/_melody_/_chord_}"
if [[ "$CHORD_STEM" != "$DATASET" && -f "data/$CHORD_STEM.txt" ]]; then
    python build_downbeat_map.py --dataset "$CHORD_STEM" --report "$REPORT"
fi

echo "================================================================"
echo "Done. The trainer picks these up automatically; the run header"
echo "will say 'Downbeat map loaded from ...' once it does."
echo "================================================================"
