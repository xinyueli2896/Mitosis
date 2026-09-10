#!/bin/bash
# SLURM (CPU): beat-align POP909 from the raw dataset with
# pop909_beat_align.py -- ONE beat map for the melody, every other track
# and the chords rendered from chord_midi.txt; first downbeat on a bar
# boundary; no lead-in artefacts. Writes <DST>/{aligned,melody,chord}/
# and <DST>/align_report.tsv, then measures the bar phase of the chord
# folder with check_downbeat_phase.py as a cross-check.
#
# Env knobs (--export=ALL,KEY=val):
#   POP909_RAW  song folders of the dataset repo
#               (default /home/xinyue.li/POP909-Dataset/POP909-git/POP909)
#   DST         output root (default /home/xinyue.li/POP909-Dataset/POP909-v3)
#   IDS         optional song ids (default: all)
#   NO_PAD      1 = beat 0 at tick 0, downbeats not put on the bar grid
#
# Downstream: stage E1 prompts from $DST/melody and $DST/chord
# (audit_wholesong_split.sbatch MEL_SRC= CHORD_SRC= STAGE_DIR=).
# The training .pt files are NOT rebuilt here (random crop windows with
# relative positions never see absolute bar phase).

#SBATCH --job-name=align_pop909
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs

POP909_RAW="${POP909_RAW:-/home/xinyue.li/POP909-Dataset/POP909-git/POP909}"
DST="${DST:-/home/xinyue.li/POP909-Dataset/POP909-v3}"
IDS="${IDS:-}"
NO_PAD="${NO_PAD:-0}"

[[ -d "$POP909_RAW/004" ]] || { echo "ERROR: $POP909_RAW has no song folders (clone the POP909-Dataset repo)"; exit 1; }

python pop909_beat_align.py --raw "$POP909_RAW" --dst "$DST" \
    ${IDS:+--ids $IDS} $([[ "$NO_PAD" == "1" ]] && echo --no-pad)

echo "================================================================"
echo "bar-phase cross-check on $DST/chord"
python check_downbeat_phase.py --raw "$POP909_RAW" --chord-dir "$DST/chord" \
    --ids 004 046 136 326 456 466 746 816
echo "report: $DST/align_report.tsv"
