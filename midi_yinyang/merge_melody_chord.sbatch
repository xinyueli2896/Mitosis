#!/bin/bash
# SLURM batch script (CPU): merge paired melody/chord midi folders into
# combined midis via merge_melody_chord.py.
# Output: <DST_DIR>/<id>.mid with tempo + MELODY + CHORD tracks (names
# forced, so the outputs work with split_melody_chord.py and the
# track-name separation in cp_transformer_inference.py).
#
# Env knobs (override via --export=ALL,KEY=val):
#   MELODY_DIR   default: POP909-Dataset/POP909-melody   (repo-relative)
#   CHORD_DIR    default: POP909-Dataset/POP909-chord
#   DST_DIR      default: POP909-Dataset/POP909-melody-chord
#   IDS          space-separated song ids; empty = all paired songs.
#
# Examples:
#   sbatch midi_yinyang/merge_melody_chord.sbatch
#   IDS="001 002 003 004 005" DST_DIR=/some/where \
#       sbatch --export=ALL midi_yinyang/merge_melody_chord.sbatch

#SBATCH --job-name=mitosis_merge_melchord
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

MELODY_DIR="${MELODY_DIR:-POP909-Dataset/POP909-melody}"
CHORD_DIR="${CHORD_DIR:-POP909-Dataset/POP909-chord}"
DST_DIR="${DST_DIR:-POP909-Dataset/POP909-melody-chord}"
IDS="${IDS:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "MELODY_DIR = $MELODY_DIR ($(ls "$MELODY_DIR" 2>/dev/null | wc -l) files)"
echo "CHORD_DIR  = $CHORD_DIR ($(ls "$CHORD_DIR" 2>/dev/null | wc -l) files)"
echo "DST_DIR    = $DST_DIR"
echo "IDS        = ${IDS:-<all paired songs>}"
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "$IDS" ]]; then
    EXTRA_ARGS+=(--ids $IDS)
fi

python merge_melody_chord.py \
    --melody "$MELODY_DIR" \
    --chord "$CHORD_DIR" \
    --dst "$DST_DIR" \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Done. Combined midis under $DST_DIR"
