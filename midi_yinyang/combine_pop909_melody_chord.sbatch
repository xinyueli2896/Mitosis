#!/bin/bash
# SLURM batch script (CPU): build combined melody+chord midis for
# beat-aligned POP909 songs via combine_pop909_melody_chord.py.
# Output: <DST_DIR>/<id>.mid with tempo + MELODY + CHORD tracks.
#
# Env knobs (override via --export=ALL,KEY=val):
#   ALIGNED_DIR  beat-aligned midis (tempo + MELODY track).
#                default: /home/xinyue.li/POP909-Dataset/POP909-aligned
#   POP909_RAW   raw dataset root with <id>/chord_midi.txt.
#                default: /home/xinyue.li/POP909-Dataset/POP909
#   DST_DIR      default: /home/xinyue.li/POP909-Dataset/POP909-melody-chord
#   IDS          space-separated song ids; empty = all songs.
#
# Examples:
#   # just 001-005 (rebuilding the lost ones):
#   sbatch --export=ALL,IDS="001 002 003 004 005" \
#          midi_yinyang/combine_pop909_melody_chord.sbatch
#   # all 909 songs:
#   sbatch midi_yinyang/combine_pop909_melody_chord.sbatch

#SBATCH --job-name=mitosis_combine_pop909
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

ALIGNED_DIR="${ALIGNED_DIR:-/home/xinyue.li/POP909-Dataset/POP909-aligned}"
POP909_RAW="${POP909_RAW:-/home/xinyue.li/POP909-Dataset/POP909}"
DST_DIR="${DST_DIR:-/home/xinyue.li/POP909-Dataset/POP909-melody-chord}"
IDS="${IDS:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "ALIGNED_DIR = $ALIGNED_DIR"
echo "POP909_RAW  = $POP909_RAW"
echo "DST_DIR     = $DST_DIR"
echo "IDS         = ${IDS:-<all songs>}"
echo "================================================================"

EXTRA_ARGS=()
if [[ -n "$IDS" ]]; then
    EXTRA_ARGS=(--ids $IDS)
fi

python combine_pop909_melody_chord.py \
    --aligned "$ALIGNED_DIR" \
    --src "$POP909_RAW" \
    --dst "$DST_DIR" \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Done. Combined midis under $DST_DIR"
