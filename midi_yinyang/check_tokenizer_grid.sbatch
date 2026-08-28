#!/bin/bash
# SLURM (CPU): tokenizer's-eye-view grid check -- are onsets quantized
# in TICK space, the only domain the tokenizer reads?
#
# Companion to check_beat_alignment.sbatch, which measures in seconds
# through the tempo map and therefore MISJUDGES beat-synced
# transcriptions (Pop1K7): tick grid clean, per-beat tempo events
# carrying performance timing. This check loads files exactly the way
# preprocess_large_midi_dataset does (XFMidi constant_tempo -- tempo
# map discarded, ticks -> subbeats) and measures onset deviation from
# the subbeat grid there. See check_tokenizer_grid.py's docstring for
# how to read the two checks together.
#
# Env knobs (--export=ALL,KEY=val):
#   FOLDERS   space-separated folders.
#             default "input/pop1k7_test_split/melody input/pop1k7_test_split/chord"
#   BEAT_DIV  default 4 (16th notes)
#
# Example:
#   sbatch midi_yinyang/check_tokenizer_grid.sbatch

#SBATCH --job-name=tokenizer_grid
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

FOLDERS="${FOLDERS:-input/pop1k7_test_split/melody input/pop1k7_test_split/chord}"
BEAT_DIV="${BEAT_DIV:-4}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "TOKENIZER-VIEW GRID CHECK   folders: $FOLDERS"
echo "================================================================"

FOLDER_ARGS=()
for f in $FOLDERS; do FOLDER_ARGS+=(--folder "$f"); done

python check_tokenizer_grid.py "${FOLDER_ARGS[@]}" --beat-div "$BEAT_DIV"
