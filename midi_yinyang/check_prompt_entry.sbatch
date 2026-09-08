#!/bin/bash
# SLURM (CPU): which of the shared-clean POP909 songs have both melody
# and chord present within the first N bars, for several N?
#
# Reads the TIME-ALIGNED source folders (the staged copies in
# input/909_matched_split are cropped to the co-entry bar and would
# all pass). Frames on the tick grid. Prints entry bar per stream, onset
# counts per window, and the songs that pass at the larger N but not
# the smaller one.
#
# Env knobs (--export=ALL,KEY=val):
#   SONGS   default: the 14 shared-clean ids
#   BARS    default "6 8"
#
# Example:
#   sbatch midi_yinyang/check_prompt_entry.sbatch
#   sbatch --export=ALL,BARS="4 6 8 10" midi_yinyang/check_prompt_entry.sbatch

#SBATCH --job-name=check_prompt_entry
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
mkdir -p /home/xinyue.li/logs

SONGS="${SONGS:-004 046 136 196 276 326 366 456 466 626 656 746 806 816}"
BARS="${BARS:-6 8}"
MEL_SRC="${MEL_SRC:-/home/xinyue.li/POP909-Dataset/POP909-melody}"
CHORD_SRC="${CHORD_SRC:-/home/xinyue.li/POP909-Dataset/POP909-chord}"

echo "Job ${SLURM_JOB_ID:-local}  songs: $SONGS   bars: $BARS"
python check_prompt_entry.py --songs $SONGS --bars $BARS \
    --mel-src "$MEL_SRC" --chord-src "$CHORD_SRC"
