#!/bin/bash
# SLURM (CPU): bar-phase diagnostic for the aligned POP909 data
# (check_downbeat_phase.py). Reads the RAW beat_midi.txt files, reports
# the first-downbeat index per song and the bar shift the current
# aligner introduces, histograms it over the corpus, and checks the
# aligned files of the listed songs.
#
# Env knobs (--export=ALL,KEY=val):
#   POP909_RAW   default /home/xinyue.li/POP909-Dataset/POP909
#   ALIGNED_DIR  default /home/xinyue.li/POP909-Dataset/POP909-aligned
#   CHORD_DIR    default /home/xinyue.li/POP909-Dataset/POP909-chord (the
#                files actually used; phase measured from chord changes)
#   IDS          default the 8 E1 songs

#SBATCH --job-name=check_downbeat_phase
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:15:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs

POP909_RAW="${POP909_RAW:-/home/xinyue.li/POP909-Dataset/POP909}"
ALIGNED_DIR="${ALIGNED_DIR:-/home/xinyue.li/POP909-Dataset/POP909-aligned}"
CHORD_DIR="${CHORD_DIR:-/home/xinyue.li/POP909-Dataset/POP909-chord}"
IDS="${IDS:-004 046 136 326 456 466 746 816}"

python check_downbeat_phase.py --raw "$POP909_RAW" --aligned "$ALIGNED_DIR" --chord-dir "$CHORD_DIR" --ids $IDS
