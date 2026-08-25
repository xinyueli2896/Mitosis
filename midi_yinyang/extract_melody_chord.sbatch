#!/bin/bash
# SLURM (CPU): extract melody + chord streams from a solo-piano corpus
# (Pop1K7, an Aria-MIDI subset, ...) into POP909-format paired folders.
#
# Task 2 (dataset expansion). POP909 shipped a labeled MELODY track and
# chord annotations; piano-only corpora ship neither, so this runs the
# literature-standard recipe via extract_melody_chord.py: skyline
# melody + duration-weighted pitch-class template chords, rendered to
# <DST_PREFIX>-melody/ and <DST_PREFIX>-chord/ with the source's own
# tempo track -- the exact folder format every downstream stage already
# consumes (check_paired_dataset.py, the melchord preprocess sbatches,
# combine_melchord_datasets.py, preprocess_pop909_yinyang.sbatch-style
# paired builds).
#
# UPSTREAM REQUIREMENT: sources must be BEAT-ALIGNED midis. Pop1K7's
# released midi_analyzed/midi_synchronized folders are (madmom beat
# tracking); RAW Aria-MIDI transcriptions are NOT -- always run
# check_beat_alignment.sbatch on the extraction output before
# tokenizing, and expect raw Aria to need a quantization stage first.
#
# Env knobs (--export=ALL,KEY=val):
#   SRC          REQUIRED. Folder of beat-aligned solo-piano midis.
#   DST_PREFIX   REQUIRED. e.g. Pop1K7-Dataset/pop1k7 produces
#                Pop1K7-Dataset/pop1k7-melody/ + .../pop1k7-chord/
#   LIMIT        optional: first N files only (use for a pilot pass).
#   CHORD_BEATS  chord window in beats (default 2).
#   MIN_MELODY_PITCH  skyline floor (default 55 = G3).
#
# Examples:
#   sbatch --exclude=cn-20 --export=ALL,\
#     SRC=Pop1K7-Dataset/midi_analyzed,DST_PREFIX=Pop1K7-Dataset/pop1k7,LIMIT=50 \
#     midi_yinyang/extract_melody_chord.sbatch          # pilot
#   sbatch --exclude=cn-20 --export=ALL,\
#     SRC=Pop1K7-Dataset/midi_analyzed,DST_PREFIX=Pop1K7-Dataset/pop1k7 \
#     midi_yinyang/extract_melody_chord.sbatch          # full corpus

#SBATCH --job-name=mitosis_extract_melchord
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=04:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

SRC="${SRC:-}"
DST_PREFIX="${DST_PREFIX:-}"
LIMIT="${LIMIT:-}"
CHORD_BEATS="${CHORD_BEATS:-2}"
MIN_MELODY_PITCH="${MIN_MELODY_PITCH:-55}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

for v in SRC DST_PREFIX; do
    [[ -n "${!v}" ]] || { echo "ERROR: $v is required (see header)"; exit 1; }
done
[[ -d "$SRC" ]] || { echo "ERROR: SRC folder not found: $SRC"; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "MELODY/CHORD EXTRACTION  $SRC -> ${DST_PREFIX}-{melody,chord}/"
echo "chord window ${CHORD_BEATS} beats  skyline floor $MIN_MELODY_PITCH"
echo "================================================================"

ARGS=(--src "$SRC" --dst-prefix "$DST_PREFIX"
      --chord-beats "$CHORD_BEATS" --min-melody-pitch "$MIN_MELODY_PITCH")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")

python extract_melody_chord.py "${ARGS[@]}"

echo "================================================================"
echo "Grid check (REQUIRED before tokenizing a new corpus):"
echo "  sbatch --exclude=cn-20 --export=ALL,FOLDERS=\"${DST_PREFIX}-melody ${DST_PREFIX}-chord\" \\"
echo "      midi_yinyang/check_beat_alignment.sbatch"
echo "================================================================"
