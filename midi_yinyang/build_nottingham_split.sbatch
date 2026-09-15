#!/bin/bash
# SLURM (CPU): stage Nottingham tunes as a melody/chord prompt split.
#
# Reads <MELODY_SRC>/<tune>.mid and <CHORD_SRC>/<tune>.mid for each tune
# in TUNES and writes <DST>/melody/<tune>.mid + <DST>/chord/<tune>.mid,
# the layout every prompt tool here reads (heldout_v5 style). A tune
# that opens with a PICKUP bar is copied as is; one that starts on the
# downbeat gets ONE EMPTY BAR of silence in front of both files, so the
# first downbeat sits a bar in either way. The bar length follows the
# file's own time signature.
#
# Pickup detection reads the chord track: chord symbols sit on
# downbeats, so a first chord that starts a fraction of a bar in marks
# that much anacrusis. The log prints, per tune, the time signature,
# the first melody and chord onsets in beats, and the decision with its
# reason -- READ IT, and force any tune the rule gets wrong with
# PICKUP="tune=1 tune=0" (1 = has a pickup, copy; 0 = none, pad).
#
# FORMAT = heldout_v5: a pure tick-space copy, programs untouched, tempo
# map replaced by one set_tempo at TEMPO bpm at tick 0 (120, as
# build_prompt_crops writes). NOT tagged here -- trial_a12 tags the
# chord at staging (CHORD_PROGRAM), exactly as the E1 tree does.
#
# Env knobs (--export=ALL,KEY=val):
#   MELODY_SRC   default /home/xinyue.li/nottingham-dataset/MIDI/melody
#   CHORD_SRC    default /home/xinyue.li/nottingham-dataset/MIDI/chords
#   DST          default input/nottingham_split
#   TUNES        default "ashover5 hpps9 morris2 playford15 xmas8"
#   TEMPO        default 120; 0 keeps the source tempo map
#   CHORD_PROGRAM empty (default) leaves programs as in the source
#   PICKUP       overrides, e.g. "morris2=1 xmas8=0"
#
# Example:
#   sbatch midi_yinyang/build_nottingham_split.sbatch
#   sbatch --export=ALL,PICKUP="xmas8=0" midi_yinyang/build_nottingham_split.sbatch

#SBATCH --job-name=build_nottingham_split
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:10:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

MELODY_SRC="${MELODY_SRC:-/home/xinyue.li/nottingham-dataset/MIDI/melody}"
CHORD_SRC="${CHORD_SRC:-/home/xinyue.li/nottingham-dataset/MIDI/chords}"
DST="${DST:-input/nottingham_split}"
TUNES="${TUNES:-ashover5 hpps9 morris2 playford15 xmas8}"
CHORD_PROGRAM="${CHORD_PROGRAM:-}"
TEMPO="${TEMPO:-120}"
PICKUP="${PICKUP:-}"

[[ -d "$MELODY_SRC" && -d "$CHORD_SRC" ]] || {
    echo "ERROR: source folders missing: $MELODY_SRC / $CHORD_SRC"; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "SRC=$MELODY_SRC | $CHORD_SRC"
echo "DST=$DST  TUNES=$TUNES  TEMPO=$TEMPO  CHORD_PROGRAM=${CHORD_PROGRAM:-<untouched>}  PICKUP=${PICKUP:-<detected>}"
echo "================================================================"

python build_nottingham_split.py --melody-dir "$MELODY_SRC" \
    --chord-dir "$CHORD_SRC" --dst "$DST" --tunes $TUNES \
    --tempo "$TEMPO" ${CHORD_PROGRAM:+--chord-program "$CHORD_PROGRAM"} \
    ${PICKUP:+--pickup $PICKUP}

echo "================================================================"
echo "Staged. Decode them with trial_a12.sbatch:"
echo "  sbatch --exclude=gpu-51,gpu-05,gpu-54 --export=ALL,MEL_SRC=$DST/melody,\\"
echo "CHORD_SRC=$DST/chord,SONG_IDS=\"$TUNES\" midi_yinyang/trial_a12.sbatch"
echo "================================================================"
