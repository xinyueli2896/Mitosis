#!/bin/bash
# SLURM (CPU): does a system's output carry OUR prompt in its prompt bars?
#
# Loads the staged prompt and each system's output over the prompt
# window and reports, per stream, onset agreement, pitch-class
# agreement, absolute-pitch agreement and the best timing shift, with a
# one-line verdict per system. Run it on the external baseline AND on
# one of ours as a control: ours copy the prompt verbatim and score 1.0
# everywhere, so any figure below that on the baseline is the baseline.
#
# The whole-song baseline in particular re-renders the chord: its
# lead-sheet image stores a chord as a chroma band and a bass band, one
# row per pitch class, and its converter writes those back out at its
# own octave and voicing. That shows up here as chord pc ~1.0 with
# chord pitch well below 1.0 -- the pitch classes came through, the
# voicing is theirs. A low pc figure, or a non-zero shift, is a real
# mismatch and the arm needs looking at.
#
# Env knobs (--export=ALL,KEY=val):
#   OUT_ROOT        E1 tree, default temp/E1_v5b
#   SYSTEMS         default "WSf A3ctcaT"
#   PROMPT_FRAMES   default 96
#   SAMPLE          which sample to read, default 0
#   EXCLUDE_SONGS   default "506 786", as eval_e1
#   SHOW            songs with the lowest chord agreement to print beat
#                   by beat, default 3
#
# Example:
#   sbatch --export=ALL,OUT_ROOT=temp/E1_v5b,SYSTEMS="WSf A3ctcaT" \
#       midi_yinyang/check_prompt_copy.sbatch

#SBATCH --job-name=check_prompt_copy
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

OUT_ROOT="${OUT_ROOT:-temp/E1_v5b}"
SYSTEMS="${SYSTEMS:-WSf A3ctcaT}"
PROMPT_FRAMES="${PROMPT_FRAMES:-96}"
SAMPLE="${SAMPLE:-0}"
EXCLUDE_SONGS="${EXCLUDE_SONGS-506 786}"
SHOW="${SHOW:-3}"
MEL_DIR="${MEL_DIR:-$OUT_ROOT/prompts/mel}"
CHORD_DIR="${CHORD_DIR:-$OUT_ROOT/prompts/chord}"

[[ -d "$MEL_DIR" && -d "$CHORD_DIR" ]] || {
    echo "ERROR: staged prompts missing under $OUT_ROOT/prompts"; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "OUT_ROOT=$OUT_ROOT  SYSTEMS=$SYSTEMS  PROMPT_FRAMES=$PROMPT_FRAMES  SAMPLE=$SAMPLE"
echo "================================================================"

python check_prompt_copy.py --out-root "$OUT_ROOT" --systems $SYSTEMS \
    --mel-dir "$MEL_DIR" --chord-dir "$CHORD_DIR" \
    --prompt-frames "$PROMPT_FRAMES" --sample "$SAMPLE" --show "$SHOW" \
    ${EXCLUDE_SONGS:+--exclude-songs $EXCLUDE_SONGS}
