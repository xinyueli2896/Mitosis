#!/bin/bash
# SLURM (CPU): make already-generated E1 outputs playable by replacing
# the tempo they inherited from the beat-synced prompt's lead-in event
# (1289 bpm on 326, 1324 on 046, plausible-looking but arbitrary values
# on the rest) with the prompt's main tempo. Any single-event output
# whose tempo differs from that by more than 2% is rewritten, so the
# 120 bpm hybrids and the 90 bpm whole-song files are brought to the
# same tempo as well. Ticks are untouched, so scores do not change
# (verified: identical scorer streams at 1289 and 58 bpm, and after the
# rewrite); the staged prompts/references under <root>/prompts are
# skipped.
#
# Env knobs (--export=ALL,KEY=val):
#   ROOTS    space-separated output roots (default the two matched groups)
#   DRY_RUN  1 = list what would change, write nothing
#
# Example:
#   sbatch --export=ALL,DRY_RUN=1 midi_yinyang/fix_output_tempo.sbatch
#   sbatch midi_yinyang/fix_output_tempo.sbatch

#SBATCH --job-name=fix_output_tempo
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
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

ROOTS="${ROOTS:-temp/E1_matched_p6 temp/E1_matched_p8}"
DRY_RUN="${DRY_RUN:-0}"
ARGS=()
for r in $ROOTS; do ARGS+=(--root "$r"); done
[[ "$DRY_RUN" == "1" ]] && ARGS+=(--dry-run)

python fix_output_tempo.py "${ARGS[@]}"
