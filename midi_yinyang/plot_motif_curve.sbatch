#!/bin/bash
# SLURM (CPU): developedness against motif length -- for melody motifs
# of MIN_NOTES..MAX_NOTES notes, the share of the continuation's motifs
# that are an exact copy of / a transposition of / a transformation
# (inversion, retrograde, contour) of a prompt motif, one curve per
# system with the ground truth dashed, in NESTED layers (each contains
# the previous) and in motifs per continuation (plot_motif_curve.py).
# Writes
# <OUT>.{pdf,png} (shares) and <OUT>_delta.{pdf,png} (minus ground
# truth, paired per song).
#
# Env knobs (--export=ALL,KEY=val):
#   MANIFEST   default results/E1_p80_v5b92_manifest.tsv
#   OUT        default the manifest path with _manifest.tsv -> _motif_curve
#   OUT_ROOT   default temp/E1_v5b (reference dirs default to prompts/{mel,chord})
#   EVAL_MEL / EVAL_CHORD   reference dirs
#   PROMPT_LENGTH 80   GEN_LENGTH 416
#   MIN_NOTES 3   MAX_NOTES 8
#   EXCLUDE_SYSTEMS  comma-separated systems to leave off

#SBATCH --job-name=plot_motif_curve
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR="${REPO_DIR:-/home/xinyue.li/Mitosis}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

MANIFEST="${MANIFEST:-results/E1_p80_v5b92_manifest.tsv}"
OUT="${OUT:-${MANIFEST%_manifest.tsv}_motif_curve}"
OUT_ROOT="${OUT_ROOT:-temp/E1_v5b}"
EVAL_MEL="${EVAL_MEL:-$OUT_ROOT/prompts/mel}"
EVAL_CHORD="${EVAL_CHORD:-$OUT_ROOT/prompts/chord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-80}"
GEN_LENGTH="${GEN_LENGTH:-416}"
MIN_NOTES="${MIN_NOTES:-3}"
MAX_NOTES="${MAX_NOTES:-8}"
EXCLUDE_SYSTEMS="${EXCLUDE_SYSTEMS:-}"

[[ -f "$MANIFEST" ]] || { echo "ERROR: no $MANIFEST -- run score_e1.sbatch first"; exit 1; }

for y in raw delta; do
    suffix=""; [[ "$y" == delta ]] && suffix="_delta"
    python plot_motif_curve.py --manifest "$MANIFEST" \
        --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
        --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
        --min-notes "$MIN_NOTES" --max-notes "$MAX_NOTES" --y "$y" \
        --out "${OUT}${suffix}" ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"}
done
