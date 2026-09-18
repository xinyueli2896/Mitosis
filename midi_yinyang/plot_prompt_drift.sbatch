#!/bin/bash
# SLURM (CPU): prompt drift across the continuation -- pitch-class JSD
# against the prompt, bar by bar, for every scored system and the
# ground truth, plus the paired delta (plot_prompt_drift.py).
#
# Writes <OUT>.{pdf,png} (raw statistics) and <OUT>_delta.{pdf,png}
# (paired difference against the ground truth), each a 2 x 2 grid:
# rows pitch-class JSD / onset similarity (EXTRA=1 adds groove and
# scale consistency to the prompt), columns melody /
# chord.
#
# Env knobs (--export=ALL,KEY=val):
#   MANIFEST   default results/E1_p80_v5b92_manifest.tsv
#   OUT        path without extension; default the manifest path with
#              _manifest.tsv -> _drift
#   OUT_ROOT   default temp/E1_v5b (reference dirs default to its
#              prompts/{mel,chord})
#   EVAL_MEL / EVAL_CHORD   reference dirs
#   PROMPT_LENGTH 80   GEN_LENGTH 416
#   WINDOW     bar (default: each bar alone) | cumulative (bars 0..i)
#   BARS       bars per window, default 1
#   EXCLUDE_SYSTEMS  comma-separated systems to leave off
#   XTICK      x tick every N bars, default 4
#
# Example:
#   sbatch --export=ALL,MANIFEST=results/E1_p80_v5b92_manifest.tsv \
#       midi_yinyang/plot_prompt_drift.sbatch

#SBATCH --job-name=plot_prompt_drift
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
OUT="${OUT:-${MANIFEST%_manifest.tsv}_drift}"
OUT_ROOT="${OUT_ROOT:-temp/E1_v5b}"
EVAL_MEL="${EVAL_MEL:-$OUT_ROOT/prompts/mel}"
EVAL_CHORD="${EVAL_CHORD:-$OUT_ROOT/prompts/chord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-80}"
GEN_LENGTH="${GEN_LENGTH:-416}"
WINDOW="${WINDOW:-bar}"
BARS="${BARS:-1}"
EXCLUDE_SYSTEMS="${EXCLUDE_SYSTEMS:-}"
XTICK="${XTICK:-4}"
EXTRA="${EXTRA:-0}"

[[ -f "$MANIFEST" ]] || { echo "ERROR: no $MANIFEST -- run score_e1.sbatch first"; exit 1; }

echo "MANIFEST=$MANIFEST  OUT=$OUT  WINDOW=$WINDOW  BARS=$BARS  PROMPT=$PROMPT_LENGTH  TOTAL=$GEN_LENGTH"
python plot_prompt_drift.py --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --window "$WINDOW" --bars "$BARS" --out "$OUT" --xtick "$XTICK" $([[ "$EXTRA" == "1" ]] && echo --extra) \
    ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"}
