#!/bin/bash
# SLURM (CPU): rescore ONLY the corpus-pooled divergences of an existing
# E1 run and rebuild the general-quality table from them. No generation,
# no per-song metrics, no paired table: eval_metrics --pooled-only over
# the manifest that score_e1 already wrote, then e1_quality_table.
#
# Use it after a change to the pooled sheet (e.g. the per-sample pooling
# of 2026-09-23) when the per-song CSV is still valid.
#
# Env knobs (--export=ALL,KEY=val):
#   MANIFEST   the E1 manifest TSV (required), e.g.
#              results/E1_p80_v5b92_lora_manifest.tsv
#   OUT_ROOT   the E1 tree whose staged prompts are the references
#              (default temp/E1_v5b)
#   TAG        results prefix; writes results/<TAG>_pooled.csv and
#              figures/<TAG>_quality.tex (default: the manifest's stem
#              without _manifest)
#   PROMPT_LENGTH / GEN_LENGTH   scored window (default 80 / 416)
#   POOLED_BOOT   bootstrap replicates for the song-mixed interval
#              (default 2000; the per-sample values do not use it)
#   QUALITY_METRICS  h3 (default here) | gcsc   table columns
#   QUALITY_PER_SAMPLE  1 (default here): per-sample pooled values
#   BOLD_BEST / BOLD_EXCLUDE / GROUP_COLUMN   table marks and layout
#              (default 1 / S1 / 1)
#   EXCLUDE_SYSTEMS  internal names left off the table, space-separated
#
#   sbatch --export=ALL,MANIFEST=results/E1_p80_v5b92_lora_manifest.tsv \
#       midi_yinyang/score_pooled.sbatch

#SBATCH --job-name=score_pooled
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
set +u; source ~/.bashrc 2>/dev/null || true; set -u

CONDA_ENV="${CONDA_ENV-mitosis}"
REPO_DIR="${REPO_DIR:-/home/xinyue.li/Mitosis}"
[[ -d "$REPO_DIR" ]] || REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
fi
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs 2>/dev/null || true

MANIFEST="${MANIFEST:?set MANIFEST=<E1 manifest TSV>}"
OUT_ROOT="${OUT_ROOT:-temp/E1_v5b}"
TAG="${TAG:-$(basename "${MANIFEST%_manifest.tsv}")}"
PROMPT_LENGTH="${PROMPT_LENGTH:-80}"
GEN_LENGTH="${GEN_LENGTH:-416}"
POOLED_BOOT="${POOLED_BOOT:-2000}"
[[ -f "$MANIFEST" ]] || { echo "ERROR: $MANIFEST not found"; exit 1; }
[[ -d "$OUT_ROOT/prompts/mel" && -d "$OUT_ROOT/prompts/chord" ]] || {
    echo "ERROR: no staged prompts under $OUT_ROOT/prompts"; exit 1; }
mkdir -p results figures

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "POOLED-ONLY RESCORE  manifest=$MANIFEST  refs=$OUT_ROOT/prompts  window=$PROMPT_LENGTH..$GEN_LENGTH"
echo "-> results/${TAG}_pooled.csv  figures/${TAG}_quality.tex"
echo "================================================================"

python eval_metrics.py --task melchord --manifest "$MANIFEST" \
    --ref-a-dir "$OUT_ROOT/prompts/mel" --ref-b-dir "$OUT_ROOT/prompts/chord" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --pooled-only --pooled-out "results/${TAG}_pooled.csv" \
    --pooled-boot "$POOLED_BOOT" \
    --pooled-boot-mode "${POOLED_BOOT_MODE:-songs}" --pooled-weight "${POOLED_WEIGHT:-song}"

python e1_quality_table.py --pooled "results/${TAG}_pooled.csv" \
    --out "figures/${TAG}_quality.tex" \
    --metrics "${QUALITY_METRICS:-h3}" \
    $([[ "${QUALITY_PER_SAMPLE:-1}" == "1" ]] && echo --per-sample) \
    $([[ "${BOLD_BEST:-1}" == "1" ]] || echo --no-bold) \
    --bold-exclude "${BOLD_EXCLUDE-S1}" \
    $([[ "${GROUP_COLUMN:-1}" == "1" ]] && echo --group-column) \
    ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"}
