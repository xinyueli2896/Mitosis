#!/bin/bash
# SLURM (CPU): score several E1 runs as ONE table.
#
# The matched POP909 test set is staged in prompt-length groups
# (audit_wholesong_split: p6 = both streams present in the first 6
# bars, p8 = only by bar 8, intros kept intact), and eval_e1.sbatch
# takes one PROMPT_LENGTH per run, so each group is its own run with
# its own results/E1_<run>_metrics.csv. This job concatenates those
# CSVs (checking the columns agree) and runs the paired table over the
# union of songs. Prompt length is a per-song property shared by every
# system, so it cancels in the per-song paired differences; the scored
# continuation has the same length in every group by construction
# (GEN_LENGTH = PROMPT_LENGTH + 320).
#
# Env knobs (--export=ALL,KEY=val):
#   RUNS        run suffixes, space-separated (default "p96_matched
#               p128_matched": results/E1_p96_matched_metrics.csv and
#               results/E1_p128_matched_metrics.csv)
#   RESULT_TAG  default matched: writes results/E1_mixed_<TAG>_*
#   BASELINE    default A3
#
# Example:
#   sbatch --export=ALL,RUNS="p96_matched p128_matched",RESULT_TAG=matched midi_yinyang/merge_e1.sbatch

#SBATCH --job-name=merge_e1
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
mkdir -p /home/xinyue.li/logs results

RUNS="${RUNS:-p96_matched p128_matched}"
RESULT_TAG="${RESULT_TAG:-matched}"
BASELINE="${BASELINE:-A3}"

MERGED="results/E1_mixed_${RESULT_TAG}_metrics.csv"
TABLE="results/E1_mixed_${RESULT_TAG}_table.md"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  merge E1 runs: $RUNS  ->  $MERGED"
echo "================================================================"
INPUTS=()
for r in $RUNS; do
    f="results/E1_${r}_metrics.csv"
    [[ -f "$f" ]] || { echo "ERROR: $f missing (run E1 for group $r first)"; exit 1; }
    INPUTS+=("$f")
done

python merge_e1_metrics.py --out "$MERGED" "${INPUTS[@]}"

echo "--- comparison table (paired Wilcoxon vs $BASELINE, union of songs) ---"
python aggregate_eval_results.py --csv "$MERGED" \
    --task melchord --baseline "$BASELINE" --mode co \
    --one-sided --complete-cases --out-md "$TABLE"
python aggregate_eval_results.py --csv "$MERGED" \
    --task melchord --baseline "$BASELINE" --mode co \
    --out-md "${TABLE%.md}_twosided.md"

echo "================================================================"
echo "  table   : $TABLE   (one-sided, complete cases, vs $BASELINE)"
echo "  2-sided : ${TABLE%.md}_twosided.md"
echo "  metrics : $MERGED"
echo "================================================================"
echo "RESULT TABLE:"
cat "$TABLE"
