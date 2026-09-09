#!/bin/bash
# SLURM (CPU): the second results table -- coherence, repetition and
# general symbolic statistics, reference-free, on the E1 outputs already
# generated. Scores each prompt group from its E1 manifest against its
# staged references, merges the groups, and writes the IEEE-style LaTeX
# table with the reference pair as a row. No generation, no GPU.
#
# Env knobs (--export=ALL,KEY=val):
#   COH_GROUPS "run:out_root:prompt:total" entries, space-separated
#              (default the two matched groups)
#   RESULT_TAG default matched -> results/E1_mixed_<TAG>_coherence.csv
#   BASELINE   default A3ctcaT (the paper's decode: alternating commit, T=1)
#
# Example:
#   sbatch midi_yinyang/score_coherence.sbatch

#SBATCH --job-name=score_coherence
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs results

COH_GROUPS="${COH_GROUPS:-p96_matched:temp/E1_matched_p6:96:416 p128_matched:temp/E1_matched_p8:128:448}"
RESULT_TAG="${RESULT_TAG:-matched}"
BASELINE="${BASELINE:-A3ctcaT}"

CSVS=()
for g in $COH_GROUPS; do
    IFS=: read -r run root prompt total <<< "$g"
    manifest="results/E1_${run}_manifest.tsv"
    [[ -f "$manifest" ]] || { echo "ERROR: $manifest missing (run the E1 score step for $run first)"; exit 1; }
    out="results/E1_${run}_coherence.csv"
    echo "================================================================"
    echo "[coherence] $run  manifest=$manifest  refs=$root/prompts  window=[$prompt,$total)"
    python coherence_metrics.py --manifest "$manifest" \
        --ref-a-dir "$root/prompts/mel" --ref-b-dir "$root/prompts/chord" \
        --prompt-frames "$prompt" --total-frames "$total" --out "$out"
    CSVS+=("$out")
done

MERGED="results/E1_mixed_${RESULT_TAG}_coherence.csv"
python merge_e1_metrics.py --out "$MERGED" "${CSVS[@]}"
python coherence_metrics.py --table "$MERGED" --baseline "$BASELINE" \
    --out "results/E1_paper_${RESULT_TAG}_coherence"
echo "----- results/E1_paper_${RESULT_TAG}_coherence.tex -----"
cat "results/E1_paper_${RESULT_TAG}_coherence.tex"
