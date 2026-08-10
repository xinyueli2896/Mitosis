#!/bin/bash
# SLURM: quantify long-term structure of existing generation outputs.
# (CPU, no GPU -- scores midis already on disk, no inference.)
#
# Wraps structure_metrics.py: per system, bar-level repetition,
# phrase/section periodicity (chroma + rhythm at lags 4/8 bars), and
# tonal drift vs the prompt. Read every model row against the REF row.
#
# Env knobs (--export=ALL,KEY=val):
#   SYSTEMS      REQUIRED. Space-separated NAME:FOLDER specs; folders
#                are scanned recursively, so E1 duet_multi layouts and
#                single-stream output dirs both work. Include the
#                ground-truth as one of them, e.g. REF:<folder>.
#   PROMPT_BARS  default 4 (= PROMPT_LENGTH 64). Use 6 for 96-frame runs.
#   MAX_BARS     default 24 -- truncates REF's full songs to the same
#                horizon the models generated (PROMPT+GEN = 384 frames).
#   SPLIT_STREAMS default 1: also report melody-only / chord-only rows
#                (melchord program conventions). Set 0 for drumnondrum.
#
# Example (melchord E1 run + ground truth):
#   sbatch --export=ALL,SYSTEMS="A2:temp/e1_melchord_<id>/a2 \
#       B1:temp/e1_melchord_<id>/b1 S1:temp/e1_melchord_<id>/s1 \
#       REF:input/pop909_split_merged" \
#       midi_yinyang/structure_metrics.sbatch

#SBATCH --job-name=structure_metrics
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
PROMPT_BARS="${PROMPT_BARS:-4}"
MAX_BARS="${MAX_BARS:-24}"
SPLIT_STREAMS="${SPLIT_STREAMS:-1}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

if [[ -z "${SYSTEMS:-}" ]]; then
    echo "ERROR: SYSTEMS is required, e.g."
    echo "  sbatch --export=ALL,SYSTEMS=\"A2:temp/e1_.../a2 S1:temp/e1_.../s1 \\"
    echo "      REF:input/pop909_split_merged\" midi_yinyang/structure_metrics.sbatch"
    exit 1
fi

SYS_ARGS=()
for spec in $SYSTEMS; do
    SYS_ARGS+=(--system "$spec")
done
SPLIT_FLAG=()
[[ "$SPLIT_STREAMS" == "1" ]] && SPLIT_FLAG=(--split-streams)

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "LONG-TERM STRUCTURE METRICS   (CPU; no GPU needed)"
echo "SYSTEMS: $SYSTEMS"
echo "PROMPT_BARS=$PROMPT_BARS  MAX_BARS=$MAX_BARS  SPLIT_STREAMS=$SPLIT_STREAMS"
echo "================================================================"

python structure_metrics.py "${SYS_ARGS[@]}" \
    --prompt-bars "$PROMPT_BARS" --max-bars "$MAX_BARS" \
    "${SPLIT_FLAG[@]}"
