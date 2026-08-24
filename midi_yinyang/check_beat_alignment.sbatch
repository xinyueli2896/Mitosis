#!/bin/bash
# SLURM (CPU): can these MIDIs be represented on the CP 16th-note grid?
#
# Motivated by jigs108.mid sounding "not beat aligned": the tokenizer
# quantizes onsets to step = 60/tempo/4, and Nottingham's compound-meter
# tunes (jigs 6/8, slip jigs 9/8) are converted with TRIPLET
# subdivisions -- every offbeat eighth sits a third of a step off the
# grid and gets SNAPPED a 32nd away at tokenization. The source file
# sounds fine; the tokenized song (training data, prompts, references,
# and anything decoded) lurches. Verified on the in-repo samples:
# ashover2.mid shows the exact 0.33-step deviation cluster.
#
# Flags per file: TRIPLET (deviations cluster at 1/3 step), OFF-GRID
# (other misalignment), METER (bar is not 16 steps -- 3/4, 6/8, 2/4,
# mixed signatures -- which mis-slices every 16-frames-per-bar
# assumption downstream even when the notes align).
#
# Env knobs (--export=ALL,KEY=val):
#   FOLDERS   space-separated folders to sweep.
#             default "input/not_split/melody input/not_split/chord"
#   FILE      basename to print even if clean (e.g. jigs108.mid)
#
# Example:
#   sbatch --export=ALL,FILE=jigs108.mid \
#       midi_yinyang/check_beat_alignment.sbatch

#SBATCH --job-name=beat_alignment
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

FOLDERS="${FOLDERS:-input/not_split/melody input/not_split/chord}"
FILE="${FILE:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

ARGS=()
for f in $FOLDERS; do
    [[ -d "$f" ]] || { echo "ERROR: folder not found: $f"; exit 1; }
    ARGS+=(--folder "$f")
done
[[ -n "$FILE" ]] && ARGS+=(--file "$FILE")

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "BEAT-ALIGNMENT CHECK   folders: $FOLDERS"
echo "================================================================"

python check_beat_alignment.py "${ARGS[@]}"

echo "================================================================"
echo "TRIPLET files are corrupted BY TOKENIZATION everywhere they are"
echo "used (training, prompts, references). Short-term: exclude them."
echo "Long-term: a grid divisible by 3 (beat_div 6/12) -- a full"
echo "retokenization decision, not a per-run knob."
echo "================================================================"
