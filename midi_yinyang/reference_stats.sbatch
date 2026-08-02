#!/bin/bash
# SLURM: score the H1 metrics on the GROUND TRUTH itself. (CPU, no GPU)
#
# Wraps reference_stats.py, which runs eval_metrics' own load_streams +
# h1_metrics on the reference pair against ITSELF over the same window
# the evaluation uses. This establishes the ceiling every model row has
# been implicitly compared against but which was never measured:
#
#   survival_b        fraction of bars the REAL chord is active. B.1's
#                     ~0.66 has been read as a deficit against S1's 0.930
#                     and A.2's 1.000 -- but those are other MODELS. If
#                     the real chord itself scores ~0.7, B.1 is
#                     reproducing the data and S1/A.2 over-produce; if it
#                     scores ~0.95, B.1's deficit is real and, since more
#                     training fixed the leader but not the follower,
#                     architectural.
#   density_drift_b   whether the real music thins toward the end -- the
#                     baseline for reading any model's "fading out".
#   density_ratio_*   1.0 BY CONSTRUCTION (ref vs itself). Anything else
#                     means the melody/chord pairing is broken and every
#                     reference-relative metric in the results is suspect.
#
# Pure file reading (pretty_midi + numpy); runs on the CPU partition.
#
# Env knobs (--export=ALL,KEY=val):
#   REF_A          melody/drum folder   (default input/pop909_split/melody)
#   REF_B          chord/nondrum folder (default input/pop909_split/chord)
#   TASK           melchord (default) | drumnondrum
#   PROMPT_LENGTH  default 64  (frames; 16/bar -- window START, matching
#                  the eval: prompt bars are excluded from scoring)
#   GEN_LENGTH     default 384 (window END, total frames incl. prompt)
#
# Examples:
#   sbatch midi_yinyang/reference_stats.sbatch
#   sbatch --export=ALL,PROMPT_LENGTH=96,GEN_LENGTH=416 \
#       midi_yinyang/reference_stats.sbatch

#SBATCH --job-name=reference_stats
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

TASK="${TASK:-melchord}"
REF_A="${REF_A:-input/pop909_split/melody}"
REF_B="${REF_B:-input/pop909_split/chord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

for d in "$REF_A" "$REF_B"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: reference folder not found: $d"
        exit 1
    fi
done

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "GROUND-TRUTH H1 BASELINE   task=$TASK   (CPU; no GPU needed)"
echo "REF_A=$REF_A  REF_B=$REF_B"
echo "window: frames [$PROMPT_LENGTH, $GEN_LENGTH)"\
" = bars $((PROMPT_LENGTH / 16))-$((GEN_LENGTH / 16))"
echo "================================================================"

python reference_stats.py \
    --ref-a-dir "$REF_A" --ref-b-dir "$REF_B" \
    --task "$TASK" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH"

echo "================================================================"
echo "How to read it against the model tables:"
echo "  * a model NEAR these values reproduces the data's own behaviour;"
echo "    far ABOVE survival/density is over-producing, not 'better'."
echo "  * density_ratio_a / _b must be exactly 1.0 -- anything else means"
echo "    the melody/chord folders are mispaired and all reference-"
echo "    relative results are suspect."
echo "================================================================"
