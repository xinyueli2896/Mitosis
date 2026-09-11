#!/bin/bash
# SLURM (CPU): gate for the prompt set: every prompt must start on a true downbeat before
# training. See audit_prompt_phase.py. Run it after build_prompt_crops and
#
#   sbatch midi_yinyang/audit_prompt_phase.sbatch

#SBATCH --job-name=audit_prompt_phase
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:20:00

set -euo pipefail
source ~/.bashrc
conda activate mitosis
cd /home/xinyue.li/Mitosis/midi_yinyang
mkdir -p /home/xinyue.li/logs
POP=/home/xinyue.li/POP909-Dataset
SET="${SET:-input/heldout_v5}"
REPORT="${REPORT:-$POP/POP909-v5/align_grid_report.tsv}"
# 6 bars: the lead (1 bar + pickup, or 2 bars) plus MEL_BARS=4 sounding
# bars. Must match what build_prompt_crops wrote, or the "generation
# starts on a downbeat" test is asking about the wrong frame.
PROMPT_FRAMES="${PROMPT_FRAMES:-96}"
GEN_FRAMES="${GEN_FRAMES:-320}"
echo "SET=$SET  REPORT=$REPORT  PROMPT_FRAMES=$PROMPT_FRAMES"

python audit_prompt_phase.py --set "$SET" --report "$REPORT" \
    --prompt-frames "$PROMPT_FRAMES" --gen-frames "$GEN_FRAMES"
