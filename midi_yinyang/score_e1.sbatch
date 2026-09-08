#!/bin/bash
# SLURM (CPU): score an existing E1 run -- no generation, no GPU.
#
# Scoring (manifest -> eval_metrics -> paired table) is pretty_midi and
# numpy; the only reason a score-only run ever took a GPU node is that
# eval_e1.sbatch's directives request one and directives cannot be
# conditional. This wrapper runs eval_e1.sbatch's own body on a CPU
# node with SKIP_INFER=1 forced, so every knob eval_e1 understands
# (OUT_ROOT, MEL_FOLDER/CHORD_FOLDER, RESULT_TAG, BASELINE, SYSTEMS,
# PROMPT_LENGTH ...) passes straight through. Use it to rescore after a
# metrics change, a baseline change, or a change of song list.
#
# Example:
#   sbatch --export=ALL,MEL_FOLDER=input/pop909_split_ws/melody,\
# CHORD_FOLDER=input/pop909_split_ws/chord,OUT_ROOT=temp/E1_ws,RESULT_TAG=ws \
#       midi_yinyang/score_e1.sbatch

#SBATCH --job-name=score_e1
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
export SKIP_INFER=1
export PIANO_COPIES="${PIANO_COPIES:-0}"
exec bash /home/xinyue.li/Mitosis/midi_yinyang/eval_e1.sbatch
