#!/bin/bash
# SLURM (CPU): stage E1's prompt folders once, so several GPU jobs can
# generate disjoint SYSTEMS lists in PARALLEL on the same OUT_ROOT.
#
# Runs eval_e1.sbatch's body with PREP_ONLY=1: copies the test split to
# OUT_ROOT/prompts/{mel,chord}, builds merged_tagged, writes
# OUT_ROOT/prompts/.ready and exits. GPU jobs that find .ready reuse the
# prompts instead of rebuilding them, so they never race. Every eval_e1
# knob passes through (MEL_FOLDER, CHORD_FOLDER, OUT_ROOT, ...).
#
# Recipe (two GPUs, one tree):
#   J0=$(sbatch --parsable --export=ALL,MEL_FOLDER=...,CHORD_FOLDER=...,OUT_ROOT=temp/E1_matched midi_yinyang/prep_e1.sbatch)
#   JA=$(sbatch --parsable --dependency=afterok:$J0 --export=ALL,...,SYSTEMS="A3 S1 S-scratch"  midi_yinyang/eval_e1.sbatch)
#   JB=$(sbatch --parsable --dependency=afterok:$J0 --export=ALL,...,SYSTEMS="A1 P-mc P-cm WS" midi_yinyang/eval_e1.sbatch)
#   sbatch --dependency=afterok:$JA:$JB --export=ALL,...,RESULT_TAG=matched midi_yinyang/score_e1.sbatch
# (each GPU job also scores its own systems at the end; the final
#  score_e1 is the one that sees all of them.)

#SBATCH --job-name=prep_e1
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
export PREP_ONLY=1
exec bash /home/xinyue.li/Mitosis/midi_yinyang/eval_e1.sbatch
