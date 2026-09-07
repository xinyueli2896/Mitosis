#!/bin/bash
# SLURM (CPU): match our held-out POP909 songs with whole-song-gen's
# validation split, and stage the shared-clean set for E1.
#
# whole-song-gen's held-out set is its 9:1 validation split
# (data/pop909_split/split.npz, 91 songs). Ours is FramedDataset's
# split='val' (dataset index % 10 == 0, ~91 songs) plus the 5 songs cut
# out into input/pop909_split. A fair test song for a table with WS in
# it is one NEITHER side trained on: their valid ∩ our held-out.
#
# Runs audit_wholesong_split.py: prints both splits, where our current
# 5 test songs fall in theirs, the shared-clean intersection, and
# copies those songs' melody/chord midis to STAGE_DIR/{melody,chord}
# so E1 can run with MEL_FOLDER/CHORD_FOLDER pointed there.
#
# Env knobs (--export=ALL,KEY=val):
#   DATASETS    our POP909 dataset stems, space-separated (default:
#               pop909_chord_cp4_v2 pop909_chord_cp8_v2
#               pop909_melchord_tagged_cp16_v2 -- cp4 for A1/A3/the
#               specialists, cp8 for YinYang, tagged cp16 for S1/
#               S-scratch). Each has its own index%10 validation set;
#               all are printed and intersected.
#   STAGE_DIR   default input/pop909_split_ws
#   MEL_SRC / CHORD_SRC   full per-song POP909 stream midis
#               (default POP909-Dataset/POP909-{melody,chord})
#
# Example:
#   sbatch midi_yinyang/audit_wholesong_split.sbatch

#SBATCH --job-name=audit_wholesong_split
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:15:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"
WS="$MY/external/whole_song_gen"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs

DATASETS="${DATASETS:-pop909_chord_cp4_v2 pop909_chord_cp8_v2 pop909_melchord_tagged_cp16_v2}"
STAGE_DIR="${STAGE_DIR:-input/pop909_split_ws}"
MEL_SRC="${MEL_SRC:-POP909-Dataset/POP909-melody}"
CHORD_SRC="${CHORD_SRC:-POP909-Dataset/POP909-chord}"
SPLIT="$WS/data/pop909_split/split.npz"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "whole-song split audit   theirs=$SPLIT"
echo "ours: data/{$DATASETS}.{txt,length.pt}  test=input/pop909_split/melody"
echo "stage -> $STAGE_DIR  from $MEL_SRC , $CHORD_SRC"
echo "================================================================"
[[ -f "$SPLIT" ]] || { echo "ERROR: $SPLIT not found (run setup_wholesong / infer_wholesong_uncond once to unpack the data payload)"; exit 1; }

python audit_wholesong_split.py \
    --split "$SPLIT" \
    --datasets $DATASETS \
    --mel-src "$MEL_SRC" --chord-src "$CHORD_SRC" \
    --stage-dir "$STAGE_DIR"

echo "================================================================"
echo "E1 on the shared-clean set:"
echo "  sbatch --export=ALL,MEL_FOLDER=$STAGE_DIR/melody,CHORD_FOLDER=$STAGE_DIR/chord,OUT_ROOT=temp/E1_ws,RESULT_TAG=ws \\"
echo "      midi_yinyang/eval_e1.sbatch"
echo "================================================================"
