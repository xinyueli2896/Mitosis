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
#               the cp4 melody/chord pair (A1, A3, specialists), the
#               cp8 pair, the cp8 '_chord_mel' dual file (YinYang) and
#               the tagged cp16 merge (S1/S-scratch)). Each file's
#               idx%10==0 songs are printed and intersected; a final
#               per-system table checks the shared-clean songs against
#               the TRAINING split of every E1 system (two dataset
#               classes, two rules -- idx%10==0 is untrained under both).
#   STAGE_DIR   default input/909_matched_split
#   MEL_SRC / CHORD_SRC   the TIME-ALIGNED per-song POP909 stream midis,
#               /home/xinyue.li/POP909-Dataset/POP909-{melody,chord}.
#               NOT the relative POP909-Dataset/... under midi_yinyang:
#               those are the raw files with POP909's original tempo
#               maps and off-grid timing (job 217259: 26 tempo events
#               and 80% off-grid onsets on 456), which staged a test
#               set on a different grid from the models' training data.
#               004 is taken from here too, so all songs are one kind
#               of file.
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

DATASETS="${DATASETS:-pop909_chord_cp4_v2 pop909_melody_cp4_v2 pop909_chord_cp8_v2 pop909_melody_cp8_v2 pop909_cp8_v2_chord_mel pop909_melchord_tagged_cp16_v2}"
STAGE_DIR="${STAGE_DIR:-input/909_matched_split}"
MEL_SRC="${MEL_SRC:-/home/xinyue.li/POP909-Dataset/POP909-melody}"
CHORD_SRC="${CHORD_SRC:-/home/xinyue.li/POP909-Dataset/POP909-chord}"
[[ -d "$MEL_SRC" && -d "$CHORD_SRC" ]] || { echo "ERROR: source folders missing: $MEL_SRC , $CHORD_SRC"; exit 1; }
# a fresh stage every run: the buggy first-tempo prompt check deleted
# files from the old stage dir, and a stale file must not survive a
# change of source folder
rm -rf "$STAGE_DIR"
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
echo "  sbatch --export=ALL,MEL_FOLDER=$STAGE_DIR/melody,CHORD_FOLDER=$STAGE_DIR/chord,OUT_ROOT=temp/E1_matched,RESULT_TAG=matched \\"
echo "      midi_yinyang/eval_e1.sbatch"
echo "================================================================"
