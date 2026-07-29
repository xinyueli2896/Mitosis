#!/bin/bash
# SLURM batch script (CPU): build the combined POP909+Nottingham
# SINGLE-STREAM (merged melody+chord) dataset used to finetune S1 with
# a matched training corpus to the combined-data A.2/A.1 duet models.
#
# Steps:
#   1. tag_nottingham_melchord.py: force the Nottingham chord track's
#      program to 48 (matching the POP909 tagged-combined convention)
#      so the single-stream tokenizer keeps melody/chord distinguishable.
#      Songs without a chord track are skipped (same population excluded
#      from the duet Nottingham pair).
#   2. tokenize the tagged folder as ONE merged stream (reuses the
#      generic pop909_melchord entrypoint via env override) ->
#      data/nottingham_melchord_cp16_v2.pt
#   3. combine_single_stream_datasets.py: POP909 (already tokenized,
#      FIRST -- preserves its held-out ids) + the new Nottingham single
#      stream -> data/melchord_pop909_nottingham_cp16_v2.pt
#
# Prerequisite: data/pop909_melchord_cp16_v2.pt must already exist
#   (preprocess_pop909_melchord.sbatch).
#
# Env knobs (override via --export=ALL,KEY=val):
#   NOTTINGHAM_DIR   raw Nottingham midi folder.
#                    default: /home/xinyue.li/nottingham-dataset/MIDI
#   CHORD_PROGRAM    program the Nottingham chord track is tagged to.
#                    default 48 (matches POP909's tagged-combined data
#                    and cp_transformer_inference.py's auto-tagging).
#   POP909_SINGLESTREAM  prefix of the existing POP909 single-stream
#                    dataset. default: data/pop909_melchord_cp16_v2
#
# After this finishes:
#   DATA=data/melchord_pop909_nottingham_cp16_v2.pt RUN_TAG=pop909nott \
#       sbatch --export=ALL midi_yinyang/finetune_pop909.sbatch

#SBATCH --job-name=mitosis_preproc_ss_combined
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

NOTTINGHAM_DIR="${NOTTINGHAM_DIR:-/home/xinyue.li/nottingham-dataset/MIDI}"
CHORD_PROGRAM="${CHORD_PROGRAM:-48}"
POP909_SINGLESTREAM="${POP909_SINGLESTREAM:-data/pop909_melchord_cp16_v2}"
TAGGED_DIR="nottingham-melchord-tagged"
OUT="data/melchord_pop909_nottingham_cp16_v2"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs data

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "NOTTINGHAM_DIR = $NOTTINGHAM_DIR"
echo "CHORD_PROGRAM  = $CHORD_PROGRAM"
echo "POP909_SINGLESTREAM = $POP909_SINGLESTREAM"
echo "================================================================"

if [[ ! -f "$POP909_SINGLESTREAM.pt" ]]; then
    echo "ERROR: $POP909_SINGLESTREAM.pt missing -- run"
    echo "  preprocess_pop909_melchord.sbatch  first."
    exit 1
fi
if [[ ! -d "$NOTTINGHAM_DIR" ]]; then
    echo "ERROR: NOTTINGHAM_DIR=$NOTTINGHAM_DIR not found."
    exit 1
fi

# --- 1. tag ------------------------------------------------------------
python tag_nottingham_melchord.py \
    --src "$NOTTINGHAM_DIR" --dst "$TAGGED_DIR" \
    --chord-program "$CHORD_PROGRAM"

# --- 2. tokenize as one merged stream -----------------------------------
POP909_COMBINED_PATH="$TAGGED_DIR" \
POP909_MELCHORD_DATASET="nottingham_melchord_cp16_v2" \
    python preprocess_large_midi_dataset.py pop909_melchord

# --- 3. combine (POP909 first) ------------------------------------------
python combine_single_stream_datasets.py \
    --source "pop909:$POP909_SINGLESTREAM" \
    --source "nottingham:data/nottingham_melchord_cp16_v2" \
    --out "$OUT"

echo "----------------------------------------------------------------"
ls -lh "$OUT"* || true
echo "================================================================"
echo "Combined single-stream melchord dataset ready: $OUT.pt"
echo "Finetune S1 on it:"
echo "  DATA=$OUT.pt RUN_TAG=pop909nott \\"
echo "      sbatch --export=ALL midi_yinyang/finetune_pop909.sbatch"
