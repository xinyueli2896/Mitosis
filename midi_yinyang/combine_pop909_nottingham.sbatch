#!/bin/bash
# SLURM batch script (CPU): combine the POP909 and Nottingham melody/chord
# .pt pairs into one training set for A.2 (POP909 first, so the
# EXPERIMENTS.md held-out POP909 ids stay held out after combination --
# see combine_melchord_datasets.py's docstring).
#
# Prerequisites: both source pairs already tokenized and pairing-checked
#   (preprocess_pop909_melchord.sbatch, preprocess_nottingham_melchord.sbatch).
#
# Output (under midi_yinyang/data/):
#   melchord_pop909_nottingham_melody_cp4_v2.pt (+ sidecars)
#   melchord_pop909_nottingham_chord_cp4_v2.pt  (+ sidecars)
# followed by the song-pairing gate.
#
# Env knobs (override via --export=ALL,KEY=val):
#   POP909_MEL / POP909_CHORD          default data/pop909_{melody,chord}_cp4_v2
#   NOTTINGHAM_MEL / NOTTINGHAM_CHORD  default data/nottingham_{melody,chord}_cp4_v2
#
# Then train with:
#   sbatch --export=ALL,TASK=melchord_pop909_nottingham \
#       midi_yinyang/train_duet_block_diffusion.sbatch

#SBATCH --job-name=mitosis_combine_melchord
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

POP909_MEL="${POP909_MEL:-data/pop909_melody_cp4_v2}"
POP909_CHORD="${POP909_CHORD:-data/pop909_chord_cp4_v2}"
NOTTINGHAM_MEL="${NOTTINGHAM_MEL:-data/nottingham_melody_cp4_v2}"
NOTTINGHAM_CHORD="${NOTTINGHAM_CHORD:-data/nottingham_chord_cp4_v2}"
OUT_MEL="data/melchord_pop909_nottingham_melody_cp4_v2"
OUT_CHORD="data/melchord_pop909_nottingham_chord_cp4_v2"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "POP909:      $POP909_MEL / $POP909_CHORD"
echo "Nottingham:  $NOTTINGHAM_MEL / $NOTTINGHAM_CHORD"
echo "================================================================"

for f in "$POP909_MEL" "$POP909_CHORD" "$NOTTINGHAM_MEL" "$NOTTINGHAM_CHORD"; do
    if [[ ! -f "$f.pt" ]]; then
        echo "ERROR: $f.pt missing -- run the corresponding preprocess sbatch first."
        exit 1
    fi
done

# POP909 listed FIRST -- preserves its songs' standalone absolute
# indices (and hence the EXPERIMENTS.md held-out ids) under the
# combined dataset's index-mod-10 split.
python combine_melchord_datasets.py \
    --source "pop909:$POP909_MEL:$POP909_CHORD" \
    --source "nottingham:$NOTTINGHAM_MEL:$NOTTINGHAM_CHORD" \
    --out-mel "$OUT_MEL" \
    --out-chord "$OUT_CHORD"

python check_paired_dataset.py "$OUT_MEL" "$OUT_CHORD"

echo "----------------------------------------------------------------"
ls -lh "$OUT_MEL"* "$OUT_CHORD"* || true
echo "================================================================"
echo "Combined melchord dataset ready."
echo "Train: sbatch --export=ALL,TASK=melchord_pop909_nottingham \\"
echo "    midi_yinyang/train_duet_block_diffusion.sbatch"
