#!/bin/bash
# SLURM batch script (CPU): combine the POP909 and Pop1K7 melody/chord
# .pt pairs into one training set (POP909 first, so the EXPERIMENTS.md
# held-out POP909 ids stay held out after combination -- see
# combine_melchord_datasets.py's docstring on index-mod-10 stability).
#
# Prerequisites: both source pairs tokenized and pairing-checked
#   preprocess_pop909_melchord.sbatch  (or the cp4 POP909 pair already
#     in data/ used by the melchord cp4 family)
#   preprocess_pop1k7_melchord.sbatch  AT THE SAME CP
#
# CP MUST MATCH between the two sources and the model you intend to
# train: the melchord cp4 family (MELCHORD_CP=4, the A.2 mg/shared
# cp4tar runs) needs BOTH sources at cp4, so run the Pop1K7 preprocess
# with MAX_POLYPHONY=4 before combining at CP=4. Mixing budgets is not
# possible -- the tokenizations are different vocabularies of frame
# content, and the file names encode the budget precisely so the guard
# below catches a mismatch as a missing file.
#
# Output (under midi_yinyang/data/):
#   melchord_pop909_pop1k7_melody_cp${CP}_v2.pt (+ sidecars)
#   melchord_pop909_pop1k7_chord_cp${CP}_v2.pt  (+ sidecars)
# followed by the song-pairing gate.
#
# Env knobs (override via --export=ALL,KEY=val):
#   CP                          polyphony budget of the sources and the
#                               output names. Default 4 (the melchord
#                               cp4 family the A.2 MoE runs use).
#   POP909_MEL / POP909_CHORD   default data/pop909_{melody,chord}_cp${CP}_v2
#   POP1K7_MEL / POP1K7_CHORD   default data/pop1k7_{melody,chord}_cp${CP}_v2
#
# Then train with:
#   sbatch --export=ALL,TASK=melchord_pop909_pop1k7,MAX_POLYPHONY=${CP} \
#       midi_yinyang/train_duet_block_diffusion.sbatch

#SBATCH --job-name=mitosis_combine_pop1k7
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

CP="${CP:-4}"
POP909_MEL="${POP909_MEL:-data/pop909_melody_cp${CP}_v2}"
POP909_CHORD="${POP909_CHORD:-data/pop909_chord_cp${CP}_v2}"
POP1K7_MEL="${POP1K7_MEL:-data/pop1k7_melody_cp${CP}_v2}"
POP1K7_CHORD="${POP1K7_CHORD:-data/pop1k7_chord_cp${CP}_v2}"
OUT_MEL="data/melchord_pop909_pop1k7_melody_cp${CP}_v2"
OUT_CHORD="data/melchord_pop909_pop1k7_chord_cp${CP}_v2"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "COMBINE melchord sources at cp${CP}"
echo "POP909: $POP909_MEL / $POP909_CHORD"
echo "Pop1K7: $POP1K7_MEL / $POP1K7_CHORD"
echo "================================================================"

for f in "$POP909_MEL" "$POP909_CHORD" "$POP1K7_MEL" "$POP1K7_CHORD"; do
    if [[ ! -f "$f.pt" ]]; then
        echo "ERROR: $f.pt missing."
        echo "  Both sources must be tokenized AT cp${CP}. For Pop1K7:"
        echo "    sbatch --exclude=cn-20 --export=ALL,MAX_POLYPHONY=${CP} \\"
        echo "        midi_yinyang/preprocess_pop1k7_melchord.sbatch"
        exit 1
    fi
done

# POP909 listed FIRST -- preserves its songs' standalone absolute
# indices (and hence the held-out ids) under the index-mod-10 split.
python combine_melchord_datasets.py \
    --source "pop909:$POP909_MEL:$POP909_CHORD" \
    --source "pop1k7:$POP1K7_MEL:$POP1K7_CHORD" \
    --out-mel "$OUT_MEL" \
    --out-chord "$OUT_CHORD"

python check_paired_dataset.py "$OUT_MEL" "$OUT_CHORD"

echo "----------------------------------------------------------------"
ls -lh "$OUT_MEL"* "$OUT_CHORD"* || true
echo "================================================================"
echo "Combined POP909+Pop1K7 melchord dataset ready (cp${CP})."
echo "Train: sbatch --export=ALL,TASK=melchord_pop909_pop1k7,MAX_POLYPHONY=${CP} \\"
echo "    midi_yinyang/train_duet_block_diffusion.sbatch"
echo "================================================================"
