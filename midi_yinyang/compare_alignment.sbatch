#!/bin/bash
# SLURM (CPU): A/B two MIDI folders on beat alignment, in TICK space.
#
# Built to answer "is the v4 aligner's output actually different from the
# POP909 melody/chord folders the models were trained on, and if so how".
# Three things per song, none of which a single-folder grid check can
# tell you:
#   * on-grid fraction for EACH folder (ticks/ppq -> beats, the only
#     domain the CP tokenizer reads)
#   * the LAG between them, by cross-correlating onset patterns over
#     +-4 beats. A folder that is systematically a beat early shows as a
#     consistent -4 sixteenths across songs.
#   * tempo-event count each: 1 means metrically exact, one-per-beat
#     means the file still carries a performance curve.
# With --a-chord/--b-chord it does the same for the chord folders.
#
# Env knobs (--export=ALL,KEY=val):
#   A_MEL, B_MEL      melody folders (required)
#   A_CHORD, B_CHORD  chord folders (optional, both or neither)
#   IDS               optional song ids (default: all shared)
#   OUT               per-song TSV (default results/compare_alignment.tsv)
#
# Default compares the ORIGINAL training folders (A) against v4 (B), so
# a negative lag means v4 is EARLY relative to what the models saw.
#
#   sbatch midi_yinyang/compare_alignment.sbatch
#   sbatch --export=ALL,IDS="004 010 136 746" \
#       midi_yinyang/compare_alignment.sbatch

#SBATCH --job-name=compare_alignment
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs results

POP="/home/xinyue.li/POP909-Dataset"
A_MEL="${A_MEL:-$POP/POP909-melody}"
A_CHORD="${A_CHORD:-$POP/POP909-chord}"
B_MEL="${B_MEL:-$POP/POP909-v4/melody}"
B_CHORD="${B_CHORD:-$POP/POP909-v4/chord}"
IDS="${IDS:-}"
OUT="${OUT:-results/compare_alignment.tsv}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "A (reference, what the models trained on):"
echo "   mel   $A_MEL"
echo "   chord $A_CHORD"
echo "B (candidate):"
echo "   mel   $B_MEL"
echo "   chord $B_CHORD"
echo "IDS=${IDS:-<all shared>}   OUT=$OUT"
echo "================================================================"

for d in "$A_MEL" "$B_MEL"; do
    [[ -d "$d" ]] || { echo "ERROR: missing folder $d"; exit 1; }
done

CH=()
if [[ -d "$A_CHORD" && -d "$B_CHORD" ]]; then
    CH=(--a-chord "$A_CHORD" --b-chord "$B_CHORD")
else
    echo "[warn] chord folders not both present; comparing melody only"
fi

python compare_alignment.py \
    --a "$A_MEL" --b "$B_MEL" "${CH[@]}" \
    ${IDS:+--ids $IDS} --out "$OUT"

echo "================================================================"
echo "Read the lag histogram first. All zeros means the two folders"
echo "agree on where the beats are and only the GRID differs; a spike at"
echo "a non-zero lag means the training data and v4 disagree about bar"
echo "phase, and every prompt built from one is wrong against the other."
echo "================================================================"
