#!/bin/bash
# SLURM batch script (CPU): split the HELD-OUT Pop1K7 test songs into
# per-stream melody/chord prompt folders for inference.
#
# ~/pop1k7/test was deliberately left out of the Pop1K7 preprocessing
# (src_001..src_004 only), so these songs are unseen by BOTH the
# POP909-only models and the pop909+pop1k7 combined models -- valid
# prompts for either. Each midi carries the release's three NAMED
# tracks (melody / chord / piano); this reuses split_named_tracks.py
# (name-based selection, meta/tempo grid preserved, piano dropped) to
# produce the POP909-format per-stream folders the inference wrappers'
# MEL_FOLDER/CHORD_FOLDER knobs expect.
#
# Output:
#   input/pop1k7_test_split/melody/<song>.mid
#   input/pop1k7_test_split/chord/<song>.mid
#
# Env knobs (--export=ALL,KEY=val):
#   SRC   default /home/xinyue.li/pop1k7/test
#   DST   default input/pop1k7_test_split
#
# Example:
#   sbatch midi_yinyang/split_pop1k7_test.sbatch

#SBATCH --job-name=mitosis_split_pop1k7_test
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

SRC="${SRC:-/home/xinyue.li/pop1k7/test}"
DST="${DST:-input/pop1k7_test_split}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "SPLIT Pop1K7 TEST SET (held-out) into per-stream prompt folders"
echo "SRC = $SRC   DST = $DST/{melody,chord}"
echo "================================================================"

[[ -d "$SRC" ]] || { echo "ERROR: SRC not found: $SRC"; exit 1; }

python split_named_tracks.py \
    --src "$SRC" \
    --dst-melody "$DST/melody" \
    --dst-chord  "$DST/chord"

N_MEL=$(ls "$DST/melody" | wc -l)
N_CHD=$(ls "$DST/chord"  | wc -l)
echo "----------------------------------------------------------------"
echo "melody: $N_MEL files   chord: $N_CHD files"
if [[ "$N_MEL" -ne "$N_CHD" || "$N_MEL" -eq 0 ]]; then
    echo "ERROR: stream counts differ or empty -- check the skip list"
    echo "       in the log above (songs missing a named track)."
    exit 1
fi
echo "================================================================"
echo "Prompt folders ready. Use with the duet inference wrappers, e.g.:"
echo "  sbatch --exclude=gpu-50,gpu-51 --export=ALL,MODES=co,\\"
echo "MAX_POLYPHONY=4,PROMPT_LENGTH=96,MAX_SONGS=10,\\"
echo "MEL_FOLDER=$DST/melody,CHORD_FOLDER=$DST/chord,CKPT=ckpt/<run>/ \\"
echo "      midi_yinyang/infer_duet_block_diffusion.sbatch"
echo "================================================================"
