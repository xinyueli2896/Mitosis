#!/bin/bash
# SLURM batch script (CPU): tokenize the Nottingham dataset into paired
# melody/chord .pt files for the melchord pipeline.
#
# Nottingham midis carry melody as instrument 0 and the rendered chord
# accompaniment as instrument 1; both streams are tokenized at
# max_polyphony=4 to match the POP909 melchord convention, so models and
# tooling are interchangeable across the two corpora. Primary purpose:
# a domain-matched counterpart to the Nottingham-trained YinYang
# conditional baseline (EXPERIMENTS.md §2.4) — enables matched-domain
# evaluation and/or Nottingham-trained duet/finetune variants.
#
# A minority of tunes (mostly "morris") carry melody only, with no
# chord track at all; the preprocessing entrypoints pre-scan the
# folder and exclude those songs from BOTH streams up front, so the
# two passes tokenize an identical file set and the pairing gate below
# passes cleanly (this is expected -- ~1% of the corpus, not a bug).
#
# Output (under midi_yinyang/data/):
#   nottingham_melody_cp4_v2.pt (+ .length.pt, .pitch_shift_range.pt, .txt)
#   nottingham_chord_cp4_v2.pt  (+ sidecars)
# followed by the song-pairing gate (check_paired_dataset.py).
#
# Env knobs (override via --export=ALL,KEY=val):
#   NOTTINGHAM_DIR  folder that DIRECTLY contains the .mid files.
#                   default: /home/xinyue.li/nottingham-dataset/MIDI
#                   (clone: https://github.com/jukedeck/nottingham-dataset)
#
# Example:
#   sbatch midi_yinyang/preprocess_nottingham_melchord.sbatch

#SBATCH --job-name=mitosis_preproc_nottingham
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

NOTTINGHAM_DIR="${NOTTINGHAM_DIR:-/home/xinyue.li/nottingham-dataset/MIDI}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs data

N_MID=$(find "$NOTTINGHAM_DIR" -maxdepth 1 -name '*.mid' 2>/dev/null | wc -l)
echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "NOTTINGHAM_DIR = $NOTTINGHAM_DIR  ($N_MID midis)"
echo "================================================================"
if [[ "$N_MID" -eq 0 ]]; then
    echo "ERROR: no midis in $NOTTINGHAM_DIR"
    echo "Clone the dataset first:"
    echo "  git clone https://github.com/jukedeck/nottingham-dataset \\"
    echo "      /home/xinyue.li/nottingham-dataset"
    exit 1
fi

NOTTINGHAM_MIDI_PATH="$NOTTINGHAM_DIR" \
    python preprocess_large_midi_dataset.py nottingham_melody
NOTTINGHAM_MIDI_PATH="$NOTTINGHAM_DIR" \
    python preprocess_large_midi_dataset.py nottingham_chord

python check_paired_dataset.py \
    data/nottingham_melody_cp4_v2 data/nottingham_chord_cp4_v2

# Listening audit: decode a handful of songs back to midi (melody /
# chord / combined, chord remapped to program 48 for separation) plus
# the per-stream notes-per-frame orientation check. Listen to these
# against the source midis before training on the data.
python audit_pt_melchord.py \
    --mel-prefix data/nottingham_melody_cp4_v2 \
    --chord-prefix data/nottingham_chord_cp4_v2 \
    --out-dir temp/nottingham_audit --n-songs 8

echo "----------------------------------------------------------------"
ls -lh data/nottingham_*_cp4_v2* || true
echo "================================================================"
echo "Nottingham melchord tokenization complete."
echo "  data/nottingham_melody_cp4_v2.pt"
echo "  data/nottingham_chord_cp4_v2.pt"
echo "Duet training on it: sbatch --export=ALL,TASK=melchord_nottingham \\"
echo "    midi_yinyang/train_duet_block_diffusion.sbatch"
