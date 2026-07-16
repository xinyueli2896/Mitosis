#!/bin/bash
# SLURM batch script: end-to-end POP909 melody/chord preprocessing for the
# melchord task (A.1/A.3/... training via tasks.py).
#
# Pipeline (each stage skippable / resumable -- stages that already have
# output are only re-run with FORCE=1):
#   1. align   preprocess_pop909_align.py:  POP909/<id>/{<id>.mid,beat_midi.txt}
#              -> POP909-aligned/<id>.mid  (beat grid matches beat_midi.txt)
#              Skipped automatically when ALIGNED_DIR already has midis
#              (e.g. you downloaded the pre-aligned set, HF: c0smic1atte/909_aligned).
#   2. melody  extract_pop909_melody.py:    aligned -> POP909-melody/<id>.mid
#              (tempo track + the track named MELODY)
#   3. chord   build_pop909_chord_midi.py:  aligned tempo + POP909/<id>/chord_midi.txt
#              -> POP909-chord/<id>.mid (bass + 3 upper voices per chord)
#              NOTE: needs the RAW dataset for chord_midi.txt even when
#              alignment was skipped.
#   4. tokenize preprocess_large_midi_dataset.py pop909_melody / pop909_chord
#              -> data/pop909_melody_cp4_v2.pt / data/pop909_chord_cp4_v2.pt
#              (+ .length.pt, .pitch_shift_range.pt, .txt manifests)
#   5. verify  check_paired_dataset.py: both manifests keep exactly the same
#              songs in the same order (FramedDataset pairs by index).
#
# Env knobs (override via --export=ALL,KEY=val):
#   POP909_RAW    raw dataset root (folders <id>/ with midi+beat+chord txt).
#                 default: POP909-Dataset/POP909
#   ALIGNED_DIR   default: POP909-Dataset/POP909-aligned
#   MELODY_DIR    default: POP909-Dataset/POP909-melody
#   CHORD_DIR     default: POP909-Dataset/POP909-chord
#   FORCE=1       re-run stages whose output already exists.
#
# After this finishes, train A.3 on melody/chord with:
#   sbatch --export=ALL,TASK=melchord midi_yinyang/train_duet_block_diffusion.sbatch

#SBATCH --job-name=mitosis_preprocess_pop909
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=04:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

POP909_RAW="${POP909_RAW:-POP909-Dataset/POP909}"
ALIGNED_DIR="${ALIGNED_DIR:-POP909-Dataset/POP909-aligned}"
MELODY_DIR="${MELODY_DIR:-POP909-Dataset/POP909-melody}"
CHORD_DIR="${CHORD_DIR:-POP909-Dataset/POP909-chord}"
FORCE="${FORCE:-0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs data

count_mid() { find "$1" -maxdepth 1 -name '*.mid' 2>/dev/null | wc -l; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "POP909_RAW  = $POP909_RAW   ($(ls "$POP909_RAW" 2>/dev/null | wc -l) entries)"
echo "ALIGNED_DIR = $ALIGNED_DIR  ($(count_mid "$ALIGNED_DIR") midis)"
echo "MELODY_DIR  = $MELODY_DIR   ($(count_mid "$MELODY_DIR") midis)"
echo "CHORD_DIR   = $CHORD_DIR    ($(count_mid "$CHORD_DIR") midis)"
echo "================================================================"

# --- 1. align ---------------------------------------------------------
if [[ "$FORCE" != "1" && "$(count_mid "$ALIGNED_DIR")" -gt 0 ]]; then
    echo "[align] $ALIGNED_DIR already populated; skipping (FORCE=1 to redo)."
else
    if [[ ! -d "$POP909_RAW" ]]; then
        echo "[align] ERROR: no aligned midis and POP909_RAW=$POP909_RAW missing."
        exit 1
    fi
    python preprocess_pop909_align.py --src "$POP909_RAW" --dst "$ALIGNED_DIR"
fi

# --- 2. melody --------------------------------------------------------
if [[ "$FORCE" != "1" && "$(count_mid "$MELODY_DIR")" -gt 0 ]]; then
    echo "[melody] $MELODY_DIR already populated; skipping."
else
    python extract_pop909_melody.py --src "$ALIGNED_DIR" --dst "$MELODY_DIR"
fi

# --- 3. chord ---------------------------------------------------------
if [[ "$FORCE" != "1" && "$(count_mid "$CHORD_DIR")" -gt 0 ]]; then
    echo "[chord] $CHORD_DIR already populated; skipping."
else
    if [[ ! -d "$POP909_RAW" ]]; then
        echo "[chord] ERROR: POP909_RAW=$POP909_RAW missing (needed for chord_midi.txt)."
        exit 1
    fi
    python build_pop909_chord_midi.py \
        --aligned "$ALIGNED_DIR" --src "$POP909_RAW" --dst "$CHORD_DIR"
fi

# Melody and chord midi sets must match BEFORE tokenizing; a song present
# on one side only would silently shift the index pairing downstream.
python - "$MELODY_DIR" "$CHORD_DIR" <<'EOF'
import os, sys
mel = {f for f in os.listdir(sys.argv[1]) if f.endswith('.mid')}
cho = {f for f in os.listdir(sys.argv[2]) if f.endswith('.mid')}
only_m, only_c = sorted(mel - cho), sorted(cho - mel)
if only_m or only_c:
    print(f'[midi-pairing] melody-only: {only_m[:10]} ({len(only_m)})')
    print(f'[midi-pairing] chord-only:  {only_c[:10]} ({len(only_c)})')
    print('[midi-pairing] FAIL: remove the unmatched midis (or fix the '
          'failed stage) so both folders hold the same song ids.')
    sys.exit(1)
print(f'[midi-pairing] OK: {len(mel)} songs present in both folders.')
EOF

# --- 4. tokenize ------------------------------------------------------
# settings.py has relative POP909_MELODY_PATH/POP909_CHORD_PATH; export the
# actual dirs via env override supported below (fall back to symlinks when
# the dirs differ from the defaults in settings.py).
if [[ "$MELODY_DIR" != "POP909-Dataset/POP909-melody" || "$CHORD_DIR" != "POP909-Dataset/POP909-chord" ]]; then
    mkdir -p POP909-Dataset
    [[ -e POP909-Dataset/POP909-melody ]] || ln -s "$(readlink -f "$MELODY_DIR")" POP909-Dataset/POP909-melody
    [[ -e POP909-Dataset/POP909-chord  ]] || ln -s "$(readlink -f "$CHORD_DIR")"  POP909-Dataset/POP909-chord
fi

if [[ "$FORCE" != "1" && -f data/pop909_melody_cp4_v2.pt ]]; then
    echo "[tokenize] data/pop909_melody_cp4_v2.pt exists; skipping melody."
else
    python preprocess_large_midi_dataset.py pop909_melody
fi
if [[ "$FORCE" != "1" && -f data/pop909_chord_cp4_v2.pt ]]; then
    echo "[tokenize] data/pop909_chord_cp4_v2.pt exists; skipping chord."
else
    python preprocess_large_midi_dataset.py pop909_chord
fi

# --- 5. verify pairing -------------------------------------------------
python check_paired_dataset.py data/pop909_melody_cp4_v2 data/pop909_chord_cp4_v2

echo "================================================================"
echo "POP909 melchord preprocessing complete."
echo "  data/pop909_melody_cp4_v2.pt"
echo "  data/pop909_chord_cp4_v2.pt"
echo "Next: sbatch --export=ALL,TASK=melchord midi_yinyang/train_duet_block_diffusion.sbatch"
echo "================================================================"
