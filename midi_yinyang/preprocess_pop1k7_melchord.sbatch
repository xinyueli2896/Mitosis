#!/bin/bash
# SLURM (CPU): Pop1K7 -> paired melody/chord .pt files for the melchord
# pipeline (task 2, dataset expansion).
#
# Input: the annotated Pop1K7 release at POP1K7_DIR with subfolders
# src_001..src_004 (the 'test' subfolder is deliberately NOT touched --
# it stays held out). Every midi carries three NAMED tracks: melody,
# chord, piano (accompaniment). The pipeline:
#
#   1. split_named_tracks.py  -- per song, write melody.mid / chord.mid
#      (meta/tempo track + the one NAMED track; selection by name, so a
#      reordered file cannot swap streams; the piano track is dropped).
#      Songs missing either track are skipped and listed.
#   2. check_tokenizer_grid.py -- grid check in TICK space, the domain
#      the tokenizer actually reads. VERIFIED on the held-out test set
#      (jobs 185995/186001): Pop1K7 is a beat-synced-transcription
#      layout -- tick grid perfectly clean (0.0% off-grid), ~one tempo
#      event per beat carrying the performance timing. The
#      seconds-domain check_beat_alignment flags such files TRIPLET/
#      OFF-GRID en masse (it assumes one constant tempo); that verdict
#      is VOID for this corpus -- do not exclude songs over it.
#      Tokenization discards the tempo map by design, so decoded output
#      is on a straight metronomic grid (no rubato).
#   3. tokenize both streams (ins_ids=['track-0'] on the split files,
#      filter=False so no per-stream drops can desync the pairing) into
#        data/pop1k7_melody_cp${MAX_POLYPHONY}_v2.pt
#        data/pop1k7_chord_cp${MAX_POLYPHONY}_v2.pt
#   4. check_paired_dataset.py -- HARD pairing gate.
#   5. audit_pt_melchord.py -- decode a handful of songs back to midi
#      under temp/pop1k7_audit; LISTEN to them against the sources
#      before training on the data.
#
# Env knobs (--export=ALL,KEY=val):
#   POP1K7_DIR     default /home/xinyue.li/pop1k7
#   SRC_SUBDIRS    default "src_001 src_002 src_003 src_004"
#   DST_PREFIX     split-folder prefix, default Pop1K7-Dataset/pop1k7
#                  (-> Pop1K7-Dataset/pop1k7-melody , .../pop1k7-chord)
#   MAX_POLYPHONY  default 8 (the certified melchord budget; run again
#                  with 4 for a cp4tar-family-compatible set -- names
#                  encode the budget, nothing clobbers).
#
# Example:
#   sbatch --exclude=cn-20 midi_yinyang/preprocess_pop1k7_melchord.sbatch

#SBATCH --job-name=mitosis_preproc_pop1k7
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=03:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

POP1K7_DIR="${POP1K7_DIR:-/home/xinyue.li/pop1k7}"
SRC_SUBDIRS="${SRC_SUBDIRS:-src_001 src_002 src_003 src_004}"
DST_PREFIX="${DST_PREFIX:-Pop1K7-Dataset/pop1k7}"
MAX_POLYPHONY="${MAX_POLYPHONY:-8}"

MEL_DIR="${DST_PREFIX}-melody"
CHD_DIR="${DST_PREFIX}-chord"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs data

SRC_ARGS=()
for d in $SRC_SUBDIRS; do
    [[ -d "$POP1K7_DIR/$d" ]] || { echo "ERROR: missing $POP1K7_DIR/$d"; exit 1; }
    SRC_ARGS+=(--src "$POP1K7_DIR/$d")
done

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "POP1K7 MELCHORD PREPROCESS   cp${MAX_POLYPHONY}"
echo "sources: $SRC_SUBDIRS under $POP1K7_DIR  ('test' untouched)"
echo "split   : $MEL_DIR , $CHD_DIR"
echo "output  : data/pop1k7_{melody,chord}_cp${MAX_POLYPHONY}_v2.pt"
echo "================================================================"

# --- 1. split by track name -----------------------------------------
python split_named_tracks.py \
    "${SRC_ARGS[@]}" \
    --dst-melody "$MEL_DIR" --dst-chord "$CHD_DIR"

N_PAIRS=$(ls "$MEL_DIR" | wc -l)
[[ "$N_PAIRS" -gt 0 ]] || { echo "ERROR: split produced no pairs"; exit 1; }

# --- 2. grid check in the TOKENIZER'S domain (tick space) ------------
# (check_beat_alignment measures seconds through the tempo map and
#  false-alarms on Pop1K7's beat-synced layout -- see the header note.)
echo "================================================================"
echo "--- tokenizer-view grid check (DIRTY songs would be corrupted"
echo "--- by tokenization; review before training) ---"
python check_tokenizer_grid.py \
    --folder "$MEL_DIR" --folder "$CHD_DIR" || true

# --- 3. tokenize both streams ----------------------------------------
echo "================================================================"
python - "$MEL_DIR" "$CHD_DIR" "$MAX_POLYPHONY" <<'PYEOF'
import sys
from preprocess_large_midi_dataset import create_npy_dataset_from_midi
mel_dir, chd_dir, poly = sys.argv[1], sys.argv[2], int(sys.argv[3])
# Split files carry exactly one note-bearing track each; filter=False so
# no per-stream drop can desync the pairing (the pairing gate below
# verifies).
for folder, name in ((mel_dir, f'pop1k7_melody_cp{poly}_v2'),
                     (chd_dir, f'pop1k7_chord_cp{poly}_v2')):
    create_npy_dataset_from_midi(folder, poly, name,
                                 ins_ids=['track-0'],
                                 scan_subfolders=False, filter=False)
PYEOF

# --- 4. pairing gate --------------------------------------------------
python check_paired_dataset.py \
    "data/pop1k7_melody_cp${MAX_POLYPHONY}_v2" \
    "data/pop1k7_chord_cp${MAX_POLYPHONY}_v2"

# --- 5. listening audit ----------------------------------------------
python audit_pt_melchord.py \
    --mel-prefix "data/pop1k7_melody_cp${MAX_POLYPHONY}_v2" \
    --chord-prefix "data/pop1k7_chord_cp${MAX_POLYPHONY}_v2" \
    --out-dir temp/pop1k7_audit --n-songs 8

echo "----------------------------------------------------------------"
ls -lh data/pop1k7_*_cp"${MAX_POLYPHONY}"_v2* || true
echo "================================================================"
echo "Pop1K7 melchord tokenization complete (cp${MAX_POLYPHONY}, ${N_PAIRS} songs)."
echo "LISTEN FIRST: temp/pop1k7_audit (decoded vs source)."
echo "Duet training on it: sbatch --export=ALL,TASK=melchord_pop1k7 \\"
echo "    midi_yinyang/train_duet_block_diffusion.sbatch"
echo "Combine with POP909/Nottingham: combine_melchord_datasets.py"
echo "================================================================"
