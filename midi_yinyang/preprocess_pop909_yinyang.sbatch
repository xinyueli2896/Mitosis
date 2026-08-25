#!/bin/bash
# SLURM (CPU): build the PAIRED single-file dataset the YinYang adapter
# trainer consumes, from the tagged merged POP909 melody+chord folder.
#
# The YinYang finetune (cp_transformer_yinyang.py) does NOT read the
# duet family's two-file melchord datasets. It expects ONE .pt where
# each frame packs BOTH streams side by side in the subseq dimension --
# the format create_npy_dataset_from_midi produces when ins_ids lists
# TWO groups. This mirrors exactly how the published Y-mc data was
# built (create_nottingham_parts: ins_ids=['track-1','track-0'], cp8):
#
#   half 0 = track-1 = CHORD     half 1 = track-0 = MELODY
#
# hence the '_chord_mel' dataset/ckpt naming convention, on which BOTH
# the inference driver and yinyang_e3_driver.py key their direction
# logic: a non-_rev ckpt is chord->mel, a _rev ckpt is mel->chord.
# Keep the name suffix '_chord_mel' or that auto-detection breaks.
#
# Input folder: the TAGGED merged POP909 folder built by
# merge_melody_chord.py (track layout: melody track then chord track;
# chord program forced to 48 -- irrelevant here since selection is by
# track index, but it keeps the folder shared with the S-model prompts).
#
# Env knobs (--export=ALL,KEY=val):
#   COMBINED_DIR   default POP909-Dataset/POP909-melody-chord-tagged
#   MAX_POLYPHONY  PER-HALF token budget. Default 8 (matches Y-mc's cp8;
#                  POP909 seventh chords are 5 notes, so 4 would drop
#                  notes). This is independent of the duet family's
#                  MELCHORD_CP -- the comparison happens on MIDI
#                  outputs, not tokens.
#   DATASET_NAME   default pop909_cp<P>_v2_chord_mel
#
# Example:
#   sbatch --exclude=cn-20 midi_yinyang/preprocess_pop909_yinyang.sbatch

#SBATCH --job-name=mitosis_preproc_yy909
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

COMBINED_DIR="${COMBINED_DIR:-POP909-Dataset/POP909-melody-chord-tagged}"
MAX_POLYPHONY="${MAX_POLYPHONY:-8}"
DATASET_NAME="${DATASET_NAME:-pop909_cp${MAX_POLYPHONY}_v2_chord_mel}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs data

[[ -d "$COMBINED_DIR" ]] || {
    echo "ERROR: COMBINED_DIR not found: $COMBINED_DIR"
    echo "Build it first with merge_melody_chord.sbatch (tagged output)."
    exit 1
}

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "YINYANG PAIRED DATASET (POP909)  ->  data/${DATASET_NAME}.pt"
echo "COMBINED_DIR=$COMBINED_DIR   per-half budget cp$MAX_POLYPHONY"
echo "half 0 = chord (track-1), half 1 = melody (track-0)  ['_chord_mel']"
echo "================================================================"

# GUARD: 'track-N' selects by note-bearing track index, so every merged
# midi must have EXACTLY two instrument tracks (melody, chord) or the
# halves silently pick the wrong content.
python - "$COMBINED_DIR" <<'PYEOF'
import sys, os, warnings
import pretty_midi
folder = sys.argv[1]
bad = []
files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.mid'))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for f in files:
        pm = pretty_midi.PrettyMIDI(os.path.join(folder, f))
        n = len([i for i in pm.instruments if len(i.notes) > 0])
        if n != 2:
            bad.append((f, n))
if bad:
    print(f'ERROR: {len(bad)} file(s) do not have exactly 2 note-bearing '
          f'tracks: {bad[:8]}')
    sys.exit(1)
print(f'[guard] all {len(files)} merged midis have exactly 2 note-bearing '
      f'tracks (melody, chord)')
PYEOF

python - "$COMBINED_DIR" "$MAX_POLYPHONY" "$DATASET_NAME" <<'PYEOF'
import sys
from preprocess_large_midi_dataset import create_npy_dataset_from_midi
folder, poly, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# Same ins_ids order as the published Nottingham Y-mc build:
# half 0 = track-1 (chord), half 1 = track-0 (melody).
create_npy_dataset_from_midi(folder, poly, name,
                             ins_ids=['track-1', 'track-0'],
                             scan_subfolders=False)
PYEOF

echo "================================================================"
ls -la "data/${DATASET_NAME}."*
echo "Next: train both directions with train_yinyang_pop909.sbatch"
echo "  DIRECTION=chord2mel (default)  and  DIRECTION=mel2chord"
echo "================================================================"
