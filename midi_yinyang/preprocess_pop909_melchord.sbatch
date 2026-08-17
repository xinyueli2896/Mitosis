#!/bin/bash
# SLURM batch script (CPU): tokenize a combined melody+chord midi folder
# into the .pt training files (single-stream CP format).
#
# Output (under midi_yinyang/data/):
#   <DATASET_NAME>.pt                    token rolls
#   <DATASET_NAME>.length.pt             per-song frame counts
#   <DATASET_NAME>.pitch_shift_range.pt  transposition ranges
#   <DATASET_NAME>.txt                   song-index manifest
#
# Env knobs (override via --export=ALL,KEY=val):
#   COMBINED_DIR   folder of combined midis.
#                  default: POP909-Dataset/POP909-melody-chord
#   DATASET_NAME   output name. default: pop909_melchord_cp16_v2
#                  Use a DIFFERENT name per source folder -- e.g.
#                  pop909_melchord_untagged_cp16_v2 for the same-program
#                  folder -- so tagged/untagged data never clobber.
#
# NOTE on stream separation: the single-stream tokenizer identifies
# streams by PROGRAM only. If COMBINED_DIR has melody and chord on the
# same program (POP909-melody-chord does), the two streams FUSE in the
# tokens and can never be separated downstream. For the finetune pipeline
# use the tagged folder (chord program 48) built by:
#   python merge_melody_chord.py --melody ... --chord ... \
#       --dst POP909-Dataset/POP909-melody-chord-tagged --chord-program 48
# (finetune_cp_transformer_pop909.sbatch does all of this automatically.)
#
# Examples:
#   sbatch midi_yinyang/preprocess_pop909_melchord.sbatch
#   COMBINED_DIR=POP909-Dataset/POP909-melody-chord-tagged \
#       sbatch --export=ALL midi_yinyang/preprocess_pop909_melchord.sbatch

#SBATCH --job-name=mitosis_preproc_melchord
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

COMBINED_DIR="${COMBINED_DIR:-POP909-Dataset/POP909-melody-chord}"
DATASET_NAME="${DATASET_NAME:-pop909_melchord_cp16_v2}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs data

N_MID=$(find "$COMBINED_DIR" -maxdepth 1 -name '*.mid' 2>/dev/null | wc -l)
echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "COMBINED_DIR = $COMBINED_DIR  ($N_MID midis)"
echo "DATASET_NAME = $DATASET_NAME"
echo "================================================================"
if [[ "$N_MID" -eq 0 ]]; then
    echo "ERROR: no midis in $COMBINED_DIR"
    exit 1
fi

# GUARD: the single-stream tokenizer separates melody from chord by
# PROGRAM ALONE. Building from a folder whose two streams share a
# program silently produces a dataset in which they are fused beyond
# recovery -- every model trained on it (S1 and S-scratch alike) then
# emits one program and cannot maintain two streams, which looks like a
# model failure but is a data-prep one. Refuse unless FORCE_UNTAGGED=1.
python - "$COMBINED_DIR" <<'PYEOF'
import glob, os, sys, warnings
import pretty_midi
folder = sys.argv[1]
files = sorted(glob.glob(os.path.join(folder, '*.mid')))[:20]
progs = set()
for f in files:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pm = pretty_midi.PrettyMIDI(f)
    progs |= {i.program for i in pm.instruments if not i.is_drum and i.notes}
print(f'[guard] non-drum programs across {len(files)} sampled midis: '
      f'{sorted(progs)}')
sys.exit(0 if len(progs) > 1 else 3)
PYEOF
rc=$?
if [[ "$rc" -eq 3 && "${FORCE_UNTAGGED:-0}" != "1" ]]; then
    echo "ERROR: $COMBINED_DIR has melody and chord on ONE program."
    echo "  The merged tokenizer keys streams on program, so this would"
    echo "  build a dataset with the streams FUSED (and any model trained"
    echo "  on it can only ever emit one program). Build the tagged"
    echo "  folder first:"
    echo "    python merge_melody_chord.py \\"
    echo "        --melody POP909-Dataset/POP909-melody \\"
    echo "        --chord  POP909-Dataset/POP909-chord \\"
    echo "        --dst    POP909-Dataset/POP909-melody-chord-tagged \\"
    echo "        --chord-program 48"
    echo "  then rerun with COMBINED_DIR=POP909-Dataset/POP909-melody-chord-tagged"
    echo "  (FORCE_UNTAGGED=1 overrides, for a deliberate fused-data ablation)."
    exit 1
fi

POP909_COMBINED_PATH="$COMBINED_DIR" \
POP909_MELCHORD_DATASET="$DATASET_NAME" \
    python preprocess_large_midi_dataset.py pop909_melchord

echo "----------------------------------------------------------------"
ls -lh "data/$DATASET_NAME"* || true
echo "================================================================"
echo "Done. Tokenized dataset: data/$DATASET_NAME.pt"
