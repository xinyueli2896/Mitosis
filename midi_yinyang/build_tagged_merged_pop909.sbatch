#!/bin/bash
# SLURM (CPU): build the TAGGED merged POP909 folder -- melody and chord
# in one midi with DISTINCT programs -- and verify the tagging.
#
# Why this matters: the merged single-stream tokenizer separates streams
# by PROGRAM ALONE. Merging with both streams on program 0 fuses them
# irreversibly, and any model trained on that data (S1 and S-scratch
# alike) can only ever emit one program -- which reads as a model
# failure but is a data-prep one. This script makes the tagged folder
# and refuses to leave a fused one behind.
#
# Output: <DST> holding <id>.mid per song, melody on MEL_PROGRAM (kept
# as-is by default) and chord forced to CHORD_PROGRAM (48, strings).
#
# Env knobs (--export=ALL,KEY=val):
#   MELODY_DIR     default /home/xinyue.li/POP909-Dataset/POP909-melody
#   CHORD_DIR      default /home/xinyue.li/POP909-Dataset/POP909-chord
#   DST            default POP909-Dataset/POP909-melody-chord-tagged
#                  (relative paths resolve under midi_yinyang/)
#   CHORD_PROGRAM  default 48 (String Ensemble 1)
#   MEL_PROGRAM    default unset -- melody keeps its own program (0)
#
# Then build the dataset and retrain the merged models:
#   sbatch --export=ALL,COMBINED_DIR=<DST>,\
# DATASET_NAME=pop909_melchord_tagged_cp16_v2 \
#       midi_yinyang/preprocess_pop909_melchord.sbatch
#   sbatch --export=ALL,DATA=data/pop909_melchord_tagged_cp16_v2.pt,\
# RUN_TAG=tagged midi_yinyang/finetune_pop909.sbatch

#SBATCH --job-name=build_tagged_merged
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

MELODY_DIR="${MELODY_DIR:-/home/xinyue.li/POP909-Dataset/POP909-melody}"
CHORD_DIR="${CHORD_DIR:-/home/xinyue.li/POP909-Dataset/POP909-chord}"
DST="${DST:-POP909-Dataset/POP909-melody-chord-tagged}"
CHORD_PROGRAM="${CHORD_PROGRAM:-48}"
MEL_PROGRAM="${MEL_PROGRAM:-}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

for d in "$MELODY_DIR" "$CHORD_DIR"; do
    [[ -d "$d" ]] || { echo "ERROR: $d not found"; exit 1; }
done
N_MEL=$(find "$MELODY_DIR" -maxdepth 1 -name '*.mid' | wc -l)
N_CHD=$(find "$CHORD_DIR" -maxdepth 1 -name '*.mid' | wc -l)

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "BUILD TAGGED MERGED POP909"
echo "MELODY_DIR=$MELODY_DIR  ($N_MEL midis)"
echo "CHORD_DIR =$CHORD_DIR  ($N_CHD midis)"
echo "DST=$DST   chord program -> $CHORD_PROGRAM"
echo "MEL_PROGRAM=${MEL_PROGRAM:-<keep source program>}"
echo "================================================================"
if [[ "$N_MEL" -eq 0 || "$N_CHD" -eq 0 ]]; then
    echo "ERROR: melody/chord folders must both hold midis."
    echo "  Run preprocess_pop909.sbatch first (it extracts melody and"
    echo "  renders the chord track)."
    exit 1
fi

EXTRA=()
[[ -n "$MEL_PROGRAM" ]] && EXTRA+=(--mel-program "$MEL_PROGRAM")

python merge_melody_chord.py \
    --melody "$MELODY_DIR" \
    --chord "$CHORD_DIR" \
    --dst "$DST" \
    --chord-program "$CHORD_PROGRAM" \
    "${EXTRA[@]}"

# Verify: the streams must land on DIFFERENT programs, or every model
# trained on this folder will fuse them (the failure this script exists
# to prevent).
echo "----------------------------------------------------------------"
python - "$DST" "$CHORD_PROGRAM" <<'PYEOF'
import glob, os, sys, warnings
import pretty_midi
dst, chord_program = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(dst, '*.mid')))
if not files:
    sys.exit('VERIFY FAILED: no midis written to ' + dst)
bad = []
for f in files[:20]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pm = pretty_midi.PrettyMIDI(f)
    insts = [(i.name, int(i.program), len(i.notes))
             for i in pm.instruments if i.notes]
    progs = {p for _, p, _ in insts}
    if len(files[:3]) and f in files[:3]:
        print(f'  {os.path.basename(f)}: {insts}')
    if chord_program not in progs or len(progs) < 2:
        bad.append((os.path.basename(f), sorted(progs)))
print(f'  wrote {len(files)} merged midis; checked {len(files[:20])}')
if bad:
    print(f'VERIFY FAILED: {len(bad)} sampled file(s) lack two distinct '
          f'programs, e.g. {bad[:3]}')
    sys.exit(1)
print(f'  VERIFIED: every sampled file has >=2 programs including '
      f'{chord_program}.')
PYEOF

echo "================================================================"
echo "Tagged merged folder ready: $DST"
echo "Next:"
echo "  sbatch --export=ALL,COMBINED_DIR=$DST,\\"
echo "DATASET_NAME=pop909_melchord_tagged_cp16_v2 \\"
echo "      midi_yinyang/preprocess_pop909_melchord.sbatch"
echo "================================================================"
