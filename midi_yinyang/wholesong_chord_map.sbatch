#!/bin/bash
# SLURM (CPU): where the whole-song baseline's chord differs from OUR
# prompt chord, the map f with ours = f(theirs), and (optionally) the
# baseline's outputs re-voiced through f as a new system.
#
# The baseline is prompted with our chord but writes its own rendering
# of it: its lead-sheet image holds a chord as a chroma band and a bass
# band, one row per pitch class, and its converter renders those back
# at its own octaves. Over the prompt bars both renderings of the SAME
# chord exist, so the map can be fitted there and checked exactly.
#
# The log has three sections:
#   1. DIFF   raw frame agreement (absolute pitch / pitch class /
#             onsets), each side's octave convention, and every frame
#             where the PITCH-CLASS content differs -- the part no
#             re-voicing can fix. Zero here means the baseline changed
#             nothing but the rendering.
#   2. FIT    the candidate maps ranked by exact agreement with our
#             prompt chord, the winner, and its residual frames.
#   3. APPLY  (APPLY=<name>) every sample of the system rewritten with
#             chord -> f(chord), melody and tempo untouched, as
#             <OUT_ROOT>/<name>/<song>/co/sample_<i>.mid. Score it as a
#             duet_multi system (eval_e1 SYSTEMS=<name> SKIP_INFER=1
#             after adding it to the system list, or score_e1 with a
#             --source line).
#
# Env knobs (--export=ALL,KEY=val):
#   OUT_ROOT       default temp/E1_v5b
#   SYSTEM         default WSf
#   APPLY          new system name to write, e.g. WSfv; empty = analyse
#   PROMPT_FRAMES  default 96
#   EXCLUDE_SONGS  default "506 786"
#   SHOW           songs to print in detail, default 4
#   FORCE_MAP      "bass,upper,fold,merge" to override the fit, e.g.
#                  36,60,1,0
#
# Examples:
#   sbatch --export=ALL,OUT_ROOT=temp/E1_v5b midi_yinyang/wholesong_chord_map.sbatch
#   sbatch --export=ALL,OUT_ROOT=temp/E1_v5b,APPLY=WSfv midi_yinyang/wholesong_chord_map.sbatch

#SBATCH --job-name=wholesong_chord_map
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

OUT_ROOT="${OUT_ROOT:-temp/E1_v5b}"
SYSTEM="${SYSTEM:-WSf}"
APPLY="${APPLY:-}"
PROMPT_FRAMES="${PROMPT_FRAMES:-96}"
EXCLUDE_SONGS="${EXCLUDE_SONGS-506 786}"
SHOW="${SHOW:-4}"
FORCE_MAP="${FORCE_MAP:-}"
CHORD_DIR="${CHORD_DIR:-$OUT_ROOT/prompts/chord}"

[[ -d "$CHORD_DIR" ]] || { echo "ERROR: no staged chord prompts at $CHORD_DIR"; exit 1; }
[[ -d "$OUT_ROOT/$SYSTEM" ]] || { echo "ERROR: no $OUT_ROOT/$SYSTEM"; exit 1; }
if [[ -n "$APPLY" && -d "$OUT_ROOT/$APPLY" ]]; then
    echo "NOTE: $OUT_ROOT/$APPLY exists; its files are overwritten"
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "OUT_ROOT=$OUT_ROOT  SYSTEM=$SYSTEM  APPLY=${APPLY:-<analyse only>}  PROMPT_FRAMES=$PROMPT_FRAMES"
echo "================================================================"

python wholesong_chord_map.py --out-root "$OUT_ROOT" --system "$SYSTEM" \
    --chord-dir "$CHORD_DIR" --prompt-frames "$PROMPT_FRAMES" --show "$SHOW" \
    ${EXCLUDE_SONGS:+--exclude-songs $EXCLUDE_SONGS} \
    ${APPLY:+--apply "$APPLY"} ${FORCE_MAP:+--force-map "$FORCE_MAP"}
