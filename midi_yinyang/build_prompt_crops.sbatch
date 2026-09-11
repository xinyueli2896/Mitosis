#!/bin/bash
# SLURM (CPU): build the held-out inference set by CROPPING each song so
# every prompt holds the same musical content.
#
# The old selection asked that a song be in whole-song-gen's split AND
# that both streams happen to sound within the first 6 or 8 bars. The
# second is a property of how long the song's intro is, so it discarded
# most of the corpus and still left prompts holding different amounts of
# music. This crops TO the melody instead:
#
#     bar 1       pickup / buffer -- the anacrusis if the song has one,
#                 silence if it starts on a bar line
#     bars 2..5   the first MEL_BARS (4) consecutive bars in which BOTH
#                 streams sound, beginning on a bar line
#
# Chord is cut on exactly the same boundary and everything before the
# pickup bar is discarded, so the streams stay in register by
# construction. Bars come from each file's OWN time signatures, so a
# song with a 6/4 bar is measured in its real bars.
#
# Env knobs (--export=ALL,KEY=val):
#   MEL_SRC, CHORD_SRC   default the v5 aligner output
#   DST                  default input/heldout_v5
#   DATASET              dataset stem whose index %% SPLIT_RATIO == 0 songs
#                        are held out (default pop909_melody_cp8_v2). Use
#                        the stem the MODEL trains on -- with PT_TAG that
#                        is e.g. pop909_melody_cp8_v2_v5 -- or the split
#                        will not match what the model never saw.
#   IDS                  explicit ids; skips the held-out filter entirely
#   FORCE_IDS            ids kept even when the normal rule rejects them.
#                        The crop point is still chosen, by the first
#                        fallback that works (melody-only bars, then any
#                        bar-aligned melody start, then any melody), and
#                        MIN_BARS is waived. The TSV's `forced` column
#                        records which fallback was used, so a forced
#                        song is never mistaken for one that passed. Use
#                        it for songs you have LISTENED to.
#   EXTRA_IDS            ids ADDED to the held-out set (default "001 002
#                        003 004 005", the songs cut before preprocessing:
#                        absent from the dataset, so the index rule cannot
#                        find them, but never trained on either). Refused
#                        if any is in the TRAIN split. Set to " " for none.
#   MEL_BARS             sounding bars in the prompt (default 4). The
#                        prompt is MEL_BARS + 2 bars: a LEAD bar, the
#                        pickup bar, then the sounding bars. The lead is
#                        the song's own bar where it has one and an empty
#                        padded bar where it does not -- pad_bars and
#                        pad_frames in the TSV say which, and the padding
#                        shifts both streams identically so they stay in
#                        register.
#   MIN_BARS             bars that must remain after the crop so there is
#                        room to generate (default 25)
#   ALIGN_REPORT         align_grid_report.tsv from the aligner. Supplies
#                        each song's IRREGULAR BAR positions, which a flat
#                        4/4 midi cannot carry once TIME_SIGS=0. Given it,
#                        the crop point is chosen to land the scored
#                        window in a metrically REGULAR stretch -- so a
#                        song with a metre change elsewhere stays usable
#                        instead of being dropped or silently scored
#                        across a bar-grid shift. Default: the report
#                        next to MEL_SRC, if one is there.
#   REQUIRE_REGULAR      1 = drop songs whose metre changes inside the
#                        scored window. Off by default so the set stays
#                        large; the TSV reports it either way and those
#                        songs put a phase shift under every bar-based
#                        metric.
#
#   sbatch midi_yinyang/build_prompt_crops.sbatch
#   sbatch --export=ALL,REQUIRE_REGULAR=1 midi_yinyang/build_prompt_crops.sbatch

#SBATCH --job-name=build_prompt_crops
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
mkdir -p /home/xinyue.li/logs

POP=/home/xinyue.li/POP909-Dataset
MEL_SRC="${MEL_SRC:-$POP/POP909-v5/melody}"
CHORD_SRC="${CHORD_SRC:-$POP/POP909-v5/chord}"
DST="${DST:-input/heldout_v5}"
DATASET="${DATASET:-pop909_melody_cp8_v2}"
SPLIT_RATIO="${SPLIT_RATIO:-10}"
IDS="${IDS:-}"
EXTRA_IDS="${EXTRA_IDS:-001 002 003 004 005}"
FORCE_IDS="${FORCE_IDS:-004}"
MEL_BARS="${MEL_BARS:-4}"
MIN_BARS="${MIN_BARS:-25}"
REQUIRE_REGULAR="${REQUIRE_REGULAR:-0}"
# default: the aligner's report sits one level above the melody folder
ALIGN_REPORT="${ALIGN_REPORT:-$(dirname "$MEL_SRC")/align_grid_report.tsv}"
[[ -f "$ALIGN_REPORT" ]] || ALIGN_REPORT=""

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "MEL_SRC=$MEL_SRC"
echo "CHORD_SRC=$CHORD_SRC"
echo "DST=$DST   DATASET=$DATASET (idx %% $SPLIT_RATIO == 0 held out)"
echo "prompt = 1 buffer bar + $MEL_BARS sounding bars; MIN_BARS=$MIN_BARS"
echo "REQUIRE_REGULAR=$REQUIRE_REGULAR   IDS=${IDS:-<held-out split>}"
echo "EXTRA_IDS=${EXTRA_IDS:-<none>}   FORCE_IDS=${FORCE_IDS:-<none>}"
echo "ALIGN_REPORT=${ALIGN_REPORT:-<none: windows cannot avoid metre changes>}"
echo "================================================================"

for d in "$MEL_SRC" "$CHORD_SRC"; do
    [[ -d "$d" ]] || { echo "ERROR: missing folder $d"; exit 1; }
done

python build_prompt_crops.py \
    --mel-src "$MEL_SRC" --chord-src "$CHORD_SRC" --dst "$DST" \
    --dataset "$DATASET" --split-ratio "$SPLIT_RATIO" \
    --mel-bars "$MEL_BARS" --min-bars "$MIN_BARS" \
    --extra-ids $EXTRA_IDS \
    ${FORCE_IDS:+--force-ids $FORCE_IDS} \
    ${ALIGN_REPORT:+--align-report "$ALIGN_REPORT"} \
    ${IDS:+--ids $IDS} \
    $([[ "$REQUIRE_REGULAR" == "1" ]] && echo --require-regular)

echo "================================================================"
echo "Set: $DST/{melody,chord}   report: $DST/prompt_crops.tsv"
echo "Every prompt is now the first $((MEL_BARS + 2)) bars of the file, so"
echo "PROMPT_LENGTH = $(( (MEL_BARS + 2) * 16 )) frames for every song --"
echo "no more per-song p6/p8 groups to merge."
echo "================================================================"
