#!/bin/bash
# SLURM (CPU): POP909 -> metrically exact MIDI with pop909_align_grid.py.
#
# beat_midi.txt is the only yardstick. Column 1 maps wall-clock seconds
# to musical time; each note's onset and offset are interpolated into
# fractional beats SEPARATELY (never onset + duration -- the tempo can
# change mid-note), SNAPPED to a subdivision grid, and written out at a
# single constant tempo so the hand-labelled tempo curve is gone and the
# note grid is metrically exact.
#
# Differs from align_pop909.sbatch / pop909_beat_align.py in the two
# steps that one never did: it snapped nothing (it rounded to 1/480 of a
# beat, so off-grid performance timing survived into the tokenizer and
# got quantized there instead), and it wrote one tempo per beat to keep
# audio time, which keeps the curve alive. Both outputs are kept; this
# one is what you want for training data and prompts.
#
# Writes <DST>/{aligned,melody,chord}/ and <DST>/align_grid_report.tsv.
#
# Env knobs (--export=ALL,KEY=val):
#   POP909_RAW  song folders of the dataset repo
#               (default /home/xinyue.li/POP909-Dataset/POP909-git/POP909)
#   DST         output root (default /home/xinyue.li/POP909-Dataset/POP909-v4)
#   IDS         optional song ids (default: all)
#   SUB         grid subdivisions per beat (default 4 = 16ths). 24 keeps
#               triplets and 32nds, which POP909's own processed release
#               uses -- but the CP tokenizer reads a 16th grid, so at 24
#               the snap happens later instead of here. Prefer 4 and use
#               the report to EXCLUDE songs a 16th grid mangles.
#   TEMPO       constant output tempo (default 120: a beat is 0.5 s)
#   ORIGIN      beat0 (default; beat 0 stays at 0, pickup kept) or
#               downbeat (shift so bar 1 starts at beat 0; the pickup
#               then lands before 0 and is dropped). NEITHER pads.
#   NO_FIX_DROPPED  1 = do not restore beats the tracker missed. Default
#               restores them: a gap ~k x the local median means k-1
#               beats were dropped and every note inside lands a beat
#               late. Restoring RENUMBERS later beats, which is the
#               point, but it moves bar phase -- check the count.
#
# READ THE REPORT BEFORE USING THE OUTPUT. Three columns decide whether
# a song is usable:
#   mel_snap_p95          how far the worst 5% of melody onsets moved,
#                         in beats. ~0 means the song was already on the
#                         grid; > 0.02 means real timing was destroyed.
#   mel_needs_sub24_frac  share of onsets sitting on the 24 grid but NOT
#                         the 4 grid. High = the song is triplet/32nd
#                         based and a 16th grid mangles it (song 326 is
#                         0.47). Exclude those, or rerun them at SUB=24.
#   frac_downbeats_on_bar share of annotated downbeats landing on a
#                         4-beat bar line. POP909's own bars are not all
#                         4 beats (column 3 is exactly every fourth beat
#                         in only 126 of 909 songs), so this is well
#                         under 1 for most songs and NO alignment can
#                         fix it -- it is a meter fact, not a bug.
#
# Downstream: stage E1 prompts from $DST/melody and $DST/chord
# (audit_wholesong_split.sbatch MEL_SRC= CHORD_SRC= STAGE_DIR=).

#SBATCH --job-name=align_pop909_grid
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

POP909_RAW="${POP909_RAW:-/home/xinyue.li/POP909-Dataset/POP909-git/POP909}"
DST="${DST:-/home/xinyue.li/POP909-Dataset/POP909-v4}"
IDS="${IDS:-}"
SUB="${SUB:-4}"
TEMPO="${TEMPO:-120}"
ORIGIN="${ORIGIN:-beat0}"
NO_FIX_DROPPED="${NO_FIX_DROPPED:-0}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "POP909_RAW=$POP909_RAW"
echo "DST=$DST  SUB=$SUB/beat  TEMPO=$TEMPO  ORIGIN=$ORIGIN"
echo "NO_FIX_DROPPED=$NO_FIX_DROPPED  IDS=${IDS:-<all>}"
echo "================================================================"

python pop909_align_grid.py \
    --raw "$POP909_RAW" --dst "$DST" \
    --sub "$SUB" --tempo "$TEMPO" --origin "$ORIGIN" \
    ${IDS:+--ids $IDS} \
    $([[ "$NO_FIX_DROPPED" == "1" ]] && echo --no-fix-dropped)

echo "================================================================"
echo "Report: $DST/align_grid_report.tsv"
echo "Verify the grid in TICK space with check_tokenizer_grid.py --"
echo "check_beat_alignment.py measures SECONDS through the tempo map and"
echo "is the wrong instrument here."
echo "================================================================"
