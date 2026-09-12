#!/bin/bash
# SLURM (GPU): the Anticipatory Music Transformer as an external SOTA
# baseline, prompted continuation of BOTH streams.
#
# Thickstun et al., arXiv:2306.08620. A general-MIDI autoregressive
# model over arrival-time event triplets, trained on Lakh MIDI. It is
# the only external reference in E1 that generates melody and chord
# TOGETHER rather than harmonising one given the other, which makes it
# the closest published comparison to what this paper does -- and,
# being trained on a different corpus and tokenization, an out-of-domain
# one. Report it as such.
#
# Run setup_amt.sbatch once first: it clones the package into an
# ISOLATED venv (the package pins its own transformers and this repo
# runs a vendored fork) and pre-caches the weights.
#
# Env knobs (--export=ALL,KEY=val):
#   MEL_FOLDER / CHORD_FOLDER
#                  the held-out set (default input/heldout_v5/{melody,
#                  chord}). The two streams are MERGED here, with the
#                  chord force-tagged to program 48, because their
#                  tokenizer carries the instrument in the note token --
#                  the tag is what makes the output splittable back into
#                  two streams. Unlike our own duets, this model never
#                  saw our corpus, so 0/48 is not a train/test mismatch
#                  for it: it is general MIDI either way.
#   MERGED_FOLDER  skip the merge and use this folder directly, e.g.
#                  E1's already-staged temp/E1_*/prompts/merged_tagged.
#   SELFTEST=1     tokenize one song, round-trip it and report the time
#                  base and instrument preservation, WITHOUT loading the
#                  model or touching a GPU. Run this first -- the driver
#                  was written against the package API, and this is what
#                  catches a signature or unit mismatch cheaply. It needs
#                  no GPU, so send it to a CPU node rather than spending
#                  a GPU queue slot on it:
#                    sbatch --gres=none -p cscc-cpu-p -q cscc-cpu-qos \
#                      --export=ALL,SELFTEST=1 \
#                      midi_yinyang/infer_amt_prompted.sbatch
#   MAX_SONGS      stop after N songs, for a smoke run
#   OUT_DIR        absolute; writes <OUT_DIR>/<song>/co/sample_<i>.mid
#   PROMPT_FRAMES  default 80.   GEN_FRAMES default 416 (TOTAL, incl.
#                  the prompt), both matching E1.
#   N_SAMPLES      default 3
#   TOP_P          default 0.98, the package README's own default
#   MODEL          default stanford-crfm/music-medium-800k
#   RESTRICT       1 (default) masks note logits to the prompt's two
#                  instruments, so the output stays two streams. Their
#                  sampler only restricts above 15 instruments, so an
#                  unrestricted run can put notes on a third instrument
#                  that belongs to neither of our streams. 0 runs the
#                  published behaviour and records what lands off-stream.
#                  Say which one a table row used.
#
#   sbatch midi_yinyang/infer_amt_prompted.sbatch

#SBATCH --job-name=infer_amt_prompted
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"
AMT_DIR="${AMT_DIR:-$MY/external/anticipation}"

MEL_FOLDER="${MEL_FOLDER:-$MY/input/heldout_v5/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-$MY/input/heldout_v5/chord}"
MERGED_FOLDER="${MERGED_FOLDER:-}"
OUT_DIR="${OUT_DIR:-$MY/temp/amt_prompted}"
SELFTEST="${SELFTEST:-0}"
MAX_SONGS="${MAX_SONGS:-}"
PROMPT_FRAMES="${PROMPT_FRAMES:-80}"
GEN_FRAMES="${GEN_FRAMES:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
TOP_P="${TOP_P:-0.98}"
MODEL="${MODEL:-stanford-crfm/music-medium-800k}"
RESTRICT="${RESTRICT:-1}"
IDS="${IDS:-}"

conda activate "$CONDA_ENV"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "MEL_FOLDER=$MEL_FOLDER   CHORD_FOLDER=$CHORD_FOLDER"
echo "MERGED_FOLDER=${MERGED_FOLDER:-<built below from the two folders>}"
echo "OUT_DIR=$OUT_DIR   SELFTEST=$SELFTEST   MAX_SONGS=${MAX_SONGS:-<all>}"
echo "PROMPT_FRAMES=$PROMPT_FRAMES  GEN_FRAMES=$GEN_FRAMES  N_SAMPLES=$N_SAMPLES"
echo "MODEL=$MODEL  TOP_P=$TOP_P  RESTRICT=$RESTRICT"
echo "================================================================"

[[ -d "$AMT_DIR/.venv" ]] || {
    echo "ERROR: no venv at $AMT_DIR/.venv -- run setup_amt.sbatch first"
    exit 1; }
# Merge the two streams into one tagged file per song, in the MITOSIS
# env (merge_melody_chord needs mido, and the AMT venv is deliberately
# minimal). Done before the venv switch below.
cd "$MY"
if [[ -z "$MERGED_FOLDER" ]]; then
    MERGED_FOLDER="$OUT_DIR/_merged_tagged"
    for d in "$MEL_FOLDER" "$CHORD_FOLDER"; do
        [[ -d "$d" ]] || { echo "ERROR: missing folder $d"; exit 1; }
    done
    n_mel=$(find "$MEL_FOLDER" -maxdepth 1 -name '*.mid' | wc -l)
    n_chd=$(find "$CHORD_FOLDER" -maxdepth 1 -name '*.mid' | wc -l)
    [[ "$n_mel" -eq "$n_chd" ]] || {
        echo "ERROR: $n_mel melody vs $n_chd chord midis -- the merge pairs"
        echo "       by filename, so an unmatched song would be dropped"
        exit 1; }
    echo "[merge] $n_mel songs -> $MERGED_FOLDER (chord forced to program 48)"
    mkdir -p "$MERGED_FOLDER"
    python merge_melody_chord.py --melody "$MEL_FOLDER" \
        --chord "$CHORD_FOLDER" --dst "$MERGED_FOLDER" --chord-program 48
else
    [[ -d "$MERGED_FOLDER" ]] || {
        echo "ERROR: MERGED_FOLDER=$MERGED_FOLDER does not exist"; exit 1; }
    echo "[merge] using $MERGED_FOLDER as given"
fi

# Isolated venv: the package pins its own transformers, and the repo's
# vendored fork must not be on the path in front of it.
# shellcheck disable=SC1091
source "$AMT_DIR/.venv/bin/activate"

python "$MY/amt_prompted.py" \
    --merged-folder "$MERGED_FOLDER" \
    --out-dir "$OUT_DIR" \
    --model "$MODEL" \
    --prompt-frames "$PROMPT_FRAMES" --gen-frames "$GEN_FRAMES" \
    --n-samples "$N_SAMPLES" --top-p "$TOP_P" \
    --skip-existing \
    ${IDS:+--ids $IDS} \
    ${MAX_SONGS:+--max-songs $MAX_SONGS} \
    $([[ "$SELFTEST" == "1" ]] && echo --selftest) \
    $([[ "$RESTRICT" == "1" ]] || echo --no-restrict-instruments)

echo "================================================================"
echo "Wrote $OUT_DIR/<song>/co/sample_<i>.mid"
echo "Per-sample note counts: $OUT_DIR/amt_notes.tsv"
echo "================================================================"
