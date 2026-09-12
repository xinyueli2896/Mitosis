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
#   MERGED_FOLDER  E1's prompts/merged_tagged -- one file per song with
#                  melody on program 0 and chord on 48. That folder is
#                  exactly the right input: E1 builds it by force-tagging
#                  the chord, so the two streams are separable on the way
#                  back out.
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

MERGED_FOLDER="${MERGED_FOLDER:-$MY/temp/E1/prompts/merged_tagged}"
OUT_DIR="${OUT_DIR:-$MY/temp/amt_prompted}"
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
echo "MERGED_FOLDER=$MERGED_FOLDER ($(ls "$MERGED_FOLDER" 2>/dev/null | wc -l) files)"
echo "OUT_DIR=$OUT_DIR"
echo "PROMPT_FRAMES=$PROMPT_FRAMES  GEN_FRAMES=$GEN_FRAMES  N_SAMPLES=$N_SAMPLES"
echo "MODEL=$MODEL  TOP_P=$TOP_P  RESTRICT=$RESTRICT"
echo "================================================================"

[[ -d "$AMT_DIR/.venv" ]] || {
    echo "ERROR: no venv at $AMT_DIR/.venv -- run setup_amt.sbatch first"
    exit 1; }
[[ -d "$MERGED_FOLDER" ]] || {
    echo "ERROR: missing $MERGED_FOLDER. E1 builds it while staging"
    echo "       prompts; run eval_e1 with PREP_ONLY=1 to make it."
    exit 1; }

# Isolated venv: the package pins its own transformers, and the repo's
# vendored fork must not be on the path in front of it.
# shellcheck disable=SC1091
source "$AMT_DIR/.venv/bin/activate"
cd "$MY"

python "$MY/amt_prompted.py" \
    --merged-folder "$MERGED_FOLDER" \
    --out-dir "$OUT_DIR" \
    --model "$MODEL" \
    --prompt-frames "$PROMPT_FRAMES" --gen-frames "$GEN_FRAMES" \
    --n-samples "$N_SAMPLES" --top-p "$TOP_P" \
    --skip-existing \
    ${IDS:+--ids $IDS} \
    $([[ "$RESTRICT" == "1" ]] || echo --no-restrict-instruments)

echo "================================================================"
echo "Wrote $OUT_DIR/<song>/co/sample_<i>.mid"
echo "Per-sample note counts: $OUT_DIR/amt_notes.tsv"
echo "================================================================"
