#!/bin/bash
# SLURM (GPU): Frechet Music Distance for an E1 run.
#
# Adds FMD to the quality sheet as a sixth panel. It behaves like a
# pooled metric -- one value per system, a bootstrap interval over
# songs, reference level 0 -- so the plotter draws it with the same
# point-and-interval mark as the pooled JSDs.
#
# Three stages:
#   1. crop every midi in the E1 manifest to the SCORED WINDOW. FMD
#      embeds whole files, and the prompt is byte-identical between
#      every system and the reference, so leaving it in would be shared
#      signal compressing the very differences the metric measures.
#   2. embed once per file with CLaMP 2 (GPU), cached as .npy
#   3. Frechet distance per system plus a song-level bootstrap, in numpy
#
# READ THE NULL CALIBRATION FIRST. 93 songs x 3 samples is 279 points
# against a 768-dimensional embedding, so the sample covariance is
# singular and the estimate is dominated by covariance-estimation error.
# Stage 3 therefore scores one half of the REFERENCE against the other
# half and prints it: that is what a perfect system gets at this n. If
# the systems' spread is not clearly larger, FMD cannot separate them
# here, and the fix is more samples per song (N_SAMPLES=10 for an
# FMD-only generation pass), not a different metric.
#
# Env knobs (--export=ALL,KEY=val):
#   MANIFEST   default results/E1_p96_v5b93_manifest.tsv
#   EVAL_MEL / EVAL_CHORD   reference dirs; default the E1 tree's
#                           staged prompts under OUT_ROOT
#   OUT_ROOT   default temp/E1_v5b (only used to default the two above)
#   OUT        CSV; default the manifest path with _manifest.tsv -> _fmd.csv
#   TREE       cropped-midi + embedding cache; default temp/fmd_<tag>
#   PROMPT_LENGTH 96   GEN_LENGTH 416   N_BOOT 500
#   REF_WINDOWS  1 (default). Cut this many consecutive windows from
#                each reference song. With 93 held-out songs the
#                reference covariance is 93 points in 768 dimensions --
#                rank 92, badly singular, and the binding constraint on
#                the whole metric. REF_WINDOWS=4 gives up to 372, which
#                is still short of 768 but much better conditioned. The
#                generated side cannot grow this way: systems produced
#                one window each, so more generated points means
#                N_SAMPLES > 3 and a new generation pass.
#   FORCE_PREP=1  rebuild the cropped tree and drop cached embeddings
#
# Example:
#   sbatch --exclude=gpu-51,gpu-05,gpu-54 \
#     --export=ALL,MANIFEST=results/E1_p96_v5b93_manifest.tsv \
#     midi_yinyang/eval_fmd.sbatch

#SBATCH --job-name=eval_fmd
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=06:00:00

set -euo pipefail
source ~/.bashrc
set +o noclobber

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"
VENV="$MY/external/fmd/.venv"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs "$MY/external/fmd"

OUT_ROOT="${OUT_ROOT:-temp/E1_v5b}"
MANIFEST="${MANIFEST:-results/E1_p96_v5b93_manifest.tsv}"
[[ -f "$MANIFEST" ]] || { echo "ERROR: no $MANIFEST -- run score_e1.sbatch first"; exit 1; }
EVAL_MEL="${EVAL_MEL:-$OUT_ROOT/prompts/mel}"
EVAL_CHORD="${EVAL_CHORD:-$OUT_ROOT/prompts/chord}"
OUT="${OUT:-${MANIFEST%_manifest.tsv}_fmd.csv}"
_tag="$(basename "${MANIFEST%_manifest.tsv}")"
TREE="${TREE:-temp/fmd_$_tag}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_BOOT="${N_BOOT:-500}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "MANIFEST=$MANIFEST"
echo "window  = frames $PROMPT_LENGTH..$GEN_LENGTH"
echo "TREE    = $TREE      OUT = $OUT"
echo "================================================================"

# ---- 1. venv (idempotent) ---------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "--- creating the FMD venv (torch + transformers; not the "
    echo "    mitosis env, whose transformers is the vendored roformer) ---"
    python -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet frechet-music-distance
fi
"$VENV/bin/python" -c "import frechet_music_distance" || {
    echo "ERROR: the FMD venv is broken; delete $VENV and resubmit"; exit 1; }

# ---- 2. crop to the scored window -------------------------------------
if [[ "${FORCE_PREP:-0}" == "1" ]]; then
    echo "[prep] FORCE_PREP=1: dropping $TREE"
    rm -rf "$TREE"
fi
python fmd_prepare.py --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --out "$TREE" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --ref-windows "${REF_WINDOWS:-1}"

# ---- 3. embed + score --------------------------------------------------
# CLaMP 2 weights are fetched from HuggingFace on first use; if the
# compute nodes have no outbound network, run this once on a login node
# with the same venv to warm the cache.
"$VENV/bin/python" fmd_score.py --tree "$TREE" --out "$OUT" \
    --n-boot "$N_BOOT"

echo "================================================================"
echo "Done -> $OUT"
echo "The plotter picks this up automatically: the quality sheet looks"
echo "for <csv>_fmd.csv beside the metrics CSV and draws fmd as its"
echo "first panel."
echo "Read the _null_ref_vs_ref row first. It is what a PERFECT system"
echo "scores at this sample size; system scores are only meaningful to"
echo "the extent they exceed it."
echo "================================================================"
