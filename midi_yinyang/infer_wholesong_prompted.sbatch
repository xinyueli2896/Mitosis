#!/bin/bash
# SLURM (GPU): T3' -- whole-song-gen PROMPTED lead-sheet continuation
# (block (i) of the E1 external comparison; EXPERIMENTS.md 2.6).
#
# Drives wholesong_prompted.py inside the whole_song_gen venv: per test
# song, 6-bar ground-truth prompts at the counterpoint and lead-sheet
# levels through their own create_canvas(prompt=...) inpainting path,
# their generation loop unchanged. Output: duet_multi layout
# <OUT_DIR>/<songid>/co/sample_<i>.mid with MELODY/CHORD tracks
# (programs 0/48) -- directly scoreable by the eval harness alongside
# our co decodes (same 6-bar prompt, frames 96..416 scored).
#
# Env knobs (--export=ALL,KEY=val):
#   PROMPT_BARS default 6        N_SAMPLES default 3
#   CROPS_TSV   prompt_crops.tsv from build_prompt_crops. Each song's
#               representation is cropped to that song's crop_bar BEFORE
#               prompting, so the baseline's bar 1 is OUR bar 1. Without
#               it the baseline is prompted from bar 0 -- usually the
#               intro -- while our systems see the melody entry, and the
#               two are not answering the same question.
#   SONG_IDS    default "1 2 3 4 5"; with CROPS_TSV set and SONG_IDS
#               empty, the ids are taken FROM the TSV (kept rows only)
#   OUT_DIR     default temp/wholesong_prompted
#   WS_FORM     gt (default) | generated -- where the counterpoint
#               stage's key+phrase background comes from.
#                 gt        GROUND TRUTH over the whole song, the
#                           generated region included. Their published
#                           form-conditioned protocol, and an ORACLE IN
#                           THE BASELINE'S FAVOUR: it is the only true
#                           ground-truth leak into the generated span.
#                 generated their OWN form stage (results_default/frm---,
#                           covered by setup_wholesong.sbatch), prompted
#                           with the GT form of the same PROMPT_BARS and
#                           left to predict key and phrase across the
#                           rest. This is the matched comparison against
#                           a prompt-only system.
#               Song LENGTH stays oracle under both: a fixed-length
#               continuation has to know where to stop for the reference
#               to line up. Report which one a table row used.
#
# Example:
#   sbatch midi_yinyang/infer_wholesong_prompted.sbatch

#SBATCH --job-name=infer_wholesong_prompted
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
WS="$MY/external/whole_song_gen"

conda activate "$CONDA_ENV"
mkdir -p /home/xinyue.li/logs

PROMPT_BARS="${PROMPT_BARS:-6}"
N_SAMPLES="${N_SAMPLES:-3}"
CROPS_TSV="${CROPS_TSV:-}"
SONG_IDS="${SONG_IDS:-}"
if [[ -z "$SONG_IDS" ]]; then
    if [[ -n "$CROPS_TSV" && -f "$CROPS_TSV" ]]; then
        # kept rows only: a dropped row has a non-empty `dropped` column
        SONG_IDS=$(awk -F'\t' 'NR>1 && $NF=="" {print $1+0}' "$CROPS_TSV" | tr '\n' ' ')
        echo "[ids] taken from $CROPS_TSV"
    else
        SONG_IDS="1 2 3 4 5"
    fi
fi
OUT_DIR="${OUT_DIR:-$MY/temp/wholesong_prompted}"
WS_FORM="${WS_FORM:-gt}"
case "$WS_FORM" in
    gt|generated) ;;
    *) echo "ERROR: WS_FORM must be gt or generated, got '$WS_FORM'"; exit 1 ;;
esac

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "T3': whole-song PROMPTED continuation"
echo "PROMPT_BARS=$PROMPT_BARS  N_SAMPLES=$N_SAMPLES"
echo "CROPS_TSV=${CROPS_TSV:-<none: prompting from bar 0>}"
echo "SONG_IDS=$(echo $SONG_IDS | wc -w) songs"
echo "WS_FORM=$WS_FORM$([[ "$WS_FORM" == gt ]] && echo '  (ORACLE form over the generated region)')"
echo "OUT_DIR=$OUT_DIR"
echo "================================================================"

[[ -d "$WS/.venv" ]] || { echo "ERROR: run setup_wholesong.sbatch first"; exit 1; }
# shellcheck disable=SC1091
source "$WS/.venv/bin/activate"

# Flatten the doubly-nested zip payloads (idempotent; same block as the
# unconditional job -- a no-op once done).
shopt -s dotglob
for d in pretrained_models results_default data; do
    if [[ -d "$WS/$d/$d" ]]; then
        echo "[layout] flattening $d/$d -> $d"
        mv "$WS/$d/$d"/* "$WS/$d/"
        rmdir "$WS/$d/$d"
    fi
done
shopt -u dotglob

# Dependency preflight: their repo ships no requirements file and its
# inference path imports a long tail at module scope (mir_eval, sklearn,
# networkx, matplotlib, omegaconf/requests via lpips, torchvision).
# wholesong_deps.py probes the real import chain, installs what is
# missing, and verifies torch was not swapped.
cd "$WS"
export PYTHONPATH="$WS${PYTHONPATH:+:$PYTHONPATH}"
python "$MY/wholesong_deps.py"

python "$MY/wholesong_prompted.py" \
    --song-ids $SONG_IDS \
    --prompt-bars "$PROMPT_BARS" \
    --n-samples "$N_SAMPLES" --skip-existing \
    --form "$WS_FORM" \
    ${CROPS_TSV:+--crops-tsv "$CROPS_TSV"} \
    --out-dir "$OUT_DIR"

echo "================================================================"
echo "Done. Outputs: $OUT_DIR/<songid>/co/sample_<i>.mid (duet_multi)."
echo "Score alongside our A3 co decodes with build_eval_manifest"
echo "(--source WS:duet_multi:$OUT_DIR) + eval_metrics, same window."
echo "================================================================"
