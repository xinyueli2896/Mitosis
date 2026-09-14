#!/bin/bash
# SLURM (GPU): regenerate ONLY the songs a cascade arm (P-mc / P-cm) is
# missing, and splice them back into the scored tree.
#
# WHY THIS EXISTS. pipeline_cogen.sbatch is folder-driven end to end:
# every stage globs its input directory, and only stage A has
# --skip-existing. Rerunning an arm to recover three songs therefore
# redoes stages B, C and D for all 95 -- stage C is the YinYang decode,
# the expensive one -- and E1's is_complete cannot ask for less, since
# it is all-or-nothing per system.
#
# So restrict the INPUT instead. This wrapper finds the songs whose
# 4_final/<sid>/co/ holds fewer than N_SAMPLES samples, stages just
# those prompts into a subset folder of symlinks, runs the ordinary
# pipeline over that subset in a scratch root, and copies the finished
# songs back. The cascade code is untouched: the arm it produces is the
# same arm, run over fewer songs.
#
# Each recovered song is regenerated WHOLE. A song with 1 of 3 samples
# has its 4_final/<sid> removed before the new one lands, so a partial
# song cannot leave a stale sample_2 from an older run sitting beside
# two fresh ones.
#
# Env knobs (--export=ALL,KEY=val):
#   E1_ROOT     required. The E1 output root, e.g. temp/E1_v5b. Prompts
#               are read from $E1_ROOT/prompts/{mel,chord} -- the SAME
#               staged crops the original run used, so a refilled song
#               is prompted identically to its neighbours -- and the
#               arm is $E1_ROOT/$PIPELINE.
#   PIPELINE    P-mc (default) | P-cm
#   N_SAMPLES   3       PROMPT_LENGTH 96      GEN_LENGTH 416
#   MAX_POLYPHONY 4     (the specialists are cp4; see eval_e1.sbatch)
#   CKPT_SPEC / CKPT_Y  default to E1's for the chosen direction
#   DRY_RUN=1   list the missing songs and exit without generating
#   KEEP_WORK=1 (default) leave the scratch root for inspection
#
# Example:
#   sbatch --exclude=gpu-51,gpu-05,gpu-54 \
#     --export=ALL,E1_ROOT=temp/E1_v5b,PIPELINE=P-mc \
#     midi_yinyang/fill_pipeline_gaps.sbatch

#SBATCH --job-name=fill_pipeline_gaps
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
set +o noclobber

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs

E1_ROOT="${E1_ROOT:-}"
[[ -n "$E1_ROOT" ]] || { echo "ERROR: E1_ROOT is required (e.g. temp/E1_v5b)"; exit 1; }
PIPELINE="${PIPELINE:-P-mc}"
case "$PIPELINE" in P-mc|P-cm) ;; *) echo "ERROR: PIPELINE must be P-mc or P-cm"; exit 1 ;; esac

N_SAMPLES="${N_SAMPLES:-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
MAX_POLYPHONY="${MAX_POLYPHONY:-4}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_WORK="${KEEP_WORK:-1}"

# Same defaults as eval_e1.sbatch: a refilled song must come off the
# same two checkpoints as the 92 it joins, or the arm is a mixture.
if [[ "$PIPELINE" == "P-mc" ]]; then
    CKPT_SPEC="${CKPT_SPEC:-${CKPT_SMEL:-ckpt/cp_transformer_v0.42_size1_pop909ft_mel_cp4_batch_24_schedule}}"
    CKPT_Y="${CKPT_Y:-${CKPT_Y_M2C:-ckpt/cp_transformer_yinyang_v5.1_lora_batch_8_pop909_cp8_v2_chord_mel_rev_mask0.25-10-step1}}"
else
    CKPT_SPEC="${CKPT_SPEC:-${CKPT_SCHORD:-ckpt/cp_transformer_v0.42_size1_pop909ft_chord_cp4_batch_24_schedule}}"
    CKPT_Y="${CKPT_Y:-${CKPT_Y_C2M:-ckpt/cp_transformer_yinyang_v5.1_lora_batch_8_pop909_cp8_v2_chord_mel_mask0.25-10-step1}}"
fi

EVAL_MEL="$E1_ROOT/prompts/mel"
EVAL_CHORD="$E1_ROOT/prompts/chord"
TARGET="$E1_ROOT/$PIPELINE/4_final"

for d in "$EVAL_MEL" "$EVAL_CHORD"; do
    [[ -d "$d" ]] || { echo "ERROR: no $d -- is E1_ROOT right?"; exit 1; }
done
# OUT_ROOT must start with temp/: stage A builds temp/<save_name>/ and
# can write nowhere else (pipeline_cogen.sbatch enforces it too).
[[ "$E1_ROOT" == temp/* ]] || { echo "ERROR: E1_ROOT must start with temp/"; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "FILL GAPS  $PIPELINE  under $E1_ROOT"
echo "prompts : $EVAL_MEL + $EVAL_CHORD"
echo "target  : $TARGET  (need $N_SAMPLES samples/song)"
echo "spec    : $CKPT_SPEC"
echo "cond    : $CKPT_Y"
echo "================================================================"

# ---- 1. which songs are short -----------------------------------------
# nullglob arrays, not `ls | wc -l`: under `set -euo pipefail` an ls on a
# directory that does not exist yet -- exactly the case being reported --
# aborts the script instead of reporting zero.
shopt -s nullglob
MISSING=()
TOTAL=0
for f in "$EVAL_MEL"/*.mid "$EVAL_MEL"/*.MID; do
    sid=$(basename "$f"); sid="${sid%.*}"
    TOTAL=$((TOTAL + 1))
    have=("$TARGET/$sid/co/"sample_*.mid)
    if [[ ${#have[@]} -lt $N_SAMPLES ]]; then
        MISSING+=("$sid")
        echo "  [short] $sid: ${#have[@]}/$N_SAMPLES"
    fi
done
shopt -u nullglob

echo "[scan] $TOTAL songs, ${#MISSING[@]} short: ${MISSING[*]:-<none>}"
if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "$PIPELINE is already complete. Nothing to do."
    exit 0
fi
if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1: stopping before generation."
    exit 0
fi

# ---- 2. subset prompts ------------------------------------------------
WORK="$E1_ROOT/$PIPELINE/_fill_${SLURM_JOB_ID:-local}"
SUB="$WORK/prompts"
rm -rf "$WORK"
mkdir -p "$SUB/mel" "$SUB/chord"
for sid in "${MISSING[@]}"; do
    for pair in "mel:$EVAL_MEL" "chord:$EVAL_CHORD"; do
        role="${pair%%:*}"; src="${pair#*:}"
        if   [[ -f "$src/$sid.mid" ]]; then ln -s "$(readlink -f "$src/$sid.mid")" "$SUB/$role/$sid.mid"
        elif [[ -f "$src/$sid.MID" ]]; then ln -s "$(readlink -f "$src/$sid.MID")" "$SUB/$role/$sid.mid"
        else echo "ERROR: no $role prompt for $sid under $src"; exit 1; fi
    done
done
echo "[subset] staged ${#MISSING[@]} songs -> $SUB"

# ---- 3. the ordinary cascade, over the subset -------------------------
# Run inside THIS allocation (its #SBATCH lines are ignored when invoked
# as a script), so the arm produced here comes off the same code path as
# the E1 arm it is patching.
PIPELINE="$PIPELINE" CKPT_SPEC="$CKPT_SPEC" CKPT_Y="$CKPT_Y" STAGE_B=y \
OUT_ROOT="$WORK" MEL_FOLDER="$SUB/mel" CHORD_FOLDER="$SUB/chord" \
PROMPT_LENGTH="$PROMPT_LENGTH" GEN_LENGTH="$GEN_LENGTH" \
N_SAMPLES="$N_SAMPLES" MAX_POLYPHONY="$MAX_POLYPHONY" \
    bash "$MY/pipeline_cogen.sbatch"

# ---- 4. splice back ---------------------------------------------------
echo "--- merging into $TARGET ---"
mkdir -p "$TARGET"
shopt -s nullglob
FILLED=(); STILL=()
for sid in "${MISSING[@]}"; do
    new=("$WORK/4_final/$sid/co/"sample_*.mid)
    if [[ ${#new[@]} -lt $N_SAMPLES ]]; then
        echo "  [FAIL] $sid: the refill produced ${#new[@]}/$N_SAMPLES"
        STILL+=("$sid")
        continue
    fi
    # whole-song replacement: a partial song's old samples go, so the
    # scored folder never mixes two runs of the same song.
    rm -rf "${TARGET:?}/$sid"
    cp -r "$WORK/4_final/$sid" "$TARGET/$sid"
    got=("$TARGET/$sid/co/"sample_*.mid)
    echo "  [ok]   $sid: ${#got[@]} samples"
    FILLED+=("$sid")
done
shopt -u nullglob

if [[ "$KEEP_WORK" != "1" ]]; then
    rm -rf "$WORK"
    echo "[work] removed $WORK"
else
    echo "[work] kept $WORK (KEEP_WORK=0 to delete)"
fi

echo "================================================================"
echo "filled ${#FILLED[@]}: ${FILLED[*]:-<none>}"
if [[ ${#STILL[@]} -gt 0 ]]; then
    echo "STILL SHORT ${#STILL[@]}: ${STILL[*]}"
    echo "These failed inside the cascade, not in the bookkeeping -- read"
    echo "the stage A/C output above for this job before rerunning."
    exit 1
fi
echo "$PIPELINE complete. Rerun eval_e1.sbatch to score (is_complete will"
echo "now pass for this arm, so generation is skipped)."
echo "================================================================"
