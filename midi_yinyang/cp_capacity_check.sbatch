#!/bin/bash
# SLURM: measure the onsets-per-frame distribution and certify the cp
# (max_polyphony) budgets against it. (CPU, no GPU)
#
# Wraps cp_capacity_check.py. Two ways to run it:
#
#   DEFAULT (no env): the full battery for our experiments --
#     1. SURVEY    melchord streams at budget 128 -- reads the raw
#                  histogram tail so the budget can be DESIGNED from
#                  data instead of guessed.
#     2. SURVEY    drumnondrum streams at budget 128, plus the pooled
#                  drums+nondrum view (what the merged baseline sees).
#     3. CERTIFY   melchord melody+chord streams at cp4  (duet side).
#     4. CERTIFY   drumnondrum drum+nondrum streams at cp16 (duet side).
#     5. CERTIFY   POOLED drums+nondrum at cp16 -- the S0 merged-stream
#                  budget, which can clip even when each stream alone
#                  fits. This is the run most likely to fail.
#   The job exits nonzero if any CERTIFY step clips (surveys never fail).
#
#   CUSTOM (FOLDERS set): one single check on your own folders, e.g. the
#   training corpora or the Nottingham held-out set:
#     sbatch --export=ALL,FOLDERS="input/nottingham_split/melody \
#         input/nottingham_split/chord",BUDGET=4 \
#         midi_yinyang/cp_capacity_check.sbatch
#     sbatch --export=ALL,FOLDERS="dir1 dir2",BUDGET=16,POOLED=1 ...
#
# Env knobs (--export=ALL,KEY=val):
#   FOLDERS            space-separated midi folders -> custom single run
#   BUDGET             budget for the custom run (required with FOLDERS)
#   POOLED             1 = sum folders per song before measuring (merged
#                      view); default per-stream
#   MELCHORD_BUDGET    battery melchord budget      (default 4)
#   DRUMNONDRUM_BUDGET battery drumnondrum budget   (default 16)
#   MEL_FOLDER / CHORD_FOLDER / DRUM_FOLDER / NONDRUM_FOLDER
#                      battery folder overrides; defaults match eval_e1:
#                      input/pop909_split/{melody,chord}
#                      input/rwc_test_prompts_split/{drum,nondrum}

#SBATCH --job-name=cp_capacity
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CP CAPACITY CHECK   (CPU; no GPU needed)"
echo "================================================================"

# ---------------- custom single run ----------------
if [[ -n "${FOLDERS:-}" ]]; then
    if [[ -z "${BUDGET:-}" ]]; then
        echo "ERROR: FOLDERS is set but BUDGET is not."
        echo "  sbatch --export=ALL,FOLDERS=\"dir1 dir2\",BUDGET=4 \\"
        echo "      midi_yinyang/cp_capacity_check.sbatch"
        exit 1
    fi
    read -r -a FOLDER_ARR <<< "$FOLDERS"
    POOLED_FLAG=()
    [[ "${POOLED:-0}" == "1" ]] && POOLED_FLAG=(--pooled)
    echo "CUSTOM RUN: budget=cp$BUDGET pooled=${POOLED:-0}"
    printf '  %s\n' "${FOLDER_ARR[@]}"
    python cp_capacity_check.py --budget "$BUDGET" \
        "${POOLED_FLAG[@]}" "${FOLDER_ARR[@]}"
    exit $?
fi

# ---------------- default battery ----------------
MELCHORD_BUDGET="${MELCHORD_BUDGET:-4}"
DRUMNONDRUM_BUDGET="${DRUMNONDRUM_BUDGET:-16}"
MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
DRUM_FOLDER="${DRUM_FOLDER:-input/rwc_test_prompts_split/drum}"
NONDRUM_FOLDER="${NONDRUM_FOLDER:-input/rwc_test_prompts_split/nondrum}"

echo "battery config:"
echo "  melchord    : $MEL_FOLDER  $CHORD_FOLDER  @ cp$MELCHORD_BUDGET"
echo "  drumnondrum : $DRUM_FOLDER  $NONDRUM_FOLDER  @ cp$DRUMNONDRUM_BUDGET"
echo "  pooled      : drums+nondrum @ cp$DRUMNONDRUM_BUDGET (S0 merged view)"

fail=0

echo ""
echo "-------- 1/5 SURVEY melchord streams (histogram only) ----------"
python cp_capacity_check.py --budget 128 \
    "$MEL_FOLDER" "$CHORD_FOLDER" || true

echo ""
echo "-------- 2/5 SURVEY drumnondrum streams + pooled ---------------"
python cp_capacity_check.py --budget 128 \
    "$DRUM_FOLDER" "$NONDRUM_FOLDER" || true
python cp_capacity_check.py --budget 128 --pooled \
    "$DRUM_FOLDER" "$NONDRUM_FOLDER" || true

echo ""
echo "-------- 3/5 CERTIFY melchord streams @ cp$MELCHORD_BUDGET ------"
python cp_capacity_check.py --budget "$MELCHORD_BUDGET" \
    "$MEL_FOLDER" "$CHORD_FOLDER" || fail=1

echo ""
echo "-------- 4/5 CERTIFY drumnondrum streams @ cp$DRUMNONDRUM_BUDGET -"
python cp_capacity_check.py --budget "$DRUMNONDRUM_BUDGET" \
    "$DRUM_FOLDER" "$NONDRUM_FOLDER" || fail=1

echo ""
echo "-------- 5/5 CERTIFY POOLED drums+nondrum @ cp$DRUMNONDRUM_BUDGET"
python cp_capacity_check.py --budget "$DRUMNONDRUM_BUDGET" --pooled \
    "$DRUM_FOLDER" "$NONDRUM_FOLDER" || fail=1

echo ""
echo "================================================================"
if [[ "$fail" -eq 0 ]]; then
    echo "ALL CERTIFICATIONS PASS: no budget clips its corpus. The cp"
    echo "difference across representations is lossless on this data --"
    echo "cite the survey histograms (steps 1-2) as the design evidence."
else
    echo "CERTIFICATION FAILED: at least one budget clips (see VERDICT"
    echo "lines above -- the pooled S0 view in step 5 is the usual"
    echo "culprit). Read the survey histogram to pick a budget covering"
    echo "the tail, then either re-tokenize at that cp or disclose the"
    echo "measured drop rate in the comparison."
fi
echo "================================================================"
exit "$fail"
