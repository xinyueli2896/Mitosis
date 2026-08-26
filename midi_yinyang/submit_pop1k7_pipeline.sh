#!/bin/bash
# Submit the whole Pop1K7 -> combined-corpus -> training chain in one go,
# with SLURM dependencies so each stage starts only when the previous one
# SUCCEEDS. Run this on the login node; then go to sleep.
#
#   1. preprocess_pop1k7_melchord.sbatch   (CPU) tokenize Pop1K7 at cp$CP
#   2. combine_pop909_pop1k7.sbatch        (CPU) merge with POP909
#   3. train_duet_block_diffusion.sbatch   (GPU) A.2 + per-modality gates
#
# afterok semantics: stage N+1 runs only if stage N exits 0. Every
# sbatch here uses `set -euo pipefail`, and stage 2 additionally gates
# on the pairing check, so a bad dataset can never reach training --
# the downstream jobs stay queued as DependencyNeverSatisfied instead
# (cancel them with scancel; see the note this script prints).
#
# Env knobs (all optional; defaults match the A.2 MoE cp4 family):
#   CP              polyphony budget for BOTH sources. Default 4.
#   RUN_TAG         training run tag. Default 'combined'.
#   LR_TOTAL_STEPS  training schedule length. Default 10000.
#   EXCLUDE_GPU     nodes to exclude for the GPU stage.
#                   Default gpu-50,gpu-51.
#   SKIP_PREP=1     skip stages 1-2 (data already built) and submit
#                   only the training job, with no dependency.
#
# Usage:
#   bash midi_yinyang/submit_pop1k7_pipeline.sh
#   LR_TOTAL_STEPS=100000 RUN_TAG=combined_long \
#       bash midi_yinyang/submit_pop1k7_pipeline.sh

set -euo pipefail

CP="${CP:-4}"
RUN_TAG="${RUN_TAG:-combined}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-10000}"
EXCLUDE_GPU="${EXCLUDE_GPU:-gpu-50,gpu-51}"
SKIP_PREP="${SKIP_PREP:-0}"

cd "$(dirname "$0")/.."   # repo root; sbatch paths below are repo-relative

TRAIN_EXPORT="ALL,TASK=melchord_pop909_pop1k7,MAX_POLYPHONY=${CP}"
TRAIN_EXPORT="${TRAIN_EXPORT},TIME_ROPE_ALIGNED=1,MOE_MODALITY_GATES=1"
TRAIN_EXPORT="${TRAIN_EXPORT},RUN_TAG=${RUN_TAG},LR_TOTAL_STEPS=${LR_TOTAL_STEPS}"

echo "================================================================"
echo "Pop1K7 pipeline: cp${CP}  run_tag=${RUN_TAG}  steps=${LR_TOTAL_STEPS}"
echo "================================================================"

if [[ "$SKIP_PREP" == "1" ]]; then
    J3=$(sbatch --parsable --exclude="$EXCLUDE_GPU" \
        --export="$TRAIN_EXPORT" \
        midi_yinyang/train_duet_block_diffusion.sbatch)
    echo "[3/3] train            job $J3  (no dependency, SKIP_PREP=1)"
else
    J1=$(sbatch --parsable --export="ALL,MAX_POLYPHONY=${CP}" \
        midi_yinyang/preprocess_pop1k7_melchord.sbatch)
    echo "[1/3] preprocess Pop1K7 job $J1"

    J2=$(sbatch --parsable --dependency="afterok:${J1}" \
        --export="ALL,CP=${CP}" \
        midi_yinyang/combine_pop909_pop1k7.sbatch)
    echo "[2/3] combine w/ POP909 job $J2  (after $J1 succeeds)"

    J3=$(sbatch --parsable --dependency="afterok:${J2}" \
        --exclude="$EXCLUDE_GPU" --export="$TRAIN_EXPORT" \
        midi_yinyang/train_duet_block_diffusion.sbatch)
    echo "[3/3] train A.2 mg      job $J3  (after $J2 succeeds)"
fi

echo "----------------------------------------------------------------"
echo "Watch:   squeue -u \$USER"
echo "Logs:    ~/logs/mitosis_preproc_pop1k7_*.out"
echo "         ~/logs/mitosis_combine_pop1k7_*.out"
echo "         ~/logs/mitosis_duet_block_diffusion_*.out"
echo ""
echo "If a stage FAILS, the jobs after it show reason"
echo "(DependencyNeverSatisfied) in squeue and never run -- read that"
echo "stage's log, fix, then scancel the stragglers and resubmit."
echo "================================================================"
