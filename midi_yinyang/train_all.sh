#!/bin/bash
# Submit training jobs for ALL active model variants in one go.
#
# This is a SUBMITTER (run it on the login node, NOT via sbatch): it
# fires one sbatch per variant and exits. Each variant trains as its
# own independent SLURM job on its own GPUs.
#
#   A.1  M2CIntraCrossAttn      train_intra_cross_attn.sbatch
#   A.3  M2CDuetBlockDiffusion  train_duet_block_diffusion.sbatch
#   B.1  M2CDuetAnticipatory    train_duet_anticipatory.sbatch
#   C.1  M2CDuetRehearsal       train_duet_rehearsal.sbatch
#   C.2  M2CDuetPrefix          train_duet_prefix.sbatch
#
# A.2 M2CDuetBlockAttn is superseded by A.3 and NOT submitted by default;
# pass WITH_A2=1 to include it.
#
# Knobs (env vars):
#   TASK=melchord          switch dataset task (default drumnondrum).
#   GPUS_PER_JOB=2         request N gpus per job (default: whatever each
#                          sbatch header says, typically 4). Useful when
#                          the partition is congested.
#   SKIP_BASE=1            skip A.1
#   SKIP_DIFFUSION=1       skip A.3
#   SKIP_ANTICIPATORY=1    skip B.1
#   SKIP_REHEARSAL=1       skip C.1
#   SKIP_PREFIX=1          skip C.2
#   WITH_A2=1              also submit A.2 (train_duet_block.sbatch)
#   DRY_RUN=1              print the sbatch commands without submitting.
#
# Variants whose sbatch supports auto-resume (A.3, B.1) will continue
# from their last.ckpt automatically; the others resume only if you pass
# CKPT explicitly to their individual sbatch (not through this script).
#
# Examples:
#   bash midi_yinyang/train_all.sh
#   SKIP_BASE=1 SKIP_REHEARSAL=1 bash midi_yinyang/train_all.sh
#   GPUS_PER_JOB=2 DRY_RUN=1 bash midi_yinyang/train_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="${TASK:-drumnondrum}"
GPUS_PER_JOB="${GPUS_PER_JOB:-}"
DRY_RUN="${DRY_RUN:-0}"
WITH_A2="${WITH_A2:-0}"

SKIP_BASE="${SKIP_BASE:-0}"
SKIP_DIFFUSION="${SKIP_DIFFUSION:-0}"
SKIP_ANTICIPATORY="${SKIP_ANTICIPATORY:-0}"
SKIP_REHEARSAL="${SKIP_REHEARSAL:-0}"
SKIP_PREFIX="${SKIP_PREFIX:-0}"

SBATCH_ARGS=()
if [[ -n "$GPUS_PER_JOB" ]]; then
    SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_JOB}")
fi

submit() {
    local label="$1"
    local script="$2"

    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        echo "[$label] ERROR: $script not found; skipping."
        return
    fi
    local cmd=(sbatch "${SBATCH_ARGS[@]}" --export="ALL,TASK=$TASK" "$SCRIPT_DIR/$script")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$label] DRY RUN: ${cmd[*]}"
        return
    fi
    local out
    out=$("${cmd[@]}")
    echo "[$label] $out"
}

echo "================================================================"
echo "Submitting training for all active variants  (task: $TASK)"
if [[ -n "$GPUS_PER_JOB" ]]; then echo "GPUs per job override: $GPUS_PER_JOB"; fi
if [[ "$DRY_RUN" == "1" ]]; then echo "DRY RUN -- nothing will be submitted"; fi
echo "================================================================"

if [[ "$SKIP_BASE" != "1" ]]; then
    submit "A.1 intra_cross_attn " "train_intra_cross_attn.sbatch"
else
    echo "[A.1 intra_cross_attn ] skipped (SKIP_BASE=1)"
fi

if [[ "$WITH_A2" == "1" ]]; then
    submit "A.2 duet_block       " "train_duet_block.sbatch"
fi

if [[ "$SKIP_DIFFUSION" != "1" ]]; then
    submit "A.3 duet_block_diff  " "train_duet_block_diffusion.sbatch"
else
    echo "[A.3 duet_block_diff  ] skipped (SKIP_DIFFUSION=1)"
fi

if [[ "$SKIP_ANTICIPATORY" != "1" ]]; then
    submit "B.1 duet_anticipatory" "train_duet_anticipatory.sbatch"
else
    echo "[B.1 duet_anticipatory] skipped (SKIP_ANTICIPATORY=1)"
fi

if [[ "$SKIP_REHEARSAL" != "1" ]]; then
    submit "C.1 duet_rehearsal   " "train_duet_rehearsal.sbatch"
else
    echo "[C.1 duet_rehearsal   ] skipped (SKIP_REHEARSAL=1)"
fi

if [[ "$SKIP_PREFIX" != "1" ]]; then
    submit "C.2 duet_prefix      " "train_duet_prefix.sbatch"
else
    echo "[C.2 duet_prefix      ] skipped (SKIP_PREFIX=1)"
fi

echo "================================================================"
echo "Done. Check queue with:  squeue -u \$USER"
