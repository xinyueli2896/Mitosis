#!/bin/bash
# SLURM: does a trained model use its query slots?
#
# Runs probe_slot_use.py -- does a trained model USE its query slots,
# or has it learned to ignore them? Over identical val batches it
# reruns the model on several query frames, varying ONLY which slot is
# filled and with what (both masked / true frame / displaced frame /
# transposed frame) and reads the query logits. Each condition fills
# ONE slot and is read by both streams: filling the melody slot is the
# melody's OWN channel and the chord's PARTNER channel at once.
#
# The headline is CONTENT = CE(partner decoy) - CE(partner true): the
# true partner against a WRONG one that is equally legal and equally
# present, so only the partner's identity varies. That is the only
# component that can carry coordination -- the rest of PARTNER GAIN is
# the model merely preferring a real frame over the mask token. A model
# that learned to discard untrusted slot content shows CONTENT ~ 0 and
# decoy sensitivity ~ 0, which would mean parametric corruptions (A.7's
# displacement, and pitch shift equally) are invertible shortcuts.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_A3, CKPT_A7, CKPT_A8
#                     run dirs (best-val auto-resolved); all optional,
#                     at least one required. A query_block>1 checkpoint
#                     (A.8) additionally gets the block-regime section:
#                     LEAK (can the offset-0 slot read true future frames
#                     from its neighbours?) and REGIME (does the lone-slot
#                     val_loss score a regime training never visits?).
#   CKPTS             extra NAME=PATH entries, space-separated, for
#                     anything beyond the three fixed slots -- e.g.
#                     CKPTS="A8last=ckpt/<run>/last.ckpt" to probe a
#                     run's FINAL weights next to its best-val pick
#                     (a path ending in last.ckpt is honoured as-is
#                     here, unlike everywhere else in the repo).
#   TASK_A3, TASK_A7  corpus to probe each on, when the checkpoints are
#                     NOT corpus-matched. Both reference checkpoints are
#                     plain melchord (POP909), so leave these unset --
#                     the combined POP909+Pop1K7 files in the A.3 run
#                     directory's name are a RUN_TAG, not its task, and
#                     only exist at cp4 anyway.
#   TASK              corpus for any checkpoint with no TASK_* (melchord)
#   MAX_POLYPHONY     REQUIRED, no default. cp of the datasets to probe;
#                     must match what the checkpoints trained on. The
#                     A.1/A.3-A.8 query-ablation family is cp4. A
#                     mismatch loads cleanly and silently degrades every
#                     number here, so this wrapper refuses to guess.
#                     Confirm from each run's training log:
#                       grep 'Data for dataset' ~/logs/<train job>.out
#   N_FRAMES          query frames scored per batch (default 8), evenly
#                     spaced over the middle half and averaged within
#                     the batch. The batch stays the independent unit;
#                     this only cuts the noise inside it.
#   LAG               decoy displacement in frames (default 16 = 1 bar)
#   SEMITONES         transposition for the shift condition (default 1)
#
# Forward passes only -- no decoding, no training. A few minutes.
#
# Example:
#   sbatch --export=ALL,MAX_POLYPHONY=4,CKPT_A3=ckpt/<run>,CKPT_A7=ckpt/<run> \\
#       midi_yinyang/probe_slot_use.sbatch

#SBATCH --job-name=probe_slot_use
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "slot-use probe"
echo "================================================================"

CKPT_A3="${CKPT_A3:-}"
CKPT_A7="${CKPT_A7:-}"
CKPT_A8="${CKPT_A8:-}"
CKPTS="${CKPTS:-}"
TASK="${TASK:-melchord}"
TASK_A3="${TASK_A3:-}"
TASK_A7="${TASK_A7:-}"
TASK_A8="${TASK_A8:-}"
N_FRAMES="${N_FRAMES:-8}"
LAG="${LAG:-16}"
SEMITONES="${SEMITONES:-1}"

# tasks.py resolves the melchord dataset paths from MELCHORD_CP, the
# same way train_duet_block_diffusion.sbatch does. 8 is the melchord*
# default there, but the query-ablation duet family is cp4 -- so there
# is no default that is right for everyone, and the echo below is the
# only place a mismatch becomes visible at all.
if [[ -z "${MAX_POLYPHONY:-}" ]]; then
    echo "ERROR: set MAX_POLYPHONY explicitly (the query-ablation duet"
    echo "family is cp4; train_duet_block_diffusion.sbatch defaults"
    echo "melchord* to 8). There is no safe default: every number this"
    echo "probe prints is a cross-entropy, and the wrong cp loads"
    echo "cleanly and degrades all of them without erroring."
    exit 1
fi
export MELCHORD_CP="$MAX_POLYPHONY"
echo "MAX_POLYPHONY=$MAX_POLYPHONY -> data/pop909_{melody,chord}_cp${MELCHORD_CP}_v2.pt"
echo "Confirm against each run's log: grep 'Data for dataset' ~/logs/<job>.out"

ARGS=()
[[ -n "$CKPT_A3" ]] && ARGS+=(--ckpt "A3=$CKPT_A3${TASK_A3:+=$TASK_A3}")
[[ -n "$CKPT_A7" ]] && ARGS+=(--ckpt "A7=$CKPT_A7${TASK_A7:+=$TASK_A7}")
[[ -n "$CKPT_A8" ]] && ARGS+=(--ckpt "A8=$CKPT_A8${TASK_A8:+=$TASK_A8}")
for spec in $CKPTS; do ARGS+=(--ckpt "$spec"); done
if [[ ${#ARGS[@]} -eq 0 ]]; then
    echo "ERROR: set at least one of CKPT_A3 / CKPT_A7 / CKPT_A8 / CKPTS"; exit 1
fi

python probe_slot_use.py "${ARGS[@]}" --task "$TASK" \
    --n_frames "$N_FRAMES" --lag "$LAG" --semitones "$SEMITONES"

echo "================================================================"
printf "Read OWN GAIN, PARTNER GAIN and DECOY per stream (bottom of each\nblock); the closing note explains what each pattern implies.\n"
echo "================================================================"
