#!/bin/bash
# SLURM: does a trained model use its query slots?
#
# Runs probe_slot_use.py -- does a trained model USE its query slots,
# or has it learned to ignore them? For a fixed query frame it reruns
# the model over identical val batches, varying ONLY which slot is
# filled and with what (both masked / true frame / displaced frame /
# transposed frame) and reads the query logits. Each condition fills
# ONE slot and is read by both streams: filling the melody slot is the
# melody's OWN channel and the chord's PARTNER channel at once.
#
# PARTNER GAIN = CE(both masked) - CE(partner true) is the information
# the same-instant channel actually carries; DECOY SENSITIVITY is how
# far a wrong-but-legal partner moves the prediction. A model that
# learned to discard untrusted slot content shows gain ~ 0 and
# sensitivity ~ 0 -- which would mean parametric corruptions (A.7's
# displacement, and pitch shift equally) are invertible shortcuts, and
# would settle whether to try another one.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_A3, CKPT_A7  run dirs (best-val auto-resolved); both optional,
#                     at least one required
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
TASK="${TASK:-melchord}"
TASK_A3="${TASK_A3:-}"
TASK_A7="${TASK_A7:-}"
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
if [[ ${#ARGS[@]} -eq 0 ]]; then
    echo "ERROR: set CKPT_A3 and/or CKPT_A7"; exit 1
fi

python probe_slot_use.py "${ARGS[@]}" --task "$TASK" \
    --lag "$LAG" --semitones "$SEMITONES"

echo "================================================================"
printf "Read OWN GAIN, PARTNER GAIN and DECOY per stream (bottom of each\nblock); the closing note explains what each pattern implies.\n"
echo "================================================================"
