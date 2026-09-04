#!/bin/bash
# SLURM: does a trained model use its query slots?
#
# Runs probe_slot_use.py -- does a trained model USE its query slots,
# or has it learned to ignore them? For a fixed query frame it reruns
# the model over identical val batches, varying ONLY the slot contents
# (both masked / own true frame / partner true frame / partner
# displaced / partner transposed) and reads the query logits.
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
#   TASK_A3, TASK_A7  corpus to probe each on. The two reference
#                     checkpoints are NOT corpus-matched (A.3 trained on
#                     pop909+pop1k7, A.7 on pop909), so probe each on
#                     its own and compare the "% of own" column, which
#                     divides out how predictable the corpus is.
#   TASK              corpus for any checkpoint with no TASK_* (melchord)
#   LAG               decoy displacement in frames (default 16 = 1 bar)
#   SEMITONES         transposition for the shift condition (default 1)
#
# Forward passes only -- no decoding, no training. A few minutes.
#
# Example:
#   sbatch midi_yinyang/probe_slot_use.sbatch

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

# tasks.py resolves the melchord dataset paths from MELCHORD_CP; every
# melchord* run trains at max_polyphony 8, so pin the probe to the same
# tokenization rather than inheriting whatever the shell had.
export MELCHORD_CP="${MELCHORD_CP:-8}"
echo "MELCHORD_CP=$MELCHORD_CP  (must match the training runs' cp)"

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
