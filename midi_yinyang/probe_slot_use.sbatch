#!/bin/bash
# SLURM (CPU): audit the A.8 query-block variant.
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
#   LAG               decoy displacement in frames (default 16 = 1 bar)
#   SEMITONES         transposition for the shift condition (default 1)
#
# Takes seconds. Run after any change to
# cp_transformer_m2c_duet_block_diffusion.py, and before launching the
# first QUERY_BLOCK>1 training run.
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
LAG="${LAG:-16}"
SEMITONES="${SEMITONES:-1}"
ARGS=()
[[ -n "$CKPT_A3" ]] && ARGS+=(--ckpt "A3=$CKPT_A3")
[[ -n "$CKPT_A7" ]] && ARGS+=(--ckpt "A7=$CKPT_A7")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    echo "ERROR: set CKPT_A3 and/or CKPT_A7"; exit 1
fi

python probe_slot_use.py "${ARGS[@]}" --lag "$LAG" --semitones "$SEMITONES"

echo "================================================================"
echo "Read PARTNER GAIN and DECOY SENSITIVITY per model (bottom of\neach block); the closing note explains what each pattern implies."
echo "================================================================"
