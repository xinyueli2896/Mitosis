#!/bin/bash
# SLURM: how far have the cross-stream gates trained? (CPU, no GPU)
#
# Wraps check_gates.py, which reports per gate parameter the bias and
# ||weight||.
#
# READ ||W||, NOT THE BIAS. A gate is sigmoid(W.x + b). At warm-start W
# is EXACTLY zero, so the bias alone fixes the output and sigmoid(bias)
# is the open fraction. Once W trains off zero the gate modulates on its
# input and the bias barely moves -- A.2 at 43k steps generates well
# (survival_b = 1.000) with its biases still at ~-9.8 and |W| ~ 0.99.
# Judging on the bias would call that model's gates shut, which is why
# the verdict keys on |W|.
#
# |W| ~ 0 means no cross-stream information can flow. Otherwise the
# mechanism is live and |W| is only meaningful compared ACROSS runs.
#
# Only gated duet variants are checked by default; the single-stream
# finetunes (S0/S1/S-mel/S-chord) have no gates and are skipped.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPTS       space-separated run dirs and/or .ckpt files. Default:
#               every melchord duet run dir that exists (A.1, B.1, A.2).
#   INIT_BIAS   the GATE_INIT_BIAS the runs started from (default -10.0);
#               used to decide whether a gate has moved off its init.
#
# Examples:
#   sbatch midi_yinyang/check_gates.sbatch
#   sbatch --export=ALL,CKPTS="ckpt/<run dir>",INIT_BIAS=-2.0 \
#       midi_yinyang/check_gates.sbatch

#SBATCH --job-name=check_gates
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
INIT_BIAS="${INIT_BIAS:--10.0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

# Gated duet variants only. Globs are expanded and filtered to what
# exists, so a pattern matching nothing is dropped rather than passed
# through as a literal path.
if [[ -z "${CKPTS:-}" ]]; then
    CANDIDATES=(
        ckpt/m2c_intra_cross_attn_*melchord*
        ckpt/m2c_duet_anticipatory_*melchord*
        ckpt/m2c_duet_block_diffusion_*melchord*
    )
    TARGETS=()
    for c in "${CANDIDATES[@]}"; do
        [[ -e "$c" ]] && TARGETS+=("$c")
    done
else
    read -r -a TARGETS <<< "$CKPTS"
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "ERROR: nothing to check."
    echo "  No melchord duet run dirs matched under $(pwd)/ckpt, and CKPTS"
    echo "  was unset. Note the single-stream finetunes have no gates."
    echo "  Pass targets explicitly:"
    echo "    sbatch --export=ALL,CKPTS=\"ckpt/<run dir>\" \\"
    echo "        midi_yinyang/check_gates.sbatch"
    exit 1
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CROSS-STREAM GATE REPORT   (CPU; no GPU needed)"
echo "INIT_BIAS=$INIT_BIAS   (gates start at W=0, bias=$INIT_BIAS)"
echo "targets (${#TARGETS[@]}):"
printf '  %s\n' "${TARGETS[@]}"
echo "================================================================"

# check_gates.py exits 1 when a checkpoint's gate W never left zero;
# that is a finding, not a crash, so capture it rather than abort.
rc=0
python check_gates.py --init-bias "$INIT_BIAS" "${TARGETS[@]}" || rc=$?

echo "================================================================"
if [[ "$rc" -eq 0 ]]; then
    echo "Every checkpoint has gate weights trained off their zero init,"
    echo "so none of them is a case of cross-stream conditioning never"
    echo "switching on. Read mean |W| COMPARATIVELY across runs; the bias"
    echo "staying near its init is normal and says nothing on its own."
else
    echo "UNTRAINED GATES FOUND (exit $rc): some gate W is still at its"
    echo "zero init, so no cross-stream information can flow through it."
    echo "Raise GATE_INIT_BIAS and regenerate the warm-start init:"
    echo ""
    echo "  sbatch --gres=gpu:2 --export=ALL,TASK=melchord_pop909_nottingham,\\"
    echo "BATCH_SIZE=8,GATE_INIT_BIAS=-2.0,VAL_CHECK_INTERVAL=100,RUN_TAG=gb2 \\"
    echo "      midi_yinyang/train_duet_anticipatory.sbatch"
    echo ""
    echo "INIT_CKPT encodes the bias, so a new value regenerates the"
    echo "warm-start instead of reusing the cached one."
fi
echo "================================================================"
exit "$rc"
