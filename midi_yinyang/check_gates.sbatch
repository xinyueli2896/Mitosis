#!/bin/bash
# SLURM: did the cross-stream gates ever open? (CPU, no GPU needed)
#
# Wraps check_gates.py, which reads a resolved checkpoint and reports per
# gate parameter: sigmoid(bias) -- the open fraction -- and ||weight||.
#
# Why it matters. The duet variants warm-start from a single-stream CP
# transformer, which has no cross-stream parameters, so those are baked
# in SILENT: gate_m/gate_c weight=0, bias=GATE_INIT_BIAS (default -10,
# sigmoid ~ 4.5e-5). At step 0 each modality just reproduces the
# pretrained block on its own stream, and the entire duet mechanism has
# to be learned through those gates. On a ~1.9k-song corpus val turns
# upward within ~1k steps, so best-val selection can hand back a
# checkpoint whose gates never opened -- a model that is functionally
# two independent single-stream models, with a val_loss that reflects
# the inherited backbone rather than any trained duet behaviour.
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
echo "INIT_BIAS=$INIT_BIAS  -> gates start at sigmoid($INIT_BIAS)"
echo "targets (${#TARGETS[@]}):"
printf '  %s\n' "${TARGETS[@]}"
echo "================================================================"

# check_gates.py exits 1 when a checkpoint's gates never moved; that is
# a finding, not a crash, so capture it rather than let set -e abort.
rc=0
python check_gates.py --init-bias "$INIT_BIAS" "${TARGETS[@]}" || rc=$?

echo "================================================================"
if [[ "$rc" -eq 0 ]]; then
    echo "All checked checkpoints have gates off their initial value."
    echo "Cross-stream conditioning is at least partly trained in each."
else
    echo "UNOPENED GATES FOUND (exit $rc)."
    echo "That checkpoint carries essentially no cross-stream conditioning,"
    echo "so the follower stream has no partner information to work with"
    echo "however good its val_loss looks. On a corpus too small to open the"
    echo "gates before val turns, retrain with the gates started open:"
    echo ""
    echo "  sbatch --gres=gpu:2 --export=ALL,TASK=melchord_pop909_nottingham,\\"
    echo "BATCH_SIZE=8,GATE_INIT_BIAS=-2.0,VAL_CHECK_INTERVAL=100,RUN_TAG=gb2 \\"
    echo "      midi_yinyang/train_duet_anticipatory.sbatch"
    echo ""
    echo "sigmoid(-2) ~ 0.12 rather than 4.5e-5, so cross-conditioning is"
    echo "live from step 0. INIT_CKPT now encodes the bias, so a new value"
    echo "regenerates the warm-start instead of reusing the cached one."
    echo "Judge the result on survival_b from diagnose_b1_follower.sbatch,"
    echo "NOT on val_loss -- more cross-stream capacity can raise val while"
    echo "improving generation."
fi
echo "================================================================"
exit "$rc"
