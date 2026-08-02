#!/bin/bash
# SLURM: checkpoint integrity + best-val report, on CPU.
#
# Wraps check_ckpts.py, which torch.loads every checkpoint with
# map_location='cpu' and reports per file: size, whether it actually
# loads, its global_step and tensor count -- then what resolve_best_ckpt
# would SELECT for each directory. No GPU is involved, so this runs on
# the CPU partition rather than occupying a GPU slot to read files.
#
# Two things it answers that matter in practice:
#   * is a checkpoint truncated? A cut-short write surfaces later as
#     "PytorchStreamReader failed reading zip archive: failed finding
#     central directory" partway through an evaluation, long after the
#     job that produced it.
#   * which checkpoint will actually be used? resolve_best_ckpt takes
#     the smallest val_loss in a directory, so a superseded run sharing
#     that directory can silently win. global_step tells you whether the
#     pick sits at a real minimum or at the first validation.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPTS           space-separated run dirs and/or .ckpt files.
#                   Default: every melchord run dir that exists
#                   (A.2, B.1, A.1, and the single-stream finetunes).
#   QUICK=1         structural check only; skip the real torch.load.
#                   Fast, but a file can pass this and still fail to
#                   load, so it is a screen and not a verdict.
#   DELETE_CORRUPT=1  DELETE files that fail to load. Irreversible --
#                   off by default, and it prints what it removed.
#
# Examples:
#   sbatch midi_yinyang/check_ckpts.sbatch
#   sbatch --export=ALL,CKPTS="ckpt/m2c_duet_anticipatory_v1.0_large_gnl12_melchord_pop909_nottingham_vci100_k16_batch_16_schedule" \
#       midi_yinyang/check_ckpts.sbatch

#SBATCH --job-name=check_ckpts
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

QUICK="${QUICK:-0}"
DELETE_CORRUPT="${DELETE_CORRUPT:-0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

# Default target set: every melchord run dir present. Globs are expanded
# and filtered to what exists, so a pattern that matches nothing is
# dropped rather than passed through as a literal path.
if [[ -z "${CKPTS:-}" ]]; then
    CANDIDATES=(
        ckpt/m2c_duet_block_diffusion_*melchord*
        ckpt/m2c_duet_anticipatory_*melchord*
        ckpt/m2c_intra_cross_attn_*melchord*
        ckpt/cp_transformer_v0.42_size1_pop909ft*
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
    echo "  No melchord run dirs matched under $(pwd)/ckpt, and CKPTS was unset."
    echo "  Pass targets explicitly:"
    echo "    sbatch --export=ALL,CKPTS=\"ckpt/<run dir> ckpt/<other>\" \\"
    echo "        midi_yinyang/check_ckpts.sbatch"
    exit 1
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CHECKPOINT INTEGRITY + BEST-VAL REPORT   (CPU; no GPU needed)"
echo "QUICK=$QUICK  DELETE_CORRUPT=$DELETE_CORRUPT"
echo "targets (${#TARGETS[@]}):"
printf '  %s\n' "${TARGETS[@]}"
echo "================================================================"

ARGS=()
[[ "$QUICK" == "1" ]] && ARGS+=(--quick)
if [[ "$DELETE_CORRUPT" == "1" ]]; then
    echo "*** DELETE_CORRUPT=1: unloadable checkpoints WILL BE REMOVED ***"
    ARGS+=(--delete-corrupt)
fi

# check_ckpts.py exits 1 when any checkpoint is corrupt; that is a
# result, not a crash, so capture it rather than letting set -e abort.
rc=0
python check_ckpts.py "${ARGS[@]}" "${TARGETS[@]}" || rc=$?

echo "================================================================"
if [[ "$rc" -eq 0 ]]; then
    echo "All checkpoints loadable."
else
    echo "CORRUPT CHECKPOINT(S) FOUND (exit $rc). Options:"
    echo "  * re-run with --export=ALL,DELETE_CORRUPT=1 to remove them, so"
    echo "    the resolver falls through to the best SURVIVING checkpoint;"
    echo "  * or pin a known-good file explicitly, e.g."
    echo "      CKPT_A2=ckpt/<run dir>/<good file>.ckpt"
    echo "  Usual cause: the write was cut short -- full disk or quota, job"
    echo "  killed mid-save, or the file was copied while being written."
fi
echo "Read the [resolve] lines above: they show which checkpoint each"
echo "directory would actually contribute to an evaluation, and its"
echo "global_step tells you whether that pick sits at a real val minimum"
echo "or merely at the first validation."
echo "================================================================"
exit "$rc"
