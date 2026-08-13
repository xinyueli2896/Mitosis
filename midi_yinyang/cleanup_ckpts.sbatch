#!/bin/bash
# SLURM (CPU): reclaim disk from ckpt/ run dirs -- keep only the
# best-val checkpoint per run. Wraps cleanup_ckpts.py.
#
# SAFE BY DEFAULT: DELETE=0 is a dry run that only prints what would be
# removed and the bytes reclaimable. Read that log FIRST, then resubmit
# with DELETE=1.
#
# Env knobs (--export=ALL,KEY=val):
#   DELETE      1 = actually delete. Default 0 (dry run).
#   PURGE_LAST  1 = also delete last.ckpt. Do NOT use while any run may
#               still resume (e.g. a training in progress) unless it is
#               protected by EXCLUDE.
#   PURGE_FIN   1 = also delete *.fin.ckpt files.
#   EXCLUDE     space-separated run-dir NAME globs to skip entirely,
#               e.g. "*cp4tar* *pop909scratch*" for active runs.
#   ROOT        default ckpt
#
# Examples:
#   sbatch midi_yinyang/cleanup_ckpts.sbatch                      # dry run
#   sbatch --export=ALL,DELETE=1 midi_yinyang/cleanup_ckpts.sbatch
#   sbatch --export=ALL,DELETE=1,PURGE_LAST=1,EXCLUDE="*cp4tar*" \
#       midi_yinyang/cleanup_ckpts.sbatch

#SBATCH --job-name=cleanup_ckpts
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
DELETE="${DELETE:-0}"
PURGE_LAST="${PURGE_LAST:-0}"
PURGE_FIN="${PURGE_FIN:-0}"
EXCLUDE="${EXCLUDE:-}"
ROOT="${ROOT:-ckpt}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

ARGS=(--root "$ROOT")
[[ "$DELETE" == "1" ]] && ARGS+=(--delete)
[[ "$PURGE_LAST" == "1" ]] && ARGS+=(--purge-last)
[[ "$PURGE_FIN" == "1" ]] && ARGS+=(--purge-fin)
for g in $EXCLUDE; do
    ARGS+=(--exclude "$g")
done

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CKPT CLEANUP  root=$ROOT  DELETE=$DELETE  PURGE_LAST=$PURGE_LAST"
echo "PURGE_FIN=$PURGE_FIN  EXCLUDE=${EXCLUDE:-<none>}"
[[ "$DELETE" != "1" ]] && echo "*** DRY RUN: nothing will be removed ***"
echo "================================================================"
df -h "$REPO_DIR" | tail -1

python cleanup_ckpts.py "${ARGS[@]}"

echo "----------------------------------------------------------------"
df -h "$REPO_DIR" | tail -1
