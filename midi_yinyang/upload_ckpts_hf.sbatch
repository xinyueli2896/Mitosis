#!/bin/bash
# SLURM (CPU): archive checkpoint families to a PRIVATE Hugging Face
# repo, verify, and optionally free the local space.
#
# Runs upload_ckpts_hf.py: each matched ckpt/ run-directory is uploaded
# to the repo at the same relative path, then verified file-by-file
# (name + size against the remote tree). Local deletion happens ONLY
# with DELETE=1 and ONLY for directories whose verification passed.
#
# ONE-TIME SETUP (login node, before first use):
#   huggingface-cli login        # paste a WRITE token from
#                                # https://huggingface.co/settings/tokens
#
# Env knobs (--export=ALL,KEY=val):
#   HF_REPO    REQUIRED, e.g. myname/mitosis-ckpt-archive
#              (created private automatically if it does not exist)
#   PATTERNS   default: "m2c_duet_anticipatory_* m2c_duet_rehearsal_* m2c_duet_prefix_*"
#   DRY_RUN=1  list matched dirs + sizes, upload nothing
#   DELETE=1   remove each local dir after ITS verification passes
#
# Suggested flow: DRY_RUN=1 first; then upload with defaults (keeps
# local copies); listen-check one restored file if paranoid; then rerun
# with DELETE=1 (already-uploaded dirs verify instantly and delete).
#
# Restore later:
#   huggingface-cli download <HF_REPO> --include "<dir_name>/*" --local-dir ckpt/
#
# Example:
#   sbatch --export=ALL,HF_REPO=myname/mitosis-ckpt-archive,DRY_RUN=1 \
#       midi_yinyang/upload_ckpts_hf.sbatch

#SBATCH --job-name=upload_ckpts_hf
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=24:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

if [[ -z "${HF_REPO:-}" || "$HF_REPO" == YOURNAME/* ]]; then
    echo "ERROR: set HF_REPO to YOUR real namespace, e.g."
    echo "       HF_REPO=\$(huggingface-cli whoami | head -1)/mitosis-ckpt-archive"
    exit 1
fi
PATTERNS="${PATTERNS:-m2c_duet_anticipatory_* m2c_duet_rehearsal_* m2c_duet_prefix_*}"

# Faster multipart uploads when the plugin is present; harmless if not.
python - <<'PY' && export HF_HUB_ENABLE_HF_TRANSFER=1 || true
import hf_transfer  # noqa: F401
PY

EXTRA=()
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA+=(--dry-run)
[[ "${DELETE:-0}" == "1" ]] && EXTRA+=(--delete)

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "ARCHIVE CKPTS -> hf.co/$HF_REPO (private)"
echo "PATTERNS = $PATTERNS"
echo "DRY_RUN=${DRY_RUN:-0}  DELETE=${DELETE:-0}"
echo "================================================================"

# shellcheck disable=SC2086
python upload_ckpts_hf.py --repo "$HF_REPO" --patterns $PATTERNS "${EXTRA[@]}"

echo "================================================================"
echo "Done. Space check:"
du -sh ckpt 2>/dev/null || true
echo "================================================================"
