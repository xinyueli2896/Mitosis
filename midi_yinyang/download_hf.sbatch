#!/bin/bash
# Download files from a Hugging Face model repo (the ones upload_hf.sbatch
# put there) into this checkout. Runs under SLURM on the cluster or as a
# plain `bash` script on a machine without SLURM (the #SBATCH lines are
# comments there); see the two examples at the bottom of this header.
#
# Env knobs (--export=ALL,KEY=val under sbatch, KEY=val bash ... elsewhere):
#   HF_REPO    repo id, e.g. <user>/duet-melchord-dense   (REQUIRED)
#   INCLUDE    space-separated glob patterns of repo paths to fetch;
#              default: everything. Directories were uploaded under
#              their own name, files at the repo root, so
#              INCLUDE="*.ckpt" takes the root files and
#              INCLUDE="heldout_v5/*" one uploaded directory.
#   DEST       local directory the repo tree is written into
#              (default ckpt/). A repo path a/b.mid lands at DEST/a/b.mid.
#   HF_TOKEN   only needed for a private repo; pass it through the
#              environment, never write it into a script.
#
# On the cluster:
#   sbatch --export=ALL,HF_REPO=<user>/duet-melchord-dense,INCLUDE="*.ckpt" \
#       midi_yinyang/download_hf.sbatch
# On a machine without SLURM, from the repo root:
#   HF_REPO=<user>/duet-melchord-dense INCLUDE="*.ckpt" bash midi_yinyang/download_hf.sbatch
#
# The result is printed as a file list with sizes. Re-running is cheap:
# files already present with the right size are not fetched again.

#SBATCH --job-name=download_hf
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=06:00:00

set -euo pipefail

# Repo root: the cluster checkout, else where the script was submitted or
# run from (run it from the repo root on a machine without SLURM).
REPO_DIR="${REPO_DIR:-}"
if [[ -z "$REPO_DIR" ]]; then
    for cand in "${SLURM_SUBMIT_DIR:-}" "$PWD" /home/xinyue.li/Mitosis; do
        [[ -n "$cand" && -d "$cand/midi_yinyang" ]] && { REPO_DIR="$cand"; break; }
    done
fi
[[ -n "$REPO_DIR" && -d "$REPO_DIR/midi_yinyang" ]] || {
    echo "ERROR: no midi_yinyang/ under the submit/current directory; run from the repo root or set REPO_DIR"; exit 1; }
# conda env: the cluster's `mitosis` by default; CONDA_ENV= (empty) skips
# activation on a machine whose python already has the packages.
CONDA_ENV="${CONDA_ENV-mitosis}"
if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    source ~/.bashrc
    conda activate "$CONDA_ENV"
fi
cd "$REPO_DIR/midi_yinyang"
[[ -n "${SLURM_JOB_ID:-}" ]] && mkdir -p /home/xinyue.li/logs

: "${HF_REPO:?set HF_REPO=<user>/<repo>}"
INCLUDE="${INCLUDE:-}"
DEST="${DEST:-ckpt}"
mkdir -p "$DEST"

python -c "import huggingface_hub" 2>/dev/null || pip install -q "huggingface_hub>=0.24"

python - "$HF_REPO" "$DEST" $INCLUDE <<'EOF'
import os, sys
from huggingface_hub import snapshot_download

repo, dest, *patterns = sys.argv[1:]
kw = dict(repo_id=repo, repo_type='model', local_dir=dest,
          token=os.environ.get('HF_TOKEN') or None)
if patterns:
    kw['allow_patterns'] = patterns
print(f'downloading {repo} -> {dest}/  patterns={patterns or "<all>"}')
root = snapshot_download(**kw)
total = 0
for d, _, files in os.walk(root):
    if '.cache' in d.split(os.sep):
        continue
    for f in sorted(files):
        p = os.path.join(d, f)
        if os.path.islink(p) or f.startswith('.'):
            continue
        sz = os.path.getsize(p)
        total += sz
        print(f'{sz / 1e9:8.2f} GB  {os.path.relpath(p, dest)}')
print(f'total {total / 1e9:.2f} GB under {dest}/')
EOF
