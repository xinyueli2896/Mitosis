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
#   DEST       local directory the files are written into (default
#              ckpt/). A repo path a/b.mid lands at DEST/a/b.mid.
#   HF_TOKEN   only needed for a private repo; pass it through the
#              environment, never write it into a script.
#   CONDA_ENV  default mitosis; skipped when that env is already active
#              or when set empty (CONDA_ENV=)
#
# On the cluster:
#   sbatch --export=ALL,HF_REPO=<user>/duet-melchord-dense,INCLUDE="*.ckpt" \
#       midi_yinyang/download_hf.sbatch
# On a machine without SLURM, from the repo root:
#   HF_REPO=<user>/duet-melchord-dense INCLUDE="*.ckpt" bash midi_yinyang/download_hf.sbatch
#
# Only the requested files land under DEST, at their repo paths, with no
# cache or metadata directories beside them. A file already present at
# its full size is skipped, so a rerun resumes.

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

# Repo root: the submit/current directory (run it from the repo root on
# a machine without SLURM), else the cluster checkout.
REPO_DIR="${REPO_DIR:-}"
if [[ -z "$REPO_DIR" ]]; then
    for cand in "${SLURM_SUBMIT_DIR:-}" "$PWD" /home/xinyue.li/Mitosis; do
        [[ -n "$cand" && -d "$cand/midi_yinyang" ]] && { REPO_DIR="$cand"; break; }
    done
fi
[[ -n "$REPO_DIR" && -d "$REPO_DIR/midi_yinyang" ]] || {
    echo "ERROR: no midi_yinyang/ under the submit/current directory; run from the repo root or set REPO_DIR"; exit 1; }
# conda env (default mitosis): skipped when CONDA_ENV is empty or that env
# is already the active one (a machine whose ~/.bashrc has no conda hook,
# run from an activated shell); otherwise activated through the hook of
# the conda on PATH, not ~/.bashrc, so 'conda init' is not required.
CONDA_ENV="${CONDA_ENV-mitosis}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH; activate the env yourself and pass CONDA_ENV="; exit 1; }
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
cd "$REPO_DIR/midi_yinyang"
# the cluster log dir (its #SBATCH --output); nothing to do elsewhere
mkdir -p /home/xinyue.li/logs 2>/dev/null || true

: "${HF_REPO:?set HF_REPO=<user>/<repo>}"
INCLUDE="${INCLUDE:-}"
DEST="${DEST:-ckpt}"
mkdir -p "$DEST"

python -c "import huggingface_hub" 2>/dev/null || pip install -q "huggingface_hub>=0.24"

python - "$HF_REPO" "$DEST" $INCLUDE <<'PY'
import fnmatch, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

repo, dest, *patterns = sys.argv[1:]
token = os.environ.get('HF_TOKEN') or None
api = HfApi(token=token)
files = [f for f in api.list_repo_tree(repo, repo_type='model', recursive=True)
         if isinstance(f, RepoFile)]
if patterns:
    files = [f for f in files if any(fnmatch.fnmatch(f.path, p) for p in patterns)]
if not files:
    sys.exit(f'nothing in {repo} matches {patterns or "<all>"}')
print(f'{repo} -> {dest}/  ({len(files)} file(s), patterns={patterns or "<all>"})')
# Each file is fetched into a scratch cache beside the target and MOVED
# to its final path, so the target folder ends up holding the files
# alone: no .cache/ metadata tree, no snapshot symlinks.
tmp = os.path.join(dest, '.hf_download_tmp')
total = 0
for f in files:
    dst = os.path.join(dest, f.path)
    size = f.size or 0
    total += size
    if os.path.isfile(dst) and os.path.getsize(dst) == size:
        print(f'{size / 1e9:8.2f} GB  {f.path}  (present, skipped)')
        continue
    got = hf_hub_download(repo, f.path, repo_type='model', cache_dir=tmp, token=token)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    shutil.move(os.path.realpath(got), dst)
    print(f'{os.path.getsize(dst) / 1e9:8.2f} GB  {f.path}')
shutil.rmtree(tmp, ignore_errors=True)
print(f'total {total / 1e9:.2f} GB under {dest}/')
PY
