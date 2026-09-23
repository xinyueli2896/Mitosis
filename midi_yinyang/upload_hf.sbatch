#!/bin/bash
# SLURM (CPU): upload checkpoint files or directories to a PUBLIC Hugging
# Face model repo, creating the repo if it does not exist.
#
# Env knobs (--export=ALL,KEY=val):
#   HF_TOKEN   a write token from https://huggingface.co/settings/tokens
#              (REQUIRED; pass it through --export, never write it down)
#   HF_REPO    target repo id, e.g. xinyueli/duet-melchord  (REQUIRED)
#   PATHS      space-separated local files or directories to upload
#              (REQUIRED). A directory is uploaded recursively under its
#              own name; a file goes to the repo root.
#   CARD       optional local README.md to upload as the model card
#   PRIVATE    1 to create the repo private instead (default 0 = public)
#
# Example:
#   sbatch --export=ALL,HF_TOKEN=hf_xxx,HF_REPO=xinyueli/duet-melchord,\
#   PATHS="ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mg_melchord_cp4tar_batch_8_schedule/best.ckpt ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt" \
#       midi_yinyang/upload_hf.sbatch
#
# Large files (> 5 GB) are handled by the hub's chunked upload; a
# directory with many files is committed in one go.

#SBATCH --job-name=upload_hf
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
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR="${REPO_DIR:-/home/xinyue.li/Mitosis}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

: "${HF_TOKEN:?set HF_TOKEN=<write token> on the --export line}"
: "${HF_REPO:?set HF_REPO=<user>/<repo>}"
: "${PATHS:?set PATHS=\"<file-or-dir> ...\"}"
CARD="${CARD:-}"
PRIVATE="${PRIVATE:-0}"

python -c "import huggingface_hub" 2>/dev/null || pip install -q "huggingface_hub>=0.24"

for p in $PATHS; do
    [[ -e "$p" ]] || { echo "ERROR: $p does not exist"; exit 1; }
done

python - "$HF_REPO" "$PRIVATE" "$CARD" $PATHS <<'EOF'
import os, sys
from huggingface_hub import HfApi

repo, private, card, *paths = sys.argv[1:]
api = HfApi(token=os.environ['HF_TOKEN'])
url = api.create_repo(repo_id=repo, repo_type='model', private=(private == '1'), exist_ok=True)
print('repo:', url)
for p in paths:
    if os.path.isdir(p):
        name = os.path.basename(p.rstrip('/'))
        print(f'uploading directory {p} -> {name}/')
        api.upload_folder(folder_path=p, path_in_repo=name, repo_id=repo, repo_type='model',
                          commit_message=f'add {name}')
    else:
        print(f'uploading file {p} -> {os.path.basename(p)}')
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p), repo_id=repo,
                        repo_type='model', commit_message=f'add {os.path.basename(p)}')
if card:
    print('uploading model card', card)
    api.upload_file(path_or_fileobj=card, path_in_repo='README.md', repo_id=repo,
                    repo_type='model', commit_message='model card')
print('done:', url)
EOF
