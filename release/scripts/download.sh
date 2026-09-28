#!/usr/bin/env bash
# Download released files from a Hugging Face model repository into duet/.
#
# The repository mirrors duet/'s layout (ckpt/..., data/..., input/...),
# so every file lands where the other scripts look for it.
#
#   HF_REPO=<owner>/<repo> bash scripts/download.sh                 # everything
#   HF_REPO=<owner>/<repo> bash scripts/download.sh 'ckpt/*' 'input/*'
#
# HF_TOKEN is read from the environment for private repositories.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${HF_REPO:?set HF_REPO=<owner>/<repo>}"

python - "$HF_REPO" "$DUET" "$@" <<'PY'
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
tmp = os.path.join(dest, '.hf_download_tmp')
for f in files:
    dst = os.path.join(dest, f.path)
    if os.path.isfile(dst) and os.path.getsize(dst) == (f.size or 0):
        print(f'present  {f.path}')
        continue
    got = hf_hub_download(repo, f.path, repo_type='model', cache_dir=tmp, token=token)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    shutil.move(os.path.realpath(got), dst)
    print(f'fetched  {f.path}')
shutil.rmtree(tmp, ignore_errors=True)
PY
