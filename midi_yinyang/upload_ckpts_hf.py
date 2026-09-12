"""Archive checkpoint run-directories to a Hugging Face repo.

Uploads each matched directory under ckpt/ to the repo at the same
relative path, then VERIFIES the upload by listing the remote tree and
comparing every file's size against the local one. Nothing is deleted
unless --delete is passed AND that directory verified clean -- and even
then each removal is logged file by file.

The default is PRIVATE -- archival, not publication -- and the repo
stays a personal storage bucket until you flip it. --public creates it
world-readable instead, which is what a large archive usually needs
since public repos get a far more generous storage allowance. Weigh
that once: published weights can be mirrored or indexed, and deleting
the repo later does not reliably unpublish them.

Auth: run `huggingface-cli login` once on the login node (stores the
token under ~/.cache/huggingface/), or export HF_TOKEN.

Usage (via upload_ckpts_hf.sbatch):
    python upload_ckpts_hf.py --repo <user>/<repo> \
        --patterns 'm2c_duet_anticipatory_*' 'm2c_duet_rehearsal_*' \
                   'm2c_duet_prefix_*' \
        [--dry-run] [--delete]
"""

import argparse
import glob
import os
import shutil
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    sys.exit('huggingface_hub is not installed in this env -- '
             'pip install -U huggingface_hub')


def resolve_token():
    """Find a WORKING write token, trying every place one could live.

    huggingface_hub's own lookup is: HF_TOKEN env var first, then the
    token file under HF_HOME (only falling back to ~/.cache/huggingface
    when HF_HOME is unset). On this cluster HF_HOME points into
    /scratch, which may be node-local -- so a login-node login can be
    invisible to a compute node, and a stale HF_TOKEN in ~/.bashrc
    silently overrides a good file. Instead of trusting the implicit
    lookup, try each candidate, VALIDATE it with whoami(), and say
    which one won (or why every one failed).
    """
    cands = []
    for var in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        v = os.environ.get(var)
        if v is not None:
            cands.append((f'env {var}', v.strip()))
    file_cands = []
    if os.environ.get('HF_TOKEN_PATH'):
        file_cands.append(os.environ['HF_TOKEN_PATH'])
    if os.environ.get('HF_HOME'):
        file_cands.append(os.path.join(os.environ['HF_HOME'], 'token'))
    file_cands.append(os.path.expanduser('~/.cache/huggingface/token'))
    user = os.environ.get('USER', '')
    if user:
        file_cands.append(f'/scratch/{user}/cache/huggingface/token')
    for f in file_cands:
        if os.path.isfile(f):
            with open(f) as fh:
                cands.append((f'file {f}', fh.read().strip()))
    if not cands:
        sys.exit('[auth] no token found anywhere (no HF_TOKEN env, no '
                 'token file). Run `huggingface-cli login` on the login '
                 'node first.')
    for src, tok in cands:
        if not tok:
            print(f'[auth] {src}: EMPTY -- skipping (an empty env var '
                  f'overrides good file tokens in the default lookup!)')
            continue
        try:
            who = HfApi(token=tok).whoami()
            print(f'[auth] {src}: VALID, user={who["name"]}')
            return tok
        except Exception as e:                        # noqa: BLE001
            print(f'[auth] {src}: rejected by the Hub ({e})')
    sys.exit('[auth] every candidate token failed validation -- '
             're-create a WRITE token and re-run huggingface-cli login.')


def local_files(root):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, os.path.dirname(root))] = os.path.getsize(p)
    return out


def verify(api, repo, folder_name, want):
    """Compare remote tree of <folder_name> against {relpath: size}."""
    got = {}
    for entry in api.list_repo_tree(repo, path_in_repo=folder_name,
                                    recursive=True):
        size = getattr(entry, 'size', None)
        if size is not None:
            got[entry.path] = size
    missing = [p for p in want if p not in got]
    mismatched = [p for p, s in want.items()
                  if p in got and got[p] != s]
    return missing, mismatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True,
                    help='e.g. <hf-username>/mitosis-ckpt-archive')
    ap.add_argument('--ckpt-root', default='ckpt')
    ap.add_argument('--patterns', nargs='+', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--delete', action='store_true',
                    help='remove each LOCAL dir after ITS verification '
                         'passes. Off by default.')
    ap.add_argument('--public', action='store_true',
                    help='create the repo PUBLIC. Off by default: a new '
                         'repo is private, which is what archival wants. '
                         'Public repos get a far more generous storage '
                         'allowance, which is the usual reason to want '
                         'this -- but the weights become world-readable '
                         'and may be mirrored or indexed even if the repo '
                         'is deleted later, so it is not a setting to '
                         'flip back and forth. Ignored for a repo that '
                         'already exists: change visibility in the repo '
                         'settings instead.')
    args = ap.parse_args()

    dirs = []
    for pat in args.patterns:
        dirs += sorted(glob.glob(os.path.join(args.ckpt_root, pat)))
    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]
    if not dirs:
        sys.exit(f'no directories matched {args.patterns} '
                 f'under {args.ckpt_root}/')

    total = 0
    print(f'{"size (GB)":>10}  directory')
    plans = []
    for d in dirs:
        want = local_files(d)
        sz = sum(want.values())
        total += sz
        plans.append((d, want, sz))
        print(f'{sz / 1e9:10.2f}  {d}')
    vis = 'PUBLIC' if args.public else 'private'
    print(f'{total / 1e9:10.2f}  TOTAL -> {args.repo} ({vis})')
    if args.dry_run:
        print('[dry-run] stopping before any upload.')
        return

    token = resolve_token()
    api = HfApi(token=token)
    # exist_ok=True means `private` is only honoured when the repo is
    # CREATED here; an existing repo keeps whatever visibility it has.
    create_repo(args.repo, repo_type='model', private=not args.public,
                exist_ok=True, token=token)
    info = api.repo_info(args.repo, repo_type='model', token=token)
    actual = 'PUBLIC' if not getattr(info, 'private', True) else 'private'
    print(f'[repo] {args.repo} is {actual}')
    if args.public and actual != 'PUBLIC':
        print('[repo] WARNING: --public was passed but the repo is still '
              'private -- it already existed. Change it in the repo '
              'settings on huggingface.co, or upload to a new name.')

    failures = []
    for d, want, sz in plans:
        name = os.path.basename(d)
        print(f'\n=== uploading {name} ({sz / 1e9:.2f} GB) ===', flush=True)
        api.upload_folder(
            repo_id=args.repo,
            folder_path=d,
            path_in_repo=name,
            commit_message=f'archive {name}',
        )
        missing, mismatched = verify(api, args.repo, name, want)
        if missing or mismatched:
            failures.append(name)
            print(f'[VERIFY FAIL] {name}: {len(missing)} missing, '
                  f'{len(mismatched)} size-mismatched')
            for p in (missing + mismatched)[:10]:
                print(f'    {p}')
            print('    -> NOT deleting this directory.')
            continue
        print(f'[verified] {name}: {len(want)} file(s), sizes match remote.')
        if args.delete:
            print(f'[delete] removing local {d}')
            shutil.rmtree(d)

    print('\n================ SUMMARY ================')
    print(f'uploaded+verified: {len(plans) - len(failures)}/{len(plans)} '
          f'dirs, {total / 1e9:.2f} GB scanned')
    if failures:
        print(f'FAILED verification (kept locally): {failures}')
        sys.exit(1)
    if not args.delete:
        print('local copies KEPT (rerun with DELETE=1 to remove verified '
              'dirs, or rm them yourself).')
    print(f'restore any dir later with:')
    print(f'  huggingface-cli download {args.repo} '
          f'--include "<dir_name>/*" --local-dir ckpt/')


if __name__ == '__main__':
    main()
