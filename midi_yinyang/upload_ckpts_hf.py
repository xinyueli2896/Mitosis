"""Archive checkpoint run-directories to a private Hugging Face repo.

Uploads each matched directory under ckpt/ to the repo at the same
relative path, then VERIFIES the upload by listing the remote tree and
comparing every file's size against the local one. Nothing is deleted
unless --delete is passed AND that directory verified clean -- and even
then each removal is logged file by file.

The point is archival, not publication: create the repo PRIVATE (the
default here) and it stays a personal storage bucket until you flip it.

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
    print(f'{total / 1e9:10.2f}  TOTAL -> {args.repo} (private)')
    if args.dry_run:
        print('[dry-run] stopping before any upload.')
        return

    api = HfApi()
    create_repo(args.repo, repo_type='model', private=True, exist_ok=True)

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
