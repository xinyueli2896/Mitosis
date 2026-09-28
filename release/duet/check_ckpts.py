"""Report which checkpoints in a run directory are actually loadable.

Definitive companion to ckpt_utils._ckpt_problem: that check is cheap
(size + zip central-directory presence) so it can run inside the
resolver, whereas this script ATTEMPTS A REAL torch.load and reports
the true verdict per file, plus what resolve_best_ckpt would pick.

Use it when an inference job dies with
    PytorchStreamReader failed reading zip archive:
    failed finding central directory
which means a checkpoint file is truncated or otherwise corrupt.

Usage:
    python check_ckpts.py ckpt/<run dir> [more dirs or files ...]
    python check_ckpts.py --delete-corrupt ckpt/<run dir>   # prune them

Exit status is 1 if any checkpoint is corrupt, so it can gate a job.
"""

import argparse
import os
import sys

import torch

from ckpt_utils import _ckpt_problem, resolve_best_ckpt


def iter_ckpts(paths):
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith('.ckpt'):
                    yield os.path.join(p, name)
        elif os.path.isfile(p):
            yield p
        else:
            print(f'[warn] {p!r} is neither a file nor a directory')


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f}{unit}' if unit == 'B' else f'{n:.1f}{unit}'
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='run directories and/or ckpt files')
    ap.add_argument('--delete-corrupt', action='store_true',
                    help='DELETE files that fail to load (irreversible)')
    ap.add_argument('--quick', action='store_true',
                    help='cheap structural check only; skip real torch.load')
    args = ap.parse_args()

    files = list(iter_ckpts(args.paths))
    if not files:
        raise SystemExit('no .ckpt files found')

    print(f'{"file":<62} {"size":>9}  verdict')
    print('-' * 100)
    corrupt, ok = [], []
    for f in files:
        size = os.path.getsize(f) if os.path.exists(f) else 0
        cheap = _ckpt_problem(f)
        if cheap is not None:
            verdict = f'CORRUPT ({cheap})'
            corrupt.append(f)
        elif args.quick:
            verdict = 'ok (structure only)'
            ok.append(f)
        else:
            try:
                ck = torch.load(f, map_location='cpu', weights_only=False)
                step = (ck.get('global_step', '?')
                        if isinstance(ck, dict) else '?')
                n_keys = (len(ck.get('state_dict', {}))
                          if isinstance(ck, dict) else '?')
                verdict = f'ok (global_step={step}, {n_keys} tensors)'
                ok.append(f)
                del ck
            except Exception as e:
                verdict = f'CORRUPT ({type(e).__name__}: {str(e)[:60]})'
                corrupt.append(f)
        print(f'{os.path.basename(f):<62} {human(size):>9}  {verdict}')

    print('-' * 100)
    print(f'{len(ok)} loadable, {len(corrupt)} corrupt')

    if corrupt and args.delete_corrupt:
        for f in corrupt:
            os.remove(f)
            print(f'[deleted] {f}')
        print('Re-run resolve to pick among the survivors.')
    elif corrupt:
        print('\nCorrupt files found. Options:')
        print('  * re-run with --delete-corrupt to remove them, then the')
        print('    resolver will pick the best SURVIVING checkpoint;')
        print('  * or pass a specific good ckpt file to the eval via')
        print('    CKPT_A2=<dir>/<good file>.ckpt')
        print('Likely cause: the write was cut short (full disk/quota, job')
        print('killed mid-save, or the file was copied while being written).')

    # Show what the resolver would choose for each directory.
    for p in args.paths:
        if os.path.isdir(p):
            try:
                print(f'\n[resolve] {p} -> {os.path.basename(resolve_best_ckpt(p))}')
            except Exception as e:
                print(f'\n[resolve] {p} -> FAILED: {e}')

    return 1 if corrupt else 0


if __name__ == '__main__':
    sys.exit(main())
