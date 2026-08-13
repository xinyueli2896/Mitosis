"""Reclaim disk from run directories: in each ckpt/<run>/ dir, keep only
the LOWEST-val_loss checkpoint (the one every resolver picks) and delete
the other val_loss-tagged files.

SAFE BY DEFAULT:
  * dry run unless --delete is passed -- prints what would go and the
    bytes reclaimed, deletes nothing;
  * last.ckpt is KEPT unless --purge-last: it is the resume point, and
    deleting it from a run that is still training (or might be extended)
    breaks the auto-resume loop;
  * *.fin.ckpt files are KEPT unless --purge-fin;
  * files without a val_loss= tag in their name are never touched, so
    loose checkpoints (the S0 pretrain, warm-start inits, YinYang
    downloads) are structurally exempt;
  * directories with fewer than two val_loss-tagged files are skipped.

Usage:
    python cleanup_ckpts.py                # dry run over ckpt/
    python cleanup_ckpts.py --delete       # actually delete
    python cleanup_ckpts.py --root ckpt --exclude '*cp4tar*' --delete
"""

import argparse
import fnmatch
import os
import re
import sys

VAL_RE = re.compile(r'val_loss=([0-9]+(?:\.[0-9]+)?)')


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f}{unit}'
        n /= 1024.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='ckpt')
    p.add_argument('--delete', action='store_true', default=False,
                   help='actually delete (default: dry run)')
    p.add_argument('--purge-last', action='store_true', default=False,
                   help='also delete last.ckpt (NOT for runs that may '
                        'resume -- e.g. a training still in progress)')
    p.add_argument('--purge-fin', action='store_true', default=False,
                   help='also delete *.fin.ckpt files')
    p.add_argument('--exclude', action='append', default=[],
                   metavar='GLOB',
                   help='skip run dirs whose NAME matches (repeatable), '
                        "e.g. --exclude '*cp4tar*' for a run still "
                        'training')
    args = p.parse_args()

    total = 0
    victims = []
    for entry in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, entry)
        if not os.path.isdir(run_dir):
            continue
        if any(fnmatch.fnmatch(entry, g) for g in args.exclude):
            print(f'[skip] {entry} (excluded)')
            continue
        tagged = []
        extras = []
        for f in os.listdir(run_dir):
            path = os.path.join(run_dir, f)
            if not os.path.isfile(path):
                continue
            m = VAL_RE.search(f)
            if m:
                tagged.append((float(m.group(1)), path))
            elif f == 'last.ckpt' and args.purge_last:
                extras.append(path)
            elif f.endswith('.fin.ckpt') and args.purge_fin:
                extras.append(path)
        if len(tagged) < 2 and not extras:
            continue
        tagged.sort()
        keep = tagged[0][1] if tagged else None
        drop = [pth for _, pth in tagged[1:]] + extras
        if keep:
            print(f'[keep] {os.path.relpath(keep, args.root)} '
                  f'(val_loss={tagged[0][0]})')
        for pth in drop:
            sz = os.path.getsize(pth)
            total += sz
            victims.append(pth)
            print(f'[{"DELETE" if args.delete else "would delete"}] '
                  f'{os.path.relpath(pth, args.root)}  ({human(sz)})')

    verdict = ('deleted' if args.delete
               else 'reclaimable (dry run -- re-run with --delete)')
    print(f'\n{len(victims)} file(s), {human(total)} {verdict}')
    if args.delete:
        for pth in victims:
            os.remove(pth)
    return 0


if __name__ == '__main__':
    sys.exit(main())
