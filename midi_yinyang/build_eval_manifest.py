"""Build the eval_metrics.py manifest TSV by scanning inference output
folders.

eval_metrics.py --manifest reads rows of
    system <TAB> mode <TAB> song <TAB> sample <TAB> path
where path is one midi (both streams in one file) or "melA;chordB" for
systems that emit per-stream files. This script produces that TSV from
the layouts the inference scripts actually write, so Phase 2 needs no
hand-editing.

Supported layouts (--layout):

  duet      <root>/<song>/<mode>.mid
            Written by the *_combined.py 5-mode scripts (A.1, A.2,
            i.e. infer_duet_block_diffusion.sbatch / infer_all_rwc.sbatch).
            One sample per (song, mode); sample id is always 0.

  duet_multi  <root>/<song>/<mode>/sample_<i>_temp<T>.mid
            Written by cp_transformer_m2c_jointattn_inference.py's
            run_folder (per-mode dirs, n-samples per mode).

  cond      <root>/<song>/<direction>_temp<T>.mid
            Written by the C.1 / C.2 drum->nondrum scripts
            (cp_transformer_m2c_duet_{rehearsal,prefix}_inference.py).
            mode is taken from the filename stem before "_temp".

  single    <root>/<song>.mid_temp<T>_continuation_<i>.mid
            Written by cp_transformer_inference.py (S0/S1/specialists).
            Prompt renders (*_prompt.mid) are skipped. Song id is the
            source midi stem; sample id is the continuation index.
            Use --mode to label these rows (default: co).

Repeat --source once per system:
    --source <system>:<layout>:<root>

Example (E1, melchord):
    python build_eval_manifest.py \
        --source "A.2:duet:temp/duet_block_diffusion_12345" \
        --source "S1:single:temp/pop909_singlestream_s1" \
        --songs 001 011 021 031 041 051 061 071 081 091 \
        --modes co \
        --out results/e1_melchord_manifest.tsv

Song filtering (--songs) is recommended: it both restricts scoring to
the held-out evaluation ids and reports any (system, song) cell that is
MISSING, so a silently-failed inference run cannot quietly shrink the
comparison.
"""

import argparse
import os
import re
from collections import defaultdict


def _stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def scan_duet(root, single_sample=True):
    """<root>/<song>/<mode>.mid"""
    rows = []
    if not os.path.isdir(root):
        return rows
    for song in sorted(os.listdir(root)):
        song_dir = os.path.join(root, song)
        if not os.path.isdir(song_dir):
            continue
        for f in sorted(os.listdir(song_dir)):
            if not f.lower().endswith('.mid'):
                continue
            rows.append((_stem(f), song, '0', os.path.join(song_dir, f)))
    return rows


def scan_duet_multi(root):
    """<root>/<song>/<mode>/sample_<i>_temp<T>.mid"""
    rows = []
    if not os.path.isdir(root):
        return rows
    for song in sorted(os.listdir(root)):
        song_dir = os.path.join(root, song)
        if not os.path.isdir(song_dir):
            continue
        for mode in sorted(os.listdir(song_dir)):
            mode_dir = os.path.join(song_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
            for f in sorted(os.listdir(mode_dir)):
                m = re.match(r'sample_(\d+)_temp', f)
                if not m or not f.lower().endswith('.mid'):
                    continue
                rows.append((mode, song, m.group(1),
                             os.path.join(mode_dir, f)))
    return rows


def scan_cond(root):
    """<root>/<song>/<direction>_temp<T>.mid"""
    rows = []
    if not os.path.isdir(root):
        return rows
    for song in sorted(os.listdir(root)):
        song_dir = os.path.join(root, song)
        if not os.path.isdir(song_dir):
            continue
        for f in sorted(os.listdir(song_dir)):
            if not f.lower().endswith('.mid'):
                continue
            mode = re.split(r'_temp', _stem(f))[0]
            rows.append((mode, song, '0', os.path.join(song_dir, f)))
    return rows


def scan_single(root, mode_label):
    """<root>/<song>.mid_temp<T>_continuation_<i>.mid"""
    rows = []
    if not os.path.isdir(root):
        return rows
    for f in sorted(os.listdir(root)):
        if not f.lower().endswith('.mid'):
            continue
        m = re.match(r'(.+?)\.mid_temp[\d.]+_continuation_(\d+)\.mid$', f)
        if not m:
            continue          # skips *_prompt.mid and anything unexpected
        rows.append((mode_label, m.group(1), m.group(2),
                     os.path.join(root, f)))
    return rows


SCANNERS = {
    'duet': lambda root, mode_label: scan_duet(root),
    'duet_multi': lambda root, mode_label: scan_duet_multi(root),
    'cond': lambda root, mode_label: scan_cond(root),
    'single': scan_single,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', action='append', required=True,
                   help='<system>:<layout>:<root>, repeatable. Layouts: '
                        + ', '.join(sorted(SCANNERS)))
    p.add_argument('--out', required=True)
    p.add_argument('--songs', nargs='*', default=None,
                   help='restrict to these song ids and report missing cells')
    p.add_argument('--modes', nargs='*', default=None,
                   help='restrict to these modes')
    p.add_argument('--mode', default='co',
                   help="mode label for 'single' layout rows (default: co)")
    args = p.parse_args()

    want_songs = set(args.songs) if args.songs else None
    want_modes = set(args.modes) if args.modes else None

    rows = []
    seen = defaultdict(set)     # (system, mode) -> {songs}
    for src in args.source:
        try:
            system, layout, root = src.split(':', 2)
        except ValueError:
            raise SystemExit(f'--source must be system:layout:root, got {src!r}')
        if layout not in SCANNERS:
            raise SystemExit(f'unknown layout {layout!r}; choose from '
                             f'{sorted(SCANNERS)}')
        found = SCANNERS[layout](root, args.mode)
        if not found:
            print(f'[{system}] WARNING: no midis found under {root} '
                  f'(layout={layout})')
        kept = 0
        for mode, song, sample, path in found:
            if want_songs is not None and song not in want_songs:
                continue
            if want_modes is not None and mode not in want_modes:
                continue
            rows.append((system, mode, song, sample, path))
            seen[(system, mode)].add(song)
            kept += 1
        print(f'[{system}] layout={layout} root={root}: '
              f'{kept} rows kept ({len(found)} found)')

    # Coverage report: every (system, mode) should cover every wanted song.
    if want_songs:
        for (system, mode), songs in sorted(seen.items()):
            missing = sorted(want_songs - songs)
            if missing:
                print(f'[coverage] {system}/{mode} MISSING {len(missing)} '
                      f'song(s): {missing}')
        modes_by_system = defaultdict(set)
        for (system, mode) in seen:
            modes_by_system[system].add(mode)
        all_modes = set(m for ms in modes_by_system.values() for m in ms)
        for system, modes in sorted(modes_by_system.items()):
            absent = sorted(all_modes - modes)
            if absent:
                print(f'[coverage] {system} has NO rows for mode(s): {absent}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for row in rows:
            f.write('\t'.join(row) + '\n')
    print(f'\nwrote {len(rows)} rows -> {args.out}')
    print('score with: python eval_metrics.py --task <task> --manifest '
          f'{args.out} --ref-a-dir <...> --ref-b-dir <...> --out results.csv')


if __name__ == '__main__':
    main()
