"""Which songs have both streams present within the first N bars?

For each song id, reads the melody and chord midis from the TIME-ALIGNED
source folders (not the staged copies: those are cropped to the
co-entry bar by audit_wholesong_split), frames onsets on the TICK grid
(the grid the tokenizer reads; 16 frames per bar), and prints per song
the bar of each stream's first onset and the number of onsets of each
stream inside the first N bars for every N in --bars. Ends with, for
each consecutive pair of N values, the songs that have both streams
non-empty at the larger N but not at the smaller one.

Usage (via check_prompt_entry.sbatch):
    python check_prompt_entry.py --songs 004 046 ... --bars 6 8
"""
import argparse
import os
import re

import pretty_midi

from eval_metrics import frame_fn

FPB = 16


def find(folder, sid):
    for f in sorted(os.listdir(folder)):
        m = re.search(r'(?<!\d)(\d{3})(?!\d)', f)
        if m and int(m.group(1)) == sid and f.lower().endswith('.mid'):
            return os.path.join(folder, f)
    return None


def onsets(fn):
    pm = pretty_midi.PrettyMIDI(fn)
    fr = frame_fn(pm)
    return sorted(fr(n.start) for ins in pm.instruments for n in ins.notes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mel-src', default='/home/xinyue.li/POP909-Dataset/POP909-melody')
    p.add_argument('--chord-src', default='/home/xinyue.li/POP909-Dataset/POP909-chord')
    p.add_argument('--songs', nargs='+', required=True, help='3-digit ids')
    p.add_argument('--bars', nargs='+', type=int, default=[6, 8])
    p.add_argument('--min-notes', type=int, default=1,
                   help='a stream counts as present with at least this many '
                        'onsets inside the window')
    args = p.parse_args()
    bars = sorted(set(args.bars))

    hdr = f'{"song":<6}{"entry mel":>10}{"entry chd":>10}'
    for b in bars:
        hdr += f'{f"mel<{b}b":>9}{f"chd<{b}b":>9}{f"both<{b}b":>10}'
    print(f'onset counts on the tick grid, {FPB} frames/bar; entry = bar of first onset')
    print(hdr)
    ok = {b: set() for b in bars}
    for s in args.songs:
        sid = int(s)
        fm, fc = find(args.mel_src, sid), find(args.chord_src, sid)
        if not (fm and fc):
            print(f'{sid:03d}   MISSING  mel={fm} chord={fc}')
            continue
        om, oc = onsets(fm), onsets(fc)
        em = int(om[0] // FPB) if om else None
        ec = int(oc[0] // FPB) if oc else None
        row = f'{sid:03d}   {str(em):>9} {str(ec):>9}'
        for b in bars:
            nm = sum(1 for x in om if x < b * FPB)
            nc = sum(1 for x in oc if x < b * FPB)
            both = nm >= args.min_notes and nc >= args.min_notes
            if both:
                ok[b].add(sid)
            row += f'{nm:>9}{nc:>9}{"yes" if both else "NO":>10}'
        print(row)

    fmt = lambda ids: ' '.join(f'{i:03d}' for i in sorted(ids)) or '(none)'
    print()
    for b in bars:
        print(f'both streams present within {b} bars: {len(ok[b])}  {fmt(ok[b])}')
    for lo, hi in zip(bars, bars[1:]):
        print(f'present within {hi} bars but NOT within {lo}: {fmt(ok[hi] - ok[lo])}')


if __name__ == '__main__':
    main()
