"""What do the H1 stream metrics score on the GROUND TRUTH itself?

Every H1 number is being read against an implicit assumption about what
"healthy" looks like, and that assumption was never measured. B.1's
follower scores survival_b ~ 0.66, which looks like a deficit next to
S1's 0.930 and A.2's 1.000 -- but those are other MODELS, not the data.
If the real chord track is itself active in only two thirds of its bars,
0.66 is correct behaviour and it is S1/A.2 that over-produce.

This scores the reference against itself over the same window the
evaluation uses, reusing eval_metrics' own loaders and h1_metrics so the
numbers are directly comparable to any results table:

  survival_a / survival_b   fraction of bars the real stream is active
  empty_rate_a / _b         silent-frame fraction of the real stream
  density_drift_a / _b      does the real music thin out over 24 bars?
  density_ratio_*           1.0 by construction (ref vs itself); a sanity
                            check that the pairing is right

Usage:
    python reference_stats.py \
        --ref-a-dir input/pop909_split/melody \
        --ref-b-dir input/pop909_split/chord \
        --prompt-frames 64 --total-frames 384
"""

import argparse
import math
import os
from glob import glob

from eval_metrics import load_streams, h1_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref-a-dir', required=True, help='melody / drum midis')
    p.add_argument('--ref-b-dir', required=True, help='chord / nondrum midis')
    p.add_argument('--task', default='melchord',
                   choices=['melchord', 'drumnondrum'])
    p.add_argument('--prompt-frames', type=int, default=64)
    p.add_argument('--total-frames', type=int, default=384)
    p.add_argument('--mel-programs', default='0,24')
    p.add_argument('--chord-programs', default='48')
    args = p.parse_args()
    args.mel_programs = {int(x) for x in args.mel_programs.split(',')}
    args.chord_programs = {int(x) for x in args.chord_programs.split(',')}

    # Both extension cases: the RWC split uses uppercase .MID
    # (RM-P*.SMF_SYNC.MID), POP909 lowercase .mid.
    def _midis(d):
        return {os.path.splitext(os.path.basename(f))[0]: f
                for pat in ('*.mid', '*.MID')
                for f in sorted(glob(os.path.join(d, pat)))}
    a_files = _midis(args.ref_a_dir)
    b_files = _midis(args.ref_b_dir)
    songs = sorted(set(a_files) & set(b_files))
    if not songs:
        raise SystemExit(
            f'no paired songs between {args.ref_a_dir} and {args.ref_b_dir} '
            f'({len(a_files)} vs {len(b_files)} files; they are paired by '
            f'filename stem)')
    missing = sorted((set(a_files) ^ set(b_files)))
    if missing:
        print(f'[warn] {len(missing)} unpaired song(s) ignored: {missing[:8]}')

    lo, hi = args.prompt_frames, args.total_frames
    print(f'{len(songs)} paired song(s); scoring frames [{lo}, {hi}) '
          f'= bars {lo // 16}-{hi // 16}\n')

    rows = []
    for sid in songs:
        try:
            ra, rb = load_streams([a_files[sid], b_files[sid]], args.task,
                                  args.mel_programs, args.chord_programs, hi)
            # Reference vs ITSELF: every "generated" stream is the real one.
            sa, sb = ra.slice(lo, hi), rb.slice(lo, hi)
            rows.append(h1_metrics(sa, sb, sa, sb, args.task))
        except Exception as e:
            print(f'  [skip] {sid}: {e!r}')

    if not rows:
        raise SystemExit('nothing scored')

    keys = ['survival_a', 'survival_b', 'survival_min',
            'density_ratio_a', 'density_ratio_b',
            'density_drift_a', 'density_drift_b',
            'empty_rate_a', 'empty_rate_b']
    print(f'{"metric":<20}{"ground truth":>14}{"n":>5}')
    print('-' * 39)
    for k in keys:
        vals = [r[k] for r in rows
                if k in r and not math.isnan(float(r[k]))]
        if not vals:
            print(f'{k:<20}{"--":>14}{0:>5}')
            continue
        print(f'{k:<20}{sum(vals) / len(vals):>14.4f}{len(vals):>5}')

    print("""
This is the CEILING, not a target to beat. A model scoring near these
values is reproducing the data's own behaviour; scoring far ABOVE
survival/density is over-producing, not doing better. density_ratio_*
must read 1.0 -- anything else means the reference pairing is wrong.""")


if __name__ == '__main__':
    main()
