"""Concatenate eval_metrics CSVs from several E1 runs into one.

Used by merge_e1.sbatch to score prompt-length groups (p6 / p8 songs)
as one table. Checks that every input has the same columns, that no
(system, mode, song, sample) row appears twice across inputs, and
prints the songs contributed per input so the union is visible in the
log. Adds a 'run' column naming the source file stem.
"""
import argparse
import csv
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    header, rows, seen = None, [], set()
    for fn in args.inputs:
        with open(fn) as fh:
            rd = csv.DictReader(fh)
            cols = list(rd.fieldnames or [])
            if header is None:
                header = cols
            elif cols != header:
                raise SystemExit(f'column mismatch: {fn} has {cols[:8]}... vs '
                                 f'{header[:8]}... -- rescore with the same '
                                 f'eval_metrics version')
            n, songs = 0, set()
            run = os.path.basename(fn).replace('_metrics.csv', '')
            for r in rd:
                key = (r.get('system'), r.get('mode'), r.get('song'), r.get('sample'))
                if key in seen:
                    raise SystemExit(f'duplicate row {key} in {fn}: the runs '
                                     f'overlap in songs')
                seen.add(key)
                r['run'] = run
                rows.append(r)
                n += 1
                songs.add(r.get('song'))
            print(f'[merge] {fn}: {n} rows, {len(songs)} songs: '
                  f'{" ".join(sorted(songs))}')
    if not rows:
        raise SystemExit('no rows')
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=header + ['run'])
        w.writeheader()
        w.writerows(rows)
    all_songs = sorted({r.get('song') for r in rows})
    systems = sorted({r.get('system') for r in rows})
    print(f'[merge] wrote {len(rows)} rows, {len(all_songs)} songs, '
          f'systems {" ".join(systems)} -> {args.out}')


if __name__ == '__main__':
    main()
