"""Why do all systems show the SAME numbers in an E3 table?

E3 gives the model the complete ground-truth partner stream. Every
conditional decoder copies that stream into its output VERBATIM --
A.2's make_actions_conditional returns ('given', condition[:, t, :]) for
every frame, and C.1/C.2 set `m_tokens = drum_tokens[:, t, :]` inside
the AR loop. So the given stream in A.2's output, B.1's output and
C.1's output is the same audio, note for note.

Consequence: every metric computed on the given stream is IDENTICAL for
every system by construction. In mel2chord that is nine of the roughly
twenty rows (the whole `_a` family plus mel_stepwise_delta,
mel_poly_rate and chord_tone_cov_ref). Those rows are not evidence about
any model and never were. Only rows on the GENERATED stream, and the
cross-stream metrics, discriminate.

That leaves a real question this script answers: are the rows that
SHOULD differ also identical? If so the eval really is broken, and the
two usual causes are distinguished here:

  * MANIFEST ALIASING -- two systems' rows point at the same files, so
    the same midis are scored twice under different names.
  * IDENTICAL GENERATION -- different paths, but the generated stream's
    notes hash the same, i.e. the systems really did produce the same
    music (a shared seed, or the wrong checkpoint loaded twice).

Usage:
    python check_e3_identical.py \
        --csv results/e3_melchord_p64_metrics.csv \
        --manifest results/e3_melchord_p64_manifest.tsv
"""

import argparse
import csv
import hashlib
import math
import os
from collections import defaultdict

from eval_metrics import GIVEN_STREAM_BY_MODE, H_GROUPS, STREAM_OF

ALL_METRICS = [k for h in ('H3', 'H2', 'H1') for k in H_GROUPS[h]]
TOL = 1e-9


def _f(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def load_csv(path):
    """-> per_song[(system, mode, song)][metric] = mean over samples"""
    acc = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row.get('system', '?'), row.get('mode', '-'),
                   row.get('song', '?'))
            for k, v in row.items():
                if k in ('system', 'mode', 'song', 'sample'):
                    continue
                x = _f(v)
                if x is not None:
                    acc[key][k].append(x)
    return {k: {m: sum(vs) / len(vs) for m, vs in d.items()}
            for k, d in acc.items()}


def check_metrics(per_song):
    """Per mode, split metrics into expected-identical and suspicious."""
    verdict_clean = True
    modes = sorted({m for (_, m, _) in per_song})
    for mode in modes:
        systems = sorted({s for (s, m, _) in per_song if m == mode})
        given = GIVEN_STREAM_BY_MODE.get(mode)
        print(f'\n--- mode={mode}  systems={systems} ---')
        if len(systems) < 2:
            print('    only one system: nothing to compare.')
            continue
        print(f'    GIVEN stream: {given or "none (co-generation)"}')

        by_design, real_diff, suspicious = [], [], []
        for metric in ALL_METRICS:
            # per-system vector over the songs every system defines it on
            song_sets = [{song for (sy, m, song), vals in per_song.items()
                          if sy == s and m == mode and metric in vals}
                         for s in systems]
            common = sorted(set.intersection(*song_sets)) if song_sets else []
            if not common:
                continue
            vecs = {s: [per_song[(s, mode, song)][metric] for song in common]
                    for s in systems}
            base = vecs[systems[0]]
            identical = all(
                all(abs(a - b) <= TOL for a, b in zip(base, vecs[s]))
                for s in systems[1:])
            expected = (given is not None
                        and STREAM_OF.get(metric, 'unknown') in (given, 'ref'))
            if identical and expected:
                by_design.append(metric)
            elif identical:
                suspicious.append((metric, len(common)))
                verdict_clean = False
            elif expected:
                # a given-stream metric that is NOT identical: the copies
                # differ, so at least one system is not copying cleanly
                real_diff.append(('!' + metric, len(common)))
            else:
                real_diff.append((metric, len(common)))

        print(f'    identical BY DESIGN (given/reference stream): '
              f'{len(by_design)} metric(s)')
        if by_design:
            print('      ' + ', '.join(by_design))
        print(f'    differ across systems: {len(real_diff)} metric(s)')
        if real_diff:
            print('      ' + ', '.join(f'{m}(n={n})' for m, n in real_diff))
            if any(m.startswith('!') for m, _ in real_diff):
                print('      ! = a GIVEN-stream metric that is NOT identical:'
                      ' that system is not copying the conditioning stream'
                      ' cleanly (check its tempo and its decode).')
        if suspicious:
            print(f'    *** IDENTICAL BUT SHOULD NOT BE: '
                  f'{len(suspicious)} metric(s)')
            print('      ' + ', '.join(f'{m}(n={n})' for m, n in suspicious))
        if not real_diff and not suspicious:
            print('    *** every comparable metric was dropped (no common '
                  'songs): the systems share no scorable song.')
            verdict_clean = False
    return verdict_clean


def check_manifest(path):
    """Do two systems' rows point at the same files?"""
    print(f'\n--- manifest {path} ---')
    if not os.path.exists(path):
        print('    not found -- skipping (pass --manifest to enable)')
        return True
    by_path = defaultdict(set)
    roots = defaultdict(set)
    n = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            system, mode, song, sample, p = parts[:5]
            n += 1
            by_path[(mode, song, sample, p)].add(system)
            roots[system].add(os.sep.join(p.split(os.sep)[:3]))
    print(f'    {n} rows')
    for system, rs in sorted(roots.items()):
        print(f'      {system}: {sorted(rs)}')
    shared = {k: v for k, v in by_path.items() if len(v) > 1}
    if shared:
        print(f'    *** {len(shared)} path(s) claimed by MORE THAN ONE '
              f'system -- the same midi is scored under several names:')
        for k, v in list(shared.items())[:5]:
            print(f'      {sorted(v)} <- {k[3]}')
        return False
    print('    every system has its own files (no aliasing).')
    return True


def _stream_hash(path, want_name):
    """Hash the notes of one named track, so two files can be compared
    on the GENERATED stream alone."""
    import warnings
    import pretty_midi
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pm = pretty_midi.PrettyMIDI(path)
    notes = []
    for inst in pm.instruments:
        if (inst.name or '').strip().lower() == want_name:
            notes.extend((round(n.start, 4), round(n.end, 4), n.pitch)
                         for n in inst.notes)
    if not notes:
        return None
    h = hashlib.sha1()
    for t in sorted(notes):
        h.update(repr(t).encode())
    return h.hexdigest()[:12]


def check_audio(manifest, limit=3):
    """Compare the GENERATED stream's notes across systems, song by song."""
    print(f'\n--- generated-stream note hashes (first {limit} song(s)) ---')
    if not os.path.exists(manifest):
        print('    no manifest -- skipping')
        return True
    rows = defaultdict(dict)          # (mode, song) -> system -> path
    with open(manifest) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            system, mode, song, sample, p = parts[:5]
            if sample != '0':
                continue              # one draw is enough to compare
            rows[(mode, song)][system] = p
    ok = True
    for (mode, song) in sorted(rows)[:limit]:
        given = GIVEN_STREAM_BY_MODE.get(mode)
        if given is None:
            continue
        gen_name = 'chord' if given == 'a' else 'melody'
        giv_name = 'melody' if given == 'a' else 'chord'
        hashes, ghashes = {}, {}
        for system, p in sorted(rows[(mode, song)].items()):
            if not os.path.exists(p):
                continue
            try:
                hashes[system] = _stream_hash(p, gen_name)
                ghashes[system] = _stream_hash(p, giv_name)
            except Exception as e:
                print(f'    [skip] {p}: {e!r}')
        if len(hashes) < 2:
            continue
        uniq = len(set(hashes.values()))
        print(f'    {mode}/{song}  given({giv_name}) '
              f'{"SHARED" if len(set(ghashes.values())) == 1 else "DIFFERS"}'
              f'   generated({gen_name}) {uniq} distinct value(s) '
              f'over {len(hashes)} system(s)')
        for system, hv in sorted(hashes.items()):
            print(f'        {system:6s} gen={hv}  given={ghashes.get(system)}')
        if uniq == 1:
            print('        *** all systems generated the SAME notes.')
            ok = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True,
                   help='results/e3_<task>_<suffix>_metrics.csv')
    p.add_argument('--manifest', default=None,
                   help='the matching *_manifest.tsv (enables the aliasing '
                        'and note-hash checks)')
    p.add_argument('--songs', type=int, default=3,
                   help='songs to note-hash (default 3)')
    args = p.parse_args()
    if args.manifest is None:
        guess = args.csv.replace('_metrics.csv', '_manifest.tsv')
        args.manifest = guess if os.path.exists(guess) else None

    print('=' * 70)
    print('E3 "ALL SYSTEMS LOOK THE SAME" DIAGNOSTIC')
    print(f'csv      = {args.csv}')
    print(f'manifest = {args.manifest or "<none>"}')
    print('=' * 70)

    per_song = load_csv(args.csv)
    metrics_clean = check_metrics(per_song)
    manifest_clean = check_manifest(args.manifest) if args.manifest else True
    audio_clean = check_audio(args.manifest, args.songs) if args.manifest else True

    print('\n' + '=' * 70)
    print('VERDICT')
    if metrics_clean and manifest_clean and audio_clean:
        print('  NOT A BUG. The rows that are identical across systems are')
        print('  the ones computed on the GIVEN stream -- the real partner,')
        print('  copied verbatim into every system\'s output. They are the')
        print('  same music in every column by design and say nothing about')
        print('  any model. Read the generated stream\'s rows and the cross-')
        print('  stream metrics; aggregate_eval_results marks the given rows')
        print('  (=) and leaves them untested.')
        print('  In mel2chord the generated stream is CHORD, so read the _b')
        print('  rows, harmonic_rhythm_jsd and chord_tone_cov_delta. In')
        print('  chord2mel it is MELODY: read the _a rows -- and note that')
        print('  harmonic_rhythm_jsd, the H3 primary, is on the given stream')
        print('  there and cannot discriminate.')
    else:
        if not manifest_clean:
            print('  MANIFEST ALIASING: systems share output paths, so the')
            print('  same midis were scored under several names. Check the')
            print('  --source roots in eval_e3.sbatch.')
        if not audio_clean:
            print('  IDENTICAL GENERATION: different files, same generated')
            print('  notes. The systems really did produce the same music --')
            print('  check that each --ckpt is the intended run, and that')
            print('  SKIP_INFER=1 is not reusing an older run\'s outputs.')
        if not metrics_clean:
            print('  METRICS: at least one metric on the GENERATED stream is')
            print('  identical across systems (listed above), or no song is')
            print('  common to all of them. Both point upstream of scoring.')
    print('=' * 70)


if __name__ == '__main__':
    main()
