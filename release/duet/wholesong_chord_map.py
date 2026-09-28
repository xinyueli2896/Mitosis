"""Where the whole-song baseline's chord differs from ours, and the map
that takes one to the other.

The baseline is prompted with OUR chord (wholesong_our_prompt.py), but
what it writes out is not our file: its lead-sheet image stores a
chord as two pitch-class bands, chroma and bass, and its own converter
renders those back to notes at its own octave and segmentation. So the
prompt bars of its output are a RE-RENDERING of our prompt chord, and
its continuation is in the same rendering. For the chord-side metrics
that see absolute pitch (voicing JSD, unique beat ratio on the full
state, FMD) that is a convention difference, not a musical one.

This tool does three things, on the prompt bars where both versions of
the same chord exist:

  1. DIFF    frame by frame: pitch-class set, bass pitch class, absolute
             pitch set, onset frames -- and prints where they disagree.
  2. FIT     the map f with  ours = f(theirs)  as a small family of
             re-voicings: bass to one octave base, the remaining pitch
             classes to another, the bass class folded into the upper
             voicing or not, contiguous identical segments merged or
             not. Every candidate is scored by exact frame agreement
             with our prompt chord and the best is reported, with its
             residual. Our own renderer puts the bass at 36+pc and every
             chord tone at 60+pc (build_pop909_chord_midi.py), so the
             expected winner is (36, 60, fold); the fit says whether
             the data agrees, and how closely.
  3. APPLY   (--apply NAME) write every output of the system with its
             chord track replaced by f(chord), melody and tempo
             untouched, as system NAME in the same tree, so the scorer
             can read it like any other duet_multi system. The prompt
             bars come out equal to ours wherever the fit was exact,
             and the continuation is in our rendering.

What f cannot repair is content: a pitch class the baseline dropped or
added in the prompt bars is a real change, and the DIFF section lists
those frames so they can be counted, not hidden.

Usage (via wholesong_chord_map.sbatch):
  python wholesong_chord_map.py --out-root temp/E1_v5b --system WSf \
      --chord-dir temp/E1_v5b/prompts/chord [--apply WSfv]
"""

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import frame_fn, BEAT_DIV     # noqa: E402

PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ---------------------------------------------------------------------------
# notes <-> frames
# ---------------------------------------------------------------------------
def notes_to_frames(pm, notes):
    """[(f0, f1, pitch)] on the file's tick grid, f1 exclusive, f1 > f0."""
    to_f = frame_fn(pm)
    out = []
    for n in notes:
        f0 = int(round(to_f(n.start)))
        f1 = max(f0 + 1, int(round(to_f(n.end))))
        out.append((f0, f1, n.pitch))
    return sorted(out)


def chord_instrument(pm):
    """The CHORD track of a combined file: by name, else program 48."""
    for inst in pm.instruments:
        if (inst.name or '').strip().lower() == 'chord':
            return inst
    for inst in pm.instruments:
        if inst.program == 48 and not inst.is_drum:
            return inst
    return None


def load_ours(path):
    pm = pretty_midi.PrettyMIDI(path)
    notes = [n for inst in pm.instruments for n in inst.notes]
    return notes_to_frames(pm, notes)


def load_theirs(path):
    pm = pretty_midi.PrettyMIDI(path)
    inst = chord_instrument(pm)
    if inst is None:
        raise ValueError(f'{path}: no CHORD track')
    return pm, inst, notes_to_frames(pm, inst.notes)


def window(notes, P):
    return [(f0, f1, p) for f0, f1, p in notes if f0 < P]


def segments(notes):
    """Notes grouped by onset frame: {f0: [(f1, pitch), ...]}, sorted."""
    seg = defaultdict(list)
    for f0, f1, p in notes:
        seg[f0].append((f1, p))
    return dict(sorted(seg.items()))


def frame_sets(notes, P):
    """Per frame: absolute pitch set, pitch-class set, bass pc (or None)."""
    abs_ = [set() for _ in range(P)]
    for f0, f1, p in notes:
        for f in range(max(f0, 0), min(f1, P)):
            abs_[f].add(p)
    pcs = [{p % 12 for p in s} for s in abs_]
    bass = [(min(s) % 12) if s else None for s in abs_]
    return abs_, pcs, bass


def onset_frames(notes, P):
    return {f0 for f0, _f1, _p in notes if 0 <= f0 < P}


def pcname(pcs):
    return '.'.join(PC_NAMES[p] for p in sorted(pcs)) or '-'


# ---------------------------------------------------------------------------
# the map family
# ---------------------------------------------------------------------------
def cap_voices(notes, max_voices):
    """Keep the LOWEST max_voices pitches of every segment.

    This is what our tokenizer does to a chord with more notes than
    its polyphony budget: the renderer writes bass then chord tones
    ascending, the tokenizer keeps the first `budget` it meets, so the
    top tone goes. Applying the same cut to the baseline's chord puts
    it in the representation our cp4 arms actually live in.
    """
    if not max_voices:
        return notes
    out = []
    for f0, members in segments(notes).items():
        keep = sorted(members, key=lambda m: m[1])[:max_voices]
        out.extend((f0, f1, p) for f1, p in keep)
    return sorted(out)


def revoice(notes, bass_base, upper_base, fold, merge):
    """f: re-render each chord segment of `notes` in the given convention.

    A segment is the set of notes sharing an onset frame; its pitch
    classes and its bass (lowest pitch) are what the baseline's image
    carried, so they are what the map keeps. Duration is the segment's
    longest note. With `merge`, a segment that repeats the previous one
    exactly and starts where it ended is absorbed into it.
    """
    seg = segments(notes)
    out_segs = []
    for f0, members in seg.items():
        pcs = frozenset(p % 12 for _f1, p in members)
        bass_pc = min(p for _f1, p in members) % 12
        f1 = max(f1_ for f1_, _p in members)
        if merge and out_segs:
            pf0, pf1, ppcs, pbass = out_segs[-1]
            if ppcs == pcs and pbass == bass_pc and pf1 == f0:
                out_segs[-1] = (pf0, f1, ppcs, pbass)
                continue
        out_segs.append((f0, f1, pcs, bass_pc))
    out = []
    for f0, f1, pcs, bass_pc in out_segs:
        upper = set(pcs) if fold else set(pcs) - {bass_pc}
        out.append((f0, f1, bass_base + bass_pc))
        out.extend((f0, f1, upper_base + pc) for pc in sorted(upper))
    return sorted(out)


def agreement(a, b, P):
    """Exact absolute-pitch frame agreement, pc agreement, onset Jaccard."""
    aa, ap, _ = frame_sets(a, P)
    ba, bp, _ = frame_sets(b, P)
    tot = hit_abs = hit_pc = 0
    for f in range(P):
        if not aa[f] and not ba[f]:
            continue
        tot += 1
        hit_abs += aa[f] == ba[f]
        hit_pc += ap[f] == bp[f]
    oa, ob = onset_frames(a, P), onset_frames(b, P)
    jac = len(oa & ob) / len(oa | ob) if (oa | ob) else 1.0
    return (hit_abs / tot if tot else 1.0, hit_pc / tot if tot else 1.0, jac)


def octave_bases(notes):
    """(bass base counter, upper base counter): pitch - pc per role."""
    bass_c, upper_c = Counter(), Counter()
    for _f0, members in segments(notes).items():
        pitches = sorted(p for _f1, p in members)
        bass_c[pitches[0] - pitches[0] % 12] += 1
        for p in pitches[1:]:
            upper_c[p - p % 12] += 1
    return bass_c, upper_c


def fold_rate(notes):
    """Share of segments whose bass pitch class also appears above it."""
    n = hit = 0
    for _f0, members in segments(notes).items():
        pitches = sorted(p for _f1, p in members)
        if len(pitches) < 2:
            continue
        n += 1
        hit += (pitches[0] % 12) in {p % 12 for p in pitches[1:]}
    return hit / n if n else float('nan')


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def find_outputs(root, system, song):
    return sorted(glob.glob(os.path.join(root, system, song, 'co',
                                         'sample_*.mid')))


def write_applied(pm, inst, new_notes, path):
    """Replace `inst`'s notes with new_notes (frames) and write pm to path."""
    tpf = pm.resolution / BEAT_DIV
    vel = int(np.median([n.velocity for n in inst.notes])) if inst.notes else 80
    inst.notes = [
        pretty_midi.Note(velocity=vel, pitch=int(p),
                         start=pm.tick_to_time(int(round(f0 * tpf))),
                         end=pm.tick_to_time(int(round(f1 * tpf))))
        for f0, f1, p in new_notes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pm.write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--system', default='WSf')
    ap.add_argument('--chord-dir', required=True,
                    help='OUR staged chord prompts, <song>.mid')
    ap.add_argument('--prompt-frames', type=int, default=96)
    ap.add_argument('--song-ids', nargs='*', default=None)
    ap.add_argument('--exclude-songs', nargs='*', default=[])
    ap.add_argument('--show', type=int, default=4,
                    help='songs with the most residual frames to print')
    ap.add_argument('--apply', default='',
                    help='write f(output) for EVERY sample of the system '
                         'as this system name in the same tree')
    ap.add_argument('--max-voices', type=int, default=0,
                    help='after the map, keep only the LOWEST n pitches of '
                         'each chord segment -- the cut our cp4 tokenizer '
                         'makes (bass + 3 tones; the top tone of a seventh '
                         'chord goes). 0 = no cap. Applied in APPLY and, '
                         'for the check, to OUR prompt as well.')
    ap.add_argument('--force-map', default='',
                    help='use this map instead of the fitted one: '
                         'bass_base,upper_base,fold,merge e.g. 36,60,1,0')
    args = ap.parse_args()
    P = args.prompt_frames

    if args.song_ids:
        songs = list(args.song_ids)
    else:
        songs = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(args.chord_dir)
                       if f.lower().endswith('.mid'))
    excl = {s.zfill(3) for s in args.exclude_songs}
    songs = [s for s in songs if s.zfill(3) not in excl]

    pairs = []          # (song, ours, theirs, pm, inst, path)
    missing = []
    for s in songs:
        outs = find_outputs(args.out_root, args.system, s)
        if not outs:
            missing.append(s)
            continue
        ours = window(cap_voices(load_ours(os.path.join(args.chord_dir,
                                                        f'{s}.mid')),
                                 args.max_voices), P)
        pm, inst, theirs_all = load_theirs(outs[0])
        pairs.append((s, ours, window(theirs_all, P), pm, inst, outs[0]))
    print(f'system {args.system}: {len(pairs)} songs with output, '
          f'{len(missing)} without'
          + (f' ({" ".join(missing)})' if missing else ''))
    if not pairs:
        raise SystemExit('nothing to compare')

    # ---- 1. DIFF: what differs, before any map ------------------------
    print('\n=== 1. DIFF, raw, over the prompt bars ===')
    raw = np.array([agreement(o, t, P) for _s, o, t, *_ in pairs])
    print(f'  frame agreement   abs pitch {raw[:, 0].mean():.3f}   '
          f'pitch class {raw[:, 1].mean():.3f}   onset Jaccard '
          f'{raw[:, 2].mean():.3f}')
    ob, ou = Counter(), Counter()
    tb, tu = Counter(), Counter()
    for _s, o, t, *_ in pairs:
        b, u = octave_bases(o); ob += b; ou += u
        b, u = octave_bases(t); tb += b; tu += u
    def top(c):
        return ', '.join(f'{k}({v})' for k, v in c.most_common(3)) or '-'
    print(f'  OURS   bass octave base {top(ob)}   upper {top(ou)}   '
          f'bass pc folded into upper: '
          f'{np.nanmean([fold_rate(o) for _s, o, *_ in pairs]):.2f}')
    print(f'  THEIRS bass octave base {top(tb)}   upper {top(tu)}   '
          f'bass pc folded into upper: '
          f'{np.nanmean([fold_rate(t) for _s, _o, t, *_ in pairs]):.2f}')
    n_seg_o = np.mean([len(segments(o)) for _s, o, *_ in pairs])
    n_seg_t = np.mean([len(segments(t)) for _s, _o, t, *_ in pairs])
    print(f'  chord segments per prompt: ours {n_seg_o:.1f}, theirs '
          f'{n_seg_t:.1f}')
    # content: pitch-class disagreement frames, the part no map fixes
    bad_pc = []
    for s, o, t, *_ in pairs:
        _, po, _ = frame_sets(o, P)
        _, pt, _ = frame_sets(t, P)
        frames = [f for f in range(P) if (po[f] or pt[f]) and po[f] != pt[f]]
        bad_pc.append((len(frames), s, frames, po, pt))
    n_bad = sum(b[0] for b in bad_pc)
    print(f'  pitch-class CONTENT differs on {n_bad} frames over '
          f'{len(pairs)} songs ({n_bad / len(pairs):.1f} per song of {P})')
    # WHERE in the prompt: a difference confined to the last bar means
    # the baseline was prompted with one bar less and generated it
    by_bar = Counter(f // 16 for _n, _s, frames, _po, _pt in bad_pc
                     for f in frames)
    print('  differing frames by bar: '
          + '  '.join(f'bar {b + 1}: {by_bar.get(b, 0)}'
                      for b in range((P + 15) // 16)))
    per_song_last = sum(1 for _n, _s, frames, _po, _pt in bad_pc
                        if any(f >= P - 16 for f in frames))
    print(f'  songs whose LAST prompt bar differs: {per_song_last}/{len(pairs)}')
    for n, s, frames, po, pt in sorted(bad_pc, reverse=True)[:args.show]:
        if n == 0:
            break
        print(f'    song {s}: {n} frames -- first few:')
        for f in frames[:6]:
            print(f'      frame {f:3d} (bar {f // 16 + 1} beat '
                  f'{(f % 16) // 4 + 1}): ours {pcname(po[f]):16s} '
                  f'theirs {pcname(pt[f])}')

    # ---- 2. FIT f -------------------------------------------------------
    print('\n=== 2. FIT  ours = f(theirs) ===')
    bass_cands = sorted({k for k, _ in ob.most_common(2)} | {36, 48})
    upper_cands = sorted({k for k, _ in ou.most_common(2)} | {48, 60})
    results = []
    for bb in bass_cands:
        for ub in upper_cands:
            for fold in (True, False):
                for merge in (False, True):
                    sc = np.array([agreement(o, cap_voices(
                        revoice(t, bb, ub, fold, merge), args.max_voices), P)
                                   for _s, o, t, *_ in pairs])
                    results.append((sc[:, 0].mean(), sc[:, 1].mean(),
                                    sc[:, 2].mean(), bb, ub, fold, merge))
    results.sort(key=lambda r: (-r[0], -r[2]))
    print('  abs    pc     onset   bass  upper  fold  merge')
    for r in results[:6]:
        print(f'  {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}   {r[3]:3d}   {r[4]:3d}'
              f'   {int(r[5])}     {int(r[6])}')
    if args.force_map:
        bb, ub, fold, merge = (int(x) for x in args.force_map.split(','))
        fold, merge = bool(fold), bool(merge)
        print(f'  using FORCED map bass {bb} upper {ub} fold {int(fold)} '
              f'merge {int(merge)}')
    else:
        _a, _p, _o, bb, ub, fold, merge = results[0]
        print(f'  best map: bass -> {bb}+pc, upper -> {ub}+pc, '
              f'{"bass folded into upper" if fold else "bass not duplicated"}'
              f', {"contiguous repeats merged" if merge else "segments kept"}')
    best_abs = results[0][0]
    if best_abs > 0.999:
        print('  the map reproduces our prompt chord EXACTLY: the '
              'difference was rendering only')
    elif best_abs > 0.95:
        print('  the map reproduces our prompt chord on '
              f'{100 * best_abs:.1f}% of frames; the residual below is '
              'content or segmentation the map does not cover')
    else:
        print(f'  WARNING: best map agrees on only {100 * best_abs:.1f}% '
              'of frames -- the difference is not a voicing convention. '
              'Read the DIFF section.')

    # residual after the map
    resid = []
    for s, o, t, *_ in pairs:
        ft = cap_voices(revoice(t, bb, ub, fold, merge), args.max_voices)
        ao, _, _ = frame_sets(o, P)
        at, _, _ = frame_sets(ft, P)
        frames = [f for f in range(P) if (ao[f] or at[f]) and ao[f] != at[f]]
        resid.append((len(frames), s, frames, ao, at))
    n_res = sum(r[0] for r in resid)
    print(f'  residual after f: {n_res} frames over {len(pairs)} songs; '
          f'songs fully reproduced: {sum(1 for r in resid if r[0] == 0)}'
          f'/{len(pairs)}')
    for n, s, frames, ao, at in sorted(resid, reverse=True)[:args.show]:
        if n == 0:
            break
        print(f'    song {s}: {n} frames -- first few:')
        for f in frames[:6]:
            print(f'      frame {f:3d}: ours {sorted(ao[f])}  '
                  f'f(theirs) {sorted(at[f])}')

    # ---- 3. APPLY --------------------------------------------------------
    if args.apply:
        print(f'\n=== 3. APPLY f to every sample of {args.system} '
              f'-> {args.apply} ===')
        n_files = 0
        for s in songs:
            for src in find_outputs(args.out_root, args.system, s):
                pm, inst, theirs_all = load_theirs(src)
                new = cap_voices(revoice(theirs_all, bb, ub, fold, merge),
                                 args.max_voices)
                dst = os.path.join(args.out_root, args.apply, s, 'co',
                                   os.path.basename(src))
                write_applied(pm, inst, new, dst)
                n_files += 1
        print(f'  wrote {n_files} files under {args.out_root}/{args.apply}/')
        print(f'  map: bass -> {bb}+pc, upper -> {ub}+pc, fold={int(fold)}, '
              f'merge={int(merge)}; melody and tempo untouched'
              + (f'; chords capped to the lowest {args.max_voices} voices'
                 if args.max_voices else ''))


if __name__ == '__main__':
    main()
