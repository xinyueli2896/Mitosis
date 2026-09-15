"""Does a system's output carry OUR prompt in its prompt bars?

Every E1 system is prompted with the same staged crops, and the scored
window starts after them -- but what a system writes INTO the prompt
bars of its own output is its business, and for an external baseline
it is where the prompt actually got altered on the way in. This tool
loads the staged prompt and each system's output over frames
[0, PROMPT_FRAMES) and reports, per stream, how far the two agree:

  onset     Jaccard of the frames that carry an onset -- rhythm
  pc        share of sounding frames whose PITCH-CLASS set matches
  pitch     share of sounding frames whose absolute-pitch set matches
  shift     the frame offset that best aligns the onsets (0 = none)
  notes     onset counts, ours vs theirs

and a one-line verdict per system. The one that matters for the
whole-song baseline: high `pc` with low `pitch` on the chord stream
means the pitch classes came through and only the VOICING changed --
its lead-sheet image stores a chord as a chroma band and a bass band,
one row per pitch class, and its own converter writes those back out
at its own octave. Low `pc`, or a non-zero shift, is a real mismatch.

Usage (via check_prompt_copy.sbatch):
  python check_prompt_copy.py --out-root temp/E1_v5b --systems WSf A3ctcaT \
      --mel-dir temp/E1_v5b/prompts/mel --chord-dir temp/E1_v5b/prompts/chord
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import load_streams          # noqa: E402


def find_output(root, system, song, sample):
    """Path of one output file under the three E1 layouts, or None."""
    # duet outputs carry a temperature suffix (sample_0_temp0.9.mid),
    # the cascades and external arms do not (sample_0.mid): glob both
    for sub in ((song, 'co'), ('4_final', song, 'co')):
        hits = sorted(glob.glob(os.path.join(root, system, *sub,
                                             f'sample_{sample}*.mid')))
        if hits:
            return hits[0]
    single = sorted(glob.glob(
        os.path.join(root, system, f'{song}.mid_temp*_continuation_*.mid')))
    if len(single) > sample:
        return single[sample]
    return None


def _onset_set(s):
    return {f for f, ps in enumerate(s.onsets) if ps}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def best_shift(ours, theirs, max_shift=8):
    """Frame offset k maximising onset overlap after shifting THEIRS by k."""
    a = _onset_set(ours)
    b = _onset_set(theirs)
    if not a or not b:
        return 0, 0.0
    best = (0, -1.0)
    for k in range(-max_shift, max_shift + 1):
        j = _jaccard(a, {f + k for f in b})
        if j > best[1] + 1e-9:
            best = (k, j)
    return best


def compare(ours, theirs):
    n = ours.n_frames
    on = _jaccard(_onset_set(ours), _onset_set(theirs))
    pc_hit = pc_tot = ab_hit = 0
    for f in range(n):
        so, st = ours.sounding[f], theirs.sounding[f]
        if not so and not st:
            continue
        pc_tot += 1
        if so == st:
            pc_hit += 1
        if ours.sounding_abs[f] == theirs.sounding_abs[f]:
            ab_hit += 1
    pc = pc_hit / pc_tot if pc_tot else 1.0
    ab = ab_hit / pc_tot if pc_tot else 1.0
    k, _ = best_shift(ours, theirs)
    return dict(onset=on, pc=pc, pitch=ab, shift=k,
                n_ours=ours.n_onsets(), n_theirs=theirs.n_onsets())


def note_diff(ours, theirs):
    """Note-level differences: what THEIRS has that OURS does not, and
    the reverse, as (frame, pitch, role, dup) with role = where the
    extra pitch sits in that frame's chord (highest / lowest / inner /
    alone) and dup = it doubles a pitch our chord already holds at that
    frame (a set comparison would never see that). Onsets are compared
    as multisets per frame, so a duplicated note counts."""
    from collections import Counter
    extra, missing = [], []
    for f in range(ours.n_frames):
        co, ct = Counter(ours.onsets[f]), Counter(theirs.onsets[f])
        for pitch, k in (ct - co).items():
            snd = theirs.sounding_abs[f]
            role = ('alone' if len(snd) <= 1 else 'highest'
                    if pitch >= max(snd) else 'lowest'
                    if pitch <= min(snd) else 'inner')
            extra.extend([(f, pitch, role, pitch in co)] * k)
        for pitch, k in (co - ct).items():
            snd = ours.sounding_abs[f]
            role = ('alone' if len(snd) <= 1 else 'highest'
                    if pitch >= max(snd) else 'lowest'
                    if pitch <= min(snd) else 'inner')
            missing.extend([(f, pitch, role, pitch in ct)] * k)
    return extra, missing


def capped(stream, n):
    """The stream with only the lowest n pitches of each onset frame."""
    from eval_metrics import Stream
    out = Stream([], 1.0, stream.n_frames)
    for f in range(stream.n_frames):
        keep = sorted(zip(stream.onsets[f], stream.dur_at[f]))[:n]
        for pitch, d in keep:
            out.onsets[f].append(pitch)
            out.dur_at[f].append(d)
            out.durations.append(d)
            for g in range(f, min(f + d, stream.n_frames)):
                out.sounding[g].add(pitch % 12)
                out.sounding_abs[g].add(pitch)
    return out


def pc_name(pcs):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return '.'.join(names[p] for p in sorted(pcs)) or '-'


def show_bars(ours, theirs, bars=2, per_bar=16, step=4):
    """Beat-by-beat pitch-class sets of the chord stream, ours vs theirs."""
    for f in range(0, min(bars * per_bar, ours.n_frames), step):
        o = pc_name(ours.sounding[f])
        t = pc_name(theirs.sounding[f])
        flag = '' if ours.sounding[f] == theirs.sounding[f] else '   <-- differs'
        print(f'      frame {f:3d}  ours {o:18s} theirs {t:18s}{flag}')


def verdict(m_mel, m_chd):
    """One line per system from the two streams' mean figures."""
    lines = []
    for name, m in (('melody', m_mel), ('chord', m_chd)):
        if m['pc'] >= 0.95 and m['onset'] >= 0.9 and m['pitch'] >= 0.95:
            lines.append(f'{name}: copied faithfully')
        elif m['pc'] >= 0.9 and m['pitch'] < 0.7:
            lines.append(f'{name}: SAME pitch classes, DIFFERENT voicing/'
                         f'octave (re-rendered, not copied)')
        elif abs(m['shift']) >= 1 and m['onset'] < 0.8:
            lines.append(f'{name}: TIMING OFFSET of {m["shift"]:+d} frames')
        elif m['onset'] < 0.8 and m['pc'] >= 0.9:
            lines.append(f'{name}: pitch classes match but the RHYTHM '
                         f'differs (durations/segmentation changed)')
        else:
            lines.append(f'{name}: CONTENT DIFFERS (pc agreement '
                         f'{m["pc"]:.2f}, onset {m["onset"]:.2f})')
    return '; '.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', required=True)
    p.add_argument('--systems', nargs='+', required=True)
    p.add_argument('--mel-dir', required=True)
    p.add_argument('--chord-dir', required=True)
    p.add_argument('--prompt-frames', type=int, default=96)
    p.add_argument('--sample', type=int, nargs='+', default=[0],
                   help='sample index(es) to read; each song x sample is a row')
    p.add_argument('--song-ids', nargs='*', default=None,
                   help='default: every staged song in --mel-dir')
    p.add_argument('--exclude-songs', nargs='*', default=[])
    p.add_argument('--show', type=int, default=3,
                   help='print beat-by-beat chord detail for the N songs '
                        'with the lowest chord agreement')
    p.add_argument('--max-voices', type=int, default=0,
                   help='cap OUR chord prompt to its lowest n notes per '
                        'onset before comparing -- the cut the cp4 '
                        'tokenizer makes -- so a system living under that '
                        'budget is checked against what it could hold')
    p.add_argument('--task', default='melchord')
    p.add_argument('--mel-programs', default='0,24')
    p.add_argument('--chord-programs', default='48')
    args = p.parse_args()
    mel_programs = {int(x) for x in args.mel_programs.split(',')}
    chord_programs = {int(x) for x in args.chord_programs.split(',')}

    if args.song_ids:
        songs = list(args.song_ids)
    else:
        songs = sorted(os.path.splitext(f)[0] for f in os.listdir(args.mel_dir)
                       if f.lower().endswith('.mid'))
    excl = {s.zfill(3) for s in args.exclude_songs}
    songs = [s for s in songs if s.zfill(3) not in excl]
    P = args.prompt_frames
    print(f'prompt window: frames [0, {P})   songs: {len(songs)}   '
          f'sample: {args.sample}')

    prompts = {}
    for s in songs:
        pm_ = os.path.join(args.mel_dir, f'{s}.mid')
        pc_ = os.path.join(args.chord_dir, f'{s}.mid')
        if not (os.path.exists(pm_) and os.path.exists(pc_)):
            print(f'  [skip] {s}: staged prompt missing')
            continue
        pa_, pb_ = load_streams([pm_, pc_], args.task, mel_programs,
                                chord_programs, P)
        if args.max_voices:
            pb_ = capped(pb_, args.max_voices)
        prompts[s] = (pa_, pb_)

    for system in args.systems:
        rows = []
        missing = 0
        for s, (pa, pb) in prompts.items():
            for k in args.sample:
                f = find_output(args.out_root, system, s, k)
                if f is None:
                    missing += 1
                    continue
                ga, gb = load_streams([f], args.task, mel_programs,
                                      chord_programs, P)
                tag = s if len(args.sample) == 1 else f'{s}/s{k}'
                rows.append((tag, compare(pa, ga), compare(pb, gb), (pb, gb),
                             (pa, ga), f))
        print('\n' + '=' * 72)
        print(f'{system}: {len(rows)} song-sample(s) compared, {missing} '
              f'without output')
        if not rows:
            continue
        keys = ('onset', 'pc', 'pitch', 'shift')
        mean = {}
        for st_i, st in ((1, 'melody'), (2, 'chord')):
            m = {k: float(np.mean([r[st_i][k] for r in rows])) for k in keys}
            m['shift'] = int(round(np.median([r[st_i]['shift'] for r in rows])))
            n_o = np.mean([r[st_i]['n_ours'] for r in rows])
            n_t = np.mean([r[st_i]['n_theirs'] for r in rows])
            mean[st] = m
            print(f'  {st:7s} onset {m["onset"]:.3f}  pc {m["pc"]:.3f}  '
                  f'pitch {m["pitch"]:.3f}  shift {m["shift"]:+d}  '
                  f'notes ours {n_o:.1f} / theirs {n_t:.1f}')
        print(f'  verdict: {verdict(mean["melody"], mean["chord"])}')
        worst = sorted(rows, key=lambda r: r[2]['pc'])[:args.show]
        if args.show and worst and worst[0][2]['pc'] < 0.999:
            print(f'  lowest chord agreement, beat by beat, first 2 bars:')
            for s, _mm, mc, (pb, gb), _mel, _f in worst:
                print(f'    song {s}: chord pc {mc["pc"]:.2f} pitch '
                      f'{mc["pitch"]:.2f} onset {mc["onset"]:.2f} '
                      f'shift {mc["shift"]:+d}')
                show_bars(pb, gb)

        # ---- note-level: extra and missing NOTES, by where they sit ----
        # A frame whose pitch SET matches can still differ in notes: a
        # doubled pitch, or a note that starts and stops inside a frame.
        # And where the set differs by one pitch, this says which one --
        # the top of the chord, the bottom, or an inner voice.
        from collections import Counter
        for st, idx in (('chord', 3), ('melody', 4)):
            ex_all, mi_all = [], []
            for r in rows:
                o, t = r[idx]
                ex, mi = note_diff(o, t)
                ex_all.extend((r[0], *e) for e in ex)
                mi_all.extend((r[0], *m) for m in mi)
            n_tot = sum(r[idx][0].n_onsets() for r in rows)
            print(f'  {st}: {len(ex_all)} extra note(s) in theirs, '
                  f'{len(mi_all)} missing, over {n_tot} of our notes')
            if ex_all:
                roles = Counter(e[3] for e in ex_all)
                dups = sum(1 for e in ex_all if e[4])
                songs_hit = len({e[0] for e in ex_all})
                print(f'    extra by position: '
                      + ', '.join(f'{k} {v}' for k, v in roles.most_common())
                      + f'; doubling a pitch we already hold: {dups}; '
                        f'songs affected: {songs_hit}')
                for sng, fr, pitch, role, dup in ex_all[:args.show * 3]:
                    print(f'      song {sng} frame {fr:3d} (bar {fr // 16 + 1} '
                          f'beat {(fr % 16) // 4 + 1}) pitch {pitch} '
                          f'{role}{" (duplicate)" if dup else ""}')
            if mi_all:
                roles = Counter(m[3] for m in mi_all)
                print(f'    missing by position: '
                      + ', '.join(f'{k} {v}' for k, v in roles.most_common()))
                for sng, fr, pitch, role, dup in mi_all[:args.show * 3]:
                    print(f'      song {sng} frame {fr:3d} pitch {pitch} {role}')


if __name__ == '__main__':
    main()
