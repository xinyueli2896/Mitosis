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
    cands = [
        os.path.join(root, system, song, 'co', f'sample_{sample}.mid'),
        os.path.join(root, system, '4_final', song, 'co',
                     f'sample_{sample}.mid'),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
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
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--song-ids', nargs='*', default=None,
                   help='default: every staged song in --mel-dir')
    p.add_argument('--exclude-songs', nargs='*', default=[])
    p.add_argument('--show', type=int, default=3,
                   help='print beat-by-beat chord detail for the N songs '
                        'with the lowest chord agreement')
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
        prompts[s] = load_streams([pm_, pc_], args.task, mel_programs,
                                  chord_programs, P)

    for system in args.systems:
        rows = []
        missing = 0
        for s, (pa, pb) in prompts.items():
            f = find_output(args.out_root, system, s, args.sample)
            if f is None:
                missing += 1
                continue
            ga, gb = load_streams([f], args.task, mel_programs,
                                  chord_programs, P)
            rows.append((s, compare(pa, ga), compare(pb, gb), (pb, gb)))
        print('\n' + '=' * 72)
        print(f'{system}: {len(rows)} songs compared, {missing} without output')
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
            for s, _mm, mc, (pb, gb) in worst:
                print(f'    song {s}: chord pc {mc["pc"]:.2f} pitch '
                      f'{mc["pitch"]:.2f} onset {mc["onset"]:.2f} '
                      f'shift {mc["shift"]:+d}')
                show_bars(pb, gb)


if __name__ == '__main__':
    main()
