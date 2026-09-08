"""Are the E1 reference files what the scorer thinks they are?

The 14-song E1 table shows every system -- S1 and WS included -- at
harmonic_rhythm_jsd ~0.8, duration_jsd_b ~0.83 and density_ratio_a
~5-7 against the reference, with most H3 rows at n=8-9 of 14. On the
5-song set the same rows were 0.08, 0.04, ~1 and n=5. Six models
cannot all be five times denser than the truth; the reference files
(13 copied from POP909-Dataset/POP909-{melody,chord}, plus 004 from the
old test split) are the suspects.

For every file in the given folders this prints what eval_metrics
would compute from it: the tempo it frames by, tick resolution, note
count, how many notes fall inside the scored window (frames
prompt..total at BEAT_DIV frames per beat of THAT tempo), and how far
the file extends in frames. A reference whose window is empty or that
ends before the window gives exactly the symptoms above.

With --metrics it also prints, per song, a few reference-dependent and
reference-free rows for two systems from the E1 metrics CSV, so song
004 (old-split file) can be compared with the other 13 directly.
"""
import argparse
import csv
import glob
import os
import statistics

import pretty_midi

from eval_metrics import _file_tempo, frame_fn, BEAT_DIV, FRAMES_PER_BAR


def describe(path, prompt, total):
    pm = pretty_midi.PrettyMIDI(path)
    tempo = _file_tempo(pm)
    _, tempi = pm.get_tempo_changes()
    step = 60.0 / tempo / BEAT_DIV
    notes = [n for inst in pm.instruments for n in inst.notes]
    end_frame = max((n.end for n in notes), default=0.0) / step
    in_win = sum(1 for n in notes if prompt * step <= n.start < total * step)
    in_prompt = sum(1 for n in notes if n.start < prompt * step)
    # the same counts on the TICK grid (what the tokenizer and, now,
    # the scorer use); a large gap between the two columns is the
    # spurious-initial-tempo signature.
    fr = frame_fn(pm)
    tick_win = sum(1 for n in notes if prompt <= fr(n.start) < total)
    tick_prompt = sum(1 for n in notes if fr(n.start) < prompt)
    res = getattr(pm, 'resolution', None)
    return dict(file=os.path.basename(path), tempo=tempo, n_tempi=len(tempi),
                res=res, n_notes=len(notes), n_prompt=in_prompt, n_win=in_win,
                tick_prompt=tick_prompt, tick_win=tick_win,
                end_frame=end_frame, n_ins=len(pm.instruments),
                names=[(i.name or '')[:8] for i in pm.instruments])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', action='append', required=True)
    p.add_argument('--prompt', type=int, default=96)
    p.add_argument('--total', type=int, default=416)
    p.add_argument('--metrics', default=None, help='E1 metrics CSV')
    p.add_argument('--systems', nargs='+', default=['A3', 'S1'])
    args = p.parse_args()

    for folder in args.folder:
        files = sorted(glob.glob(os.path.join(folder, '*.mid'))
                       + glob.glob(os.path.join(folder, '*.MID')))
        print(f'\n--- {folder}  ({len(files)} files)  window = frames '
              f'[{args.prompt}, {args.total}) at {BEAT_DIV}/beat, '
              f'{FRAMES_PER_BAR}/bar ---')
        print(f'{"file":<10}{"tempo":>7}{"#tempi":>7}{"res":>6}{"ins":>4}'
              f'{"notes":>7}{"prompt":>7}{"window":>7}{"end_fr":>8}'
              f'{"|tick:pr":>9}{"win":>6}  names')
        for f in files:
            try:
                d = describe(f, args.prompt, args.total)
            except Exception as e:
                print(f'{os.path.basename(f):<10} READ FAILED {e!r}')
                continue
            flag = ''
            if d['n_win'] == 0:
                flag = '  <- EMPTY WINDOW'
            elif d['end_frame'] < args.total:
                flag = f'  <- ends before frame {args.total}'
            print(f'{d["file"]:<10}{d["tempo"]:>7.1f}{d["n_tempi"]:>7}'
                  f'{str(d["res"]):>6}{d["n_ins"]:>4}{d["n_notes"]:>7}'
                  f'{d["n_prompt"]:>7}{d["n_win"]:>7}{d["end_frame"]:>8.0f}'
                  f'{d["tick_prompt"]:>9}{d["tick_win"]:>6}'
                  f'  {d["names"]}{flag}')

    if args.metrics and os.path.isfile(args.metrics):
        with open(args.metrics) as fh:
            rows = list(csv.DictReader(fh))
        cols = rows[0].keys() if rows else []
        sys_col = next((c for c in cols if c.lower() in ('system', 'model')), None)
        song_col = next((c for c in cols if c.lower() in ('song', 'song_id')), None)
        want = [c for c in ('density_ratio_a', 'harmonic_rhythm_jsd',
                            'duration_jsd_b', 'survival_min', 'coupling',
                            'chord_tone_cov') if c in cols]
        if not (sys_col and song_col and want):
            print(f'\n[metrics] could not find system/song columns in {list(cols)[:12]}')
            return
        print(f'\n--- per-song rows from {args.metrics} (mean over samples) ---')
        print(f'{"system":<10}{"song":<6}' + ''.join(f'{c:>20}' for c in want))
        for system in args.systems:
            by_song = {}
            for r in rows:
                if r[sys_col] != system:
                    continue
                by_song.setdefault(r[song_col], []).append(r)
            for song in sorted(by_song):
                vals = []
                for c in want:
                    xs = []
                    for r in by_song[song]:
                        try:
                            xs.append(float(r[c]))
                        except (ValueError, TypeError):
                            pass
                    vals.append(f'{statistics.mean(xs):>20.3f}' if xs else f'{"nan":>20}')
                print(f'{system:<10}{song:<6}' + ''.join(vals))


if __name__ == '__main__':
    main()
