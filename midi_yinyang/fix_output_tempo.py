"""Rewrite the playback tempo of generated midis that inherited a
spurious ~990 bpm from a beat-synced prompt, keeping every tick.

The E1 outputs for the time-aligned POP909 songs were written at the
prompt's FIRST tempo event, the ~0.06 s lead-in (~990 bpm). Ticks are
what the tokenizer and the scorer read, so the numbers are unaffected;
only playback is. This walks the given roots, and for every .mid whose
tempo map is a single event above --max-bpm it replaces that event with
the main tempo of the matching prompt (midi_tempo.main_tempo of
<prompts>/<song id>.mid), or --bpm if no prompt matches. Files with a
sane tempo, or with a real tempo map (more than one event), are left
alone -- the staged prompts and references are never touched because
they live under <root>/prompts, which is skipped.

Usage (via fix_output_tempo.sbatch):
    python fix_output_tempo.py --root temp/E1_matched_p6 --root temp/E1_matched_p8
"""
import argparse
import os
import re

import mido

from midi_tempo import main_tempo


def song_id(relpath):
    """First 3-digit id in the path (outputs live under <system>/<song>/...)."""
    m = re.search(r'(?<!\d)(\d{3})(?!\d)', relpath)
    return m.group(1) if m else None


def tempo_events(m):
    out, t = [], 0
    for msg in mido.merge_tracks(m.tracks):
        t += msg.time
        if msg.type == 'set_tempo':
            out.append((t, msg.tempo))
    return out


def rewrite(path, bpm):
    m = mido.MidiFile(path)
    new_tempo = mido.bpm2tempo(bpm)
    for tr in m.tracks:
        for i, msg in enumerate(tr):
            if msg.type == 'set_tempo':
                tr[i] = msg.copy(tempo=new_tempo)
    m.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', action='append', required=True)
    p.add_argument('--max-bpm', type=float, default=300.0)
    p.add_argument('--bpm', type=float, default=120.0, help='fallback')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    n_fixed = n_seen = 0
    for root in args.root:
        prompts = os.path.join(root, 'prompts', 'mel')
        cache = {}
        for dirpath, dirs, files in os.walk(root):
            if os.path.abspath(dirpath).startswith(os.path.abspath(os.path.join(root, 'prompts'))):
                dirs[:] = []
                continue
            for f in files:
                if not f.lower().endswith('.mid'):
                    continue
                fn = os.path.join(dirpath, f)
                n_seen += 1
                try:
                    ev = tempo_events(mido.MidiFile(fn))
                except Exception as e:      # noqa: BLE001
                    print(f'[skip] {fn}: {e!r}')
                    continue
                if len(ev) != 1 or mido.tempo2bpm(ev[0][1]) <= args.max_bpm:
                    continue
                sid = song_id(os.path.relpath(fn, root))
                if sid not in cache:
                    pf = os.path.join(prompts, f'{sid}.mid')
                    cache[sid] = main_tempo(pf, args.bpm) if os.path.isfile(pf) else args.bpm
                bpm = cache[sid]
                print(f'[fix] {fn}: {mido.tempo2bpm(ev[0][1]):.0f} -> {bpm:.1f} bpm'
                      f'{"  (dry run)" if args.dry_run else ""}')
                if not args.dry_run:
                    rewrite(fn, bpm)
                n_fixed += 1
    print(f'[done] {n_fixed} of {n_seen} midis rewritten')


if __name__ == '__main__':
    main()
