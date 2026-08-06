"""Make piano LISTENING COPIES of generated midis, without touching the
files the evaluation reads.

Why not just write piano programs at inference: in the melchord merged
representation the chord stream is identified by its program (48,
strings) -- that is both how the tokenizer separates the streams in the
prompt and how eval_metrics attributes notes in S0/S1 outputs
(chord_programs={48}). Rewriting programs in the scored files would
re-fuse the streams for the scorer. So the originals stay exactly as
generated, and this script emits parallel copies for listening with
every non-drum instrument set to one program (default 0, Acoustic Grand
-- the GM slot a Steinway soundfont sits on; the actual timbre is the
player's synth, MIDI can only pick the slot).

Track structure, names, notes and timing are preserved; only
program_change events are rewritten (drum channel 9 untouched). Files
are re-emitted via mido, so no re-quantization occurs.

Usage:
    python render_piano_copies.py temp/e1_melchord_1234/A.2 temp/e1_S1_out
        -> temp/e1_melchord_1234/A.2_piano , temp/e1_S1_out_piano
    python render_piano_copies.py --program 0 --suffix _piano <folders...>
"""

import argparse
import os

import mido


def pianoize_file(src, dst, program):
    mid = mido.MidiFile(src)
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for track in mid.tracks:
        t = mido.MidiTrack()
        for msg in track:
            if msg.type == 'program_change' and msg.channel != 9:
                t.append(msg.copy(program=program))
            else:
                t.append(msg.copy())
        out.tracks.append(t)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.save(dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folders', nargs='+',
                   help='output folders (searched recursively for .mid)')
    p.add_argument('--program', type=int, default=0,
                   help='GM program for every non-drum track (default 0, '
                        'Acoustic Grand Piano)')
    p.add_argument('--suffix', default='_piano',
                   help="suffix for the copy folder (default '_piano')")
    args = p.parse_args()

    total = 0
    for folder in args.folders:
        folder = folder.rstrip('/')
        if not os.path.isdir(folder):
            print(f'[skip] not a directory: {folder}')
            continue
        dst_root = folder + args.suffix
        n = 0
        for root, _, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith(('.mid', '.midi')):
                    continue
                src = os.path.join(root, f)
                rel = os.path.relpath(src, folder)
                try:
                    pianoize_file(src, os.path.join(dst_root, rel),
                                  args.program)
                    n += 1
                except Exception as e:
                    print(f'  [fail] {src}: {e!r}')
        print(f'{folder} -> {dst_root}  ({n} file(s))')
        total += n
    print(f'done: {total} listening cop{"y" if total == 1 else "ies"}. '
          f'Scored originals untouched.')


if __name__ == '__main__':
    main()
