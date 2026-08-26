"""Split multi-track midis into per-stream melody/chord files by TRACK
NAME, preserving the tempo/meta grid -- the Pop1K7 counterpart of
extract_pop909_melody.py.

The annotated Pop1K7 release carries three named tracks per song:
"melody", "chord" and "piano" (the accompaniment). The melchord
pipeline wants POP909-format per-stream folders (one melody midi + one
chord midi per song, each = meta/tempo track + the one named track), so
downstream tooling -- tokenization with ins_ids=['track-0'], pairing
gate, merge/prompt scripts -- works unchanged. Selection is by NAME,
not track index, so a reordered file cannot silently swap streams.

Meta handling: if track 0 carries no notes it is copied verbatim (the
usual type-1 layout); otherwise a fresh meta track is synthesized from
the set_tempo / time_signature / key_signature events found anywhere in
the file, so the grid survives either layout.

Songs missing either named track are skipped and listed. Basename
collisions across --src folders abort (they would silently overwrite).

Usage (via preprocess_pop1k7_melchord.sbatch):
    python split_named_tracks.py \\
        --src /path/pop1k7/src_001 [--src ...] \\
        --dst-melody Pop1K7-Dataset/pop1k7-melody \\
        --dst-chord  Pop1K7-Dataset/pop1k7-chord
"""

import argparse
import os
from glob import glob

import mido


def find_track(mid, name):
    want = name.strip().lower()
    hits = [t for t in mid.tracks
            if (t.name or '').strip().lower() == want]
    return hits[0] if len(hits) == 1 else (hits, None)[0] if hits else None


def has_notes(track):
    return any(m.type == 'note_on' and m.velocity > 0 for m in track)


META_TYPES = ('set_tempo', 'time_signature', 'key_signature')


def build_meta_track(mid):
    """Fresh meta track from the global events found anywhere."""
    events = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type in META_TYPES:
                events.append((t, msg))
    events.sort(key=lambda e: e[0])
    out = mido.MidiTrack()
    prev = 0
    for t, msg in events:
        out.append(msg.copy(time=t - prev))
        prev = t
    out.append(mido.MetaMessage('end_of_track', time=0))
    return out


def split_one(path, dst_mel, dst_chd, melody_name, chord_name):
    mid = mido.MidiFile(path)
    mel = find_track(mid, melody_name)
    chd = find_track(mid, chord_name)
    if mel is None or chd is None:
        present = [repr((t.name or '').strip()) for t in mid.tracks]
        return f'missing named track (tracks: {", ".join(present)})'
    meta = (mid.tracks[0] if not has_notes(mid.tracks[0])
            else build_meta_track(mid))
    name = os.path.basename(path)
    for track, dst in ((mel, dst_mel), (chd, dst_chd)):
        out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
        out.tracks.append(meta)
        out.tracks.append(track)
        out.save(os.path.join(dst, name))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', action='append', required=True)
    p.add_argument('--dst-melody', required=True)
    p.add_argument('--dst-chord', required=True)
    p.add_argument('--melody-name', default='melody')
    p.add_argument('--chord-name', default='chord')
    args = p.parse_args()

    files = []
    for src in args.src:
        got = sorted(glob(os.path.join(src, '*.mid'))
                     + glob(os.path.join(src, '*.MID'))
                     + glob(os.path.join(src, '*.midi')))
        if not got:
            raise SystemExit(f'no midis under {src}')
        files.extend(got)
    seen = {}
    for f in files:
        b = os.path.basename(f)
        if b in seen:
            raise SystemExit(f'basename collision: {b} in both '
                             f'{seen[b]} and {os.path.dirname(f)} -- '
                             f'outputs would overwrite')
        seen[b] = os.path.dirname(f)

    os.makedirs(args.dst_melody, exist_ok=True)
    os.makedirs(args.dst_chord, exist_ok=True)
    print(f'[split] {len(files)} midis from {len(args.src)} folder(s)')

    skipped = []
    for i, f in enumerate(files):
        try:
            why = split_one(f, args.dst_melody, args.dst_chord,
                            args.melody_name, args.chord_name)
        except Exception as e:
            why = f'{type(e).__name__}: {e}'
        if why:
            skipped.append((os.path.basename(f), why))
        if (i + 1) % 200 == 0:
            print(f'[split] {i + 1}/{len(files)}')

    ok = len(files) - len(skipped)
    print(f'[split] wrote {ok} melody/chord pairs -> '
          f'{args.dst_melody} , {args.dst_chord}')
    if skipped:
        print(f'[split] skipped {len(skipped)}:')
        for n, why in skipped[:20]:
            print(f'  {n}: {why}')
        if len(skipped) > 20:
            print(f'  ... and {len(skipped) - 20} more')


if __name__ == '__main__':
    main()
