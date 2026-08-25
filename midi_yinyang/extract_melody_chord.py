"""Corpus-agnostic melody/chord extraction for solo-piano MIDI corpora
(Pop1K7, Aria-MIDI subsets, ...) into the POP909 melody/chord folder
format the whole melchord pipeline already consumes.

POP909 got its streams for free (a labeled MELODY track + human chord
annotations rendered by build_pop909_chord_midi.py). Piano-only corpora
have neither, so this applies the literature-standard recipe for
Pop1K7-style data:

  MELODY  skyline: at every onset instant keep the highest-pitched
          note; a kept note is truncated when the next skyline onset
          starts. Notes below --min-melody-pitch never become melody
          (left-hand bass runs).
  CHORD   duration-weighted pitch-class template matching per window of
          --chord-beats beats (default 2), over the same chord
          vocabulary build_pop909_chord_midi.py renders (maj, min, dim,
          aug, sus2/4, 6ths, 7ths). Windows with too little pitch mass
          emit no chord (like POP909's 'N' labels). The chosen chord is
          rendered exactly like the POP909 chord midis: bass note near
          C2 plus chord tones near C4.

Output per song: <dst>-melody/<name>.mid and <dst>-chord/<name>.mid,
each carrying the SOURCE's meta/tempo track (so the grid stays
identical) plus one named track (MELODY / CHORD). Songs are skipped
(and listed) when unreadable, shorter than --min-beats, or when the
skyline keeps fewer than --min-melody-notes notes.

IMPORTANT upstream requirement: the source midis must be BEAT-ALIGNED
(tempo map matches the audio beats). Pop1K7's midi_analyzed/
midi_synchronized folders are (madmom beat tracking); raw Aria-MIDI
transcriptions are NOT -- run check_beat_alignment.sbatch on any new
corpus's extraction output before tokenizing, and expect Aria to need a
quantization stage first.

Usage (via extract_melody_chord.sbatch):
    python extract_melody_chord.py --src <folder of piano midis> \\
        --dst-prefix <corpus name, e.g. pop1k7> [--limit N]
"""

import argparse
import os
import warnings
from glob import glob

import mido
import pretty_midi
from joblib import Parallel, delayed

# Same rendering register as build_pop909_chord_midi.py.
BASS_CENTER = 36    # ~C2
CHORD_CENTER = 60   # ~C4
PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

QUALITY_INTERVALS = {
    'maj': (0, 4, 7),
    'min': (0, 3, 7),
    'dim': (0, 3, 6),
    'aug': (0, 4, 8),
    'sus2': (0, 2, 7),
    'sus4': (0, 5, 7),
    'maj6': (0, 4, 7, 9),
    'min6': (0, 3, 7, 9),
    '7': (0, 4, 7, 10),
    'maj7': (0, 4, 7, 11),
    'min7': (0, 3, 7, 10),
}
# Template weights: chord tones positive, the root emphasized, non-chord
# tones penalized. Precomputed per (root, quality).
TEMPLATES = []
for root in range(12):
    for quality, ivs in QUALITY_INTERVALS.items():
        tones = frozenset((root + iv) % 12 for iv in ivs)
        TEMPLATES.append((root, quality, tones))


def skyline(notes, min_pitch):
    """Classic skyline: keep the highest-pitched note at each onset
    instant; truncate each kept note at the next kept onset."""
    onsets = {}
    for n in notes:
        if n.pitch < min_pitch:
            continue
        key = round(n.start, 4)
        if key not in onsets or n.pitch > onsets[key].pitch:
            onsets[key] = n
    picked = sorted(onsets.values(), key=lambda n: n.start)
    out = []
    for i, n in enumerate(picked):
        end = n.end
        if i + 1 < len(picked):
            end = min(end, picked[i + 1].start)
        if end - n.start > 1e-3:
            out.append((n.start, end, n.pitch))
    return out


def estimate_chords(notes, beats, chord_beats, min_mass):
    """Duration-weighted pitch-class template matching per window."""
    out = []   # (start_time, end_time, root, quality)
    for w in range(0, len(beats) - 1, chord_beats):
        t0 = beats[w]
        t1 = beats[min(w + chord_beats, len(beats) - 1)]
        if t1 <= t0:
            continue
        mass = [0.0] * 12
        bass_pitch, bass_start = None, None
        for n in notes:
            ov = min(n.end, t1) - max(n.start, t0)
            if ov <= 0:
                continue
            mass[n.pitch % 12] += ov
            if bass_pitch is None or n.pitch < bass_pitch:
                bass_pitch = n.pitch
        total = sum(mass)
        if total < min_mass * (t1 - t0):
            continue
        best, best_score = None, float('-inf')
        for root, quality, tones in TEMPLATES:
            score = sum(m if pc in tones else -m
                        for pc, m in enumerate(mass))
            score += 0.5 * mass[root]                    # root emphasis
            if bass_pitch is not None and bass_pitch % 12 == root:
                score += 0.25 * total                    # bass agreement
            if score > best_score:
                best_score, best = score, (root, quality)
        out.append((t0, t1, best[0], best[1]))
    # merge consecutive identical chords
    merged = []
    for seg in out:
        if merged and merged[-1][2:] == seg[2:] and \
                abs(merged[-1][1] - seg[0]) < 1e-3:
            merged[-1] = (merged[-1][0], seg[1], seg[2], seg[3])
        else:
            merged.append(list(seg) if isinstance(seg, tuple) else seg)
            merged[-1] = list(merged[-1])
    return [tuple(m) for m in merged]


def write_stream(src_mido, pm, events, out_path, track_name, is_chord):
    """events: melody [(start, end, pitch)] or chord segments."""
    out = mido.MidiFile(ticks_per_beat=src_mido.ticks_per_beat)
    out.tracks.append(src_mido.tracks[0])      # meta/tempo, same grid
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=track_name, time=0))
    msgs = []
    if is_chord:
        for (t0, t1, root, quality) in events:
            pitches = [BASS_CENTER + root]
            for iv in QUALITY_INTERVALS[quality]:
                pitches.append(CHORD_CENTER + root + iv
                               - (12 if root + iv >= 12 else 0))
            tick0 = int(round(pm.time_to_tick(t0)))
            tick1 = max(tick0 + 1, int(round(pm.time_to_tick(t1))))
            for p in pitches:
                msgs.append((tick0, 1, mido.Message(
                    'note_on', note=p, velocity=80, time=0)))
                msgs.append((tick1, 0, mido.Message(
                    'note_off', note=p, velocity=0, time=0)))
    else:
        for (start, end, pitch) in events:
            tick0 = int(round(pm.time_to_tick(start)))
            tick1 = max(tick0 + 1, int(round(pm.time_to_tick(end))))
            msgs.append((tick0, 1, mido.Message(
                'note_on', note=pitch, velocity=100, time=0)))
            msgs.append((tick1, 0, mido.Message(
                'note_off', note=pitch, velocity=0, time=0)))
    msgs.sort(key=lambda m: (m[0], m[1]))
    prev = 0
    for tick, _, msg in msgs:
        msg.time = tick - prev
        prev = tick
        track.append(msg)
    out.tracks.append(track)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)


def process_one(path, mel_dir, chd_dir, args):
    name = os.path.splitext(os.path.basename(path))[0] + '.mid'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pm = pretty_midi.PrettyMIDI(path)
            src = mido.MidiFile(path)
        notes = [n for inst in pm.instruments if not inst.is_drum
                 for n in inst.notes]
        beats = pm.get_beats()
        if len(beats) < args.min_beats:
            return (name, f'too short ({len(beats)} beats)')
        mel = skyline(notes, args.min_melody_pitch)
        if len(mel) < args.min_melody_notes:
            return (name, f'skyline kept only {len(mel)} notes')
        chords = estimate_chords(notes, beats, args.chord_beats,
                                 args.min_chord_mass)
        if not chords:
            return (name, 'no chord windows above mass threshold')
        write_stream(src, pm, mel, os.path.join(mel_dir, name),
                     'MELODY', is_chord=False)
        write_stream(src, pm, chords, os.path.join(chd_dir, name),
                     'CHORD', is_chord=True)
        return (name, None)
    except Exception as e:  # unreadable / malformed
        return (name, f'{type(e).__name__}: {e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True,
                   help='folder of beat-aligned solo-piano midis')
    p.add_argument('--dst-prefix', required=True,
                   help='output prefix: <prefix>-melody/ and <prefix>-chord/')
    p.add_argument('--limit', type=int, default=None,
                   help='process only the first N files (sorted)')
    p.add_argument('--chord-beats', type=int, default=2,
                   help='chord window in beats (default 2 = half a 4/4 bar)')
    p.add_argument('--min-melody-pitch', type=int, default=55,
                   help='skyline ignores notes below this (default G3=55)')
    p.add_argument('--min-melody-notes', type=int, default=32)
    p.add_argument('--min-beats', type=int, default=32)
    p.add_argument('--min-chord-mass', type=float, default=0.5,
                   help='min duration-weighted pitch mass per second of '
                        'window for a chord to be emitted')
    p.add_argument('--jobs', type=int, default=-1)
    args = p.parse_args()

    files = sorted(glob(os.path.join(args.src, '*.mid'))
                   + glob(os.path.join(args.src, '*.MID'))
                   + glob(os.path.join(args.src, '*.midi')))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no midi files under {args.src}')
    mel_dir = args.dst_prefix + '-melody'
    chd_dir = args.dst_prefix + '-chord'
    print(f'[extract] {len(files)} files  {args.src} -> '
          f'{mel_dir}/ + {chd_dir}/')

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(process_one)(f, mel_dir, chd_dir, args) for f in files)
    skipped = [(n, why) for (n, why) in results if why is not None]
    print(f'\n[extract] wrote {len(results) - len(skipped)} song pairs, '
          f'skipped {len(skipped)}')
    for n, why in skipped[:20]:
        print(f'  skip {n}: {why}')
    if len(skipped) > 20:
        print(f'  ... and {len(skipped) - 20} more')
    print('\nNEXT: verify the grid before tokenizing --')
    print(f'  sbatch --export=ALL,FOLDERS="{mel_dir} {chd_dir}" '
          f'midi_yinyang/check_beat_alignment.sbatch')


if __name__ == '__main__':
    main()
