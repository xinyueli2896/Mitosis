"""Stage Nottingham tunes as a melody/chord prompt split, pickup-aware.

For every tune named, reads <melody_dir>/<tune>.mid and
<chord_dir>/<tune>.mid, decides whether the tune opens with a PICKUP
(anacrusis), and writes the pair to <dst>/melody and <dst>/chord:

  pickup      copied as is -- the partial opening bar already puts the
              first downbeat off the start
  no pickup   ONE EMPTY BAR of silence inserted in front of both files,
              so the tune's first downbeat also sits one bar in

The bar length comes from the file's own time signature and ticks per
beat, so a 6/8 jig gets a 6/8 bar. Tempo and time-signature meta stay
at tick 0; only the notes move.

HOW A PICKUP IS DETECTED. abc2midi writes the anacrusis as a partial
first bar starting at tick 0 -- there is no meta event for it -- and
the ABC convention shortens the last bar to compensate, so the total
length is a bar multiple either way. What does mark it is the CHORD
track: chord symbols sit on downbeats, and a tune whose first chord
starts at a positive offset shorter than a bar has that much pickup in
front of its first downbeat. That is the rule used here; the melody's
first onset and the offset are printed per tune so the call can be
read off the log, and --pickup overrides it by name for any tune where
the chords do not follow the convention.

FORMAT MATCHES heldout_v5, which build_prompt_crops writes: a pure
tick-space window of the source with every program untouched, and the
tempo map replaced by ONE set_tempo at tick 0 (--tempo, default 120 as
there; 0 keeps the source map). No tag is applied here: the staging
step of the decode wrapper (trial_a12 CHORD_PROGRAM) does that, the
same way the E1 tree tags its prompts, so the split stays a faithful
copy of the source. --chord-program is there only to tag on purpose.

Usage (via build_nottingham_split.sbatch):
  python build_nottingham_split.py --melody-dir .../MIDI/melody \
      --chord-dir .../MIDI/chords --dst input/nottingham_split \
      --tunes ashover_5 hpps_9 morris_2 playford_15 xmas_8
"""

import argparse
import glob
import os
import shutil

import mido


def find_file(folder, tune):
    for cand in (f'{tune}.mid', f'{tune}.MID', f'{tune}.midi'):
        p = os.path.join(folder, cand)
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(folder, f'{tune}*.mid')))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SystemExit(f'{tune}: ambiguous in {folder}: {hits}')
    raise SystemExit(f'{tune}: not found in {folder} '
                     f'(looked for {tune}.mid and {tune}*.mid)')


def time_signature(mid):
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == 'time_signature':
                return msg.numerator, msg.denominator
    return 4, 4


def first_note_tick(mid):
    best = None
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                best = t if best is None else min(best, t)
                break
    return best


def last_tick(mid):
    return max((sum(m.time for m in tr) for tr in mid.tracks), default=0)


def shift_notes(mid, ticks):
    """Move every non-meta message later by `ticks`; meta stays put.

    Done on absolute times per track, so a tempo or time-signature
    event at 0 keeps its place and the notes that followed it move.
    """
    for tr in mid.tracks:
        t = 0
        events = []
        for msg in tr:
            t += msg.time
            events.append((t + (0 if msg.is_meta else ticks), msg))
        # end_of_track must remain last
        events.sort(key=lambda e: (e[0], 0 if e[1].type != 'end_of_track'
                                   else 1))
        prev = 0
        for abs_t, msg in events:
            msg.time = abs_t - prev
            prev = abs_t
        tr[:] = [m for _t, m in events]
    return mid


def force_tempo(mid, bpm):
    """Drop every set_tempo and put one at tick 0 of track 0.

    Ticks are musical positions; the tempo map only says how fast they
    go by, so no note moves. This is the heldout_v5 convention."""
    for tr in mid.tracks:
        t, events = 0, []
        for msg in tr:
            t += msg.time
            if msg.type != 'set_tempo':
                events.append((t, msg))
        prev = 0
        for abs_t, msg in events:
            msg.time = abs_t - prev
            prev = abs_t
        tr[:] = [m for _t, m in events]
    mid.tracks[0].insert(0, mido.MetaMessage(
        'set_tempo', tempo=mido.bpm2tempo(bpm), time=0))


def set_program(mid, prog):
    """Every note-carrying track gets program `prog` (rewritten or added)."""
    for tr in mid.tracks:
        has_notes = any(m.type == 'note_on' for m in tr)
        if not has_notes:
            continue
        seen = False
        ch = next((m.channel for m in tr if m.type == 'note_on'), 0)
        for m in tr:
            if m.type == 'program_change':
                m.program = prog
                seen = True
        if not seen:
            tr.insert(0, mido.Message('program_change', program=prog,
                                      channel=ch, time=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--melody-dir', required=True)
    ap.add_argument('--chord-dir', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--tunes', nargs='+', required=True)
    ap.add_argument('--chord-program', type=int, default=None,
                    help='tag the chord track with this program. Default '
                         'none: programs stay as in the source, like '
                         'heldout_v5')
    ap.add_argument('--tempo', type=float, default=120.0,
                    help='replace the tempo map with one set_tempo at this '
                         'bpm at tick 0, as build_prompt_crops does for '
                         'heldout_v5 (default 120). 0 keeps the source map.')
    ap.add_argument('--pickup', nargs='*', default=[],
                    help='override the detection: tune=1 (has a pickup, '
                         'copy as is) or tune=0 (none, pad a bar)')
    args = ap.parse_args()
    force = {}
    for item in args.pickup:
        k, v = item.split('=')
        force[k] = bool(int(v))

    mel_out = os.path.join(args.dst, 'melody')
    chd_out = os.path.join(args.dst, 'chord')
    os.makedirs(mel_out, exist_ok=True)
    os.makedirs(chd_out, exist_ok=True)

    print(f'{"tune":14s} {"sig":5s} {"bar":>6s} {"mel@":>7s} {"chd@":>7s} '
          f'{"len(bars)":>9s}  decision')
    for tune in args.tunes:
        mp = find_file(args.melody_dir, tune)
        cp = find_file(args.chord_dir, tune)
        mel = mido.MidiFile(mp)
        chd = mido.MidiFile(cp)
        if mel.ticks_per_beat != chd.ticks_per_beat:
            raise SystemExit(f'{tune}: melody {mel.ticks_per_beat} tpb but '
                             f'chord {chd.ticks_per_beat} tpb')
        tpb = mel.ticks_per_beat
        num, den = time_signature(mel)
        bar = int(tpb * 4 * num / den)
        m0 = first_note_tick(mel)
        c0 = first_note_tick(chd)
        length_bars = last_tick(mel) / bar

        if tune in force:
            pickup = force[tune]
            why = 'forced'
        elif c0 is None:
            pickup = (m0 or 0) % bar != 0 and 0 < (m0 or 0) < bar
            why = 'no chords; from melody onset'
        else:
            off = c0 % bar
            pickup = 0 < c0 < bar and (m0 is not None and m0 < c0)
            why = (f'first chord {c0 / tpb:.2f} beats in, melody starts '
                   f'{(m0 or 0) / tpb:.2f}: '
                   + ('anacrusis' if pickup else 'downbeat at 0'))
            if c0 >= bar and off == 0 and m0 is not None and m0 < c0:
                why += ' (chord enters a full bar later; treated as no pickup)'

        action = 'copy as is' if pickup else f'pad 1 bar ({bar} ticks)'
        print(f'{tune:14s} {num}/{den:<3d} {bar:6d} '
              f'{(m0 or 0) / tpb:7.2f} {(c0 if c0 is not None else -1) / tpb:7.2f} '
              f'{length_bars:9.2f}  {"pickup" if pickup else "NO pickup"}: '
              f'{action}  [{why}]')

        if not pickup:
            shift_notes(mel, bar)
            shift_notes(chd, bar)
        if args.tempo and args.tempo > 0:
            force_tempo(mel, args.tempo)
            force_tempo(chd, args.tempo)
        if args.chord_program is not None:
            set_program(chd, args.chord_program)
        mel.save(os.path.join(mel_out, f'{tune}.mid'))
        chd.save(os.path.join(chd_out, f'{tune}.mid'))

    print(f'\nwrote {len(args.tunes)} pair(s) -> {mel_out} / {chd_out}; '
          f'programs {"untouched" if args.chord_program is None else args.chord_program}; '
          f'tempo {"as source" if not args.tempo else f"one set_tempo at {args.tempo:g} bpm"}')


if __name__ == '__main__':
    main()
