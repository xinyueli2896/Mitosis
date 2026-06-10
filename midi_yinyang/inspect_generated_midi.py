"""Inspect a generated MIDI from the M2CJointAttn inference scripts.

Per-track breakdown:
  - is_drum vs program id
  - note count
  - unique pitch count
  - pitch range
  - note count BEFORE vs AFTER the prompt boundary (if --prompt-length given)
    -- detects "model emits notes in the prompt but nothing in the generated
       portion" (i.e. EOS-collapse on a stream)

Run:
    python inspect_generated_midi.py temp/<run>/<song>/co.mid
    python inspect_generated_midi.py --prompt-length 64 temp/<run>/<song>/co.mid
"""

import argparse
import os

import pretty_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('midi', help='Path to a generated .mid file.')
    ap.add_argument('--prompt-length', type=int, default=None,
                    help='Prompt length in timesteps (subbeats). At default '
                         'tempo (120) and beat_div=4, 1 timestep = 0.125s. If '
                         'set, the script splits each track into "prompt" and '
                         '"generated" parts and reports note counts in each.')
    ap.add_argument('--tempo', type=float, default=120.0)
    ap.add_argument('--beat-div', type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.midi):
        print(f'[err] not found: {args.midi}')
        return

    midi = pretty_midi.PrettyMIDI(args.midi)
    timestep_s = 60.0 / args.tempo / args.beat_div
    boundary_s = (args.prompt_length * timestep_s
                  if args.prompt_length is not None else None)

    print(f'\nfile:     {args.midi}')
    print(f'duration: {midi.get_end_time():.2f} s '
          f'({midi.get_end_time() / (4 * 60.0 / args.tempo):.2f} bars at {args.tempo} BPM)')
    if boundary_s is not None:
        print(f'prompt boundary: t={args.prompt_length} timesteps = '
              f'{boundary_s:.2f}s')
    print(f'tracks:   {len(midi.instruments)}')
    print()

    total_notes = 0
    drum_notes = 0
    pitched_notes = 0
    for i, inst in enumerate(midi.instruments):
        pitches = sorted({n.pitch for n in inst.notes})
        n_total = len(inst.notes)
        total_notes += n_total
        if inst.is_drum:
            drum_notes += n_total
        else:
            pitched_notes += n_total

        if boundary_s is None:
            print(f'  track {i:2d}  {"DRUM" if inst.is_drum else f"PROG{inst.program:03d}":7s}  '
                  f'notes={n_total:5d}  uniq_p={len(pitches):3d}  '
                  f'p_range=({min(pitches) if pitches else None}..'
                  f'{max(pitches) if pitches else None})')
        else:
            in_prompt = sum(1 for n in inst.notes if n.start < boundary_s)
            in_gen = n_total - in_prompt
            tag_pct_gen = f'{100 * in_gen / max(n_total, 1):.0f}%'
            print(f'  track {i:2d}  {"DRUM" if inst.is_drum else f"PROG{inst.program:03d}":7s}  '
                  f'notes={n_total:5d}  uniq_p={len(pitches):3d}  '
                  f'p_range=({min(pitches) if pitches else None}..'
                  f'{max(pitches) if pitches else None})  '
                  f'prompt={in_prompt}  gen={in_gen} ({tag_pct_gen})')

    print()
    print(f'total notes across all tracks: {total_notes}')
    print(f'  drum:    {drum_notes:5d}  ({100*drum_notes/max(total_notes,1):.0f}%)')
    print(f'  pitched: {pitched_notes:5d}  ({100*pitched_notes/max(total_notes,1):.0f}%)')

    # Verdict
    if pitched_notes == 0:
        print('\n[verdict] ZERO pitched notes anywhere. '
              'Either the non-drum side completely collapsed to silence, '
              'OR all pitched-program notes are being mis-routed.')
    elif boundary_s is not None:
        # Check per-track whether the generated portion is empty.
        empty_gen_tracks = []
        for inst in midi.instruments:
            in_gen = sum(1 for n in inst.notes if n.start >= boundary_s)
            if in_gen == 0 and len(inst.notes) > 0:
                empty_gen_tracks.append(
                    'DRUM' if inst.is_drum else f'PROG{inst.program:03d}'
                )
        if empty_gen_tracks:
            print(f'\n[verdict] These tracks have notes in the prompt but '
                  f'ZERO in the generated portion: {empty_gen_tracks}. '
                  f'Likely EOS-collapse on those streams during generation.')


if __name__ == '__main__':
    main()
