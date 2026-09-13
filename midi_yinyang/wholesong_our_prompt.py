"""Build whole-song-gen's lead-sheet prompt image from OUR prompt midis.

The baseline is prompted by SONG ID: it rebuilds the song from POP909
through its own analysis, so its prompt is POP909's melody and chord,
not ours. Ours differ in quantisation (our 16th grid), in beat
alignment (pop909_align_grid's downbeats), and in chord voicing. This
module produces the array the model actually reads, filled from
input/heldout_v5/{melody,chord}/<song>.mid, so the baseline is prompted
with the same music every other E1 system gets.

THE TARGET FORMAT is fixed by two functions in their repo, and this
file mirrors them rather than inventing an encoding:

  note_matrix_to_piano_roll      (data_utils/utils/format_converter.py)
      piano_roll = zeros((2, L, 128))
      piano_roll[0, onset, pitch] = 1                   <- channel 0 = ONSET
      piano_roll[1, onset+1: onset+dur, pitch] = 1      <- channel 1 = SUSTAIN

  chord_mat_to_chord_roll        (same file)
      chord_roll[2, start, :]            = chroma       onset
      chord_roll[3, start+1: start+dur]  = chroma       sustain
      chord_roll[4, start, bass]         = 1            onset
      chord_roll[5, start+1: start+dur]  = 1            sustain

and LeadSheetDataset.lang_to_img composes them into ONE 128-row image,
in this order -- the chord bands OVERWRITE the melody in rows 24..47:

      img[0:2, :, 0:128]  = mel_roll
      img[0:2, :, 36:48]  = chd_roll[2:4]     chroma, one row per pc
      img[0:2, :, 24:36]  = chd_roll[4:6]     bass,   one row per pc

CHROMA IS ABSOLUTE, BASS IS ABSOLUTE HERE. Their chord matrix stores
the bass RELATIVE to the root and chord_mat_to_chord_roll converts it
(`bass = (chord_content[-1] + root) % 12`) while writing chroma
untouched -- so what lands in the image is absolute pitch classes in
both bands. That is what we can read off a midi, and it is why this
module needs no chord symbol, no root, and no Harte parse: a voicing
cannot give a root, and the image never carries one.

Our renderer makes the bass unambiguous. build_pop909_chord_midi.py
writes the bass at 36+pc and the upper tones at 60+pc, two octaves
apart, so the lowest sounding note is always the bass voice.

WHAT THIS DELIBERATELY REPRODUCES, not corrects: our renderer folds the
bass interval into the upper voicing (`intervals.add(bass_interval)`),
so a slash chord like C/D yields a chroma containing D where the
annotation's would not. The point of the swap is that the baseline sees
the chord OUR systems see, so the rendering's own choices come with it.

Self-test (no GPU, no venv, no model):
    python wholesong_our_prompt.py --selftest \
        --melody input/heldout_v5/melody/001.mid \
        --chord  input/heldout_v5/chord/001.mid
"""

import argparse

import numpy as np
import pretty_midi

# Image rows for the two chord bands, per lang_to_img.
CHROMA_LO, CHROMA_HI = 36, 48
BASS_LO, BASS_HI = 24, 36
# Rows the chord bands occupy, hence rows a melody note would lose.
CHORD_ROWS = range(BASS_LO, CHROMA_HI)


def _merged_notes(path):
    """Every note in the file, whatever track it sits on.

    Our crops carry one instrument per file, but a stray empty track or
    a program-change split would silently halve the prompt if we took
    instruments[0].
    """
    pm = pretty_midi.PrettyMIDI(path)
    notes = [n for inst in pm.instruments for n in inst.notes]
    tempi = pm.get_tempo_changes()[1]
    bpm = float(tempi[0]) if len(tempi) else 120.0
    return notes, bpm


def _to_steps(notes, sec_per_step, skip_steps, n_steps):
    """[(onset_step, pitch, duration_steps)], clipped to the window.

    skip_steps drops the crop's leading pad: our midis begin with
    pad_frames of silence so the melody's first downbeat lands at frame
    32, and the baseline's levels are cropped at crop_bar with no pad,
    so the prompt is taken from the first REAL frame.
    """
    out = []
    for n in notes:
        on = int(round(n.start / sec_per_step)) - skip_steps
        off = int(round(n.end / sec_per_step)) - skip_steps
        dur = max(off - on, 1)
        if on < 0:                      # head inside the padded region
            dur += on
            on = 0
        if dur <= 0 or on >= n_steps:
            continue
        out.append((on, int(n.pitch), min(dur, n_steps - on)))
    return out


def build_lsh_prompt(melody_path, chord_path, n_steps, skip_steps=0,
                     bpm=None, verbose=False):
    """The (2, n_steps, 128) lead-sheet prompt image, from our midis."""
    mel_notes, mel_bpm = _merged_notes(melody_path)
    chd_notes, chd_bpm = _merged_notes(chord_path)
    bpm = bpm or mel_bpm
    if abs(mel_bpm - chd_bpm) > 1e-6:
        raise ValueError(
            f'melody is {mel_bpm} BPM but chord is {chd_bpm} BPM; the two '
            f'crops must share a tempo or their grids do not line up'
        )
    sec_per_step = 60.0 / bpm / 4.0          # one step = one sixteenth

    img = np.zeros((2, n_steps, 128), dtype=np.int64)

    # --- melody: exactly note_matrix_to_piano_roll -------------------
    mel = _to_steps(mel_notes, sec_per_step, skip_steps, n_steps)
    lost = 0
    for on, pitch, dur in mel:
        if pitch in CHORD_ROWS:
            lost += 1
        img[0, on, pitch] = 1
        img[1, on + 1: on + dur, pitch] = 1

    # --- chord: segments, then the two absolute-pitch-class bands ----
    chd = _to_steps(chd_notes, sec_per_step, skip_steps, n_steps)
    segs = {}
    for on, pitch, dur in chd:
        segs.setdefault(on, []).append((pitch, dur))
    starts = sorted(segs)
    for i, start in enumerate(starts):
        members = segs[start]
        # A chord runs to its own longest note or to the next chord,
        # whichever comes first -- a held note must not paint over the
        # chord that replaced it.
        own_end = start + max(d for _, d in members)
        next_start = starts[i + 1] if i + 1 < len(starts) else n_steps
        end = min(own_end, next_start, n_steps)
        chroma = sorted({p % 12 for p, _ in members})
        bass = min(p for p, _ in members) % 12
        for pc in chroma:
            img[0, start, CHROMA_LO + pc] = 1
            img[1, start + 1: end, CHROMA_LO + pc] = 1
        img[0, start, BASS_LO + bass] = 1
        img[1, start + 1: end, BASS_LO + bass] = 1

    if verbose:
        print(f'  [our-prompt] {len(mel)} melody notes, {len(starts)} chord '
              f'segments over {n_steps} steps at {bpm:g} BPM'
              f'{f" (skipped {skip_steps} padded steps)" if skip_steps else ""}')
        if lost:
            print(f'  [our-prompt] WARNING {lost} melody note(s) in rows '
                  f'{BASS_LO}..{CHROMA_HI - 1}, overwritten by the chord '
                  f'bands -- their own lang_to_img does the same, but a '
                  f'melody this low is unusual and worth a look')
    return img


def _selftest(a):
    """Round-trip: build the image, decode it, compare to the source."""
    notes, bpm = _merged_notes(a.melody)
    sec_per_step = 60.0 / bpm / 4.0
    n_steps = a.steps or (
        max(int(round(n.end / sec_per_step)) for n in notes) + 1)
    img = build_lsh_prompt(a.melody, a.chord, n_steps,
                           skip_steps=a.skip, verbose=True)

    print(f'\n  image {img.shape}, dtype {img.dtype}, '
          f'{int(img.sum())} set cells')
    for name, lo, hi in (('melody', CHROMA_HI, 128),
                         ('chroma', CHROMA_LO, CHROMA_HI),
                         ('bass', BASS_LO, BASS_HI)):
        on = int(img[0, :, lo:hi].sum())
        su = int(img[1, :, lo:hi].sum())
        print(f'  {name:7} rows {lo:3}-{hi - 1:3}: {on:5} onsets, '
              f'{su:6} sustain')
        if on == 0:
            print(f'  FAIL: {name} band is empty')
            return 1

    # every step with a chord must name exactly one bass pitch class
    bad = [t for t in range(n_steps)
           if img[:, t, BASS_LO:BASS_HI].sum() > 1]
    if bad:
        print(f'  FAIL: {len(bad)} step(s) with more than one bass class, '
              f'first at {bad[0]}')
        return 1
    print('  bass band is single-valued at every step  OK')

    # onset and sustain must not both be set for one cell
    both = int((img[0] & img[1]).sum())
    if both:
        print(f'  FAIL: {both} cell(s) marked onset AND sustain')
        return 1
    print('  no cell is both onset and sustain  OK')

    src = _to_steps(notes, sec_per_step, a.skip, n_steps)
    kept = [n for n in src if n[1] >= CHROMA_HI]
    got = int(img[0, :, CHROMA_HI:128].sum())
    print(f'  melody onsets above row {CHROMA_HI}: {len(kept)} in, '
          f'{got} out  {"OK" if got == len(kept) else "MISMATCH"}')
    return 0 if got == len(kept) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--melody', required=True)
    ap.add_argument('--chord', required=True)
    ap.add_argument('--steps', type=int, default=0,
                    help='window length; default the melody\'s own length')
    ap.add_argument('--skip', type=int, default=0,
                    help="the crop's pad_frames, dropped from the front")
    a = ap.parse_args()
    raise SystemExit(_selftest(a) if a.selftest else 0)


if __name__ == '__main__':
    main()
