"""Whole-Song Gen's song folders for the Nottingham held-out prompts.

Whole-Song Gen reads a song as POP909 ships it after their own
cleaning (data/pop909_w_structure_label/<id>/): melody.txt,
finalized_chord.txt and a human phrase label. Nottingham has none of
these, so this writes the same three files for each staged prompt pair
(prompts/mel/<tune>.mid + prompts/chord/<tune>.mid), built from OUR
crops, so the baseline is prompted with exactly the notes every other
system gets:

  melody.txt           "pitch duration" per line, sixteenth steps, pitch 0
                       a rest (their read_melody). Monophonic: where two
                       melody notes start together the highest is kept,
                       and a note is cut where the next one starts.
  finalized_chord.txt  "symbol ... duration" per line, in beats (their
                       read_chord parses the symbol with mir_eval). The
                       chord sounding at each beat is read off our chord
                       track and named by template matching (root, quality
                       and, when the lowest note is not the root, a slash
                       bass); a set no template fits is written as a list
                       of degrees over its lowest note, which mir_eval
                       also reads. Silence is N.
  human_label1.txt     phrase label, e.g. i1A8A8B8B8. Nottingham carries
                       no phrase annotation, so this is a heuristic and
                       the one input here that is ours rather than the
                       data's: an optional pickup bar is an intro (i),
                       then 8-bar phrases from the first downbeat, a
                       phrase taking the letter of an earlier one whose
                       melody it repeats on >= --same-frac of its frames
                       (abc2midi writes repeats out, so AABB tunes read
                       A8A8B8B8). Only the prompt bars' phrases reach the
                       model under --form generated; key is estimated by
                       their own analysis.

LEADING SILENCE. Our crops put the first full downbeat at frame 32,
so every crop opens with one (pickup) or two empty bars. Whole-Song
Gen cannot represent bars we padded, so, as wholesong_prompted.py does
for POP909's padded crops, each song starts at the first bar that has
a note, its prompt is shortened by the bars skipped, and its output is
shifted right by them again on write. crops.tsv carries that offset
(crop_bar 0, pad_frames = skipped bars x 16) in the columns
wholesong_prompted.py reads.

Usage (scripts/external/wholesong.sh generate, DATASET=nottingham):
    python build_nottingham_wholesong.py --mel-dir temp/E1_nottingham/prompts/mel \\
        --chord-dir temp/E1_nottingham/prompts/chord --dst temp/E1_nottingham/ws_songs
"""
import argparse
import csv
import glob
import os

import pretty_midi

from eval_metrics import frame_fn

STEPS_PER_BEAT = 4
STEPS_PER_BAR = 16          # 4/4 and 2/2 only (build_nottingham_prompts)

# intervals above the root, without the root -> Harte quality
TEMPLATES = {
    (4, 7): 'maj', (3, 7): 'min', (3, 6): 'dim', (4, 8): 'aug',
    (4, 7, 10): '7', (4, 7, 11): 'maj7', (3, 7, 10): 'min7',
    (3, 6, 9): 'dim7', (3, 6, 10): 'hdim7', (3, 7, 11): 'minmaj7',
    (4, 7, 9): 'maj6', (3, 7, 9): 'min6', (5, 7): 'sus4', (2, 7): 'sus2',
}
DEGREE = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']
PC_NAME = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def chord_symbol(pitches):
    """Harte symbol for a set of sounding pitches (mir_eval-readable)."""
    if not pitches:
        return 'N'
    bass = min(pitches) % 12
    pcs = {p % 12 for p in pitches}
    # the bass as root first, then every other pitch class
    for root in [bass] + sorted(pcs - {bass}):
        iv = tuple(sorted((pc - root) % 12 for pc in pcs if pc != root))
        if iv in TEMPLATES:
            sym = f'{PC_NAME[root]}:{TEMPLATES[iv]}'
            if root != bass:
                sym += '/' + DEGREE[(bass - root) % 12]
            return sym
    degrees = ','.join(DEGREE[(pc - bass) % 12] for pc in sorted(
        pcs, key=lambda pc: (pc - bass) % 12))
    return f'{PC_NAME[bass]}:({degrees})'


def frames(path):
    """[(f0, f1, pitch)] of every note, on the file's sixteenth grid."""
    pm = pretty_midi.PrettyMIDI(path)
    to_f = frame_fn(pm)
    out = []
    for inst in pm.instruments:
        for n in inst.notes:
            f0 = int(round(to_f(n.start)))
            out.append((f0, max(f0 + 1, int(round(to_f(n.end)))), n.pitch))
    return sorted(out)


def melody_rows(notes):
    """(pitch, dur) rows with rests, monophonic, from step 0."""
    by_onset = {}
    for f0, f1, p in notes:
        if f0 not in by_onset or p > by_onset[f0][1]:
            by_onset[f0] = (f1, p)
    onsets = sorted(by_onset)
    rows, cursor = [], 0
    for i, f0 in enumerate(onsets):
        f1, p = by_onset[f0]
        if i + 1 < len(onsets):
            f1 = min(f1, onsets[i + 1])
        if f0 > cursor:
            rows.append((0, f0 - cursor))
        rows.append((p, f1 - f0))
        cursor = f1
    return rows


def chord_rows(notes, n_beats):
    """(symbol, beats) rows, one chord per beat, run-length merged."""
    per_beat = []
    for b in range(n_beats):
        s = b * STEPS_PER_BEAT
        sounding = [p for f0, f1, p in notes if f0 <= s < f1]
        if not sounding:        # a chord that enters inside the beat
            sounding = [p for f0, f1, p in notes
                        if s <= f0 < s + STEPS_PER_BEAT]
        per_beat.append(chord_symbol(sounding))
    rows = []
    for sym in per_beat:
        if rows and rows[-1][0] == sym:
            rows[-1][1] += 1
        else:
            rows.append([sym, 1])
    return rows


def phrase_label(mel_notes, n_bars, pickup_bars, phrase_bars, same_frac):
    """i<pickup> then <letter><len>... over n_bars."""
    top = {}
    for f0, f1, p in mel_notes:
        for f in range(f0, f1):
            top[f] = max(top.get(f, 0), p)

    def block(b0, nb):
        return [top.get(f, 0) for f in range(b0 * STEPS_PER_BAR,
                                             (b0 + nb) * STEPS_PER_BAR)]

    label = f'i{pickup_bars}' if pickup_bars else ''
    seen = []                   # (letter, frames)
    b = pickup_bars
    while b < n_bars:
        nb = min(phrase_bars, n_bars - b)
        cur = block(b, nb)
        letter = None
        for lt, ref in seen:
            m = min(len(cur), len(ref))
            if sum(x == y for x, y in zip(cur[:m], ref[:m])) >= same_frac * m:
                letter = lt
                break
        if letter is None:
            letter = chr(ord('A') + len(seen))
            seen.append((letter, cur))
        label += f'{letter}{nb}'
        b += nb
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mel-dir', required=True)
    ap.add_argument('--chord-dir', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--downbeat-frame', type=int, default=32,
                    help='where our crops put the first full downbeat')
    ap.add_argument('--phrase-bars', type=int, default=8)
    ap.add_argument('--same-frac', type=float, default=0.75)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    rows = []
    for mel_path in sorted(glob.glob(os.path.join(a.mel_dir, '*.mid'))):
        tune = os.path.splitext(os.path.basename(mel_path))[0]
        mel, chd = frames(mel_path), frames(os.path.join(a.chord_dir, f'{tune}.mid'))
        first = min(n[0] for n in mel + chd)
        skip_bars = first // STEPS_PER_BAR
        skip = skip_bars * STEPS_PER_BAR
        mel = [(f0 - skip, f1 - skip, p) for f0, f1, p in mel]
        chd = [(f0 - skip, f1 - skip, p) for f0, f1, p in chd]
        end = max(n[1] for n in mel + chd)
        n_bars = -(-end // STEPS_PER_BAR)
        pickup_bars = max(0, (a.downbeat_frame - skip) // STEPS_PER_BAR)

        d = os.path.join(a.dst, tune)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'melody.txt'), 'w') as fh:
            fh.writelines(f'{p} {dur}\n' for p, dur in melody_rows(mel))
        crows = chord_rows(chd, n_bars * STEPS_PER_BAR // STEPS_PER_BEAT)
        with open(os.path.join(d, 'finalized_chord.txt'), 'w') as fh:
            fh.writelines(f'{sym} {dur}\n' for sym, dur in crows)
        label = phrase_label(mel, n_bars, pickup_bars, a.phrase_bars, a.same_frac)
        with open(os.path.join(d, 'human_label1.txt'), 'w') as fh:
            fh.write(label + '\n')
        rows.append({'id': tune, 'crop_bar': 0, 'pad_frames': skip,
                     'bars': n_bars, 'phrases': label,
                     'chords': len(crows),
                     'templated': sum(':(' not in s and s != 'N' for s, _ in crows)})

    with open(os.path.join(a.dst, 'crops.tsv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    n_ch = sum(r['chords'] for r in rows)
    n_tp = sum(r['templated'] for r in rows)
    print(f'[ws-songs] {len(rows)} tunes -> {a.dst}; chord segments named by '
          f'template {n_tp}/{n_ch}; pad bars skipped: '
          f'{sorted({r["pad_frames"] // STEPS_PER_BAR for r in rows})}')


if __name__ == '__main__':
    main()
