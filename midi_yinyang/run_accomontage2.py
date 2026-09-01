"""T8 driver: harmonize our POP909 test melodies with AccoMontage2.

For each melody midi in --melody-dir:
  1. find its phrase segmentation string in the hierarchical-structure-
     analysis labels (human_label1.txt per song id);
  2. find its tonic key: POP909 key annotation file when present, else
     Krumhansl-Schmuckler estimation from the melody's pitch-class
     histogram (deterministic; report which source was used);
  3. run chorderator: set_melody -> set_meta -> set_segmentation ->
     generate_save; keep chord_gen.mid (melody + generated chords).

Defensive by design: every song runs in a try/except, failures are
reported per-song with the reason, and a summary table closes the run.
The known integration risks it surfaces explicitly:
  * label alphabet -- the human labels use i/o/x/X segments
    ('i12A8...o8'), richer than the demo's 'A8B8A8B8'; we pass the raw
    string and record acceptance/rejection per song;
  * bar-count alignment -- our extracted melody may disagree with the
    label's total bar count (pickups); we report the two counts.

Usage (via run_accomontage2.sbatch):
    python run_accomontage2.py --melody-dir ../input/pop909_split/melody \
        --labels-dir external/hierarchical-structure-analysis/POP909 \
        --out-dir temp/accomontage2_pop909 [--limit 3]
"""

import argparse
import os
import re
import shutil
import sys
import traceback

# Krumhansl-Schmuckler key profiles (Krumhansl 1990).
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
               'F#', 'G', 'G#', 'A', 'A#', 'B']


def ks_estimate_key(midi_path):
    """(tonic_name, mode) via Krumhansl-Schmuckler on duration-weighted
    pitch classes."""
    import pretty_midi
    import numpy as np
    pm = pretty_midi.PrettyMIDI(midi_path)
    hist = np.zeros(12)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            hist[n.pitch % 12] += n.end - n.start
    if hist.sum() == 0:
        return 'C', 'maj'
    best = (-2.0, 'C', 'maj')
    for shift in range(12):
        rolled = np.roll(hist, -shift)
        for profile, mode in ((KS_MAJOR, 'maj'), (KS_MINOR, 'min')):
            r = np.corrcoef(rolled, profile)[0, 1]
            if r > best[0]:
                best = (r, PITCH_NAMES[shift], mode)
    return best[1], best[2]


def read_pop909_key(song_id, pop909_dirs):
    """Return (tonic, mode) from a POP909 key annotation if one exists.

    CONTRACT (registered with the user): the POP909-Dataset clone under
    external/ is a KEY-ANNOTATION SOURCE ONLY. Its midis are NOT beat
    aligned and must never be fed to harmonization -- melodies always
    come from --melody-dir (our beat-aligned extracted split).
    """
    for root in pop909_dirs:
        f = os.path.join(root, song_id, 'key_audio.txt')
        if os.path.isfile(f):
            with open(f) as fh:
                line = fh.readline().split()
            if len(line) >= 3:                    # start end key
                key = line[2]                     # e.g. 'Gb:maj' / 'A:min'
                m = re.match(r'([A-Ga-g])([#b]?):?(maj|min)?', key)
                if m:
                    # Uppercase the LETTER only ('Gb' must not become
                    # 'GB'), then map flats to their sharp enharmonics
                    # for chorderator's Key vocabulary.
                    tonic = m.group(1).upper() + m.group(2)
                    flat2sharp = {'Cb': 'B', 'Db': 'C#', 'Eb': 'D#',
                                  'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#',
                                  'Bb': 'A#'}
                    tonic = flat2sharp.get(tonic, tonic)
                    return tonic, (m.group(3) or 'maj')
    return None


def read_phrase_label(song_id, labels_dir):
    f = os.path.join(labels_dir, song_id, 'human_label1.txt')
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        return fh.readline().strip()


def label_bar_count(label):
    return sum(int(n) for n in re.findall(r'[A-Za-z](\d+)', label))


def melody_bar_count(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    end = pm.get_end_time()
    # 4/4 at the file's tempo; downbeats are authoritative when present.
    try:
        db = pm.get_downbeats()
        return len(db)
    except Exception:                             # noqa: BLE001
        tempo = pm.estimate_tempo() or 120.0
        return int(round(end / (4 * 60.0 / tempo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--melody-dir', required=True)
    ap.add_argument('--labels-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pop909-dirs', nargs='*', default=[],
                    help='key_audio.txt sources ONLY (never melodies); '
                         'the sbatch passes the external/ clone '
                         'explicitly because the job runs from the '
                         'accomontage2 repo root')
    ap.add_argument('--limit', type=int, default=0,
                    help='>0: smoke test on the first N songs only')
    args = ap.parse_args()

    import chorderator as cdt

    melodies = sorted(
        f for f in os.listdir(args.melody_dir) if f.lower().endswith('.mid'))
    if args.limit > 0:
        melodies = melodies[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    ok, failed = [], []
    for fname in melodies:
        stem = os.path.splitext(fname)[0]
        m = re.search(r'(\d{3})', stem)
        song_id = m.group(1) if m else stem
        mel_path = os.path.join(args.melody_dir, fname)
        out_name = os.path.join(args.out_dir, song_id)
        print(f'\n=== {song_id} ({fname}) ===', flush=True)
        try:
            label = read_phrase_label(song_id, args.labels_dir)
            if not label:
                raise RuntimeError(f'no phrase label for {song_id}')
            key = read_pop909_key(song_id, args.pop909_dirs)
            src = 'annotation'
            if key is None:
                key = ks_estimate_key(mel_path)
                src = 'KS-estimated'
            tonic, mode = key
            lb, mb = label_bar_count(label), melody_bar_count(mel_path)
            print(f'  label   : {label}  ({lb} bars)')
            print(f'  melody  : {mb} bars')
            print(f'  key     : {tonic} {mode}  [{src}]')
            if abs(lb - mb) > 2:
                print(f'  WARNING: bar-count mismatch (label {lb} vs '
                      f'melody {mb}) -- alignment may be off')

            cdt.set_melody(mel_path)
            cdt.set_meta(tonic=tonic,
                         mode=('min' if mode.startswith('min') else 'maj'))
            cdt.set_segmentation(label)
            cdt.set_note_shift(0)
            cdt.set_output_style(cdt.Style.POP_STANDARD)
            cdt.generate_save(out_name, task='chord_and_textured_chord',
                              log=True, wav=False)

            produced = [p for p in os.listdir(out_name)
                        if p.endswith('.mid')] if os.path.isdir(out_name) else []
            if not any('chord_gen' in p for p in produced):
                raise RuntimeError(f'no chord_gen.mid produced (saw {produced})')
            print(f'  OK -> {out_name}: {sorted(produced)}')
            ok.append(song_id)
        except Exception as e:                    # noqa: BLE001
            print(f'  FAILED: {e!r}')
            traceback.print_exc(limit=3)
            failed.append((song_id, repr(e)))
            shutil.rmtree(out_name, ignore_errors=True)

    print('\n================ SUMMARY ================')
    print(f'ok: {len(ok)}/{len(melodies)}')
    if failed:
        print('failed:')
        for sid, err in failed:
            print(f'  {sid}: {err[:120]}')
    # Smoke-test semantics: any failure in a small --limit run is a
    # setup problem worth fixing before the full sweep.
    if failed and args.limit > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
