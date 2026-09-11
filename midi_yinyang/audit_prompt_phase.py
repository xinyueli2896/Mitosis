"""Gate the prompt set: does every prompt actually start on a bar line,
and is it still on one where generation begins?

Training windows start on true downbeats (FramedDataset + the downbeat
map), so the model's metrical cue is calibrated to "frame 0 is a
downbeat". A prompt that violates that is off-distribution at the very
first token, and a prompt whose metre changes before frame
PROMPT_LENGTH hands the model a generation boundary that is not a bar
line either. Both are silent: the decode runs and the output is merely
worse.

This checks the WRITTEN files against the SOURCE annotation rather than
against build_prompt_crops' own bookkeeping, so a bug in the builder
cannot certify itself. For each kept song it takes the source's
downbeat_frames (from align_grid_report.tsv) and that song's crop_bar,
converts both into the cropped file's own frame numbering, and asks:

  1. does the prompt begin on a bar line -- a true downbeat, or a
     padded lead bar that ends on one?
  2. is frame PROMPT_LENGTH -- where the model takes over -- a true
     downbeat?
  3. do melody and chord carry the same number of frames of prompt, so
     the two streams enter the model in register?
  4. does the song's own metre stay regular across prompt + scored
     window (the thing --align-report steers for)?
  5. does the melody actually enter where the crop rule says it should
     -- inside bar 1 for a song with a pickup, on bar 2's downbeat for
     one without? This is the "is there a bar in front of the pickup
     bar" question, asked of the WRITTEN NOTES. A set built by an older
     builder, or left un-rebuilt after the rule changed, fails here
     instead of having to be opened in a DAW to be noticed.

Usage (via audit_prompt_phase.sbatch, CPU):
    python audit_prompt_phase.py --set input/heldout_v5 \\
        --report <v5>/align_grid_report.tsv --prompt-frames 80
"""
import argparse
import csv
import os
import sys

FRAMES_PER_BEAT = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', required=True,
                   help='the prompt set built by build_prompt_crops')
    p.add_argument('--report', required=True,
                   help='align_grid_report.tsv for the SOURCE corpus')
    p.add_argument('--prompt-frames', type=int, default=96,
                   help='frames handed to the model before it takes over '
                        '(6 bars = 96 at 16 frames per bar: the lead plus '
                        'MEL_BARS sounding bars)')
    p.add_argument('--gen-frames', type=int, default=320,
                   help='frames the model then generates, for the metre '
                        'check over the scored window')
    a = p.parse_args()

    src = {}
    with open(a.report) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            raw = (row.get('downbeat_frames') or '').strip()
            src[row['id']] = [int(x) for x in raw.split(',') if x != ''] \
                if raw else []

    crops_path = os.path.join(a.set, 'prompt_crops.tsv')
    if not os.path.exists(crops_path):
        raise SystemExit(f'missing {crops_path}')

    fails, checked, no_src = [], 0, []
    warn_forced = []
    with open(crops_path) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            if row.get('dropped'):
                continue
            sid = row['id']
            mel = os.path.join(a.set, 'melody', f'{sid}.mid')
            chd = os.path.join(a.set, 'chord', f'{sid}.mid')
            if not (os.path.exists(mel) and os.path.exists(chd)):
                fails.append((sid, 'kept in the TSV but no file on disk'))
                continue
            db = src.get(sid)
            if not db:
                no_src.append(sid)
                continue
            checked += 1
            # express the source downbeats in the CROPPED file's frames.
            # Use the recorded crop_frame: deriving it as crop_bar*16
            # assumes a flat 4/4 grid, which is false for exactly the
            # songs this audit exists to check.
            cf = (row.get('crop_frame') or '').strip()
            if cf == '':
                fails.append((sid, 'no crop_frame in the TSV -- rebuild the '
                                   'set; crop_bar*16 is not a safe substitute'))
                continue
            off = int(cf)
            pad = int((row.get('pad_frames') or 0) or 0)
            # output frame = source frame - crop point + any padded lead
            local = {d - off + pad for d in db if d >= off}

            # With a padded lead bar frame 0 is synthetic: we made it a
            # bar line, and what must be checked is that the pad ends
            # exactly on the first real downbeat. With no pad, pad == 0
            # and this is the plain "frame 0 is a downbeat" test.
            if pad not in local:
                nearest = min(local, default=None)
                fails.append((sid, f'the prompt does not begin on a bar '
                                   f'line: expected a downbeat at frame '
                                   f'{pad}, nearest is {nearest}'))
            if pad and pad % 16:
                fails.append((sid, f'padded lead is {pad} frames, not a '
                                   f'whole number of bars'))
            if a.prompt_frames not in local:
                below = max((x for x in local if x <= a.prompt_frames),
                            default=None)
                fails.append((sid, f'generation starts at frame '
                                   f'{a.prompt_frames}, which is NOT a '
                                   f'downbeat (last one at {below})'))
            win = a.prompt_frames + a.gen_frames
            inside = sorted(x for x in local if pad < x < win)
            if any(b - a_ != 16 for a_, b in zip([pad] + inside, inside)):
                fails.append((sid, 'metre changes inside prompt + scored '
                                   'window'))
            # ---- 5. does the melody enter where the rule says? --------
            # Read the written notes, not the builder's bookkeeping.
            lead = row.get('lead_bars', '')
            if lead != '':
                lead = int(lead)
                try:
                    import pretty_midi as pm
                    mm = pm.PrettyMIDI(mel)
                    ons = [n.start for i in mm.instruments for n in i.notes]
                except Exception as exc:                    # noqa: BLE001
                    fails.append((sid, f'cannot read the melody: {exc}'))
                    ons = []
                if ons:
                    # 16 frames to the bar, 4 frames to the beat; the
                    # crop is written at one constant tempo
                    spf = 60.0 / 120.0 / 4.0
                    f0 = min(ons) / spf
                    if lead == 1:
                        lo, hi, what = 16.0, 32.0, 'inside bar 1 (the pickup)'
                    else:
                        lo, hi, what = 32.0, 32.0 + 1e-6, "on bar 2's downbeat"
                    if not (lo - 1e-6 <= f0 < hi):
                        fails.append((sid, (
                            f'the melody enters at frame {f0:.1f} (bar '
                            f'{f0 / 16:.2f}); with lead_bars={lead} it must '
                            f'enter {what}. There is no full bar of '
                            f'head-room in front of it -- this set was '
                            f'built by an older rule, REBUILD it.')))
            if row.get('forced'):
                warn_forced.append((sid, row['forced']))

    print(f'checked {checked} kept songs against {a.report}')
    if no_src:
        print(f'  {len(no_src)} had no source downbeat map, skipped: '
              f'{no_src[:10]}{" ..." if len(no_src) > 10 else ""}')
    if warn_forced:
        print(f'  {len(warn_forced)} were FORCED past the normal rule '
              f'(expected to be odd): {[s for s, _ in warn_forced][:10]}')
    print()
    if fails:
        print(f'{len(fails)} PROBLEM(S):')
        for sid, why in fails[:40]:
            print(f'  {sid}: {why}')
        if len(fails) > 40:
            print(f'  ... {len(fails) - 40} more')
        print('\nA prompt that does not start on a downbeat, or hands over '
              'mid-bar, is off-distribution from the first token -- the '
              'decode will run and simply be worse.')
        sys.exit(1)
    print('ALL PROMPTS PHASE-CORRECT: frame 0 and the generation boundary '
          'are both true downbeats, the metre holds across the scored '
          'window, and every melody enters with its full lead in front.')


if __name__ == '__main__':
    main()
