"""Why does a merged single-stream output (S1 / S0 / S-scratch) carry only
ONE program, when its prompt carried two?

In the merged representation the PROGRAM *is* the stream identity: the
tokenizer separates melody from chord by program alone (which is why
merge_melody_chord tags chords as 48). So an all-program-0 output means
the chord stream is genuinely absent, not merely re-timbred. This is
specific to the merged models -- the duet models (A.1/A.2/B.1) carry
stream identity in the architecture and write both streams on program 0
BY DESIGN, distinguished by track name, so they are not affected.

Three places the second program can be lost, checked in pipeline order:

  1. TRAINING DATA   the merged .pt was built from an untagged folder,
                     so every note is program 0 and the model never saw
                     a second program.
  2. PROMPT / RESTORE the prompt itself lacks the tagged program, or
                     tag_chord_track re-mapped it: that helper defaults
                     orig_prog to 0 when the CHORD track has no
                     program_change, so 48 is restored to 0 on output.
  3. THE MODEL       data and prompt are fine, the prompt region of the
                     output still shows both programs, but the GENERATED
                     region has only one -- the model never emits the
                     chord program. Sparse output (few notes/frame) then
                     points at the local decoder emitting EOS after the
                     first slot, which in training is the melody note.

Usage:
    python check_program_collapse.py \
        --data data/pop909_melchord_cp16_v2.pt \
        --prompt-folder temp/e1_melchord_165497/prompts/merged_tagged \
        --out temp/e1_melchord_Sscratch_p96 \
        --prompt-frames 96 --chord-program 48
"""

import argparse
import os
import warnings
from collections import Counter
from glob import glob

import pretty_midi

try:
    import mido
except ImportError:
    mido = None


def _tempo(pm, default=120.0):
    _, tempi = pm.get_tempo_changes()
    t = float(tempi[0]) if len(tempi) else default
    return t if t > 0 else default


def check_data(path, chord_program):
    print(f'\n[1] TRAINING DATA  {path}')
    if not path:
        print('    (skipped: --data not given)')
        return None
    if not os.path.exists(path):
        print(f'    NOT FOUND -- skipping')
        return None
    try:
        import numpy as np
        import torch
    except ImportError as e:
        print(f'    cannot load ({e!r}); skipping')
        return None
    arr = np.asarray(torch.load(path, weights_only=True))
    progs = arr.reshape(-1, 4)[:, 0]
    progs = progs[progs < 128]          # 255 = pad, 254 = eos
    vals, counts = np.unique(progs, return_counts=True)
    dist = dict(zip(vals.tolist(), counts.tolist()))
    print(f'    programs: {dist}')
    if chord_program not in dist:
        print(f'    *** program {chord_program} ABSENT from the training '
              f'data: the merged .pt was built from an UNTAGGED folder.')
        return False
    print(f'    program {chord_program} present '
          f'({dist[chord_program]:,} notes) -- data is fine.')
    return True


def check_prompts(folder, chord_program, limit=3):
    print(f'\n[2] PROMPT FILES  {folder}')
    if not folder:
        print('    (skipped: --prompt-folder not given)')
        return None
    files = sorted(glob(os.path.join(folder, '*.mid'))
                   + glob(os.path.join(folder, '*.MID')))[:limit]
    if not files:
        print('    no midis found -- skipping')
        return None
    ok = True
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pm = pretty_midi.PrettyMIDI(f)
        insts = [(i.name, int(i.program), len(i.notes))
                 for i in pm.instruments]
        print(f'    {os.path.basename(f)}: {insts}')
        progs = {i.program for i in pm.instruments}
        if chord_program not in progs:
            ok = False
            # the restore_map trap: CHORD track with no program_change
            if mido is not None:
                for t in mido.MidiFile(f).tracks:
                    if (t.name or '').strip().lower() == 'chord':
                        has_pc = any(m.type == 'program_change' for m in t)
                        if not has_pc:
                            print('      *** track named CHORD has NO '
                                  'program_change: tag_chord_track will '
                                  f'restore {chord_program} -> 0 on output.')
    if ok:
        print(f'    all sampled prompts carry program {chord_program}.')
    else:
        print(f'    *** program {chord_program} MISSING from prompts.')
    return ok


def check_outputs(out_dir, prompt_frames, chord_program, limit=4):
    print(f'\n[3] GENERATED OUTPUTS  {out_dir}')
    files = sorted(glob(os.path.join(out_dir, '**', '*.mid'), recursive=True))
    files = [f for f in files if '_prompt' not in os.path.basename(f)][:limit]
    if not files:
        print('    no midis found')
        return None
    prompt_has = gen_has = False
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pm = pretty_midi.PrettyMIDI(f)
        step = 60.0 / _tempo(pm) / 4.0
        pre, gen = Counter(), Counter()
        gen_frames = set()
        for inst in pm.instruments:
            for n in inst.notes:
                fr = int(round(n.start / step))
                if fr < prompt_frames:
                    pre[inst.program] += 1
                else:
                    gen[inst.program] += 1
                    gen_frames.add(fr)
        dens = (sum(gen.values()) / len(gen_frames)) if gen_frames else 0.0
        fmt = lambda c: {int(k): int(v) for k, v in sorted(c.items())}
        print(f'    {os.path.basename(f)[:44]:46s}'
              f'prompt={fmt(pre)}  generated={fmt(gen)}  '
              f'notes/active-frame={dens:.2f}')
        prompt_has |= chord_program in pre
        gen_has |= chord_program in gen
    return prompt_has, gen_has


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=None,
                   help='merged training .pt (e.g. pop909_melchord_cp16_v2.pt)')
    p.add_argument('--prompt-folder', default=None,
                   help='merged, program-tagged prompts fed to inference')
    p.add_argument('--out', required=True,
                   help='the temp/<save_name> dir holding the continuations')
    p.add_argument('--prompt-frames', type=int, default=96)
    p.add_argument('--chord-program', type=int, default=48)
    args = p.parse_args()

    print('=' * 68)
    print('SINGLE-STREAM PROGRAM-COLLAPSE DIAGNOSTIC')
    print(f'chord program under test: {args.chord_program}')
    print('=' * 68)

    data_ok = check_data(args.data, args.chord_program)
    prompts_ok = check_prompts(args.prompt_folder, args.chord_program)
    out = check_outputs(args.out, args.prompt_frames, args.chord_program)

    print('\n' + '=' * 68)
    print('VERDICT')
    if data_ok is False:
        print(f'  DATA: the training set has no program {args.chord_program}.')
        print('  The model could not possibly emit it. Rebuild the merged')
        print('  dataset from the --chord-program 48 folder')
        print('  (preprocess_pop909_melchord.sbatch) and retrain.')
    elif prompts_ok is False:
        print('  PROMPT/RESTORE: the prompts fed to inference lack the chord')
        print('  program (or tag_chord_track restores it to 0 because the')
        print('  CHORD track carries no program_change). Fix the prompt')
        print('  folder -- the model is not at fault.')
    elif out is None:
        print('  No outputs to judge; rerun with a valid --out.')
    else:
        prompt_has, gen_has = out
        if gen_has:
            print('  NO COLLAPSE: the generated region contains both')
            print('  programs. If it still sounds like one instrument, you')
            print('  are listening to a *_piano copy.')
        elif prompt_has:
            print('  MODEL: prompt region has both programs, generated region')
            print(f'  has only one -- the model never emits program '
                  f'{args.chord_program}.')
            print('  For a from-scratch model this is a capability failure,')
            print('  not a pipeline bug: without the multi-instrument')
            print('  pretrain it never learned the program token. A low')
            print('  notes/active-frame above points at the local decoder')
            print('  emitting EOS after the first slot (the melody note), so')
            print('  chord notes are never written at all.')
            print('  Report it as measured: in the merged representation the')
            print('  program IS the stream identity, so this model cannot')
            print('  maintain two streams -- the sharpest form of the H1')
            print('  weakness the duet representation avoids by construction.')
        else:
            print('  Neither region carries the chord program -- the input to')
            print('  inference was already single-program. Check [2].')
    print('=' * 68)


if __name__ == '__main__':
    main()
