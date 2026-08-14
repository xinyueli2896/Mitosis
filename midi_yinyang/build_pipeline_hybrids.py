"""Glue for the PIPELINE co-generation baselines (P-mc / P-cm).

The pipeline: a single-stream specialist freely continues ONE stream
(stage A, cp_transformer_inference), then the YinYang conditional model
generates the OTHER stream conditioned on it (stage B,
yinyang_e3_driver). No mutual conditioning anywhere -- the foil for the
joint duet models in E1.

Two subcommands:

build     stage A outputs + ground-truth partner prompts -> two-track
          hybrid midis the YinYang driver can consume. Track order
          follows the Nottingham convention (track 0 = melody, track 1
          = chord). The generated stream is full length; the partner
          carries ONLY its prompt (first --prompt-frames), which is
          exactly what cond_continuation expects (conditioning track
          given in full, target track prompted). Partner notes are
          resampled by FRAME INDEX from their own tempo grid onto the
          generated file's grid, so mismatched tempi cannot skew
          alignment. Hybrids are named <song>__s<i>.mid, one per
          stage-A sample.

assemble  the YinYang driver's per-hybrid output tree ->
          duet_multi layout for E1 scoring:
          <final>/<song>/co/sample_<i>_temp<T>.mid. The driver already
          named the tracks MELODY/CHORD by role.

Usage:
    python build_pipeline_hybrids.py build \
        --gen-folder temp/<stageA save_name> \
        --partner-folder input/pop909_split/chord \
        --generated-role mel --prompt-frames 96 --temperature 1.0 \
        --out temp/pipeline_mc/hybrids

    python build_pipeline_hybrids.py assemble \
        --ydriver-out temp/pipeline_mc/y \
        --direction mel2chord --temperature 1.0 \
        --final temp/pipeline_mc/P-mc
"""

import argparse
import os
import re
import shutil
import warnings
from glob import glob

import pretty_midi

CONT_RE = re.compile(r'^(?P<song>.+)_temp(?P<temp>[0-9.]+)_continuation_'
                     r'(?P<idx>\d+)\.mid$')
HYBRID_RE = re.compile(r'^(?P<song>.+)__s(?P<idx>\d+)$')


def _bpm(pm):
    _, tempi = pm.get_tempo_changes()
    b = float(tempi[0]) if len(tempi) else 120.0
    return b if b > 0 else 120.0


def _all_notes(pm):
    return [n for inst in pm.instruments if not inst.is_drum
            for n in inst.notes]


def build(args):
    files = sorted(glob(os.path.join(args.gen_folder, '*_continuation_*.mid')))
    if not files:
        raise SystemExit(f'no stage-A continuations in {args.gen_folder}')
    os.makedirs(args.out, exist_ok=True)
    n = 0
    missing = []
    for f in files:
        m = CONT_RE.match(os.path.basename(f))
        if not m or float(m.group('temp')) != args.temperature:
            continue
        # cp_transformer_inference names outputs from the basename WITH
        # its extension ('001.mid_temp1.0_continuation_0.mid'), so the
        # captured song still carries '.mid'. Strip it, or every partner
        # lookup asks for '001.mid.mid' and silently finds nothing.
        song = m.group('song')
        if song.lower().endswith('.mid'):
            song = song[:-4]
        idx = int(m.group('idx'))
        partner = None
        for ext in ('.mid', '.MID'):
            cand = os.path.join(args.partner_folder, song + ext)
            if os.path.exists(cand):
                partner = cand
                break
        if partner is None:
            print(f'[skip] {song}: no {song}.mid in {args.partner_folder}')
            missing.append(song)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gen_pm = pretty_midi.PrettyMIDI(f)
            part_pm = pretty_midi.PrettyMIDI(partner)
        gen_bpm = _bpm(gen_pm)
        part_bpm = _bpm(part_pm)
        gen_step = 60.0 / gen_bpm / 4.0
        part_step = 60.0 / part_bpm / 4.0

        # Partner prompt: keep notes STARTING within the first
        # prompt-frames on the partner's own grid; re-express on the
        # generated file's grid by frame index so tempi need not match.
        cutoff = args.prompt_frames * part_step
        partner_inst = pretty_midi.Instrument(
            program=0,
            name=('CHORD' if args.generated_role == 'mel' else 'MELODY'))
        for note in _all_notes(part_pm):
            if note.start >= cutoff:
                continue
            scale = gen_step / part_step
            partner_inst.notes.append(pretty_midi.Note(
                velocity=note.velocity, pitch=note.pitch,
                start=note.start * scale,
                end=min(note.end, cutoff) * scale,
            ))

        gen_inst = pretty_midi.Instrument(
            program=0,
            name=('MELODY' if args.generated_role == 'mel' else 'CHORD'))
        for note in _all_notes(gen_pm):
            gen_inst.notes.append(pretty_midi.Note(
                velocity=note.velocity, pitch=note.pitch,
                start=note.start, end=note.end,
            ))

        out_pm = pretty_midi.PrettyMIDI(initial_tempo=gen_bpm)
        # Nottingham convention: track 0 = melody, track 1 = chord.
        if args.generated_role == 'mel':
            out_pm.instruments.extend([gen_inst, partner_inst])
        else:
            out_pm.instruments.extend([partner_inst, gen_inst])
        out_pm.write(os.path.join(args.out, f'{song}__s{idx}.mid'))
        n += 1
    print(f'built {n} hybrid file(s) -> {args.out}')
    if n == 0:
        have = sorted(os.path.basename(p) for p in
                      glob(os.path.join(args.partner_folder, '*.mid')))[:5]
        raise SystemExit(
            'no hybrids built.\n'
            f'  stage-A files seen : {len(files)} in {args.gen_folder}\n'
            f'  songs without partner: {missing[:5]}\n'
            f'  partner folder holds : {have}\n'
            '  (also check --temperature matches the stage-A filenames)')


def build_cond(args):
    """Stage-B variant for an IN-DOMAIN conditional (C.1/C.2) instead of
    the two-track YinYang driver. Those scripts take two FOLDERS of
    single-stream midis paired by filename -- conditioning (mod_a) and
    the target's prompt (mod_b) -- so no hybrid files are needed: the
    stage-A continuation is already a single-stream midi, and the
    partner ground truth is copied under the matching name (the C
    inference caps it to --prompt-length itself)."""
    files = sorted(glob(os.path.join(args.gen_folder, '*_continuation_*.mid')))
    if not files:
        raise SystemExit(f'no stage-A continuations in {args.gen_folder}')
    os.makedirs(args.out_cond, exist_ok=True)
    os.makedirs(args.out_prompt, exist_ok=True)
    n, missing = 0, []
    for f in files:
        m = CONT_RE.match(os.path.basename(f))
        if not m or float(m.group('temp')) != args.temperature:
            continue
        song = m.group('song')
        if song.lower().endswith('.mid'):
            song = song[:-4]
        idx = int(m.group('idx'))
        partner = None
        for ext in ('.mid', '.MID'):
            cand = os.path.join(args.partner_folder, song + ext)
            if os.path.exists(cand):
                partner = cand
                break
        if partner is None:
            print(f'[skip] {song}: no {song}.mid in {args.partner_folder}')
            missing.append(song)
            continue
        stem = f'{song}__s{idx}.mid'
        shutil.copyfile(f, os.path.join(args.out_cond, stem))
        shutil.copyfile(partner, os.path.join(args.out_prompt, stem))
        n += 1
    print(f'paired {n} (conditioning, prompt) file(s) -> '
          f'{args.out_cond} , {args.out_prompt}')
    if n == 0:
        raise SystemExit(
            f'nothing paired. stage-A files: {len(files)}; '
            f'songs without partner: {missing[:5]}')


def assemble(args):
    # Two producer layouts land here:
    #   Y driver      <root>/<song>__s<i>/<direction>/sample_0_temp<T>.mid
    #   C.1/C.2 (1 sample)  <root>/<song>__s<i>/drum2nondrum_temp<T>.mid
    run_dirs = sorted(d for d in glob(os.path.join(args.ydriver_out, '*'))
                      if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f'no per-song dirs under {args.ydriver_out}')
    n = 0
    for run_dir in run_dirs:
        hybrid_name = os.path.basename(run_dir)
        m = HYBRID_RE.match(hybrid_name)
        if not m:
            print(f'[skip] {hybrid_name}: not a <song>__s<i> name')
            continue
        song, idx = m.group('song'), int(m.group('idx'))
        candidates = [
            os.path.join(run_dir, args.direction,
                         f'sample_0_temp{args.temperature}.mid'),
            os.path.join(run_dir,
                         f'drum2nondrum_temp{args.temperature}.mid'),
        ]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            print(f'[warn] no output for {hybrid_name} '
                  f'(looked for {[os.path.basename(c) for c in candidates]})')
            continue
        dst_dir = os.path.join(args.final, song, 'co')
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, os.path.join(
            dst_dir, f'sample_{idx}_temp{args.temperature}.mid'))
        n += 1
    print(f'assembled {n} sample(s) -> {args.final} (duet_multi, mode=co)')
    if n == 0:
        raise SystemExit('nothing assembled')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build')
    b.add_argument('--gen-folder', required=True,
                   help='stage-A output dir (temp/<save_name>)')
    b.add_argument('--partner-folder', required=True,
                   help='ground-truth folder of the OTHER stream')
    b.add_argument('--generated-role', choices=['mel', 'chord'],
                   required=True,
                   help="which stream stage A generated ('mel' for P-mc)")
    b.add_argument('--prompt-frames', type=int, default=96)
    b.add_argument('--temperature', type=float, default=1.0)
    b.add_argument('--out', required=True)

    c = sub.add_parser('build-cond')
    c.add_argument('--gen-folder', required=True)
    c.add_argument('--partner-folder', required=True)
    c.add_argument('--temperature', type=float, default=1.0)
    c.add_argument('--out-cond', required=True,
                   help='folder for the generated CONDITIONING stream')
    c.add_argument('--out-prompt', required=True,
                   help="folder for the target's ground-truth prompt")

    a = sub.add_parser('assemble')
    a.add_argument('--ydriver-out', required=True)
    a.add_argument('--direction', required=True,
                   choices=['mel2chord', 'chord2mel'])
    a.add_argument('--temperature', type=float, default=1.0)
    a.add_argument('--final', required=True)

    args = p.parse_args()
    {'build': build, 'build-cond': build_cond, 'assemble': assemble}[args.cmd](args)


if __name__ == '__main__':
    main()
