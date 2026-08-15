"""Run the published YinYang conditional baseline (Y-mc / Y-dn) under OUR
E3 protocol, and lay the outputs out so the existing scoring chain reads
them directly.

The upstream driver (cp_transformer_yinyang_inference.eval_model) runs a
hardcoded demo file list. This wrapper keeps its model loading and its
cond_continuation generation untouched, but:

  * loops over a FOLDER of two-stream source midis (the same songs as
    the rest of E3), with our prompt length / gen length / sample count;
  * relocates each generated sample into the `duet_multi` manifest
    layout (<out>/<song>/<mode>/sample_<i>_temp<T>.mid) under the
    E3 direction label, so build_eval_manifest scans it with
    --source "Y:duet_multi:<out>" and no sed is needed;
  * for melchord, renames the tracks to MELODY/CHORD by ROLE so
    eval_metrics' named-track branch attributes streams correctly --
    cond_continuation writes unnamed instruments whose programs are
    direction-dependent (its fixed_program argument), which would
    otherwise mis-attribute under the default program fallback. For
    drumnondrum the is_drum flag already separates streams; names are
    left alone.

Presets (task, direction) -> upstream config, mirroring eval_model's
hardcoded blocks:

  melchord    mel2chord  : ins_ids=[track-0, track-1], fixed_program=[64, 0]
              chord2mel  : ins_ids=[track-1, track-0], fixed_program=[0, 64]
              (Nottingham two-track layout: track 0 = melody, 1 = chord.
               Y-mc checkpoints: '_chord_mel' ckpts, '_rev' = mel->chord.)
  drumnondrum drum2nondrum: ins_ids=[drum, nondrum]   (Y-dn: 'drums_nondrum')
              nondrum2drum: ins_ids=[nondrum, drum]   ('_rev' ckpts)

Pick the checkpoint whose own direction matches --direction (the ckpt
filename encodes it); this wrapper cannot check that for you.

Usage:
    python yinyang_e3_driver.py \
        --ckpt mel_to_chord/<file>.ckpt \
        --midi-folder ~/nottingham-heldout \
        --task melchord --direction mel2chord \
        --output-dir temp/e3_yinyang_mc \
        --prompt-length 64 --gen-length 384 --n-samples 3
"""

import argparse
import os
import shutil
from glob import glob

import mido
import pretty_midi
import torch

from cp_transformer_fine_tune import RoFormerSymbolicTransformerInjected
from cp_transformer_yinyang import RoformerYinyang
from cp_transformer_yinyang_inference import cond_continuation

# --- peft compatibility shim -------------------------------------------
# Recent peft versions duck-type a ZERO-ARG `get_base_model()` on the
# model they wrap (tied-embedding check in inject_adapter). The CP
# lineage defines an unrelated `get_base_model(config)` (builds the HF
# RoFormer backbone), so peft's call crashes with a missing-argument
# TypeError. Upstream YinYang ran an older peft without the check.
# Shim: zero-arg calls return the module itself (what peft expects of a
# raw base model); explicit config calls keep the original behaviour.
_orig_get_base_model = RoFormerSymbolicTransformerInjected.get_base_model


def _peft_compatible_get_base_model(self, config=None):
    if config is None:
        return self
    return _orig_get_base_model(self, config)


RoFormerSymbolicTransformerInjected.get_base_model = (
    _peft_compatible_get_base_model
)

PRESETS = {
    ('melchord', 'mel2chord'): dict(
        ins_ids=['track-0', 'track-1'], fixed_program=[64, 0],
        cond_name='MELODY', target_name='CHORD'),
    ('melchord', 'chord2mel'): dict(
        ins_ids=['track-1', 'track-0'], fixed_program=[0, 64],
        cond_name='CHORD', target_name='MELODY'),
    ('drumnondrum', 'drum2nondrum'): dict(
        ins_ids=['drum', 'nondrum'], fixed_program=[None, None],
        cond_name=None, target_name=None),
    ('drumnondrum', 'nondrum2drum'): dict(
        ins_ids=['nondrum', 'drum'], fixed_program=[None, None],
        cond_name=None, target_name=None),
}


def source_tempo(path, default=120.0):
    """Initial tempo of the source midi. cond_continuation renders BOTH
    streams on the frame grid at output_bpm, so passing the source's own
    tempo keeps the result at the song's real speed; the upstream default
    (150, or the 120 this driver used to hardcode) plays every song at
    one fixed tempo. Scoring is unaffected either way -- eval_metrics
    frames each file by its own tempo -- but listening is not."""
    try:
        pm = pretty_midi.PrettyMIDI(path)
        _, tempi = pm.get_tempo_changes()
        t = float(tempi[0]) if len(tempi) else default
        return t if t > 0 else default
    except Exception as e:
        print(f'[tempo] failed to read {path}: {e!r}; using {default}')
        return default


def load_yinyang(ckpt):
    """Mirror eval_model's resolution: accept ckpt/<name>, <dir>/<name>,
    or an absolute path."""
    for cand in (ckpt, os.path.join('ckpt', ckpt)):
        if os.path.isfile(cand):
            model = RoformerYinyang.load_from_checkpoint(cand, strict=False)
            model.save_name = os.path.basename(cand)
            model.cuda()
            model.eval()
            print(f'[load] {cand}')
            return model
    raise FileNotFoundError(f'checkpoint not found: {ckpt!r} '
                            f'(tried as-is and under ckpt/)')


def rename_streams(path, target_program, target_name, cond_name):
    """Name tracks by ROLE (melchord only): the instrument carrying
    target_program is the generated stream; everything else non-drum is
    the conditioning stream. Uses mido so timing is untouched."""
    mid = mido.MidiFile(path)
    for track in mid.tracks:
        prog = None
        has_notes = False
        for msg in track:
            if msg.type == 'program_change':
                prog = msg.program
            if msg.type == 'note_on':
                has_notes = True
        if not has_notes:
            continue
        name = target_name if prog == target_program else cond_name
        # drop existing track_name events, insert ours at t=0
        track[:] = [m for m in track if m.type != 'track_name']
        track.insert(0, mido.MetaMessage('track_name', name=name, time=0))
    mid.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True,
                   help='YinYang ckpt (path, or name under ckpt/)')
    p.add_argument('--midi-folder', required=True,
                   help='two-stream source midis (both tracks in one file)')
    p.add_argument('--task', required=True,
                   choices=['melchord', 'drumnondrum'])
    p.add_argument('--direction', required=True,
                   choices=['mel2chord', 'chord2mel',
                            'drum2nondrum', 'nondrum2drum'])
    p.add_argument('--output-dir', required=True,
                   help='duet_multi layout root for the manifest')
    p.add_argument('--prompt-length', type=int, default=64)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--n-samples', type=int, default=3)
    p.add_argument('--output-bpm', type=float, default=None,
                   help='render tempo. Default: each source file\'s own '
                        'initial tempo (so the result plays at the song\'s '
                        'real speed). Pass a number to force one tempo for '
                        'every song, as the upstream demo driver does.')
    p.add_argument('--max-songs', type=int, default=None)
    args = p.parse_args()

    key = (args.task, args.direction)
    if key not in PRESETS:
        raise SystemExit(f'direction {args.direction!r} is not defined for '
                         f'task {args.task!r}')
    preset = PRESETS[key]

    model = load_yinyang(args.ckpt)

    files = sorted(glob(os.path.join(args.midi_folder, '*.mid'))
                   + glob(os.path.join(args.midi_folder, '*.MID')))
    if args.max_songs:
        files = files[:args.max_songs]
    if not files:
        raise SystemExit(f'no midis in {args.midi_folder}')
    print(f'{len(files)} source midis; direction={args.direction} '
          f'ins_ids={preset["ins_ids"]}')

    # cond_continuation writes to temp/prompt{P}_{save_name}/...
    src_dir = f'temp/prompt{args.prompt_length}_{model.save_name}'
    ins_str = ','.join(preset['ins_ids'])

    for i, f in enumerate(files):
        base = os.path.basename(f)
        sid = os.path.splitext(base)[0]
        print(f'=== [{i + 1}/{len(files)}] {sid}')
        try:
            out_bpm = (args.output_bpm if args.output_bpm is not None
                       else source_tempo(f))
            print(f'  [tempo] output_bpm={out_bpm:.2f}')
            cond_continuation(
                model, f,
                prompt_length=args.prompt_length,
                generation_length=args.gen_length,
                temperature=args.temperature,
                n_samples=args.n_samples,
                ins_ids=preset['ins_ids'],
                fixed_program=preset['fixed_program'],
                output_bpm=out_bpm,
            )
        except Exception as e:
            print(f'  failed: {e!r}')
            continue

        mode_dir = os.path.join(args.output_dir, sid, args.direction)
        os.makedirs(mode_dir, exist_ok=True)
        for s in range(args.n_samples):
            src = os.path.join(
                src_dir,
                f'{base}_temp{args.temperature}_continuation_{s}'
                f'[{ins_str}].mid')
            if not os.path.exists(src):
                print(f'  [warn] missing sample: {src}')
                continue
            dst = os.path.join(
                mode_dir, f'sample_{s}_temp{args.temperature}.mid')
            shutil.copyfile(src, dst)
            if preset['target_name'] is not None:
                rename_streams(dst,
                               target_program=preset['fixed_program'][1],
                               target_name=preset['target_name'],
                               cond_name=preset['cond_name'])
        print(f'  -> {mode_dir}')

    print(f'\ndone. Manifest source line:\n'
          f'  --source "Y:duet_multi:{args.output_dir}"')


if __name__ == '__main__':
    main()
