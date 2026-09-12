"""External SOTA baseline: the Anticipatory Music Transformer, prompted.

The published model (Thickstun et al., arXiv:2306.08620; weights on the
HuggingFace hub under stanford-crfm) is a general-MIDI autoregressive
model over ARRIVAL-TIME event triplets (time, duration, note), with
note = pitch x instrument. It is trained on Lakh MIDI, not on POP909,
and it is the only external reference here that generates BOTH streams
itself rather than harmonising one given the other.

The E1 arm is co-generation, so this runs it the way E1 runs every
other system: prompt with the first PROMPT_LENGTH frames of both
streams and let the model continue to GEN_LENGTH. The anticipation
machinery (controls interleaved DELTA seconds ahead) is what the model
was trained with and stays available, but is NOT used here -- controls
would make this conditional generation, which is E3's question, not
E1's. `generate` runs in its own AR mode when no controls are passed.

    prompt (merged, melody program 0 / chord program 48)
      -> midi_to_events          their tokenizer
      -> clip to [0, prompt_s]   so nothing after the boundary leaks in
                                 as an anticipated control
      -> generate(start=prompt_s, end=total_s)
      -> events_to_midi
      -> split by program into named MELODY / CHORD instruments

TIME BASE. Our frames are sixteenth notes at a fixed 120 BPM, so one
frame is 0.125 s and the arithmetic is exact: an 80-frame prompt is
10 s, a 416-frame window is 52 s. Their events_to_midi writes
ticks_per_beat = TIME_RESOLUTION//2 at two beats per second, i.e. 120
BPM, so the output lands on the same grid the rest of E1 uses.

STREAM SEPARATION is the one real adaptation, and it is disclosed
rather than hidden. Their sampler only restricts the instrument set
once 15 instruments are in play (`instr_logits`), so a two-instrument
prompt leaves the model free to bring in a third -- and a note on a
third instrument belongs to neither of our streams. --restrict-instruments
(default on) masks note logits to the instruments the prompt uses,
reusing their own masking code through the same hook. Turn it off to
measure what the unmodified model does; the TSV records how many notes
then land off-stream.

Usage (via infer_amt_prompted.sbatch, GPU):
    python amt_prompted.py --merged-folder temp/E1/prompts/merged_tagged \\
        --out-dir temp/E1/AMT --prompt-frames 80 --gen-frames 416 \\
        --n-samples 3 --model stanford-crfm/music-medium-800k
"""
import argparse
import csv
import os
from glob import glob


FRAMES_PER_SECOND = 8.0          # sixteenths at 120 BPM


def restrict_instruments(instrs):
    """Mask note logits to `instrs`, via the sampler's own hook.

    Their instr_logits already masks exactly this way; it just declines
    to do so below 15 instruments. Replacing it with an unconditional
    version over the prompt's instrument set is the smallest change
    that keeps the output splittable into two streams.
    """
    from anticipation.config import MAX_INSTR, MAX_PITCH
    from anticipation.vocab import NOTE_OFFSET

    keep = set(int(i) for i in instrs)

    def _instr_logits(logits, full_history):
        for instr in range(MAX_INSTR):
            if instr not in keep:
                logits[NOTE_OFFSET + instr * MAX_PITCH:
                       NOTE_OFFSET + (instr + 1) * MAX_PITCH] = -float('inf')
        return logits

    return _instr_logits


def split_by_program(mid, mel_program, chord_program):
    """Their events_to_midi writes one track per instrument, unnamed.
    eval_metrics splits streams by NAME first, so name them here; notes
    on any other instrument are returned separately rather than being
    quietly folded into one of the two.
    """
    import pretty_midi as pm

    out = pm.PrettyMIDI(initial_tempo=120.0)
    mel = pm.Instrument(program=mel_program, name='MELODY')
    chd = pm.Instrument(program=chord_program, name='CHORD')
    stray = []
    for inst in mid.instruments:
        if inst.is_drum:
            stray.extend(inst.notes)
        elif inst.program == mel_program:
            mel.notes.extend(inst.notes)
        elif inst.program == chord_program:
            chd.notes.extend(inst.notes)
        else:
            stray.extend(inst.notes)
    out.instruments += [mel, chd]
    return out, len(mel.notes), len(chd.notes), len(stray)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--merged-folder', required=True,
                   help="E1's prompts/merged_tagged: one file per song "
                        'holding both streams, melody on --mel-program and '
                        'chord on --chord-program')
    p.add_argument('--out-dir', required=True,
                   help='writes <out-dir>/<song>/co/sample_<i>.mid, the '
                        "duet_multi layout E1 scores")
    p.add_argument('--model', default='stanford-crfm/music-medium-800k')
    p.add_argument('--prompt-frames', type=int, default=80)
    p.add_argument('--gen-frames', type=int, default=416,
                   help='TOTAL frames including the prompt, matching E1')
    p.add_argument('--n-samples', type=int, default=3)
    p.add_argument('--top-p', type=float, default=0.98,
                   help="the package README's sampling default")
    p.add_argument('--mel-program', type=int, default=0)
    p.add_argument('--chord-program', type=int, default=48)
    p.add_argument('--restrict-instruments', dest='restrict',
                   action='store_true', default=True)
    p.add_argument('--no-restrict-instruments', dest='restrict',
                   action='store_false',
                   help='let the model choose any instrument, as published. '
                        'The output is then not reliably two streams; the '
                        'TSV records the notes that land off-stream.')
    p.add_argument('--skip-existing', action='store_true',
                   help='skip a song whose samples are all written')
    p.add_argument('--ids', nargs='*', default=None)
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from anticipation import ops
    from anticipation.convert import midi_to_events, events_to_midi
    import anticipation.sample as amt_sample

    prompt_s = a.prompt_frames / FRAMES_PER_SECOND
    total_s = a.gen_frames / FRAMES_PER_SECOND
    print(f'[amt] prompt {a.prompt_frames} frames = {prompt_s:g}s; '
          f'total {a.gen_frames} frames = {total_s:g}s; '
          f'n_samples={a.n_samples} top_p={a.top_p}')

    files = sorted(glob(os.path.join(a.merged_folder, '*.mid')))
    if a.ids:
        want = {str(i).zfill(3) for i in a.ids}
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in want]
    if not files:
        raise SystemExit(f'no midis in {a.merged_folder}')
    print(f'[amt] {len(files)} songs')

    print(f'[amt] loading {a.model}')
    model = AutoModelForCausalLM.from_pretrained(a.model).cuda().eval()

    if a.restrict:
        amt_sample.instr_logits = restrict_instruments(
            {a.mel_program, a.chord_program})
        print(f'[amt] note logits restricted to instruments '
              f'{{{a.mel_program}, {a.chord_program}}} -- see the module '
              f'docstring; --no-restrict-instruments for the published '
              f'behaviour')

    rows, ok, failed = [], [], []
    for idx, f in enumerate(files):
        sid = os.path.splitext(os.path.basename(f))[0]
        out_dir = os.path.join(a.out_dir, sid, 'co')
        print(f'\n[{idx + 1}/{len(files)}] {sid}', flush=True)
        if a.skip_existing and all(
                os.path.exists(os.path.join(out_dir, f'sample_{i}.mid'))
                for i in range(a.n_samples)):
            print('  all samples already on disk, skipped')
            ok.append(sid)
            continue
        try:
            events = midi_to_events(f)
            # Anything after the boundary would be re-read by generate()
            # as an anticipated CONTROL -- the model would be told the
            # future it is meant to be predicting. Clip it away.
            prompt = ops.clip(events, 0, prompt_s, clip_duration=False,
                              seconds=True)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(a.n_samples):
                with torch.no_grad():
                    gen = amt_sample.generate(
                        model, start_time=prompt_s, end_time=total_s,
                        inputs=prompt, top_p=a.top_p)
                mid = events_to_midi(ops.combine(gen, []))
                split, n_mel, n_chd, n_stray = split_by_program(
                    mid, a.mel_program, a.chord_program)
                path = os.path.join(out_dir, f'sample_{i}.mid')
                split.write(path)
                rows.append(dict(id=sid, sample=i, mel_notes=n_mel,
                                 chord_notes=n_chd, off_stream_notes=n_stray))
                print(f'  sample {i}: mel {n_mel}, chord {n_chd}'
                      + (f', OFF-STREAM {n_stray} (dropped)'
                         if n_stray else ''))
            ok.append(sid)
        except Exception as e:                            # noqa: BLE001
            import traceback
            print(f'  FAILED {sid}: {e!r}')
            traceback.print_exc(limit=4)
            failed.append(sid)

    if rows:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, 'amt_notes.tsv'), 'w',
                  newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                               delimiter='\t')
            w.writeheader()
            w.writerows(rows)
        stray = sum(r['off_stream_notes'] for r in rows)
        kept = sum(r['mel_notes'] + r['chord_notes'] for r in rows)
        print(f'\n[amt] {kept} notes kept, {stray} dropped as off-stream '
              f'({100.0 * stray / max(kept + stray, 1):.2f}%)')
        if stray and a.restrict:
            print('[amt] WARNING: off-stream notes with the instrument '
                  'restriction ON means the mask is not being applied -- '
                  'check that anticipation.sample.instr_logits is the hook '
                  'add_token actually calls in the installed version.')

    print(f'\nok: {len(ok)}/{len(files)}')
    if failed:
        print(f'FAILED: {failed}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
