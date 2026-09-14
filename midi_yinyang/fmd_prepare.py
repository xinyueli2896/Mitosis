"""Crop an E1 manifest's midis to the SCORED WINDOW for FMD embedding.

FMD embeds whole files. Handing it the raw outputs would embed the
prompt too -- and the prompt is byte-identical between every system and
the reference, so it is shared signal that pulls every system's distance
toward the reference by the same amount and compresses the differences
the metric exists to show. This is the whole-file-durations mistake in
another guise, so the window is cut first: frames
[--prompt-frames, --total-frames), the same span eval_metrics scores.

Writes, under --out:
    _reference/<song>.mid          one per song
    <system>/<song>__s<sample>.mid one per generated sample

Run in the mitosis env; fmd_score.py then runs in the FMD venv.
"""

import argparse
import os
import warnings

import pretty_midi

BEAT_DIV = 4


def frame_fn(pm):
    """seconds -> frames on the file's tick grid (eval_metrics.frame_fn)."""
    tpf = pm.resolution / BEAT_DIV
    return lambda t: pm.time_to_tick(t) / tpf


MEL_PROGRAMS = {0, 24}
CHORD_PROGRAMS = {48}


def _canon(inst, task='melchord'):
    """Which stream an instrument is, by eval_metrics' own rules."""
    name = (inst.name or '').strip().lower()
    if name in ('melody', 'chord'):
        return name
    if inst.program in MEL_PROGRAMS and not inst.is_drum:
        return 'melody'
    if inst.program in CHORD_PROGRAMS and not inst.is_drum:
        return 'chord'
    return None


def crop(paths, lo, hi, out_path, out_bpm=120.0):
    """Notes whose ONSET falls in [lo, hi), re-timed to start at 0.

    Onset-based, matching how every other metric treats the window: a
    note held over from the prompt belongs to the prompt. Ends are
    clipped to the window so a long final note cannot extend the file
    and change its embedding.
    """
    step = 60.0 / out_bpm / BEAT_DIV
    # CANONICAL two-track output, identical for reference and generated.
    # The reference arrives as two single-stream files and a system's
    # output as one two-track file, with whatever programs each wrote.
    # Handing those differences to the encoder would let it separate
    # reference from generated on instrument assignment alone, which is
    # not what the distance is supposed to measure. Both are re-emitted
    # as MELODY on program 0 and CHORD on program 48.
    buckets = {'melody': [], 'chord': []}
    for p in paths:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pm = pretty_midi.PrettyMIDI(p)
        to_f = frame_fn(pm)
        for inst in pm.instruments:
            which = _canon(inst)
            if which is None:
                continue
            for n in inst.notes:
                f0 = to_f(n.start)
                if not (lo <= f0 < hi):
                    continue
                f1 = min(to_f(n.end), hi)
                buckets[which].append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=(f0 - lo) * step,
                    end=max(f1 - lo, f0 - lo + 1) * step))
    if not any(buckets.values()):
        return False
    out = pretty_midi.PrettyMIDI(initial_tempo=out_bpm)
    for which, prog in (('melody', 0), ('chord', 48)):
        if buckets[which]:
            inst = pretty_midi.Instrument(program=prog, name=which.upper())
            inst.notes = buckets[which]
            out.instruments.append(inst)
    out.write(out_path)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ref-a-dir', required=True)
    p.add_argument('--ref-b-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--prompt-frames', type=int, default=96)
    p.add_argument('--total-frames', type=int, default=416)
    p.add_argument('--ref-windows', type=int, default=1,
                   help='cut this many consecutive windows of the scored '
                        'length from each REFERENCE song instead of one. '
                        'The reference covariance is estimated from N '
                        'points in 768 dimensions and N is the binding '
                        'constraint; extra windows are more real music '
                        'from the same songs, which is what the reference '
                        'distribution is meant to describe. The generated '
                        'side is untouched -- systems only produced the '
                        'one window.')
    args = p.parse_args()

    lo, hi = args.prompt_frames, args.total_frames
    ref_dir = os.path.join(args.out, '_reference')
    os.makedirs(ref_dir, exist_ok=True)

    seen_ref, n_gen, empty = set(), 0, []
    for line in open(args.manifest):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        system, _mode, song, sample, path = line.split('\t')[:5]
        d = os.path.join(args.out, system)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, f'{song}__s{sample}.mid')
        if not os.path.exists(dst):
            if crop(path.split(';'), lo, hi, dst):
                n_gen += 1
            else:
                empty.append(f'{system}/{song}#{sample}')
        else:
            n_gen += 1
        if song not in seen_ref:
            span = hi - lo
            srcs = [os.path.join(args.ref_a_dir, f'{song}.mid'),
                    os.path.join(args.ref_b_dir, f'{song}.mid')]
            for w in range(args.ref_windows):
                suffix = '' if w == 0 else f'__w{w}'
                rdst = os.path.join(ref_dir, f'{song}{suffix}.mid')
                if os.path.exists(rdst):
                    continue
                if not crop(srcs, lo + w * span, hi + w * span, rdst):
                    break      # ran off the end of the song
            seen_ref.add(song)

    print(f'[fmd-prep] {len(seen_ref)} reference songs, {n_gen} generated '
          f'files -> {args.out}')
    if empty:
        # An empty crop is a silent continuation. It cannot be embedded,
        # so it silently leaves that system with fewer points than the
        # others -- which flatters it, since a Frechet estimate improves
        # with n. Name them.
        print(f'[fmd-prep] WARNING {len(empty)} crops held no notes and were '
              f'skipped: {" ".join(empty[:10])}'
              + (' ...' if len(empty) > 10 else ''))


if __name__ == '__main__':
    main()
