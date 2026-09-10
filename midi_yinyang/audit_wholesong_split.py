"""Match our held-out POP909 songs with whole-song-gen's validation split.

whole-song-gen trains on POP909 with a 9:1 train/valid split
(data/pop909_split/split.npz in their data payload; valid index i is
song i+1). They have no separate test set -- valid IS their held-out
set. Our A-family, specialists and S1 train on the POP909 .pt datasets
with the FramedDataset rule: a song is held out (split='val') iff its
dataset index is a multiple of 10; plus the 5 songs cut out entirely
into input/pop909_split (the E1 test set so far).

A song is a fair test song for a table containing WS only if NEITHER
side trained on it. This script prints:

  * their valid ids, and whether split.npz equals the seed-1234
    regeneration of their create_train_valid_split (it should)
  * our val ids (from data/<dataset>.txt + .length.pt) and our test ids
  * where each of our current test songs sits in THEIR split
  * the SHARED-CLEAN set: their valid  ∩  (our val ∪ our test)

and, with --stage-dir, copies the shared-clean songs' melody/chord midis
UNCROPPED into <stage-dir>/p<N>/{melody,chord}/, N being the smallest
prompt length in bars (--prompt-bars, default 6 8) at which both streams
are present, so E1 runs once per group (PROMPT_LENGTH=16N) and
merge_e1.sbatch scores the groups together.

Usage (via audit_wholesong_split.sbatch):
    python audit_wholesong_split.py --split external/whole_song_gen/data/pop909_split/split.npz
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
import torch

TRAIN_LENGTH = 384      # FramedDataset target length (cp_transformer_m2c_moe)


def song_id(name):
    """'001.mid', '/x/y/001-v1.mid', '001' -> 1; None if no 3-digit id."""
    m = re.search(r'(?<!\d)(\d{3})(?!\d)', os.path.basename(name))
    return int(m.group(1)) if m else None


def their_split(fn):
    z = np.load(fn)
    return (set((z['train_inds'] + 1).tolist()),
            set((z['valid_inds'] + 1).tolist()))


def seed_regen(n=909, train_factor=9, seed=1234):
    np.random.seed(seed)
    ids = np.arange(n)
    n_train = int(n * train_factor / (train_factor + 1))
    tr = np.sort(np.random.choice(ids, n_train, replace=False))
    va = np.sort(np.setdiff1d(ids, tr))
    return set((tr + 1).tolist()), set((va + 1).tolist())


def our_val(dataset_txt, length_pt, split_ratio=10):
    names = {}
    with open(dataset_txt) as f:
        for line in f:
            i, name = line.rstrip('\n').split('\t', 1)
            names[int(i)] = name
    lengths = torch.load(length_pt)
    is_valid = (lengths >= TRAIN_LENGTH)
    idx = torch.arange(len(lengths))[is_valid]
    val_idx = idx[idx % split_ratio == 0].tolist()
    train_idx = idx[idx % split_ratio != 0].tolist()
    to_ids = lambda ix: {song_id(names[i]) for i in ix if song_id(names[i]) is not None}
    return to_ids(train_idx), to_ids(val_idx), names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, help='their split.npz')
    p.add_argument('--datasets', nargs='+',
                   default=['pop909_chord_cp4_v2', 'pop909_melody_cp4_v2',
                            'pop909_chord_cp8_v2', 'pop909_melody_cp8_v2',
                            'pop909_cp8_v2_chord_mel',
                            'pop909_melchord_tagged_cp16_v2'],
                   help='dataset stems under data/ (the .txt and .length.pt '
                        'next to each give index -> song). Our systems train '
                        'on different files -- cp4 (A1/A3/specialists), cp8 '
                        '(YinYang), tagged cp16 (S1/S-scratch) -- and each '
                        'has its own index%%10 validation set; the audit '
                        'reports every one and intersects them.')
    p.add_argument('--test-folder', default='input/pop909_split/melody')
    p.add_argument('--mel-src', default='/home/xinyue.li/POP909-Dataset/POP909-melody',
                   help='the TIME-ALIGNED per-song stream folder (absolute); '
                        'the relative POP909-Dataset/... under midi_yinyang '
                        'holds the raw files and must not be used')
    p.add_argument('--chord-src', default='/home/xinyue.li/POP909-Dataset/POP909-chord')
    p.add_argument('--stage-dir', default=None,
                   help='copy shared-clean songs here as {melody,chord}/<id>.mid')
    p.add_argument('--min-prompt-notes', type=int, default=1,
                   help='a staged song must have at least this many notes '
                        'in BOTH streams inside the prompt, else it is '
                        'removed from the stage dir: a prompt with an '
                        'empty stream (an intro before the melody enters, '
                        'a chord track that starts late) is not a '
                        'co-generation prompt, every system gets it wrong '
                        'in its own way, and the song only adds noise.')
    p.add_argument('--prompt-bars', type=int, nargs='+', default=[6, 8],
                   help='candidate prompt lengths in bars; each song gets the '
                        'smallest with both streams present and is staged '
                        'into <stage-dir>/p<N>/')
    p.add_argument('--gen-frames', type=int, default=320,
                   help='frames scored after the prompt (E1: GEN_LENGTH - '
                        'PROMPT_LENGTH); a song whose streams end before '
                        'prompt + this is excluded')
    p.add_argument('--max-offgrid', type=float, default=0.05,
                   help='exclude a song if more than this fraction of its '
                        'onsets are off the 16th tick grid (or cluster at '
                        'thirds: triplet-encoded)')
    p.add_argument('--keep-offgrid', type=int, nargs='*', default=[326, 816],
                   help='song ids whose off-grid flag is reported but does '
                        'not exclude them (checked by ear/score: 326, 816)')
    p.add_argument('--exclude', type=int, nargs='*', default=[],
                   help='song ids dropped from the stage regardless of the checks '
                        '(2026-09-10: 456 -- 2-, 3- and 6-beat bars from bar 2 on, '
                        'so its chords leave the 4-beat grid inside the prompt)')
    p.add_argument('--min-kept', type=int, default=8,
                   help='exit 1 if fewer songs survive the prompt check, so '
                        'a dependent E1 chain holds instead of scoring a '
                        'one-song "table"')
    args = p.parse_args()

    fmt = lambda s: ' '.join(f'{i:03d}' for i in sorted(s))

    ws_train, ws_valid = their_split(args.split)
    rg_train, rg_valid = seed_regen()
    print(f'[theirs] n_train={len(ws_train)} n_valid={len(ws_valid)}')
    print(f'[theirs] valid ids: {fmt(ws_valid)}')
    print(f'[theirs] split.npz == seed-1234 regeneration: '
          f'{"YES" if ws_valid == rg_valid else "NO -- their payload was split differently"}')

    our_train, our_val_ids, names = set(), None, {}
    for stem in args.datasets:
        txt, lp = f'data/{stem}.txt', f'data/{stem}.length.pt'
        if not (os.path.isfile(txt) and os.path.isfile(lp)):
            print(f'[ours] {stem}: files missing under data/ -- skipped')
            continue
        tr, va, nm = our_val(txt, lp)
        print(f'[ours] {stem}: {len(nm)} songs; train={len(tr)} val={len(va)}'
              f' (index % 10 == 0)  val ids: {fmt(va)}')
        our_train |= tr
        our_val_ids = va if our_val_ids is None else (our_val_ids & va)
        names = names or nm
    if our_val_ids is None:
        sys.exit('no dataset files found under data/')
    print(f'[ours] validation songs common to ALL listed datasets: '
          f'{len(our_val_ids)}  ({fmt(our_val_ids)})')
    print(f'[ours] songs in ANY training split: {len(our_train)}')
    our_test = set()
    if os.path.isdir(args.test_folder):
        for f in os.listdir(args.test_folder):
            s = song_id(f)
            if s is not None:
                our_test.add(s)
    print(f'[ours] test ids (cut before preprocessing): {fmt(our_test)}  '
          f'({args.test_folder})')
    in_data = our_test & (our_train | our_val_ids)
    if in_data:
        print(f'[ours] WARNING: test songs present in a dataset file: {fmt(in_data)}')
    both = our_train & ws_valid
    print(f'[ours] our TRAIN songs inside their valid: {len(both)} '
          f'(these are fair for WS but not for us)')

    print('\n[current E1 test songs in THEIR split]')
    for s in sorted(our_test):
        where = ('their TRAIN -- WS trained on it' if s in ws_train
                 else 'their VALID -- clean' if s in ws_valid else 'absent')
        print(f'  {s:03d}: {where}')

    ours_heldout = our_val_ids | our_test
    shared = ours_heldout & ws_valid
    print(f'\n[SHARED-CLEAN] their valid ∩ our held-out (val ∪ test): '
          f'{len(shared)} songs')
    print(f'  {fmt(shared)}')
    print('  neither side trained on these; use them as the E1 test set '
          'for any table that contains WS.')

    # ---- per-system check: are the shared-clean songs outside every E1
    # system's TRAINING split? Two dataset classes are in play:
    #   cp_transformer_m2c_moe.FramedDataset (A1, A3):
    #       train = idx % 10 != 0            val = idx % 10 == 0
    #   cp_transformer.FramedDataset (S1, S-scratch, specialists, YinYang):
    #       train = idx % 10 > 1   val = idx % 10 == 1   test = idx % 10 == 0
    # Under BOTH, idx % 10 == 0 is never trained on -- so a song is safe
    # for a system iff it sits at idx % 10 == 0 in that system's dataset
    # file (or was cut out before preprocessing). That is what is checked
    # here, file by file, for the files each system trains on.
    E1 = [
        ('A1',        'cp_transformer_m2c_moe', ['pop909_chord_cp4_v2', 'pop909_melody_cp4_v2']),
        ('A3',        'cp_transformer_m2c_moe', ['pop909_chord_cp4_v2', 'pop909_melody_cp4_v2']),
        ('S1',        'cp_transformer',         ['pop909_melchord_tagged_cp16_v2']),
        ('S-scratch', 'cp_transformer',         ['pop909_melchord_tagged_cp16_v2']),
        ('P-mc S-mel specialist',   'cp_transformer', ['pop909_melody_cp4_v2']),
        ('P-cm S-chord specialist', 'cp_transformer', ['pop909_chord_cp4_v2']),
        ('P-mc/P-cm YinYang',       'cp_transformer', ['pop909_cp8_v2_chord_mel']),
    ]
    print('\n[E1 clean check] shared-clean songs vs each system\'s training split')
    all_ok = True
    for sysname, cls, stems in E1:
        for stem in stems:
            txt, lp = f'data/{stem}.txt', f'data/{stem}.length.pt'
            if not (os.path.isfile(txt) and os.path.isfile(lp)):
                print(f'  {sysname:<26} {stem:<34} FILE MISSING -- cannot verify')
                all_ok = False
                continue
            tr, va, nm = our_val(txt, lp)
            leaked = shared & tr
            status = 'CLEAN' if not leaked else f'LEAK: {fmt(leaked)}'
            print(f'  {sysname:<26} {stem:<34} {cls:<24} {status}')
            all_ok &= not leaked
    print(f'  WS                         their valid split                           '
          f'{"CLEAN" if shared <= ws_valid else "LEAK"}')
    print(f'  -> {"ALL E1 SYSTEMS CLEAN on the shared set" if all_ok else "NOT clean -- see LEAK / MISSING lines"}')

    if args.stage_dir:
        ok = 0
        # The 5 test songs were MOVED out of the source folders into
        # input/pop909_split/{melody,chord} before preprocessing, so a
        # shared-clean song that is one of them (004) lives there, not
        # in the source folder. Search both.
        test_root = os.path.dirname(args.test_folder)   # input/pop909_split
        for sub, src in (('melody', args.mel_src), ('chord', args.chord_src)):
            dst = os.path.join(args.stage_dir, sub)
            os.makedirs(dst, exist_ok=True)
            sources = [d for d in (src, os.path.join(test_root, sub))
                       if os.path.isdir(d)]
            if not sources:
                print(f'[stage] no source folder for {sub} ({src}) -- not staged')
                continue
            for s in sorted(shared):
                found = None
                for d in sources:
                    cands = [f for f in os.listdir(d) if song_id(f) == s
                             and f.lower().endswith('.mid')]
                    if cands:
                        found = os.path.join(d, cands[0])
                        break
                if found is None:
                    print(f'[stage] {sub}: no midi for song {s:03d} in {sources}')
                    continue
                shutil.copyfile(found, os.path.join(dst, f'{s:03d}.mid'))
                ok += 1
        print(f'[stage] {ok} files -> {args.stage_dir}/{{melody,chord}}')
        n_m = len(os.listdir(os.path.join(args.stage_dir, 'melody')))
        n_c = len(os.listdir(os.path.join(args.stage_dir, 'chord')))
        if n_m != n_c or n_m != len(shared):
            print(f'[stage] WARNING: melody={n_m} chord={n_c} expected={len(shared)}')
            sys.exit(1)

        # ---- prompt check: both streams must be present in the prompt,
        # counted on the TICK grid (frame = tick / (resolution/4)) --
        # the grid the tokenizer reads. The first version of this check
        # framed by the file's first tempo event and, on POP909's
        # spurious initial tempi, judged 12 of 14 prompts empty and
        # deleted them. Also drop songs whose onsets are off the 16th
        # grid IN TICK SPACE (check_tokenizer_grid): tokenization snaps
        # them a 32nd away, so they are corrupted as prompts and as
        # references alike. The grid check must be the TICK-space one:
        # the second version of this block used check_beat_alignment,
        # which measures onsets in seconds against the file's FIRST
        # tempo event -- the same first-tempo bug again -- and on the
        # time-aligned files (job 217364 chain) flagged 13 of 14 songs
        # OFF-GRID/TRIPLET, leaving a 1-song stage and a void E1 table.
        # Meter is still checked from the time-signature events (a 3/4
        # song does not have 16-frame bars, so '6 bars = 96 frames' is
        # wrong for it).
        import mido
        import pretty_midi
        from eval_metrics import frame_fn
        from check_tokenizer_grid import analyze as tick_grid

        def meter_flags(fn):
            m = mido.MidiFile(fn)
            sigs = sorted({(msg.numerator, msg.denominator)
                           for tr in m.tracks for msg in tr
                           if msg.type == 'time_signature'})
            out = []
            bad = [f'{n}/{d}' for n, d in sigs if int(n * 16 / d) != 16]
            if bad:
                out.append(f'METER({",".join(bad)})')
            if len(sigs) > 1:
                out.append('MIXED-METER')
            return out

        # ---- prompt-length groups. Job 217761 (tick-space check on the
        # time-aligned files) found the melody absent from the first 6
        # bars of 8 of the 14 songs: POP909 intros, chords alone with
        # the melody entering at bar 5-14. The intro is kept intact
        # (the decision was NOT to crop it away): instead each song is
        # assigned the SMALLEST prompt length in --prompt-bars (default
        # 6 8) for which both streams have at least --min-prompt-notes
        # onsets inside the prompt, and is staged UNCROPPED into
        # <stage>/p<N>/{melody,chord}. E1 then runs once per group with
        # PROMPT_LENGTH=16*N and GEN_LENGTH=16*N+--gen-frames, so the
        # scored continuation has the same length in every group, and
        # merge_e1.sbatch scores the groups together (prompt length is
        # a per-song property shared by every system, so it cancels in
        # the paired per-song differences). A song whose streams are
        # not both present at the largest N, that is off the tick grid
        # (unless in --keep-offgrid), or that ends before the scored
        # window, is excluded. prompt_bars.tsv records the assignment.
        FPB = 16
        groups = sorted(set(args.prompt_bars))

        def load(fn):
            pm = pretty_midi.PrettyMIDI(fn)
            fr = frame_fn(pm)
            return sorted((fr(n.start), fr(n.end))
                          for ins in pm.instruments for n in ins.notes)

        print(f'\n[prompt] prompt = the first N bars of the song, N = smallest of '
              f'{groups} with both streams present ({args.min_prompt_notes}+ onsets),'
              f' on the tick grid; scored window = the {args.gen_frames} frames after it.'
              f' off-grid = onset fraction off the 16th TICK grid (mel/chd);'
              f' tempo-ev = tempo events in the file (ignored by the tokenizer)')
        cols = ''.join(f'{f"m/c<{b}b":>10}' for b in groups)
        print(f'  {"song":<6}{"entry(m/c)":>11}{cols}{"N":>4}{"end":>7}{"off-grid":>16}'
              f'{"tempo-ev":>10}  grid flags        verdict')
        dropped, assign = [], {}
        for s in sorted(shared):
            data, gflags, offs, tev, first = {}, [], [], [], {}
            for sub in ('melody', 'chord'):
                fn = os.path.join(args.stage_dir, sub, f'{s:03d}.mid')
                data[sub] = load(fn)
                first[sub] = int(data[sub][0][0] // FPB) if data[sub] else None
                try:
                    r = tick_grid(fn, 4)
                    offs.append(r['off_frac'])
                    tev.append(r['tempo_events'])
                    if r['n'] and r['trip_frac'] > args.max_offgrid:
                        gflags.append('TRIPLET')
                    elif r['n'] and r['off_frac'] > args.max_offgrid:
                        gflags.append('OFF-GRID')
                    gflags += meter_flags(fn)
                except Exception as e:      # noqa: BLE001
                    gflags.append(f'check-failed({e!r})')
            counts = {b: {sub: sum(1 for st, _ in data[sub] if st < b * FPB)
                          for sub in data} for b in groups}
            n_bars = next((b for b in groups
                           if all(v >= args.min_prompt_notes
                                  for v in counts[b].values())), None)
            end = min((max((en for _, en in data[sub]), default=0.0)
                       for sub in data), default=0.0)
            reasons = []
            if n_bars is None:
                reasons.append(f'a stream empty in the first {groups[-1]} bars')
            elif end < n_bars * FPB + args.gen_frames:
                reasons.append(f'ends at frame {end:.0f} < {n_bars * FPB + args.gen_frames}')
            if gflags and s in args.keep_offgrid:
                gflags = [f'{g}(kept)' for g in gflags]
            elif gflags:
                reasons.append('off the 16th tick grid / meter')
            if s in args.exclude:
                reasons.append('excluded by --exclude')
            verdict = 'ok' if not reasons else f'EXCLUDED ({"; ".join(reasons)})'
            off_s = '/'.join(f'{o:.1%}' for o in offs) or '-'
            tev_s = '/'.join(str(t) for t in tev) or '-'
            ent_s = '/'.join('-' if first[k] is None else str(first[k])
                             for k in ('melody', 'chord'))
            cnt_s = ''.join(f'{counts[b]["melody"]}/{counts[b]["chord"]:<4}'.rjust(10)
                            for b in groups)
            print(f'  {s:03d}   {ent_s:>11}{cnt_s}{str(n_bars or "-"):>4}{end:>7.0f}'
                  f'{off_s:>16}{tev_s:>10}  {",".join(sorted(set(gflags))) or "-":<18}'
                  f'{verdict}')
            for sub in ('melody', 'chord'):
                src_fn = os.path.join(args.stage_dir, sub, f'{s:03d}.mid')
                if reasons:
                    os.remove(src_fn)
                    continue
                dst_dir = os.path.join(args.stage_dir, f'p{n_bars}', sub)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src_fn, os.path.join(dst_dir, f'{s:03d}.mid'))
            if reasons:
                dropped.append(s)
            else:
                assign[s] = n_bars
        for sub in ('melody', 'chord'):
            d = os.path.join(args.stage_dir, sub)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        kept = sorted(shared - set(dropped))
        with open(os.path.join(args.stage_dir, 'prompt_bars.tsv'), 'w') as fh:
            fh.write('song\tprompt_bars\tprompt_frames\tgen_length\n')
            for s in kept:
                fh.write(f'{s:03d}\t{assign[s]}\t{assign[s] * FPB}\t'
                         f'{assign[s] * FPB + args.gen_frames}\n')
        print(f'[prompt] kept {len(kept)}: {fmt(kept)}')
        for b in groups:
            ids = [s for s in kept if assign[s] == b]
            print(f'[prompt]   p{b} ({len(ids)}): {fmt(ids) if ids else "(none)"}'
                  f'   -> {args.stage_dir}/p{b}/{{melody,chord}}'
                  f'   PROMPT_LENGTH={b * FPB} GEN_LENGTH={b * FPB + args.gen_frames}')
        print(f'[prompt] assignment recorded in {args.stage_dir}/prompt_bars.tsv')
        if dropped:
            print(f'[prompt] removed: {fmt(dropped)}')
        if len(kept) < args.min_kept:
            # hold the afterok chain instead of letting E1 run and print
            # a table over one song with every +-0.000
            print(f'[prompt] ERROR: only {len(kept)} songs kept, fewer than '
                  f'--min-kept {args.min_kept}; the stage is not a test set. '
                  f'Inspect the columns above before lowering the bar.')
            sys.exit(1)


if __name__ == '__main__':
    main()
