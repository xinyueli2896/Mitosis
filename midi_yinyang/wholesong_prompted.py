"""T3': whole-song-gen PROMPTED lead-sheet continuation (block (i)).

Their README ships no prompted CLI, but the released inference library
implements it: every level's create_canvas takes prompt= (content
written into the canvas, mask=1 over the prompt region) and the sampler
inpaints around known cells. This driver calls those functions with the
argument they already accept -- no custom sampling logic -- and encodes
the prompt with THEIR data pipeline, so every representation decision
stays the authors'.

Per test song (POP909 ids, default 1-5):
  1. read + analyze the song with their read_pop909_data /
     analyze_pop909_dataset (ground-truth languages at every level);
  2. counterpoint stage: background = the song's GT form channels
     (key + phrase, the form-conditioned protocol -- oracle form is an
     advantage GIVEN TO THE BASELINE), prompt = GT reduced lead sheet
     for the first --prompt-bars bars (nbpm rows per bar);
  3. lead-sheet stage: background = the GENERATED counterpoint expanded
     through their own expand_background, prompt = GT lead sheet for
     the same bars (nbpm*nspb rows per bar);
  4. write one midi per sample: MELODY / CHORD tracks (programs 0/48)
     via their piano_roll_to_note_mat, full length including the
     prompt region -- the eval harness scores frames prompt..total as
     usual. Layout: <out>/<songid>/co/sample_<i>.mid (duet_multi).

The acc stage is skipped (out of scope) and the frm model is never
loaded (form is ground truth).

Run from the whole_song_gen repo root (the sbatch handles PYTHONPATH):
    python wholesong_prompted.py --song-ids 1 2 3 4 5 \
        --prompt-bars 6 --n-samples 3 --out-dir <abs path>
"""

import argparse
import os
import sys

import numpy as np
import pretty_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--song-ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--prompt-bars', type=int, default=6)
    ap.add_argument('--n-samples', type=int, default=3)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--bpm', type=float, default=90.0,
                    help='their output convention; scoring re-derives '
                         'the grid from the file tempo either way')
    args = ap.parse_args()

    # their repo root must be the cwd (resource paths are relative)
    from data_utils.read_pop909_data import (analyze_pop909_dataset,
                                             read_pop909_dataset)
    from data_utils.pytorch_datasets.counterpoint_dataset import \
        CounterpointDataset
    from data_utils.pytorch_datasets.leadsheet_dataset import LeadSheetDataset
    from data_utils.midi_output import note_mat_to_notes, piano_roll_to_note_mat
    from inference.generation_operations import (CounterpointGenOp,
                                                 LeadSheetGenOp)
    from model import get_model_path
    from params import params_ctp, params_lsh
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctp_path, _, _ = get_model_path('results_default/ctp-a-b-/v-default',
                                    'default')
    lsh_path, _, _ = get_model_path('results_default/lsh-a-b-/v-default',
                                    'default')
    ctp_op = CounterpointGenOp(params_ctp, ctp_path, device,
                               use_autoreg_cond=True, use_external_cond=False)
    lsh_op = LeadSheetGenOp(params_lsh, lsh_path, device,
                            use_autoreg_cond=True, use_external_cond=False)

    ok, failed = [], []
    for sid in args.song_ids:
        name = str(sid).zfill(3)
        print(f'\n=== song {name} ===', flush=True)
        try:
            dataset = read_pop909_dataset(song_ids=[sid])
            analyses = analyze_pop909_dataset(dataset)
            nbpm, nspb = 4, 4

            # GT language images, full length, via THEIR dataset builders
            # (shift 0, no augmentation).
            ctp_ds = CounterpointDataset(analyses, shift_high=0, shift_low=0,
                                         random_pitch_aug=False,
                                         use_autoreg_cond=True,
                                         use_external_cond=False)
            ctp_ds.store_key(0, 0)
            ctp_ds.store_phrase(0)
            ctp_ds.store_red_mel(0, 0)
            ctp_ds.store_red_chd(0, 0)
            L_beats = ctp_ds.lengths[0]
            ctp_img = ctp_ds.lang_to_img(0, 0, L_beats, tgt_lgth=L_beats)

            lsh_ds = LeadSheetDataset(analyses, shift_high=0, shift_low=0,
                                      random_pitch_aug=False,
                                      use_autoreg_cond=True,
                                      use_external_cond=False)
            lsh_ds.store_key(0, 0)
            lsh_ds.store_phrase(0)
            lsh_ds.store_red_mel(0, 0)
            lsh_ds.store_red_chd(0, 0)
            lsh_ds.store_mel(0, 0)
            lsh_ds.store_chd(0, 0)
            L_16 = lsh_ds.lengths[0]
            lsh_img = lsh_ds.lang_to_img(0, 0, L_16, tgt_lgth=L_16)

            p_beats = args.prompt_bars * nbpm
            p_16 = args.prompt_bars * nbpm * nspb
            print(f'  L = {L_beats} beats / {L_16} sixteenths; '
                  f'prompt = {p_beats} beats / {p_16} sixteenths')

            n = args.n_samples
            # ---- counterpoint: GT form background, GT 6-bar prompt ----
            ctp_bg = np.repeat(ctp_img[np.newaxis, 2:], n, axis=0)
            ctp_prompt = np.repeat(ctp_img[np.newaxis, 0:2, 0:p_beats],
                                   n, axis=0)
            canvas, slices, gen_max_l = ctp_op.create_canvas(
                ctp_bg, n, nbpm, ctp_prompt)
            ctp_songs = ctp_op.generation(canvas, slices, gen_max_l,
                                          n_sample=n)
            ctp_out = np.stack(ctp_songs, 0)[:, :, 0:L_beats]

            # ---- lead sheet: generated ctp as background (their own
            #      expand path), GT 6-bar prompt ----
            lsh_bg = lsh_op.expand_background(ctp_out, nspb)[:, :, 0:L_16]
            lsh_prompt = np.repeat(lsh_img[np.newaxis, 0:2, 0:p_16],
                                   n, axis=0)
            canvas, slices, gen_max_l = lsh_op.create_canvas(
                lsh_bg, n, nbpm, nspb, lsh_prompt)
            lsh_songs = lsh_op.generation(canvas, slices, gen_max_l)

            out_dir = os.path.join(args.out_dir, name, 'co')
            os.makedirs(out_dir, exist_ok=True)
            for i, song in enumerate(lsh_songs):
                pair = song[0:2, 0:L_16]
                nmat_mel, nmat_chd = piano_roll_to_note_mat(
                    pair, True, seperate_chord=True)
                notes_mel = note_mat_to_notes(nmat_mel, args.bpm, unit=0.25)
                notes_chd = note_mat_to_notes(nmat_chd, args.bpm, unit=0.25)
                pm = pretty_midi.PrettyMIDI(initial_tempo=args.bpm)
                mel = pretty_midi.Instrument(0, name='MELODY')
                mel.notes = notes_mel
                chd = pretty_midi.Instrument(48, name='CHORD')
                chd.notes = notes_chd
                pm.instruments += [mel, chd]
                path = os.path.join(out_dir, f'sample_{i}.mid')
                pm.write(path)
                print(f'  wrote {path}')
            ok.append(name)
        except Exception as e:                     # noqa: BLE001
            import traceback
            print(f'  FAILED {name}: {e!r}')
            traceback.print_exc(limit=4)
            failed.append(name)

    print('\n================ SUMMARY ================')
    print(f'ok: {len(ok)}/{len(args.song_ids)}')
    if failed:
        print(f'failed: {failed}')
        sys.exit(1)


if __name__ == '__main__':
    main()
