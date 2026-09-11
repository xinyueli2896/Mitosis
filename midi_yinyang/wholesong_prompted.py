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
  2. counterpoint stage: prompt = GT reduced lead sheet for the first
     --prompt-bars bars (nbpm rows per bar). The background (key +
     phrase) depends on --form:
       gt        the song's GROUND-TRUTH form channels for the whole
                 song, including the region being generated. This is
                 their form-conditioned protocol and an oracle GIVEN
                 TO THE BASELINE. Default, so earlier runs reproduce.
       generated their OWN form stage, prompted with the GT form of the
                 same --prompt-bars bars and left to predict key and
                 phrase across the rest. Removes the only ground-truth
                 leak into the generated region, which is what makes
                 the comparison with a prompt-only system fair.
     NOTE even under `generated` the song LENGTH stays oracle: the
     generated form is truncated to the GT bar count, because a
     fixed-length continuation protocol has to know where to stop for
     the reference to be comparable. Length is a far weaker advantage
     than key + phrase, but it is an advantage and the paper should
     say so.
  3. lead-sheet stage: background = the GENERATED counterpoint expanded
     through their own expand_background, prompt = GT lead sheet for
     the same bars (nbpm*nspb rows per bar);
  4. write one midi per sample: MELODY / CHORD tracks (programs 0/48)
     via their piano_roll_to_note_mat, full length including the
     prompt region -- the eval harness scores frames prompt..total as
     usual. Layout: <out>/<songid>/co/sample_<i>.mid (duet_multi).

The acc stage is skipped (out of scope). The frm model is loaded only
when --form generated.

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
    ap.add_argument('--crops-tsv', default=None,
                    help="prompt_crops.tsv from build_prompt_crops. Each "
                         "song's representation is cropped to that song's "
                         "crop_bar BEFORE prompting, so the baseline's bar 1 "
                         "is OUR bar 1. Without it the baseline is prompted "
                         "from bar 0 of the song -- usually the intro -- "
                         "while our systems see the melody entry, and the "
                         "two are not answering the same question. Songs "
                         "absent from the TSV, or dropped in it, are "
                         "skipped.")
    ap.add_argument('--form', choices=('gt', 'generated'), default='gt',
                    help="where the counterpoint stage's key+phrase "
                         "background comes from. gt = ground truth over "
                         "the WHOLE song (their form-conditioned "
                         "protocol, an oracle in the baseline's favour); "
                         "generated = their own form stage prompted with "
                         "the GT form of the prompt bars only. Song "
                         "length stays oracle either way.")
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
    from data_utils.pytorch_datasets.form_dataset import FormDataset
    from data_utils.midi_output import note_mat_to_notes, piano_roll_to_note_mat
    from inference.generation_operations import (CounterpointGenOp,
                                                 FormGenOp, LeadSheetGenOp)
    from inference.utils import quantize_generated_form_batch
    from data_utils.pytorch_datasets.const import LANGUAGE_DATASET_PARAMS
    # The dataset constructors default to the COUNTERPOINT geometry
    # (n_channels=10); the lead-sheet level is 12 channels. Their own
    # loader passes these from const.py per level; so must we, or the
    # 6 phrase channels written at [6:12] overflow a 10-channel image.
    P_CTP = LANGUAGE_DATASET_PARAMS['counterpoint']
    P_LSH = LANGUAGE_DATASET_PARAMS['lead_sheet']
    P_FRM = LANGUAGE_DATASET_PARAMS['form']
    from model import get_model_path
    from params import params_ctp, params_frm, params_lsh
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
    frm_op = None
    if args.form == 'generated':
        # AUTOREG_PARAMS has no 'form' entry -- the form level is the top
        # of the cascade and conditions on nothing, so both cond flags
        # are off, matching their own form_generation().
        frm_path, _, _ = get_model_path('results_default/frm---/v-default',
                                        'default')
        frm_op = FormGenOp(params_frm, frm_path, device,
                           use_autoreg_cond=False, use_external_cond=False)
        print('[form] generated: key+phrase predicted past the prompt '
              '(no oracle form over the generated region)')
    else:
        print('[form] gt: ORACLE key+phrase over the whole song, '
              "including the generated region (baseline's favour)")

    crops = {}
    if args.crops_tsv:
        import csv as _csv
        with open(args.crops_tsv) as fh:
            for row in _csv.DictReader(fh, delimiter='\t'):
                if row.get('dropped'):
                    continue
                cb = row.get('crop_bar', '')
                if cb != '':
                    crops[row['id']] = int(cb)
        print(f'[crop] {len(crops)} songs with a crop point from '
              f'{args.crops_tsv}')

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
                                         max_l=P_CTP['max_l'], h=P_CTP['h'],
                                         n_channels=P_CTP['n_channel'],
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
                                      max_l=P_LSH['max_l'], h=P_LSH['h'],
                                      n_channels=P_LSH['n_channel'],
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

            # Crop THEIR representation to our crop point, so bar 1 of
            # the baseline's song is bar 1 of ours. Slicing the language
            # images is the honest way to do it: every level keeps its
            # own encoding, only the window moves.
            if crops:
                if name not in crops:
                    raise ValueError(
                        f'no crop point for {name} in {args.crops_tsv} '
                        f'(dropped there, or not a held-out song)')
                cb = crops[name]
                off_beats, off_16 = cb * nbpm, cb * nbpm * nspb
                if off_beats >= L_beats or off_16 >= L_16:
                    raise ValueError(
                        f'crop_bar {cb} is past the end of {name} '
                        f'({L_beats} beats / {L_16} sixteenths)')
                ctp_img = ctp_img[:, off_beats:]
                lsh_img = lsh_img[:, off_16:]
                L_beats -= off_beats
                L_16 -= off_16
                print(f'  cropped to bar {cb} '
                      f'(-{off_beats} beats / -{off_16} sixteenths)')

            p_beats = args.prompt_bars * nbpm
            p_16 = args.prompt_bars * nbpm * nspb
            print(f'  L = {L_beats} beats / {L_16} sixteenths; '
                  f'prompt = {p_beats} beats / {p_16} sixteenths')

            n = args.n_samples
            # ---- form background ----------------------------------
            # ctp channels [0:2] are reduced mel + reduced chd (what the
            # stage predicts); [2:10] are key + phrase, its background.
            # That is the same 8-channel geometry the form level emits,
            # which is why expand_background can feed one to the other.
            n_bars = L_beats // nbpm
            if args.form == 'gt':
                ctp_bg = np.repeat(ctp_img[np.newaxis, 2:], n, axis=0)
            else:
                frm_ds = FormDataset(analyses, shift_high=0, shift_low=0,
                                     max_l=P_FRM['max_l'], h=P_FRM['h'],
                                     n_channels=P_FRM['n_channel'],
                                     random_pitch_aug=False)
                frm_ds.store_key(0, 0)
                frm_ds.store_phrase(0)
                # FormDataset rows are BARS (ctp rows are beats), so its
                # own length is the bar count; prefer it over L_beats //
                # nbpm where they disagree, and say so if they do.
                frm_bars = frm_ds.lengths[0]
                if frm_bars != n_bars:
                    print(f'  [form] bar count {frm_bars} != L_beats//nbpm '
                          f'{n_bars}; using {frm_bars}')
                    n_bars = frm_bars
                frm_img = frm_ds.lang_to_img(0, 0, frm_bars,
                                             tgt_lgth=frm_bars)
                frm_prompt = np.repeat(
                    frm_img[np.newaxis, :, 0:args.prompt_bars], n, axis=0)
                f_canvas, f_slices, f_max_l = frm_op.create_canvas(
                    n_sample=n, prompt=frm_prompt)
                frm_raw = frm_op.generation(f_canvas, f_slices, f_max_l,
                                            quantize=False, n_sample=n)
                frm_q, frm_lengths, frm_phrases = \
                    quantize_generated_form_batch(np.stack(frm_raw, 0))
                for i, (li, pi) in enumerate(zip(frm_lengths, frm_phrases)):
                    flag = ' SHORT' if li < n_bars else ''
                    print(f'  [form] sample {i}: predicted {li} bars'
                          f'{flag}, phrases {pi}')
                # Truncate (or right-pad) the predicted form to the GT bar
                # count: the continuation protocol scores a fixed span, so
                # length stays oracle even here. A sample whose predicted
                # song end falls short is padded with its own last bar
                # rather than with -1, which the ctp stage would read as
                # "no background".
                frm_use = np.zeros((n, P_FRM['n_channel'], n_bars,
                                    P_FRM['h']), dtype=frm_q.dtype)
                take = min(n_bars, frm_q.shape[2])
                frm_use[:, :, 0:take] = frm_q[:, :, 0:take]
                if take < n_bars:
                    frm_use[:, :, take:] = frm_q[:, :, take - 1:take]
                ctp_bg = ctp_op.expand_background(frm_use, nbpm)[:, :,
                                                                0:L_beats]
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
