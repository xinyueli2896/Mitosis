"""Diagnose the A.12 (self-draft) decode: does the draft the harmonizers
see at decode time look like the draft they were trained on?

The training probe builds the AR-head draft as a per-position sample
from the TEACHER-FORCED AR logits: every sub-token is drawn given the
ground-truth prefix of its frame, so the draft keeps the true frame's
note count, EOS position and program tokens, its errors never compound,
and no validity mask is applied. The decode builds it with
local_sampling, FREE-RUNNING from the content hidden state with the
validity mask. This script puts the two side by side on real songs and
a real checkpoint, for a set of target frames of each song:

  1. the draft itself against the ground-truth frame: sub-token
     agreement, note-count error, EOS-position error, structural
     validity (program / pitch-duration alternation, pad after EOS),
     share of drafts identical to the ground truth
  2. the REVISION the harmonizers make of each draft: cross-entropy and
     argmax accuracy of the harmonizer output against the ground-truth
     frame when the harmonizer holds (a) the teacher-forced draft,
     (b) the free-running draft, (c) the ground truth itself (oracle),
     (d) the mask (no draft)

If (b) is much worse than (a), the harmonizers were trained on drafts
they never meet at decode time, and --sc_ar_free_run is the fix. If
(a) and (b) are alike, the draft path is not the problem and the
search moves on (k tags, temperature, the local decoder's own
sampling).

Everything runs teacher-forced on ground-truth content up to the
target frame, so it isolates the draft mismatch from exposure bias.

Usage (8-GPU machine, plain python; the cluster wraps it in sbatch):
  python diag_a12_draft.py --ckpt ckpt/<A12 run dir> \
      --mel-folder input/subjective/melody --chord-folder input/subjective/chord \
      --max-polyphony 8 --chord-program 48 --prompt-frames 80
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_transformer_m2c_duet_block_diffusion_inference import load_model  # noqa: E402
from cp_transformer_m2c_moe_inference import _load_prompt_tokens         # noqa: E402
from ckpt_utils import pick_device                              # noqa: E402


def structural_ok(row, eos, pad):
    """program / pitch-dur alternation up to EOS, pad after EOS, no pad
    before EOS."""
    row = list(row)
    if eos in row:
        i = row.index(eos)
        if i % 2 != 0:                       # EOS must sit on a program step
            return False
        if any(t != pad for t in row[i + 1:]):
            return False
        body = row[:i]
    else:
        body = row
    if any(t == pad for t in body):
        return False
    return len(body) % 2 == 0


def n_notes(row, eos, pad):
    row = list(row)
    body = row[:row.index(eos)] if eos in row else [t for t in row if t != pad]
    return len(body) // 2


def eos_pos(row, eos):
    row = list(row)
    return row.index(eos) if eos in row else len(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mel-folder', required=True)
    ap.add_argument('--chord-folder', required=True)
    ap.add_argument('--max-polyphony', type=int, default=8)
    ap.add_argument('--chord-program', type=int, default=48,
                    help='retag the chord track before tokenising (as the '
                         'E1 staging does); -1 = leave as is')
    ap.add_argument('--prompt-frames', type=int, default=80)
    ap.add_argument('--total-frames', type=int, default=416)
    ap.add_argument('--n-targets', type=int, default=12,
                    help='target frames per song, evenly spaced over the '
                         'continuation')
    ap.add_argument('--temperature', type=float, default=1.0,
                    help='draft temperature for both sources (training: '
                         'sc_draft_temp; decode: A3_FINAL_TEMP)')
    ap.add_argument('--seeds', type=int, default=3,
                    help='draft samples per target frame')
    ap.add_argument('--max-songs', type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    model = load_model(args.ckpt)
    model.to(device).eval()
    model.sampling_top_p = None
    tok = model.tokenizer
    eos, pad = tok.eos_token, tok.pad_token
    K = model.diffusion_K
    print(f'[diag] K={K}  sc_ar_free_run_flag='
          f'{int(getattr(model, "sc_ar_free_run_flag", torch.tensor(0)))}  '
          f'sc_draft_temp={float(getattr(model, "sc_draft_temp", 0.0))}')

    songs = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(args.mel_folder, '*.mid')))
    if args.max_songs:
        songs = songs[:args.max_songs]

    stats = {src: {'agree': [], 'note_err': [], 'eos_err': [], 'valid': [],
                   'identical': []} for src in ('teacher_forced', 'free_run')}
    rev = {src: {'ce': [], 'acc': []}
           for src in ('teacher_forced', 'free_run', 'oracle', 'mask')}

    import mido, tempfile
    for song in songs:
        mp = os.path.join(args.mel_folder, f'{song}.mid')
        cp = os.path.join(args.chord_folder, f'{song}.mid')
        if args.chord_program >= 0:
            m = mido.MidiFile(cp); seen = False
            for tr in m.tracks:
                for msg in tr:
                    if msg.type == 'program_change':
                        msg.program = args.chord_program; seen = True
            if not seen:
                for tr in m.tracks:
                    if any(msg.type == 'note_on' for msg in tr):
                        tr.insert(0, mido.Message('program_change',
                                                  program=args.chord_program,
                                                  time=0)); break
            tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
            m.save(tmp.name); cp = tmp.name
        mel = _load_prompt_tokens(model, mp, args.max_polyphony)
        chd = _load_prompt_tokens(model, cp, args.max_polyphony)
        T = min(mel.shape[1], chd.shape[1], args.total_frames)
        S = max(mel.shape[2], chd.shape[2])
        def padS(t):
            return F.pad(t, (0, S - t.shape[2]), value=pad) if t.shape[2] < S else t
        mel, chd = padS(mel[:, :T]), padS(chd[:, :T])
        x_full = torch.stack([mel, chd], dim=2).view(1, 2 * T, S).to(device)
        targets = np.linspace(args.prompt_frames, T - 1, args.n_targets).astype(int)

        for t in targets:
            t = int(t)
            x = x_full[:, :2 * (t + 1)]                  # content through frame t
            gt_m, gt_c = x[0, 2 * t], x[0, 2 * t + 1]
            with torch.no_grad():
                model._stash_h = True
                ar_logits, q_logits_masked, _ = model.forward(
                    x, T_query=t, k_m=K, k_c=K)
                model._stash_h = False
                hg = model._last_h_clean_global
                V = tok.n_tokens
                ar4 = ar_logits.view(1, 2 * (t + 1), S, V)
                # mask-only revision (no draft): the probe's own harmonizer output
                ql = q_logits_masked.view(2, S, V)
                for si, gt in ((0, gt_m), (1, gt_c)):
                    keep = gt != pad
                    rev['mask']['ce'].append(float(F.cross_entropy(ql[si][keep], gt[keep])))
                    rev['mask']['acc'].append(float((ql[si].argmax(-1)[keep] == gt[keep]).float().mean()))
                for _seed in range(args.seeds):
                    drafts = {}
                    # (a) teacher-forced per-position sample, as the training probe
                    tf_m = model._draft_from_logits(ar4[:, 2 * t], temp=args.temperature)[0]
                    tf_c = model._draft_from_logits(ar4[:, 2 * t + 1], temp=args.temperature)[0]
                    drafts['teacher_forced'] = (tf_m, tf_c)
                    # (b) free-running local_sampling, as the decode
                    fr_m = model.local_sampling(hg[:, 2 * t], max_subseq_len=S,
                                                temperature=args.temperature,
                                                token_type_id=0)[0]
                    fr_c = model.local_sampling(hg[:, 2 * t + 1], max_subseq_len=S,
                                                temperature=args.temperature,
                                                token_type_id=1)[0]
                    drafts['free_run'] = (fr_m, fr_c)
                    for src, (dm, dc) in drafts.items():
                        for d, gt in ((dm, gt_m), (dc, gt_c)):
                            keep = gt != pad
                            stats[src]['agree'].append(float((d[keep] == gt[keep]).float().mean()))
                            stats[src]['note_err'].append(abs(n_notes(d.tolist(), eos, pad) - n_notes(gt.tolist(), eos, pad)))
                            stats[src]['eos_err'].append(abs(eos_pos(d.tolist(), eos) - eos_pos(gt.tolist(), eos)))
                            stats[src]['valid'].append(float(structural_ok(d.tolist(), eos, pad)))
                            stats[src]['identical'].append(float(bool((d == gt).all())))
                    drafts['oracle'] = (gt_m, gt_c)
                    # revision: harmonizers hold the draft at k=0
                    for src, (dm, dc) in drafts.items():
                        emb_m = model._encode_frame(dm.view(1, S), 0)
                        emb_c = model._encode_frame(dc.view(1, S), 1)
                        on = torch.ones(1, dtype=torch.bool, device=device)
                        _ar, q_logits, _ = model.forward(
                            x, T_query=t, k_m=0, k_c=0,
                            sc_mask_m=on, sc_emb_m=emb_m,
                            sc_mask_c=on, sc_emb_c=emb_c,
                            sc_toks_m=dm.view(1, S), sc_toks_c=dc.view(1, S))
                        ql = q_logits.view(2, S, V)
                        for si, gt in ((0, gt_m), (1, gt_c)):
                            keep = gt != pad
                            rev[src]['ce'].append(float(F.cross_entropy(ql[si][keep], gt[keep])))
                            rev[src]['acc'].append(float((ql[si].argmax(-1)[keep] == gt[keep]).float().mean()))
        print(f'[diag] {song}: {len(targets)} target frames done')

    print('\n== DRAFT vs GROUND TRUTH (both streams, all targets, all seeds) ==')
    print(f'{"source":16s} {"agree":>7s} {"note_err":>9s} {"eos_err":>8s} {"valid":>6s} {"identical":>10s}')
    for src, d in stats.items():
        print(f'{src:16s} {np.mean(d["agree"]):7.3f} {np.mean(d["note_err"]):9.2f} '
              f'{np.mean(d["eos_err"]):8.2f} {np.mean(d["valid"]):6.3f} {np.mean(d["identical"]):10.3f}')
    print('\n== HARMONIZER REVISION of each draft (k=0), vs ground truth ==')
    print(f'{"harmonizer holds":16s} {"CE":>7s} {"acc":>7s}')
    for src, d in rev.items():
        print(f'{src:16s} {np.mean(d["ce"]):7.3f} {np.mean(d["acc"]):7.3f}')
    print('\nRead: if free_run CE >> teacher_forced CE, the harmonizers were '
          'trained on drafts they never meet at decode; retrain with '
          'SC_AR_FREE_RUN=1. If the two are alike, the draft path is not '
          'the problem.')


if __name__ == '__main__':
    main()
