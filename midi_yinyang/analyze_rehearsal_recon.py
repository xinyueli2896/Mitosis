"""Can the rehearsal model RECONSTRUCT the conditioning stream from its
prefix -- and does that ability hold across the sequence?

C.1's design promise: the suffix's mod_a slots retrieve mod_a[k] out of
the prefix (the slot predicting a_k never holds a_k -- it holds a_{k-1}
under shift-2, b_{k-1} under shift-1 -- so exact reconstruction REQUIRES
the retrieval). Training supervises this with the Brier term, but
standard inference never exercises it: the suffix's mod_a slots are
teacher-forced from the prompt and the model's own reconstruction is
discarded. val_recon_loss shows the aggregate; what it hides is the
POSITION structure.

This analyzer runs val batches teacher-forced and reports mod_a
reconstruction accuracy AS A FUNCTION OF FRAME INDEX k:

  - Under the LEGACY geometry the rotary distance from the mod_a query
    to a_k in the prefix grew as k -- prediction: accuracy DECAYS with k.
  - Under prefix_stride2 the distance is constant -- prediction: FLAT.

So the same curve that answers "can it reconstruct" also empirically
validates (or falsifies) the geometry fix, on checkpoints that already
exist.

Interpretation caveat printed with the results: under shift-2 the slot
holds a_{k-1}, so decent accuracy can come from LOCAL CONTINUATION of
the stream rather than prefix retrieval; under shift-1 the slot holds
b_{k-1} and continuation is impossible -- accuracy IS retrieval. The
free-running counterpart (infer_duet_rehearsal.sbatch MODE=reconstruct)
closes the remaining gap.

C.1-only: C.2's forward emits no mod_a logits at all.

Usage (via analyze_rehearsal_recon.sbatch):
    python analyze_rehearsal_recon.py --ckpt ckpt/<C1A or C1B run>/ \\
        --task melchord_nottingham --batches 4
"""

import argparse
import os
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--batches', type=int, default=4,
                   help='val batches to average over')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--model-size', default='large')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--bins', type=int, default=12,
                   help='frame-index buckets for the printed table')
    p.add_argument('--out-csv', default=None,
                   help='per-frame curve CSV (default results/recon_curve_'
                        '<ckpt dir name>.csv)')
    args = p.parse_args()

    from cp_transformer_m2c_duet_rehearsal_inference import load_model
    from cp_transformer_m2c_moe import FramedDataset, TRAIN_LENGTH
    from tasks import get_task

    task = get_task(args.task)
    net = load_model(args.ckpt, model_size=args.model_size,
                     moe_num_experts=args.moe_num_experts,
                     moe_topk=args.moe_topk)
    net.eval()
    pad = net.tokenizer.pad_token
    sh1 = bool(net.suffix_shift1)
    ps2 = bool(net.prefix_stride2)

    print('=' * 72)
    print('REHEARSAL RECONSTRUCTION -- teacher-forced, position-resolved')
    print(f'ckpt={args.ckpt}')
    print(f'task={task.name}  scheme: prefix_stride2={ps2} '
          f'suffix_shift1={sh1}')
    if sh1:
        print('shift-1: the slot predicting a_k holds b_{k-1} -- local')
        print('continuation is impossible, so accuracy here IS prefix '
              'retrieval.')
    else:
        print('shift-2: the slot predicting a_k holds a_{k-1} -- accuracy '
              'can mix')
        print('prefix retrieval with local stream continuation; compare '
              'against the')
        print('free-running MODE=reconstruct to separate them.')
    print('=' * 72)

    ds = FramedDataset(task.mod_b_path, TRAIN_LENGTH, args.batch_size,
                       split='val', mel_path=task.mod_a_path)

    T = None
    a_corr = a_tot = b_corr = b_tot = None
    a_frame_ok = a_frames = None
    it = iter(ds)
    for bi in range(args.batches):
        try:
            batch = list(next(it))
        except StopIteration:
            print(f'[data] val exhausted after {bi} batches')
            break
        x_mel, x_acc, ps = batch[0], batch[1], batch[2]
        with torch.no_grad():
            xm, xa = net.preprocess(x_mel, ps, y=x_acc)
            B, seq_len, S = xm.shape
            x = torch.stack([xm, xa], dim=2).view(B, seq_len * 2, S)
            logits, _ = net.forward(x)
            V = net.tokenizer.n_tokens
            lg = logits.view(B, seq_len * 2, S, V)
        if T is None:
            T = seq_len
            a_corr = torch.zeros(T); a_tot = torch.zeros(T)
            b_corr = torch.zeros(T); b_tot = torch.zeros(T)
            a_frame_ok = torch.zeros(T); a_frames = torch.zeros(T)
        pred = lg.argmax(-1)                       # [B, 2T, S]
        for parity, corr, tot in ((0, a_corr, a_tot), (1, b_corr, b_tot)):
            tgt = x[:, parity::2]                  # [B, T, S]
            pr = pred[:, parity::2]
            np_mask = (tgt != pad)
            hit = (pr == tgt) & np_mask
            corr += hit.sum(dim=(0, 2)).float()
            tot += np_mask.sum(dim=(0, 2)).float()
            if parity == 0:
                # frame-exact: every non-pad token of the frame correct
                ok = (hit.sum(-1) == np_mask.sum(-1)) & (np_mask.sum(-1) > 0)
                a_frame_ok += ok.sum(0).float()
                a_frames += (np_mask.sum(-1) > 0).sum(0).float()
        print(f'[batch {bi + 1}/{args.batches}] done')

    if T is None:
        sys.exit('no data')
    a_acc = (a_corr / a_tot.clamp_min(1)).numpy()
    b_acc = (b_corr / b_tot.clamp_min(1)).numpy()
    fr_acc = (a_frame_ok / a_frames.clamp_min(1)).numpy()

    print(f'\nper-position curves over T={T} frames '
          f'({args.batches} batch(es), {int(a_tot.sum())} mod_a tokens):\n')
    print(f'  {"frames":>12} {"a tok acc":>10} {"a frame-exact":>14} '
          f'{"b tok acc":>10}   (b = the generative stream, as a floor)')
    bw = max(1, T // args.bins)
    for s in range(0, T, bw):
        e = min(T, s + bw)
        print(f'  {f"{s:>4}-{e - 1:<4}":>12} '
              f'{a_acc[s:e].mean():>10.3f} {fr_acc[s:e].mean():>14.3f} '
              f'{b_acc[s:e].mean():>10.3f}')

    # decay statistics on the mod_a token curve
    import numpy as np
    k = np.arange(T)
    slope = float(np.polyfit(k, a_acc, 1)[0])
    q = T // 4
    first_q, last_q = float(a_acc[:q].mean()), float(a_acc[-q:].mean())
    delta = last_q - first_q
    print(f'\n  mod_a acc: mean {a_acc.mean():.3f}   '
          f'first quartile {first_q:.3f} -> last quartile {last_q:.3f} '
          f'(delta {delta:+.3f})   slope*T {slope * T:+.3f}')

    print('\n' + '=' * 72)
    print('VERDICT')
    if a_acc.mean() < b_acc.mean() + 0.05:
        print('  NO RECONSTRUCTION: mod_a accuracy is at the generative '
              'floor -- the')
        print('  suffix is not retrieving the conditioning stream from the '
              'prefix at all.')
        print('  Check val_recon_loss in wandb; if it never fell, the Brier '
              'term did not')
        print('  train (was recon_weight 0 on this run?).')
    elif delta < -0.05:
        print(f'  RETRIEVAL DECAYS with position ({delta:+.3f} first->last '
              f'quartile).')
        if ps2:
            print('  Under prefix_stride2 the rotary distance is constant, '
                  'so this decay is')
            print('  NOT the old geometry bug -- look at attention span / '
                  'gate behaviour.')
        else:
            print('  Consistent with the LEGACY geometry: the query-to-'
                  'prefix rotary distance')
            print('  grows with k, and retrieval accuracy falls with it. '
                  'This is the direct')
            print('  empirical signature of the bug prefix_stride2 fixes.')
    else:
        print(f'  RETRIEVAL HOLDS across the sequence (delta {delta:+.3f}, '
              f'mean {a_acc.mean():.3f}).')
        if ps2:
            print('  As predicted for the fixed geometry: constant rotary '
                  'distance, flat curve.')
    print('=' * 72)

    out = args.out_csv or os.path.join(
        'results', f'recon_curve_{os.path.basename(os.path.normpath(args.ckpt))}.csv')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        f.write('k,a_tok_acc,a_frame_exact,b_tok_acc\n')
        for i in range(T):
            f.write(f'{i},{a_acc[i]:.5f},{fr_acc[i]:.5f},{b_acc[i]:.5f}\n')
    print(f'per-frame curve -> {out}')


if __name__ == '__main__':
    main()
