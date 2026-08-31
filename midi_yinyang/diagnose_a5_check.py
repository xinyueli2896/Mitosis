"""Why is A.5 worse than A.3 even at REFINE_STEPS=0? Three checks.

Loads the A.5 and A.3 checkpoints and runs both over the SAME
deterministic validation batches. No training, no decoding -- pure
measurement. Everything prints as a labelled report.

  CHECK 1 -- keep-mask sign (the unaudited-bug check).
    Run the A.5 model in train mode and read _last_query_kept_frac,
    the fraction of query positions its loss actually scored.
      ~0.75  correct (0.5 masked + half of the unmasked kept via
             self-conditioning at self_cond_prob=0.5)
      ~0.25  INVERTED keep mask: the run trained on the copy
             positions only, and the whole A.5 result is a bug.

  CHECK 2 -- effective re-weighting factor.
    For BOTH models, compute the query CE twice on identical
    corruption draws (same seed): once averaged over all non-pad
    positions (the A.3 objective) and once over corrupted-only (the
    A.5 objective). The ratio corrupted-only/all is the factor by
    which A.5 silently scaled the query term's gradient share at
    unchanged query_loss_weight. ~2x supports the re-weighting
    explanation of the degradation.

  CHECK 3 -- eval-mode losses on identical batches.
    net.eval() pins k=K, where the two objectives coincide, so
    val_ar_loss / val_query_loss are directly comparable numbers, on
    the same data, from the same code, for both checkpoints. If
    A.5's ar_loss is clearly higher, the shared backbone was dragged
    by the query gradient; if they match, the by-ear gap lives
    somewhere eval CE cannot see (sampling regime), which points back
    at decoding or at checkpoint choice.

Usage (via diagnose_a5_check.sbatch):
    python diagnose_a5_check.py --ckpt_a5 <file> --ckpt_a3 <file>
"""

import argparse
import os

import torch
import torch.nn.functional as F

from cp_transformer_m2c_moe import FramedDataset, TRAIN_LENGTH
from cp_transformer_m2c_duet_block_diffusion_inference import load_model
from tasks import get_task


def collect_batches(task, batch_size, n_batches):
    val = FramedDataset(task.mod_b_path, TRAIN_LENGTH, batch_size,
                        split='val', mel_path=task.mod_a_path)
    loader = torch.utils.data.DataLoader(val, batch_size=None, num_workers=0)
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        out.append(b)
    return out


def to_dev(batch, device):
    return [t.to(device) if torch.is_tensor(t) else t for t in batch]


def train_mode_stats(net, batches, device, seed0):
    """Train-mode loss() sweeps: kept_frac and CE under both objectives."""
    net.train()
    kept, ce_all, ce_kept = [], [], []
    flag0 = int(net.mask_revealed_query_loss_flag.item())
    with torch.no_grad():
        for i, b in enumerate(batches):
            b = to_dev(b, device)
            # Same seed for both flag settings -> identical corruption
            # draws, so the two CE numbers differ ONLY in what is scored.
            net.mask_revealed_query_loss_flag.fill_(1)
            torch.manual_seed(seed0 + i)
            net.loss(*b)
            kept.append(float(net._last_query_kept_frac))
            ce_kept.append(float(net._last_query_loss))
            net.mask_revealed_query_loss_flag.fill_(0)
            torch.manual_seed(seed0 + i)
            net.loss(*b)
            ce_all.append(float(net._last_query_loss))
    net.mask_revealed_query_loss_flag.fill_(flag0)
    m = lambda v: sum(v) / max(len(v), 1)
    return m(kept), m(ce_all), m(ce_kept)


def eval_mode_stats(net, batches, device):
    net.eval()
    ar, q, tot = [], [], []
    with torch.no_grad():
        for b in batches:
            b = to_dev(b, device)
            loss, _aux = net.loss(*b)
            ar.append(float(net._last_ar_loss))
            q.append(float(net._last_query_loss))
            tot.append(float(loss))
    m = lambda v: sum(v) / max(len(v), 1)
    return m(ar), m(q), m(tot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_a5', required=True)
    p.add_argument('--ckpt_a3', required=True)
    p.add_argument('--task', default='melchord')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--n_batches', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = get_task(args.task)
    print(f'[data] mod_a={task.mod_a_path}  mod_b={task.mod_b_path}')
    batches = collect_batches(task, args.batch_size, args.n_batches)
    print(f'[data] {len(batches)} deterministic val batches x '
          f'batch_size {args.batch_size}')

    nets = {}
    for name, path in (('A5', args.ckpt_a5), ('A3', args.ckpt_a3)):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        step = ck.get('global_step')
        del ck
        net = load_model(path).to(device)
        flag = int(net.mask_revealed_query_loss_flag.item())
        print(f'[{name}] {os.path.basename(path)}  global_step={step}  '
              f'mask_revealed_query_loss_flag={flag}')
        if name == 'A5' and flag != 1:
            print(f'[{name}] WARNING: expected flag=1 on the A.5 ckpt -- '
                  f'is this the right file?')
        nets[name] = net

    print('\n=== CHECK 1: keep-mask sign (A.5 model, train-mode draws) ===')
    kept5, all5, hard5 = train_mode_stats(nets['A5'], batches, device,
                                          args.seed + 1000)
    print(f'  A5 query_kept_frac = {kept5:.3f}   (expected ~0.75)')
    if kept5 < 0.45:
        print('  VERDICT: ~0.25-ish -> keep mask INVERTED. The run trained '
              'on the copy positions only; A.5\'s result is a bug, not a '
              'finding.')
    elif kept5 > 0.95:
        print('  VERDICT: ~1.0 -> the mask never engaged (flag not applied '
              'in training?). Check the run\'s wandb config.')
    else:
        print('  VERDICT: keep mask direction looks CORRECT.')

    print('\n=== CHECK 2: effective re-weighting factor ===')
    kept3, all3, hard3 = train_mode_stats(nets['A3'], batches, device,
                                          args.seed + 1000)
    for name, ca, ck_, kf in (('A5', all5, hard5, kept5),
                              ('A3', all3, hard3, kept3)):
        ratio = ck_ / ca if ca else float('nan')
        print(f'  {name}: query CE over ALL positions = {ca:.4f}   '
              f'over CORRUPTED-only = {ck_:.4f}   ratio = {ratio:.2f}x   '
              f'(kept_frac {kf:.3f})')
    print('  The ratio is the factor by which A.5\'s objective scaled the '
          'query term\'s\n  gradient share at unchanged query_loss_weight. '
          '~2x supports the\n  re-weighting explanation.')

    print('\n=== CHECK 3: eval-mode losses, identical batches ===')
    print(f'  {"":4} {"ar_loss":>10} {"query_loss":>12} {"total":>10}')
    rows = {}
    for name in ('A3', 'A5'):
        ar, q, tot = eval_mode_stats(nets[name], batches, device)
        rows[name] = (ar, q, tot)
        print(f'  {name:4} {ar:10.4f} {q:12.4f} {tot:10.4f}')
    d_ar = rows['A5'][0] - rows['A3'][0]
    d_q = rows['A5'][1] - rows['A3'][1]
    print(f'  A5-A3 {d_ar:+10.4f} {d_q:+12.4f}')
    print('  ar_loss clearly higher for A5 -> the query gradient dragged '
          'the shared\n  backbone (re-weighting story). ar_loss ~equal -> '
          'eval CE cannot see the\n  by-ear gap; suspect decoding regime '
          'or checkpoint choice instead.')


if __name__ == '__main__':
    main()
