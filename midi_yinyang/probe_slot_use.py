"""Does the model USE its query slots, or has it learned to ignore them?

The A.7 post-mortem hypothesis: a PARAMETRIC corruption (time
displacement, and pitch shift would be the same) can be undone by a
shortcut -- "detect that this content is displaced, discard it" -- so
the model reaches low query loss by IGNORING the slot rather than by
learning to coordinate. If true, the same-instant channel carries
nothing at decode, which is what coupling below the no-slot control
looks like.

This probe measures that directly, with no training and no decoding.
For a fixed query frame t it runs the model several times over the SAME
validation batches, changing only what the two query slots contain, and
reads the query logits:

  ctx            both slots masked            -> prediction from history alone
  own_true       own slot = its true frame    -> can it read its own slot at all?
  partner_true   partner slot = its true frame, own masked
  partner_decoy  partner slot = a DISPLACED real frame (t + lag), own masked
  partner_shift  partner slot = a TRANSPOSED real frame, own masked

The two numbers that decide the question, per stream:

  PARTNER GAIN  = CE(ctx) - CE(partner_true)
      how much knowing the partner's true frame improves this stream's
      prediction. This IS the same-instant channel's information
      content. Near zero => the channel is dead.
  DECOY SENSITIVITY = divergence(partner_decoy, ctx)
      how far a wrong-but-legal partner moves the prediction. Near zero
      => the model ignores slot content it judges untrustworthy (the
      shortcut story). Large, with CE worse than ctx => it is misled
      instead (the opposite story).

Slot contents are injected through the model's own self-conditioning
override (k=0 plus sc_mask/sc_emb), which both the A.3 and A.7 forwards
honour, so nothing about the model is monkey-patched. Models run in
eval mode, where A.7's decoy branch is inactive by construction -- the
probe measures LEARNED BEHAVIOUR under controlled inputs, not the
training-time kernel.

Usage (via probe_slot_use.sbatch):
    python probe_slot_use.py --ckpt A3=ckpt/<run>/ --ckpt A7=ckpt/<run>/
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


def interleave(net, batch, device):
    """Batch -> the interleaved, preprocessed tensor forward() expects."""
    x_mel, x_acc, pitch_shift = (t.to(device) if torch.is_tensor(t) else t
                                 for t in batch)
    x_mel, x_acc = net.preprocess(x_mel, pitch_shift, y=x_acc)
    B, T, S = x_mel.shape
    x = torch.stack([x_mel, x_acc], dim=2).view(B, T * 2, S)
    return x, T, S


def transpose_frame(net, tokens, semitones):
    """Shift the pitch-dur tokens of a frame, keeping programs/EOS/pad.

    pitch-dur ids are pitch + (dur+1)*128, so adding n semitones is
    +n on the id, clamped inside the pitch-dur range.
    """
    tok = net.tokenizer
    out = tokens.clone()
    S = tokens.shape[1]
    is_pd = (torch.arange(S, device=tokens.device) % 2 == 1)
    real = is_pd.unsqueeze(0) & (tokens != tok.pad_token) & \
        (tokens != tok.eos_token) & (tokens < tok.n_normal_tokens)
    out[real] = (out[real] + semitones).clamp(128, tok.n_normal_tokens - 1)
    return out


def frame_ce(logits, target, pad):
    """Mean CE over non-pad token positions of one frame."""
    V = logits.shape[-1]
    keep = (target != pad)
    if keep.sum() == 0:
        return float('nan')
    ce = F.cross_entropy(logits[keep].view(-1, V).float(),
                         target[keep].view(-1), reduction='mean')
    return float(ce)


def frame_divergence(logits_a, logits_b, target, pad):
    """Jensen-Shannon divergence between two predictive distributions,
    plus argmax disagreement, over non-pad positions."""
    keep = (target != pad)
    if keep.sum() == 0:
        return float('nan'), float('nan')
    pa = F.softmax(logits_a[keep].float(), dim=-1)
    pb = F.softmax(logits_b[keep].float(), dim=-1)
    m = 0.5 * (pa + pb)
    kl = lambda p, q: (p * (p.clamp_min(1e-12).log()
                            - q.clamp_min(1e-12).log())).sum(-1)
    js = (0.5 * kl(pa, m) + 0.5 * kl(pb, m)).mean()
    disagree = (pa.argmax(-1) != pb.argmax(-1)).float().mean()
    return float(js), float(disagree)


def split_slots(q_logits, B):
    """Query logits come back FLATTENED as [B*2, S, V] (local_decode's
    layout: index = b*2 + slot), not [B, 2, S, V]. Normalise, so the
    melody slot is [:, 0] and the chord slot [:, 1]."""
    if q_logits.dim() == 3:
        S, V = q_logits.shape[-2], q_logits.shape[-1]
        q_logits = q_logits.view(B, -1, S, V)
    return q_logits[:, 0], q_logits[:, 1]


@torch.no_grad()
def run_condition(net, x, t, K, mode, lag, semitones, device):
    """One forward with the slots set up for `mode`. Returns query
    logits [B, 2, S, V] (index 0 = melody slot, 1 = chord slot)."""
    B, L, S = x.shape
    T_full = L // 2
    full = lambda v: torch.full((B,), v, dtype=torch.long, device=device)
    none_sc = (None, None)

    def decoy_emb(mod, kind):
        t2 = (t + lag) % T_full
        tokens = x[:, 2 * t2 + mod]
        if kind == 'shift':
            tokens = transpose_frame(net, x[:, 2 * t + mod], semitones)
        return net._encode_frame(tokens, mod)

    k_m, k_c = full(K), full(K)
    sc_m, e_m = none_sc
    sc_c, e_c = none_sc
    if mode == 'ctx':
        pass
    elif mode == 'own_true':
        k_m = full(0)
    elif mode == 'partner_true':
        k_c = full(0)
    elif mode in ('partner_decoy', 'partner_shift'):
        k_c = full(0)
        sc_c = torch.ones(B, dtype=torch.bool, device=device)
        e_c = decoy_emb(1, 'shift' if mode == 'partner_shift' else 'lag')
    else:
        raise ValueError(mode)

    _, q_logits, _ = net(x, T_query=t, k_m=k_m, k_c=k_c,
                         sc_mask_m=sc_m, sc_emb_m=e_m,
                         sc_mask_c=sc_c, sc_emb_c=e_c)
    return q_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True,
                   help='NAME=PATH, repeatable (e.g. A3=ckpt/<run>/)')
    p.add_argument('--task', default='melchord')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--n_batches', type=int, default=25)
    p.add_argument('--lag', type=int, default=16,
                   help='decoy displacement in frames (16 = one bar)')
    p.add_argument('--semitones', type=int, default=1,
                   help='transposition for the pitch-shift condition')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = get_task(args.task)
    batches = collect_batches(task, args.batch_size, args.n_batches)
    print(f'[data] {len(batches)} val batches x {args.batch_size}')

    MODES = ['ctx', 'own_true', 'partner_true', 'partner_decoy',
             'partner_shift']
    for spec in args.ckpt:
        name, path = spec.split('=', 1)
        net = load_model(path).to(device)
        net.eval()
        K = net.diffusion_K
        pad = net.tokenizer.pad_token
        print(f'\n{"=" * 66}\n{name}: {os.path.basename(path.rstrip("/"))}\n'
              f'{"=" * 66}')

        acc = {m: {'ce_m': [], 'ce_c': [], 'js_m': [], 'js_c': [],
                   'dis_m': [], 'dis_c': []} for m in MODES}
        for batch in batches:
            x, T_full, S = interleave(net, batch, device)
            t = T_full // 2                      # a mid-sequence frame
            logits = {m: run_condition(net, x, t, K, m, args.lag,
                                       args.semitones, device)
                      for m in MODES}
            tgt_m, tgt_c = x[:, 2 * t], x[:, 2 * t + 1]
            B = x.shape[0]
            for m in MODES:
                lm, lc = split_slots(logits[m], B)
                acc[m]['ce_m'].append(frame_ce(lm, tgt_m, pad))
                acc[m]['ce_c'].append(frame_ce(lc, tgt_c, pad))
                cm, cc = split_slots(logits['ctx'], B)
                js, dis = frame_divergence(lm, cm, tgt_m, pad)
                acc[m]['js_m'].append(js)
                acc[m]['dis_m'].append(dis)
                js, dis = frame_divergence(lc, cc, tgt_c, pad)
                acc[m]['js_c'].append(js)
                acc[m]['dis_c'].append(dis)

        mean = lambda v: sum(v) / max(len(v), 1)
        print(f'{"condition":<16}{"CE mel":>9}{"CE chd":>9}'
              f'{"JS vs ctx (m/c)":>20}{"argmax diff (m/c)":>22}')
        for m in MODES:
            a = acc[m]
            print(f'{m:<16}{mean(a["ce_m"]):>9.4f}{mean(a["ce_c"]):>9.4f}'
                  f'{mean(a["js_m"]):>10.4f}/{mean(a["js_c"]):<9.4f}'
                  f'{mean(a["dis_m"]):>11.3f}/{mean(a["dis_c"]):<10.3f}')

        gain_m = mean(acc['ctx']['ce_m']) - mean(acc['partner_true']['ce_m'])
        gain_c = mean(acc['ctx']['ce_c']) - mean(acc['partner_true']['ce_c'])
        own_m = mean(acc['ctx']['ce_m']) - mean(acc['own_true']['ce_m'])
        harm_m = mean(acc['partner_decoy']['ce_m']) - mean(acc['ctx']['ce_m'])
        sens_m = mean(acc['partner_decoy']['js_m'])
        shift_m = mean(acc['partner_shift']['js_m'])
        print(f'\n  PARTNER GAIN   mel {gain_m:+.4f}   chord {gain_c:+.4f}'
              f'   (>0: the same-instant channel carries information)')
        print(f'  OWN-SLOT GAIN  mel {own_m:+.4f}'
              f'   (sanity: reading its own true frame must help a lot)')
        print(f'  DECOY  sensitivity(JS) {sens_m:.4f}   harm(CE) {harm_m:+.4f}')
        print(f'  SHIFT  sensitivity(JS) {shift_m:.4f}')
        verdict = ('IGNORES untrusted slot content'
                   if sens_m < 0.05 * max(own_m, 1e-6) or sens_m < 1e-3
                   else 'RESPONDS to slot content')
        print(f'  -> {verdict}')

    print(f'\n{"=" * 66}')
    print('How to read: compare PARTNER GAIN across models -- it is the '
          'information\nthe same-instant edge actually carries. If A.7 '
          'gains far less than A.3 while\nits DECOY sensitivity is ~0, '
          'the parametric corruption taught it to discard\nthe channel, '
          'and pitch-shift (also parametric, also invertible) would '
          'meet\nthe same fate. If A.7 gains as much but is MISLED '
          '(harm > 0), the channel\nlives and the failure is elsewhere.')


if __name__ == '__main__':
    main()
