"""Does the STARTING POINT of the revise round treat the two streams alike?

The A.12 symmetric design rests on a claim about the AR drafts that
seed every decode: melody t and chord t are drafted BLIND TO EACH
OTHER, so neither stream enters the revise round with an advantage.
The claim was first made by reading `_build_masks`, which is exactly
the kind of reasoning that has been wrong here before, so this audit
tests it CAUSALLY instead -- perturb one stream's frame t and watch
whether the other stream's frame-t logits move.

Checks:

  1. melody t's AR logits do not move when chord t is changed.
  2. chord t's AR logits do not move when melody t is changed.
  3. (positive control, so 1-2 are not vacuous) melody t's AR logits DO
     move when chord t-1 changes, and chord t's when melody t-1 does.
  4. the blindness holds at every frame, not just one.
  5. the two query slots DO see each other: perturbing the chord slot's
     content moves the melody slot's logits and vice versa. This is the
     channel the revise round runs on, so it must be open in BOTH
     directions -- the starting point is symmetric because neither
     stream can see the other, the revise round because both can.

If 1-3 hold, the AR seed is symmetric in the only sense that matters:
whatever each stream knows about the other, it knows through frame
t-1, and neither can read the partner's current frame.

Usage (via audit_stream_symmetry.sbatch, CPU):
    python audit_stream_symmetry.py
"""
import sys

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []
TOL = 1e-5


def check(cond, msg):
    print(f'  [{"PASS" if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(seed=0):
    torch.manual_seed(seed)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False, moe_num_experts=4, moe_topk=2,
        global_num_layers=3, gate_init_bias=0.0, diffusion_K=1,
        slot_sees_prev_frame=True, cond_slot_prob=0.0,
        self_cond_prob=0.0, moe_aux_clean_only=True)
    # eval: the local encoder has dropout, so two identical training-mode
    # forwards would not agree and every comparison below would be noise.
    net.eval()
    return net


def batch(B=4, T=12, S=8, seed=1):
    torch.manual_seed(seed)
    x = torch.zeros(B, T, S // 4, 4, dtype=torch.long)
    x[..., 0] = 24
    x[..., 1] = torch.randint(40, 80, (B, T, S // 4))
    x[..., 2] = torch.randint(0, 8, (B, T, S // 4))
    x[..., 3] = 64
    x[:, :, 1, 0] = 254
    x[:, :, 1, 1] = 255
    y = x.clone()
    y[..., 1] = torch.randint(40, 80, (B, T, S // 4))
    return (x.view(B, T, S), y.view(B, T, S), torch.zeros(B, dtype=torch.long))


def interleave(net, b):
    B, T_full = b[0].shape[0], b[0].shape[1]
    x_m, x_c = net.preprocess(b[0], b[2], y=b[1])
    return torch.stack([x_m, x_c], dim=2).view(B, 2 * T_full, -1), T_full


def run(net, xx, t_j, k=0):
    """One forward at pinned levels. Returns (ar_logits, query_logits)."""
    B = xx.shape[0]
    kk = torch.full((B, 1), k, dtype=torch.long)
    with torch.no_grad():
        ar, q, _ = net.forward(xx, T_query=(t_j,), k_m=kk, k_c=kk)
    # Both come back flattened over the row axis: ar is [B*2T, S, V] and
    # q is [B*2Q, S, V]. Restore the row axis so a row can be indexed by
    # its interleaved position.
    ar = ar.reshape(B, -1, ar.shape[-2], ar.shape[-1])
    q = q.reshape(B, -1, q.shape[-2], q.shape[-1])
    return ar, q


def perturb(xx, row, seed):
    """Replace the PITCH field of one interleaved row with new values."""
    out = xx.clone()
    g = torch.Generator().manual_seed(seed)
    n = out.shape[-1]
    # cp frames are (program, pitch-dur) pairs; move the odd positions,
    # which carry content, and leave structure (program/EOS) alone.
    idx = torch.arange(1, n, 2)
    out[:, row, idx] = torch.randint(
        40, 80, (out.shape[0], idx.numel()), generator=g)
    return out


def main():
    net = build()
    b = batch()
    xx, T_full = interleave(net, b)
    t = 6                      # the frame under test
    t_j = 8                    # query frame, kept clear of t

    print('--- 1/2. the AR drafts of frame t are blind to each other ---')
    ar0, _ = run(net, xx, t_j)
    # row 2t predicts melody t, row 2t+1 predicts chord t.
    base_m, base_c = ar0[:, 2 * t], ar0[:, 2 * t + 1]

    ar_c, _ = run(net, perturb(xx, 2 * t + 1, 11), t_j)   # move chord t
    d_m = float((ar_c[:, 2 * t] - base_m).abs().max())
    check(d_m < TOL,
          f'melody {t} AR logits unmoved by chord {t} (max |d| {d_m:.2e})')

    ar_m, _ = run(net, perturb(xx, 2 * t, 12), t_j)       # move melody t
    d_c = float((ar_m[:, 2 * t + 1] - base_c).abs().max())
    check(d_c < TOL,
          f'chord {t} AR logits unmoved by melody {t} (max |d| {d_c:.2e})')

    print('--- 3. positive control: frame t-1 DOES reach across ---')
    ar_cp, _ = run(net, perturb(xx, 2 * (t - 1) + 1, 13), t_j)
    d_mp = float((ar_cp[:, 2 * t] - base_m).abs().max())
    check(d_mp > TOL,
          f'melody {t} AR logits DO move with chord {t - 1} '
          f'(max |d| {d_mp:.2e})')
    ar_mp, _ = run(net, perturb(xx, 2 * (t - 1), 14), t_j)
    d_cp = float((ar_mp[:, 2 * t + 1] - base_c).abs().max())
    check(d_cp > TOL,
          f'chord {t} AR logits DO move with melody {t - 1} '
          f'(max |d| {d_cp:.2e})')

    print('--- 4. blindness holds at every frame ---')
    bad = []
    for tt in range(1, T_full - 1):
        a_c, _ = run(net, perturb(xx, 2 * tt + 1, 100 + tt), t_j)
        a_m, _ = run(net, perturb(xx, 2 * tt, 200 + tt), t_j)
        dm = float((a_c[:, 2 * tt] - ar0[:, 2 * tt]).abs().max())
        dc = float((a_m[:, 2 * tt + 1] - ar0[:, 2 * tt + 1]).abs().max())
        if dm >= TOL or dc >= TOL:
            bad.append((tt, dm, dc))
    check(not bad,
          f'frames 1..{T_full - 2}: neither stream reads the partner\'s '
          f'own frame' + (f' -- LEAKS AT {bad}' if bad else ''))

    print('--- 5. the two query slots DO see each other (both ways) ---')
    # k=0 puts each slot's ground-truth frame in it, so moving frame t_j
    # in one stream moves that slot's CONTENT. The partner slot reaches
    # it only through the frame pass: with slot_sees_prev_frame the
    # slots read clean rows up to prediction-frame t_j, which hold frame
    # t_j - 1, so the perturbed clean row (prediction-frame t_j + 1) is
    # masked out for both slots and the slot link is the only route.
    _, q0 = run(net, xx, t_j)
    q_m0, q_c0 = q0[:, 0], q0[:, 1]
    _, q_pc = run(net, perturb(xx, 2 * t_j + 1, 21), t_j)
    d_slot_m = float((q_pc[:, 0] - q_m0).abs().max())
    check(d_slot_m > TOL,
          f'melody slot reads the chord slot (max |d| {d_slot_m:.2e})')
    _, q_pm = run(net, perturb(xx, 2 * t_j, 22), t_j)
    d_slot_c = float((q_pm[:, 1] - q_c0).abs().max())
    check(d_slot_c > TOL,
          f'chord slot reads the melody slot (max |d| {d_slot_c:.2e})')

    print()
    if fails:
        print(f'{len(fails)} CHECK(S) FAILED:')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED -- the AR seed is symmetric (neither stream '
          'sees the other\'s current frame); the revise round is symmetric '
          '(both slots see each other)')


if __name__ == '__main__':
    main()
