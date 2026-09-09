"""Gate for A.10 (same-frame edge) before any training: does the stack do
exactly what cp_transformer_m2c_intra_cross_attn._build_masks says?

On a tiny randomly-initialised M2CIntraCrossAttn(same_frame=True) with
THREE global layers (so transitive leaks through the stack are
exercised -- the first A.10 design passed on one layer and failed on
two), with a teacher-forced interleaved input h = [sos_m, sos_c, m_0,
c_0, ...] (row 2t holds m_{t-1} and predicts m_t; row 2t+1 holds c_{t-1}
and predicts c_t), checks:

  1. direction=None reproduces A.1: masks equal the base masks and the
     stack's output equals a same_frame=False model's.
  2. direction 0 everywhere (chord follows melody), for every frame t:
     perturbing m_t (row 2t+2) changes the CHORD row 2t+1 and no row
     before it, and NOT the mel row 2t; perturbing c_t (row 2t+3)
     changes nothing at or before row 2t+2.
  3. direction 1 everywhere (melody follows chord): perturbing c_t
     changes the MEL row 2t and no row before it, and NOT the chord row
     2t+1 (its own target) nor row 2t+2; perturbing m_t changes nothing
     at or before row 2t+1.
  4. per-FRAME random directions (what training draws and the
     alternating decode produces): the frame-t statements of 2/3 hold
     for each frame under its own direction.
  5. decode consistency: under a random per-frame history, the
     follower's row computed from the truncated input the two-forward
     decode uses (rows through 2t+3, zero filler at the unknown row)
     equals the teacher-forced full-sequence row, and does not depend
     on the filler value.
  6. batched directions: a batch with mixed direction tensors gives,
     per item, the same output as that item alone.

Usage (via audit_same_frame.sbatch, CPU):
    python audit_same_frame.py
"""
import sys

import torch

from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn

fails = []


def check(cond, msg):
    print(f'  [{"PASS" if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(same_frame):
    torch.manual_seed(0)
    net = M2CIntraCrossAttn(large=False, with_velocity=False,
                            moe_num_experts=2, moe_topk=1,
                            global_num_layers=3, gate_init_bias=0.0,
                            same_frame=same_frame)
    net.eval()
    return net


def run(net, h, direction):
    net._sf_dir = direction
    try:
        with torch.no_grad():
            out, _ = net._global_interaction(h)
    finally:
        net._sf_dir = None
    return out


def changed_rows(a, b, tol=1e-6):
    return (a - b).abs().amax(dim=(0, 2)) > tol       # [2T] bool


def frame_checks(net, h, D, label, frames):
    """D: LongTensor [1, T]. For each frame t in `frames`, the
    perturbation statements under d_t = D[0, t]."""
    H = h.shape[-1]
    base = run(net, h, D)

    def perturbed(row):
        h2 = h.clone()
        h2[:, row] = torch.randn(H)
        return h2

    ok_all = True
    for t in frames:
        d = int(D[0, t])
        r_m, r_c, k_m, k_c = 2 * t, 2 * t + 1, 2 * t + 2, 2 * t + 3
        ch_m = changed_rows(base, run(net, perturbed(k_m), D))
        ch_c = changed_rows(base, run(net, perturbed(k_c), D))
        # Rows >= 2t+2 (frame t+1 on) may legitimately change: m_t and c_t
        # are in their past (the partner's previous frame is an encoding
        # key in both directions). Frame-t rows and earlier are the test.
        if d == 0:
            conds = [
                (bool(ch_m[r_c]), f't={t} d=0: m_t changes the chord row {r_c}'),
                (not bool(ch_m[r_m]), f't={t} d=0: m_t does NOT change the mel row {r_m} (its own target)'),
                (not bool(ch_m[:r_c].any()), f't={t} d=0: m_t changes no row before {r_c}'),
                (not bool(ch_c[:k_m].any()), f't={t} d=0: c_t changes nothing at or before row {r_c}'),
            ]
        else:
            conds = [
                (bool(ch_c[r_m]), f't={t} d=1: c_t changes the mel row {r_m}'),
                (not bool(ch_c[r_c]), f't={t} d=1: c_t does NOT change the chord row {r_c} (its own target)'),
                (not bool(ch_c[:r_m].any()), f't={t} d=1: c_t changes no row before {r_m}'),
                (not bool(ch_m[:k_m].any()), f't={t} d=1: m_t changes nothing at or before row {r_c}'),
            ]
        for cond, msg in conds:
            if not cond:
                ok_all = False
                print(f'    [FAIL] {msg}')
    check(ok_all, f'{label}: all per-frame statements hold for frames {list(frames)}')


def main():
    torch.manual_seed(1)
    T = 8
    net = build(True)
    net0 = build(False)
    net0.load_state_dict(net.state_dict(), strict=False)
    H = net.hidden_size
    B = 1
    h = torch.randn(B, 2 * T, H)
    frames = range(1, T - 1)

    print('--- 1. direction=None reproduces A.1 ---')
    layer = net.global_layers[0]
    mM, mC = layer._build_masks(2 * T, h.device, None)
    bM, bC = layer._build_base_masks(2 * T, h.device)
    check(torch.equal(mM, bM) and torch.equal(mC, bC), 'masks equal the base masks')
    o_none = run(net, h, None)
    o_a1 = run(net0, h, None)
    check(float((o_none - o_a1).abs().max()) < 1e-5,
          'same_frame model with direction=None == plain A.1 model')

    print('--- 2. direction 0 everywhere: chord follows melody ---')
    frame_checks(net, h, torch.zeros(1, T, dtype=torch.long), 'direction 0', frames)
    print('--- 3. direction 1 everywhere: melody follows chord ---')
    frame_checks(net, h, torch.ones(1, T, dtype=torch.long), 'direction 1', frames)
    print('--- 4. per-frame random directions ---')
    torch.manual_seed(7)
    D = torch.randint(0, 2, (1, T))
    print(f'    directions: {D[0].tolist()}')
    frame_checks(net, h, D, 'random per-frame', frames)
    D_alt = (torch.arange(T) % 2).view(1, T)
    frame_checks(net, h, D_alt, 'alternating (the decode rule)', frames)

    print('--- 5. decode consistency (truncated input == teacher-forced row; filler-free) ---')
    full = run(net, h, D)
    ok = True
    for t in frames:
        d = int(D[0, t])
        r_m, r_c, k_m, k_c = 2 * t, 2 * t + 1, 2 * t + 2, 2 * t + 3
        D_tr = D[:, :t + 2]
        follower_row = r_c if d == 0 else r_m
        unknown_row = k_c if d == 0 else k_m
        h_dec = h[:, :k_c + 1].clone()
        h_dec[:, unknown_row] = 0.0
        trunc = run(net, h_dec, D_tr)
        h_dec2 = h_dec.clone()
        h_dec2[:, unknown_row] = torch.randn(H)
        trunc2 = run(net, h_dec2, D_tr)
        e1 = float((full[:, follower_row] - trunc[:, follower_row]).abs().max())
        e2 = float((trunc[:, follower_row] - trunc2[:, follower_row]).abs().max())
        if e1 >= 1e-5 or e2 >= 1e-5:
            ok = False
            print(f'    [FAIL] t={t} d={d}: full-vs-truncated {e1:.2e}, filler sensitivity {e2:.2e}')
    check(ok, 'follower row from rows[:2t+4] equals the full-sequence row and ignores the filler, every frame')

    print('--- 6. batched directions ---')
    hb = torch.cat([h, torch.randn(1, 2 * T, H)], dim=0)
    Db = torch.cat([D, 1 - D], dim=0)
    mixed = run(net, hb, Db)
    solo0 = run(net, hb[:1], Db[:1])
    solo1 = run(net, hb[1:], Db[1:])
    check(float((mixed[0] - solo0[0]).abs().max()) < 1e-5 and
          float((mixed[1] - solo1[0]).abs().max()) < 1e-5,
          'per-item outputs identical in a mixed batch and alone')
    # [B] directions still accepted (broadcast over frames)
    solo_b = run(net, hb[:1], torch.zeros(1, dtype=torch.long))
    solo_bt = run(net, hb[:1], torch.zeros(1, T, dtype=torch.long))
    check(float((solo_b - solo_bt).abs().max()) < 1e-6, 'direction [B] == direction [B, T] broadcast')

    print()
    if fails:
        print(f'{len(fails)} CHECK(S) FAILED:')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED -- A.10 masks behave as documented; safe to train')


if __name__ == '__main__':
    main()
