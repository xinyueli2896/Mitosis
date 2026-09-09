"""Gate for A.10 (same-frame edge) before any training: does the mask do
exactly what cp_transformer_m2c_intra_cross_attn._build_masks says?

On a tiny randomly-initialised M2CIntraCrossAttn(same_frame=True), with
a teacher-forced interleaved input h = [sos_m, sos_c, m_0, c_0, ...]
(row 2t holds m_{t-1} and predicts m_t; row 2t+1 holds c_{t-1} and
predicts c_t), checks:

  1. direction=None reproduces A.1: masks equal the base masks and the
     global stack's output equals a same_frame=False model's.
  2. direction 0 (chord follows melody): perturbing row 2t+2 (m_t)
     changes the CHORD row 2t+1 and no row before it; the MEL row 2t is
     unchanged; perturbing row 2t+3 (c_t) changes nothing at or before
     row 2t+2.
  3. direction 1 (melody follows chord): perturbing row 2t+3 (c_t)
     changes the MEL row 2t and no row before it; the chord row 2t+1 is
     unchanged; perturbing row 2t+2 (m_t) changes nothing at or before
     row 2t+1.
  4. decode consistency: the follower's row computed from the truncated
     input the two-forward decode uses (rows through the leader's key,
     with the zero filler under direction 1) equals the teacher-forced
     full-sequence row -- i.e. the follower reads nothing past its
     leader's key.
  5. batched directions: a batch with mixed directions gives, per item,
     the same output as that item alone with its own direction.

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
                            global_num_layers=2, gate_init_bias=0.0,
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


def main():
    torch.manual_seed(1)
    T = 6
    net = build(True)
    net0 = build(False)
    net0.load_state_dict(net.state_dict(), strict=False)
    H = net.hidden_size
    B = 1
    h = torch.randn(B, 2 * T, H)

    print('--- 1. direction=None reproduces A.1 ---')
    layer = net.global_layers[0]
    mM, mC = layer._build_masks(2 * T, h.device, None)
    bM, bC = layer._build_base_masks(2 * T, h.device)
    check(torch.equal(mM, bM) and torch.equal(mC, bC), 'masks equal the base masks')
    o_none = run(net, h, None)
    o_a1 = run(net0, h, None)
    check(float((o_none - o_a1).abs().max()) < 1e-5,
          'same_frame model with direction=None == plain A.1 model')

    t = 2
    r_m, r_c, k_m, k_c = 2 * t, 2 * t + 1, 2 * t + 2, 2 * t + 3

    def perturbed(row):
        h2 = h.clone()
        h2[:, row] = torch.randn(H)
        return h2

    for d, name in ((0, 'chord follows melody'), (1, 'melody follows chord')):
        print(f'--- {2 + d}. direction {d}: {name} ---')
        dir_t = torch.full((B,), d, dtype=torch.long)
        base = run(net, h, dir_t)
        ch_m = changed_rows(base, run(net, perturbed(k_m), dir_t))
        ch_c = changed_rows(base, run(net, perturbed(k_c), dir_t))
        if d == 0:
            check(bool(ch_m[r_c]), f'perturbing m_t (row {k_m}) changes the chord row {r_c}')
            check(not bool(ch_m[r_m]), f'... and NOT the mel row {r_m}')
            check(not bool(ch_m[:r_c].any()), f'... and no row before {r_c}')
            check(not bool(ch_c[:k_m + 1].any()), f'perturbing c_t (row {k_c}) changes nothing at or before row {k_m}')
        else:
            check(bool(ch_c[r_m]), f'perturbing c_t (row {k_c}) changes the mel row {r_m}')
            check(not bool(ch_c[r_c]), f'... and NOT the chord row {r_c}')
            check(not bool(ch_c[:r_m].any()), f'... and no row before {r_m}')
            check(not bool(ch_m[:r_c + 1].any()), f'perturbing m_t (row {k_m}) changes nothing at or before row {r_c}')

    print('--- 4. decode consistency (truncated input == teacher-forced row) ---')
    dir0 = torch.zeros(B, dtype=torch.long)
    full0 = run(net, h, dir0)
    h_dec0 = h[:, :k_c + 1].clone()
    h_dec0[:, k_c] = 0.0                                # the decode's zero filler at c_t
    trunc0 = run(net, h_dec0, dir0)
    check(float((full0[:, r_c] - trunc0[:, r_c]).abs().max()) < 1e-5,
          'direction 0: chord row from rows[:2t+4] with a zero filler at c_t equals the full-sequence row')
    dir1 = torch.ones(B, dtype=torch.long)
    full1 = run(net, h, dir1)
    h_dec = h[:, :k_c + 1].clone()
    h_dec[:, k_m] = 0.0                                 # the decode's zero filler at m_t
    trunc1 = run(net, h_dec, dir1)
    check(float((full1[:, r_m] - trunc1[:, r_m]).abs().max()) < 1e-5,
          'direction 1: mel row from rows[:2t+4] with a zero filler at m_t equals the full-sequence row')

    print('--- 5. batched directions ---')
    hb = torch.cat([h, torch.randn(1, 2 * T, H)], dim=0)
    mixed = run(net, hb, torch.tensor([0, 1]))
    solo0 = run(net, hb[:1], torch.tensor([0]))
    solo1 = run(net, hb[1:], torch.tensor([1]))
    check(float((mixed[0] - solo0[0]).abs().max()) < 1e-5 and
          float((mixed[1] - solo1[0]).abs().max()) < 1e-5,
          'per-item outputs identical in a mixed batch and alone')

    print()
    if fails:
        print(f'{len(fails)} CHECK(S) FAILED:')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED -- A.10 masks behave as documented; safe to train')


if __name__ == '__main__':
    main()
