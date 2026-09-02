"""A.7 (decoy_corruption) audit: does lag-graded decoy corruption do
exactly what it claims, and nothing else?

Claims checked:

  1. EVAL EQUIVALENCE (val_loss comparability). The decoy branch is
     training-only: in eval mode an A.7 model with identical weights
     produces BIT-IDENTICAL forward outputs to a plain A.3 model at
     k_m=k_c=K -- validation still pins the fully-masked setting, so
     val_loss is computed identically across the whole A-family.
  2. k=0 IS THE TRUTH. _decoy_frame_slot at k=0 returns exactly the
     stream's own frame embedding at t (lag 0, no mask), revealed=True.
  3. LAG DRAW RESPECTS THE BINS. _draw_decoy_lags: k=0 -> 0; k=1..K-1
     magnitudes inside decoy_lag_bins[k-1] inclusive; k=K magnitudes in
     [1, T_full-1]; both signs occur.
  4. CIRCULAR, PARITY-CORRECT GATHER. The decoy content equals
     h[:, 2*((t+lag) mod T) + mod] -- wraps at both window edges, and
     mod selects the SAME stream (melody slots never receive chord
     frames or vice versa).
  5. RESIDUAL MASK ONLY AT k=K. residual=1.0 -> every k=K slot is the
     mask embedding; residual=0.0 -> none is; k<K never draws it.
  6. SELF-COND OVERRIDE. Items with sc_mask set carry sc_emb as content
     (any k), and are never marked revealed.
  7. REVEALED BOOKKEEPING. After a training forward,
     _last_query_revealed is True exactly where k==0 and no self-cond.
  8. TARGETS UNTOUCHED + GRADIENTS FLOW. x is not modified in place;
     loss() with the decoy branch active is finite and backward()
     produces gradients (k_emb included).
  9. LOUD MISUSE. decoy+token_level_mask raises; decoy+
     mask_revealed_query_loss raises; wrong bin count raises.
  10. FLAG TRAVELS BY VALUE. Both plain and A.7 checkpoints carry
     decoy_corruption_flag (0/1).

Runs on CPU in seconds. Usage (via audit_decoy_corruption.sbatch):
    python audit_decoy_corruption.py
"""

import sys

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []


def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(decoy, **kw):
    torch.manual_seed(0)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False,
        moe_num_experts=4, moe_topk=2,
        global_num_layers=2, diffusion_K=4,
        decoy_corruption=decoy, **kw,
    )
    net.eval()
    return net


def make_frame(net, n_notes, prog):
    tok = net.tokenizer
    S = 8
    f = torch.full((S,), tok.pad_token, dtype=torch.long)
    for n in range(n_notes):
        f[2 * n] = prog
        f[2 * n + 1] = 128 + 40 * n
    f[2 * n_notes] = tok.eos_token
    return f


def toy_batch(net, B=2, T=8):
    return torch.stack([
        torch.stack([make_frame(net, (b + i) % 3 + 1,
                                24 if i % 2 == 0 else 0)
                     for i in range(2 * T)])
        for b in range(B)
    ])


def coded_h(B, T_full, H):
    """Deterministic 'embeddings' encoding (stream, frame) in two dims,
    so gather mistakes are unmissable."""
    h = torch.zeros(B, 2 * T_full, H)
    for pos in range(2 * T_full):
        h[:, pos, 0] = pos % 2          # stream parity
        h[:, pos, 1] = pos // 2         # frame index
    return h


def main():
    K = 4
    net = build(True)
    plain = build(False)
    # Identical weights: A.7's extra state is the flag buffer only.
    plain.load_state_dict(
        {k: v for k, v in net.state_dict().items()
         if k != 'decoy_corruption_flag'}, strict=False)
    B, T = 2, 8
    x = toy_batch(net, B, T)

    print('--- 1. eval equivalence at k=K (val_loss comparability) ---')
    torch.manual_seed(7)
    ar_a, q_a, _ = net(x, T_query=T - 1,
                       k_m=torch.full((B,), K, dtype=torch.long),
                       k_c=torch.full((B,), K, dtype=torch.long))
    torch.manual_seed(7)
    ar_b, q_b, _ = plain(x, T_query=T - 1,
                         k_m=torch.full((B,), K, dtype=torch.long),
                         k_c=torch.full((B,), K, dtype=torch.long))
    check(torch.equal(ar_a, ar_b) and torch.equal(q_a, q_b),
          'eval-mode A.7 forward == plain A.3 forward, bit-identical')

    print('--- 2/4/5/6. slot construction on coded embeddings ---')
    H = net.hidden_size
    h = coded_h(B, T, H)
    mask_emb = torch.full((B, 1, H), 999.0)
    t_j, k0 = 3, torch.zeros(B, dtype=torch.long)
    slot, rev = net._decoy_frame_slot(h, t_j, k0, None, None, 0, T, mask_emb)
    check(torch.equal(slot, h[:, 2 * t_j:2 * t_j + 1]),
          'k=0: slot is exactly the true frame embedding')
    check(bool(rev.all()), 'k=0: revealed=True without self-cond')

    fixed_lag = 6                        # wraps: (3+6) % 8 = 1
    orig_draw = net._draw_decoy_lags
    net._draw_decoy_lags = lambda k_t, T_full: torch.full(
        (k_t.shape[0],), fixed_lag, dtype=torch.long)
    for mod in (0, 1):
        k2 = torch.full((B,), 2, dtype=torch.long)
        slot, rev = net._decoy_frame_slot(h, t_j, k2, None, None, mod,
                                          T, mask_emb)
        want = h[:, 2 * ((t_j + fixed_lag) % T) + mod].unsqueeze(1)
        check(torch.equal(slot, want),
              f'mod={mod}: decoy == h at 2*((t+lag) mod T)+{mod} '
              f'(circular wrap, same stream)')
        check(bool((slot[:, 0, 0] == mod).all()),
              f'mod={mod}: stream parity of the gathered frame is {mod}')
        check(not bool(rev.any()), f'mod={mod}: k>0 never marked revealed')
    net._draw_decoy_lags = orig_draw

    kK = torch.full((B,), K, dtype=torch.long)
    net.decoy_mask_residual = 1.0
    slot, _ = net._decoy_frame_slot(h, t_j, kK, None, None, 0, T, mask_emb)
    check(torch.equal(slot, mask_emb),
          'residual=1.0: every k=K slot is the mask embedding')
    net.decoy_mask_residual = 0.0
    slot, _ = net._decoy_frame_slot(h, t_j, kK, None, None, 0, T, mask_emb)
    check(not bool((slot == 999.0).any()),
          'residual=0.0: no k=K slot is the mask embedding')
    net.decoy_mask_residual = 1.0
    k3 = torch.full((B,), 3, dtype=torch.long)
    slot, _ = net._decoy_frame_slot(h, t_j, k3, None, None, 0, T, mask_emb)
    check(not bool((slot == 999.0).any()),
          'k<K never draws the residual mask even at residual=1.0')
    net.decoy_mask_residual = 0.25

    sc_mask = torch.tensor([True, False])
    sc_emb = torch.full((B, 1, H), -5.0)
    slot, rev = net._decoy_frame_slot(h, t_j, k0, sc_mask, sc_emb, 0,
                                      T, mask_emb)
    check(torch.equal(slot[0], sc_emb[0]) and
          torch.equal(slot[1], h[1, 2 * t_j:2 * t_j + 1]),
          'self-cond override replaces content only where its mask is set')
    check(bool(rev.tolist() == [False, True]),
          'self-conditioned items are never marked revealed')

    print('--- 3. lag draw respects the bins ---')
    T_big = 512
    draws_ok, sign_seen = True, set()
    torch.manual_seed(3)
    for level in range(0, K + 1):
        k_t = torch.full((4096,), level, dtype=torch.long)
        lags = net._draw_decoy_lags(k_t, T_big)
        mag = lags.abs()
        if level == 0:
            draws_ok &= bool((lags == 0).all())
        elif level == K:
            draws_ok &= bool(((mag >= 1) & (mag <= T_big - 1)).all())
        else:
            lo, hi = net.decoy_lag_bins[level - 1]
            draws_ok &= bool(((mag >= lo) & (mag <= hi)).all())
        if level > 0:
            sign_seen |= {int(s) for s in torch.sign(lags).unique().tolist()}
    check(draws_ok, 'magnitudes: 0 at k=0; inside bins at k=1..K-1; '
                    '[1, T-1] at k=K')
    check(sign_seen == {-1, 1}, 'both shift directions occur')

    print('--- 7/8. training forward: bookkeeping, targets, grads ---')
    net.train()
    torch.manual_seed(11)
    x_before = x.clone()
    k_m = torch.tensor([0, K], dtype=torch.long)
    k_c = torch.tensor([2, 0], dtype=torch.long)
    net.self_cond_prob = 0.0
    ar, q, aux = net(x, T_query=4, k_m=k_m, k_c=k_c)
    check(torch.equal(x, x_before), 'input tokens untouched by the forward')
    revealed = net._last_query_revealed          # [B, 2Q, S]
    want_rev = torch.tensor([[True, False], [False, True]])
    got_rev = revealed[:, :, 0]
    check(bool((got_rev == want_rev).all()),
          '_last_query_revealed true exactly where k==0 (no self-cond)')
    x_mel = x[:, 0::2]
    x_acc = x[:, 1::2]
    out = net.loss(x_mel, x_acc, torch.zeros(B, dtype=torch.long))
    loss = out[0] if isinstance(out, (tuple, list)) else out
    check(bool(torch.isfinite(loss)), 'training loss is finite')
    loss.backward()
    check(net.k_emb_m.weight.grad is not None and
          bool(torch.isfinite(net.k_emb_m.weight.grad).all()),
          'gradients flow (k_emb included)')

    print('--- 9. loud misuse ---')
    for kw, name in (
        (dict(token_level_mask=True), 'decoy + token_level_mask'),
        (dict(mask_revealed_query_loss=True),
         'decoy + mask_revealed_query_loss'),
        (dict(decoy_lag_bins=[(1, 4)]), 'wrong bin count'),
    ):
        try:
            build(True, **kw)
            check(False, f'{name} raises')
        except ValueError:
            check(True, f'{name} raises')

    print('--- 10. flag travels by value ---')
    check(int(net.state_dict()['decoy_corruption_flag']) == 1 and
          int(build(False).state_dict()['decoy_corruption_flag']) == 0,
          'decoy_corruption_flag present with the right value in both')

    print('\n' + '=' * 60)
    if fails:
        print(f'{len(fails)} FAILURE(S):')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED -- A.7 decoy corruption is safe to train.')


if __name__ == '__main__':
    main()
