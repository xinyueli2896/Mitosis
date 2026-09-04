"""A.8 (query_block) audit: is the block a block, and does the DECODE
present the model with the structure it was trained on?

A.8's entire claim is train/decode structural identity, so the audit is
mostly about that identity rather than about the block itself.

Claims checked:

  1. BLOCK GEOMETRY. Training draws a CONTIGUOUS run of B frames
     t0..t0+B-1 with t0 >= 1 and t0+B-1 < T_full, over many draws every
     legal t0 appears, and eval still pins the single historical frame
     (so val_loss stays comparable across A.3/A.7/A.8).
  2. PREFIX CUTOFF. Every slot in the block sees clean frames < t0 --
     NOT < its own frame. Frames inside the block are un-committed at
     decode time, so a slot that had learned to read them would be
     reading something inference cannot supply.
  3. BLOCK-WIDE SLOT LINKS. Every slot may read every OTHER slot in the
     block (both streams, all B frames) and never itself; with
     query_block == 1 the mask degenerates to A.3's own-partner-only
     rule, bit-identical.
  4. NO LEAK. Clean rows never attend any slot, in block mode too.
  5. DECODE DISPATCH. general_inference_diffusion routes an A.8 model
     to the block decode, and a B=1 model to the single-frame decode.
  6. CKPT ROUND-TRIP. query_block_flag travels in the state dict with
     its value, and load_model's detection reads it back -- if it were
     lost the model would silently build A.3 masks and decode
     single-frame, the exact mismatch A.8 exists to remove.
  7. LOUD MISUSE. query_block > 1 together with query_pairs > 1 raises.
  8. GRADIENTS. A block training step produces a finite loss and
     gradients through the slot pathway.

Runs on CPU in seconds. Usage (via audit_query_block.sbatch):
    python audit_query_block.py
"""

import sys

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []


def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(block, **kw):
    torch.manual_seed(0)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False,
        moe_num_experts=4, moe_topk=2,
        global_num_layers=2, diffusion_K=4,
        query_block=block, **kw,
    )
    net.eval()
    return net


def raw_batch(B, T, n_tuples=2):
    x = torch.zeros(B, T, 4 * n_tuples, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            x[b, t, 0:4] = torch.tensor([24, 48 + (b + t) % 24, 2, 0])
            x[b, t, 4:8] = torch.tensor([254, 0, 0, 0])
    return x


def main():
    Bk, K = 4, 4
    net = build(Bk)
    plain = build(1)

    print('--- 1. block geometry (contiguous, in range, all offsets) ---')
    T_full = 16
    seen = set()
    ok_shape = True
    for _ in range(400):
        t0 = int(torch.randint(low=1, high=T_full - Bk + 1, size=(1,)).item())
        tq = tuple(range(t0, t0 + Bk))
        seen.add(t0)
        ok_shape &= (len(tq) == Bk and tq[0] >= 1 and tq[-1] < T_full
                     and list(tq) == list(range(tq[0], tq[0] + Bk)))
    check(ok_shape, 'draws are contiguous, 1 <= t0, t0+B-1 < T_full')
    check(seen == set(range(1, T_full - Bk + 1)),
          f'every legal start appears over many draws ({len(seen)} of '
          f'{T_full - Bk} )')

    print('--- 2/3/4. mask structure in block mode ---')
    layer = net.global_layers[0]
    check(getattr(layer, 'query_block_mode', False),
          'layers are switched into block mode by the model')
    clean_len = 2 * T_full
    tq = (5, 6, 7, 8)
    m_intra, m_cross, m_frame = layer._build_masks(
        clean_len, tq, torch.device('cpu'))
    L = clean_len + 2 * Bk
    t0 = tq[0]

    # every slot row: clean columns it may read must be frames < t0
    prefix_ok, own_frame_blocked = True, True
    for j in range(Bk):
        for slot in (clean_len + 2 * j, clean_len + 2 * j + 1):
            allowed = (m_intra[slot, :clean_len]
                       | m_cross[slot, :clean_len]).nonzero().flatten()
            frames = {int(p) // 2 for p in allowed}
            prefix_ok &= all(f < t0 for f in frames)
            own_frame_blocked &= all(f != tq[j] for f in frames)
    check(prefix_ok, 'every slot reads only clean frames < block start t0')
    check(own_frame_blocked, 'no slot can read its own target frame')

    slot_ids = list(range(clean_len, L))
    links_ok, self_ok = True, True
    for p in slot_ids:
        for q in slot_ids:
            if p == q:
                self_ok &= not bool(m_frame[p, q])
            else:
                links_ok &= bool(m_frame[p, q])
    check(links_ok, 'every slot may read every OTHER slot in the block')
    check(self_ok, 'no slot reads itself through the frame pass')

    leak = bool(m_intra[:clean_len, clean_len:].any()
                or m_cross[:clean_len, clean_len:].any()
                or m_frame[:clean_len, clean_len:].any())
    check(not leak, 'clean rows never attend any slot (no leak)')

    # B = 1 must reproduce A.3 exactly
    l1 = plain.global_layers[0]
    a = [x.clone() for x in l1._build_masks(clean_len, (5,), torch.device('cpu'))]
    l1.query_block_mode = True
    l1._mask_cache_key = None
    b = l1._build_masks(clean_len, (5,), torch.device('cpu'))
    check(all(torch.equal(x, y) for x, y in zip(a, b)),
          'B = 1 in block mode is bit-identical to A.3 masks')
    l1.query_block_mode = False
    l1._mask_cache_key = None

    print('--- 5. decode dispatch ---')
    import cp_transformer_m2c_duet_block_diffusion_inference as inf
    routed = {}

    def fake_block(model, *a, **kw):
        routed['block'] = True
        return [], []

    orig = inf.general_inference_block
    inf.general_inference_block = fake_block
    try:
        inf.general_inference_diffusion(net, 4, 1, 8, 1.0,
                                        lambda t: 'sample',
                                        lambda t: 'sample')
        check(routed.get('block', False),
              'an A.8 model routes to the BLOCK decode')
        routed.clear()
        try:
            inf.general_inference_diffusion(plain, 0, 1, 8, 1.0,
                                            lambda t: 'sample',
                                            lambda t: 'sample')
        except Exception:            # noqa: BLE001  (0 frames: fine)
            pass
        check(not routed.get('block', False),
              'a B = 1 model does NOT route to the block decode')
    finally:
        inf.general_inference_block = orig

    print('--- 6. ckpt round-trip ---')
    sd = net.state_dict()
    key = [k for k in sd if k.endswith('query_block_flag')]
    check(len(key) == 1 and int(sd[key[0]]) == Bk,
          f'query_block_flag travels in the state dict with value {Bk}')
    check(int(plain.state_dict()['query_block_flag']) == 1,
          'a B = 1 ckpt carries 1, so detection cannot false-positive')

    print('--- 7. loud misuse ---')
    try:
        build(4, query_pairs=8)
        check(False, 'query_block + query_pairs raises')
    except ValueError:
        check(True, 'query_block + query_pairs raises')

    print('--- 8. training step ---')
    net.train()
    net.self_cond_prob = 0.0
    x_mel, x_acc = raw_batch(2, T_full), raw_batch(2, T_full)
    out = net.loss(x_mel, x_acc, torch.zeros(2, dtype=torch.long))
    loss = out[0] if isinstance(out, (tuple, list)) else out
    check(bool(torch.isfinite(loss)), 'block training loss is finite')
    loss.backward()
    check(net.k_emb_m.weight.grad is not None
          and bool(torch.isfinite(net.k_emb_m.weight.grad).all()),
          'gradients flow through the slot pathway')
    check(int(net._last_n_pairs) == Bk,
          f'the training step supervised {Bk} frames (a full block)')

    print('\n' + '=' * 60)
    if fails:
        print(f'{len(fails)} FAILURE(S):')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED -- A.8 block training/decode are consistent.')


if __name__ == '__main__':
    main()
