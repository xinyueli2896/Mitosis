"""Audit for A.9 / A.3f (slot_sees_prev_frame, query_pairs=-1).

What the query slot for frame t can read from the clean stream is a
question about one mask line, so ask the mask directly rather than
infer it from a training curve.

The clean stream is shifted one frame right: clean row p holds frame
p//2 - 1 and PREDICTS frame p//2. Rows 2t and 2t+1 hold frame t-1 (both
streams) and predict t; row 2t+2 is the first to hold frame t itself.

  historical mask (flag off): slot admits rows predicting < t
                              -> content up to t-2 only
  fixed mask      (flag on) : slot admits rows predicting <= t
                              -> content up to t-1, never frame t

Checks:
  1. flag off: the slot for t cannot reach rows 2t, 2t+1 (documents the
     off-by-one every A-family checkpoint was trained with)
  2. flag on : the slot reaches rows 2t, 2t+1 and NOT 2t+2, 2t+3 (no
     target leak), in intra and cross alike
  3. both    : clean rows never see slots; a slot sees only its own
     partner among slots, at Q=all as at Q=1
  4. Q=all   : a real training step with gradients, n_pairs = T-1,
     every frame 1..T-1 supervised once, finite loss; eval path stays
     single-frame
  5. the flag round-trips through state_dict and switches the layers
  6. the vectorised rotary index assignment equals the historical loop

Exit 0 = all checks passed; the A.9 / A.3f training commands are
unblocked.
"""
import sys

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []


def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(**kw):
    torch.manual_seed(0)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False,
        moe_num_experts=4, moe_topk=2,
        global_num_layers=2, diffusion_K=4, **kw,
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


def reach(layer, clean_len, tq, slot_row):
    """clean rows the given slot row may read (intra | cross)."""
    m_intra, m_cross, m_frame = layer._build_masks(
        clean_len, tq, torch.device('cpu'))
    allowed = (m_intra[slot_row, :clean_len]
               | m_cross[slot_row, :clean_len]).nonzero().flatten()
    return {int(p) for p in allowed}, (m_intra, m_cross, m_frame)


def main():
    T_full, t = 16, 7
    clean_len = 2 * T_full
    old = build()
    new = build(slot_sees_prev_frame=True)

    print('--- 1. historical mask: slot blind to frame t-1 ---')
    rows, _ = reach(old.global_layers[0], clean_len, t, clean_len)
    check(not getattr(old.global_layers[0], 'slot_sees_prev_frame', False),
          'flag off leaves the layers in the historical mode')
    check(2 * t not in rows and 2 * t + 1 not in rows,
          f'slot for t={t} cannot reach rows {2*t},{2*t+1} (content t-1)')
    check(max(rows) == 2 * t - 1,
          f'its last visible clean row is {max(rows)} (content t-2)')

    print('--- 2. fixed mask: slot reaches t-1, never t ---')
    lay = new.global_layers[0]
    check(getattr(lay, 'slot_sees_prev_frame', False),
          'flag on switches the layers')
    for slot_row, name in ((clean_len, 'melody slot'),
                           (clean_len + 1, 'chord slot')):
        rows, masks = reach(lay, clean_len, t, slot_row)
        check({2 * t, 2 * t + 1} <= rows,
              f'{name} reaches rows {2*t},{2*t+1} (content t-1, both streams)')
        check(2 * t + 2 not in rows and 2 * t + 3 not in rows,
              f'{name} does not reach rows {2*t+2},{2*t+3} (content t = target)')
        check(max(rows) == 2 * t + 1,
              f'{name} last visible clean row is exactly {2*t+1}')
        m_intra, m_cross, _ = masks
        mod = slot_row - clean_len            # 0 = melody, 1 = chord
        same = 2 * t + mod
        other = 2 * t + (1 - mod)
        check(bool(m_intra[slot_row, same]) and bool(m_cross[slot_row, other]),
              f'{name}: own-stream t-1 via intra, other-stream t-1 via cross')
    rows_old, _ = reach(old.global_layers[0], clean_len, t, clean_len)
    rows_new, _ = reach(lay, clean_len, t, clean_len)
    check(rows_new - rows_old == {2 * t, 2 * t + 1},
          'the fix adds exactly rows 2t and 2t+1 and nothing else')

    print('--- 3. clean rows never see slots; slot sees only its partner ---')
    tq_all = tuple(range(1, T_full))
    for net, label in ((old, 'flag off'), (new, 'flag on')):
        layer = net.global_layers[0]
        for tq in (t, tq_all):
            m_intra, m_cross, m_frame = layer._build_masks(
                clean_len, tq, torch.device('cpu'))
            n_pairs = 1 if isinstance(tq, int) else len(tq)
            L = clean_len + 2 * n_pairs
            clean_to_slot = (m_intra[:clean_len, clean_len:]
                             | m_cross[:clean_len, clean_len:]
                             | m_frame[:clean_len, clean_len:])
            check(not clean_to_slot.any(),
                  f'{label}, Q={n_pairs}: no clean row reads any slot')
            slot_slot = (m_intra[clean_len:, clean_len:]
                         | m_cross[clean_len:, clean_len:]
                         | m_frame[clean_len:, clean_len:])
            diag = torch.eye(2 * n_pairs, dtype=torch.bool)
            partner = torch.zeros_like(slot_slot)
            for j in range(n_pairs):
                partner[2 * j, 2 * j + 1] = True
                partner[2 * j + 1, 2 * j] = True
            check(bool(((slot_slot & ~diag) == partner).all()),
                  f'{label}, Q={n_pairs}: each slot reads exactly its partner')

    print('--- 4. Q=all training step ---')
    net = build(slot_sees_prev_frame=True, query_pairs=-1)
    check(net.query_pairs == -1 and net.slot_sees_prev_frame,
          'constructor accepts query_pairs=-1 with the flag')
    net.train()
    B = 2
    x_mel, x_acc = raw_batch(B, T_full), raw_batch(B, T_full)
    ps = torch.zeros(B, dtype=torch.long)
    loss, aux = net.loss(x_mel, x_acc, ps)
    check(torch.isfinite(loss).all(), f'loss finite ({float(loss):.4f})')
    check(net._last_n_pairs == T_full - 1,
          f'n_pairs = T-1 = {net._last_n_pairs}')
    check(net._last_T_query == 1, 'first query frame is 1')
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    check(len(grads) > 0 and all(torch.isfinite(g).all() for g in grads),
          'gradients flow and are finite')
    net.eval()
    with torch.no_grad():
        net.loss(x_mel, x_acc, ps)
    check(net._last_n_pairs == 1 and net._last_T_query == T_full - 1,
          'eval path stays single-frame at T-1')

    print('--- 5. flag round-trips through the checkpoint ---')
    sd = new.state_dict()
    key = next(k for k in sd if k.endswith('slot_sees_prev_frame_flag'))
    check(int(sd[key].item()) == 1, f'{key} == 1 in state_dict')
    fresh = build(slot_sees_prev_frame=True)
    fresh.load_state_dict(sd, strict=False)
    check(all(getattr(l, 'slot_sees_prev_frame', False)
              for l in fresh.global_layers),
          'a model rebuilt with the flag switches every layer')
    sd_old = old.state_dict()
    key_old = next(k for k in sd_old if k.endswith('slot_sees_prev_frame_flag'))
    check(int(sd_old[key_old].item()) == 0,
          'a flag-off model stores 0 (historical ckpts lack the key, '
          'load_model treats absent as 0)')

    print('--- 6. vectorised rotary indices == historical loop ---')
    L = clean_len + 2 * len(tq_all)
    positions = torch.arange(L)
    tq_t = torch.as_tensor(tq_all)
    positions[clean_len::2] = 2 * tq_t + 2
    positions[clean_len + 1::2] = 2 * tq_t + 3
    ref = torch.arange(L)
    for j, t_j in enumerate(tq_all):
        ref[clean_len + 2 * j] = 2 * t_j + 2
        ref[clean_len + 2 * j + 1] = 2 * t_j + 3
    check(bool((positions == ref).all()), 'identical for Q=all')

    print()
    if fails:
        print(f'{len(fails)} CHECK(S) FAILED:')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
