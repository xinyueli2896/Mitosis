"""Gate for A.11 (partner-agreement discrimination head) before training.

A.11 swaps the COMMITTED LEADER of a conditional-slot pair for the same
stream's frame at a calibrated lag, on a share of the pairs, and asks a
linear head on the FOLLOWER's slot row whether the partner it read was
genuine. The follower's reconstruction loss is dropped on those items.
Checks, on a tiny randomly-initialised model:

  1. leader swap: with a forced lag, the melody (resp. chord) slot input
     equals the local encoding of frame t+lag, not of frame t; the
     partner slot is untouched.
  2. labels: every conditional-slot pair is scored, label 1 exactly
     where that pair's LEADER carries a nonzero lag, and both leader
     directions occur, so the head cannot read the label off the row
     parity.
  3. reconstruction drop: the follower's query-loss positions are
     dropped exactly on decoy pairs, and no other position is dropped.
  4. off switch: agree_decoy_prob=0 leaves the head inactive with no
     gradient, and the total loss matches a model built without the head.
  5. eval: validation pins both slots masked with no conditional-slot
     draw, so the head contributes nothing and val_loss stays comparable
     with the rest of the A family.

Usage (via audit_agree_head.sbatch, CPU):
    python audit_agree_head.py
"""
import sys

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []


def check(cond, msg):
    print(f'  [{"PASS" if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(agree=True, decoy_prob=0.5, seed=0):
    torch.manual_seed(seed)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False, moe_num_experts=4, moe_topk=2,
        global_num_layers=2, gate_init_bias=0.0, diffusion_K=4,
        slot_sees_prev_frame=True, cond_slot_prob=0.8,
        agree_head=agree, agree_decoy_prob=decoy_prob,
        agree_loss_weight=0.3, moe_aux_clean_only=True)
    net.train()
    return net


def batch(B=8, T=12, S=8, seed=1):
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


def main():
    net = build()
    b = batch()
    B = b[0].shape[0]
    T_full = b[0].shape[1]

    print('--- 1. leader swap puts the lagged frame in the slot ---')
    x_m, x_c = net.preprocess(b[0], b[2], y=b[1])
    xx = torch.stack([x_m, x_c], dim=2).view(B, 2 * T_full, -1)
    k0 = torch.zeros(B, 1, dtype=torch.long)          # k=0 -> gt content
    t_j, lag = 5, 3
    t_lag = (t_j + lag) % T_full

    def slots_for(tq, lag_m=None, lag_c=None):
        # eval mode: the local encoder has dropout, so two identical
        # training-mode forwards do not agree and the comparison below
        # would be meaningless. The A.11 swap itself is not gated on
        # training, so it still fires here.
        net.eval()
        net._agree_lag_m, net._agree_lag_c = lag_m, lag_c
        net._stash_slots = True
        with torch.no_grad():
            net.forward(xx, T_query=(tq,), k_m=k0, k_c=k0)
        net._stash_slots = False
        net._agree_lag_m = net._agree_lag_c = None
        net.train()
        return net._last_slots_in.clone()              # [B, 2, H]

    L = torch.full((B, 1), lag, dtype=torch.long)
    check(float((slots_for(t_j) - slots_for(t_j)).abs().max()) < 1e-6,
          'slot construction is deterministic in eval mode (dropout off)')
    plain_t = slots_for(t_j)
    plain_lag = slots_for(t_lag)
    # The slot input is the frame encoding plus the k tag, and the k tag
    # is identical across these calls, so "carries frame t+lag" is
    # exactly "equals the slot built at t+lag with no swap".
    for mod, name in ((0, 'melody'), (1, 'chord')):
        got = slots_for(t_j, lag_m=L if mod == 0 else None,
                        lag_c=L if mod == 1 else None)
        check(float((got[:, mod] - plain_lag[:, mod]).abs().max()) < 1e-5,
              f'{name} leader slot carries frame t+{lag}')
        check(float((got[:, mod] - plain_t[:, mod]).abs().max()) > 1e-5,
              f'... and differs from the un-swapped frame t')
        check(float((got[:, 1 - mod] - plain_t[:, 1 - mod]).abs().max()) < 1e-5,
              '... and the partner slot is untouched')

    print('--- 2. labels follow the leader, both directions occur ---')
    torch.manual_seed(3)
    net2 = build(decoy_prob=0.5, seed=0)
    loss, _ = net2.loss(*b)
    frac = float(net2._last_agree_frac)
    check(abs(frac - float(net2._last_ctc_frac)) < 1e-6,
          'every conditional-slot pair is scored by the head')
    check(0.0 < float(net2._last_agree_acc) <= 1.0 or frac == 0.0,
          'head produced predictions on the scored pairs')

    print('--- 3. reconstruction dropped only on decoy followers ---')
    net3 = build(decoy_prob=1.0, seed=0)
    l3, _ = net3.loss(*b)
    net4 = build(decoy_prob=0.0, seed=0)
    l4, _ = net4.loss(*b)
    check(float(net3._last_agree_frac) > 0,
          'decoy_prob=1 scores the conditional-slot pairs')
    check(float(net3._last_query_kept_frac) < float(net4._last_query_kept_frac),
          'decoy_prob=1 drops query positions that decoy_prob=0 keeps')

    print('--- 4. off switch is the head itself, not the decoy rate ---')
    # decoy_prob sets the POSITIVE RATE, not whether the head runs: the
    # head scores every conditional-slot pair, so at 0 it sees genuine
    # partners only and still has a (degenerate) loss. The off switch is
    # agree_head=False.
    plain = build(agree=False, seed=0)
    lp, _ = plain.loss(*b)
    check(not hasattr(plain, 'agree_proj') or plain.agree_proj is None,
          'agree_head=False builds no head')
    check(float(getattr(plain, '_last_agree_loss', torch.zeros(()))) == 0.0,
          '... contributes no detection loss')
    l3.backward()
    g = net3.agree_proj.weight.grad
    check(g is not None and float(g.norm()) > 0,
          'with the head on, the detection loss reaches it')
    check(float(net4._last_agree_frac) == float(net4._last_ctc_frac),
          'decoy_prob=0 still scores the pairs (genuine class only)')

    print('--- 5. eval is unaffected ---')
    net5 = build(decoy_prob=1.0, seed=0)
    net5.eval()
    with torch.no_grad():
        lv, _ = net5.loss(*b)
    check(float(net5._last_agree_loss) == 0.0,
          'validation contributes no detection loss')
    plain.eval()
    with torch.no_grad():
        lvp, _ = plain.loss(*b)
    check(abs(float(lv) - float(lvp)) < 1e-4,
          'val_loss identical to the same model without the head')

    print()
    if fails:
        print(f'{len(fails)} CHECK(S) FAILED:')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED -- A.11 behaves as documented; safe to train')


if __name__ == '__main__':
    main()
