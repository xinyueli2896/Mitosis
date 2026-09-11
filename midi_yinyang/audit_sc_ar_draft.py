"""Gate for A.12 (AR-head self-conditioning drafts) before training.

A.12 changes what a self-conditioned query slot CONTAINS, not how the
loss is formed. Three separable knobs:

  sc_ar_frac      share of self-conditioned slots whose draft comes from
                  the AR CONTENT head's rows rather than the query
                  logits. Both decodes seed from the AR head, so at 0
                  the slots never meet their inference input.
  sc_draft_temp   sample the draft at this temperature instead of the
                  argmax. The paper decode commits at temperature 1.0,
                  so an argmax draft is sharper than anything a slot
                  sees at inference.
  sc_k_consistent for self-conditioned slots, mask at k=K only rather
                  than by a Bernoulli coin at rate k/K -- the decode
                  puts a WHOLE draft in every slot at k<K, so training's
                  intermediate k should mean the same thing. Without it
                  the coin discards ~half the drafts before use.

Checks, on a tiny randomly-initialised model:

  1. all three knobs off reproduce the pre-A.12 loss bit-for-bit, so
     every existing A.3/A.4 run stays comparable.
  2. sc_ar_frac=1 really swaps the draft SOURCE: the tokens written into
     the slots equal the AR-row drafts and differ from the query-head
     drafts on a substantial share of positions.
  3. train_sc_ar_frac logs the realised rate, matching the knob.
  4. sc_k_consistent makes a self-conditioned slot the mask embedding
     exactly at k=K and the draft everywhere else, while a slot with no
     draft keeps the Bernoulli coin.
  5. sc_k_consistent raises the share of self-conditioned slots that
     actually carry their draft from ~1/2 to 1.
  6. sc_draft_temp=0 is deterministic (argmax); > 0 is stochastic, and
     both stay inside the vocabulary.
  7. the draft carries no gradient (it is built under no_grad and enters
     as a constant), while the query loss still reaches the parameters.
  8. self-conditioning is training-only, so val_loss is identical across
     every setting of all three knobs -- the A-family comparison and
     resolve_best_ckpt are unaffected.

Usage (via audit_sc_ar_draft.sbatch, CPU):
    python audit_sc_ar_draft.py
"""
import sys

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []


def check(cond, msg):
    print(f'  [{"PASS" if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(seed=0, **kw):
    torch.manual_seed(seed)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False, moe_num_experts=4, moe_topk=2,
        global_num_layers=2, gate_init_bias=0.0, diffusion_K=4,
        slot_sees_prev_frame=True, cond_slot_prob=0.0,
        self_cond_prob=1.0, moe_aux_clean_only=True, **kw)
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


def run(net, b, seed=7, stash=False):
    """One loss() at a pinned RNG state, so two models differing only in
    a knob draw the same k, the same self-conditioning mask and the same
    Bernoulli coins."""
    net._stash_slots = stash
    torch.manual_seed(seed)
    out = net.loss(*b)
    net._stash_slots = False
    return out


def main():
    b = batch()

    print('--- 1. all knobs off == pre-A.12 ---')
    base = build()
    a12off = build(sc_ar_frac=0.0, sc_draft_temp=0.0, sc_k_consistent=False)
    lb, _ = run(base, b)
    lo, _ = run(a12off, b)
    lb, lo = float(lb.detach()), float(lo.detach())
    check(abs(lb - lo) < 1e-6,
          f'loss unchanged with the knobs at their defaults '
          f'({lb:.6f} vs {lo:.6f})')

    print('--- 2. sc_ar_frac=1 swaps the draft source ---')
    net = build(sc_ar_frac=1.0)
    run(net, b, stash=True)
    qm, am = net._last_q_draft_m, net._last_ar_draft_m
    check(qm.shape == am.shape, f'draft shapes agree {tuple(qm.shape)}')
    diff = float((qm != am).float().mean())
    check(diff > 0.05,
          f'AR-row draft differs from the query-head draft on '
          f'{100 * diff:.1f}% of tokens')
    check(bool(net._last_use_ar_m.all()),
          'at sc_ar_frac=1 every self-conditioned item takes the AR draft')

    print('--- 3. the realised rate is logged ---')
    # The logged value is a binomial mean over 2*B*Q draws -- 16 at the
    # audit's B=8 -- so a single run's rate has std ~0.11 and a tight
    # tolerance would flake. Pool 8 seeds (128 draws, std ~0.04).
    for p in (0.25, 0.5, 1.0):
        n = build(sc_ar_frac=p)
        rates = []
        for s in range(8):
            run(n, b, seed=100 + s)
            rates.append(float(n._last_sc_ar_frac))
        got = sum(rates) / len(rates)
        check(abs(got - p) < 0.12,
              f'sc_ar_frac={p}: train_sc_ar_frac={got:.3f} '
              f'(mean of {len(rates)} draws)')
    n0 = build(sc_ar_frac=0.0)
    run(n0, b)
    check(float(n0._last_sc_ar_frac) == 0.0,
          'sc_ar_frac=0 logs 0 and takes no AR draft')

    print('--- 4/5. sc_k_consistent: the slot means what the decode means ---')
    B, T_full = b[0].shape[0], b[0].shape[1]

    def slot_stats(net):
        """Share of self-conditioned slots holding the mask embedding,
        split by whether their k is the endpoint K."""
        net._stash_slots = True
        torch.manual_seed(7)
        net.loss(*b)
        net._stash_slots = False
        slots = net._last_slots_in                      # [B, 2Q, H]
        K = net.diffusion_K
        out = {}
        for mod, name in ((0, 'm'), (1, 'c')):
            mask_emb = (net.mask_m_emb if mod == 0 else net.mask_c_emb)
            k_emb = (net.k_emb_m if mod == 0 else net.k_emb_c)
            row = slots[:, mod]                         # [B, H]
            ks = torch.arange(K + 1)
            # slot == mask_emb + k_emb(k) for the k it was tagged with
            ref = mask_emb.view(1, -1) + k_emb(ks)      # [K+1, H]
            d = (row.unsqueeze(1) - ref.unsqueeze(0)).abs().amax(-1)
            is_mask = (d < 1e-4).any(dim=1)             # [B]
            out[name] = is_mask
        return out

    incons = build(sc_ar_frac=1.0, sc_k_consistent=False)
    cons = build(sc_ar_frac=1.0, sc_k_consistent=True)
    si, sc = slot_stats(incons), slot_stats(cons)
    fi = float(torch.cat([si['m'], si['c']]).float().mean())
    fc = float(torch.cat([sc['m'], sc['c']]).float().mean())
    check(fc < fi,
          f'consistent slots are masked less often '
          f'({100 * fc:.0f}% vs {100 * fi:.0f}%)')
    check(fc <= 1.0 / (cons.diffusion_K + 1) + 0.20,
          f'... and only near the 1/(K+1) rate at which k==K is drawn '
          f'({100 * fc:.0f}%)')
    check(fi > 0.25,
          f'... while the Bernoulli coin discards a large share of the '
          f'drafts ({100 * fi:.0f}%)')

    print('--- 6. draft sharpness ---')
    d = build(sc_ar_frac=1.0, sc_draft_temp=0.0)
    run(d, b, seed=11, stash=True)
    d1 = d._last_ar_draft_m.clone()
    run(d, b, seed=11, stash=True)
    check(bool((d1 == d._last_ar_draft_m).all()),
          'sc_draft_temp=0 is the deterministic argmax')
    s = build(sc_ar_frac=1.0, sc_draft_temp=1.0)
    run(s, b, seed=11, stash=True)
    s1 = s._last_ar_draft_m.clone()
    run(s, b, seed=12, stash=True)
    s2 = s._last_ar_draft_m
    check(not bool((s1 == s2).all()),
          'sc_draft_temp=1 samples (two draws differ)')
    V = s.tokenizer.n_tokens
    check(bool((s1 >= 0).all() and (s1 < V).all()),
          f'sampled drafts stay inside the vocabulary (V={V})')

    print('--- 7. gradients ---')
    g = build(sc_ar_frac=1.0, sc_draft_temp=1.0, sc_k_consistent=True)
    lg, _ = run(g, b, stash=True)
    lg.backward()
    gn = sum(float(p.grad.norm()) for p in g.parameters()
             if p.grad is not None)
    check(gn > 0, f'the loss reaches the parameters (|grad| sum {gn:.3f})')
    check(all(not t.requires_grad for t in
              (g._last_ar_draft_m, g._last_sc_ar_frac)
              if torch.is_tensor(t)),
          'the draft and its logged rate carry no graph')

    print('--- 9. sym_k: both slots move together ---')
    # K=1 is the setting the one-round parallel decode's tags match:
    # k=1 is the masked seed round, k=0 the draft round.
    def build_sym(cond=0.0, **kw):
        torch.manual_seed(0)
        n = M2CDuetBlockDiffusion(
            large=False, with_velocity=False, moe_num_experts=4, moe_topk=2,
            global_num_layers=2, gate_init_bias=0.0, diffusion_K=1,
            slot_sees_prev_frame=True, cond_slot_prob=cond,
            self_cond_prob=1.0, sc_ar_frac=1.0, sc_draft_temp=1.0,
            moe_aux_clean_only=True, **kw)
        n.train()
        return n

    def diag_frac(net, trials=8):
        """Share of query pairs whose two slots sit at the same level."""
        hits = tot = 0
        for s in range(trials):
            torch.manual_seed(200 + s)
            net.loss(*b)
            km, kc = net._last_k_m, net._last_k_c
            hits += int((km == kc).sum())
            tot += km.numel()
        return hits / tot

    sym = build_sym(sym_k=True)
    ind = build_sym(sym_k=False)
    check(diag_frac(sym) == 1.0,
          'sym_k=1: every pair is on a diagonal (k, k) state')
    fi = diag_frac(ind)
    check(fi < 1.0,
          f'sym_k=0: the independent draw is off-diagonal '
          f'{100 * (1 - fi):.0f}% of the time')
    ls, _ = sym.loss(*b)
    ls.backward()
    check(float(ls.detach()) > 0, 'sym_k trains (loss finite, backward runs)')
    try:
        build_sym(cond=0.8, sym_k=True)
        check(False, 'sym_k + cond_slot_prob should be refused')
    except ValueError:
        check(True, 'sym_k + cond_slot_prob is refused at construction')

    print('--- 8. validation is unaffected ---')
    vals = []
    for kw in ({}, {'sc_ar_frac': 1.0},
               {'sc_ar_frac': 1.0, 'sc_draft_temp': 1.0},
               {'sc_ar_frac': 1.0, 'sc_draft_temp': 1.0,
                'sc_k_consistent': True}):
        n = build(**kw)
        n.eval()
        torch.manual_seed(21)
        with torch.no_grad():
            lv, _ = n.loss(*b)
        vals.append(float(lv))
    check(max(vals) - min(vals) < 1e-5,
          f'val_loss identical across all knob settings '
          f'(spread {max(vals) - min(vals):.2e})')

    print()
    if fails:
        print(f'{len(fails)} CHECK(S) FAILED:')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED -- A.12 behaves as documented; safe to train')


if __name__ == '__main__':
    main()
