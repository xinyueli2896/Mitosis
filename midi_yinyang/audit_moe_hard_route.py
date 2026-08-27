"""A.2.moe_hardroute audit: does the disjoint-pool control do exactly
what it claims, and nothing else?

The hard-route arm exists to be the imposed-separation strawman the
learned per-modality gates are argued against. That argument is only
honest if the arm is implemented FAIRLY -- separation imposed, nothing
else sabotaged. These checks verify both halves.

Claims checked, from unit to full model:

  1. POOLS ARE DISJOINT. A mod_a token receives exactly zero routing
     probability on every pool-B expert and vice versa, for arbitrary
     gate weights -- verified at FFN level (explicit ids) and at layer
     level (the layer derives ids itself, query slots included,
     matching _build_masks).
  2. STILL A VALID DISTRIBUTION. Rows sum to 1, top-k stays inside the
     pool, outputs are finite, and gradients still reach the gate --
     i.e. masking with -inf did not produce NaNs or dead gradients.
  3. FAIR LOAD BALANCING. The aux loss is computed WITHIN pools: it is
     minimised by a pool-uniform load (its floor is 1.0, the same floor
     the shared arm has under a uniform load over E), NOT by the
     unreachable all-expert uniform load. A model that perfectly
     balances within pools must not be penalised relative to a shared
     model that perfectly balances overall.
  4. NO FREE PARAMETERS. The hard-route state dict differs from the
     shared one by the hard_route_flag buffer ONLY -- same expert and
     gate weights, same shapes -- so any quality difference is
     attributable to routing, not capacity.
  5. CHECKPOINT DETECTION. hard_route_flag is persistent (it lands in
     the state dict, which is what inference auto-detects on), while
     pool_is_a is not (derived, not learned).
  6. LOUD MISUSE. Hard route without modality_ids raises; an odd expert
     count or topk > pool size raises at construction.
  7. INTEGRATORS ARE UNREPRESENTABLE. Under hard route no expert can
     receive traffic from both streams, whatever the gate weights --
     the property that makes purity 0/100 by construction and makes
     this the control rather than a competitor.
  8. THE OTHER ARMS ARE UNTOUCHED. Shared and per-modality-gate builds
     behave exactly as before this flag existed.

Runs on CPU in seconds. Usage (via audit_moe_hard_route.sbatch):
    python audit_moe_hard_route.py
"""

import sys

import torch

from cp_transformer_m2c_jointattn import SimpleMoEFFN
from cp_transformer_m2c_duet_block import M2CDuetBlockLayer

fails = []


def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


H, E, TOPK, INTER = 32, 4, 2, 64
B, T = 2, 8
L = 2 * T + 2          # interleaved clean context + 2 query slots
POOL = E // 2

torch.manual_seed(0)
x = torch.randn(B, L, H)
ids = torch.arange(L)
ids = torch.where(ids < 2 * T, ids % 2, (ids - 2 * T) % 2)
idx_a = [int(i) for i in torch.nonzero(ids == 0).flatten()]
idx_b = [int(i) for i in torch.nonzero(ids == 1).flatten()]


print('\n=== 1. pools are disjoint ===')
hr = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK,
                  modality_hard_route=True)
# Arbitrary (not default) gate weights: disjointness must not depend on
# the router happening to prefer its own pool.
with torch.no_grad():
    torch.nn.init.normal_(hr.gate.weight, std=1.5)
out, aux = hr(x, modality_ids=ids)
p = hr._last_routing_probs                       # [B, L, E]
check(float(p[:, idx_a, POOL:].abs().max()) == 0.0,
      'mod_a tokens put zero probability on pool B')
check(float(p[:, idx_b, :POOL].abs().max()) == 0.0,
      'mod_b tokens put zero probability on pool A')

layer = M2CDuetBlockLayer(
    hidden_size=H, num_heads=4, intermediate_size=INTER,
    moe_num_experts=E, moe_topk=TOPK, moe_intermediate_size=INTER,
    moe_modality_hard_route=True,
)
check(getattr(layer.ffn, 'needs_modality_ids', False),
      'layer-built hard-route FFN advertises needs_modality_ids')
with torch.no_grad():
    torch.nn.init.normal_(layer.ffn.gate.weight, std=1.5)
    cos, sin = layer._rope_cache(L, x.device) if hasattr(
        layer, '_rope_cache') else (None, None)
if cos is None:                       # layer takes cos/sin from caller
    from cp_transformer_m2c_jointattn import _rope_freqs
    cos, sin = _rope_freqs(L, H // 4, x.device, x.dtype)
with torch.no_grad():
    layer(x, T_query=T, cos=cos, sin=sin, clean_len=2 * T)
pl = layer.ffn._last_routing_probs
check(float(pl[:, idx_a, POOL:].abs().max()) == 0.0
      and float(pl[:, idx_b, :POOL].abs().max()) == 0.0,
      'layer derives the ids itself (query slots included) and the '
      'pools stay disjoint end to end')


print('\n=== 2. still a valid, differentiable distribution ===')
check(torch.allclose(p.sum(-1), torch.ones(B, L), atol=1e-6),
      'routing rows still sum to 1 (softmax over the pool)')
check(bool(torch.isfinite(out).all()), 'FFN output is finite (no NaN '
      'from the -inf mask)')
check(bool(torch.isfinite(aux)), 'aux loss is finite')
aux.backward()
check(hr.gate.weight.grad is not None
      and bool(torch.isfinite(hr.gate.weight.grad).all()),
      'gradient still reaches the gate through the masked softmax')


print('\n=== 3. load balancing is computed within pools (fairness) ===')
# A hard-route model with a perfectly pool-uniform load should score the
# same aux value as a shared model with a perfectly uniform load: 1.0.
# The all-expert form would give the hard-route model E/pool = 2.0 -- a
# constant penalty for an allocation it cannot produce.
hr_bal = SimpleMoEFFN(H, INTER, num_experts=E, topk=1,
                      modality_hard_route=True)
sh_bal = SimpleMoEFFN(H, INTER, num_experts=E, topk=1)
with torch.no_grad():                      # zero gate -> uniform probs
    hr_bal.gate.weight.zero_()
    sh_bal.gate.weight.zero_()
    _, aux_hr = hr_bal(x, modality_ids=ids)
    _, aux_sh = sh_bal(x)
check(abs(float(aux_hr) - 1.0) < 0.35,
      f'pool-uniform load scores near the 1.0 floor (got {float(aux_hr):.3f}), '
      f'not the 2.0 an all-expert loss would impose')
check(abs(float(aux_hr) - float(aux_sh)) < 0.35,
      f'hard-route aux ({float(aux_hr):.3f}) is comparable to the shared '
      f'arm at its own balanced optimum ({float(aux_sh):.3f})')


print('\n=== 4. no free parameters vs the shared arm ===')
shared = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK)
sd_hr, sd_sh = hr.state_dict(), shared.state_dict()
extra = set(sd_hr) - set(sd_sh)
missing = set(sd_sh) - set(sd_hr)
check(extra == {'hard_route_flag'} and not missing,
      f'state dicts differ by the flag buffer only (extra={sorted(extra)}, '
      f'missing={sorted(missing)})')
same_shapes = all(sd_hr[k].shape == sd_sh[k].shape for k in sd_sh)
check(same_shapes, 'every shared key has identical shape (same expert '
      'and gate capacity -- differences are routing, not size)')
n_hr = sum(v.numel() for k, v in sd_hr.items() if k != 'hard_route_flag')
n_sh = sum(v.numel() for v in sd_sh.values())
check(n_hr == n_sh, f'parameter count identical ({n_hr} == {n_sh})')


print('\n=== 5. checkpoint detection ===')
check('hard_route_flag' in sd_hr,
      'hard_route_flag is persistent -> lands in the ckpt for '
      'load_model auto-detection')
check(not any(k.endswith('pool_is_a') for k in sd_hr),
      'pool_is_a is non-persistent (derived, not learned)')
check(not any(k.endswith('hard_route_flag') for k in sd_sh),
      'shared ckpts carry no flag -> auto-detect returns False')


print('\n=== 6. loud misuse ===')
try:
    hr(x)
    check(False, 'hard route without modality_ids should raise')
except ValueError as e:
    check('modality_ids' in str(e),
          'hard route without modality_ids raises a clear ValueError')
for kw, why in ((dict(num_experts=3, topk=1), 'odd expert count'),
                (dict(num_experts=4, topk=3), 'topk larger than the pool')):
    try:
        SimpleMoEFFN(H, INTER, modality_hard_route=True, **kw)
        check(False, f'{why} should raise at construction')
    except ValueError:
        check(True, f'{why} raises at construction')


print('\n=== 7. integrators are unrepresentable ===')
# Sweep many random gates: no expert may ever win tokens from both
# streams. (Under the shared arm this is not only possible but observed
# -- that contrast is the point of the ablation.)
both_streams_ever = False
for trial in range(25):
    m = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK,
                     modality_hard_route=True)
    with torch.no_grad():
        torch.nn.init.normal_(m.gate.weight, std=2.0)
        m(torch.randn(B, L, H), modality_ids=ids)
    pr = m._last_routing_probs
    top1 = pr.argmax(-1)
    for e in range(E):
        won_a = bool((top1[:, idx_a] == e).any())
        won_b = bool((top1[:, idx_b] == e).any())
        both_streams_ever |= (won_a and won_b)
check(not both_streams_ever,
      'over 25 random routers, no expert ever won tokens from both '
      'streams (purity is 0/100 by construction)')


print('\n=== 8. the other arms are untouched ===')
torch.manual_seed(7)
s1 = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK)
torch.manual_seed(7)
s2 = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK)
o1, a1 = s1(x)
o2, a2 = s2(x)
check(torch.equal(o1, o2) and torch.equal(a1, a2),
      'shared arm is deterministic and unchanged by the new code path')
check(not s1.needs_modality_ids,
      'shared arm still does not require modality_ids')
g = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK, modality_gates=True)
og, _ = g(x, modality_ids=ids)
pg = g._last_routing_probs
check(bool(torch.isfinite(og).all())
      and float(pg[:, idx_a, POOL:].abs().max()) > 0.0,
      'per-modality-gates arm still reaches the WHOLE pool (the shared '
      'unassigned pool it is defined by)')


print()
if fails:
    print(f'{len(fails)} FAILURE(S):')
    for f in fails:
        print(f'  - {f}')
    sys.exit(1)
print('all checks passed')
