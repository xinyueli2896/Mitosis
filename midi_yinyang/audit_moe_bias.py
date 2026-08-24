"""A.2.moe_improved audit: does the per-modality router bias do exactly
what it claims, and nothing else?

Claims checked, from unit to full model:

  1. WARM-START EQUIVALENCE. A bias-on module/model with the bias at its
     zero init produces bit-identical outputs to the bias-off build with
     the same weights. (If this fails, enabling the knob silently changes
     the model before training has learned anything.)
  2. CONTENT PATHWAY ISOLATION. _last_content_probs is the softmax of
     the UNBIASED logits: identical whether the bias is zero or huge.
     The probes' success metric depends on this cache being clean.
  3. BIAS STEERS ROUTING. With a saturating bias (+/-50) the full
     routing's top-1 follows the modality id at every position.
  4. LAYOUT. The modality ids the layer feeds the FFN match
     _build_masks' own modality derivation, including the two appended
     query slots (clean_len -> mod_a, clean_len+1 -> mod_b).
  5. LOUD MISUSE. A bias-on FFN called without modality_ids raises;
     bias-on vs bias-off state dicts differ exactly by the
     ffn.modality_bias keys (so mismatched resumes fail loudly).

Runs on CPU in seconds. Usage (via audit_moe_bias.sbatch):
    python audit_moe_bias.py
"""

import sys

import torch

from cp_transformer_m2c_jointattn import SimpleMoEFFN
from cp_transformer_m2c_duet_block import M2CDuetBlockLayer
from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []


def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


H, E, TOPK, INTER = 32, 4, 2, 64
B, T = 2, 8            # frames per modality; layer sequence L = 2T + 2

torch.manual_seed(0)

# ---------------------------------------------------------------- level 1
print('\n=== level 1: SimpleMoEFFN unit ===')
ffn_off = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK)
ffn_on = SimpleMoEFFN(H, INTER, num_experts=E, topk=TOPK,
                      modality_bias=True)
missing, unexpected = ffn_on.load_state_dict(ffn_off.state_dict(),
                                             strict=False)
check(missing == ['modality_bias'] and not unexpected,
      'state dicts differ exactly by the modality_bias key')

x = torch.randn(B, 2 * T + 2, H)
ids = torch.arange(2 * T + 2) % 2
with torch.no_grad():
    out_off, aux_off = ffn_off(x)
    out_on, aux_on = ffn_on(x, modality_ids=ids)
check(torch.equal(out_off, out_on) and torch.equal(aux_off, aux_on),
      'zero bias == bias-off, bit-identical (warm-start equivalence)')
check(torch.equal(ffn_on._last_content_probs, ffn_on._last_routing_probs),
      'zero bias: content probs == full probs')
content_at_zero = ffn_on._last_content_probs.clone()

with torch.no_grad():
    # saturating bias: mod_a -> expert 0, mod_b -> expert 1
    ffn_on.modality_bias.zero_()
    ffn_on.modality_bias[0, 0] = 50.0
    ffn_on.modality_bias[1, 1] = 50.0
    ffn_on(x, modality_ids=ids)
top1 = ffn_on._last_routing_probs.argmax(-1)              # [B, L]
want = ids.unsqueeze(0).expand(B, -1)
check(bool((top1 == want).all()),
      'saturating bias steers every position\'s top-1 to its modality')
check(torch.equal(ffn_on._last_content_probs, content_at_zero),
      'content probs unaffected by the bias (probe pathway is clean)')

try:
    ffn_on(x)
    check(False, 'bias-on forward without modality_ids raises')
except ValueError:
    check(True, 'bias-on forward without modality_ids raises')

# ---------------------------------------------------------------- level 2
print('\n=== level 2: M2CDuetBlockLayer layout ===')
mk = dict(hidden_size=H, num_heads=4, intermediate_size=INTER,
          moe_num_experts=E, moe_topk=TOPK, moe_intermediate_size=INTER)
layer_off = M2CDuetBlockLayer(**mk)
layer_on = M2CDuetBlockLayer(**mk, moe_modality_bias=True)
missing, unexpected = layer_on.load_state_dict(layer_off.state_dict(),
                                               strict=False)
check(missing == ['ffn.modality_bias'] and not unexpected,
      'layer state dicts differ exactly by ffn.modality_bias')

clean_len, L = 2 * T, 2 * T + 2
T_query = T - 2
h = torch.randn(B, L, H)
head_dim = H // 4
from cp_transformer_m2c_jointattn import _rope_freqs
cos, sin = _rope_freqs(L, head_dim, device=h.device, dtype=h.dtype)
with torch.no_grad():
    h_off, aux_off = layer_off(h, T_query, cos, sin, clean_len)
    h_on, aux_on = layer_on(h, T_query, cos, sin, clean_len)
check(torch.equal(h_off, h_on) and torch.equal(aux_off, aux_on),
      'layer forward: zero bias == bias-off, bit-identical')

# the ids the layer built must match _build_masks' modality derivation
layer_on._build_masks(clean_len, T_query, h.device)
pos = torch.arange(L)
is_clean = pos < clean_len
mask_modality = torch.where(is_clean, pos % 2, (pos - clean_len) % 2)
with torch.no_grad():
    layer_on.ffn.modality_bias.zero_()
    layer_on.ffn.modality_bias[0, 0] = 50.0
    layer_on.ffn.modality_bias[1, 1] = 50.0
    layer_on(h, T_query, cos, sin, clean_len)
top1 = layer_on.ffn._last_routing_probs.argmax(-1)
check(bool((top1 == mask_modality.unsqueeze(0)).all()),
      'layer feeds the FFN the same modality layout _build_masks uses '
      '(query slots included)')

# ---------------------------------------------------------------- level 3
print('\n=== level 3: M2CDuetBlockDiffusion end to end ===')
mkm = dict(large=False, with_velocity=False, moe_num_experts=E,
           moe_topk=TOPK, global_num_layers=2, diffusion_K=2)
net_off = M2CDuetBlockDiffusion(**mkm)
net_on = M2CDuetBlockDiffusion(**mkm, moe_modality_bias=True)
missing, unexpected = net_on.load_state_dict(net_off.state_dict(),
                                             strict=False)
check(sorted(missing) == sorted(
          f'global_layers.{i}.ffn.modality_bias' for i in range(2))
      and not unexpected,
      'model state dicts differ exactly by the per-layer bias keys')

S = 6
x_tok = torch.randint(0, 200, (B, 2 * T, S))
net_off.eval(), net_on.eval()
torch.manual_seed(1)
with torch.no_grad():
    ar_off, q_off, _ = net_off.forward(x_tok, T_query=T_query)
torch.manual_seed(1)   # forward draws Bernoulli masks for the slots
with torch.no_grad():
    ar_on, q_on, _ = net_on.forward(x_tok, T_query=T_query)
check(torch.equal(ar_off, ar_on) and torch.equal(q_off, q_on),
      'full forward: zero bias == bias-off, bit-identical logits')

with torch.no_grad():
    for lyr in net_on.global_layers:
        lyr.ffn.modality_bias.zero_()
        lyr.ffn.modality_bias[0, 2] = 50.0
        lyr.ffn.modality_bias[1, 3] = 50.0
    net_on.forward(x_tok, T_query=T_query)
ok = True
for lyr in net_on.global_layers:
    top1 = lyr.ffn._last_routing_probs.argmax(-1)
    Lm = top1.shape[1]
    p = torch.arange(Lm)
    want = torch.where(p < Lm - 2, p % 2, (p - (Lm - 2)) % 2)
    ok = ok and bool((top1 == (want + 2).unsqueeze(0)).all())
check(ok, 'end to end: saturating bias steers routing in every layer')

print()
if fails:
    print(f'{len(fails)} FAILURE(S):')
    for f in fails:
        print(f'  - {f}')
    sys.exit(1)
print('all checks passed')
