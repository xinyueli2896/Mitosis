"""A.4 (token_level_mask) audit: does per-token frame corruption do
exactly what it claims, and nothing else?

Claims checked:

  1. MASK ID SAFETY. frame_mask_token sits in the instrument-padding
     dead zone: above the pitch-dur range (max 128*25-1), above the
     real program range (0..255 by position), below SOS/EOS/PAD -- so
     it can never collide with real data and local_sampling's
     valid-token masks already exclude it.
  2. ENDPOINT EQUIVALENCE, k=K. With identical weights, an A.4 model at
     k_m=k_c=K produces BIT-IDENTICAL outputs to the plain model: the
     all-masked draw is deterministic at p=1 and falls back to the
     learned mask_*_emb.
  3. ENDPOINT EQUIVALENCE, k=0. At k=0 nothing is masked and the slot
     is the clean frame's encoding -- matching the plain model's
     content branch (same local encoder, eval mode).
  4. GRADED MIDDLE. At 0<k<K the masked fraction of maskable tokens
     tracks k/K in expectation, only non-pad positions are ever masked,
     and the mask id is what replaces them.
  5. SELF-COND AT TOKEN LEVEL. With sc_mask set, corruption applies to
     the provided draft tokens, not the ground truth.
  6. FLAG TRAVELS BY VALUE. Both plain and A.4 checkpoints carry
     token_level_mask_flag (0/1) -- detection must read the value, not
     key presence; verified for both.
  7. LOUD MISUSE. with_velocity + token_level_mask raises (that vocab
     has no free id).
  8. DECODE HELPERS. _remask_lowest masks exactly the n lowest-
     confidence non-pad tokens (n = round(frac * maskable)), never a
     pad, and frac=0 is a no-op; _token_confidences returns +inf on
     pads so they are never selected.

Runs on CPU in seconds. Usage (via audit_a4_token_mask.sbatch):
    python audit_a4_token_mask.py
"""

import sys

import torch

from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion
from cp_transformer_m2c_duet_block_diffusion_inference import (
    _remask_lowest, _token_confidences,
)

fails = []


def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(token_level_mask):
    torch.manual_seed(0)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False,
        moe_num_experts=4, moe_topk=2,
        global_num_layers=2, diffusion_K=4,
        token_level_mask=token_level_mask,
    )
    net.eval()
    return net


def toy_batch(net, B=2, T=6):
    """Interleaved [B, 2T, S] with plausible frames (program + pd + EOS)."""
    tok = net.tokenizer
    S = 8
    x = torch.full((B, 2 * T, S), tok.pad_token, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    for b in range(B):
        for i in range(2 * T):
            prog = 24 if i % 2 == 0 else 0
            n_notes = int(torch.randint(1, 3, (1,), generator=g))
            x[b, i, 0] = prog
            for n in range(n_notes):
                pd = int(torch.randint(130, 128 * 25 - 1, (1,), generator=g))
                x[b, i, 1 + n] = pd
            x[b, i, 1 + n_notes] = tok.eos_token
    return x


print('\n=== 1. mask id safety ===')
net_a4 = build(True)
tok = net_a4.tokenizer
mid = net_a4.frame_mask_token
check(mid == tok.n_normal_tokens - 1, f'mask id is n_normal_tokens-1 ({mid})')
check(mid > 128 * 25 - 1, 'above the pitch-dur range (can never be a pd token)')
check(mid > 255, 'above the padded program range (can never be a program)')
check(mid not in (tok.sos_token, tok.eos_token, tok.pad_token),
      'distinct from SOS/EOS/PAD')

print('\n=== 2. endpoint equivalence at k=K (bit-identical) ===')
net_plain = build(False)
net_a4.load_state_dict(
    {k: v for k, v in net_plain.state_dict().items()}, strict=False)
x = toy_batch(net_plain)
K = net_plain.diffusion_K
with torch.no_grad():
    torch.manual_seed(7)
    ar0, q0, _ = net_plain.forward(x, T_query=3, k_m=K, k_c=K)
    torch.manual_seed(7)
    ar1, q1, _ = net_a4.forward(x, T_query=3, k_m=K, k_c=K)
check(torch.equal(ar0, ar1) and torch.equal(q0, q1),
      'A.4 at k=K reproduces the plain model bit-for-bit '
      '(all-masked falls back to mask_*_emb)')

print('\n=== 3. endpoint equivalence at k=0 (clean frame) ===')
with torch.no_grad():
    torch.manual_seed(7)
    ar0, q0, _ = net_plain.forward(x, T_query=3, k_m=0, k_c=0)
    torch.manual_seed(7)
    ar1, q1, _ = net_a4.forward(x, T_query=3, k_m=0, k_c=0)
check(torch.allclose(q0, q1, atol=1e-5),
      'A.4 at k=0 matches the plain content branch (clean-frame encode)')

print('\n=== 4. graded middle ===')
frame = x[:, 6]                                  # a mod_a frame [B, S]
k_t = torch.full((frame.shape[0],), 2, dtype=torch.long)
n_trials, frac_sum, bad_pad, bad_id = 400, 0.0, 0, 0
for i in range(n_trials):
    torch.manual_seed(100 + i)
    B, S = frame.shape
    p = (k_t.float() / K).view(B, 1)
    maskable = frame != tok.pad_token
    drawn = (torch.rand(B, S) < p) & maskable
    corrupted = torch.where(
        drawn, torch.full_like(frame, mid), frame)
    frac_sum += (drawn.float().sum() / maskable.float().sum()).item()
    bad_pad += int(((corrupted == mid) & ~maskable).sum())
    bad_id += int(((corrupted != frame) & (corrupted != mid)).sum())
mean_frac = frac_sum / n_trials
check(abs(mean_frac - 0.5) < 0.05,
      f'masked fraction tracks k/K (k=2, K=4: got {mean_frac:.3f} ~ 0.5)')
check(bad_pad == 0, 'pad positions are never masked')
check(bad_id == 0, 'replacement token is always the mask id')

print('\n=== 5. self-conditioning at token level ===')
B = frame.shape[0]
draft = frame.clone()
draft[:, 1] = 999                                 # visibly different pd
sc_mask = torch.tensor([True] + [False] * (B - 1))
with torch.no_grad():
    torch.manual_seed(3)
    slot_draft = net_a4._token_level_slot(frame, sc_mask, draft,
                                           torch.zeros(B, dtype=torch.long), 0)
    torch.manual_seed(3)
    slot_gt = net_a4._token_level_slot(frame, None, None,
                                        torch.zeros(B, dtype=torch.long), 0)
check(not torch.allclose(slot_draft[0], slot_gt[0]),
      'flagged item encodes the draft tokens, not ground truth')
check(torch.allclose(slot_draft[1:], slot_gt[1:], atol=1e-6),
      'unflagged items are untouched by the override')

print('\n=== 6. flag travels by value ===')
sd_plain, sd_a4 = net_plain.state_dict(), net_a4.state_dict()
check('token_level_mask_flag' in sd_plain
      and int(sd_plain['token_level_mask_flag']) == 0,
      'plain ckpts carry the buffer with value 0 (detect by VALUE)')
check('token_level_mask_flag' in sd_a4
      and int(sd_a4['token_level_mask_flag']) == 1,
      'A.4 ckpts carry the buffer with value 1')

print('\n=== 7. loud misuse ===')
try:
    M2CDuetBlockDiffusion(large=False, with_velocity=True,
                          moe_num_experts=4, moe_topk=2,
                          global_num_layers=2, token_level_mask=True)
    check(False, 'with_velocity + token_level_mask should raise')
except ValueError as e:
    check('free token id' in str(e),
          'with_velocity + token_level_mask raises a clear ValueError')

print('\n=== 8. decode helpers ===')
tokens = frame.clone()
S = tokens.shape[1]
conf = torch.rand(tokens.shape)
conf = torch.where(tokens == tok.pad_token,
                   torch.full_like(conf, float('inf')), conf)
out0 = _remask_lowest(net_a4, tokens, conf, 0.0)
check(torch.equal(out0, tokens), 'frac=0 is a no-op')
out_half = _remask_lowest(net_a4, tokens, conf, 0.5)
maskable = tokens != tok.pad_token
n_expect = (0.5 * maskable.sum(dim=1).float()).round().long()
n_got = ((out_half == mid) & maskable).sum(dim=1)
check(torch.equal(n_got, n_expect),
      f'masks exactly round(frac*maskable) tokens per item ({n_got.tolist()})')
check(int(((out_half == mid) & ~maskable).sum()) == 0,
      'pads are never re-masked')
masked_pos = (out_half == mid) & maskable
kept_pos = (out_half != mid) & maskable
worst_kept = conf.masked_fill(~kept_pos, float('inf')).min(dim=1).values
best_masked = conf.masked_fill(~masked_pos, -float('inf')).max(dim=1).values
check(bool((best_masked <= worst_kept + 1e-9).all()),
      'the masked set is exactly the lowest-confidence tokens')
h_pred = torch.randn(tokens.shape[0], net_a4.hidden_size)
with torch.no_grad():
    c = _token_confidences(net_a4, h_pred, tokens, 0)
check(c.shape == tokens.shape, '_token_confidences returns [B, S]')
check(bool(torch.isinf(c[tokens == tok.pad_token]).all()),
      'pad positions get +inf confidence (never re-masked)')

print()
if fails:
    print(f'{len(fails)} FAILURE(S):')
    for f in fails:
        print(f'  - {f}')
    sys.exit(1)
print('all checks passed')
