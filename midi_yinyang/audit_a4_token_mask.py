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
  4b. STRUCTURAL EXEMPTION (FIX 2). Only odd (pitch-dur) positions are
     maskable: programs and the terminating EOS, which live at even
     positions, survive every k. A frame whose EOS was maskable reads as
     unfinished and inflates density across refinement rounds -- the
     first A.4 run's actual failure mode.
  9. MASK ROW INIT (FIX 1). _init_frame_mask_row() sets the mask id's
     embedding row to the mean of the real token rows (0..128*25-1),
     runs only once (flag buffer), is a no-op for non-A.4 models, and
     fires from on_train_start so the warm-start state_dict cannot
     overwrite it.
  10. SILENT FRAMES. A frame with no pitch-dur token has nothing to
     corrupt, so it is masked all-or-nothing with the same probability
     k/K: never at k=0, always at k=K, and it can never reveal its
     silence for free at intermediate k.
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


def make_frame(net, n_notes, prog, g=None):
    """One cp frame: (program, pitch-dur) PAIRS terminated by EOS.

    Even positions carry programs and the EOS, odd positions carry
    pitch-durs -- the layout FIX 2's exemption relies on. n_notes=0 is a
    silent frame (EOS at position 0, no maskable token at all).
    """
    tok = net.tokenizer
    S = 8
    f = torch.full((S,), tok.pad_token, dtype=torch.long)
    for n in range(n_notes):
        f[2 * n] = prog
        f[2 * n + 1] = int(torch.randint(128, 128 * 25, (1,), generator=g)) \
            if g is not None else 128 + n
    f[2 * n_notes] = tok.eos_token
    return f


def toy_batch(net, B=2, T=6):
    """Interleaved [B, 2T, S] of plausible frames, silent ones included."""
    x = torch.stack([
        torch.stack([
            make_frame(net, (b + i) % 4, 24 if i % 2 == 0 else 0,
                       torch.Generator().manual_seed(1000 * b + i))
            for i in range(2 * T)
        ])
        for b in range(B)
    ])
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
# Copy the WEIGHTS across, never the scheme flags: token_level_mask_flag
# is 0 in the plain model and copying it would silently turn net_a4 into
# a plain model for every check below.
SCHEME_FLAGS = ('token_level_mask_flag', 'frame_mask_row_init_flag')
net_a4.load_state_dict(
    {k: v for k, v in net_plain.state_dict().items()
     if k not in SCHEME_FLAGS}, strict=False)
check(net_a4.token_level_mask,
      'net_a4 is still an A.4 model after the weight copy')
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

print('\n=== 4. graded middle (real draw, via _corrupt_frame_tokens) ===')
frame = x[:, 6]                                  # a mod_a frame [B, S]
B, S = frame.shape
is_pd = (torch.arange(S) % 2 == 1).view(1, S)
maskable = is_pd & (frame != tok.pad_token)
k_t = torch.full((B,), 2, dtype=torch.long)
n_trials, frac_sum, bad_pad, bad_id, bad_struct = 400, 0.0, 0, 0, 0
for i in range(n_trials):
    torch.manual_seed(100 + i)
    corrupted, _ = net_a4._corrupt_frame_tokens(frame, k_t)
    drawn = corrupted == mid
    frac_sum += (drawn.float().sum() / maskable.float().sum()).item()
    bad_pad += int((drawn & (frame == tok.pad_token)).sum())
    bad_id += int(((corrupted != frame) & (corrupted != mid)).sum())
    bad_struct += int((drawn & ~is_pd.expand(B, S)).sum())
mean_frac = frac_sum / n_trials
check(abs(mean_frac - 0.5) < 0.05,
      f'masked fraction tracks k/K (k=2, K=4: got {mean_frac:.3f} ~ 0.5)')
check(bad_pad == 0, 'pad positions are never masked')
check(bad_id == 0, 'replacement token is always the mask id')

print('\n=== 4b. structural exemption (FIX 2) ===')
check(bad_struct == 0,
      'even positions (programs + EOS) are never masked over 400 draws')
eos_kept = 0
for i in range(200):
    torch.manual_seed(500 + i)
    for kk in range(K + 1):
        c, _ = net_a4._corrupt_frame_tokens(
            frame, torch.full((B,), kk, dtype=torch.long))
        eos_kept += int(((frame == tok.eos_token) & (c == mid)).sum())
check(eos_kept == 0, 'the terminating EOS survives at EVERY k (0..K)')
c_full, fully_full = net_a4._corrupt_frame_tokens(
    frame, torch.full((B,), K, dtype=torch.long))
check(bool((c_full[maskable] == mid).all()),
      'k=K masks every pitch-dur token')
check(bool(fully_full.all()),
      'k=K flags every item for the whole-frame mask embedding')

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

print('\n=== 9. mask row init (FIX 1) ===')
net_fresh = build(True)
emb = net_fresh.local_embedding.weight
expected = emb[:128 * 25].mean(dim=0).clone()
check(int(net_fresh.frame_mask_row_init_flag) == 0,
      'a fresh A.4 model has NOT yet initialised the row')
check(not torch.allclose(emb[mid], expected, atol=1e-4),
      'the row starts as the random/pretrained vector (the bug being fixed)')
net_fresh._init_frame_mask_row()
check(torch.allclose(net_fresh.local_embedding.weight[mid], expected,
                     atol=1e-6),
      f'row {mid} becomes the mean of the {128 * 25} real token embeddings')
check(int(net_fresh.frame_mask_row_init_flag) == 1, 'the flag buffer records it')
with torch.no_grad():
    net_fresh.local_embedding.weight[mid] += 1.0      # pretend it trained
trained = net_fresh.local_embedding.weight[mid].clone()
net_fresh._init_frame_mask_row()
check(torch.equal(net_fresh.local_embedding.weight[mid], trained),
      'a second call is a no-op (a resumed run keeps its trained row)')
net_hook = build(True)
net_hook.on_train_start()
check(int(net_hook.frame_mask_row_init_flag) == 1,
      'on_train_start fires it -- AFTER the warm-start state_dict has loaded')
check(torch.allclose(
    net_hook.local_embedding.weight[mid],
    net_hook.local_embedding.weight[:128 * 25].mean(dim=0), atol=1e-6),
      'the hook produces the same row as the direct call')
net_plain.on_train_start()
check(int(net_plain.frame_mask_row_init_flag) == 0,
      'no-op for non-A.4 models (no frame_mask_token to touch)')
check('frame_mask_row_init_flag' in net_plain.state_dict()
      and 'frame_mask_row_init_flag' in net_a4.state_dict(),
      'the flag travels inside every checkpoint of both kinds')

print('\n=== 10. silent frames ===')
silent = make_frame(net_a4, 0, 24).view(1, -1).expand(4, -1).contiguous()
n_sil = silent.shape[0]
c0, f0 = net_a4._corrupt_frame_tokens(silent, torch.zeros(n_sil,
                                                          dtype=torch.long))
check(torch.equal(c0, silent) and not bool(f0.any()),
      'k=0 leaves a silent frame clean and unflagged')
cK, fK = net_a4._corrupt_frame_tokens(
    silent, torch.full((n_sil,), K, dtype=torch.long))
check(bool(fK.all()),
      'k=K flags it, so it cannot reveal its silence at the endpoint')
hits = 0
for i in range(400):
    torch.manual_seed(900 + i)
    _, f = net_a4._corrupt_frame_tokens(
        silent, torch.full((n_sil,), 2, dtype=torch.long))
    hits += int(f.sum())
rate = hits / (400 * n_sil)
check(abs(rate - 0.5) < 0.05,
      f'at k=2/K=4 it is fully masked ~k/K of the time (got {rate:.3f})')

print()
if fails:
    print(f'{len(fails)} FAILURE(S):')
    for f in fails:
        print(f'  - {f}')
    sys.exit(1)
print('all checks passed')
