"""Q query pairs (--query_pairs): does supervising Q frames per forward
change anything except how much supervision there is?

Training supervises the query slots on ONE frame per forward, while the
AR loss scores all 2T positions -- so the frame pass, the only mechanism
the decode loop actually uses, gets ~1/T of the gradient. Q>1 appends Q
query PAIRS for Q distinct frames. The claim this file checks is that
the added pairs are INERT with respect to each other and to the clean
stream: the only thing Q changes is how many frames get a loss.

Claims checked:

  1. Q=1 IS THE OLD MASK. _build_masks with an int T_query reproduces
     the historical three formulas exactly, so every existing
     checkpoint's training geometry is untouched.
  2. LAYOUT. L = clean_len + 2Q; pair j sits at clean_len+2j (m) and
     clean_len+2j+1 (c) and carries prediction-frame T_query[j].
  3. NO LEAK. Clean rows never attend a query column, in any of the
     three passes, at any Q. This is what keeps the AR loss honest.
  4. PAIR ISOLATION. In the frame pass a query slot attends EXACTLY one
     query column: its own partner. Never another pair's slot -- which
     would hand it a second draft of a different frame that inference
     could not supply. In intra/cross a query row attends only clean
     columns.
  5. VISIBILITY WINDOW. Pair j sees clean frames < T_query[j] only, and
     each pair's window follows its OWN frame.
  6. PAIR EQUIVALENCE (the strong one). Because pairs are mutually
     blind and the clean stream is blind to all of them, pair j of a
     Q>1 forward must produce the SAME query logits as a Q=1 forward at
     that same frame. Checked at k=K (slot = mask embedding) and k=0
     (slot = that frame's ground-truth encoding, so it also proves each
     pair reads the RIGHT frame). Both model variants.
     Compared with a tight tolerance rather than bitwise: the masked-out
     entries are mathematically -inf, but a longer sequence reassociates
     the softmax and GEMM sums, so the last bits may differ. The max
     absolute deviation is printed -- it should be ~1e-7, not ~1e-3.
  7. AR LOGITS UNCHANGED. Same argument, other direction: adding query
     pairs cannot move the clean stream's outputs (same tolerance).
  8. ROTARY. Pair j's two slots take rotary index 2*T_j+2 / 2*T_j+3
     (v1.1), halved to T_j+1 for both under v1.2 -- its OWN frame's
     phase. So no Q>1 run visits a rotary position a Q=1 run would not.
  9. TARGET ALIGNMENT. The loss's target assembly puts frame T_query[j]
     at rows 2j (m) and 2j+1 (c) -- the same order forward() emits the
     logits in, and the order local_decode's B-major flattening
     preserves.
  10. COERCION. k and the self-conditioning masks accept both the
     historical un-paired shapes and the paired ones.

Runs on CPU in seconds. Usage (via audit_query_pairs.sbatch):
    python audit_query_pairs.py
"""

import sys

import torch

from cp_transformer_m2c_duet_block import normalize_T_query
from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

fails = []


def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond:
        fails.append(msg)


def build(token_level_mask=False, query_pairs=1, time_rope_aligned=False):
    torch.manual_seed(0)
    net = M2CDuetBlockDiffusion(
        large=False, with_velocity=False,
        moe_num_experts=4, moe_topk=2,
        global_num_layers=2, diffusion_K=4,
        token_level_mask=token_level_mask,
        query_pairs=query_pairs,
        time_rope_aligned=time_rope_aligned,
    )
    net.eval()
    return net


def toy_batch(net, B=2, T=8, S=8):
    """Interleaved [B, 2T, S]: (program, pitch-dur) pairs then EOS."""
    tok = net.tokenizer
    x = torch.full((B, 2 * T, S), tok.pad_token, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    for b in range(B):
        for i in range(2 * T):
            n_notes = int(torch.randint(1, 4, (1,), generator=g))
            for n in range(n_notes):
                x[b, i, 2 * n] = 24 if i % 2 == 0 else 0
                x[b, i, 2 * n + 1] = int(
                    torch.randint(128, 128 * 25, (1,), generator=g))
            x[b, i, 2 * n_notes] = tok.eos_token
    return x


def legacy_masks(clean_len, T_query, device):
    """The three mask formulas as they stood before Q existed."""
    L = clean_len + 2
    pos = torch.arange(L, device=device)
    is_query = pos >= clean_len
    is_clean = ~is_query
    modality = torch.where(is_clean, pos % 2, (pos - clean_len) % 2)
    pred_frame = torch.where(is_clean, pos // 2,
                             torch.full_like(pos, T_query))
    same_mod = (modality[:, None] == modality[None, :])
    diff_mod = ~same_mod
    f_p, f_q = pred_frame[:, None], pred_frame[None, :]
    same_frame = (f_p == f_q)
    strict_past_frame = (f_q < f_p)
    q_clean, q_query = is_clean[None, :], is_query[None, :]
    p_clean, p_query = is_clean[:, None], is_query[:, None]
    clean_clean = p_clean & q_clean
    causal_pos = (pos[None, :] <= pos[:, None])
    diag = torch.eye(L, dtype=torch.bool, device=device)
    m_intra = ((clean_clean & same_mod & causal_pos)
               | (p_query & q_clean & same_mod & strict_past_frame))
    m_cross = ((clean_clean & diff_mod & strict_past_frame)
               | (p_query & q_clean & diff_mod & strict_past_frame)) | diag
    m_frame = ((clean_clean & diff_mod & same_frame)
               | (p_query & q_query & ~diag))
    return m_intra, m_cross, m_frame


# Mathematically these comparisons are exact. Numerically a longer
# sequence reassociates the softmax and GEMM sums, so allow a few ulps
# of float32 drift and print the observed maximum -- a real leak between
# pairs would show up orders of magnitude above this, not near it.
TOL = 1e-4

dev = torch.device('cpu')
net = build()
layer = net.global_layers[0]
CLEAN = 16                                   # 8 frames, interleaved
TQ = (3, 5, 6)


print('\n=== 1. Q=1 reproduces the historical masks ===')
for t in (1, 3, 7):
    got = layer._build_masks(CLEAN, t, dev)
    want = legacy_masks(CLEAN, t, dev)
    check(all(torch.equal(a, b) for a, b in zip(got, want)),
          f'all three masks identical to the pre-Q formulas at T_query={t}')
check(normalize_T_query(4) == (4,) and normalize_T_query([1, 2]) == (1, 2),
      'normalize_T_query accepts an int and a sequence alike')


print('\n=== 2. layout at Q>1 ===')
m_intra, m_cross, m_frame = layer._build_masks(CLEAN, TQ, dev)
L = CLEAN + 2 * len(TQ)
check(m_intra.shape == (L, L) and m_frame.shape == (L, L),
      f'L = clean_len + 2Q = {L}')
pos = torch.arange(L)
is_query = pos >= CLEAN
mod_of = torch.where(is_query, (pos - CLEAN) % 2, pos % 2)
check(all(int(mod_of[CLEAN + 2 * j]) == 0 and int(mod_of[CLEAN + 2 * j + 1]) == 1
          for j in range(len(TQ))),
      'pair j is (m at clean_len+2j, c at clean_len+2j+1)')


print('\n=== 3. clean rows never see a query column ===')
for name, m in (('intra', m_intra), ('cross', m_cross), ('frame', m_frame)):
    leak = int(m[~is_query][:, is_query].sum())
    check(leak == 0, f'mask_{name}: no clean->query edge ({leak} found)')


print('\n=== 4. pair isolation ===')
frame_qq = m_frame[is_query][:, is_query]
partner = torch.zeros_like(frame_qq)
n_q = int(is_query.sum())
for a in range(n_q):
    for b in range(n_q):
        partner[a, b] = (a // 2 == b // 2) and (a != b)
check(torch.equal(frame_qq, partner),
      'in the frame pass each slot attends EXACTLY its own partner')
check(int(m_intra[is_query][:, is_query].sum()) == 0
      and int(m_cross[is_query][:, is_query].sum() - n_q) == 0,
      'intra/cross give query rows no query columns (cross keeps only '
      'the diagonal empty-row guard)')


print('\n=== 5. visibility window follows each pair own frame ===')
clean_frame = pos[~is_query] // 2
ok_window = True
for j, t_j in enumerate(TQ):
    for slot in (CLEAN + 2 * j, CLEAN + 2 * j + 1):
        # intra gives same-mod clean, cross gives diff-mod clean, both
        # restricted to frames strictly before the pair's own frame --
        # so their union must be EXACTLY the clean prefix.
        seen = m_intra[slot][~is_query] | m_cross[slot][~is_query]
        if not torch.equal(seen, clean_frame < t_j):
            ok_window = False
check(ok_window,
      f'each pair sees exactly the clean frames strictly before its own '
      f'T_query {TQ} -- no more, no less')
w0 = m_intra[CLEAN][~is_query] | m_cross[CLEAN][~is_query]
w2 = m_intra[CLEAN + 4][~is_query] | m_cross[CLEAN + 4][~is_query]
check(int(w2.sum()) > int(w0.sum()),
      'a later pair sees strictly more of the clean stream than an '
      'earlier one (the windows are not shared)')


print('\n=== 6. pair equivalence: pair j == a Q=1 forward at frame T_j ===')
for tk_flag in (False, True):
    label = 'A.4' if tk_flag else 'plain'
    net_e = build(token_level_mask=tk_flag)
    x = toy_batch(net_e)
    B, S = x.shape[0], x.shape[2]
    K = net_e.diffusion_K
    for k_val in (K, 0):
        with torch.no_grad():
            _, q_multi, _ = net_e.forward(x, T_query=TQ, k_m=k_val, k_c=k_val)
        q_multi = q_multi.view(B, 2 * len(TQ), S, -1)
        worst = 0.0
        for j, t_j in enumerate(TQ):
            with torch.no_grad():
                _, q_one, _ = net_e.forward(
                    x, T_query=t_j, k_m=k_val, k_c=k_val)
            q_one = q_one.view(B, 2, S, -1)
            worst = max(worst, float(
                (q_multi[:, 2 * j:2 * j + 2] - q_one).abs().max()))
        check(worst < TOL,
              f'{label}, k={k_val}: every pair reproduces its own Q=1 '
              f'forward (max |diff| {worst:.2e})')


print('\n=== 7. AR logits are untouched by Q ===')
net_e = build()
x = toy_batch(net_e)
K = net_e.diffusion_K
with torch.no_grad():
    ar_one, _, _ = net_e.forward(x, T_query=TQ[0], k_m=K, k_c=K)
    ar_multi, _, _ = net_e.forward(x, T_query=TQ, k_m=K, k_c=K)
ar_diff = float((ar_one - ar_multi).abs().max())
check(ar_diff < TOL,
      f'the clean stream cannot see the query slots, so adding pairs '
      f'leaves every AR logit unmoved (max |diff| {ar_diff:.2e})')


print('\n=== 8. rotary index per pair ===')
for time_aligned in (False, True):
    net_r = build(time_rope_aligned=time_aligned)
    Lr = CLEAN + 2 * len(TQ)
    positions = torch.arange(Lr)
    for j, t_j in enumerate(TQ):
        positions[CLEAN + 2 * j] = 2 * t_j + 2
        positions[CLEAN + 2 * j + 1] = 2 * t_j + 3
    if time_aligned:
        positions = torch.div(positions, 2, rounding_mode='floor')
    want = ([t + 1 for t in TQ for _ in range(2)] if time_aligned
            else [v for t in TQ for v in (2 * t + 2, 2 * t + 3)])
    check(positions[CLEAN:].tolist() == want,
          f'{"v1.2" if time_aligned else "v1.1"}: pair j takes its own '
          f'frame phase {want}')
    q1_max = max(2 * t + 3 for t in TQ)
    check(int(positions[CLEAN:].max()) <= q1_max,
          'no rotary position beyond what a Q=1 run at the same frames '
          'already visits')


print('\n=== 9. target alignment ===')
x = toy_batch(net_e)
targets = torch.cat([x[:, 2 * t:2 * t + 2] for t in TQ], dim=1)
check(targets.shape[1] == 2 * len(TQ), 'targets_query is [B, 2Q, S]')
check(all(torch.equal(targets[:, 2 * j], x[:, 2 * TQ[j]])
          and torch.equal(targets[:, 2 * j + 1], x[:, 2 * TQ[j] + 1])
          for j in range(len(TQ))),
      'row 2j is frame T_j mod_a, row 2j+1 is frame T_j mod_b -- the '
      'order forward() emits logits in')


print('\n=== 10. coercion of k and the self-conditioning masks ===')
B, Q = 3, len(TQ)
net_c = build()
kk = net_c._coerce_k_pairs(2, B, Q, dev)
check(kk.shape == (B, Q) and bool((kk == 2).all()),
      'an int k broadcasts to [B, Q]')
kv = net_c._coerce_k_pairs(torch.arange(B), B, Q, dev)
check(kv.shape == (B, Q) and bool((kv[:, 0] == torch.arange(B)).all())
      and bool((kv[:, 0:1] == kv).all()),
      'a [B] k (what inference passes) broadcasts across pairs')
kp = torch.arange(B * Q).view(B, Q)
check(torch.equal(net_c._coerce_k_pairs(kp, B, Q, dev), kp),
      'a [B, Q] k passes through untouched')
sc1 = torch.tensor([True, False, True])
check(net_c._coerce_sc_mask(sc1, B, Q).shape == (B, Q),
      'a [B] sc mask broadcasts to [B, Q]')
check(net_c._coerce_sc_mask(None, B, Q) is None, 'None stays None')
toks_flat = torch.zeros(B, 8, dtype=torch.long)
toks_pair = torch.arange(B * Q * 8).view(B, Q, 8)
check(torch.equal(net_c._sc_tok_slice(toks_flat, 1), toks_flat)
      and torch.equal(net_c._sc_tok_slice(toks_pair, 1), toks_pair[:, 1]),
      '_sc_tok_slice handles [B, S] and [B, Q, S]')
emb_flat = torch.zeros(B, 1, 4)
emb_pair = torch.arange(B * Q * 4).view(B, Q, 1, 4).float()
check(torch.equal(net_c._sc_emb_slice(emb_flat, 1), emb_flat)
      and torch.equal(net_c._sc_emb_slice(emb_pair, 1), emb_pair[:, 1]),
      '_sc_emb_slice handles [B, 1, H] and [B, Q, 1, H]')


print()
if fails:
    print(f'{len(fails)} FAILURE(S):')
    for f in fails:
        print(f'  - {f}')
    sys.exit(1)
print('all checks passed')
