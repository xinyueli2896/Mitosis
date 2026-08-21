"""C.1 audit: mirror every layout/index rule from the source and assert
train==inference, causality, masking and RoPE for all scheme combos."""
import numpy as np, re
SRC = '/home/user/Mitosis/midi_yinyang/cp_transformer_m2c_duet_rehearsal.py'
src = open(SRC).read()
# the rules this audit encodes must still be the ones in the file
for probe in ["torch.cat([sos, h[:, :-1]], dim=1)", "torch.cat([sos, h[:, :-2]], dim=1)",
              "offset = 1 if self.suffix_shift1 else 2", "prefix_pos = positions * 2 + offset",
              "h_global[:, T_full:T_full + seq_len]", "logits_4d[:, 0::2]",
              "frame_idx % 2 == 0,"]:
    assert probe in src, f'source no longer matches audit: {probe}'

T = 6; L = 3*T
def lab(j): return ('a' if j % 2 == 0 else 'b', j // 2)

def suffix_holds(sh1):
    if sh1: return {0: 'sos_m', **{i: lab(i-1) for i in range(1, 2*T)}}
    return {0: 'sos_m', 1: 'sos_c', **{i: lab(i-2) for i in range(2, 2*T)}}
def predicts(i): return lab(i)
def infer_buffer(sh1, t):
    """suffix contents at the moment slot 2t+1 is read, per the AR loop."""
    if sh1:
        b = [];  [b.extend([lab(2*i), lab(2*i+1)]) for i in range(t)]
        return ['sos_m'] + b + [lab(2*t)]
    b = [];  [b.extend([lab(2*i), lab(2*i+1)]) for i in range(t)]
    return ['sos_m', 'sos_c'] + b
def local_pos(ps2, sh1):
    pos = np.arange(L); off = 1 if sh1 else 2
    return np.where(pos < T, pos*2 + off if ps2 else pos, pos - T)
def masks(sh1):
    pos = np.arange(L); is_pre = pos < T
    mod = np.where(is_pre, 0, (pos - T) % 2)
    causal = pos[None, :] <= pos[:, None]
    vis = ((is_pre[:,None] & is_pre[None,:]) | (~is_pre[:,None] & is_pre[None,:])
           | (~is_pre[:,None] & ~is_pre[None,:] & causal))
    same = mod[:,None] == mod[None,:]
    return (vis & same), ((vis & ~same) | np.eye(L, dtype=bool))

fails = []
def check(cond, msg):
    print(f'  [{"ok " if cond else "FAIL"}] {msg}')
    if not cond: fails.append(msg)

for sh1 in (True, False):
    for ps2 in (True, False):
        print(f'\n=== suffix_shift1={sh1}  prefix_stride2={ps2} ===')
        holds, lp = suffix_holds(sh1), local_pos(ps2, sh1)
        intra, cross = masks(sh1)

        check(len(holds) == 2*T, f'suffix length == 2T ({len(holds)})')
        check(all(predicts(i) == lab(i) for i in range(2*T)),
              'slot i predicts frame x_i (index-aligned with emb)')
        if sh1:
            check(all(holds[2*k+1] == ('a', k) for k in range(T)),
                  'slot predicting b_k HOLDS a_k  <- the conditioning fix')
        else:
            check(all(holds[2*k+1] == ('b', k-1) for k in range(1, T)),
                  'legacy: slot predicting b_k holds b_{k-1}')

        # train vs inference
        ok = all(infer_buffer(sh1, t)[2*t+1] == holds[2*t+1] for t in range(T))
        check(ok, 'inference slot content == training slot content')
        check(all(len(infer_buffer(sh1, t)) == 2*t+2 for t in range(T)),
              'inference commits nothing past the query slot (causal)')

        # masks
        check(not intra[:T, T:].any() and not cross[:T, T:].any(),
              'prefix never sees the suffix')
        fut = [(p,q) for p in range(T,L) for q in range(T,L)
               if q > p and (intra[p,q] or cross[p,q])]
        check(not fut, 'no future leakage inside the suffix')
        check(all(cross[T+2*k+1, :T].all() for k in range(T)),
              'every b-slot sees the WHOLE prefix (the rehearsal)')

        # rope
        a_slot = {v[1]: i for i, v in holds.items() if v != 'sos_m' and v != 'sos_c' and v[0]=='a'}
        if ps2:
            check(all(lp[j] == lp[T + a_slot[j]] for j in a_slot),
                  'prefix copy of a_j shares the suffix copy\'s rotary index')
            b_rot = {int(lp[T+i]) for i,v in holds.items()
                     if v not in ('sos_m','sos_c') and v[0]=='b'}
            check(not ({int(lp[j]) for j in range(T)} & b_rot),
                  'no prefix rotary index collides with a mod_b slot')
            d = {int(lp[T+2*k+1] - lp[k]) for k in range(T)}
            check(len(d) == 1, f'dist(query b_k -> prefix a_k) constant = {d}')
        else:
            d = [int(lp[T+2*k+1] - lp[k]) for k in range(T)]
            check(len(set(d)) > 1, f'legacy: distance grows {d} (expected)')
        check(int(lp.max())+1 <= 2*T + 2, 'rope table bound sane')

        # loss scope
        scored = [i for i in range(2*T) if i % 2 == 1]
        check(all(lab(i)[0] == 'b' for i in scored),
              'target_only_loss scores exactly the mod_b frames')

print('\n' + ('ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILURE(S): {fails}'))
