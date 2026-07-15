"""Train/inference PARITY test: teacher-force each variant's inference
context construction and assert its prediction hidden states numerically
match the training forward's at the same positions.

This is the property every model comparison depends on: the model is
evaluated exactly as trained. It mechanically catches sos-assembly
drift, _encode_frame vs local_encode drift, RoPE indexing drift, mask
drift, shift/off-by-one drift.

All parameters are RANDOMIZED (except token_type_embeddings, which are
zeroed+frozen in production for these variants) so that any code-path
difference produces a numeric difference -- fresh-init zeros would mask
e.g. the sos_offset bug.
"""
import os, sys, types
import importlib.machinery

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'transformers_roformer_moe', 'src'))


class _StubAnything:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return _StubAnything


for _name in ('pretty_midi', 'xf_midi', 'mido', 'six'):
    try:
        __import__(_name)
    except Exception:
        _m = _StubModule(_name)
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules[_name] = _m

import torch

torch.manual_seed(0)
B, T, POLY = 1, 8, 4
RAW = POLY * 4

COMMON = dict(large=False, with_velocity=False,
              moe_num_experts=2, moe_topk=1, global_num_layers=2)

ATOL, RTOL = 3e-4, 1e-4
results = []


def make_stream(drum):
    x = torch.zeros(B, T, POLY, 4, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            n = torch.randint(1, 3, (1,)).item()
            for s in range(POLY):
                if s < n:
                    x[b, t, s, 0] = 127 if drum else torch.randint(0, 120, (1,)).item()
                    x[b, t, s, 1] = torch.randint(30, 90, (1,)).item()
                    x[b, t, s, 2] = torch.randint(0, 4, (1,)).item()
                    x[b, t, s, 3] = torch.randint(40, 100, (1,)).item()
                elif s == n:
                    x[b, t, s, 0] = 254
                else:
                    x[b, t, s, 1] = 255
    return x.view(B, T, RAW)


def randomize(net):
    g = torch.Generator().manual_seed(1234)
    for name, p in net.named_parameters():
        if 'token_type_embeddings' in name:
            continue   # zero+frozen in production; keep to avoid false alarms
        with torch.no_grad():
            p.copy_(torch.empty_like(p).normal_(0, 0.05, generator=g))
    net.eval()
    return net


def training_h_clean(net, x):
    """Replicates the clean-stream construction of the m2c training
    forward (local_encode with parity token types + shift-by-2 + sos)."""
    batch_size, seq_len, subseq_len = x.shape
    idx = torch.arange(seq_len)
    frame_type = (idx % 2).long()
    tt = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
    sos_t = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
    tt = torch.cat([sos_t, tt], dim=-1)
    h, emb = net.local_encode(x, tt)
    h = h.view(batch_size, seq_len, -1)
    sos = net._assemble_sos(batch_size, h.device, h.dtype)
    return torch.cat([sos, h[:, :-2]], dim=1), emb


def inference_committed(net, x, t):
    """Replicates the decode loops' committed-history construction:
    per-frame _encode_frame + the sos the SHIPPED code uses."""
    from cp_transformer_m2c_moe_inference import build_inference_sos
    hs = []
    for j in range(t):
        hs.append(net._encode_frame(x[:, 2 * j], 0))
        hs.append(net._encode_frame(x[:, 2 * j + 1], 1))
    h_buf = (torch.cat(hs, dim=1) if hs
             else torch.zeros(B, 0, net.hidden_size))
    sos = build_inference_sos(net, B, x.device, h_buf.dtype)
    return torch.cat([sos, h_buf], dim=1)


def check(label, fn):
    try:
        max_diff = fn()
        ok = max_diff < ATOL
        results.append((label, 'PASS' if ok else f'FAIL max_diff={max_diff:.2e}'))
        print(f'{label}: max_diff={max_diff:.2e} {"PASS" if ok else "FAIL"}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append((label, f'ERROR: {type(e).__name__}: {e}'))


# ---------------------------------------------------------------- A.1
from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn

def parity_a1():
    net = randomize(M2CIntraCrossAttn(**COMMON, gate_init_bias=-10.0))
    x_mel, x_acc = make_stream(True), make_stream(False)
    xm, xc = net.preprocess(x_mel, torch.zeros(B, dtype=torch.long), y=x_acc)
    x = torch.stack([xm, xc], dim=2).view(B, T * 2, -1)
    with torch.no_grad():
        h_clean, _ = training_h_clean(net, x)
        h_tr, _ = net._global_interaction(h_clean)
        diffs = []
        for t in (1, 5):
            h_in = inference_committed(net, x, t)
            h_inf, _ = net._global_interaction(h_in)
            diffs.append((h_inf[:, -2] - h_tr[:, 2 * t]).abs().max().item())
            diffs.append((h_inf[:, -1] - h_tr[:, 2 * t + 1]).abs().max().item())
    return max(diffs)

check('A.1 IntraCrossAttn   ', parity_a1)

# ---------------------------------------------------------------- B.1
from cp_transformer_m2c_duet_anticipatory import M2CDuetAnticipatory

def parity_b1():
    net = randomize(M2CDuetAnticipatory(**COMMON, gate_init_bias=-10.0,
                                         anticipation_frames=2))
    x_mel, x_acc = make_stream(True), make_stream(False)
    xm, xc = net.preprocess(x_mel, torch.zeros(B, dtype=torch.long), y=x_acc)
    x = torch.stack([xm, xc], dim=2).view(B, T * 2, -1)
    with torch.no_grad():
        h_clean, _ = training_h_clean(net, x)
        h_tr, _ = net._global_interaction(h_clean)
        diffs = []
        for t in (1, 4):
            h_in = inference_committed(net, x, t)
            h_inf, _ = net._global_interaction(h_in)
            diffs.append((h_inf[:, -2] - h_tr[:, 2 * t]).abs().max().item())
            diffs.append((h_inf[:, -1] - h_tr[:, 2 * t + 1]).abs().max().item())
    return max(diffs)

check('B.1 Anticipatory     ', parity_b1)

# ---------------------------------------------------------------- A.3
from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion

def parity_a3():
    K = 2
    diffs = []
    x_mel, x_acc = make_stream(True), make_stream(False)
    for t in (1, 5):
        # Training side: T_full = t+1 makes the training clean stream
        # exactly [sos, pairs 0..t-1] = the inference context at step t.
        net = randomize(M2CDuetBlockDiffusion(
            **COMMON, gate_init_bias=-10.0, diffusion_K=K,
            self_cond_prob=0.0))
        xm, xc = net.preprocess(x_mel[:, :t + 1], torch.zeros(B, dtype=torch.long),
                                 y=x_acc[:, :t + 1])
        x = torch.stack([xm, xc], dim=2).view(B, (t + 1) * 2, -1)
        with torch.no_grad():
            _, q_logits_tr, _ = net.forward(
                x, T_query=t,
                k_m=torch.full((B,), K, dtype=torch.long),
                k_c=torch.full((B,), K, dtype=torch.long))
            # Inference side: committed pairs 0..t-1 + masked slots,
            # exactly as general_inference_diffusion round r=K builds it
            # (v1.1 aligned scheme -> no padding).
            hs = []
            for j in range(t):
                hs.append(net._encode_frame(x[:, 2 * j], 0))
                hs.append(net._encode_frame(x[:, 2 * j + 1], 1))
            h_clean = torch.cat(
                [net._assemble_sos(B, x.device, hs[0].dtype)] + hs, dim=1)
            slot_m = (net.mask_m_emb.view(1, 1, -1)
                      + net.k_emb_m(torch.tensor(K)).view(1, 1, -1)).expand(B, 1, -1)
            slot_c = (net.mask_c_emb.view(1, 1, -1)
                      + net.k_emb_c(torch.tensor(K)).view(1, 1, -1)).expand(B, 1, -1)
            h_in = torch.cat([h_clean, slot_m, slot_c], dim=1)
            h_g, _ = net._run_global_stack(h_in, T_query=max(t, 1))
            # decode with the same teacher-forced emb as training
            seq_len = (t + 1) * 2
            idx = torch.arange(seq_len)
            ft = (idx % 2).long()
            tt = torch.cat([
                ft.unsqueeze(0).unsqueeze(-1).expand(B, seq_len, 1),
                ft.unsqueeze(0).unsqueeze(-1).expand(B, seq_len, x.shape[-1]),
            ], dim=-1)
            _, emb = net.local_encode(x, tt)
            emb_r = emb.view(B, seq_len, x.shape[-1], -1)
            emb_q = torch.cat([emb_r[:, 2 * t:2 * t + 1],
                                emb_r[:, 2 * t + 1:2 * t + 2]], dim=1)
            q_logits_inf = net.local_decode(
                h_g[:, -2:], emb_q.view(B * 2, x.shape[-1], -1))
            diffs.append((q_logits_inf - q_logits_tr).abs().max().item())
    return max(diffs)

check('A.3 BlockDiffusion   ', parity_a3)

# ---------------------------------------------------------------- C.1
from cp_transformer_m2c_duet_rehearsal import M2CDuetRehearsal

def parity_c1():
    net = randomize(M2CDuetRehearsal(**COMMON, gate_init_bias=-10.0,
                                      recon_weight=1.0))
    x_mel, x_acc = make_stream(True), make_stream(False)
    xm, xc = net.preprocess(x_mel, torch.zeros(B, dtype=torch.long), y=x_acc)
    x = torch.stack([xm, xc], dim=2).view(B, T * 2, -1)
    with torch.no_grad():
        # training side (replicates rehearsal.forward)
        h_clean, _ = training_h_clean(net, x)   # [sos, shifted] (2T)
        seq_len = T * 2
        idx = torch.arange(seq_len)
        ft = (idx % 2).long()
        tt = torch.cat([
            ft.unsqueeze(0).unsqueeze(-1).expand(B, seq_len, 1),
            ft.unsqueeze(0).unsqueeze(-1).expand(B, seq_len, x.shape[-1]),
        ], dim=-1)
        h, _ = net.local_encode(x, tt)
        h = h.view(B, seq_len, -1)
        h_drum = h[:, 0::2]
        h_full_tr = torch.cat([h_drum, h_clean], dim=1)
        h_tr, _ = net._run_global_stack(h_full_tr, T=T)
        diffs = []
        for t in (1, 5):
            # inference side (replicates rehearsal inference loop)
            hd = torch.cat([net._encode_frame(x[:, 2 * j], 0)
                            for j in range(T)], dim=1)
            hs = []
            for j in range(t):
                hs.append(net._encode_frame(x[:, 2 * j], 0))
                hs.append(net._encode_frame(x[:, 2 * j + 1], 1))
            committed = torch.cat(
                [net._assemble_sos(B, x.device, hd.dtype)] + hs, dim=1)
            pad = torch.zeros(B, 2 * T - committed.shape[1], net.hidden_size)
            h_full_inf = torch.cat([hd, committed, pad], dim=1)
            h_inf, _ = net._run_global_stack(h_full_inf, T=T)
            diffs.append((h_inf[:, T + 2 * t] - h_tr[:, T + 2 * t]).abs().max().item())
            diffs.append((h_inf[:, T + 2 * t + 1] - h_tr[:, T + 2 * t + 1]).abs().max().item())
    return max(diffs)

check('C.1 Rehearsal        ', parity_c1)

# ---------------------------------------------------------------- C.2
from cp_transformer_m2c_duet_prefix import M2CDuetPrefix

def parity_c2():
    net = randomize(M2CDuetPrefix(**COMMON, gate_init_bias=-10.0))
    x_mel, x_acc = make_stream(True), make_stream(False)
    xm, xc = net.preprocess(x_mel, torch.zeros(B, dtype=torch.long), y=x_acc)
    x = torch.stack([xm, xc], dim=2).view(B, T * 2, -1)
    with torch.no_grad():
        seq_len = T * 2
        idx = torch.arange(seq_len)
        ft = (idx % 2).long()
        tt = torch.cat([
            ft.unsqueeze(0).unsqueeze(-1).expand(B, seq_len, 1),
            ft.unsqueeze(0).unsqueeze(-1).expand(B, seq_len, x.shape[-1]),
        ], dim=-1)
        h, _ = net.local_encode(x, tt)
        h = h.view(B, seq_len, -1)
        h_drum = h[:, 0::2]
        h_nd = h[:, 1::2]
        sos_n = (net.global_sos + net.sos_offset_c).view(1, 1, -1).expand(B, 1, -1)
        h_nd_shift = torch.cat([sos_n, h_nd[:, :-1]], dim=1)
        h_tr, _ = net._run_global_stack(
            torch.cat([h_drum, h_nd_shift], dim=1), T=T)
        diffs = []
        for t in (1, 5):
            hd = torch.cat([net._encode_frame(x[:, 2 * j], 0)
                            for j in range(T)], dim=1)
            hn = [net._encode_frame(x[:, 2 * j + 1], 1) for j in range(t)]
            committed = torch.cat([sos_n.to(hd.dtype)] + hn, dim=1)
            pad = torch.zeros(B, T - committed.shape[1], net.hidden_size)
            h_inf, _ = net._run_global_stack(
                torch.cat([hd, committed, pad], dim=1), T=T)
            diffs.append((h_inf[:, T + t] - h_tr[:, T + t]).abs().max().item())
    return max(diffs)

check('C.2 Prefix           ', parity_c2)

print('\n================ SUMMARY ================')
for label, status in results:
    print(f'{label} {status}')
bad = [r for r in results if r[1] != 'PASS']
sys.exit(1 if bad else 0)
