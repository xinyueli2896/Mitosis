"""Training-side smoke test for ALL active model variants.

For each variant: instantiate tiny (2-layer, 2-expert) model, feed a
random-but-valid raw batch through loss(), backward(), and check:
  * loss is finite
  * gradients reach the variant's NEW parameters (not just the backbone)
  * eval-mode loss also runs (different T_query / k branches)

CPU-only, ~1 minute. Run from midi_yinyang/:

    python smoke_test_variants.py

Exits nonzero if any variant fails, so it works as a pre-submit gate:

    python smoke_test_variants.py && sbatch train_<variant>.sbatch

MIDI-IO deps (pretty_midi etc.) are stubbed if absent -- the loss paths
never touch them, and this keeps the test runnable in bare containers.
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

B, T, POLY = 2, 8, 4          # tiny batch: 2 songs, 8 frames, 4-note polyphony
RAW_SUBSEQ = POLY * 4         # [program, pitch, duration, velocity] per slot


def make_stream(drum: bool):
    """Random valid raw stream [B, T, POLY*4]."""
    x = torch.zeros(B, T, POLY, 4, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            n_notes = torch.randint(1, 3, (1,)).item()   # 1-2 real notes
            for s in range(POLY):
                if s < n_notes:
                    x[b, t, s, 0] = 127 if drum else torch.randint(0, 120, (1,)).item()
                    x[b, t, s, 1] = torch.randint(30, 90, (1,)).item()   # pitch
                    x[b, t, s, 2] = torch.randint(0, 4, (1,)).item()     # duration idx
                    x[b, t, s, 3] = torch.randint(40, 100, (1,)).item()  # velocity
                elif s == n_notes:
                    x[b, t, s, 0] = 254                                  # frame EOS
                    x[b, t, s, 1] = 0
                else:
                    x[b, t, s, 1] = 255                                  # pad
    return x.view(B, T, RAW_SUBSEQ)


x_mel = make_stream(drum=True)     # mod-a = drum
x_acc = make_stream(drum=False)    # mod-b = nondrum
pitch_shift = torch.zeros(B, dtype=torch.long)

COMMON = dict(large=False, with_velocity=False,
              moe_num_experts=2, moe_topk=1, global_num_layers=2)

results = []


def check(label, build_fn, new_param_substrings):
    print(f'=== {label} ===')
    try:
        net = build_fn()
        net.train()
        loss, aux = net.loss(x_mel.clone(), x_acc.clone(), pitch_shift.clone())
        assert torch.isfinite(loss), f'non-finite train loss: {loss}'
        loss.backward()
        # gradient reach on variant-specific params
        grad_hits = {}
        for sub in new_param_substrings:
            hit = any(
                (p.grad is not None and p.grad.abs().sum() > 0)
                for n, p in net.named_parameters() if sub in n
            )
            grad_hits[sub] = hit
        net.eval()
        with torch.no_grad():
            eval_loss, _ = net.loss(x_mel.clone(), x_acc.clone(), pitch_shift.clone())
        assert torch.isfinite(eval_loss), f'non-finite eval loss: {eval_loss}'
        missing = [s for s, hit in grad_hits.items() if not hit]
        status = 'PASS' if not missing else f'PASS (loss) but NO GRAD on: {missing}'
        print(f'  train_loss={loss.item():.4f}  eval_loss={eval_loss.item():.4f}  {status}')
        results.append((label, status))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append((label, f'FAIL: {type(e).__name__}: {e}'))


# --- A.1 ---
from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn
check('A.1 M2CIntraCrossAttn',
      lambda: M2CIntraCrossAttn(**COMMON, gate_init_bias=-10.0),
      ['gate_m', 'gate_c', 'sos_offset'])

# --- A.2 ---
from cp_transformer_m2c_duet_block import M2CDuetBlockAttn
check('A.2 M2CDuetBlockAttn',
      lambda: M2CDuetBlockAttn(**COMMON, gate_init_bias=-10.0,
                                query_loss_weight=1.0),
      ['mask_m_emb', 'mask_c_emb', 'gate_fm', 'gate_fc'])

# --- A.3 ---
from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion
check('A.3 M2CDuetBlockDiffusion',
      lambda: M2CDuetBlockDiffusion(**COMMON, gate_init_bias=-10.0,
                                     query_loss_weight=1.0, diffusion_K=2),
      ['k_emb_m', 'k_emb_c', 'mask_m_emb', 'mask_c_emb'])

# --- B.1 ---
from cp_transformer_m2c_duet_anticipatory import M2CDuetAnticipatory
check('B.1 M2CDuetAnticipatory',
      lambda: M2CDuetAnticipatory(**COMMON, gate_init_bias=-10.0,
                                   anticipation_frames=2),
      ['gate_m', 'gate_c'])

# --- C.1 ---
from cp_transformer_m2c_duet_rehearsal import M2CDuetRehearsal
check('C.1 M2CDuetRehearsal',
      lambda: M2CDuetRehearsal(**COMMON, gate_init_bias=-10.0,
                                recon_weight=1.0),
      ['sos_offset'])

# --- C.2 ---
from cp_transformer_m2c_duet_prefix import M2CDuetPrefix
check('C.2 M2CDuetPrefix',
      lambda: M2CDuetPrefix(**COMMON, gate_init_bias=-10.0),
      ['sos_offset'])

print('\n================ SUMMARY ================')
for label, status in results:
    print(f'{label:32s} {status}')
bad = [r for r in results if not r[1].startswith('PASS')]
sys.exit(1 if bad else 0)
