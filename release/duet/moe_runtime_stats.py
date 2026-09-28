"""Accumulate MoE routing statistics across EVERY forward pass of a
process -- the free-running counterpart to analyze_moe_routing.py.

The analyzer measures routing on one teacher-forced pass over clean
validation data. During generation the model routes its OWN sampled
content, and that distribution could differ. Activate this module by
env var (no CLI plumbing needed -- works under every driver that goes
through the variant's load_model):

    MOE_ROUTING_STATS=results/routing_stats_<job>.json

Every SimpleMoEFFN forward is wrapped; two statistics accumulate per
layer:

  all       every routing decision executed during decoding. NOTE the
            decode loop re-forwards the full growing sequence each step
            (and each refinement round), so early positions are counted
            many times -- fine for the dead-expert question (any win
            counts), biased for balance. Reported for completeness.
  frontier  only the last 4 positions of each forward (the newest
            committed frame pair + the two query slots) -- the routing
            of tokens at the moment they are being generated. The
            cleaner free-running load statistic.

An expert is DEAD in either statistic if it never appears in any
token's top-k. The report prints at process exit and the raw counts go
to the JSON for figures/.
"""

import atexit
import json
import os

import torch


def attach(net, out_path):
    layers = [l for l in getattr(net, 'global_layers', [])
              if getattr(l, 'ffn', None) is not None
              and hasattr(l.ffn, 'num_experts')]
    if not layers:
        print('[routing-stats] no MoE layers found; not attaching')
        return None
    stats = []
    for li, layer in enumerate(layers):
        ffn = layer.ffn
        E, k = ffn.num_experts, min(ffn.topk, ffn.num_experts)
        acc = {'layer': li, 'E': E, 'topk': k,
               'top1': torch.zeros(E, dtype=torch.long),
               'in_topk': torch.zeros(E, dtype=torch.long),
               'n': 0,
               'f_top1': torch.zeros(E, dtype=torch.long),
               'f_in_topk': torch.zeros(E, dtype=torch.long),
               'f_n': 0}
        stats.append(acc)
        orig = ffn.forward

        def wrapped(*args, _orig=orig, _ffn=ffn, _acc=acc, **kw):
            out = _orig(*args, **kw)
            pr = _ffn._last_routing_probs            # [B, L, E], detached
            B, L, E_ = pr.shape
            flat = pr.reshape(-1, E_)
            top = flat.topk(_acc['topk'], dim=-1).indices.cpu()
            _acc['top1'] += torch.bincount(top[:, 0], minlength=E_)
            _acc['in_topk'] += torch.bincount(top.reshape(-1),
                                              minlength=E_)
            _acc['n'] += flat.shape[0]
            fr = pr[:, max(0, L - 4):].reshape(-1, E_)
            ftop = fr.topk(_acc['topk'], dim=-1).indices.cpu()
            _acc['f_top1'] += torch.bincount(ftop[:, 0], minlength=E_)
            _acc['f_in_topk'] += torch.bincount(ftop.reshape(-1),
                                                minlength=E_)
            _acc['f_n'] += fr.shape[0]
            return out

        ffn.forward = wrapped

    print(f'[routing-stats] attached to {len(stats)} MoE layers; '
          f'report + {out_path} at exit')
    atexit.register(_report, stats, out_path)
    return stats


def _report(stats, out_path):
    if not stats or stats[0]['n'] == 0:
        print('[routing-stats] no forwards recorded')
        return
    print('\n' + '=' * 72)
    print('FREE-RUNNING ROUTING STATS (accumulated over every forward '
          'this process)')
    print('  all: every decision incl. re-forwarded history (dead-expert '
          'check is exact,')
    print('  loads biased toward early positions). frontier: last 4 '
          'positions per forward')
    print('  -- the newest frame pair + query slots; the clean '
          'free-running load.')
    print('=' * 72)
    dead_all, dead_frontier = [], []
    for scope, key_n, key1, keyk, dead in (
            ('ALL', 'n', 'top1', 'in_topk', dead_all),
            ('FRONTIER', 'f_n', 'f_top1', 'f_in_topk', dead_frontier)):
        print(f'\n  [{scope}]')
        print(f'  {"layer":>5} {"n":>10}  per-expert top-1 load'
              f'{"":<14} dead(top-k)')
        for a in stats:
            n = a[key_n]
            loads = (a[key1].float() / max(n, 1)).tolist()
            d = [e for e in range(a['E']) if a[keyk][e] == 0]
            dead.extend((a['layer'], e) for e in d)
            print(f'  {a["layer"]:>5} {n:>10}  '
                  + ' '.join(f'{v:.3f}' for v in loads)
                  + f'   {d if d else "-"}')
    print('\n  VERDICT: '
          + ('NO DEAD EXPERTS in free-running decoding (all + frontier).'
             if not dead_all and not dead_frontier
             else f'DEAD (layer, expert): all={dead_all} '
                  f'frontier={dead_frontier}'))
    print('=' * 72)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump([{k: (v.tolist() if torch.is_tensor(v) else v)
                    for k, v in a.items()} for a in stats], f)
    print(f'[routing-stats] wrote {out_path}')
