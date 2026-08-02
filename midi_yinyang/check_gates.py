"""How far have the cross-stream gates trained?

The duet variants warm-start from a single-stream CP transformer, which
has no cross-stream parameters, so those are baked in silent:

    gate_m.weight = 0                 (EXACTLY zero)
    gate_m.bias   = GATE_INIT_BIAS    (default -10)

READ THE WEIGHT NORM, NOT THE BIAS. A gate is sigmoid(W.x + b), so it is
input-dependent. Only while W is exactly zero does the bias alone decide
the output, making sigmoid(bias) the open fraction. Once W trains away
from zero the gate modulates on its input and the bias barely needs to
move -- so a near-init bias says nothing about whether the gate works.

This was learned the hard way. A.2 at 43k steps generates well
(survival_b = 1.000) yet its biases still sit at ~-9.8, i.e.
sigmoid(bias) ~ 6e-5; an earlier version of this script judged on that
and declared the healthy model's gates shut. Its |W| had reached ~0.99,
which is what actually shows the gates trained.

So: |W| ~ 0 means no cross-stream information can flow at all. |W| well
above zero means the mechanism is live, and the number is meaningful
only COMPARED ACROSS RUNS -- it is not a pass/fail test.

Usage:
    python check_gates.py ckpt/<run dir>                 # resolves best-val
    python check_gates.py ckpt/<run dir>/<file>.ckpt     # a specific file
    python check_gates.py ckpt/a/... ckpt/b/...          # compare runs
"""

import argparse
import math
import os
import re
import sys

import torch

from ckpt_utils import resolve_best_ckpt


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def gate_rows(state):
    """-> [(param_name, mean_bias, open_fraction, weight_norm)]"""
    rows = []
    for key, val in state.items():
        # per-block cross gates (gate_m/gate_c) and frame gates
        # (gate_fm/gate_fc) on the DuetBlock family
        if not re.search(r'gate_[a-z]*\.bias$', key):
            continue
        try:
            b = float(val.float().mean().item())
        except Exception:
            continue
        wkey = key[: -len('bias')] + 'weight'
        w = state.get(wkey)
        try:
            wn = float(w.float().norm().item()) if w is not None else float('nan')
        except Exception:
            wn = float('nan')
        rows.append((key, b, sigmoid(b), wn))
    rows.sort()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='run dirs and/or ckpt files')
    ap.add_argument('--init-bias', type=float, default=-10.0,
                    help='the GATE_INIT_BIAS the run started from '
                         '(default -10.0); used to flag gates that never moved')
    ap.add_argument('--w-eps', type=float, default=0.05,
                    help='mean ||W|| below this counts as untrained '
                         '(W is EXACTLY 0 at init)')
    args = ap.parse_args()

    any_stuck = False
    for path in args.paths:
        try:
            resolved = resolve_best_ckpt(path)
        except Exception as e:
            print(f'[skip] {path}: {e}')
            continue
        ck = torch.load(resolved, map_location='cpu', weights_only=False)
        state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
        step = ck.get('global_step', '?') if isinstance(ck, dict) else '?'
        rows = gate_rows(state)

        print('=' * 78)
        print(f'{os.path.basename(resolved)}')
        print(f'  global_step = {step}   gate params found: {len(rows)}')
        if not rows:
            print('  no gate_*.bias parameters -- not a gated duet variant?')
            continue

        print(f'  {"param":<44}{"bias":>9}{"open":>10}{"|W|":>10}')
        for name, b, op, wn in rows:
            print(f'  {name:<44}{b:>9.3f}{op:>10.5f}{wn:>10.4f}')

        biases = [b for _, b, _, _ in rows]
        opens = [o for _, _, o, _ in rows]
        wnorms = [w for _, _, _, w in rows if not math.isnan(w)]
        moved = sum(1 for b in biases if abs(b - args.init_bias) > 0.05)
        mean_open = sum(opens) / len(opens)
        mean_w = sum(wnorms) / len(wnorms) if wnorms else float('nan')
        print(f'  ---- bias-only open {mean_open:.5f} (NOT the true open '
              f'fraction; see below)')
        print(f'  ---- mean |W| {mean_w:.4f}   bias moved from init: '
              f'{moved}/{len(biases)}')
        # A gate is sigmoid(W.x + b) -- input-dependent. At init W is EXACTLY
        # zero, so the bias alone decides the output and sigmoid(bias) is the
        # open fraction. Once |W| > 0 that stops being true: the gate is
        # modulated by its input and the bias barely has to move. Judging on
        # sigmoid(bias) therefore condemns healthy models -- A.2 at 43k steps
        # generates well (survival_b = 1.000) with mean |W| 0.99 and its bias
        # still at -9.8. Untrained means W never left zero.
        if mean_w < args.w_eps:
            any_stuck = True
            print(f'  *** GATES UNTRAINED: mean |W| {mean_w:.4f} < {args.w_eps} --')
            print('      W is still at its zero init, so the gate output is')
            print('      fixed at sigmoid(bias) regardless of input and no')
            print('      cross-stream information can flow.')
        else:
            print(f'  ok: W has left its zero init, so the gates modulate on')
            print(f'      input. Compare mean |W| ACROSS runs for relative')
            print(f'      cross-stream development; it is not an on/off test.')
        del ck, state
    print('=' * 78)
    if any_stuck:
        print('At least one checkpoint has gates whose W never left zero --')
        print('no cross-stream information can flow through them at all.')
        print('Raise GATE_INIT_BIAS and regenerate the warm-start init (the')
        print('INIT_CKPT path now encodes the bias, so a new value gets its')
        print('own init rather than reusing the cached one).')
    else:
        print('All gates have trained W off zero, so none of these')
        print('checkpoints is a case of "cross-stream conditioning never')
        print('switched on". Differences in generation quality between them')
        print('have to be explained by something else -- compare mean |W|')
        print('and global_step across runs rather than reading any single')
        print('run as pass/fail.')
    return 1 if any_stuck else 0


if __name__ == '__main__':
    sys.exit(main())
