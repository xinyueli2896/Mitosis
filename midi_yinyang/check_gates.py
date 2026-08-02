"""Did the cross-stream gates ever open?

The duet variants warm-start from a single-stream CP transformer, and the
parameters that carry information BETWEEN the two streams do not exist in
that source. They are baked in silent:

    gate_m.weight = 0
    gate_m.bias   = GATE_INIT_BIAS   (default -10 -> sigmoid ~ 4.5e-5)

so at step 0 each modality reproduces what the pretrained block computes
on its own stream in isolation, and the entire duet mechanism has to be
LEARNED through those gates. If training stops before they open -- which
best-val selection can force on a small corpus, where val turns upward
within ~1k steps -- the model is still effectively two independent
single-stream models, and the stream that depends on cross-conditioning
(the follower/chord) degenerates while the other coasts on inherited
weights.

This reports, per gate parameter: sigmoid(bias) -- the open fraction --
and ||weight||, which is 0 at init and only becomes non-zero if the gate
learned to modulate on its input. A checkpoint whose gates all still sit
at the init value is undertrained for the duet task no matter what its
val_loss says.

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
        print(f'  ---- mean open fraction {mean_open:.5f}   '
              f'gates moved from init: {moved}/{len(biases)}   '
              f'mean |W| {sum(wnorms) / len(wnorms) if wnorms else float("nan"):.4f}')
        if moved == 0 or mean_open < 1e-3:
            any_stuck = True
            print('  *** GATES STILL AT INIT: this checkpoint has essentially no')
            print('      cross-stream conditioning. Its val_loss reflects the')
            print('      inherited single-stream backbone, not a trained duet.')
        del ck, state

    print('=' * 78)
    if any_stuck:
        print('At least one checkpoint has unopened gates. On a corpus too small')
        print('to train them before val turns, raise GATE_INIT_BIAS (e.g. -4 or')
        print('-2) so they start partly open instead of at sigmoid(-10)=4.5e-5,')
        print('and regenerate the warm-start init -- the INIT_CKPT path now')
        print('encodes the bias, so a new value gets its own init rather than')
        print('silently reusing the cached one.')
    return 1 if any_stuck else 0


if __name__ == '__main__':
    sys.exit(main())
