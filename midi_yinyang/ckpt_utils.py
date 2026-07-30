"""Shared checkpoint-selection helper.

Lives in its own module so BOTH the duet inference scripts and the
single-stream `cp_transformer_inference.py` resolve checkpoints
identically. Before this existed only the duet path auto-selected the
best-val ckpt, so an evaluation that pointed the duet systems at a run
directory and the single-stream baseline at `last.ckpt` was silently
comparing best-val models against a LAST model -- a real bias whenever
the baseline overfits after its val minimum (which the POP909 S1
finetune does).
"""

import os
import re


def resolve_best_ckpt(path):
    """Resolve a ckpt argument into an actual file path, preferring the
    val_loss-tagged ckpt with the smallest val_loss over `last.ckpt`.

    Behaviour:
      * If `path` is a directory, scan for `*.val_loss=*.ckpt` files in it,
        pick the one with the smallest val_loss. Fall back to `last.ckpt`
        in the same directory if no val_loss-tagged ckpts are present.
      * If `path` is a file named `last.ckpt` AND there are val_loss-tagged
        siblings in the same directory, prefer the best of those.
      * Otherwise honor the exact file path passed.

    Returns the resolved absolute-or-as-given path. Raises FileNotFoundError
    if no ckpt can be found.
    """
    if os.path.isdir(path):
        directory = path
        search = True
    elif os.path.isfile(path):
        directory = os.path.dirname(path) or '.'
        # Only auto-redirect from 'last.ckpt'; honor any other explicit file.
        search = os.path.basename(path).lower() == 'last.ckpt'
        if not search:
            return path
    else:
        raise FileNotFoundError(
            f'[resolve_best_ckpt] {path!r} is neither a file nor a directory'
        )

    pattern = re.compile(r'val_loss=([0-9.]+)\.ckpt$')
    candidates = []
    try:
        for fname in os.listdir(directory):
            m = pattern.search(fname)
            if m:
                try:
                    val_loss = float(m.group(1))
                    candidates.append((val_loss, fname))
                except ValueError:
                    continue
    except OSError as e:
        raise FileNotFoundError(
            f'[resolve_best_ckpt] could not list {directory!r}: {e!r}'
        )

    if not candidates:
        last_path = os.path.join(directory, 'last.ckpt')
        if os.path.exists(last_path):
            print(f'[ckpt] no val_loss-tagged ckpts in {directory}; '
                  f'using last.ckpt')
            return last_path
        raise FileNotFoundError(
            f'[resolve_best_ckpt] no ckpt files (val_loss or last.ckpt) '
            f'found in {directory!r}'
        )

    candidates.sort()
    best_val_loss, best_fname = candidates[0]
    best_path = os.path.join(directory, best_fname)
    print(f'[ckpt] auto-selected {best_fname} (val_loss={best_val_loss:.5f}) '
          f'from {len(candidates)} val_loss-tagged ckpt(s) in {directory}')
    return best_path
