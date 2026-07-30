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
import zipfile


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
    # Skip unreadable candidates (truncated / mid-write / zero-byte) and
    # fall through to the next-best, so a single bad file cannot kill an
    # evaluation that has other perfectly good checkpoints.
    skipped = []
    for val_loss, fname in candidates:
        path_i = os.path.join(directory, fname)
        problem = _ckpt_problem(path_i)
        if problem is None:
            if skipped:
                print(f'[ckpt] WARNING: skipped {len(skipped)} unreadable '
                      f'ckpt(s) before this one: ' +
                      '; '.join(f'{n} ({why})' for n, why in skipped))
            print(f'[ckpt] auto-selected {fname} (val_loss={val_loss:.5f}) '
                  f'from {len(candidates)} val_loss-tagged ckpt(s) in '
                  f'{directory}')
            return path_i
        skipped.append((fname, problem))

    last_path = os.path.join(directory, 'last.ckpt')
    if os.path.exists(last_path) and _ckpt_problem(last_path) is None:
        print(f'[ckpt] WARNING: all {len(candidates)} val_loss-tagged ckpt(s) '
              f'in {directory} are unreadable (' +
              '; '.join(f'{n}: {why}' for n, why in skipped) +
              '); falling back to last.ckpt')
        return last_path

    raise FileNotFoundError(
        f'[resolve_best_ckpt] every candidate in {directory!r} is unreadable: '
        + '; '.join(f'{n} ({why})' for n, why in skipped)
        + '. Typical causes: the training job is still WRITING the file '
          '(wait for it to finish, or pass an older ckpt explicitly), the '
          'write was truncated by a full disk/quota, or the file was '
          'copied while being written.'
    )


def _ckpt_problem(path):
    """Return None if `path` looks like a loadable torch checkpoint, else a
    short human-readable reason. Cheap: no tensor deserialization.

    Modern torch checkpoints are zip archives; the classic corruption
    symptom (`PytorchStreamReader failed reading zip archive: failed
    finding central directory`) is exactly a zip whose end-of-central-
    directory record is missing, which zipfile.is_zipfile also detects.
    Legacy non-zip (pure-pickle) checkpoints do not start with 'PK' and
    are passed through untouched.
    """
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return f'stat failed: {e}'
    if size == 0:
        return 'zero bytes'
    try:
        with open(path, 'rb') as f:
            magic = f.read(2)
    except OSError as e:
        return f'unreadable: {e}'
    if magic != b'PK':
        return None          # legacy pickle-format ckpt; leave it alone
    if not zipfile.is_zipfile(path):
        return (f'truncated zip ({size} bytes; no central directory) -- '
                f'still being written, or the write was cut short')
    return None
