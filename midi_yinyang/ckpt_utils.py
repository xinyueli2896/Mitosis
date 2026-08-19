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
    metric-tagged ckpt with the smallest value over `last.ckpt`.

    The tag is whatever metric the run was CHECKPOINTED on (Lightning
    names the file after ModelCheckpoint's `monitor`), so it is
    'val_loss' for most runs but e.g. 'val_ce_loss_nondrum' for a C.1
    run trained with --monitor.

    Behaviour:
      * If `path` is a directory, scan for `*.<val_metric>=*.ckpt` files in
        it, pick the one with the smallest value. Fall back to `last.ckpt`
        in the same directory if no metric-tagged ckpts are present.
        Raises if the directory mixes TWO different monitored metrics,
        whose values are not on a common scale.
      * If `path` is a file named `last.ckpt` AND there are metric-tagged
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

    # Lightning names the file after the MONITORED metric, so the tag is
    # not always 'val_loss': a run trained with
    # --monitor val_ce_loss_nondrum writes '...val_ce_loss_nondrum=1.234'.
    # Matching 'val_loss=' literally missed those entirely and fell back to
    # last.ckpt -- silently reintroducing the exact best-vs-last bias this
    # module exists to prevent.
    pattern = re.compile(r'\.(val[A-Za-z0-9_]*)=([0-9.]+)\.ckpt$')
    by_tag = {}
    try:
        for fname in os.listdir(directory):
            m = pattern.search(fname)
            if m:
                try:
                    by_tag.setdefault(m.group(1), []).append(
                        (float(m.group(2)), fname))
                except ValueError:
                    continue
    except OSError as e:
        raise FileNotFoundError(
            f'[resolve_best_ckpt] could not list {directory!r}: {e!r}'
        )

    # Two different monitored metrics in one directory are not on a common
    # scale, so "smallest number wins" would pick across runs by accident.
    # Refuse rather than choose wrong -- a silent bad pick here contaminates
    # every downstream evaluation.
    if len(by_tag) > 1:
        raise FileNotFoundError(
            f'[resolve_best_ckpt] {directory!r} holds ckpts tagged with '
            f'DIFFERENT monitored metrics: {sorted(by_tag)}. Their values '
            f'are not comparable, so the best-of cannot be resolved. Pass an '
            f'explicit ckpt file, or keep one monitor per run directory '
            f'(RUN_TAG= gives a rerun its own dir).'
        )
    metric = next(iter(by_tag), 'val_loss')
    candidates = by_tag.get(metric, [])

    if not candidates:
        last_path = os.path.join(directory, 'last.ckpt')
        if os.path.exists(last_path):
            print(f'[ckpt] no metric-tagged ckpts in {directory}; '
                  f'using last.ckpt')
            return last_path
        raise FileNotFoundError(
            f'[resolve_best_ckpt] no ckpt files (metric-tagged or last.ckpt) '
            f'found in {directory!r}'
        )

    candidates.sort()
    # Skip unreadable candidates (truncated / mid-write / zero-byte) and
    # fall through to the next-best, so a single bad file cannot kill an
    # evaluation that has other perfectly good checkpoints.
    deep = os.environ.get('CKPT_DEEP_CHECK', '1') != '0'
    skipped = []
    for val_loss, fname in candidates:
        path_i = os.path.join(directory, fname)
        problem = _ckpt_problem(path_i, deep=deep)
        if problem is None:
            if skipped:
                print(f'[ckpt] WARNING: skipped {len(skipped)} unreadable '
                      f'ckpt(s) before this one: ' +
                      '; '.join(f'{n} ({why})' for n, why in skipped))
            print(f'[ckpt] auto-selected {fname} ({metric}={val_loss:.5f}) '
                  f'from {len(candidates)} {metric}-tagged ckpt(s) in '
                  f'{directory}')
            return path_i
        skipped.append((fname, problem))

    last_path = os.path.join(directory, 'last.ckpt')
    if os.path.exists(last_path) and _ckpt_problem(last_path, deep=deep) is None:
        print(f'[ckpt] WARNING: all {len(candidates)} {metric}-tagged ckpt(s) '
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


def _ckpt_problem(path, deep=False):
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
    if deep:
        # The structural check can still pass files that PyTorch's own
        # reader rejects (e.g. trailing junk after the central directory,
        # or a corrupt member). Only a real load is authoritative.
        try:
            import torch
            torch.load(path, map_location='meta', weights_only=False)
        except Exception:
            try:
                import torch
                torch.load(path, map_location='cpu', weights_only=False)
            except Exception as e:
                return f'{type(e).__name__}: {str(e)[:70]}'
    return None
