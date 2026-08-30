"""Why is the best saved checkpoint worse than the best logged val_loss?

Opens every .ckpt in a run directory and prints what the trainer and the
ModelCheckpoint callback actually recorded -- the ground truth that
settles a wandb-vs-disk disagreement:

  per file : global_step, epoch, file mtime/size
  callback : best_model_score, best_model_path, kth_best_model_path,
             kth_value, current_score, and the full best_k_models map
             (path -> score) as it stood when that file was written.

How to read the output:

  * A ckpt whose global_step is LATER than the best file's step, while
    best_model_score still shows the older (worse) value, means the
    callback saw the better score and refused it -- monitor/aggregation
    problem.
  * best_k_models listing paths that no longer exist on disk means
    something outside the trainer deleted files (a cleanup job, or a
    second job sharing this dirpath), and the callback then had no
    reason to re-save.
  * last.ckpt's global_step tells you how far training actually got in
    the process that owns this directory -- if it is far below the
    wandb curve's last step, the wandb run and this directory are not
    the same process (concurrent jobs / a resumed run writing
    elsewhere).
  * Two files with the same epoch and different mtimes hours apart,
    with a gap in between, is the signature of a resume that started a
    fresh callback (best_k_models empty at restart).

Usage (via inspect_ckpt_callback.sbatch):
    python inspect_ckpt_callback.py --dir ckpt/<run_dir>
"""

import argparse
import os
import time
from glob import glob

import torch

CB_KEY_HINT = 'ModelCheckpoint'


def fmt_time(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))


def cb_state(ck):
    """Return (callback_key, state_dict) for the ModelCheckpoint entry."""
    cbs = ck.get('callbacks', {}) if isinstance(ck, dict) else {}
    for k, v in cbs.items():
        if CB_KEY_HINT in str(k):
            return str(k), v
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    args = p.parse_args()

    files = sorted(glob(os.path.join(args.dir, '*.ckpt')),
                   key=lambda f: os.path.getmtime(f))
    if not files:
        raise SystemExit(f'no .ckpt files under {args.dir}')

    print('=' * 78)
    print(f'CHECKPOINT / CALLBACK INSPECTION   {args.dir}')
    print(f'{len(files)} file(s), listed oldest-written first')
    print('=' * 78)

    on_disk = {os.path.abspath(f) for f in files}

    for f in files:
        size_gb = os.path.getsize(f) / 1e9
        print(f'\n--- {os.path.basename(f)}')
        print(f'    mtime {fmt_time(os.path.getmtime(f))}   {size_gb:.2f} GB')
        try:
            ck = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:                       # noqa: BLE001
            print(f'    [!] could not load: {e}')
            continue
        gs = ck.get('global_step')
        ep = ck.get('epoch')
        print(f'    global_step={gs}   epoch={ep}')

        key, st = cb_state(ck)
        if st is None:
            print('    [!] no ModelCheckpoint state in this file')
            continue
        print(f'    callback: {key}')
        for field in ('monitor', 'best_model_score', 'current_score',
                      'kth_value', 'best_model_path', 'kth_best_model_path',
                      'last_model_path'):
            if field in st:
                v = st[field]
                if hasattr(v, 'item'):
                    try:
                        v = v.item()
                    except Exception:                # noqa: BLE001
                        pass
                if isinstance(v, str) and v:
                    v = os.path.basename(v)
                print(f'      {field:<22} {v}')
        bk = st.get('best_k_models', {})
        if bk:
            print(f'      best_k_models ({len(bk)}):')
            for path, score in sorted(bk.items(), key=lambda kv: float(kv[1])):
                sc = score.item() if hasattr(score, 'item') else score
                missing = '' if os.path.abspath(path) in on_disk \
                    else '   <-- NOT ON DISK'
                print(f'        {float(sc):.5f}  {os.path.basename(path)}'
                      f'{missing}')

    print('\n' + '=' * 78)
    print('READ THE last.ckpt ROW FIRST: its global_step is how far the')
    print('process owning this directory actually got. Compare that with the')
    print('last step on the wandb curve, and compare best_model_score with')
    print('the curve minimum -- see this file\'s docstring for what each')
    print('mismatch means.')
    print('=' * 78)


if __name__ == '__main__':
    main()
