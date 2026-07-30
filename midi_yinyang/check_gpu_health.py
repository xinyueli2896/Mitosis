"""Pre-flight GPU health check for the training sbatch scripts.

Catches the failure mode where SLURM allocates N GPUs (so
CUDA_VISIBLE_DEVICES lists N devices and torchrun spawns N ranks) but
the driver on that node cannot actually enumerate/open them. Without
this check the job dies deep inside Lightning's device setup with a
~200-line DeferredCudaCallError whose root message is the cryptic

    device >= 0 && device < num_gpus INTERNAL ASSERT FAILED ...
    device=2, num_gpus=

which reads like a code bug but is a bad-node problem: the fix is to
resubmit (SLURM will usually place the job elsewhere) and, if it
recurs on the same host, to report that node.

Exits 0 when every expected device is usable, 1 otherwise.

Usage (in an sbatch, before torchrun):
    python check_gpu_health.py --expect "$NGPUS" || exit 1
"""

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expect', type=int, default=None,
                   help='number of GPUs the job expects to use')
    args = p.parse_args()

    host = os.environ.get('SLURMD_NODENAME') or os.uname().nodename
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')

    try:
        import torch
    except Exception as e:
        print(f'[gpu-check] FAIL: cannot import torch: {e!r}')
        return 1

    if not torch.cuda.is_available():
        print(f'[gpu-check] FAIL on host={host}: torch.cuda.is_available() '
              f'is False (CUDA_VISIBLE_DEVICES={cvd})')
        return 1

    count = torch.cuda.device_count()
    print(f'[gpu-check] host={host}  CUDA_VISIBLE_DEVICES={cvd}  '
          f'device_count={count}')

    # Actually touch every device: device_count() can report devices the
    # driver later fails to open, which is exactly the fault this guards.
    bad = []
    for i in range(count):
        try:
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f'[gpu-check]   cuda:{i} {name} sm_{cap[0]}{cap[1]} '
                  f'{free / 2**30:.1f}/{total / 2**30:.1f} GiB free')
        except Exception as e:
            bad.append(i)
            print(f'[gpu-check]   cuda:{i} UNUSABLE: {type(e).__name__}: {e}')

    if bad:
        print(f'[gpu-check] FAIL on host={host}: devices {bad} are visible '
              f'but unusable -- this node\'s GPUs are unhealthy. Resubmit '
              f'(SLURM will usually pick another node); if it recurs on the '
              f'same host, report that node to the cluster admins.')
        return 1

    if args.expect is not None and count < args.expect:
        print(f'[gpu-check] FAIL on host={host}: expected {args.expect} '
              f'usable GPUs, found {count}. Either the allocation did not '
              f'materialize or the node is degraded -- resubmit, or lower '
              f'--gres/BATCH_SIZE to match.')
        return 1

    print(f'[gpu-check] OK: {count} usable GPU(s) on {host}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
