#!/usr/bin/env bash
# Copy a checkpoint with some *_flag config buffers set (duet/set_ckpt_flags.py).
# Paths are relative to duet/. CPU only.
#   sbatch [res] slurm/run.sbatch set_ckpt_flags.sh <in.ckpt> <out_dir> key=value ...
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python set_ckpt_flags.py "$@"
