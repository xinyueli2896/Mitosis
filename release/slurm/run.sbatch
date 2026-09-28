#!/bin/bash
# Run any stage script under SLURM. Resources are set on the command line,
# so one template serves every stage:
#
#   sbatch --gres=gpu:2 --time=48:00:00 slurm/run.sbatch train.sh duet
#   sbatch --gres=gpu:1 --time=12:00:00 slurm/run.sbatch generate.sh duet
#   sbatch --cpus-per-task=4 --time=02:00:00 slurm/run.sbatch evaluate.sh
#
# Chain stages with sbatch's own dependencies, so each stays a plain job:
#   J1=$(sbatch --parsable --gres=gpu:1 slurm/run.sbatch generate.sh duet)
#   sbatch --dependency=afterok:$J1 slurm/run.sbatch evaluate.sh
#
# Activate your environment in ~/.bashrc or export it before sbatch
# (--export=ALL is the default).
#SBATCH --job-name=duet
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
mkdir -p logs
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STAGE="${1:?usage: sbatch [resources] slurm/run.sbatch <script> [args]}"; shift
bash "$ROOT/scripts/$STAGE" "$@"
