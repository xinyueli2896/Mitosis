#!/bin/bash
# Time-grid pretraining 2/3: tokenization statistics for FRAME_CODECS
# (default "1 3 6 12", i.e. 20/60/120/240 ms frames) at MAX_POLYPHONY:
# empty frames, notes per frame, notes the budget drops, context length.
# Written to duet/data/$CORPUS.stats_cp<P>.json. CPU, minutes. See
# pretrain_time_prep.sbatch for the chain.
#SBATCH --job-name=pretrain_time_stats
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=deploy
#SBATCH --qos=deploy
set -eo pipefail
source ~/.bashrc || true
# many ~/.bashrc files return early in batch shells, before conda init
command -v conda >/dev/null && eval "$(conda shell.bash hook)"
conda activate mitosis
set -u
ROOT="${SLURM_SUBMIT_DIR:?submit from release/}"
cd "$ROOT"; mkdir -p logs
bash scripts/pretrain_time.sh stats
