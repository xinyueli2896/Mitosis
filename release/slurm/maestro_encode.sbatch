#!/bin/bash
# MAESTRO v3 stage 2 of 3: encode every recording with EnCodec 32 kHz
# (facebook/encodec_32khz, MusicGen's codec: 50 Hz x 4 codebooks), one
# shard per visible GPU, codes cached per piece under CODES_DIR. A
# resubmitted job only encodes what is missing. Per-shard logs:
# logs/maestro_encode_shard<i>.log. See maestro_download.sbatch for the chain.
#SBATCH --job-name=maestro_encode
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --time=12:00:00
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
nvidia-smi --query-gpu=index,name,memory.used --format=csv
bash scripts/maestro.sh encode
