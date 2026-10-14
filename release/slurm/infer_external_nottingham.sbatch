#!/bin/bash
# AMT and Whole-Song Gen on the Nottingham held-out prompts, one 8-GPU
# node: each GPU runs one AMT slice and one Whole-Song Gen slice
# (SHARD=k/8, every 8th tune), then Whole-Song Gen's chord re-voicing
# runs once over all of them.
#
#   sbatch slurm/infer_external_nottingham.sbatch
#
# Setup is idempotent (clones, venvs and Whole-Song Gen's Drive
# downloads are skipped when present). Outputs:
#   duet/temp/E1_nottingham/AMT  and  duet/temp/E1_nottingham/WSfv4
# Per-slice logs: logs/{amt,wholesong}_nottingham_<k>.log
# FRESH=1 (default) deletes earlier Whole-Song Gen outputs first (they
# predate the prompt-leak fix); AMT outputs are kept and finished tunes
# skipped.
#SBATCH --job-name=infer_ext_nottingham
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
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
export DATASET=nottingham
N=8

if [[ "${FRESH:-1}" == 1 ]]; then
    rm -rf -- duet/temp/E1_nottingham/WSf duet/temp/E1_nottingham/WSfv4
fi

bash scripts/external/amt.sh setup
bash scripts/external/wholesong.sh setup
bash scripts/external/wholesong.sh prep      # stages prompts + song folders once

pids=(); names=()
for ((k = 0; k < N; k++)); do
    CUDA_VISIBLE_DEVICES=$k SHARD=$k/$N bash scripts/external/amt.sh generate \
        > "logs/amt_nottingham_$k.log" 2>&1 &
    pids+=($!); names+=("amt $k")
    CUDA_VISIBLE_DEVICES=$k SHARD=$k/$N bash scripts/external/wholesong.sh generate \
        > "logs/wholesong_nottingham_$k.log" 2>&1 &
    pids+=($!); names+=("wholesong $k")
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] ${names[$i]}"
    else
        echo "[FAILED] ${names[$i]} (see its log)"; fail=1
    fi
done

bash scripts/external/wholesong.sh map
echo "[infer] AMT -> duet/temp/E1_nottingham/AMT, Whole-Song Gen -> duet/temp/E1_nottingham/WSfv4"
exit $fail
