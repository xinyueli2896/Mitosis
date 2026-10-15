#!/bin/bash
# Generation for every in-house system on the Nottingham held-out prompts,
# one 8-GPU node. Each GPU k takes shard k/8 of the tunes (every 8th) for
# every system, the systems running side by side on it, so all GPUs carry
# the same load and finish together.
#
#   sbatch slurm/infer_nottingham.sbatch
#
# Checkpoints: each system's Nottingham run directory from training (the
# loader picks its best val_loss checkpoint); pass CKPT_<system>=<path>
# through --export to override one. Finished samples are skipped, so a
# resubmitted job resumes. Outputs: duet/temp/E1_nottingham/<system id>/
# (A3ctcaT, A3stale, D1, A1, S1, S-scratch). Per-call logs:
# logs/gen_nottingham_<system>_<k>.log
#SBATCH --job-name=infer_nottingham
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
SYSTEMS="${SYSTEMS:-duet duet_stale dense no_draft ss_finetuned ss_scratch}"

bash scripts/stage_prompts.sh                # once, before the parallel calls

pids=(); names=()
for ((k = 0; k < N; k++)); do
    for s in $SYSTEMS; do
        v="CKPT_$s"
        env CUDA_VISIBLE_DEVICES=$k SHARD=$k/$N ${!v:+CKPT="${!v}"} \
            bash scripts/generate.sh "$s" > "logs/gen_nottingham_${s}_$k.log" 2>&1 &
        pids+=($!); names+=("$s shard $k")
    done
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] ${names[$i]}"
    else
        echo "[FAILED] ${names[$i]} (see its log)"; fail=1
    fi
done

for s in $SYSTEMS; do
    id=$(DATASET=nottingham bash -c "source scripts/common.sh >/dev/null; system_id $s")
    echo "[infer] $s -> duet/temp/E1_nottingham/$id: $(find "duet/temp/E1_nottingham/$id" -name '*.mid' 2>/dev/null | wc -l) midi files"
done
exit $fail
