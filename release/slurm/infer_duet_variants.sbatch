#!/bin/bash
# Generation for the Duet variants of README "Improving Duet's
# co-generation", one 8-GPU node. Each GPU k takes shard k/8 of the
# songs for every system, the systems running side by side on it.
#
#   sbatch slurm/infer_duet_variants.sbatch                              # Nottingham
#   sbatch --export=ALL,DATASET=pop909 slurm/infer_duet_variants.sbatch  # POP909
#   sbatch --export=ALL,SYSTEMS="duet_v2 duet_v2c48 duet_prev" slurm/infer_duet_variants.sbatch
#
# Default SYSTEMS: the four decodes of the existing duet checkpoint
# (duet_joint duet_jointmix duet_k0 duet_bestar; POP909 leaves out
# duet_bestar, the released checkpoint has no best_ar/ -- pass
# CKPT_duet_bestar=<run dir>/best_ar to add it). Checkpoints default to
# each system's run directory (released files on POP909); CKPT_<system>
# overrides one. Finished samples are skipped, so a resubmitted job
# resumes. Outputs: duet/temp/E1[_nottingham]/<system id>/. Per-call
# logs: logs/gen_<dataset>_<system>_<k>.log
#SBATCH --job-name=infer_duet_variants
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
export DATASET="${DATASET:-nottingham}"
N=8
if [[ "$DATASET" == nottingham ]]; then
    SYSTEMS="${SYSTEMS:-duet_joint duet_jointmix duet_k0 duet_bestar}"
else
    SYSTEMS="${SYSTEMS:-duet_joint duet_jointmix duet_k0${CKPT_duet_bestar:+ duet_bestar}}"
fi

bash scripts/stage_prompts.sh                # once, before the parallel calls

pids=(); names=()
for ((k = 0; k < N; k++)); do
    for s in $SYSTEMS; do
        v="CKPT_$s"
        env CUDA_VISIBLE_DEVICES=$k SHARD=$k/$N ${!v:+CKPT="${!v}"} \
            bash scripts/generate.sh "$s" > "logs/gen_${DATASET}_${s}_$k.log" 2>&1 &
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
    id=$(bash -c "source scripts/common.sh >/dev/null; system_id $s")
    out=$(bash -c "source scripts/common.sh >/dev/null; echo \$OUT_ROOT")
    echo "[infer] $s -> duet/$out/$id: $(find "duet/$out/$id" -name '*.mid' 2>/dev/null | wc -l) midi files"
done
exit $fail
