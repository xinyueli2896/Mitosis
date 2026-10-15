#!/usr/bin/env bash
# Train every in-house system on one multi-GPU machine, without SLURM.
#
#   DATASET=nottingham NOTTINGHAM_RAW=/path/to/nottingham-dataset/MIDI \
#       bash scripts/train_all.sh 2>&1 | tee logs/train_all.log
#
# Runs train.sh for each system with its own CUDA_VISIBLE_DEVICES and
# packs the runs onto the machine's GPUs: the longest run starts first,
# and a waiting run starts as soon as enough GPUs are free. Each run
# keeps the GPU count of the paper recipe, so its learning rate and
# global batch are the paper's (train.sh scales the LR by GPU count):
#
#   dense                       2 GPUs, 100k steps   (the long pole)
#   duet, duet_stale, no_draft  2 GPUs,  50k steps
#   ss_finetuned                3 GPUs,  global batch 36
#   ss_scratch                  1 GPU,   global batch 12
#
# On 8 GPUs the four two-stream runs start together; the single-stream
# runs take the first GPUs freed, and everything is done when dense is.
#
# If the training tensors are missing, `prepare_data.sh train` and
# `eval` run first (NOTTINGHAM_RAW or POP909_RAW must then be set).
# Re-running resumes: train.sh picks up each duet run's last.ckpt.
#
# Knobs (environment):
#   GPU_IDS   GPUs to use, e.g. "0 1 2 3 4 5 6 7" (default: every GPU
#             nvidia-smi lists)
#   SYSTEMS   which runs, longest first (default: all six below)
#   GPUS_<system>  override one run's GPU count, e.g. GPUS_dense=4
#             (that run's LR then follows train.sh's scaling)
#   plus everything train.sh reads (BACKBONE, STALE_EVERY, WANDB, ...)
# Per-run logs: logs/train_<dataset>_<system>.log under release/.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-pop909}"
export DATASET
SYSTEMS="${SYSTEMS:-dense duet duet_stale no_draft ss_finetuned ss_scratch}"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

gpus_for() {
    local v="GPUS_$1"
    if [[ -n "${!v:-}" ]]; then echo "${!v}"; return; fi
    case "$1" in
        ss_finetuned) echo 3 ;;
        ss_scratch)   echo 1 ;;
        *)            echo 2 ;;
    esac
}

if [[ -z "${GPU_IDS:-}" ]]; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
read -r -a ALL_GPUS <<< "$GPU_IDS"
[[ ${#ALL_GPUS[@]} -gt 0 ]] || { echo "ERROR: no GPUs (set GPU_IDS)" >&2; exit 1; }
for s in $SYSTEMS; do
    if [[ "$(gpus_for "$s")" -gt ${#ALL_GPUS[@]} ]]; then
        echo "ERROR: $s needs $(gpus_for "$s") GPUs, only ${#ALL_GPUS[@]} available" >&2; exit 1
    fi
done

# ---- data, once, before anything trains -------------------------------
case "$DATASET" in
    nottingham) STEM=nottingham ;;
    *)          STEM=pop909 ;;
esac
if [[ ! -f "$ROOT/duet/data/${STEM}_melody_cp4_v2.pt" ]]; then
    echo "[train_all] preparing $DATASET data"
    bash "$ROOT/scripts/prepare_data.sh" train || exit 1
    bash "$ROOT/scripts/prepare_data.sh" eval || exit 1
fi

# ---- pack the runs onto the GPUs --------------------------------------
declare -A OWNER      # gpu -> pid holding it
declare -A RUN_OF     # pid -> system
declare -A RC         # system -> exit code
declare -A STARTED    # system -> start time
queue=($SYSTEMS)
port=$((29500 + RANDOM % 1000))

free_gpus() {
    local g out=()
    for g in "${ALL_GPUS[@]}"; do [[ -z "${OWNER[$g]:-}" ]] && out+=("$g"); done
    echo "${out[@]}"
}

reap() {
    local pid g s
    for pid in "${!RUN_OF[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        wait "$pid"; RC[${RUN_OF[$pid]}]=$?
        s="${RUN_OF[$pid]}"
        echo "[train_all] $(date '+%F %T') $s finished, exit ${RC[$s]}," \
             "$(( ($(date +%s) - STARTED[$s]) / 60 )) min"
        for g in "${!OWNER[@]}"; do [[ "${OWNER[$g]}" == "$pid" ]] && unset "OWNER[$g]"; done
        unset "RUN_OF[$pid]"
    done
}

while [[ ${#queue[@]} -gt 0 || ${#RUN_OF[@]} -gt 0 ]]; do
    reap
    launched=0
    # first queued run that fits; longer runs are earlier in the queue
    for i in "${!queue[@]}"; do
        s="${queue[$i]}"; need="$(gpus_for "$s")"
        read -r -a free <<< "$(free_gpus)"
        [[ ${#free[@]} -ge $need ]] || continue
        use=("${free[@]:0:$need}")
        devs="$(IFS=,; echo "${use[*]}")"
        port=$((port + 1))
        log="$LOGDIR/train_${DATASET}_${s}.log"
        echo "[train_all] $(date '+%F %T') $s on GPU $devs -> $log"
        CUDA_VISIBLE_DEVICES="$devs" MASTER_PORT="$port" \
            bash "$ROOT/scripts/train.sh" "$s" > "$log" 2>&1 &
        pid=$!
        RUN_OF[$pid]="$s"; STARTED[$s]=$(date +%s)
        for g in "${use[@]}"; do OWNER[$g]=$pid; done
        unset "queue[$i]"; queue=("${queue[@]}")
        launched=1
        break
    done
    [[ $launched -eq 1 ]] || sleep 30
done

echo "[train_all] summary ($DATASET):"
fail=0
for s in $SYSTEMS; do
    echo "  $s: exit ${RC[$s]:-?}"
    [[ "${RC[$s]:-1}" == 0 ]] || fail=1
done
exit $fail
