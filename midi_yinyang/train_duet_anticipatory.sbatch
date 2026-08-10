#!/bin/bash
# SLURM batch script: multi-GPU training of M2CDuetAnticipatory (B.1).
# DuetAttn architecture; the only difference is the input-side reindexing
# -- drum is shifted ahead of nondrum by ANTICIPATION_FRAMES frames in
# the interleaved sequence.
#
# WARM-START AND AUTO-RESUME
#   Three-tier startup logic (run again with no flags to keep training):
#
#   1. If CKPT is explicitly set, full Lightning resume from that ckpt.
#      (Can also be set to a regular DuetAttn A.1 ckpt -- state dict is
#      identical, so the architecture loads cleanly and you start from a
#      trained model.)
#   2. Else if the default run directory already contains last.ckpt
#      (i.e. a prior run of this same sbatch), AUTO-RESUME from it.
#      LR_TOTAL_STEPS auto-bumps to (ckpt's global_step + 50000) so the
#      OneCycle schedule extends and the next run does another ~50k steps.
#   3. Else cold-start: init from the base single-stream CP-transformer
#      pretrained ckpt via init_pretrained_into_intra_cross_attn.py
#      (B.1 reuses A.1's init -- state-dict-identical).
#
# Knobs (all optional):
#   --gres=gpu:N            how many GPUs to allocate (1..4).
#                           DEFAULT IS NOW 2. The run directory name
#                           encodes the GLOBAL batch (BATCH_SIZE x NGPUS),
#                           so changing the GPU count changes the run dir
#                           and auto-resume will NOT find a run trained at
#                           a different GPU count. To keep the previous
#                           4-GPU global batch of 16 on 2 GPUs, pass
#                           BATCH_SIZE=8 (memory permitting); otherwise the
#                           2-GPU default gives global batch 8 and a new
#                           run dir, i.e. a fresh cold-start run.
#   --export=ALL,TASK=...   drumnondrum (default), melchord,
#                           melchord_nottingham, or melchord_pop909_nottingham.
#                           Data max_polyphony (cp16 vs cp8) is auto-set from
#                           TASK; override with MAX_POLYPHONY if needed.
#   --export=ALL,CKPT=...   explicit override for the resume ckpt. If
#                           unset, the script auto-detects the run's
#                           last.ckpt.
#   --export=ALL,LR_TOTAL_STEPS=N    total steps the OneCycle schedule
#                           spans. Default: 50000 for a cold start,
#                           (prior_step + 50000) on auto-resume.
#   --export=ALL,ANTICIPATION_FRAMES=16  drum lookahead in frames.
#                                        Default 16 (= 1 bar at 16th-note resolution).
#   --export=ALL,FRESH_SCHEDULE=1    opt-in: load weights only, drop
#                           optimizer + scheduler + step counter.
#   --export=ALL,MAX_LR=..  override LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=..  per-GPU batch size (default 4).
#   --export=ALL,GATE_INIT_BIAS=-4  raise if gates stay at init.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first batch to .mid.

#SBATCH --job-name=mitosis_duet_anticipatory
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

# Reuse the A.1 init ckpt -- state dict is identical, only the
# loss-time data reindexing differs.
PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt

CKPT="${CKPT:-}"
FRESH_SCHEDULE="${FRESH_SCHEDULE:-0}"
# Empty default; resolved below to 50000 on cold start or to
# (prior_step + 50000) on auto-resume.
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-}"
# Steps between val evaluations. The default 500 resolves a minimum
# that arrives inside the first ~1k steps with one or two points --
# enough to make a real descent look like a monotonic rise, and to
# place the saved checkpoint badly. Lower it on the small melchord
# corpora (VAL_CHECK_INTERVAL=100).
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
# Anti-exposure-bias augmentation for the sparse stream: prob per
# frame that a silent run of CTX_CORRUPT_LEN frames begins in the acc
# INPUT context while CE targets stay clean. 0 = off (default,
# behaviour unchanged). See the comment in cp_transformer_m2c_moe.loss.
CTX_CORRUPT_PROB="${CTX_CORRUPT_PROB:-0.0}"
CTX_CORRUPT_LEN="${CTX_CORRUPT_LEN:-8}"
MAX_LR="${MAX_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:--10.0}"
# The warm-start init BAKES gate_init_bias into the checkpoint
# (gate_m/gate_c weight=0, bias=GATE_INIT_BIAS), and the step below
# skips regeneration when INIT_CKPT already exists. With a fixed
# filename that silently reused a cached init built at a DIFFERENT
# bias, so raising GATE_INIT_BIAS was a no-op: training loaded the
# old gates and the knob appeared to do nothing. Encode the bias in
# the path so each value gets its own init.
GATE_TAG="${GATE_INIT_BIAS//./p}"; GATE_TAG="${GATE_TAG//-/m}"
INIT_CKPT="ckpt/m2c_intra_cross_attn_init_gb${GATE_TAG}/m2c_intra_cross_attn_init_gb${GATE_TAG}.ckpt"
ANTICIPATION_FRAMES="${ANTICIPATION_FRAMES:-16}"
TASK="${TASK:-drumnondrum}"
DUMP_SAMPLES_DIR="${DUMP_SAMPLES_DIR:-temp/train_samples_duet_anticipatory_${SLURM_JOB_ID}}"
# Data polyphony per stream: drumnondrum data is cp16, melchord data is
# cp8 (POP909/Nottingham; cp4 retired -- seventh chords are 5 notes).
# Only affects the sample-dump callback's row reshape, but it's on by
# default here and WILL crash on non-cp16 data at the wrong width if
# left at the python default of 16.
if [[ -z "${MAX_POLYPHONY:-}" ]]; then
    case "$TASK" in
        melchord|melchord_nottingham|melchord_pop909_nottingham) MAX_POLYPHONY=8 ;;
        *)                                                       MAX_POLYPHONY=16 ;;
    esac
fi
# tasks.py resolves melchord dataset paths from MELCHORD_CP; keep it in
# lockstep with the decode budget so data files and sample dump agree.
case "$TASK" in melchord*) export MELCHORD_CP="$MAX_POLYPHONY" ;; esac

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NGPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
NGPUS="${NGPUS:-1}"

if [[ -z "$MAX_LR" ]]; then
    case "$NGPUS" in
        1) MAX_LR=5e-5 ;;
        2) MAX_LR=7e-5 ;;
        4) MAX_LR=1e-4 ;;
        8) MAX_LR=1.5e-4 ;;
        *) MAX_LR=1e-4 ;;
    esac
fi

# --- auto-detect prior run -------------------------------------------
# The run directory mirrors the python script's default model_name. If
# the user didn't pass CKPT explicitly and last.ckpt exists there from a
# prior submission, auto-resume.
GLOBAL_BATCH=$((BATCH_SIZE * NGPUS))
RUN_DIR="ckpt/m2c_duet_anticipatory_v1.0_large_gnl12_${TASK}_k${ANTICIPATION_FRAMES}_batch_${GLOBAL_BATCH}_schedule"
if [[ -z "$CKPT" && -f "$RUN_DIR/last.ckpt" ]]; then
    CKPT="$RUN_DIR/last.ckpt"
    echo "[auto-resume] found $CKPT  (run dir: $RUN_DIR)"
elif [[ -z "$CKPT" ]]; then
    # No run at THIS global batch. Warn if runs exist for the same
    # task at a different GPU count -- the run dir encodes global
    # batch, so changing --gres silently starts a fresh run instead
    # of continuing the old one.
    _sibling=$(ls -d "${RUN_DIR%_batch_*}"_batch_*_schedule 2>/dev/null | grep -v "^${RUN_DIR}$" || true)
    if [[ -n "$_sibling" ]]; then
        echo "[warn] no last.ckpt in $RUN_DIR (global batch $GLOBAL_BATCH = $BATCH_SIZE x $NGPUS GPUs)."
        echo "[warn] but these runs exist for the same task at other global batches:"
        echo "$_sibling" | sed "s/^/[warn]   /"
        echo "[warn] This run will COLD-START. To continue one of the above instead,"
        echo "[warn] match its GPU count/BATCH_SIZE, or pass CKPT=<that dir>/last.ckpt."
    fi
fi

# --- resolve LR_TOTAL_STEPS ------------------------------------------
if [[ -z "$LR_TOTAL_STEPS" ]]; then
    if [[ -n "$CKPT" && "$FRESH_SCHEDULE" != "1" ]]; then
        LR_TOTAL_STEPS=$(python -c "
import torch
ck = torch.load('$CKPT', map_location='cpu', weights_only=False)
step = ck.get('global_step', 0) if isinstance(ck, dict) else 0
print(step + 50000)
")
        echo "[auto-resume] LR_TOTAL_STEPS = $LR_TOTAL_STEPS (ckpt global_step + 50000)"
    else
        LR_TOTAL_STEPS=50000
    fi
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Variant: m2c_duet_anticipatory (B.1)  k=$ANTICIPATION_FRAMES"
echo "Task:    $TASK   (data max_polyphony: $MAX_POLYPHONY)"
echo "Detected GPUs: $NGPUS   Per-GPU batch: $BATCH_SIZE   Global: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR   gate_init_bias: $GATE_INIT_BIAS"
if [[ -n "$CKPT" ]]; then
    if [[ "$FRESH_SCHEDULE" == "1" ]]; then
        echo "Mode: CONTINUE (load weights from CKPT=$CKPT, fresh LR schedule)"
    else
        echo "Mode: RESUME (full Lightning resume from CKPT=$CKPT)"
    fi
else
    echo "Mode: COLD START (init from PRETRAINED -> INIT_CKPT=$INIT_CKPT, reusing A.1's init)"
fi
echo "LR_TOTAL_STEPS=$LR_TOTAL_STEPS   VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL   RUN_DIR=$RUN_DIR"
echo "----------------------------------------------------------------"
nvidia-smi || true
echo "================================================================"

if [[ -z "$CKPT" ]]; then
    if [[ -f "$INIT_CKPT" ]]; then
        echo "[init] $INIT_CKPT already exists; skipping warm-start step."
    else
        if [[ ! -f "$PRETRAINED" ]]; then
            echo "[init] ERROR: pretrained ckpt not found at $PRETRAINED"
            exit 1
        fi
        echo "[init] running init_pretrained_into_intra_cross_attn.py (shared A.1 init)"
        python init_pretrained_into_intra_cross_attn.py \
            --pretrained "$PRETRAINED" \
            --output     "$INIT_CKPT" \
            --model_size large --global_num_layers 12 \
            --moe_num_experts 4 --moe_topk 2 \
            --gate_init_bias "$GATE_INIT_BIAS"
        echo "[init] done -> $INIT_CKPT"
    fi
    TRAIN_CKPT="$INIT_CKPT"
else
    if [[ ! -f "$CKPT" ]]; then
        echo "[resume] ERROR: CKPT not found at $CKPT"
        exit 1
    fi
    TRAIN_CKPT="$CKPT"
fi
echo "================================================================"
echo "[train] --checkpoint_path = $TRAIN_CKPT"
echo "================================================================"

# Pre-flight: fail fast with a clear message if the allocated GPUs
# are visible but unusable (bad node) -- otherwise the job dies deep
# inside Lightning device setup with a cryptic INTERNAL ASSERT.
python check_gpu_health.py --expect "$NGPUS" || exit 1

export MASTER_PORT=$((20000 + RANDOM % 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_TRAIN_ARGS=()
if [[ "$FRESH_SCHEDULE" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--fresh_schedule)
fi

COMMON_ARGS=(
    --task "$TASK"
    --batch_size "$BATCH_SIZE"
    --model_size large
    --moe_num_experts 4 --moe_topk 2
    --max_lr "$MAX_LR"
    --lr_total_steps "$LR_TOTAL_STEPS" \
    --val_check_interval "$VAL_CHECK_INTERVAL" \
    --ctx_corrupt_prob "$CTX_CORRUPT_PROB" \
    --ctx_corrupt_len "$CTX_CORRUPT_LEN"
    --gradient_clip_val 1.0
    --aux_loss_weight 0.01
    --moe_monitor_every_n_steps 500
    --dump_samples_dir "$DUMP_SAMPLES_DIR"
    --max_polyphony "$MAX_POLYPHONY"
    --gate_init_bias "$GATE_INIT_BIAS"
    --anticipation_frames "$ANTICIPATION_FRAMES"
    --checkpoint_path "$TRAIN_CKPT"
    --wandb
)

if [[ "$NGPUS" -le 1 ]]; then
    python cp_transformer_m2c_duet_anticipatory.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_duet_anticipatory.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
