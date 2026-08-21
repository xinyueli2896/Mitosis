#!/bin/bash
# SLURM batch script: multi-GPU training of M2CDuetRehearsal on cscc-gpu.
# Drum-prefix rehearsal context (T positions, bidirectional) + standard
# DuetAttn-style interleaved AR suffix (2T positions). Total seq length
# 3T = 1.5x DuetAttn's 2T, so VRAM is tighter -- BATCH_SIZE default 4.
#
# Knobs (all optional):
#   --gres=gpu:N            how many GPUs to allocate (1..4).
#   --export=ALL,TASK=...   drumnondrum (default) or melchord.
#   --export=ALL,CKPT=...   if set, skip init + resume from that ckpt.
#   --export=ALL,MAX_LR=..  override LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=..  per-GPU batch size (default 4).
#   --export=ALL,GATE_INIT_BIAS=-4  raise if gates stay at init.
#   --export=ALL,RECON_WEIGHT=0.5   tune Brier-MSE drum recon weight.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first batch to .mid.

#SBATCH --job-name=mitosis_duet_rehearsal
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
INIT_CKPT=ckpt/m2c_duet_rehearsal_init/m2c_duet_rehearsal_init.ckpt

CKPT="${CKPT:-}"
MAX_LR="${MAX_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:--10.0}"
RECON_WEIGHT="${RECON_WEIGHT:-1.0}"
# Checkpoint selection metric. Passed as --ckpt_monitor: torchrun parses
# the multi-GPU command line first and abbreviation-matches a bare
# --monitor against its own --monitor-interval, dying with "ambiguous
# option" before the trainer is ever reached.
# Under TARGET_ONLY_LOSS=1 (the default) val_loss IS the conditional
# objective -- mod_b CE plus the MoE aux term -- so val_loss is the right
# monitor. Set MONITOR=val_ce_loss_nondrum only when running the old
# joint objective (TARGET_ONLY_LOSS=0), where val_loss mixes in the
# given stream and the Brier term.
MONITOR="${MONITOR:-val_loss}"
# Conditional-model corrections, all ON by default (see the model
# docstring). This variant inherited its shift and loss from the
# symmetric co-generation lineage; as a conditional model those were
# wrong. SUFFIX_SHIFT1 puts mod_a[k] in the slot predicting mod_b[k], so
# every b is conditioned on its own a. TARGET_ONLY_LOSS scores mod_b
# alone (mod_a is given; the Brier term stays logged, not optimised).
# PREFIX_STRIDE2 counts the prefix in the suffix's half-frame units, so
# the rehearsal distance is constant instead of growing with k.
# Set any to 0 to reproduce a pre-fix run. Existing ckpts are unaffected
# -- the flags are buffers and decode under their trained scheme.
PREFIX_STRIDE2="${PREFIX_STRIDE2:-1}"
SUFFIX_SHIFT1="${SUFFIX_SHIFT1:-1}"
TARGET_ONLY_LOSS="${TARGET_ONLY_LOSS:-1}"
TASK="${TASK:-drumnondrum}"
# Data max_polyphony, auto-set from TASK (override with MAX_POLYPHONY).
# The melchord datasets are cp8 (16-token frames; cp4 retired -- seventh
# chords are 5 notes); the trainer defaults to 16 and the
# --dump_samples_dir callback decodes real frames with it, so an unset
# value silently mis-decodes the sample dump on non-cp16 tasks -- the
# same bug fixed in the other duet training sbatches.
if [[ -z "${MAX_POLYPHONY:-}" ]]; then
    case "$TASK" in
        # melchord* -- every variant INCLUDING the _rev tasks, which
        # only swap mod_a/mod_b roles and read the SAME cp8 files.
        # Must stay in step with the melchord* pattern that exports
        # MELCHORD_CP below: an explicit list here missed _rev, so a
        # reverse run took MAX_POLYPHONY=16 and then exported
        # MELCHORD_CP=16, sending tasks.py to cp16 data paths.
        melchord*) MAX_POLYPHONY=8 ;;
        *)         MAX_POLYPHONY=16 ;;
    esac
fi
# tasks.py resolves melchord dataset paths from MELCHORD_CP; keep it in
# lockstep with the decode budget so data files and sample dump agree.
case "$TASK" in melchord*) export MELCHORD_CP="$MAX_POLYPHONY" ;; esac
# OneCycle horizon (was hardcoded 50000; same default kept).
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-50000}"
# Steps between val evals; lower on small melchord corpora (e.g. 100) so
# the early minimum is resolved by more than one or two points.
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
# Anti-exposure-bias augmentation for the sparse stream: prob per
# frame that a silent run of CTX_CORRUPT_LEN frames begins in the acc
# INPUT context while CE targets stay clean. 0 = off (default,
# behaviour unchanged). See cp_transformer_m2c_moe._ctx_corrupt_acc.
CTX_CORRUPT_PROB="${CTX_CORRUPT_PROB:-0.0}"
CTX_CORRUPT_LEN="${CTX_CORRUPT_LEN:-8}"
# Optional run-name suffix: keeps a rerun out of an older run's ckpt dir,
# where resolve_best_ckpt would otherwise pick across both runs.
RUN_TAG="${RUN_TAG:-}"
DUMP_SAMPLES_DIR="${DUMP_SAMPLES_DIR:-temp/train_samples_duet_rehearsal_${SLURM_JOB_ID}}"

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

echo "================================================================"
echo "Job $SLURM_JOB_ID  node=$(hostname)"
echo "Variant: m2c_duet_rehearsal  (drum-prefix + interleaved AR suffix)"
echo "Task:    $TASK"
echo "Detected GPUs: $NGPUS   Per-GPU batch: $BATCH_SIZE   Global: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR   gate_init_bias: $GATE_INIT_BIAS   recon_weight: $RECON_WEIGHT"
echo "Checkpoint selection: monitor=$MONITOR"
echo "Conditional scheme: prefix_stride2=$PREFIX_STRIDE2  suffix_shift1=$SUFFIX_SHIFT1  target_only_loss=$TARGET_ONLY_LOSS"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip init)"
else
    echo "Mode: FRESH (init from PRETRAINED -> INIT_CKPT=$INIT_CKPT)"
fi
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
        echo "[init] running init_pretrained_into_duet_rehearsal.py"
        python init_pretrained_into_duet_rehearsal.py \
            --pretrained "$PRETRAINED" \
            --output     "$INIT_CKPT" \
            --model_size large --global_num_layers 12 \
            --moe_num_experts 4 --moe_topk 2 \
            --gate_init_bias "$GATE_INIT_BIAS" \
            --recon_weight "$RECON_WEIGHT"
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

if [[ "$NGPUS" -le 1 ]]; then
    python cp_transformer_m2c_duet_rehearsal.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --ctx_corrupt_prob "$CTX_CORRUPT_PROB" \
        --ctx_corrupt_len "$CTX_CORRUPT_LEN" \
        --max_polyphony "$MAX_POLYPHONY" \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 \
        --dump_samples_dir "$DUMP_SAMPLES_DIR" \
        --gate_init_bias "$GATE_INIT_BIAS" \
        --recon_weight "$RECON_WEIGHT" \
        --ckpt_monitor "$MONITOR" \
        --prefix_stride2 "$PREFIX_STRIDE2" \
        --suffix_shift1 "$SUFFIX_SHIFT1" \
        --target_only_loss "$TARGET_ONLY_LOSS" \
        --checkpoint_path "$TRAIN_CKPT" \
        --wandb
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_duet_rehearsal.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --ctx_corrupt_prob "$CTX_CORRUPT_PROB" \
        --ctx_corrupt_len "$CTX_CORRUPT_LEN" \
        --max_polyphony "$MAX_POLYPHONY" \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 \
        --dump_samples_dir "$DUMP_SAMPLES_DIR" \
        --gate_init_bias "$GATE_INIT_BIAS" \
        --recon_weight "$RECON_WEIGHT" \
        --ckpt_monitor "$MONITOR" \
        --prefix_stride2 "$PREFIX_STRIDE2" \
        --suffix_shift1 "$SUFFIX_SHIFT1" \
        --target_only_loss "$TARGET_ONLY_LOSS" \
        --checkpoint_path "$TRAIN_CKPT" \
        --wandb
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
