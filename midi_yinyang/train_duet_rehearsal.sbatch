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
# INIT_CKPT is derived below and KEYED ON GATE_INIT_BIAS: the bias is
# baked into the warm-start weights, and this script skips the init step
# whenever the file already exists -- so a single fixed path meant that
# changing GATE_INIT_BIAS silently reused an init built with the old one.

CKPT="${CKPT:-}"
MAX_LR="${MAX_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RECON_WEIGHT="${RECON_WEIGHT:-1.0}"
# Checkpoint selection metric. Passed as --ckpt_monitor: torchrun parses
# the multi-GPU command line first and abbreviation-matches a bare
# --monitor against its own --monitor-interval, dying with "ambiguous
# option" before the trainer is ever reached.
# val_loss carries the Brier retrieval term whenever RECON_WEIGHT != 0,
# so it is not purely the conditional objective. Default to ranking on
# val_ce_loss_nondrum -- mod_b CE alone -- and fall back to val_loss only
# when the Brier term is off, where the two coincide.
if [[ -z "${MONITOR:-}" ]]; then
    if [[ "${RECON_WEIGHT:-1.0}" == "0" || "${RECON_WEIGHT:-1.0}" == "0.0" ]]
    then MONITOR=val_loss
    else MONITOR=val_ce_loss_nondrum; fi
fi
# Conditional-model corrections, all ON by default (see the model
# docstring). This variant inherited its shift and loss from the
# symmetric co-generation lineage; as a conditional model those were
# wrong. SUFFIX_SHIFT1 puts mod_a[k] in the slot predicting mod_b[k], so
# every b is conditioned on its own a. TARGET_ONLY_LOSS drops the
# token-level CE on mod_a only -- the Brier RETRIEVAL term is separate,
# controlled by RECON_WEIGHT, and stays ON: the slot predicting mod_a[k]
# holds mod_a[k-1], so it must fetch mod_a[k] out of the prefix, over the
# same keys/values the conditioning path reads.
# PREFIX_STRIDE2 counts the prefix in the suffix's half-frame units, so
# the rehearsal distance is constant instead of growing with k.
# Set any to 0 to reproduce a pre-fix run. Existing ckpts are unaffected
# -- the flags are buffers and decode under their trained scheme.
PREFIX_STRIDE2="${PREFIX_STRIDE2:-1}"
SUFFIX_SHIFT1="${SUFFIX_SHIFT1:-1}"
TARGET_ONLY_LOSS="${TARGET_ONLY_LOSS:-1}"
# Cross-gate bias at init. The default DEPENDS ON THE SHIFT, because
# what the gated cross path carries depends on it:
#   shift-2: the intra path already carries mod_b's own history, so the
#            cross path is auxiliary (it brings in mod_a). -10 keeps it
#            silent at step 0, which is the warm-start trick -- the model
#            starts as a clean single-stream LM over mod_b.
#   shift-1: the slot predicting b_k HOLDS a_k, so the intra path
#            (same-parity slots) carries mod_a, and mod_b's OWN HISTORY
#            moves onto the cross path along with the prefix. Leaving the
#            gate at -10 would start the model with no access to its own
#            autoregressive history at all. 0.0 -> sigmoid = 0.5, both
#            paths live from step 0.
# Override explicitly to pin a value regardless of shift.
if [[ -z "${GATE_INIT_BIAS:-}" ]]; then
    if [[ "$SUFFIX_SHIFT1" == "1" ]]; then GATE_INIT_BIAS=0.0
    else                                        GATE_INIT_BIAS=-10.0; fi
fi
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
# Key the warm-start file on the gate bias (m=minus, p=point) so a
# different bias builds its own init instead of silently reusing another.
GATE_TAG=$(echo "$GATE_INIT_BIAS" | sed 's/-/m/g; s/\./p/g')
INIT_CKPT="${INIT_CKPT:-ckpt/m2c_duet_rehearsal_init/m2c_duet_rehearsal_init_gate${GATE_TAG}.ckpt}"
mkdir -p "$(dirname "$INIT_CKPT")"

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
echo "Warm start: $INIT_CKPT"
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
        echo "       (it was built with gate_init_bias=$GATE_INIT_BIAS -- the"
        echo "        filename is keyed on that, so this is the right file)"
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
