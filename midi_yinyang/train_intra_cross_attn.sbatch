#!/bin/bash
# SLURM batch script: multi-GPU training of M2CIntraCrossAttn on cscc-gpu.
# Mirrors train_slurm.sbatch but targets the intra-cross-attn variant
# (per-modality Q/K/V/O + two-pass intra/cross attention + per-block gates +
# shared MoE FFN). ~15-25% slower per step than jointattn due to the second
# SDPA per block; matched LR/aux/silence-aug defaults.
#
# Knobs (all optional):
#   --gres=gpu:N            how many GPUs to allocate (1..4).
#   --export=ALL,TASK=...   task name: drumnondrum (default) or melchord.
#                           Resolves mod_a/mod_b dataset paths + labels.
#   --export=ALL,CKPT=...   if set, skip init + resume from that ckpt.
#   --export=ALL,MAX_LR=..  override the LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=12 per-GPU batch size (default 8).
#   --export=ALL,GATE_INIT_BIAS=-4  raise if gates stay pinned at init.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first training batch as .mid
#                           into this dir for sanity-check listening.
#                           Default: temp/train_samples_<JOBID>.
#
# Examples:
#   sbatch midi_yinyang/train_intra_cross_attn.sbatch
#   sbatch --export=ALL,TASK=melchord midi_yinyang/train_intra_cross_attn.sbatch
#   sbatch --gres=gpu:2 midi_yinyang/train_intra_cross_attn.sbatch
#   sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt midi_yinyang/train_intra_cross_attn.sbatch

#SBATCH --job-name=mitosis_intra_cross_attn
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
# Node max is 256 GB but SLURM/kernel reserve ~16 GB, so 256G is
# rejected with "Memory specification can not be satisfied". 240G is
# safely under the cap. With mmap'd data tensors (see FramedDataset
# _robust_load mmap=True path), aggregate RAM is dominated by model
# state + activations rather than data duplication, so 240G is more
# than enough now.
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

# --- environment ---------------------------------------------------------
set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt

MAX_LR="${MAX_LR:-}"
# Default per-GPU batch size 4. intra-cross-attn's two SDPA passes per
# block roughly double attention activation memory vs jointattn, so 8
# OOMs on 40GB A100s with the size-large + 12-layer config. Override
# via --export=ALL,BATCH_SIZE=2 if you still OOM, or BATCH_SIZE=6 to
# push if you have memory headroom.
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
# OneCycle horizon. Was hardcoded at 200000, while A.2
# (train_duet_block_diffusion.sbatch) uses 50000 -- so at a given step
# A.1 sat far earlier in its cycle, at a much higher LR, than the model
# it is meant to be an ablation OF. The A.1-vs-A.2 comparison is only
# about the 同步看 (simultaneous mutual conditioning) mechanism if the
# optimisation matches, so default to A.2's horizon and allow override.
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-50000}"
# Steps between val evaluations. On the ~1.9k-song combined melchord
# corpus the val minimum arrives within the first ~1k steps, so the
# stock 500 resolves it with one or two points; lower it to place the
# checkpoint accurately.
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
# Anti-exposure-bias augmentation for the sparse stream: prob per
# frame that a silent run of CTX_CORRUPT_LEN frames begins in the acc
# INPUT context while CE targets stay clean. 0 = off (default,
# behaviour unchanged). See the comment in cp_transformer_m2c_moe.loss.
CTX_CORRUPT_PROB="${CTX_CORRUPT_PROB:-0.0}"
CTX_CORRUPT_LEN="${CTX_CORRUPT_LEN:-8}"
# D.1 scheme: 1 = time-aligned RoPE (m_t and c_t share rotary position
# t, restoring the pretrain positional geometry). Baked into the ckpt;
# use RUN_TAG (e.g. tar) so the run dir stays distinct from the legacy
# parity-RoPE run. Default 0 = legacy.
TIME_ROPE_ALIGNED="${TIME_ROPE_ALIGNED:-0}"
# Optional run-name suffix. The run directory is derived from variant +
# task + global batch, so two runs of the SAME task land in the SAME
# dir and their checkpoints intermingle. That is silently dangerous:
# resolve_best_ckpt picks the global val_loss minimum across the dir,
# so a superseded run with a lower val_loss (e.g. one that overfit
# under a wrong LR horizon) would be selected for evaluation instead
# of the run you meant. Set RUN_TAG when re-running a task under
# changed settings, or move the old dir aside first.
RUN_TAG="${RUN_TAG:-}"
# Task config: melchord (mod_a=mel, mod_b=chord) or drumnondrum
# (mod_a=drum, mod_b=nondrum). Dataset paths and display labels are
# resolved from tasks.py based on this name. Override with
# --export=ALL,TASK=melchord on the sbatch line.
TASK="${TASK:-drumnondrum}"
# Data max_polyphony, auto-set from TASK (override with MAX_POLYPHONY).
# The melchord datasets are cp8 (8 polyphony slots -> 16-token frames;
# cp4 was retired -- seventh chords render as 5 notes and cp4 dropped
# the seventh); everything else is cp16. The trainer defaults this to
# 16, and the --dump_samples_dir callback below decodes real frames
# with it, so leaving it unset silently mis-decodes -- or crashes --
# the sample dump on any non-cp16 task. train_duet_block_diffusion.sbatch
# and train_duet_anticipatory.sbatch already derive it the same way.
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
# Optional: dump the first training batch as .mid files into this dir at
# train start. Useful for sanity-check listening before letting training
# run for hours. Default empty = off.
DUMP_SAMPLES_DIR="${DUMP_SAMPLES_DIR:-temp/train_samples_${SLURM_JOB_ID}}"

# Resume default is per-TASK. It used to hard-code the drumnondrum run's
# last.ckpt for every task, so submitting TASK=melchord* without CKPT
# silently RESUMED a drumnondrum run: wrong-domain weights plus a
# restored optimizer/LR-scheduler state from another schedule. The run
# dir embeds task.name, so nothing was overwritten -- it just quietly
# produced a model that is not what the run name claims, and would break
# the A.1-vs-A.2 ablation, which requires both to warm-start identically
# from the pretrained single-stream ckpt. train_duet_block_diffusion and
# train_duet_anticipatory both default to empty (fresh init); match them
# for every task except the drumnondrum run this default was written for.
if [[ -z "${CKPT+x}" ]]; then
    case "$TASK" in
        drumnondrum) CKPT=ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_drumnondrum_batch_16_schedule/last.ckpt ;;
        *)           CKPT= ;;
    esac
fi

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

# --- detect GPU count --------------------------------------------------
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NGPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
NGPUS="${NGPUS:-1}"

# sqrt-scaled LR by NGPUS (base 5e-5 @ 1 GPU). Override via MAX_LR=...
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
echo "Variant: m2c_intra_cross_attn  (per-block gated intra+cross attn)"
echo "Task:    $TASK   (data max_polyphony: $MAX_POLYPHONY)"
echo "Detected GPUs: $NGPUS   Per-GPU batch: $BATCH_SIZE   Global: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR   gate_init_bias: $GATE_INIT_BIAS"
echo "LR schedule: OneCycle over $LR_TOTAL_STEPS steps   val every $VAL_CHECK_INTERVAL"
echo "Run tag: ${RUN_TAG:-<none -- shares the default run dir for this task>}"
if [[ -n "$CKPT" ]]; then
    echo "Mode: RESUME from CKPT=$CKPT (skip init)"
else
    echo "Mode: FRESH (init from PRETRAINED=$PRETRAINED -> INIT_CKPT=$INIT_CKPT)"
fi
echo "----------------------------------------------------------------"
nvidia-smi || true
echo "================================================================"

# --- step 1: warm-start init (only when no CKPT given) -----------------
if [[ -z "$CKPT" ]]; then
    if [[ -f "$INIT_CKPT" ]]; then
        echo "[init] $INIT_CKPT already exists; skipping warm-start step."
    else
        if [[ ! -f "$PRETRAINED" ]]; then
            echo "[init] ERROR: pretrained ckpt not found at $PRETRAINED"
            exit 1
        fi
        echo "[init] running init_pretrained_into_intra_cross_attn.py"
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

# --- step 2: training ---------------------------------------------------
# Pre-flight: fail fast with a clear message if the allocated GPUs
# are visible but unusable (bad node) -- otherwise the job dies deep
# inside Lightning device setup with a cryptic INTERNAL ASSERT.
python check_gpu_health.py --expect "$NGPUS" || exit 1

export MASTER_PORT=$((20000 + RANDOM % 10000))
# Allocator anti-fragmentation. intra-cross-attn does 2 SDPA passes per
# block, so attention activations can fragment the heap more than usual.
# Cheap to enable; helps avoid spurious OOMs near the memory ceiling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Loss weights (mel/acc/eos) all default to 1.0 in the training script,
# i.e. unweighted CE -- val_loss is comparable across runs.
if [[ "$NGPUS" -le 1 ]]; then
    python cp_transformer_m2c_intra_cross_attn.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --ctx_corrupt_prob "$CTX_CORRUPT_PROB" \
        --ctx_corrupt_len "$CTX_CORRUPT_LEN" \
        --time_rope_aligned "$TIME_ROPE_ALIGNED" \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 \
        --dump_samples_dir "$DUMP_SAMPLES_DIR" \
        --gate_init_bias "$GATE_INIT_BIAS" \
        --fresh_schedule \
        --checkpoint_path "$TRAIN_CKPT" \
        --max_polyphony "$MAX_POLYPHONY" \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} \
        --wandb
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_intra_cross_attn.py \
        --task "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" \
        --lr_total_steps "$LR_TOTAL_STEPS" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --ctx_corrupt_prob "$CTX_CORRUPT_PROB" \
        --ctx_corrupt_len "$CTX_CORRUPT_LEN" \
        --time_rope_aligned "$TIME_ROPE_ALIGNED" \
        --gradient_clip_val 1.0 \
        --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 \
        --dump_samples_dir "$DUMP_SAMPLES_DIR" \
        --gate_init_bias "$GATE_INIT_BIAS" \
        --fresh_schedule \
        --checkpoint_path "$TRAIN_CKPT" \
        --max_polyphony "$MAX_POLYPHONY" \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} \
        --wandb
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
