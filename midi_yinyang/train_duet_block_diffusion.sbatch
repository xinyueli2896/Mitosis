#!/bin/bash
# SLURM batch script: multi-GPU training of M2CDuetBlockDiffusion (A.3).
# DuetBlock with discrete-diffusion-style training at the two query
# slots: per item, per slot, a noise level k in {0..K} controls whether
# the slot is fed mask_*_emb (k=K) or the ground-truth frame embedding
# (k=0), with a learned k-embedding added on top. Trains the model to
# handle every combination of (k_m, k_c), so at inference either parallel
# diffusion or MaskGIT-style iterative refinement is valid.
#
# WARM-START AND AUTO-RESUME
#   Three-tier startup logic (run again with no flags to keep training):
#
#   1. If CKPT is explicitly set, full Lightning resume from that ckpt.
#   2. Else if the default run directory already contains last.ckpt
#      (i.e. a prior run of this same sbatch), AUTO-RESUME from it.
#      LR_TOTAL_STEPS auto-bumps to (ckpt's global_step + 50000) so the
#      OneCycle schedule extends and the next run does another ~50k steps.
#   3. Else cold-start: init from the base single-stream CP-transformer
#      pretrained ckpt via init_pretrained_into_duet_block_diffusion.py
#      (same pattern as A.1 / A.2 / C.1 / C.2). The init copies the
#      pretrained Q/K/V/O into the per-modality slots and zero-inits the
#      A.3-specific extras (mask_m_emb, mask_c_emb, k_emb_m, k_emb_c, gates).
#
#   We deliberately do NOT warm-start from an A.2 ckpt: A.2 was trained
#   only in the fully-masked regime, and warm-starting from it would
#   inherit the query-slot collapse that motivated A.3 in the first
#   place. The neutral pretrained init lets the diffusion objective
#   train the query-slot pathway across the full noise spectrum.
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
#   --export=ALL,TASK=...   drumnondrum (default) or melchord.
#   --export=ALL,CKPT=...   explicit override for the resume ckpt. If
#                           unset, the script auto-detects the run's
#                           last.ckpt. Full Lightning resume preserves
#                           optimizer momentum and scheduler state.
#   --export=ALL,LR_TOTAL_STEPS=N    total steps the OneCycle schedule
#                           spans. Default: 50000 for a cold start,
#                           (prior_step + 50000) on auto-resume. When
#                           continuing manually, set this LARGER than the
#                           prior run's value -- Lightning re-creates the
#                           scheduler with the new total_steps and loads
#                           the old last_epoch on top, so the LR keeps
#                           annealing smoothly instead of restarting.
#   --export=ALL,DIFFUSION_K=8       number of noise-level bins (default 4).
#   --export=ALL,MOE_MODALITY_BIAS=1 A.2.moe_improved: learned per-modality
#                           bias [2, E] on the MoE router logits, zero-init.
#                           Hands the router the slot-parity bit the
#                           per-modality attention projections already
#                           imprint (probes 178945/178946: ~69% of routing
#                           separation was that stamp), freeing the
#                           input-driven pathway for within-modality
#                           structure. Run dirs get an 'mb' marker after
#                           the K tag so plain and bias runs never
#                           auto-resume into each other; a mismatched
#                           explicit CKPT fails loudly at the strict load.
#                           Verify after training with
#                           analyze_moe_routing.sbatch PROBE=identical:
#                           the content-pathway stamp share should be
#                           near zero. OUTCOME (probes 179683/179684):
#                           offered, not used. Kept for ablations.
#   --export=ALL,MOE_MODALITY_GATES=1  A.2.moe_permod: per-modality
#                           router matrices gate_m/gate_c replacing the
#                           shared gate (the q_m/q_c move applied to the
#                           router). Each gate only scores its own
#                           stream, so the parity stamp cannot influence
#                           within-stream routing; the expert pool stays
#                           fully shared and UNASSIGNED -- which experts
#                           each stream uses, and whether any serves
#                           both (an integrator), is learned. Read the
#                           outcome off analyze_moe_routing's purity
#                           tables. Run dirs get an 'mg' marker after
#                           the K tag; warm start seeds both gates from
#                           the shared gate so step 0 matches the
#                           shared-router init exactly.
#   --export=ALL,MOE_MODALITY_HARD_ROUTE=1  A.2.moe_hardroute: DISJOINT
#                           expert pools -- mod_a may only reach experts
#                           [0, E/2), mod_b only [E/2, E). The imposed-
#                           separation control (MoMa / VL-MoE style) the
#                           learned gates are argued against: same
#                           parameters and activated compute, but an
#                           integrator expert serving both streams is no
#                           longer representable. Load balancing is
#                           computed within each pool so the arm is not
#                           penalised for its own architecture. Needs an
#                           even MOE_NUM_EXPERTS and MOE_TOPK <= E/2 (at
#                           the E=4/topk=2 default each stream uses BOTH
#                           of its experts every token -- routing within
#                           a pool is then weighting, not selection; run
#                           MOE_NUM_EXPERTS=8 if you want sparsity
#                           preserved inside the pools, at 2x expert
#                           parameters). Run dirs get an 'hr' marker.
#                           Expert purity is 0/100 BY CONSTRUCTION on
#                           this arm -- report within-pool content
#                           responsiveness and quality instead.
#   --export=ALL,TOKEN_LEVEL_MASK=1  A.4: per-token absorbing corruption
#                           of the query-slot frame -- at commitment
#                           level k each non-pad token is masked with
#                           prob k/K and the local encoder embeds the
#                           partial frame, so intermediate k are
#                           genuinely intermediate ("chord root known,
#                           upper voices open"). Endpoints match the
#                           plain variant exactly (all-masked ->
#                           mask_*_emb; k=0 -> clean), so shared/mg
#                           ckpts warm-start cleanly. Uses the free
#                           token id n_normal_tokens-1 (no vocab
#                           change); melchord only (velocity vocab has
#                           no free id). Run dirs get a 'tk' marker;
#                           inference auto-detects and turns on
#                           confidence-based per-token re-masking
#                           across refinement rounds (A3_TOKEN_REMASK=0
#                           to disable).
#   --export=ALL,FRESH_SCHEDULE=1    opt-in: load weights only, drop
#                           optimizer + scheduler + step counter, run a
#                           brand-new OneCycle from step 0. NOT the
#                           default -- only use if you want to change K,
#                           query_loss_weight, or other things that the
#                           optimizer state has baked in assumptions about.
#   --export=ALL,MAX_LR=...          override LR (else autoscale by NGPUS).
#   --export=ALL,BATCH_SIZE=...      per-GPU batch size (default 4).
#   --export=ALL,GATE_INIT_BIAS=-4   raise if gates stay at init.
#   --export=ALL,QUERY_LOSS_WEIGHT=2.0  bump query-slot CE weight.
#   --export=ALL,DUMP_SAMPLES_DIR=path  dump first batch to .mid.
#
# Examples:
#   sbatch midi_yinyang/train_duet_block_diffusion.sbatch
#   sbatch --export=ALL,DIFFUSION_K=8 midi_yinyang/train_duet_block_diffusion.sbatch
#   sbatch --export=ALL,CKPT=ckpt/m2c_duet_block_diffusion_.../last.ckpt \
#          midi_yinyang/train_duet_block_diffusion.sbatch

#SBATCH --job-name=mitosis_duet_block_diffusion
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

PRETRAINED=ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt
# MoE configuration (defaults = the standard 4-expert top-2 setup).
# For the E4c MoE ablation: MOE_NUM_EXPERTS=1 MOE_TOPK=1 with
# MOE_INTERMEDIATE_SIZE sized to match the activated compute of the
# standard config, plus RUN_TAG to keep the run directory distinct.
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_TOPK="${MOE_TOPK:-2}"
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-}"
RUN_TAG="${RUN_TAG:-}"
# Init ckpt path is K-dependent so different K values don't overwrite
# each other (the k_emb_* tables have different shapes); non-default MoE
# configs get their own suffix for the same reason.
DIFFUSION_K="${DIFFUSION_K:-4}"
MOE_SUFFIX=""
if [[ "$MOE_NUM_EXPERTS" != "4" || "$MOE_TOPK" != "2" || -n "$MOE_INTERMEDIATE_SIZE" ]]; then
    MOE_SUFFIX="_moe${MOE_NUM_EXPERTS}t${MOE_TOPK}${MOE_INTERMEDIATE_SIZE:+i$MOE_INTERMEDIATE_SIZE}"
fi
INIT_CKPT="ckpt/m2c_duet_block_diffusion_init_K${DIFFUSION_K}${MOE_SUFFIX}/m2c_duet_block_diffusion_init_K${DIFFUSION_K}${MOE_SUFFIX}.ckpt"

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
MAX_LR="${MAX_LR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:--10.0}"
# v1.1 default 2.0: the query pathway gets ~T_full x fewer supervised
# positions than the AR stream; upweighting rebalances. Watch
# val_ar_loss for regression; drop back toward 1.0 if it degrades.
QUERY_LOSS_WEIGHT="${QUERY_LOSS_WEIGHT:-2.0}"
SELF_COND_PROB="${SELF_COND_PROB:-0.5}"
# v1.2 scheme: 1 = time-aligned RoPE (rotary index = physical // 2, so
# m_t and c_t share rotary position t -- restores the pretrain
# positional geometry; candidate fix for the long-term-structure
# deficit). The scheme version is stamped into the run dir name
# (v1.2 vs v1.1), so runs never intermingle. Default 0 = v1.1.
TIME_ROPE_ALIGNED="${TIME_ROPE_ALIGNED:-0}"
# A.2.moe_improved: per-modality bias on the MoE router logits (see
# header). Marks the run dir with 'mb' after the K tag.
MOE_MODALITY_BIAS="${MOE_MODALITY_BIAS:-0}"
# A.2.moe_permod: per-modality router matrices over a fully shared,
# unassigned expert pool (see header). Marks the run dir with 'mg'.
MOE_MODALITY_GATES="${MOE_MODALITY_GATES:-0}"
# A.2.moe_hardroute: disjoint per-modality expert pools (see header).
# Marks the run dir with 'hr'.
MOE_MODALITY_HARD_ROUTE="${MOE_MODALITY_HARD_ROUTE:-0}"
# A.4: token-level masking at the query slots (see header). Marks the
# run dir with 'tk'.
TOKEN_LEVEL_MASK="${TOKEN_LEVEL_MASK:-0}"
TASK="${TASK:-drumnondrum}"
DUMP_SAMPLES_DIR="${DUMP_SAMPLES_DIR:-temp/train_samples_duet_block_diffusion_${SLURM_JOB_ID}}"
# Data polyphony per stream (drives the sample-dump reshape, not the model):
# drumnondrum data is cp16, POP909/Nottingham melchord data is cp8
# (cp4 retired -- seventh chords are 5 notes and cp4 dropped the seventh).
if [[ -z "${MAX_POLYPHONY:-}" ]]; then
    case "$TASK" in
        # melchord* -- every variant INCLUDING the _rev tasks, which
        # only swap mod_a/mod_b roles and read the SAME cp8 files.
        # Must stay in step with the melchord* pattern that exports
        # MELCHORD_CP below: an explicit list here missed _rev, so a
        # reverse run took MAX_POLYPHONY=16 and then exported
        # MELCHORD_CP=16, sending tasks.py to cp16 data paths.
        melchord*) MAX_POLYPHONY=8 ;;
        *)        MAX_POLYPHONY=16 ;;
    esac
fi
# tasks.py resolves melchord dataset paths from MELCHORD_CP; keep it in
# lockstep with the decode budget so data files and sample dump agree.
case "$TASK" in melchord*) export MELCHORD_CP="$MAX_POLYPHONY" ;; esac

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"

mkdir -p /home/xinyue.li/logs

# --- detect GPU count --------------------------------------------------
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
# Scheme version must match the python side's default_name derivation
# (v1.2 when TIME_ROPE_ALIGNED=1, else v1.1) or auto-resume looks in
# the wrong directory.
if [[ "$TIME_ROPE_ALIGNED" == "1" ]]; then SCHEME_VERSION=v1.2; else SCHEME_VERSION=v1.1; fi
# 'mb'/'mg' markers must match the python side's default_name derivation
# (mb_tag concatenates in that order).
MB_TAG=""
[[ "$MOE_MODALITY_BIAS" == "1" ]] && MB_TAG="${MB_TAG}mb"
[[ "$MOE_MODALITY_GATES" == "1" ]] && MB_TAG="${MB_TAG}mg"
[[ "$MOE_MODALITY_HARD_ROUTE" == "1" ]] && MB_TAG="${MB_TAG}hr"
[[ "$TOKEN_LEVEL_MASK" == "1" ]] && MB_TAG="${MB_TAG}tk"
RUN_DIR="ckpt/m2c_duet_block_diffusion_${SCHEME_VERSION}_large_gnl12_K${DIFFUSION_K}${MB_TAG}_${TASK}${RUN_TAG:+_$RUN_TAG}_batch_${GLOBAL_BATCH}_schedule"
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
# Cold start -> 50000.
# Resume (CKPT set) and no explicit LR_TOTAL_STEPS -> read the ckpt's
#   global_step and bump by 50000 so we run another ~50k steps under
#   an extended OneCycle.
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
echo "Variant: m2c_duet_block_diffusion (A.3)  K=$DIFFUSION_K  moe_modality_bias=$MOE_MODALITY_BIAS  moe_modality_gates=$MOE_MODALITY_GATES  moe_modality_hard_route=$MOE_MODALITY_HARD_ROUTE  token_level_mask=$TOKEN_LEVEL_MASK${MB_TAG:+  [$MB_TAG]}"
echo "MoE: experts=$MOE_NUM_EXPERTS topk=$MOE_TOPK intermediate=${MOE_INTERMEDIATE_SIZE:-<default 3072/expert>}  RUN_TAG=${RUN_TAG:-<none>}"
echo "Task:    $TASK   (data max_polyphony: $MAX_POLYPHONY)"
echo "Detected GPUs: $NGPUS   Per-GPU batch: $BATCH_SIZE   Global: $((BATCH_SIZE * NGPUS))"
echo "Effective max_lr: $MAX_LR   gate_init_bias: $GATE_INIT_BIAS   query_loss_weight: $QUERY_LOSS_WEIGHT"
if [[ -n "$CKPT" ]]; then
    if [[ "$FRESH_SCHEDULE" == "1" ]]; then
        echo "Mode: CONTINUE (load weights from CKPT=$CKPT, fresh LR schedule)"
    else
        echo "Mode: RESUME (full Lightning resume from CKPT=$CKPT)"
    fi
else
    echo "Mode: COLD START (init from PRETRAINED -> INIT_CKPT=$INIT_CKPT)"
fi
echo "LR_TOTAL_STEPS=$LR_TOTAL_STEPS   VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL   RUN_DIR=$RUN_DIR"
echo "----------------------------------------------------------------"
nvidia-smi || true
echo "================================================================"

# --- step 1: warm-start init -----------------------------------------
if [[ -z "$CKPT" ]]; then
    if [[ -f "$INIT_CKPT" ]]; then
        echo "[init] $INIT_CKPT already exists; skipping warm-start step."
    else
        if [[ ! -f "$PRETRAINED" ]]; then
            echo "[init] ERROR: pretrained ckpt not found at $PRETRAINED"
            exit 1
        fi
        echo "[init] running init_pretrained_into_duet_block_diffusion.py"
        INIT_EXTRA=()
        if [[ -n "$MOE_INTERMEDIATE_SIZE" ]]; then
            INIT_EXTRA+=(--moe_intermediate_size "$MOE_INTERMEDIATE_SIZE")
        fi
        python init_pretrained_into_duet_block_diffusion.py \
            --pretrained "$PRETRAINED" \
            --output     "$INIT_CKPT" \
            --model_size large --global_num_layers 12 \
            --moe_num_experts "$MOE_NUM_EXPERTS" --moe_topk "$MOE_TOPK" \
            --gate_init_bias "$GATE_INIT_BIAS" \
            --query_loss_weight "$QUERY_LOSS_WEIGHT" \
            --diffusion_K "$DIFFUSION_K" \
            "${INIT_EXTRA[@]}"
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

# --- step 2: training -------------------------------------------------
# Pre-flight: fail fast with a clear message if the allocated GPUs
# are visible but unusable (bad node) -- otherwise the job dies deep
# inside Lightning device setup with a cryptic INTERNAL ASSERT.
python check_gpu_health.py --expect "$NGPUS" || exit 1

export MASTER_PORT=$((20000 + RANDOM % 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_TRAIN_ARGS=()
if [[ -n "$MOE_INTERMEDIATE_SIZE" ]]; then
    EXTRA_TRAIN_ARGS+=(--moe_intermediate_size "$MOE_INTERMEDIATE_SIZE")
fi
if [[ -n "$RUN_TAG" ]]; then
    EXTRA_TRAIN_ARGS+=(--run_tag "$RUN_TAG")
fi
if [[ "$FRESH_SCHEDULE" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--fresh_schedule)
fi

COMMON_ARGS=(
    --task "$TASK"
    --batch_size "$BATCH_SIZE"
    --model_size large
    --moe_num_experts "$MOE_NUM_EXPERTS" --moe_topk "$MOE_TOPK"
    --max_lr "$MAX_LR"
    --lr_total_steps "$LR_TOTAL_STEPS" \
    --val_check_interval "$VAL_CHECK_INTERVAL"
    --gradient_clip_val 1.0
    --aux_loss_weight 0.01
    --moe_monitor_every_n_steps 500
    --dump_samples_dir "$DUMP_SAMPLES_DIR"
    --gate_init_bias "$GATE_INIT_BIAS"
    --max_polyphony "$MAX_POLYPHONY"
    --query_loss_weight "$QUERY_LOSS_WEIGHT"
    --self_cond_prob "$SELF_COND_PROB"
    --time_rope_aligned "$TIME_ROPE_ALIGNED"
    --moe_modality_bias "$MOE_MODALITY_BIAS"
    --moe_modality_gates "$MOE_MODALITY_GATES"
    --moe_modality_hard_route "$MOE_MODALITY_HARD_ROUTE"
    --token_level_mask "$TOKEN_LEVEL_MASK"
    --diffusion_K "$DIFFUSION_K"
    --checkpoint_path "$TRAIN_CKPT"
    --wandb
)

if [[ "$NGPUS" -le 1 ]]; then
    python cp_transformer_m2c_duet_block_diffusion.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
        cp_transformer_m2c_duet_block_diffusion.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"
fi

echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
