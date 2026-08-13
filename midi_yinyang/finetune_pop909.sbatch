#!/bin/bash
# SLURM: finetune S0 (pretrained single-stream CP transformer) on POP909
# melody+chord IN THE UPSTREAM midi-function-alignment REPO, 4 GPUs,
# wandb logging. cp_transformer.py is not modified; the finetune driver
# (finetune_pop909.py) is copied from this Mitosis checkout into the
# upstream repo automatically before launch.
#
# Env knobs (override via --export=ALL,KEY=val or shell-prefix + --export=ALL):
#   UPSTREAM_DIR    directory containing cp_transformer.py + ckpt/ + data/.
#                   default: /home/xinyue.li/Mitosis/midi_yinyang
#                   (set to another checkout, e.g. a separate
#                   midi-function-alignment clone, to run there instead)
#   MITOSIS_DIR     where finetune_pop909.py lives (copied over only when
#                   UPSTREAM_DIR differs).
#                   default: /home/xinyue.li/Mitosis/midi_yinyang
#   DATA            default data/pop909_melchord_cp16_v2.pt (relative to
#                   UPSTREAM_DIR; auto-copied from MITOSIS_DIR/data if absent)
#   PRETRAINED      default ckpt/cp_transformer_v0.42_..._schedule.epoch.00.fin.ckpt
#   RESUME          prior finetune last.ckpt; auto-detected if unset
#   MAX_LR          default 1e-5
#   LR_TOTAL_STEPS  default 2000
#   VAL_CHECK_INTERVAL  default 25 (steps between val evals)
#       Defaults retuned after the first 20k-step run overfit instantly:
#       POP909 train is ~20 steps/epoch at global batch 36-48, so the
#       val minimum sits within the first ~1-2 epochs -- a 250-step val
#       interval never even measures it. Short schedule + low LR +
#       dense validation captures the dip.
#   BATCH_SIZE      per-GPU, default 12. NOW 2 GPUs by default, so the
#                   global batch is 24 (was 48 on 4 GPUs). The run dir
#                   encodes the global batch, so a 2-GPU run lands in a
#                   DIFFERENT dir than a prior 4-GPU one (no auto-resume
#                   across that change). Pass BATCH_SIZE=24 to keep the
#                   old global batch of 48 on 2 GPUs, memory permitting.
#   RUN_TAG         optional run-name suffix
#   WANDB_PROJECT   default MusicMOE
#
# Examples:
#   sbatch midi_yinyang/finetune_pop909.sbatch
#   WANDB_PROJECT=my-space sbatch --export=ALL midi_yinyang/finetune_pop909.sbatch

#SBATCH --job-name=cpft_pop909
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
UPSTREAM_DIR="${UPSTREAM_DIR:-/home/xinyue.li/Mitosis/midi_yinyang}"
MITOSIS_DIR="${MITOSIS_DIR:-/home/xinyue.li/Mitosis/midi_yinyang}"

DATA="${DATA:-data/pop909_melchord_cp16_v2.pt}"
PRETRAINED="${PRETRAINED:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
RESUME="${RESUME:-}"
MAX_LR="${MAX_LR:-1e-5}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-2000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-25}"
BATCH_SIZE="${BATCH_SIZE:-12}"
RUN_TAG="${RUN_TAG:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-MusicMOE}"

conda activate "$CONDA_ENV"
mkdir -p /home/xinyue.li/logs

if [[ ! -d "$UPSTREAM_DIR" ]]; then
    echo "ERROR: UPSTREAM_DIR=$UPSTREAM_DIR not found."
    exit 1
fi
cd "$UPSTREAM_DIR"

# keep the driver current when running in a separate checkout
if [[ "$(readlink -f "$UPSTREAM_DIR")" != "$(readlink -f "$MITOSIS_DIR")" ]]; then
    cp -f "$MITOSIS_DIR/finetune_pop909.py" .
    echo "[setup] copied finetune_pop909.py from $MITOSIS_DIR"
fi

# auto-copy the tokenized data from the Mitosis checkout if absent here
if [[ ! -f "$DATA" && -f "$MITOSIS_DIR/$DATA" ]]; then
    mkdir -p "$(dirname "$DATA")"
    cp "$MITOSIS_DIR/$DATA" "$(dirname "$DATA")/"
    for ext in .length.pt .pitch_shift_range.pt .txt; do
        src="$MITOSIS_DIR/${DATA%.pt}$ext"
        [[ -f "$src" ]] && cp "$src" "$(dirname "$DATA")/"
    done
    echo "[setup] copied $DATA (+sidecars) from $MITOSIS_DIR"
fi

# auto-resume if this exact run already has a last.ckpt
NGPUS="${SLURM_GPUS_ON_NODE:-2}"
TAG_SUFFIX=""; [[ -n "$RUN_TAG" ]] && TAG_SUFFIX="_${RUN_TAG}"
RUN_DIR="ckpt/cp_transformer_v0.42_size1_pop909ft${TAG_SUFFIX}_batch_$((BATCH_SIZE * NGPUS))_schedule"
if [[ -z "$RESUME" && -f "$RUN_DIR/last.ckpt" ]]; then
    RESUME="$RUN_DIR/last.ckpt"
    echo "[auto-resume] $RESUME"
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)  GPUs=$NGPUS"
echo "UPSTREAM_DIR=$UPSTREAM_DIR"
echo "DATA=$DATA"
echo "PRETRAINED=$PRETRAINED"
echo "RESUME=${RESUME:-<cold start from pretrained>}"
echo "MAX_LR=$MAX_LR  LR_TOTAL_STEPS=$LR_TOTAL_STEPS  BATCH=$BATCH_SIZE x $NGPUS"
echo "WANDB_PROJECT=$WANDB_PROJECT  RUN_DIR=$RUN_DIR"
echo "================================================================"
nvidia-smi | head -n 12 || true

if [[ ! -f "$DATA" ]]; then
    echo "ERROR: $DATA missing (run preprocess_pop909_melchord.sbatch first)"
    exit 1
fi
# FROM_SCRATCH=1: random init (S-scratch baseline) -- the pretrained
# ckpt is neither required nor loaded, and the run dir stem becomes
# pop909scratch so it can never be resolved in place of a finetune.
FROM_SCRATCH="${FROM_SCRATCH:-0}"
if [[ "$FROM_SCRATCH" != "1" && -z "$RESUME" && ! -f "$PRETRAINED" ]]; then
    echo "ERROR: $PRETRAINED missing"
    exit 1
fi

EXTRA_ARGS=()
[[ -n "$RESUME" ]] && EXTRA_ARGS+=(--resume "$RESUME")
[[ -n "$RUN_TAG" ]] && EXTRA_ARGS+=(--run_tag "$RUN_TAG")
[[ "$FROM_SCRATCH" == "1" ]] && EXTRA_ARGS+=(--from_scratch)

python finetune_pop909.py \
    --data "$DATA" \
    --pretrained "$PRETRAINED" \
    --batch_size "$BATCH_SIZE" \
    --max_lr "$MAX_LR" \
    --lr_total_steps "$LR_TOTAL_STEPS" \
    --val_check_interval "$VAL_CHECK_INTERVAL" \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Done. Checkpoints under $UPSTREAM_DIR/$RUN_DIR"
