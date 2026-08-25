#!/bin/bash
# SLURM: train OUR OWN conditional YinYang (Y-909) on POP909 melody/chord,
# one direction per submission.
#
# What this is: the published YinYang recipe (arXiv:2506.15548,
# cp_transformer_yinyang.py) applied to POP909 -- the FROZEN pretrained
# CP transformer run twice (one pass fully encodes the conditioning
# stream, one pass generates the target AR) with low-rank cross-attn
# adapters every n_skip layers + LoRA on Q/V as the only trainable
# parts. Until now every melchord Y comparison used the authors' Y-mc
# ckpt, trained on NOTTINGHAM -- the domain caveat in EXPERIMENTS.md.
# This produces the POP909-matched Y ("option (a)" there): same
# adapter recipe, same defaults (mask 0.25/10, LoRA r16, n_skip 2),
# our data.
#
# DIRECTION semantics (inherited from the '_chord_mel' naming that the
# inference driver and yinyang_e3_driver auto-detect from the ckpt
# filename -- do not rename):
#   chord2mel : train_task = <DATASET_NAME>           (no _rev)
#   mel2chord : train_task = <DATASET_NAME>_rev
# Conditional models are DIRECTION-LOCKED at training time: train one
# checkpoint per direction, two submissions.
#
# Data: data/<DATASET_NAME>.pt from preprocess_pop909_yinyang.sbatch
# (paired within-frame format; NOT the duet family's two-file datasets).
# Train/val split is FramedDataset's internal 90/10; E3 test prompts
# come from the held-out prompt folders as usual, not from this file.
#
# Checkpoints land in
#   ckpt/cp_transformer_yinyang_v5.1_lora_batch_<B*G>_<train_task>_mask<p>-<l>-step1/
# with val_loss-tagged filenames (resolve_best_ckpt-compatible) plus
# .epoch=best.ckpt / .epoch=last.ckpt copies next to the dir.
#
# Env knobs (--export=ALL,KEY=val):
#   DIRECTION      chord2mel (default) | mel2chord
#   DATASET_NAME   default pop909_cp8_v2_chord_mel
#   BATCH_SIZE     per-GPU, default 4 (two full-model forwards per step;
#                  raise if memory allows)
#   LR             default 1e-4 (published default)
#   MASK_PROB      default 0.25    MASK_LENGTH default 10
#   N_SKIP         default 2 (adapter every 2 layers)
#   TRAIN_LENGTH   default 384 frames
#   EARLY_STOP     default 20 -- val checks (every 500 steps) without
#                  improvement before stopping. POP909 is small; the
#                  upstream default (no early stop, 60k steps) will
#                  overfit long past the minimum. Best ckpt is saved by
#                  val_loss either way.
#   WEIGHTS        optional: resume/continue from a prior yinyang ckpt.
#
# Examples:
#   sbatch --exclude=gpu-50 midi_yinyang/train_yinyang_pop909.sbatch
#   sbatch --exclude=gpu-50 --export=ALL,DIRECTION=mel2chord \
#       midi_yinyang/train_yinyang_pop909.sbatch

#SBATCH --job-name=mitosis_yinyang_909
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
REPO_DIR=/home/xinyue.li/Mitosis

DIRECTION="${DIRECTION:-chord2mel}"
DATASET_NAME="${DATASET_NAME:-pop909_cp8_v2_chord_mel}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
MASK_PROB="${MASK_PROB:-0.25}"
MASK_LENGTH="${MASK_LENGTH:-10}"
N_SKIP="${N_SKIP:-2}"
TRAIN_LENGTH="${TRAIN_LENGTH:-384}"
EARLY_STOP="${EARLY_STOP:-20}"
# Optional override for the frozen base ckpt; unset -> the trainer's own
# default (its filename spelling differs from the duet sbatches', so we
# do not re-hardcode it here).
WEIGHTS="${WEIGHTS:-}"
PRETRAINED="${PRETRAINED:-}"

case "$DIRECTION" in
    chord2mel) TRAIN_TASK="$DATASET_NAME" ;;
    mel2chord) TRAIN_TASK="${DATASET_NAME}_rev" ;;
    *) echo "ERROR: DIRECTION must be chord2mel or mel2chord (got '$DIRECTION')"; exit 1 ;;
esac

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

[[ -f "data/${DATASET_NAME}.pt" ]] || {
    echo "ERROR: data/${DATASET_NAME}.pt not found."
    echo "Build it first: sbatch --exclude=cn-20 midi_yinyang/preprocess_pop909_yinyang.sbatch"
    exit 1
}

NGPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)  GPUs=$NGPUS"
echo "Y-909: conditional YinYang finetune on POP909 melody/chord"
echo "DIRECTION=$DIRECTION  ->  train_task=$TRAIN_TASK"
echo "DATA=data/${DATASET_NAME}.pt   BATCH=${BATCH_SIZE}x${NGPUS}  LR=$LR"
echo "mask=${MASK_PROB}/${MASK_LENGTH}  n_skip=$N_SKIP  train_length=$TRAIN_LENGTH  early_stop=$EARLY_STOP val-checks"
echo "================================================================"
nvidia-smi || true

python check_gpu_health.py --expect "$NGPUS" || exit 1

# Log to wandb (project MusicMOE, same as the duet runs); the mitosis
# env has no tensorboard. Set WANDB=0 to fall back to a CSV logger.
WANDB="${WANDB:-1}"
export YINYANG_WANDB="$WANDB"
export WANDB_PROJECT="${WANDB_PROJECT:-MusicMOE}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"

EXTRA=()
[[ -n "$WEIGHTS" ]] && EXTRA+=(--weights_path "$WEIGHTS")
[[ -n "$PRETRAINED" ]] && EXTRA+=(--fp_path "$PRETRAINED")

python cp_transformer_yinyang.py \
    --batch_size "$BATCH_SIZE" \
    --dataset_name "$DATASET_NAME" \
    --train_task "$TRAIN_TASK" \
    --mask_prob "$MASK_PROB" \
    --mask_length "$MASK_LENGTH" \
    --lr "$LR" \
    --n_skip "$N_SKIP" \
    --train_length "$TRAIN_LENGTH" \
    --early_stopping_patience "$EARLY_STOP" \
    "${EXTRA[@]}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local} finished at $(date)"
echo "Eval under the E3 protocol with eval_yinyang.sbatch:"
echo "  CKPT_Y=<the new ckpt>  TASK=melchord  DIRECTION=$DIRECTION"
echo "================================================================"
