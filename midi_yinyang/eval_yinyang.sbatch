#!/bin/bash
# SLURM: generate the published YinYang conditional baseline (Y-mc/Y-dn)
# under OUR E3 protocol, via yinyang_e3_driver.py.
#
# Checkpoints: upstream Google Drive (README of
# github.com/music-x-lab/midi-function-alignment), subfolders
# mel_to_chord / chord_to_mel / drum_to_others / others_to_drum; place
# them under midi_yinyang/ckpt/ keeping the subfolder, and prefer the
# 'prompt_lora' checkpoints per the upstream README. Pick the ckpt whose
# own direction matches DIRECTION -- the filename encodes it ('_rev'
# flips it); this script cannot verify that for you.
#
# Inputs are the TWO-STREAM source midis (both tracks in one file), the
# same songs as the rest of E3:
#   Y-mc : the held-out Nottingham two-track files (your manual holdout
#          folder) -- the domain where Y-mc is the matched system.
#   Y-dn : the unsplit RWC test songs.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_Y        ckpt path or name under ckpt/        (required)
#   TASK          melchord | drumnondrum               (required)
#   DIRECTION     mel2chord|chord2mel|drum2nondrum|nondrum2drum (required)
#   MIDI_FOLDER   folder of two-stream source midis    (required)
#   OUT           duet_multi output root (default temp/e3_y_<task>_<jobid>)
#   PROMPT_LENGTH default 64; GEN_LENGTH default 384; N_SAMPLES default 3
#
# Afterwards, score it alongside the internal systems by rerunning
# eval_e3.sbatch with SKIP_INFER=1 and Y_OUT=<this OUT> (adds the Y rows
# to the same table; BASELINE=Y then gives paired tests vs YinYang).

#SBATCH --job-name=mitosis_eval_yinyang
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=6:00:00

set -euo pipefail
source ~/.bashrc
CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

CKPT_Y="${CKPT_Y:-}"
TASK="${TASK:-}"
DIRECTION="${DIRECTION:-}"
MIDI_FOLDER="${MIDI_FOLDER:-}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
N_SAMPLES="${N_SAMPLES:-3}"
OUT="${OUT:-temp/e3_y_${TASK:-unset}_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

for v in CKPT_Y TASK DIRECTION MIDI_FOLDER; do
    [[ -n "${!v}" ]] || { echo "ERROR: $v is required"; exit 1; }
done
[[ -d "$MIDI_FOLDER" ]] || { echo "ERROR: $MIDI_FOLDER not found"; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "YINYANG BASELINE   task=$TASK  direction=$DIRECTION"
echo "CKPT_Y=$CKPT_Y"
echo "MIDI_FOLDER=$MIDI_FOLDER  OUT=$OUT"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "================================================================"

if ! python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.get_device_name(0)}')
"; then
    echo "ERROR: no usable CUDA GPU (submit via sbatch; --exclude bad nodes)."
    exit 1
fi

python yinyang_e3_driver.py \
    --ckpt "$CKPT_Y" \
    --midi-folder "$MIDI_FOLDER" \
    --task "$TASK" --direction "$DIRECTION" \
    --output-dir "$OUT" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples "$N_SAMPLES"

echo "================================================================"
echo "Y outputs -> $OUT  (duet_multi layout, mode=$DIRECTION)"
echo "Score together with the internal systems:"
echo "  sbatch --export=ALL,TASK=$TASK,SKIP_INFER=1,Y_OUT=$OUT,\\"
echo "OUT_ROOT=<the e3 run's OUT_ROOT>,BASELINE=Y,CKPT_A2=... \\"
echo "      midi_yinyang/eval_e3.sbatch"
echo "================================================================"
