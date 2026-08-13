#!/bin/bash
# SLURM: PIPELINE co-generation baseline (P-mc / P-cm), end to end.
#
# The cascade foil for joint co-generation: a single-stream specialist
# freely continues ONE stream from its prompt, then the published
# YinYang conditional model generates the OTHER stream conditioned on
# the (generated) first -- no mutual conditioning anywhere. If the duet
# models cannot beat this in E1, joint generation is not earning its
# complexity.
#
#   PIPELINE=P-mc : S-mel continues melody -> Y (mel2chord) harmonizes.
#   PIPELINE=P-cm : S-chord continues chords -> Y (chord2mel) melodizes.
#
# Stages (all in one job):
#   A. cp_transformer_inference: specialist continues the free stream
#      (N_SAMPLES independent continuations per song).
#   B. build_pipeline_hybrids build: each continuation + the partner's
#      ground-truth PROMPT -> a two-track hybrid midi.
#   C. yinyang_e3_driver: Y generates the partner conditioned on the
#      full generated stream (1 draw per hybrid).
#   D. build_pipeline_hybrids assemble: duet_multi layout, mode=co:
#      <OUT>/<song>/co/sample_<i>_temp<T>.mid, tracks MELODY/CHORD.
#
# Score in E1 via the EXTRA_SOURCES knob, e.g.:
#   EXTRA_SOURCES="P-mc:duet_multi:temp/pipeline_mc/final"
#
# DISCLOSED CAVEAT: Y is Nottingham-trained; on POP909 prompts it runs
# out of domain. The pipeline uses the strongest available conditional
# model, which happens to be folk-trained.
#
# Env knobs (--export=ALL,KEY=val):
#   PIPELINE      P-mc (default) or P-cm
#   CKPT_SPEC     REQUIRED: the specialist run dir/ckpt
#                 (S-mel for P-mc, S-chord for P-cm)
#   CKPT_Y        REQUIRED: YinYang ckpt name/path for the direction
#                 (_rev = mel2chord for P-mc; non-rev = chord2mel)
#   MEL_FOLDER    default input/pop909_split/melody
#   CHORD_FOLDER  default input/pop909_split/chord
#   PROMPT_LENGTH default 96.  GEN_LENGTH default 416 (TOTAL).
#   N_SAMPLES     default 3.   MAX_POLYPHONY default 4 (specialist cp).
#   OUT_ROOT      default temp/pipeline_<PIPELINE>_<jobid>

#SBATCH --job-name=pipeline_cogen
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

PIPELINE="${PIPELINE:-P-mc}"
CKPT_SPEC="${CKPT_SPEC:-}"
CKPT_Y="${CKPT_Y:-}"
MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
MAX_POLYPHONY="${MAX_POLYPHONY:-4}"

case "$PIPELINE" in
    P-mc) GEN_FOLDER="$MEL_FOLDER";   PARTNER_FOLDER="$CHORD_FOLDER"
          GENERATED_ROLE=mel;         DIRECTION=mel2chord ;;
    P-cm) GEN_FOLDER="$CHORD_FOLDER"; PARTNER_FOLDER="$MEL_FOLDER"
          GENERATED_ROLE=chord;       DIRECTION=chord2mel ;;
    *) echo "ERROR: PIPELINE must be P-mc or P-cm"; exit 1 ;;
esac

OUT_ROOT="${OUT_ROOT:-temp/pipeline_${PIPELINE}_${SLURM_JOB_ID:-local}}"
STAGEA_NAME="pipeA_${PIPELINE}_${SLURM_JOB_ID:-local}"
HYBRIDS="$OUT_ROOT/hybrids"
Y_TREE="$OUT_ROOT/y"
FINAL="$OUT_ROOT/final"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT"

if [[ -z "$CKPT_SPEC" || -z "$CKPT_Y" ]]; then
    echo "ERROR: CKPT_SPEC and CKPT_Y are both required."
    exit 1
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "PIPELINE CO-GENERATION BASELINE  $PIPELINE"
echo "stage A: $CKPT_SPEC continues $GENERATED_ROLE from $GEN_FOLDER"
echo "stage B: $CKPT_Y ($DIRECTION) conditioned on it"
echo "partner prompts from $PARTNER_FOLDER"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================================================"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'no CUDA device'
print(f'[gpu] {torch.cuda.get_device_name(0)}')
EOF

echo "--- stage A: specialist continuation ---"
python cp_transformer_inference.py \
    --ckpt "$CKPT_SPEC" \
    --midi-folder "$GEN_FOLDER" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples "$N_SAMPLES" \
    --max-polyphony "$MAX_POLYPHONY" --no-split-tracks \
    --save-name "$STAGEA_NAME"

echo "--- stage B: build hybrids ---"
python build_pipeline_hybrids.py build \
    --gen-folder "temp/$STAGEA_NAME" \
    --partner-folder "$PARTNER_FOLDER" \
    --generated-role "$GENERATED_ROLE" \
    --prompt-frames "$PROMPT_LENGTH" --temperature 1.0 \
    --out "$HYBRIDS"

echo "--- stage C: YinYang conditional on the generated stream ---"
python yinyang_e3_driver.py \
    --ckpt "$CKPT_Y" \
    --midi-folder "$HYBRIDS" \
    --task melchord --direction "$DIRECTION" \
    --output-dir "$Y_TREE" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples 1

echo "--- stage D: assemble duet_multi layout ---"
python build_pipeline_hybrids.py assemble \
    --ydriver-out "$Y_TREE" \
    --direction "$DIRECTION" --temperature 1.0 \
    --final "$FINAL"

echo "================================================================"
echo "$PIPELINE complete."
echo "  final outputs: $FINAL   (duet_multi, mode=co, MELODY/CHORD named)"
echo "Score in E1 with:"
echo "  EXTRA_SOURCES=\"$PIPELINE:duet_multi:$FINAL\""
echo "================================================================"
