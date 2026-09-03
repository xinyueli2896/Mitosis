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
#   EXTRA_SOURCES="P-mc:duet_multi:temp/cascade_P-mc_<jobid>/4_final"
#
# STAGE-B DOMAIN (updated 2026-09-03): a POP909-trained YinYang
# conditional now exists (cp_transformer_yinyang_v5.1_lora_batch_8_
# pop909_cp8_v2_chord_mel[_rev]_...), so the cascade can run DOMAIN
# MATCHED -- prefer it. Direction mapping is the ckpt's own suffix:
#   P-mc (mel2chord) -> the '_rev' ckpt
#   P-cm (chord2mel) -> the non-rev ckpt
# The published Nottingham-trained ckpts remain runnable for the
# published-baseline row, but then the out-of-domain caveat must be
# disclosed with the number. Whichever is used, the label printed below
# goes into the results table.
#
# Env knobs (--export=ALL,KEY=val):
#   PIPELINE      P-mc (default) or P-cm
#   CKPT_SPEC     REQUIRED: the specialist run dir/ckpt
#                 (S-mel for P-mc, S-chord for P-cm)
#   STAGE_B       y (default) | c1 | c2 -- who writes the partner stream.
#                 y  = published YinYang ckpt (Nottingham-trained: OUT
#                      OF DOMAIN on POP909, disclose it)
#                 c1/c2 = in-domain conditional trained on the same
#                      corpus as the specialists (the strong foil)
#   CKPT_Y        required when STAGE_B=y: YinYang ckpt for the direction
#                 (_rev = mel2chord for P-mc; non-rev = chord2mel)
#   CKPT_COND     required when STAGE_B=c1/c2: the in-domain conditional
#                 run dir (task melchord for P-mc, melchord_rev for P-cm)
#   MEL_FOLDER    default input/pop909_split/melody
#   CHORD_FOLDER  default input/pop909_split/chord
#   PROMPT_LENGTH default 96.  GEN_LENGTH default 416 (TOTAL).
#   N_SAMPLES     default 3.
#   MAX_POLYPHONY the STAGE-A SPECIALIST's training cp (default 4).
#                 S-mel / S-chord are trained on pop909_{melody,chord}_
#                 cp8_v2.pt -> pass MAX_POLYPHONY=8 for them. A cp
#                 mismatch loads cleanly and silently degrades the
#                 music (CLAUDE.md). Stage C needs no cp flag: the
#                 YinYang ckpt restores its own hyperparameters.
#   OUT_ROOT      default temp/cascade_<PIPELINE>_<jobid>. MUST be under
#                 temp/ (stage A can only write there). The run tree is
#                 1_stageA_specialist / 2_hybrids / 3_y_conditional /
#                 4_final -- score 4_final.

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
# Stage B: which conditional model writes the partner stream.
#   y  (default) the published YinYang ckpt -- Nottingham-trained, so
#      OUT OF DOMAIN on POP909; keep as the published-baseline row.
#   c1 / c2      an IN-DOMAIN conditional (C.1 rehearsal / C.2 prefix)
#      trained on the same corpus as the specialists -- the strong foil,
#      where "no mutual conditioning" is the ONLY difference from the
#      duet models. Needs CKPT_COND.
STAGE_B="${STAGE_B:-y}"
CKPT_COND="${CKPT_COND:-}"
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

# Everything for one cascade run lives under ONE root, in numbered
# stage order. cp_transformer_inference can only write under temp/
# (it builds temp/<save_name>/...), so the stage-A save name is the
# root's path relative to temp/ -- which keeps its output inside the
# run root instead of in a confusing sibling directory.
OUT_ROOT="${OUT_ROOT:-temp/cascade_${PIPELINE}_${SLURM_JOB_ID:-local}}"
if [[ "$OUT_ROOT" != temp/* ]]; then
    echo "ERROR: OUT_ROOT must start with temp/ (stage A can only write"
    echo "       under temp/). Got: $OUT_ROOT"
    exit 1
fi
STAGEA_SAVE_NAME="${OUT_ROOT#temp/}/1_stageA_specialist"
STAGEA_DIR="$OUT_ROOT/1_stageA_specialist"
HYBRIDS="$OUT_ROOT/2_hybrids"
Y_TREE="$OUT_ROOT/3_y_conditional"
FINAL="$OUT_ROOT/4_final"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT"

if [[ -z "$CKPT_SPEC" ]]; then
    echo "ERROR: CKPT_SPEC is required."; exit 1
fi
case "$STAGE_B" in
    y)  [[ -n "$CKPT_Y" ]] || { echo "ERROR: STAGE_B=y needs CKPT_Y."; exit 1; }
        # yinyang_e3_driver's loader takes a FILE; accept a run dir here
        # like every other CKPT knob and resolve the best-val ckpt with
        # the shared resolver.
        if [[ -d "$CKPT_Y" ]]; then
            # resolve_best_ckpt narrates its choice on STDOUT, so send that
            # to stderr and keep stdout to the path alone -- capturing both
            # produced a bogus multi-line CKPT_Y (job 203490).
            CKPT_Y=$(python -c "
import contextlib, sys
from ckpt_utils import resolve_best_ckpt
with contextlib.redirect_stdout(sys.stderr):
    _p = resolve_best_ckpt('$CKPT_Y')
print(_p)")
            echo "[ckpt_y] run dir -> $CKPT_Y"
            [[ -f "$CKPT_Y" ]] || { echo "ERROR: resolved CKPT_Y is not a file: $CKPT_Y"; exit 1; }
        fi
        case "$CKPT_Y" in
            *pop909*) _Y_DOMAIN="POP909-trained, DOMAIN MATCHED" ;;
            *)        _Y_DOMAIN="published Nottingham-trained, OUT OF DOMAIN on POP909" ;;
        esac
        COND_LABEL="YinYang $CKPT_Y ($_Y_DOMAIN)" ;;
    c1) [[ -n "$CKPT_COND" ]] || { echo "ERROR: STAGE_B=c1 needs CKPT_COND."; exit 1; }
        COND_SCRIPT=cp_transformer_m2c_duet_rehearsal_inference.py
        COND_LABEL="C.1 rehearsal $CKPT_COND (in-domain)" ;;
    c2) [[ -n "$CKPT_COND" ]] || { echo "ERROR: STAGE_B=c2 needs CKPT_COND."; exit 1; }
        COND_SCRIPT=cp_transformer_m2c_duet_prefix_inference.py
        COND_LABEL="C.2 prefix $CKPT_COND (in-domain)" ;;
    *)  echo "ERROR: STAGE_B must be y, c1, or c2"; exit 1 ;;
esac
# Reversed direction (chord->mel): the C decoders name output tracks by
# stream POSITION, so the names must be swapped or the scorer reads the
# streams backwards.
SWAP_ARGS=()
[[ "$DIRECTION" == "chord2mel" && "$STAGE_B" != "y" ]] && SWAP_ARGS=(--swap-stream-names)

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "PIPELINE CO-GENERATION BASELINE  $PIPELINE"
echo "stage A: $CKPT_SPEC continues $GENERATED_ROLE from $GEN_FOLDER"
echo "stage B: $COND_LABEL, direction=$DIRECTION"
echo "partner prompts from $PARTNER_FOLDER"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================================================"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'no CUDA device'
print(f'[gpu] {torch.cuda.get_device_name(0)}')
EOF

echo "--- stage A: specialist continues the $GENERATED_ROLE stream ---"
python cp_transformer_inference.py \
    --ckpt "$CKPT_SPEC" \
    --midi-folder "$GEN_FOLDER" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples "$N_SAMPLES" \
    --max-polyphony "$MAX_POLYPHONY" --no-split-tracks \
    --save-name "$STAGEA_SAVE_NAME"

if [[ "$STAGE_B" == "y" ]]; then
    echo "--- stage B: hybrid = generated $GENERATED_ROLE + partner prompt ---"
    python build_pipeline_hybrids.py build \
        --gen-folder "$STAGEA_DIR" \
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
else
    echo "--- stage B: pair generated $GENERATED_ROLE with partner prompts ---"
    python build_pipeline_hybrids.py build-cond \
        --gen-folder "$STAGEA_DIR" \
        --partner-folder "$PARTNER_FOLDER" \
        --temperature 1.0 \
        --out-cond "$HYBRIDS/conditioning" \
        --out-prompt "$HYBRIDS/target_prompt"

    echo "--- stage C: in-domain conditional ($STAGE_B) ---"
    python "$COND_SCRIPT" \
        --ckpt "$CKPT_COND" \
        --drum-folder "$HYBRIDS/conditioning" \
        --nondrum-folder "$HYBRIDS/target_prompt" \
        --output-dir "$Y_TREE" \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --max-polyphony "$MAX_POLYPHONY" \
        --model-size large --moe-num-experts 4 --moe-topk 2 \
        "${SWAP_ARGS[@]}"
fi

echo "--- stage D: assemble duet_multi layout ---"
python build_pipeline_hybrids.py assemble \
    --ydriver-out "$Y_TREE" \
    --direction "$DIRECTION" --temperature 1.0 \
    --final "$FINAL"

echo "================================================================"
echo "$PIPELINE complete. Run tree:"
echo "  $OUT_ROOT/"
echo "    1_stageA_specialist/  specialist continuations (raw, single stream)"
echo "    2_hybrids/            two-track inputs handed to YinYang"
echo "    3_y_conditional/      YinYang raw output tree"
echo "    4_final/              <-- SCORE THIS (duet_multi, mode=co)"
echo ""
echo "Score in E1 by adding to the export list:"
echo "  EXTRA_SOURCES=\"$PIPELINE:duet_multi:$FINAL\""
echo "================================================================"
