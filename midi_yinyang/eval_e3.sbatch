#!/bin/bash
# SLURM: END-TO-END E3 (conditional generation, RQ3).
#
# The complete ground-truth partner stream is GIVEN; the model generates
# the remaining stream. Per EXPERIMENTS.md #E3:
#
#   melchord    : A.2 (mel2chord AND chord2mel); optionally B.1's native
#                 conditional (leader given -> follower generated, i.e.
#                 mel2chord).
#   drumnondrum : A.2, B.1, C.1 (rehearsal), C.2 (prefix), all
#                 drum -> nondrum; A.2 additionally nondrum -> drum.
#
# Baselines Y-mc / Y-dn: generate them first with eval_yinyang.sbatch,
# then pass Y_OUT=<its output root> here (adds the Y rows to the same
# table; BASELINE=Y gives paired tests vs the published model). Without
# any baseline present the tables print means only; BASELINE=<internal
# system id> (e.g. A.2) tests the others against it instead.
#
# The no-conditioning FLOOR for interpreting fit scores is the E1
# co-generation table (results/e1_<task>_p<P>_metrics.csv, mode=co):
# the same target-stream metrics scored when the partner was NOT given.
# A conditional system earns its keep by beating those numbers; no extra
# runs are needed for the floor.
#
# Mode-label normalization: A.2's drum->nondrum direction is labeled
# `mel2chord` by the 5-mode script (mod_a=mel/drum), while B.1/C.1/C.2
# write `drum2nondrum`. The manifest's mode column is rewritten
# drum2nondrum -> mel2chord after building, so every system's
# same-direction rows share one table section.
#
# Env knobs (--export=ALL,KEY=val):
#   TASK          melchord (default) | drumnondrum
#   CKPT_A2       A.2 run dir or ckpt file            (required)
#   CKPT_B1       optional (its native conditional mode)
#   CKPT_C1       optional, drumnondrum only (rehearsal)
#   CKPT_C2       optional, drumnondrum only (prefix)
#   BASELINE      system id for paired tests; default Y-mc/Y-dn (absent
#                 in the table -> means-only, by design)
#   MEL_FOLDER / CHORD_FOLDER   same task defaults as eval_e1
#   PROMPT_LENGTH default 64; GEN_LENGTH default 384 (TOTAL frames)
#   N_SAMPLES     default 3 -- honoured by A.2 and B.1. C.1/C.2 have no
#                 sampling flag and emit 1 sample each; their per-song
#                 scores are therefore noisier (disclosed, not hidden).
#   RESULT_TAG    extra suffix on result filenames
#   SKIP_INFER=1  reuse existing outputs under OUT_ROOT, score only
#   PIANO_COPIES  1/0; default 1 for melchord (listening copies)
#
# Example (melchord):
#   sbatch --export=ALL,\
# CKPT_A2=ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_melchord_pop909_nottingham_batch_8_schedule/ \
#       midi_yinyang/eval_e3.sbatch
#
# Example (drumnondrum, all four systems, internal baseline A.2):
#   sbatch --export=ALL,TASK=drumnondrum,BASELINE=A.2,\
# CKPT_A2=ckpt/m2c_duet_block_diffusion_v1.0_large_gnl12_K4_drumnondrum_batch_16_schedule/,\
# CKPT_B1=ckpt/m2c_duet_anticipatory_v1.0_large_gnl12_drumnondrum_k16_batch_16_schedule/,\
# CKPT_C1=ckpt/<rehearsal run>/,CKPT_C2=ckpt/<prefix run>/ \
#       midi_yinyang/eval_e3.sbatch

#SBATCH --job-name=mitosis_eval_e3
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

TASK="${TASK:-melchord}"
CKPT_A2="${CKPT_A2:-}"
CKPT_B1="${CKPT_B1:-}"
CKPT_C1="${CKPT_C1:-}"
CKPT_C2="${CKPT_C2:-}"
N_SAMPLES="${N_SAMPLES:-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
SKIP_INFER="${SKIP_INFER:-0}"
RESULT_TAG="${RESULT_TAG:-}"

case "$TASK" in
    melchord)
        MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
        # cp8 is the melchord convention (cp4 retired: seventh chords are
        # 5 notes). Must match the ckpts' training data -- pass
        # MAX_POLYPHONY=4 for pre-migration cp4-trained checkpoints.
        MAX_POLYPHONY="${MAX_POLYPHONY:-8}"
        BASELINE="${BASELINE:-Y-mc}"
        # C.1/C.2 on melchord are valid once their melchord trainings
        # exist (train_duet_{rehearsal,prefix}.sbatch with
        # TASK=melchord_pop909_nottingham). Passing a ckpt asserts you
        # have one; the integrity pre-flight below rejects a wrong path.
        if [[ -n "$CKPT_C1" || -n "$CKPT_C2" ]]; then
            echo "[note] C.1/C.2 on melchord: make sure these are MELCHORD"
            echo "       checkpoints -- a drumnondrum ckpt here would decode"
            echo "       melchord prompts with cp16 assumptions and score garbage."
        fi
        ;;
    drumnondrum)
        MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"
        MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
        BASELINE="${BASELINE:-Y-dn}"
        ;;
    *) echo "ERROR: unknown TASK=$TASK"; exit 1 ;;
esac

OUT_ROOT="${OUT_ROOT:-temp/e3_${TASK}_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

if [[ -z "$CKPT_A2" && -z "$CKPT_C1" && -z "$CKPT_C2" && -z "$CKPT_B1" && -z "${Y_OUT:-}" ]]; then
    echo "ERROR: nothing to evaluate -- set at least one of CKPT_A2,"
    echo "       CKPT_C1, CKPT_C2, CKPT_B1, or Y_OUT."
    exit 1
fi
[[ -z "$CKPT_A2" ]] && echo "[note] CKPT_A2 unset: A.2 SKIPPED (C/Y-only conditional run)"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E3 CONDITIONAL EVALUATION   task=$TASK"
echo "CKPT_A2 = $CKPT_A2"
echo "CKPT_B1 = ${CKPT_B1:-<unset: B.1 SKIPPED>}"
echo "CKPT_C1 = ${CKPT_C1:-<unset: C.1 SKIPPED>}"
echo "CKPT_C2 = ${CKPT_C2:-<unset: C.2 SKIPPED>}"
echo "BASELINE=$BASELINE (absent in table -> means-only)"
echo "prompts  = $MEL_FOLDER / $CHORD_FOLDER"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)  "\
"GEN_LENGTH=$GEN_LENGTH ($((GEN_LENGTH / 16)) bars)  N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================================================"
nvidia-smi | head -n 12 || true

# GPU pre-flight (all generation stages call model.cuda()).
if ! python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"; then
    echo "ERROR: no usable CUDA GPU."
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "  CAUSE: not submitted via sbatch, so --gres never applied."
    else
        echo "  CAUSE: node $(hostname) has no working GPU; resubmit with"
        echo "         sbatch --exclude=$(hostname) ..."
    fi
    exit 1
fi

CKPT_ARGS=()
[[ -n "$CKPT_A2" ]] && CKPT_ARGS+=("$CKPT_A2")
[[ -n "$CKPT_B1" ]] && CKPT_ARGS+=("$CKPT_B1")
[[ -n "$CKPT_C1" ]] && CKPT_ARGS+=("$CKPT_C1")
[[ -n "$CKPT_C2" ]] && CKPT_ARGS+=("$CKPT_C2")
if [[ ${#CKPT_ARGS[@]} -gt 0 ]]; then
    python check_ckpts.py "${CKPT_ARGS[@]}" || {
        echo "ERROR: unreadable checkpoint(s) -- see above."; exit 1; }
fi

for d in "$MEL_FOLDER" "$CHORD_FOLDER"; do
    [[ -d "$d" ]] || { echo "ERROR: $d not found"; exit 1; }
done
N_MEL=$(find "$MEL_FOLDER" -maxdepth 1 \( -name '*.mid' -o -name '*.MID' \) 2>/dev/null | wc -l)
N_CHD=$(find "$CHORD_FOLDER" -maxdepth 1 \( -name '*.mid' -o -name '*.MID' \) 2>/dev/null | wc -l)
if [[ "$N_MEL" -eq 0 || "$N_MEL" -ne "$N_CHD" ]]; then
    echo "ERROR: prompt folders must hold the SAME songs ($N_MEL vs $N_CHD)."
    exit 1
fi
SONG_IDS=$(ls "$MEL_FOLDER" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
echo "[prompts] $N_MEL songs: $SONG_IDS"

A2_OUT="$OUT_ROOT/A.2"
B1_OUT="$OUT_ROOT/B.1"
C1_OUT="$OUT_ROOT/C.1"
C2_OUT="$OUT_ROOT/C.2"

# ---------------------------------------------------------------------
# Conditional DIRECTION for the B/C systems (melchord):
#   mel2chord (default)  forward ckpts (task melchord_*): mod_a = melody
#   chord2mel            REVERSED ckpts (task melchord_*_rev): mod_a =
#                        chord conditioning, melody generated. Feeds the
#                        chord folder to the mod_a side, swaps the
#                        decoded track names (decode names by stream
#                        position), and relabels the manifest mode.
# A.2 is direction-agnostic (runs both modes natively) and unaffected.
# On drumnondrum the default mapping is the existing behaviour.
# ---------------------------------------------------------------------
DIRECTION="${DIRECTION:-mel2chord}"
case "$DIRECTION" in
    mel2chord)
        MODA_FOLDER="$MEL_FOLDER"; MODB_FOLDER="$CHORD_FOLDER"
        SWAP_ARGS=(); NORM_MODE=mel2chord ;;
    chord2mel)
        MODA_FOLDER="$CHORD_FOLDER"; MODB_FOLDER="$MEL_FOLDER"
        SWAP_ARGS=(--swap-stream-names); NORM_MODE=chord2mel ;;
    *) echo "ERROR: DIRECTION must be mel2chord or chord2mel"; exit 1 ;;
esac
echo "DIRECTION=$DIRECTION (B/C conditioning side: ${MODA_FOLDER})"

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing existing outputs"
else
    # -----------------------------------------------------------------
    # A.2 conditional modes. mel2chord = mod_a given, mod_b generated
    # (drum->nondrum on the drumnondrum task); chord2mel the reverse.
    # -----------------------------------------------------------------
    if [[ -n "$CKPT_A2" ]]; then
        echo "--- A.2 mel2chord + chord2mel ---"
        python cp_transformer_m2c_duet_block_diffusion_combined.py \
            --ckpt "$CKPT_A2" \
            --mel-folder "$MEL_FOLDER" --chord-folder "$CHORD_FOLDER" \
            --output-dir "$A2_OUT" \
            --modes mel2chord chord2mel \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY" \
            --model-size large --moe-num-experts 4 --moe-topk 2
    else
        echo "[A.2] SKIPPED (CKPT_A2 unset)"
    fi

    if [[ -n "$CKPT_B1" ]]; then
        if [[ "$DIRECTION" == "chord2mel" ]]; then
            echo "[B.1] SKIPPED: its inference has no reversed mode/naming;"
            echo "      evaluate B.1 in a DIRECTION=mel2chord run."
        else
            echo "--- B.1 drum2nondrum (leader given, its native mode) ---"
            python cp_transformer_m2c_duet_anticipatory_inference.py \
                --ckpt "$CKPT_B1" --mode drum2nondrum \
                --drum-folder "$MODA_FOLDER" --nondrum-folder "$MODB_FOLDER" \
                --output-dir "$B1_OUT" \
                --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
                --temperature 1.0 --n-samples "$N_SAMPLES" \
                --max-polyphony "$MAX_POLYPHONY" \
                --model-size large --moe-num-experts 4 --moe-topk 2
        fi
    fi

    # C.1/C.2: purpose-built conditional models (mod_a given -> mod_b
    # generated; the ckpt's task decides which stream is which, and
    # DIRECTION must match it).
    if [[ -n "$CKPT_C1" ]]; then
        echo "--- C.1 rehearsal conditional ($DIRECTION, $N_SAMPLES sample(s)/song) ---"
        python cp_transformer_m2c_duet_rehearsal_inference.py \
            --ckpt "$CKPT_C1" \
            --drum-folder "$MODA_FOLDER" --nondrum-folder "$MODB_FOLDER" \
            --output-dir "$C1_OUT" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY" \
            --model-size large --moe-num-experts 4 --moe-topk 2 \
            "${SWAP_ARGS[@]}"
    fi
    if [[ -n "$CKPT_C2" ]]; then
        echo "--- C.2 prefix conditional ($DIRECTION, $N_SAMPLES sample(s)/song) ---"
        python cp_transformer_m2c_duet_prefix_inference.py \
            --ckpt "$CKPT_C2" \
            --drum-folder "$MODA_FOLDER" --nondrum-folder "$MODB_FOLDER" \
            --output-dir "$C2_OUT" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY" \
            --model-size large --moe-num-experts 4 --moe-topk 2 \
            "${SWAP_ARGS[@]}"
    fi
fi

# ---------------------------------------------------------------------
# piano listening copies (melchord only; scored files untouched)
# ---------------------------------------------------------------------
PIANO_COPIES="${PIANO_COPIES:-}"
if [[ -z "$PIANO_COPIES" ]]; then
    [[ "$TASK" == "melchord" ]] && PIANO_COPIES=1 || PIANO_COPIES=0
fi
if [[ "$PIANO_COPIES" == "1" ]]; then
    PIANO_DIRS=()
    for d in "$A2_OUT" "$B1_OUT"; do [[ -d "$d" ]] && PIANO_DIRS+=("$d"); done
    [[ ${#PIANO_DIRS[@]} -gt 0 ]] && {
        echo "--- piano listening copies ---"
        python render_piano_copies.py "${PIANO_DIRS[@]}"; }
fi

# ---------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------
if [[ "$N_SAMPLES" -gt 1 ]]; then
    A2_LAYOUT=duet_multi; B1_LAYOUT=duet_multi; C_LAYOUT=duet_multi
else
    A2_LAYOUT=duet;       B1_LAYOUT=cond;       C_LAYOUT=cond
fi

RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
MANIFEST="results/e3_${TASK}_${RSUF}_manifest.tsv"
METRICS="results/e3_${TASK}_${RSUF}_metrics.csv"
TABLE="results/e3_${TASK}_${RSUF}_table.md"

# Optional published-baseline rows: point Y_OUT at a duet_multi root
# produced by eval_yinyang.sbatch / yinyang_e3_driver.py. Its mode
# dirs use the same direction labels, so the rows land in the same
# table sections; BASELINE=Y then gives paired tests vs YinYang.
Y_OUT="${Y_OUT:-}"
SRC_ARGS=()
[[ -d "$A2_OUT" ]] && SRC_ARGS+=(--source "A.2:${A2_LAYOUT}:${A2_OUT}")
[[ -n "$Y_OUT" && -d "$Y_OUT" ]] && SRC_ARGS+=(--source "Y:duet_multi:${Y_OUT}")
[[ -d "$B1_OUT" ]] && SRC_ARGS+=(--source "B.1:${B1_LAYOUT}:${B1_OUT}")
[[ -d "$C1_OUT" ]] && SRC_ARGS+=(--source "C.1:${C_LAYOUT}:${C1_OUT}")
[[ -d "$C2_OUT" ]] && SRC_ARGS+=(--source "C.2:${C_LAYOUT}:${C2_OUT}")

python build_eval_manifest.py "${SRC_ARGS[@]}" \
    --songs $SONG_IDS \
    --modes mel2chord chord2mel drum2nondrum \
    --out "$MANIFEST"

# Normalize the direction label so every system's rows land in ONE
# table section: B.1/C.1/C.2 write 'drum2nondrum' regardless of task;
# relabel it to the run's DIRECTION (mel2chord, or chord2mel for
# reversed-ckpt runs) to match A.2's/Y's native mode labels.
sed -i $'s/\tdrum2nondrum\t/\t'"$NORM_MODE"$'\t/g' "$MANIFEST"

python eval_metrics.py --task "$TASK" \
    --manifest "$MANIFEST" \
    --ref-a-dir "$MEL_FOLDER" --ref-b-dir "$CHORD_FOLDER" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline "$BASELINE" \
    --complete-cases \
    --out-md "$TABLE"

echo "================================================================"
echo "E3 evaluation complete."
echo "  generated midis : $OUT_ROOT"
echo "  manifest        : $MANIFEST   (drum2nondrum relabeled mel2chord)"
echo "  per-sample CSV  : $METRICS"
echo "  table           : $TABLE"
echo "Reading guide:"
echo "  * mel2chord section = partner GIVEN, target generated: read the"
echo "    TARGET stream's metrics (_b rows) plus the cross metrics"
echo "    (harmonic_rhythm_jsd, chord_tone_cov_delta) -- fit to a real"
echo "    partner is the whole question. chord2mel: read _a rows."
echo "  * FLOOR: compare against the same metrics in the E1 co table"
echo "    (results/e1_${TASK}_*_metrics.csv, mode=co) -- that is what the"
echo "    system achieves WITHOUT the partner given. Conditioning must"
echo "    beat it to mean anything."
echo "  * The GIVEN stream's rows (_a in mel2chord) are the copied real"
echo "    stream -- reference values, not generation quality."
echo "  * All systems drew $N_SAMPLES sample(s) per song."
echo "================================================================"
