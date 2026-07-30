#!/bin/bash
# SLURM: END-TO-END E1 (co-generation) evaluation.
#   1. builds eval-subset prompt folders (held-out songs only)
#   2. runs co-generation inference for every system, saving outputs
#   3. runs the quantitative analysis (manifest -> metrics -> table)
#
# Systems evaluated:
#   A.2  duet block diffusion, co mode, frozen decode schedule
#   S1   single-stream in-domain finetune  (melchord: the MATCHED baseline)
#   S0   single-stream pretrained          (unmatched lower anchor)
#
# NOT included: B.1. Its inference script
# (cp_transformer_m2c_duet_anticipatory_inference.py) implements ONLY
# conditional leader->follower decoding; there is no co-generation decode
# loop for the anticipatory re-indexing, and running a B.1 ckpt through
# the generic A.1 combined script would decode it WITHOUT the k-frame
# shift it was trained with (train/inference mismatch -> invalid row).
# Add B.1 to E1 only once a proper co-gen decoder exists.
#
# Env knobs (override via --export=ALL,KEY=val):
#   TASK          melchord (default) | drumnondrum
#   CKPT_A2       A.2 run dir or ckpt file        (required)
#   CKPT_S1       S1 ckpt file                    (melchord; optional)
#   CKPT_S0       S0 pretrained ckpt              (optional)
#   MEL_FOLDER    full melody/drum midis  (prompts AND scoring references)
#   CHORD_FOLDER  full chord/nondrum midis
#   SONGS         space-separated eval song ids
#                 default (melchord): 001 011 ... 091   (held out from
#                 BOTH the duet and single-stream splits)
#   OUT_ROOT      default temp/e1_<TASK>_<JOBID>
#   BASELINE      system id the table is tested against
#                 (default: S1 for melchord, S0 for drumnondrum)
#   N_SAMPLES     single-stream samples per song (default 3)
#   SKIP_INFER=1  reuse existing outputs under OUT_ROOT, score only
#
# Example:
#   sbatch --export=ALL,\
# CKPT_A2=ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_melchord_batch_8_schedule/,\
# CKPT_S1=ckpt/cp_transformer_v0.42_size1_pop909ft_short_batch_48_schedule/last.ckpt \
#       midi_yinyang/eval_e1.sbatch

#SBATCH --job-name=mitosis_eval_e1
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
CKPT_S1="${CKPT_S1:-}"
CKPT_S0="${CKPT_S0:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
N_SAMPLES="${N_SAMPLES:-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
SKIP_INFER="${SKIP_INFER:-0}"

case "$TASK" in
    melchord)
        MEL_FOLDER="${MEL_FOLDER:-POP909-Dataset/POP909-melody}"
        CHORD_FOLDER="${CHORD_FOLDER:-POP909-Dataset/POP909-chord}"
        SONGS="${SONGS:-001 011 021 031 041 051 061 071 081 091}"
        MAX_POLYPHONY_DUET=4
        BASELINE="${BASELINE:-S1}"
        ;;
    drumnondrum)
        MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"
        SONGS="${SONGS:-}"          # empty = all songs in the folders
        MAX_POLYPHONY_DUET=16
        BASELINE="${BASELINE:-S0}"
        ;;
    *) echo "ERROR: unknown TASK=$TASK"; exit 1 ;;
esac

OUT_ROOT="${OUT_ROOT:-temp/e1_${TASK}_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E1 CO-GENERATION EVALUATION   task=$TASK"
echo "CKPT_A2 = ${CKPT_A2:-<unset: A.2 will be SKIPPED>}"
echo "CKPT_S1 = ${CKPT_S1:-<unset: S1 will be SKIPPED>}"
echo "CKPT_S0 = ${CKPT_S0:-<unset: S0 will be SKIPPED>}"
echo "MEL_FOLDER=$MEL_FOLDER  CHORD_FOLDER=$CHORD_FOLDER"
echo "SONGS=${SONGS:-<all>}"
echo "OUT_ROOT=$OUT_ROOT   BASELINE=$BASELINE"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "================================================================"
nvidia-smi | head -n 12 || true

# ---------------------------------------------------------------------
# 1. eval-subset prompt folders
#    Inference iterates whole folders, so restricting to the held-out
#    songs here is what keeps this from decoding all ~900 POP909 songs.
#    The SAME full midis serve as prompts (first PROMPT_LENGTH frames)
#    and as scoring references (frames PROMPT_LENGTH..GEN_LENGTH).
# ---------------------------------------------------------------------
EVAL_MEL="$OUT_ROOT/prompts/mel"
EVAL_CHORD="$OUT_ROOT/prompts/chord"
EVAL_MERGED="$OUT_ROOT/prompts/merged_tagged"
mkdir -p "$EVAL_MEL" "$EVAL_CHORD"

if [[ -n "$SONGS" ]]; then
    for sid in $SONGS; do
        for pair in "$MEL_FOLDER:$EVAL_MEL" "$CHORD_FOLDER:$EVAL_CHORD"; do
            src_dir="${pair%%:*}"; dst_dir="${pair##*:}"
            src=$(ls "$src_dir/$sid".mid "$src_dir/$sid".MID 2>/dev/null | head -1 || true)
            if [[ -z "$src" ]]; then
                echo "ERROR: song $sid not found in $src_dir"; exit 1
            fi
            cp -f "$src" "$dst_dir/$sid.mid"
        done
    done
else
    cp -f "$MEL_FOLDER"/*.mid "$EVAL_MEL"/ 2>/dev/null || true
    cp -f "$CHORD_FOLDER"/*.mid "$EVAL_CHORD"/ 2>/dev/null || true
fi
N_EVAL=$(ls "$EVAL_MEL" | wc -l)
echo "[prompts] $N_EVAL eval songs -> $EVAL_MEL , $EVAL_CHORD"

# merged, program-tagged prompts for the single-stream systems: the
# tokenizer separates streams by PROGRAM only, so the chord stream must
# carry a distinct program or the two parts fuse irreversibly.
if [[ "$TASK" == "melchord" ]]; then
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" \
        --dst "$EVAL_MERGED" --chord-program 48
else
    # drumnondrum: drums are natively program 127, so a plain merge keeps
    # the streams distinguishable.
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" --dst "$EVAL_MERGED"
fi
echo "[prompts] merged+tagged -> $EVAL_MERGED"

# ---------------------------------------------------------------------
# 2. inference
# ---------------------------------------------------------------------
A2_OUT="$OUT_ROOT/A.2"
S1_OUT_NAME="e1_${TASK}_S1_${SLURM_JOB_ID:-local}"
S0_OUT_NAME="e1_${TASK}_S0_${SLURM_JOB_ID:-local}"
S1_OUT="temp/$S1_OUT_NAME"
S0_OUT="temp/$S0_OUT_NAME"

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing existing outputs"
else
    if [[ -n "$CKPT_A2" ]]; then
        echo "--- A.2 co-generation (K=4 schedule from infer defaults) ---"
        export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
        export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
        export A3_TOP_P="${A3_TOP_P:-0.95}"
        export A3_ADAPTIVE="${A3_ADAPTIVE:-1}"
        echo "[A.2] K=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P adaptive=$A3_ADAPTIVE"
        python cp_transformer_m2c_duet_block_diffusion_combined.py \
            --ckpt "$CKPT_A2" \
            --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
            --output-dir "$A2_OUT" \
            --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 \
            --max-polyphony "$MAX_POLYPHONY_DUET" \
            --model-size large --moe-num-experts 4 --moe-topk 2
    else
        echo "[A.2] SKIPPED (CKPT_A2 unset)"
    fi

    for spec in "S1:$CKPT_S1:$S1_OUT_NAME" "S0:$CKPT_S0:$S0_OUT_NAME"; do
        sys="${spec%%:*}"; rest="${spec#*:}"
        ckpt="${rest%%:*}"; name="${rest##*:}"
        if [[ -z "$ckpt" ]]; then echo "[$sys] SKIPPED (ckpt unset)"; continue; fi
        if [[ ! -e "$ckpt" ]]; then echo "[$sys] SKIPPED (ckpt not found: $ckpt)"; continue; fi
        echo "--- $sys single-stream continuation ---"
        python cp_transformer_inference.py \
            --ckpt "$ckpt" \
            --midi-folder "$EVAL_MERGED" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --save-name "$name"
    done
fi

# ---------------------------------------------------------------------
# 3. quantitative analysis
# ---------------------------------------------------------------------
MANIFEST="results/e1_${TASK}_manifest.tsv"
METRICS="results/e1_${TASK}_metrics.csv"
TABLE="results/e1_${TASK}_table.md"

SRC_ARGS=()
[[ -d "$A2_OUT" ]] && SRC_ARGS+=(--source "A.2:duet:$A2_OUT")
[[ -d "$S1_OUT" ]] && SRC_ARGS+=(--source "S1:single:$S1_OUT")
[[ -d "$S0_OUT" ]] && SRC_ARGS+=(--source "S0:single:$S0_OUT")
if [[ ${#SRC_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: no inference outputs found to score."
    exit 1
fi

SONG_ARGS=()
[[ -n "$SONGS" ]] && SONG_ARGS=(--songs $SONGS)

echo "================================================================"
echo "--- building manifest ---"
python build_eval_manifest.py "${SRC_ARGS[@]}" "${SONG_ARGS[@]}" \
    --modes co --out "$MANIFEST"

echo "--- scoring (H1-H3) ---"
python eval_metrics.py --task "$TASK" \
    --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

echo "--- comparison table (paired Wilcoxon vs $BASELINE) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline "$BASELINE" --mode co \
    --out-md "$TABLE"

echo "================================================================"
echo "E1 evaluation complete."
echo "  generated midis : $OUT_ROOT (A.2), $S1_OUT, $S0_OUT"
echo "  manifest        : $MANIFEST"
echo "  per-sample CSV  : $METRICS"
echo "  comparison table: $TABLE"
echo "Reminder: B.1's co-generation row is NOT included (no co-gen"
echo "decoder for the anticipatory re-indexing yet)."
