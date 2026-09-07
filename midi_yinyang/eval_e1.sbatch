#!/bin/bash
# SLURM (GPU): E1 co-generation evaluation, ONE JOB.
#
#   1. stages the held-aside POP909 test split as prompts
#   2. generates every system's outputs into temp/E1/<system>/
#      -- skipping any system whose folder already holds N_SAMPLES
#         co samples for every test song (FORCE regenerates)
#   3. scores everything (manifest -> metrics -> paired table)
#
# Systems (SYSTEMS knob; default all six):
#   A1    intra-cross-attn co-generation, NO query slots (control)
#   A3    the model: duet block diffusion, K=4 refinement, co mode
#   S1    POP909-finetuned single-stream cp transformer (merged
#         streams, chord on program 48), continuation -- the MATCHED
#         single-stream baseline
#   S0    the LA-pretrained single-stream model, same path, never
#         trained on POP909 -- unmatched anchor; not in the default
#         list, add S0 to SYSTEMS to include it
#   P-mc  cascade: S-mel continues melody -> POP909 YinYang (mel2chord,
#         the '_rev' ckpt) harmonizes.  Domain matched.
#   P-cm  cascade: S-chord continues chords -> POP909 YinYang
#         (chord2mel, non-rev) melodizes.  Domain matched.
#   WS    external reference: whole-song-gen PROMPTED lead-sheet
#         continuation, same 6-bar prompt, its own inpainting path.
#
# Every system: same prompt (PROMPT_LENGTH frames), same total length
# (GEN_LENGTH), N_SAMPLES per song, scored on frames
# PROMPT_LENGTH..GEN_LENGTH only. The cascades and WS run through their
# own wrappers (pipeline_cogen.sbatch, infer_wholesong_prompted.sbatch)
# executed inside this allocation, so the E1 arm is byte-for-byte the
# standalone arm.
#
# Output tree (fixed, rerun-stable):
#   temp/E1/<system>/            the system's folder
#     A1, A3, WS:  <song>/co/sample_<i>[_temp<T>].mid   (duet_multi)
#     S1, S0:      <song>.mid_temp<T>_continuation_<i>.mid (single)
#     P-mc, P-cm:  1_stageA_specialist/ 2_hybrids/ 3_y_conditional/
#                  4_final/<song>/co/sample_<i>_temp<T>.mid  (scored)
#   temp/E1/prompts/{mel,chord,merged_tagged}/  staged test split
#   results/E1_p<PROMPT_LENGTH>[_<RESULT_TAG>]_{manifest.tsv,metrics.csv,
#                                               table.md,table_twosided.md}
#
# Skip rule: a system is DONE when every staged test song has at least
# N_SAMPLES samples in its scored layout. Done systems are not
# regenerated -- so a rerun that adds one system, or a job killed
# mid-way, only generates what is missing. FORCE="A1 P-mc" (or
# FORCE=all) deletes those folders under temp/E1 and regenerates them.
#
# Env knobs (--export=ALL,KEY=val):
#   SYSTEMS        default "A1 A3 S1 P-mc P-cm WS"
#   FORCE          default empty; space-separated names, or all
#   CKPT_A1        default the melchord cp4tar intra-cross-attn run dir
#   CKPT_A3        default the A3 cp4tar run dir (K4mg_melchord_cp4tar)
#   CKPT_S1        REQUIRED for S1: the POP909-finetuned MERGED
#                  single-stream run dir (cp16; not the per-stream
#                  cp4 specialists the cascade uses)
#   CKPT_S0        default the LA-pretrained cp transformer ckpt
#   CKPT_SMEL      P-mc stage A: default the pop909ft_mel_cp4 specialist
#   CKPT_SCHORD    P-cm stage A: default the pop909ft_chord_cp4 specialist
#   CKPT_Y_M2C     P-mc stage B: default the POP909 YinYang '_rev' (mel2chord)
#   CKPT_Y_C2M     P-cm stage B: default the POP909 YinYang non-rev (chord2mel)
#   MAX_POLYPHONY_DUET   cp of A1/A3 training data (default 4)
#   MAX_POLYPHONY_SPEC   cp of the cascade specialists (default 4)
#   MAX_POLYPHONY_S1 / MAX_POLYPHONY_S0   cp of the merged single-stream
#                  models (default 16)
#   PROMPT_LENGTH  default 96 (6 bars).  GEN_LENGTH default 416 (total).
#   N_SAMPLES      default 3
#   BASELINE       system the paired table tests against (default A3)
#   MEL_FOLDER / CHORD_FOLDER   default input/pop909_split/{melody,chord}
#   OUT_ROOT       default temp/E1
#   RESULT_TAG     extra suffix on the result files
#   SKIP_INFER=1   score only, generate nothing
#   PIANO_COPIES   default 1: <system>_piano listening copies
#
# Examples:
#   sbatch midi_yinyang/eval_e1.sbatch          # all six, defaults
#   sbatch --export=ALL,SYSTEMS="A1 A3",FORCE=A1 midi_yinyang/eval_e1.sbatch
#   sbatch --export=ALL,SKIP_INFER=1,BASELINE=S1 midi_yinyang/eval_e1.sbatch

#SBATCH --job-name=eval_e1
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
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs results

# ---- knobs ------------------------------------------------------------
SYSTEMS="${SYSTEMS:-A1 A3 S1 P-mc P-cm WS}"
FORCE="${FORCE:-}"
CKPT_A1="${CKPT_A1:-ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_melchord_cp4tar_batch_8_schedule}"
CKPT_A3="${CKPT_A3:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mg_melchord_cp4tar_batch_8_schedule}"
CKPT_S1="${CKPT_S1:-}"
CKPT_S0="${CKPT_S0:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
# Cascade stage A: the cp4 POP909 specialists (cp4 is in the dir name,
# hence MAX_POLYPHONY_SPEC=4 below). Stage B: the POP909-trained
# YinYang conditionals -- direction is the ckpt's own suffix: '_rev' is
# mel2chord (P-mc), non-rev is chord2mel (P-cm). YinYang restores its
# own cp from the ckpt, so no cp flag reaches stage B.
CKPT_SMEL="${CKPT_SMEL:-ckpt/cp_transformer_v0.42_size1_pop909ft_mel_cp4_batch_24_schedule}"
CKPT_SCHORD="${CKPT_SCHORD:-ckpt/cp_transformer_v0.42_size1_pop909ft_chord_cp4_batch_24_schedule}"
CKPT_Y_M2C="${CKPT_Y_M2C:-ckpt/cp_transformer_yinyang_v5.1_lora_batch_8_pop909_cp8_v2_chord_mel_rev_mask0.25-10-step1}"
CKPT_Y_C2M="${CKPT_Y_C2M:-ckpt/cp_transformer_yinyang_v5.1_lora_batch_8_pop909_cp8_v2_chord_mel_mask0.25-10-step1}"
MAX_POLYPHONY_DUET="${MAX_POLYPHONY_DUET:-4}"
MAX_POLYPHONY_SPEC="${MAX_POLYPHONY_SPEC:-4}"
MAX_POLYPHONY_S1="${MAX_POLYPHONY_S1:-16}"
MAX_POLYPHONY_S0="${MAX_POLYPHONY_S0:-16}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
BASELINE="${BASELINE:-A3}"
MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
OUT_ROOT="${OUT_ROOT:-temp/E1}"
RESULT_TAG="${RESULT_TAG:-}"
SKIP_INFER="${SKIP_INFER:-0}"
PIANO_COPIES="${PIANO_COPIES:-1}"

# A3's decode schedule (the listening-test winner); identical to
# queryablation so E1's A3 row and the ablation's A3 row are the same
# decoder.
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
export A3_TOP_P="${A3_TOP_P:-0.95}"

[[ "$OUT_ROOT" == temp/* ]] || { echo "ERROR: OUT_ROOT must be under temp/ (got $OUT_ROOT)"; exit 1; }
mkdir -p "$OUT_ROOT"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E1 CO-GENERATION EVALUATION"
echo "SYSTEMS=$SYSTEMS   FORCE=${FORCE:-<none>}   SKIP_INFER=$SKIP_INFER"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)  GEN_LENGTH=$GEN_LENGTH  N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT=$OUT_ROOT   BASELINE=$BASELINE"
echo "A3 decode: REFINE=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P"
echo "cp: duet=$MAX_POLYPHONY_DUET  specialists=$MAX_POLYPHONY_SPEC  S1/S0=$MAX_POLYPHONY_S1/$MAX_POLYPHONY_S0"
echo "================================================================"
nvidia-smi | head -n 12 || true

# ---- 0. pre-flight ----------------------------------------------------
if [[ "$SKIP_INFER" != "1" ]]; then
    python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"
fi
for _c in "$CKPT_A1" "$CKPT_A3" "$CKPT_S1" "$CKPT_S0" "$CKPT_SMEL" "$CKPT_SCHORD" "$CKPT_Y_M2C" "$CKPT_Y_C2M"; do
    [[ -n "$_c" && -e "$_c" ]] && python check_ckpts.py "$_c" || true
done

# ---- 1. prompts: the test split, staged once -------------------------
# The same full midis serve as prompts (first PROMPT_LENGTH frames) and
# as scoring references (frames PROMPT_LENGTH..GEN_LENGTH).
EVAL_MEL="$OUT_ROOT/prompts/mel"
EVAL_CHORD="$OUT_ROOT/prompts/chord"
EVAL_MERGED="$OUT_ROOT/prompts/merged_tagged"
mkdir -p "$EVAL_MEL" "$EVAL_CHORD"
cp -f "$MEL_FOLDER"/*.mid "$MEL_FOLDER"/*.MID "$EVAL_MEL"/ 2>/dev/null || true
cp -f "$CHORD_FOLDER"/*.mid "$CHORD_FOLDER"/*.MID "$EVAL_CHORD"/ 2>/dev/null || true
N_EVAL=$(ls "$EVAL_MEL" | wc -l)
[[ "$N_EVAL" -gt 0 ]] || { echo "ERROR: no midis in $MEL_FOLDER"; exit 1; }
[[ "$N_EVAL" -eq "$(ls "$EVAL_CHORD" | wc -l)" ]] || {
    echo "ERROR: melody/chord split sizes differ; scoring pairs by filename"; exit 1; }
SONG_IDS=$(ls "$EVAL_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | sort | tr '\n' ' ')
echo "[prompts] $N_EVAL test songs: $SONG_IDS"
# merged, program-tagged copies for the single-stream S0 (the tokenizer
# separates streams by program only)
if [[ ! -d "$EVAL_MERGED" || "$(ls "$EVAL_MERGED" | wc -l)" -lt "$N_EVAL" ]]; then
    python merge_melody_chord.py --melody "$EVAL_MEL" --chord "$EVAL_CHORD" \
        --dst "$EVAL_MERGED" --chord-program 48
fi

# ---- system registry --------------------------------------------------
# folder under OUT_ROOT, the scored root inside it, and its layout.
sys_dir()    { echo "$OUT_ROOT/$1"; }
sys_root()   { case "$1" in P-mc|P-cm) echo "$OUT_ROOT/$1/4_final" ;; *) echo "$OUT_ROOT/$1" ;; esac; }
sys_layout() { case "$1" in S0|S1) echo single ;; *) echo duet_multi ;; esac; }

# DONE = every test song has >= N_SAMPLES samples in the scored layout.
is_complete() {
    local root layout sid n
    root=$(sys_root "$1"); layout=$(sys_layout "$1")
    [[ -d "$root" ]] || return 1
    for sid in $SONG_IDS; do
        case "$layout" in
            duet_multi) n=$(ls "$root/$sid/co/"sample_*.mid 2>/dev/null | wc -l) ;;
            single)     n=$(ls "$root/$sid".mid_temp*_continuation_*.mid 2>/dev/null | wc -l) ;;
        esac
        [[ "$n" -ge "$N_SAMPLES" ]] || return 1
    done
    return 0
}

is_forced() {
    [[ "$FORCE" == "all" ]] && return 0
    local f; for f in $FORCE; do [[ "$f" == "$1" ]] && return 0; done
    return 1
}

# ---- 2. generation ----------------------------------------------------
generate() {
    local name="$1" dir; dir=$(sys_dir "$name")
    case "$name" in
    A1)
        python cp_transformer_m2c_intra_cross_attn_combined.py \
            --ckpt "$CKPT_A1/" \
            --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
            --output-dir "$dir" --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY_DUET" \
            --model-size large --moe-num-experts 4 --moe-topk 2 ;;
    A3)
        python cp_transformer_m2c_duet_block_diffusion_combined.py \
            --ckpt "$CKPT_A3/" \
            --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
            --output-dir "$dir" --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY_DUET" \
            --model-size large --moe-num-experts 4 --moe-topk 2 \
            --min-chord-tokens-before-eos 0 ;;
    S0|S1)
        local ck cp
        if [[ "$name" == "S1" ]]; then ck="$CKPT_S1"; cp="$MAX_POLYPHONY_S1"
        else ck="$CKPT_S0"; cp="$MAX_POLYPHONY_S0"; fi
        [[ -n "$ck" ]] || { echo "[$name] SKIPPED: set CKPT_$name"; return 0; }
        # cp_transformer_inference writes temp/<save-name>/ itself.
        python cp_transformer_inference.py \
            --ckpt "$ck" \
            --midi-folder "$EVAL_MERGED" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$cp" \
            --save-name "${dir#temp/}" ;;
    P-mc|P-cm)
        local spec y
        if [[ "$name" == "P-mc" ]]; then spec="$CKPT_SMEL"; y="$CKPT_Y_M2C"
        else spec="$CKPT_SCHORD"; y="$CKPT_Y_C2M"; fi
        [[ -n "$spec" && -n "$y" ]] || {
            echo "[$name] SKIPPED: needs $([[ "$name" == P-mc ]] && echo CKPT_SMEL and CKPT_Y_M2C || echo CKPT_SCHORD and CKPT_Y_C2M)"
            return 0; }
        # The standalone cascade wrapper, run inside this allocation with
        # E1's prompts and output folder -- so the E1 arm IS the
        # standalone arm. (#SBATCH lines are ignored when run this way;
        # the GPU is this job's.)
        PIPELINE="$name" CKPT_SPEC="$spec" CKPT_Y="$y" STAGE_B=y \
        OUT_ROOT="$dir" MEL_FOLDER="$EVAL_MEL" CHORD_FOLDER="$EVAL_CHORD" \
        PROMPT_LENGTH="$PROMPT_LENGTH" GEN_LENGTH="$GEN_LENGTH" \
        N_SAMPLES="$N_SAMPLES" MAX_POLYPHONY="$MAX_POLYPHONY_SPEC" \
            bash "$MY/pipeline_cogen.sbatch" ;;
    WS)
        # whole-song-gen wants integer ids (its own POP909 index) and an
        # absolute OUT_DIR (it cds into its venv tree). It writes
        # <OUT_DIR>/<id zero-padded to 3>/co/sample_<i>.mid, which is
        # the split's naming.
        local ids; ids=$(for s in $SONG_IDS; do echo $((10#$s)); done | tr '\n' ' ')
        OUT_DIR="$MY/$dir" SONG_IDS="$ids" N_SAMPLES="$N_SAMPLES" \
        PROMPT_BARS="$((PROMPT_LENGTH / 16))" \
            bash "$MY/infer_wholesong_prompted.sbatch" ;;
    *)  echo "ERROR: unknown system $name"; exit 1 ;;
    esac
}

SCORED=()
for name in $SYSTEMS; do
    dir=$(sys_dir "$name")
    echo "----------------------------------------------------------------"
    if [[ "$SKIP_INFER" == "1" ]]; then
        is_complete "$name" && { echo "[$name] score-only: complete"; SCORED+=("$name"); } \
                            || echo "[$name] score-only: INCOMPLETE or missing -- not scored"
        continue
    fi
    if is_forced "$name" && [[ -d "$dir" ]]; then
        echo "[$name] FORCE: removing $dir"
        rm -rf "$dir"
    fi
    if is_complete "$name"; then
        echo "[$name] SKIP: $(sys_root "$name") already has >= $N_SAMPLES co samples for all $N_EVAL songs"
        SCORED+=("$name"); continue
    fi
    echo "[$name] generating -> $dir"
    t0=$(date +%s)
    generate "$name"
    if is_complete "$name"; then
        echo "[$name] done in $(( $(date +%s) - t0 ))s"
        SCORED+=("$name")
    else
        echo "[$name] WARNING: incomplete after generation (missing songs or samples); not scored"
    fi
done

# ---- 2b. piano listening copies (scored files keep chord on prog 48) --
if [[ "$PIANO_COPIES" == "1" && ${#SCORED[@]} -gt 0 && "$SKIP_INFER" != "1" ]]; then
    dirs=(); for name in "${SCORED[@]}"; do dirs+=("$(sys_root "$name")"); done
    python render_piano_copies.py "${dirs[@]}" || echo "[piano] copies failed (non-fatal)"
fi

# ---- 3. scoring -------------------------------------------------------
[[ ${#SCORED[@]} -gt 0 ]] || { echo "ERROR: nothing to score"; exit 1; }
RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
MANIFEST="results/E1_${RSUF}_manifest.tsv"
METRICS="results/E1_${RSUF}_metrics.csv"
TABLE="results/E1_${RSUF}_table.md"

SRC_ARGS=()
for name in "${SCORED[@]}"; do
    SRC_ARGS+=(--source "$name:$(sys_layout "$name"):$(sys_root "$name")")
done
echo "================================================================"
echo "[sources] ${SCORED[*]}"
python build_eval_manifest.py "${SRC_ARGS[@]}" --songs $SONG_IDS \
    --modes co --out "$MANIFEST"

echo "--- scoring (H3 > H2 > H1 > S) ---"
python eval_metrics.py --task melchord \
    --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

if [[ " ${SCORED[*]} " != *" $BASELINE "* ]]; then
    echo "WARNING: BASELINE=$BASELINE was not scored; using ${SCORED[0]} for the paired table"
    BASELINE="${SCORED[0]}"
fi
echo "--- comparison table (paired Wilcoxon vs $BASELINE) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task melchord --baseline "$BASELINE" --mode co \
    --one-sided --complete-cases --out-md "$TABLE"
python aggregate_eval_results.py --csv "$METRICS" \
    --task melchord --baseline "$BASELINE" --mode co \
    --out-md "${TABLE%.md}_twosided.md"

echo "================================================================"
echo "E1 complete.  systems: ${SCORED[*]}"
echo "  outputs : $OUT_ROOT/<system>/"
echo "  table   : $TABLE   (one-sided, complete cases, vs $BASELINE)"
echo "  2-sided : ${TABLE%.md}_twosided.md"
echo "  metrics : $METRICS"
echo "================================================================"
echo "RESULT TABLE:"
cat "$TABLE"
