#!/bin/bash
# SLURM: A.2 rope-geometry ablation -- v1.1 (parity RoPE) vs v1.2
# (time-aligned RoPE), same architecture, same decode settings, same
# prompts. The question under test: does restoring the pretrain's
# positional geometry (musical distance == rotary distance) recover the
# long-term structure the duet outputs lack?
#
# Runs co-generation with BOTH checkpoints, then scores twice:
#   1. structure_metrics.py -- the HEADLINE readout for this ablation:
#      repetition, 4/8-bar periodicity (chroma + rhythm), tonal drift,
#      per stream, each variant vs the ground-truth REF row.
#   2. the standard E1 chain (build_eval_manifest -> eval_metrics ->
#      aggregate) with BASELINE=A2v11, so the Wilcoxon column reads
#      directly as "did v1.2 move vs v1.2's own control".
#
# The scheme each ckpt trained with is auto-detected from its
# time_rope_aligned_flag buffer; this script VERIFIES the detection
# (v11 ckpt must report time_rope_aligned=False, v12 True) and aborts
# on a mismatch so the ablation can't silently compare the wrong pair.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_V11        REQUIRED. v1.1 (parity RoPE) A.2 run dir or ckpt.
#   CKPT_V12        REQUIRED. v1.2 (time-aligned) A.2 run dir or ckpt.
#   TASK            melchord (default) | drumnondrum
#   MEL_FOLDER / CHORD_FOLDER   eval split; defaults match eval_e1.
#   PROMPT_LENGTH   default 64 (4 bars). GEN_LENGTH default 384 (TOTAL
#                   incl. prompt). N_SAMPLES default 3.
#   MAX_POLYPHONY   prompt tokenization budget for BOTH variants
#                   (default: task convention, 8 melchord / 16 dn).
#                   MAX_POLYPHONY_V11 / MAX_POLYPHONY_V12 override per
#                   variant -- ONLY for evaluating a legacy cp4 v1.1
#                   ckpt, and that confounds the ablation with cp:
#                   the script warns loudly when they differ.
#   A3_REFINE_STEPS/A3_FINAL_TEMP/A3_TOP_P/A3_ADAPTIVE  decode knobs,
#                   same values applied to BOTH variants (defaults
#                   match eval_e1: 4 / 0.9 / 0.95 / 1).
#   SKIP_INFER=1    reuse existing outputs under OUT_ROOT, score only.
#   OUT_ROOT        default temp/a2rope_<task>_<jobid>
#   RESULT_TAG      optional suffix for the results/ filenames.
#
# Example:
#   sbatch --export=ALL,\
#     CKPT_V11=ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_melchord_pop909_nottingham_cp8_batch_16_schedule,\
#     CKPT_V12=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4_melchord_pop909_nottingham_cp8_batch_16_schedule \
#     midi_yinyang/eval_a2_rope_ablation.sbatch

#SBATCH --job-name=a2_rope_ablation
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

CKPT_V11="${CKPT_V11:-}"
CKPT_V12="${CKPT_V12:-}"
TASK="${TASK:-melchord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
N_SAMPLES="${N_SAMPLES:-3}"
SKIP_INFER="${SKIP_INFER:-0}"
RESULT_TAG="${RESULT_TAG:-}"

case "$TASK" in
    melchord)
        MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
        MAX_POLYPHONY="${MAX_POLYPHONY:-8}"
        SPLIT_STREAMS=1
        ;;
    drumnondrum)
        MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"
        MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
        SPLIT_STREAMS=0
        ;;
    *) echo "ERROR: unknown TASK=$TASK"; exit 1 ;;
esac
MAX_POLYPHONY_V11="${MAX_POLYPHONY_V11:-$MAX_POLYPHONY}"
MAX_POLYPHONY_V12="${MAX_POLYPHONY_V12:-$MAX_POLYPHONY}"

OUT_ROOT="${OUT_ROOT:-temp/a2rope_${TASK}_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "A.2 ROPE-GEOMETRY ABLATION   task=$TASK"
echo "CKPT_V11 = ${CKPT_V11:-<UNSET>}"
echo "CKPT_V12 = ${CKPT_V12:-<UNSET>}"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "MAX_POLYPHONY: v11=$MAX_POLYPHONY_V11  v12=$MAX_POLYPHONY_V12"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================================================"

if [[ -z "$CKPT_V11" || -z "$CKPT_V12" ]]; then
    echo "ERROR: both CKPT_V11 and CKPT_V12 are required."
    exit 1
fi
for c in "$CKPT_V11" "$CKPT_V12"; do
    [[ -e "$c" ]] || { echo "ERROR: ckpt not found: $c"; exit 1; }
done
if [[ "$MAX_POLYPHONY_V11" != "$MAX_POLYPHONY_V12" ]]; then
    echo "*** WARNING: variants tokenize prompts at DIFFERENT cp"
    echo "*** ($MAX_POLYPHONY_V11 vs $MAX_POLYPHONY_V12). The comparison is"
    echo "*** then geometry + cp, not the pure geometry ablation."
fi

python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'no CUDA device -- resubmit on a GPU node'
print(f'[gpu] {torch.cuda.get_device_name(0)}')
EOF

# ---------------------------------------------------------------------
# 1. stage prompts (mirrors eval_e1: both extension cases; count gate)
# ---------------------------------------------------------------------
EVAL_MEL="$OUT_ROOT/prompts/mel"
EVAL_CHORD="$OUT_ROOT/prompts/chord"
EVAL_MERGED="$OUT_ROOT/prompts/merged_tagged"
mkdir -p "$EVAL_MEL" "$EVAL_CHORD"
cp -f "$MEL_FOLDER"/*.mid "$MEL_FOLDER"/*.MID "$EVAL_MEL"/ 2>/dev/null || true
cp -f "$CHORD_FOLDER"/*.mid "$CHORD_FOLDER"/*.MID "$EVAL_CHORD"/ 2>/dev/null || true
N_EVAL=$(ls "$EVAL_MEL" 2>/dev/null | wc -l)
if [[ "$N_EVAL" -eq 0 ]]; then
    echo "ERROR: no midis collected from MEL_FOLDER=$MEL_FOLDER"; exit 1
fi
if [[ "$N_EVAL" -ne "$(ls "$EVAL_CHORD" | wc -l)" ]]; then
    echo "ERROR: melody/chord split sizes differ"; exit 1
fi
echo "[prompts] $N_EVAL eval songs"

# merged ground truth = the REF row for the structure metrics
if [[ "$TASK" == "melchord" ]]; then
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" \
        --dst "$EVAL_MERGED" --chord-program 48
else
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" --dst "$EVAL_MERGED"
fi

# ---------------------------------------------------------------------
# 2. inference: both ckpts, identical decode settings
# ---------------------------------------------------------------------
V11_OUT="$OUT_ROOT/A2v11"
V12_OUT="$OUT_ROOT/A2v12"
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
export A3_TOP_P="${A3_TOP_P:-0.95}"
export A3_ADAPTIVE="${A3_ADAPTIVE:-1}"
echo "[decode] K=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P adaptive=$A3_ADAPTIVE (both variants)"

run_variant () {
    local name="$1" ckpt="$2" out="$3" mp="$4" want_time_aligned="$5"
    local log="$OUT_ROOT/infer_${name}.log"
    echo "--- $name co-generation (ckpt: $ckpt) ---"
    python cp_transformer_m2c_duet_block_diffusion_combined.py \
        --ckpt "$ckpt" \
        --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
        --output-dir "$out" \
        --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" \
        --max-polyphony "$mp" \
        --model-size large --moe-num-experts 4 --moe-topk 2 \
        2>&1 | tee "$log"
    # Scheme verification: refuse to score a mislabeled pair.
    local detected
    detected=$(grep -o 'time_rope_aligned=[A-Za-z]*' "$log" | head -1 || true)
    echo "[$name] detected: ${detected:-<no scheme line in log>}"
    if [[ "$detected" != "time_rope_aligned=$want_time_aligned" ]]; then
        echo "ERROR: $name ckpt reports '$detected' but this slot expects"
        echo "       time_rope_aligned=$want_time_aligned. Check CKPT_V11/CKPT_V12"
        echo "       are not swapped (or the v1.2 run trained without"
        echo "       TIME_ROPE_ALIGNED=1)."
        exit 1
    fi
}

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing $V11_OUT / $V12_OUT"
else
    run_variant A2v11 "$CKPT_V11" "$V11_OUT" "$MAX_POLYPHONY_V11" False
    run_variant A2v12 "$CKPT_V12" "$V12_OUT" "$MAX_POLYPHONY_V12" True
fi

# ---------------------------------------------------------------------
# 3a. HEADLINE: long-term structure metrics
# ---------------------------------------------------------------------
RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
STRUCT_OUT="results/a2rope_${TASK}_${RSUF}_structure.txt"
SPLIT_FLAG=()
[[ "$SPLIT_STREAMS" == "1" ]] && SPLIT_FLAG=(--split-streams)
echo "================================================================"
echo "--- structure metrics (headline: read V12 vs V11 vs REF) ---"
python structure_metrics.py \
    --system "A2v11:$V11_OUT" \
    --system "A2v12:$V12_OUT" \
    --system "REF:$EVAL_MERGED" \
    --prompt-bars $((PROMPT_LENGTH / 16)) \
    --max-bars $((GEN_LENGTH / 16)) \
    "${SPLIT_FLAG[@]}" | tee "$STRUCT_OUT"

# ---------------------------------------------------------------------
# 3b. standard E1 metric chain, baseline = v1.1
# ---------------------------------------------------------------------
MANIFEST="results/a2rope_${TASK}_${RSUF}_manifest.tsv"
METRICS="results/a2rope_${TASK}_${RSUF}_metrics.csv"
TABLE="results/a2rope_${TASK}_${RSUF}_table.md"

DUET_LAYOUT=duet
[[ "$N_SAMPLES" -gt 1 ]] && DUET_LAYOUT=duet_multi
EVAL_SONG_IDS=$(ls "$EVAL_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')

echo "--- building manifest ---"
python build_eval_manifest.py \
    --source "A2v11:$DUET_LAYOUT:$V11_OUT" \
    --source "A2v12:$DUET_LAYOUT:$V12_OUT" \
    --songs $EVAL_SONG_IDS \
    --modes co --out "$MANIFEST"

echo "--- scoring ---"
python eval_metrics.py --task "$TASK" \
    --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

echo "--- comparison table (paired Wilcoxon, A2v12 vs baseline A2v11) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline A2v11 --mode co \
    --one-sided --complete-cases --out-md "$TABLE"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline A2v11 --mode co \
    --out-md "${TABLE%.md}_twosided.md"

echo "================================================================"
echo "A.2 rope ablation complete."
echo "  generated midis : $V11_OUT , $V12_OUT"
echo "  structure table : $STRUCT_OUT              <-- READ THIS FIRST"
echo "  metric table    : $TABLE  (+ _twosided)"
echo ""
echo "How to read the structure table:"
echo "  * A2v12 repetition/period4/period8 closer to REF than A2v11,"
echo "    drift_slope less negative -> the geometry fix works; extend"
echo "    the comparison to S1 (structure_metrics.sbatch) to see how"
echo "    much of the gap to the merged baseline it closes."
echo "  * A2v12 ~= A2v11 -> geometry was not the binding constraint;"
echo "    next suspects are empty-frame dilution (register tokens) and"
echo "    the split-form mechanism (D.2)."
echo "================================================================"
