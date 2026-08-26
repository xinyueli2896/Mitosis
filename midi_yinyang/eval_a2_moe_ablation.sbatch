#!/bin/bash
# SLURM: E4d router-design ablation -- shared gate vs per-modality gates
# (A.2.moe_permod), same architecture otherwise, same decode settings,
# same prompts. Optionally adds the E4c dense arm for a three-way
# routing spectrum (none / shared / per-modality).
#
# What it does:
#   1. co-generates with every arm on identical prompts and SAVES the
#      sample midis per (song, arm) under OUT_ROOT -- the listening
#      material for the ablation;
#   2. records per-arm FREE-RUNNING routing stats (MOE_ROUTING_STATS)
#      into results/, extending the utilization evidence to this run;
#   3. scores long-term structure metrics (all arms vs the ground-truth
#      REF row) and the standard E1 chain with BASELINE=A2shared, so
#      the Wilcoxon column reads directly as "did per-modality gates
#      move vs the shared router".
#
# Arm identity is VERIFIED from load_model's own log lines
# (moe_modality_gates=True/False); a swapped pair aborts before scoring.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_SHARED     REQUIRED. Standard A.2 run dir or ckpt (shared gate).
#   CKPT_MG         REQUIRED. A.2.moe_permod run dir (K4mg family).
#   CKPT_DENSE      optional. E4c dense arm (moe_num_experts=1); scored
#                   as a third system when set.
#   TASK            melchord (default) | drumnondrum
#   MEL_FOLDER / CHORD_FOLDER   eval prompts; defaults match eval_e1.
#   PROMPT_LENGTH   default 64. GEN_LENGTH default 384 (TOTAL incl.
#                   prompt). N_SAMPLES default 3.
#   MAX_POLYPHONY   prompt tokenization budget for ALL arms. Default:
#                   task convention (8 melchord / 16 dn). The cp4tar
#                   ablation family needs MAX_POLYPHONY=4.
#   A3_REFINE_STEPS/A3_FINAL_TEMP/A3_TOP_P/A3_ADAPTIVE  decode knobs,
#                   same values for every arm (defaults match eval_e1).
#   SKIP_INFER=1    reuse existing outputs under OUT_ROOT, score only.
#   OUT_ROOT        default temp/a2moe_<task>_<jobid>
#   RESULT_TAG      optional suffix for the results/ filenames.
#
# Example (the cp4tar POP909 pair):
#   sbatch --exclude=gpu-50,gpu-51 --export=ALL,MAX_POLYPHONY=4,\
# CKPT_SHARED=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4_melchord_cp4tar_batch_8_schedule,\
# CKPT_MG=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mg_melchord_cp4tar_batch_8_schedule \
#     midi_yinyang/eval_a2_moe_ablation.sbatch

#SBATCH --job-name=a2_moe_ablation
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

CKPT_SHARED="${CKPT_SHARED:-}"
CKPT_MG="${CKPT_MG:-}"
CKPT_DENSE="${CKPT_DENSE:-}"
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

OUT_ROOT="${OUT_ROOT:-temp/a2moe_${TASK}_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E4d ROUTER-DESIGN ABLATION   task=$TASK"
echo "CKPT_SHARED = ${CKPT_SHARED:-<UNSET>}"
echo "CKPT_MG     = ${CKPT_MG:-<UNSET>}"
echo "CKPT_DENSE  = ${CKPT_DENSE:-<none -- two-arm run>}"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "MAX_POLYPHONY=$MAX_POLYPHONY   OUT_ROOT=$OUT_ROOT"
echo "================================================================"

if [[ -z "$CKPT_SHARED" || -z "$CKPT_MG" ]]; then
    echo "ERROR: both CKPT_SHARED and CKPT_MG are required."
    exit 1
fi
for c in "$CKPT_SHARED" "$CKPT_MG" ${CKPT_DENSE:+"$CKPT_DENSE"}; do
    [[ -e "$c" ]] || { echo "ERROR: ckpt not found: $c"; exit 1; }
done

python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'no CUDA device -- resubmit on a GPU node'
print(f'[gpu] {torch.cuda.get_device_name(0)}')
EOF

# ---------------------------------------------------------------------
# 1. stage prompts (mirrors eval_a2_rope_ablation / eval_e1)
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

if [[ "$TASK" == "melchord" ]]; then
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" \
        --dst "$EVAL_MERGED" --chord-program 48
else
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" --dst "$EVAL_MERGED"
fi

# ---------------------------------------------------------------------
# 2. inference: every arm, identical decode settings + per-arm
#    free-running routing stats
# ---------------------------------------------------------------------
SHARED_OUT="$OUT_ROOT/A2shared"
MG_OUT="$OUT_ROOT/A2mg"
DENSE_OUT="$OUT_ROOT/A2dense"
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
export A3_TOP_P="${A3_TOP_P:-0.95}"
export A3_ADAPTIVE="${A3_ADAPTIVE:-1}"
echo "[decode] K=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P adaptive=$A3_ADAPTIVE (all arms)"

run_arm () {
    local name="$1" ckpt="$2" out="$3" want_gates="$4" experts="$5" topk="$6" inter="$7"
    local log="$OUT_ROOT/infer_${name}.log"
    local extra=()
    [[ -n "$inter" ]] && extra=(--moe-intermediate-size "$inter")
    echo "--- $name co-generation (ckpt: $ckpt) ---"
    MOE_ROUTING_STATS="results/routing_stats_${name}_${SLURM_JOB_ID:-local}.json" \
    python cp_transformer_m2c_duet_block_diffusion_combined.py \
        --ckpt "$ckpt" \
        --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
        --output-dir "$out" \
        --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" \
        --max-polyphony "$MAX_POLYPHONY" \
        --model-size large --moe-num-experts "$experts" --moe-topk "$topk" \
        "${extra[@]}" \
        2>&1 | tee "$log"
    # Arm-identity verification: refuse to score a mislabeled pair.
    if [[ -n "$want_gates" ]]; then
        local detected
        detected=$(grep -o 'moe_modality_gates=[A-Za-z]*' "$log" | head -1 || true)
        echo "[$name] detected: ${detected:-<no gates line in log>}"
        if [[ "$detected" != "moe_modality_gates=$want_gates" ]]; then
            echo "ERROR: $name ckpt reports '$detected' but this slot expects"
            echo "       moe_modality_gates=$want_gates. CKPT_SHARED/CKPT_MG"
            echo "       are likely swapped."
            exit 1
        fi
    fi
}

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing outputs under $OUT_ROOT"
else
    run_arm A2shared "$CKPT_SHARED" "$SHARED_OUT" False 4 2 ""
    run_arm A2mg     "$CKPT_MG"     "$MG_OUT"     True  4 2 ""
    if [[ -n "$CKPT_DENSE" ]]; then
        run_arm A2dense "$CKPT_DENSE" "$DENSE_OUT" "" 1 1 6144
    fi
fi

# ---------------------------------------------------------------------
# 3a. long-term structure metrics (all arms vs REF)
# ---------------------------------------------------------------------
RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
STRUCT_OUT="results/a2moe_${TASK}_${RSUF}_structure.txt"
SPLIT_FLAG=()
[[ "$SPLIT_STREAMS" == "1" ]] && SPLIT_FLAG=(--split-streams)
DENSE_SYS=()
[[ -n "$CKPT_DENSE" ]] && DENSE_SYS=(--system "A2dense:$DENSE_OUT")
echo "================================================================"
echo "--- structure metrics (all arms vs REF) ---"
python structure_metrics.py \
    --system "A2shared:$SHARED_OUT" \
    --system "A2mg:$MG_OUT" \
    "${DENSE_SYS[@]}" \
    --system "REF:$EVAL_MERGED" \
    --prompt-bars $((PROMPT_LENGTH / 16)) \
    --max-bars $((GEN_LENGTH / 16)) \
    "${SPLIT_FLAG[@]}" | tee "$STRUCT_OUT"

# ---------------------------------------------------------------------
# 3b. standard E1 metric chain, baseline = shared router
# ---------------------------------------------------------------------
MANIFEST="results/a2moe_${TASK}_${RSUF}_manifest.tsv"
METRICS="results/a2moe_${TASK}_${RSUF}_metrics.csv"
TABLE="results/a2moe_${TASK}_${RSUF}_table.md"

DUET_LAYOUT=duet
[[ "$N_SAMPLES" -gt 1 ]] && DUET_LAYOUT=duet_multi
EVAL_SONG_IDS=$(ls "$EVAL_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
DENSE_SRC=()
[[ -n "$CKPT_DENSE" ]] && DENSE_SRC=(--source "A2dense:$DUET_LAYOUT:$DENSE_OUT")

echo "--- building manifest ---"
python build_eval_manifest.py \
    --source "A2shared:$DUET_LAYOUT:$SHARED_OUT" \
    --source "A2mg:$DUET_LAYOUT:$MG_OUT" \
    "${DENSE_SRC[@]}" \
    --songs $EVAL_SONG_IDS \
    --modes co --out "$MANIFEST"

echo "--- scoring ---"
python eval_metrics.py --task "$TASK" \
    --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

echo "--- comparison table (paired Wilcoxon vs baseline A2shared) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline A2shared --mode co \
    --one-sided --complete-cases --out-md "$TABLE"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline A2shared --mode co \
    --out-md "${TABLE%.md}_twosided.md"

echo "================================================================"
echo "E4d router-design ablation complete."
echo "  SAVED SAMPLES   : $SHARED_OUT , $MG_OUT${CKPT_DENSE:+ , $DENSE_OUT}"
echo "                    (one .mid per song/sample per arm -- the"
echo "                     listening material; same prompts everywhere)"
echo "  routing stats   : results/routing_stats_A2*_${SLURM_JOB_ID:-local}.json"
echo "  structure table : $STRUCT_OUT"
echo "  metric table    : $TABLE  (+ _twosided)"
echo ""
echo "How to read: quality parity (no significant A2mg deficits) already"
echo "supports E4d -- the router redundancy costs nothing and buys the"
echo "mechanism results (emergent structure, chord-side content routing,"
echo "better balance). Gains, if any, should land on H3 stream-grammar"
echo "metrics, mirroring E4c's logic."
echo "================================================================"
