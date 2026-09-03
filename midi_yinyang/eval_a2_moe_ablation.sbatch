#!/bin/bash
# SLURM: the MoE ABLATION evaluation harness -- up to four arms spanning
# the routing spectrum, identical architecture otherwise, identical
# decode settings, identical prompts.
#
# ARM CODES (renamed 2026-09-03; D0 is the model, D1-D3 its departures):
#   D0   per-part gates -- THE DEFAULT MODEL (= A3 elsewhere):
#        per-stream routers over a fully SHARED, unassigned pool.
#        Separation is available but not imposed; specialists and
#        integrators are both representable and the split is learned.
#        (Label 'D3' in trees/results written before 2026-09-03.)
#   D1   dense: no MoE at all, one dense FFN. Cross-part interaction
#        happens ONLY in the attention block. The floor -- is expert
#        routing load-bearing at all?
#   D2   HARD ROUTE: disjoint per-part pools, separation IMPOSED
#        (MoMa / VL-MoE style). The ceiling on separation and the floor
#        on integration -- no expert can serve both streams.
#   D3   shared gate: one router over a shared pool, no per-part
#        anything. (Label 'A2' before 2026-09-03.)
#
# The D2 vs D0 contrast is the experiment's point: both give each
# stream its own routing decision, and they differ ONLY in whether the
# model may also learn to share an expert. D3 and D1 bracket
# them from below.
#
# What it does:
#   1. co-generates with every arm on identical prompts and SAVES the
#      sample midis per (song, arm) under OUT_ROOT -- the listening
#      material for the ablation;
#   2. records per-arm FREE-RUNNING routing stats (MOE_ROUTING_STATS)
#      into results/, extending the utilization evidence to this run;
#   3. scores long-term structure metrics (all arms vs the ground-truth
#      REF row) and the standard E1 chain with BASELINE=D3 (shared
#      gate), so
#      the Wilcoxon column reads directly as "did per-modality gates
#      move vs the shared router".
#
# Arm identity is VERIFIED from load_model's own log lines
# (moe_modality_gates=True/False); a swapped pair aborts before scoring.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_SHARED     REQUIRED. D3: shared-gate run dir/ckpt.
#   CKPT_MG         REQUIRED. D0: per-part-gate run dir (K4mg family).
#   CKPT_HR         optional. D2: A.2.moe_hardroute run dir (K4hr
#                   family) -- the disjoint-pool control. HR_EXPERTS/
#                   HR_TOPK (default 4/2) must match how it was TRAINED.
#   CKPT_DENSE      optional. D1: dense arm (moe_num_experts=1, inter
#                   6144).
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
CKPT_HR="${CKPT_HR:-}"
CKPT_DENSE="${CKPT_DENSE:-}"
# Hard-route pool geometry. The default 4/2 matches the other arms in
# parameters AND activated compute (2 experts per token) -- the fair
# comparison -- at the cost of making within-pool routing a weighting
# rather than a selection (top-2 of a 2-expert pool is both). Set
# HR_EXPERTS=8 HR_TOPK=2 if the hard-route ckpt was trained with
# sparsity preserved inside the pools; the arm then has 2x expert
# parameters and the table must say so.
HR_EXPERTS="${HR_EXPERTS:-4}"
HR_TOPK="${HR_TOPK:-2}"
TASK="${TASK:-melchord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
N_SAMPLES="${N_SAMPLES:-3}"
SKIP_INFER="${SKIP_INFER:-0}"
RESULT_TAG="${RESULT_TAG:-}"
# Wilcoxon baseline arm for the table. D3 (shared gate) keeps the
# historical reading; BASELINE=D0 tests every departure against the
# default model instead.
BASELINE="${BASELINE:-D3}"

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
echo "E6 MoE ABLATION (D0 / D1 / D2 / D3)   task=$TASK"
echo "CKPT_SHARED = ${CKPT_SHARED:-<UNSET>}"
echo "CKPT_MG     = ${CKPT_MG:-<UNSET>}"
echo "CKPT_HR     = ${CKPT_HR:-<none>}${CKPT_HR:+  (pools ${HR_EXPERTS}/2, topk $HR_TOPK)}"
echo "CKPT_DENSE  = ${CKPT_DENSE:-<none>}"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "MAX_POLYPHONY=$MAX_POLYPHONY   OUT_ROOT=$OUT_ROOT"
echo "================================================================"

if [[ -z "$CKPT_SHARED" || -z "$CKPT_MG" ]]; then
    echo "ERROR: both CKPT_SHARED and CKPT_MG are required."
    exit 1
fi
for c in "$CKPT_SHARED" "$CKPT_MG" ${CKPT_HR:+"$CKPT_HR"} \
         ${CKPT_DENSE:+"$CKPT_DENSE"}; do
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
SHARED_OUT="$OUT_ROOT/D3"
MG_OUT="$OUT_ROOT/D0"
HR_OUT="$OUT_ROOT/D2"
DENSE_OUT="$OUT_ROOT/D1"
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
export A3_TOP_P="${A3_TOP_P:-0.95}"
export A3_ADAPTIVE="${A3_ADAPTIVE:-1}"
echo "[decode] K=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P adaptive=$A3_ADAPTIVE (all arms)"

run_arm () {
    local name="$1" ckpt="$2" out="$3" want_gates="$4" want_hr="$5" \
          experts="$6" topk="$7" inter="$8"
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
    # Arm-identity verification from load_model's own auto-detect lines:
    # refuse to score a mislabeled set. Every router variant is
    # distinguishable this way, so a swapped CKPT_* aborts before any
    # number is produced.
    local detected
    if [[ -n "$want_gates" ]]; then
        detected=$(grep -o 'moe_modality_gates=[A-Za-z]*' "$log" | head -1 || true)
        echo "[$name] detected: ${detected:-<no gates line in log>}"
        if [[ "$detected" != "moe_modality_gates=$want_gates" ]]; then
            echo "ERROR: $name ckpt reports '$detected' but this slot expects"
            echo "       moe_modality_gates=$want_gates. The CKPT_* variables"
            echo "       are likely swapped."
            exit 1
        fi
    fi
    if [[ -n "$want_hr" ]]; then
        detected=$(grep -o 'moe_modality_hard_route=[A-Za-z]*' "$log" | head -1 || true)
        echo "[$name] detected: ${detected:-<no hard-route line in log>}"
        if [[ "$detected" != "moe_modality_hard_route=$want_hr" ]]; then
            echo "ERROR: $name ckpt reports '$detected' but this slot expects"
            echo "       moe_modality_hard_route=$want_hr. The CKPT_* variables"
            echo "       are likely swapped."
            exit 1
        fi
    fi
}

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing outputs under $OUT_ROOT"
else
    #        name      ckpt            out           gates hr    E  k  inter
    run_arm D3 "$CKPT_SHARED" "$SHARED_OUT" False False 4 2 ""
    run_arm D0     "$CKPT_MG"     "$MG_OUT"     True  False 4 2 ""
    if [[ -n "$CKPT_HR" ]]; then
        run_arm D2 "$CKPT_HR" "$HR_OUT" False True \
            "$HR_EXPERTS" "$HR_TOPK" ""
    fi
    if [[ -n "$CKPT_DENSE" ]]; then
        run_arm D1 "$CKPT_DENSE" "$DENSE_OUT" "" "" 1 1 6144
    fi
fi

# ---------------------------------------------------------------------
# 3a. long-term structure metrics (all arms vs REF)
# ---------------------------------------------------------------------
RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
STRUCT_OUT="results/a2moe_${TASK}_${RSUF}_structure.txt"
SPLIT_FLAG=()
[[ "$SPLIT_STREAMS" == "1" ]] && SPLIT_FLAG=(--split-streams)
HR_SYS=()
[[ -n "$CKPT_HR" ]] && HR_SYS=(--system "D2:$HR_OUT")
DENSE_SYS=()
[[ -n "$CKPT_DENSE" ]] && DENSE_SYS=(--system "D1:$DENSE_OUT")
echo "================================================================"
echo "--- structure metrics (all arms vs REF) ---"
python structure_metrics.py \
    --system "D3:$SHARED_OUT" \
    --system "D0:$MG_OUT" \
    "${HR_SYS[@]}" \
    "${DENSE_SYS[@]}" \
    --system "REF:$EVAL_MERGED" \
    --prompt-bars $((PROMPT_LENGTH / 16)) \
    --max-bars $((GEN_LENGTH / 16)) \
    "${SPLIT_FLAG[@]}" | tee "$STRUCT_OUT"

# ---------------------------------------------------------------------
# 3b. standard E1 metric chain, baseline = shared gate (D3) by default
# ---------------------------------------------------------------------
MANIFEST="results/a2moe_${TASK}_${RSUF}_manifest.tsv"
METRICS="results/a2moe_${TASK}_${RSUF}_metrics.csv"
TABLE="results/a2moe_${TASK}_${RSUF}_table.md"

DUET_LAYOUT=duet
[[ "$N_SAMPLES" -gt 1 ]] && DUET_LAYOUT=duet_multi
EVAL_SONG_IDS=$(ls "$EVAL_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
HR_SRC=()
[[ -n "$CKPT_HR" ]] && HR_SRC=(--source "D2:$DUET_LAYOUT:$HR_OUT")
DENSE_SRC=()
[[ -n "$CKPT_DENSE" ]] && DENSE_SRC=(--source "D1:$DUET_LAYOUT:$DENSE_OUT")

echo "--- building manifest ---"
python build_eval_manifest.py \
    --source "D3:$DUET_LAYOUT:$SHARED_OUT" \
    --source "D0:$DUET_LAYOUT:$MG_OUT" \
    "${HR_SRC[@]}" \
    "${DENSE_SRC[@]}" \
    --songs $EVAL_SONG_IDS \
    --modes co --out "$MANIFEST"

echo "--- scoring ---"
python eval_metrics.py --task "$TASK" \
    --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

echo "--- comparison table (paired Wilcoxon vs baseline $BASELINE) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline "$BASELINE" --mode co \
    --one-sided --complete-cases --out-md "$TABLE"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline "$BASELINE" --mode co \
    --out-md "${TABLE%.md}_twosided.md"

echo "================================================================"
echo "E6 MoE ablation complete."
echo "  SAVED SAMPLES   : $SHARED_OUT , $MG_OUT${CKPT_HR:+ , $HR_OUT}${CKPT_DENSE:+ , $DENSE_OUT}"
echo "                    (one .mid per song/sample per arm -- the"
echo "                     listening material; same prompts everywhere)"
echo "  routing stats   : results/routing_stats_*_${SLURM_JOB_ID:-local}.json"
echo "  structure table : $STRUCT_OUT"
echo "  metric table    : $TABLE  (+ _twosided)"
echo ""
echo "How to read: quality parity (no significant D0 deficits vs D3)"
echo "already supports E6 -- removing the router redundancy costs nothing"
echo "and buys the mechanism results (emergent structure, chord-side"
echo "content routing, better balance). The D2-vs-D0 column is the"
echo "experiment's point: if D0 wins, learned integrators do work that"
echo "imposed pools cannot express. Gains, if any, should land on H3"
echo "stream-grammar metrics."
echo "================================================================"
