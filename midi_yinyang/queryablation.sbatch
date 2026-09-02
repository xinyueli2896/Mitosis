#!/bin/bash
# SLURM (GPU): query-slot ablation, ONE JOB -- co-generation decodes for
# A1 (intra-cross-attn, NO query slots: the control) and A3/A4/A5/A6 on
# the POP909 test split, then the quantitative melody-chord consistency
# scoring, all in a single allocation (the multi-job chain version hit
# QOSMaxSubmitJobPerUserLimit). Tables: vs A3 (variant-level) and vs A1
# (design-level: do query slots help at all?).
#
# Per model: co mode, 6-bar prompt (96 frames), 416 total frames,
# 3 samples/song, identical decode schedule (REFINE 4, temp 0.9,
# top-p 0.95) -- differences between models are the training objective,
# never the decoder. CKPTs are run DIRECTORIES: best-val selection
# applies (crucial for A6, whose minima sit at ~15-27k).
#
# Listening outputs: temp/bakeoff/<model>_co/<song>/co/sample_<i>_temp*.mid
# Quantitative outputs (H2 = the melody<->chord consistency block:
# harmonic_coupling, ctnctr, pcs, mctd, chord_tone_cov, each with
# _ref/_delta; paired Wilcoxon vs A3):
#   results/queryablation_manifest.tsv
#   results/queryablation_metrics.csv
#   results/queryablation_table.md            (one-sided, complete cases)
#   results/queryablation_table_twosided.md   (conservative reference)
#
# Idempotent: each finished model decode leaves a .done marker and is
# skipped on rerun, so a timeout resumes where it stopped instead of
# regenerating everything. FORCE=1 regenerates all.
#
# Env knobs (--export=ALL,KEY=val):
#   PROMPT_LENGTH default 96 (6 bars), GEN_LENGTH default 416,
#   N_SAMPLES default 3, BASELINE default A3, FORCE default 0
#
# Example:
#   sbatch midi_yinyang/queryablation.sbatch

#SBATCH --job-name=queryablation
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

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs results temp/bakeoff

PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
BASELINE="${BASELINE:-A3}"
FORCE="${FORCE:-0}"
MEL_FOLDER=input/pop909_split/melody
CHORD_FOLDER=input/pop909_split/chord

# Identical decode schedule for every model (the listening-test winner).
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
export A3_TOP_P="${A3_TOP_P:-0.95}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "QUERY-SLOT ABLATION: A3/A4/A5/A6 co decode + H-metric scoring"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)"
echo "GEN_LENGTH=$GEN_LENGTH  N_SAMPLES=$N_SAMPLES  BASELINE=$BASELINE"
echo "REFINE=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P"
echo "================================================================"
nvidia-smi | head -n 12 || true

# GPU pre-flight (a dead GPU should fail here, not mid-decode).
python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"

# ---- 1. co-generation decodes, sequential ------------------------------
# Skip logic is CONTENT-based, not marker-based: a model's decode is
# considered done when every test song already has N_SAMPLES co samples
# in the duet_multi layout -- so outputs produced by earlier jobs count,
# and a run killed mid-model resumes at the incomplete model. The .done
# marker is kept as a fast path. FORCE=1 regenerates everything.
have_all_outputs() {
    local out="$1" sid n
    for f in "$MEL_FOLDER"/*.mid; do
        sid=$(basename "$f" .mid)
        n=$(ls "$out/$sid/co/"*.mid 2>/dev/null | wc -l)
        [[ "$n" -ge "$N_SAMPLES" ]] || return 1
    done
    return 0
}

# A1 first: intra-cross-attn, NO query slots at all -- the control the
# whole query-slot line is measured against. Different inference script,
# same decode knobs and output layout.
A1_CKPT=ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_melchord_cp4tar_batch_8_schedule
A1_OUT=temp/bakeoff/A1_co
echo "----------------------------------------------------------------"
if [[ "$FORCE" != "1" ]] && { [[ -f "$A1_OUT/.done" ]] || have_all_outputs "$A1_OUT"; }; then
    echo "[A1] SKIP (outputs already complete)"
    touch "$A1_OUT/.done" 2>/dev/null || true
else
    [[ -d "$A1_CKPT" ]] || { echo "ERROR: ckpt dir $A1_CKPT not found"; exit 1; }
    echo "[A1] decoding from $A1_CKPT/ -> $A1_OUT"
    python cp_transformer_m2c_intra_cross_attn_combined.py \
        --ckpt "$A1_CKPT/" \
        --mel-folder "$MEL_FOLDER" --chord-folder "$CHORD_FOLDER" \
        --output-dir "$A1_OUT" \
        --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" \
        --max-polyphony 4 \
        --model-size large --moe-num-experts 4 --moe-topk 2
    touch "$A1_OUT/.done"
    echo "[A1] done"
fi

MODELS=(A3 A4 A5 A6)
declare -A CKPTS=(
    [A3]=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mg_melchord_pop909_pop1k7_combined_long_batch_8_schedule
    [A4]=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A4_melchord_long_batch_8_schedule
    [A5]=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A5_melchord_long_batch_8_schedule
    [A6]=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A6_melchord_long_batch_8_schedule
)

for m in "${MODELS[@]}"; do
    ckpt="${CKPTS[$m]}"
    out="temp/bakeoff/${m}_co"
    echo "----------------------------------------------------------------"
    if [[ "$FORCE" != "1" ]] && { [[ -f "$out/.done" ]] || have_all_outputs "$out"; }; then
        echo "[$m] SKIP (outputs already complete)"
        touch "$out/.done" 2>/dev/null || true
        continue
    fi
    [[ -d "$ckpt" ]] || { echo "ERROR: ckpt dir $ckpt not found"; exit 1; }
    echo "[$m] decoding from $ckpt/ -> $out"
    python cp_transformer_m2c_duet_block_diffusion_combined.py \
        --ckpt "$ckpt/" \
        --mel-folder "$MEL_FOLDER" \
        --chord-folder "$CHORD_FOLDER" \
        --output-dir "$out" \
        --modes co \
        --prompt-length "$PROMPT_LENGTH" \
        --gen-length "$GEN_LENGTH" \
        --temperature 1.0 \
        --max-polyphony 4 \
        --model-size large \
        --moe-num-experts 4 --moe-topk 2 \
        --min-chord-tokens-before-eos 0 \
        --n-samples "$N_SAMPLES"
    touch "$out/.done"
    echo "[$m] done"
done

# ---- 2. scoring --------------------------------------------------------
# The split folders hold exactly the 5 test songs, so they serve directly
# as scoring references; the window (--prompt-frames..--total-frames)
# scores only what was GENERATED.
echo "================================================================"
SONG_IDS=$(ls "$MEL_FOLDER" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
echo "[songs] $SONG_IDS"

MANIFEST=results/queryablation_manifest.tsv
METRICS=results/queryablation_metrics.csv
TABLE=results/queryablation_table.md

echo "--- building manifest ---"
python build_eval_manifest.py \
    --source "A1:duet_multi:temp/bakeoff/A1_co" \
    --source "A3:duet_multi:temp/bakeoff/A3_co" \
    --source "A4:duet_multi:temp/bakeoff/A4_co" \
    --source "A5:duet_multi:temp/bakeoff/A5_co" \
    --source "A6:duet_multi:temp/bakeoff/A6_co" \
    --songs $SONG_IDS --modes co --out "$MANIFEST"

echo "--- scoring (H1-H3) ---"
python eval_metrics.py --task melchord \
    --manifest "$MANIFEST" \
    --ref-a-dir "$MEL_FOLDER" --ref-b-dir "$CHORD_FOLDER" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

echo "--- comparison table (paired Wilcoxon vs $BASELINE) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task melchord --baseline "$BASELINE" --mode co \
    --one-sided --complete-cases --out-md "$TABLE"
python aggregate_eval_results.py --csv "$METRICS" \
    --task melchord --baseline "$BASELINE" --mode co \
    --out-md "${TABLE%.md}_twosided.md"

# Second baseline: A1 = no query slots. "Does ANY query-slot design
# improve melody-chord consistency over plain intra-cross-attn?" is the
# design-level question; vs-A3 above is the variant-level one.
echo "--- comparison table (paired Wilcoxon vs A1, no-query-slot control) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task melchord --baseline A1 --mode co \
    --one-sided --complete-cases --out-md "${TABLE%.md}_vsA1.md"

echo "================================================================"
echo "RESULT TABLE (one-sided, complete cases, vs $BASELINE):"
cat "$TABLE"
echo "================================================================"
echo "RESULT TABLE (one-sided, complete cases, vs A1 control):"
cat "${TABLE%.md}_vsA1.md"
echo "================================================================"
echo "Listening midis: temp/bakeoff/<model>_co/<song>/co/sample_<i>_temp*.mid"
echo "Focus rows for the query-slot hypothesis: harmonic_coupling,"
echo "ctnctr, pcs, mctd, chord_tone_cov (the H2 consistency block)."
echo "================================================================"
