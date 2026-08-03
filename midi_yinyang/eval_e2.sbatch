#!/bin/bash
# SLURM: END-TO-END E2 (marginal single-stream generation, melchord).
#
# E2 asks RQ2: does the duet model RETAIN single-stream (marginal)
# capability, or did two-stream training tax it? Exactly one stream is
# generated; the partner is absent. Per EXPERIMENTS.md the pairing is
# PER-STREAM and never crossed:
#
#   A.2 `mel_only`   vs  S-mel    scored on MELODY (stream-a) metrics
#   A.2 `chord_only` vs  S-chord  scored on CHORD  (stream-b) metrics
#
# Reading the result (H-E2.1): this is a NON-INFERIORITY question, not a
# race. The specialist is expected to be at least as good -- it does
# nothing else. A.2 "retains the marginal" if it lands within a margin
# delta of the specialist on each primary H3 grammar metric; a WIN for
# A.2 is a bonus finding, not the bar. Tables are therefore two-sided --
# a one-sided superiority test points the wrong way for this hypothesis.
#
# CAVEAT to carry into the writeup: A.2's marginal modes clamp the absent
# stream and effectively decode at K~=0 (AR pathway, no refinement), so
# E2 measures the BACKBONE's marginal capability, not the refinement
# machinery.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_A2       A.2 run dir or ckpt file             (required)
#   CKPT_SMEL     S-mel  run dir or ckpt file          (required)
#   CKPT_SCHORD   S-chord run dir or ckpt file         (required)
#   MEL_FOLDER    default input/pop909_split/melody   (prompts + refs)
#   CHORD_FOLDER  default input/pop909_split/chord
#   PROMPT_LENGTH default 64 frames (4 bars; 96 = 6 bars)
#   GEN_LENGTH    default 384 TOTAL frames incl. prompt (416 with p96)
#   N_SAMPLES     default 3 (same count for all three systems)
#   RESULT_TAG    optional extra suffix on result filenames
#   SKIP_INFER=1  reuse existing outputs under OUT_ROOT, score only
#   CHORD_OUT_PROGRAMS  programs that identify CHORD content in the
#                 UNNAMED S-chord outputs (default "0,48").
#                 cp_transformer_inference writes plain program
#                 instruments with no track names, so scoring falls back
#                 to program matching; if S-chord's rows come out empty,
#                 its output program is not in this list -- inspect one
#                 midi and override.
#
# Example:
#   sbatch --export=ALL,\
# CKPT_A2=ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_melchord_pop909_nottingham_batch_8_schedule/,\
# CKPT_SMEL=ckpt/cp_transformer_v0.42_size1_pop909ft_mel_combined_batch_24_schedule/,\
# CKPT_SCHORD=ckpt/cp_transformer_v0.42_size1_pop909ft_chord_combined_batch_24_schedule/ \
#       midi_yinyang/eval_e2.sbatch

#SBATCH --job-name=mitosis_eval_e2
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

CKPT_A2="${CKPT_A2:-}"
CKPT_SMEL="${CKPT_SMEL:-}"
CKPT_SCHORD="${CKPT_SCHORD:-}"
MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
N_SAMPLES="${N_SAMPLES:-3}"
RESULT_TAG="${RESULT_TAG:-}"
SKIP_INFER="${SKIP_INFER:-0}"
CHORD_OUT_PROGRAMS="${CHORD_OUT_PROGRAMS:-0,48}"
OUT_ROOT="${OUT_ROOT:-temp/e2_melchord_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

for v in CKPT_A2 CKPT_SMEL CKPT_SCHORD; do
    if [[ -z "${!v}" ]]; then
        echo "ERROR: $v is required (E2 has no optional systems: each"
        echo "       pairing needs both its members)."
        exit 1
    fi
done

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E2 MARGINAL (SINGLE-STREAM) EVALUATION   task=melchord"
echo "CKPT_A2     = $CKPT_A2"
echo "CKPT_SMEL   = $CKPT_SMEL"
echo "CKPT_SCHORD = $CKPT_SCHORD"
echo "prompts     = $MEL_FOLDER / $CHORD_FOLDER"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)  "\
"GEN_LENGTH=$GEN_LENGTH ($((GEN_LENGTH / 16)) bars)  N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================================================"
nvidia-smi | head -n 12 || true

# GPU pre-flight (all three generation stages call model.cuda()).
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

python check_ckpts.py "$CKPT_A2" "$CKPT_SMEL" "$CKPT_SCHORD" || {
    echo "ERROR: unreadable checkpoint(s) -- see above."; exit 1; }

for d in "$MEL_FOLDER" "$CHORD_FOLDER"; do
    [[ -d "$d" ]] || { echo "ERROR: $d not found"; exit 1; }
done
N_MEL=$(ls "$MEL_FOLDER"/*.mid 2>/dev/null | wc -l)
N_CHD=$(ls "$CHORD_FOLDER"/*.mid 2>/dev/null | wc -l)
if [[ "$N_MEL" -eq 0 || "$N_MEL" -ne "$N_CHD" ]]; then
    echo "ERROR: prompt folders must hold the SAME songs ($N_MEL vs $N_CHD)."
    exit 1
fi
SONG_IDS=$(ls "$MEL_FOLDER" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
echo "[prompts] $N_MEL songs: $SONG_IDS"

A2_OUT="$OUT_ROOT/A.2"
SMEL_OUT_NAME="e2_smel_${SLURM_JOB_ID:-local}"
SCHORD_OUT_NAME="e2_schord_${SLURM_JOB_ID:-local}"

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing existing outputs"
else
    # -----------------------------------------------------------------
    # 1. A.2 marginal modes (absent stream clamped; effectively K~=0)
    # -----------------------------------------------------------------
    echo "--- A.2 mel_only + chord_only ---"
    python cp_transformer_m2c_duet_block_diffusion_combined.py \
        --ckpt "$CKPT_A2" \
        --mel-folder "$MEL_FOLDER" --chord-folder "$CHORD_FOLDER" \
        --output-dir "$A2_OUT" \
        --modes mel_only chord_only \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" \
        --max-polyphony 4 \
        --model-size large --moe-num-experts 4 --moe-topk 2

    # -----------------------------------------------------------------
    # 2. specialists. Two easy-to-miss requirements:
    #    --max-polyphony 4: they are finetuned on cp4 single-stream data
    #      (8-token frames); the default 16 is a silent train/inference
    #      mismatch.
    #    --no-split-tracks for S-chord: its prompt files carry a track
    #      named CHORD, and the default track-tagging would force that
    #      track to program 48 during tokenization -- a program the
    #      specialist never saw in training. (Harmless for S-mel; passed
    #      to both for symmetry.)
    # -----------------------------------------------------------------
    echo "--- S-mel single-stream continuation (melody prompts) ---"
    python cp_transformer_inference.py \
        --ckpt "$CKPT_SMEL" \
        --midi-folder "$MEL_FOLDER" \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" \
        --max-polyphony 4 --no-split-tracks \
        --save-name "$SMEL_OUT_NAME"

    echo "--- S-chord single-stream continuation (chord prompts) ---"
    python cp_transformer_inference.py \
        --ckpt "$CKPT_SCHORD" \
        --midi-folder "$CHORD_FOLDER" \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" \
        --max-polyphony 4 --no-split-tracks \
        --save-name "$SCHORD_OUT_NAME"
fi

# ---------------------------------------------------------------------
# 3. score: one manifest + metrics + table PER PAIRING. The pairings
#    use different modes AND different baselines, so mixing them in one
#    table would let --complete-cases intersect across unrelated
#    comparisons (the exact failure --systems was added to prevent).
# ---------------------------------------------------------------------
if [[ "$N_SAMPLES" -gt 1 ]]; then A2_LAYOUT=duet_multi; else A2_LAYOUT=duet; fi
RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"

echo "=== pairing 1: A.2(mel_only) vs S-mel (melody metrics) ==="
MAN_MEL="results/e2_melchord_${RSUF}_mel_manifest.tsv"
MET_MEL="results/e2_melchord_${RSUF}_mel_metrics.csv"
TAB_MEL="results/e2_melchord_${RSUF}_mel_table.md"
python build_eval_manifest.py \
    --source "A.2:${A2_LAYOUT}:${A2_OUT}" \
    --source "S-mel:single:temp/${SMEL_OUT_NAME}" \
    --songs $SONG_IDS --modes mel_only --mode mel_only \
    --out "$MAN_MEL"
python eval_metrics.py --task melchord \
    --manifest "$MAN_MEL" \
    --ref-a-dir "$MEL_FOLDER" --ref-b-dir "$CHORD_FOLDER" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$MET_MEL"
python aggregate_eval_results.py --csv "$MET_MEL" \
    --task melchord --baseline S-mel --mode mel_only \
    --complete-cases --systems A.2 S-mel \
    --out-md "$TAB_MEL"

echo "=== pairing 2: A.2(chord_only) vs S-chord (chord metrics) ==="
# S-chord outputs are unnamed single-instrument midis, so stream
# attribution falls back to PROGRAM matching: route its programs to
# stream b (chord) and keep stream a's fallback away from them.
MAN_CHD="results/e2_melchord_${RSUF}_chord_manifest.tsv"
MET_CHD="results/e2_melchord_${RSUF}_chord_metrics.csv"
TAB_CHD="results/e2_melchord_${RSUF}_chord_table.md"
python build_eval_manifest.py \
    --source "A.2:${A2_LAYOUT}:${A2_OUT}" \
    --source "S-chord:single:temp/${SCHORD_OUT_NAME}" \
    --songs $SONG_IDS --modes chord_only --mode chord_only \
    --out "$MAN_CHD"
python eval_metrics.py --task melchord \
    --manifest "$MAN_CHD" \
    --ref-a-dir "$MEL_FOLDER" --ref-b-dir "$CHORD_FOLDER" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --mel-programs 24 --chord-programs "$CHORD_OUT_PROGRAMS" \
    --out "$MET_CHD"
python aggregate_eval_results.py --csv "$MET_CHD" \
    --task melchord --baseline S-chord --mode chord_only \
    --complete-cases --systems A.2 S-chord \
    --out-md "$TAB_CHD"

echo "================================================================"
echo "E2 evaluation complete."
echo "  melody pairing : $TAB_MEL   (read the _a / mel_* rows)"
echo "  chord pairing  : $TAB_CHD   (read the _b / chord / harmonic rows)"
echo "  raw CSVs       : $MET_MEL , $MET_CHD"
echo "Reading guide:"
echo "  * Non-inferiority: A.2 retains the marginal if it is within a"
echo "    margin delta of the specialist on each primary H3 grammar"
echo "    metric for ITS stream. The specialist winning slightly is the"
echo "    expected outcome, not a failure."
echo "  * Ignore cross-stream rows (chord metrics in the melody table"
echo "    and vice versa) -- the partner stream is absent by design, so"
echo "    those are 0/NaN, not deficits."
echo "  * If S-chord's rows are empty/zero, its output program is not in"
echo "    CHORD_OUT_PROGRAMS ($CHORD_OUT_PROGRAMS) -- inspect one midi"
echo "    under temp/${SCHORD_OUT_NAME} and override the knob."
echo "================================================================"
