#!/usr/bin/env bash
# Generate the evaluation samples of one in-house system.
#
#   bash scripts/generate.sh duet            [CKPT=<run dir or .ckpt>]
#   bash scripts/generate.sh dense | no_draft | ss_finetuned | ss_scratch
#
# Every system continues the same staged prompts (the held-out songs,
# first PROMPT_LENGTH frames) to GEN_LENGTH frames, N_SAMPLES times, into
# temp/E1/<system id>/. Finished samples are skipped, so an interrupted
# run resumes. One GPU per call; run several systems side by side with
# CUDA_VISIBLE_DEVICES=<i>.
#
# The external baselines have their own scripts under scripts/external/.
# Keep N_SAMPLES >= 2: with a single sample the duet generators write
# <song>/co.mid instead of the <song>/co/sample_<i>.mid layout evaluate.sh reads.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SYSTEM="${1:?usage: generate.sh duet|dense|no_draft|ss_finetuned|ss_scratch}"
ID="$(system_id "$SYSTEM")"
bash "$ROOT/scripts/stage_prompts.sh"

# Default checkpoints: the released files, or your own run directories.
case "$SYSTEM" in
    duet)         CKPT="${CKPT:-ckpt/duet}" ;;
    dense)        CKPT="${CKPT:-ckpt/dense}" ;;
    no_draft)     CKPT="${CKPT:-ckpt/no_draft}" ;;
    ss_finetuned) CKPT="${CKPT:-ckpt/ss_finetuned}" ;;
    ss_scratch)   CKPT="${CKPT:-ckpt/ss_scratch}" ;;
    *) echo "generate.sh covers the in-house systems; see scripts/external/ for $SYSTEM" >&2; exit 1 ;;
esac
need "$CKPT" "Download the checkpoints (scripts/download.sh) or pass CKPT=<your run dir>."
# A directory holding a single checkpoint (a released model) is resolved
# to that file; a training run directory is handed over with a trailing
# slash, and the loader picks its best checkpoint by validation loss.
if [[ -d "$CKPT" ]]; then
    mapfile -t _found < <(ls "$CKPT"/*.ckpt 2>/dev/null)
    if [[ ${#_found[@]} -eq 1 ]]; then CKPT="${_found[0]}"; else CKPT="${CKPT%/}/"; fi
fi
OUT="$OUT_ROOT/$ID"

case "$SYSTEM" in
duet|dense)
    # Draft-and-denoise, leader alternating by frame (ctc_alt): the
    # leader's draft is sampled from its autoregressive head and written
    # into its draft token, one denoise pass reads out both streams.
    # Temperature 1, no nucleus cut.
    SHAPE=(--moe-num-experts 4 --moe-topk 2)
    [[ "$SYSTEM" == dense ]] && SHAPE=(--moe-num-experts 1 --moe-topk 1 --moe-intermediate-size 6144)
    MELCHORD_CP=4 A3_SCHEDULE=ctc_alt A3_REFINE_STEPS=4 A3_FINAL_TEMP=1.0 A3_TOP_P=1.0 \
    python cp_transformer_m2c_duet_block_diffusion_combined.py \
        --ckpt "$CKPT" --mel-folder "$PROMPTS/mel" --chord-folder "$PROMPTS/chord" \
        --output-dir "$OUT" --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony 4 --skip-existing \
        --model-size large "${SHAPE[@]}" --min-chord-tokens-before-eos 0
    ;;
no_draft)
    MELCHORD_CP=4 python cp_transformer_m2c_intra_cross_attn_combined.py \
        --ckpt "$CKPT" --mel-folder "$PROMPTS/mel" --chord-folder "$PROMPTS/chord" \
        --output-dir "$OUT" --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony 4 --skip-existing \
        --model-size large --moe-num-experts 4 --moe-topk 2
    ;;
ss_finetuned|ss_scratch)
    # single stream: continues the merged file (chords on program 48) and
    # writes temp/<save-name>/<song>.mid_temp1.0_continuation_<i>.mid
    python cp_transformer_inference.py --ckpt "$CKPT" \
        --midi-folder "$PROMPTS/merged_tagged" \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony 16 --skip-existing \
        --save-name "${OUT#temp/}"
    ;;
esac
echo "[generate] $SYSTEM -> $DUET/$OUT"
