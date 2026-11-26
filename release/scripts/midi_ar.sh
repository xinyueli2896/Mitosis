#!/usr/bin/env bash
# Autoregressive MIDI transformer (duet/midi_ar.py) over the MIDI RVQ
# codec's codes: MusicGen's LM layout (delay pattern, K heads) at the same
# 20 ms frame, so it lines up step for step with MusicGen.
#
#   train   codes made on the fly by the frozen codec. Every visible GPU;
#           resubmitting resumes from last.ckpt (chain resubmits with
#           --dependency=afterany to run past the 12 h limit).
#   sample  N pieces as MIDI into duet/$OUT (default temp/ar_samples/<run>):
#           unconditional, or continuations of EVAL_DATA pieces with
#           PROMPT_SECONDS > 0 (the prompt's original is written next to it).
#
# Knobs:
#   CODEC         codec checkpoint, or its run folder ckpt/<run>/ (the
#                 highest val_note_f1 in it)                      train
#   CORPUS (aria_dedup) MIX_DATA MIX_FRAC (0.15) VAL_DATA (maestro_val_midi)
#   WIDTH (1024) LAYERS (24) HEADS (16) FRAMES (1500 = 30 s) CONTEXT (50)
#   BATCH_SIZE (8 per GPU) ACCUMULATE (1) LR (3e-4) WARMUP (4000)
#   MAX_STEPS (500000) VAL_EVERY (2000) WANDB (0) RUN_NAME
#   CKPT          AR checkpoint or run folder (the lowest val_loss)   sample
#   EVAL_DATA (maestro_val_midi) N (8) SECONDS_PER_PIECE (30)
#   PROMPT_SECONDS (0) TEMPERATURE (1.0) TOP_K (250) OUT
#   THRESHOLD (0.8) MIN_GAP (2) FRAME_THRESHOLD (0.5): codec read-out
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGE="${1:?usage: midi_ar.sh train|sample}"

# pick_ckpt <file-or-dir> <metric> <max|min>: a file as is; in a folder, the best by metric.
pick_ckpt() {
    if [[ -f "$1" ]]; then echo "$1"; return; fi
    local order=-g best; [[ "$3" == max ]] || order=-gr
    best=$(ls "$1"/*"$2"=*.ckpt 2>/dev/null \
        | sed "s/.*$2=\([0-9.]*\)\.ckpt$/\1 &/" | sort $order | tail -1 | cut -d' ' -f2-)
    [[ -n "$best" ]] || { echo "ERROR: no *$2=*.ckpt in $1" >&2; exit 1; }
    echo "$best"
}

case "$STAGE" in
train)
    : "${CODEC:?set CODEC to the codec run folder ckpt/<run>/ or a .ckpt}"
    C=$(pick_ckpt "$CODEC" val_note_f1 max); echo "codec $C"
    CORPUS="${CORPUS:-aria_dedup}"
    need "data/$CORPUS.notes.bin" "build it with scripts/prep_data.sh"
    args=(--codec "$C" --data "data/$CORPUS" --d "${WIDTH:-1024}" --layers "${LAYERS:-24}" --heads "${HEADS:-16}"
          --frames "${FRAMES:-1500}" --context "${CONTEXT:-50}" --batch_size "${BATCH_SIZE:-8}"
          --accumulate "${ACCUMULATE:-1}" --lr "${LR:-3e-4}" --warmup "${WARMUP:-4000}"
          --max_steps "${MAX_STEPS:-500000}" --val_every "${VAL_EVERY:-2000}" --stretch "${STRETCH:-0.1}"
          --wandb "${WANDB:-0}")
    VAL_DATA="${VAL_DATA-maestro_val_midi}"
    if [[ -n "$VAL_DATA" && -f "data/$VAL_DATA.notes.bin" ]]; then
        args+=(--val_data "data/$VAL_DATA")
    else
        echo "validating on held-out pieces of $CORPUS (data/$VAL_DATA.notes.bin not found)"
    fi
    if [[ -n "${MIX_DATA:-}" ]]; then
        need "data/$MIX_DATA.notes.bin" "build it with scripts/prep_data.sh"
        args+=(--mix_data "data/$MIX_DATA" --mix_frac "${MIX_FRAC:-0.15}")
    fi
    [[ -n "${RUN_NAME:-}" ]] && args+=(--run_name "$RUN_NAME")
    python midi_ar.py train "${args[@]}"
    ;;
sample)
    : "${CKPT:?set CKPT to ckpt/<run>/ or a .ckpt}"
    C=$(pick_ckpt "$CKPT" val_loss min); echo "checkpoint $C"
    OUT="${OUT:-temp/ar_samples/$(basename "$(dirname "$C")")}"
    args=(--ckpt "$C" --out "$OUT" --n "${N:-8}" --seconds "${SECONDS_PER_PIECE:-30}"
          --prompt_seconds "${PROMPT_SECONDS:-0}" --temperature "${TEMPERATURE:-1.0}" --top_k "${TOP_K:-250}"
          --threshold "${THRESHOLD:-0.8}" --min_gap "${MIN_GAP:-2}" --frame_threshold "${FRAME_THRESHOLD:-0.5}")
    if [[ "${PROMPT_SECONDS:-0}" != 0 ]]; then
        EVAL_DATA="${EVAL_DATA:-maestro_val_midi}"
        need "data/$EVAL_DATA.notes.bin" "set EVAL_DATA to a note corpus"
        args+=(--data "data/$EVAL_DATA")
    fi
    python midi_ar.py sample "${args[@]}"
    echo "MIDI in duet/$OUT"
    ;;
*) echo "unknown stage '$STAGE' (train sample)" >&2; exit 1 ;;
esac
