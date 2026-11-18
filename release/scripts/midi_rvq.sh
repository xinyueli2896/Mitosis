#!/usr/bin/env bash
# MIDI RVQ-VAE (duet/midi_rvq.py): piano MIDI -> 4 codes per 20 ms frame,
# the same shape as MusicGen's EnCodec 32 kHz codes, so MIDI frame f and
# codec frame f are the same 20 ms.
#
#   train   on a note corpus (scripts/prep_data.sh or maestro.sh pack),
#           validated on MAESTRO's validation split when it is packed.
#           Every visible GPU; resubmitting resumes from last.ckpt.
#   eval    whole-piece reconstruction note F1 (onset / +offset / +velocity)
#           on EVAL_DATA, written next to the checkpoint as JSON.
#   export  codes for every piece of each MAESTRO split:
#           data/maestro_<split>.midi_codes.npy, row-aligned with .codes.npy.
#
# Knobs:
#   CORPUS        training corpus under data/ (default time_piano)
#   VAL_DATA      validation corpus (default maestro_validation if packed,
#                 else the corpus's own held-out pieces)
#   N_Q (4) N_CODES (1024) DIM (128 code size) WIDTH (512) N_RES (8) N_ATTN (4)
#   FRAMES (500 = 10 s per window) BATCH_SIZE (16 per GPU) LR (3e-4)
#   MAX_STEPS (200000) VAL_EVERY (2000) WANDB (0) RUN_NAME
#   CKPT          a checkpoint, or a run folder ckpt/<run>/ (eval, export:
#                 the best val_note_f1 in it)
#   EVAL_DATA (maestro_test) MAX_PIECES (50)   eval
#   SPLITS (train validation test)             export: MAESTRO splits
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGE="${1:?usage: midi_rvq.sh train|eval|export}"

# best_ckpt <file-or-dir>: a file as is; in a folder, the highest val_note_f1.
best_ckpt() {
    if [[ -f "$1" ]]; then echo "$1"; return; fi
    local best
    best=$(ls "$1"/*val_note_f1=*.ckpt 2>/dev/null \
        | sed 's/.*val_note_f1=\([0-9.]*\)\.ckpt$/\1 &/' | sort -g | tail -1 | cut -d' ' -f2-)
    [[ -n "$best" ]] || { echo "ERROR: no *val_note_f1=*.ckpt in $1" >&2; exit 1; }
    echo "$best"
}

case "$STAGE" in
train)
    CORPUS="${CORPUS:-time_piano}"
    need "data/$CORPUS.notes.bin" "build it with scripts/prep_data.sh (or maestro.sh pack)"
    args=(--data "data/$CORPUS" --n_q "${N_Q:-4}" --n_codes "${N_CODES:-1024}" --dim "${DIM:-128}"
          --d "${WIDTH:-512}" --n_res "${N_RES:-8}" --n_attn "${N_ATTN:-4}" --frames "${FRAMES:-500}"
          --batch_size "${BATCH_SIZE:-16}" --lr "${LR:-3e-4}" --max_steps "${MAX_STEPS:-200000}"
          --val_every "${VAL_EVERY:-2000}" --stretch "${STRETCH:-0.1}" --wandb "${WANDB:-0}")
    VAL_DATA="${VAL_DATA-maestro_validation}"
    if [[ -n "$VAL_DATA" && -f "data/$VAL_DATA.notes.bin" ]]; then
        args+=(--val_data "data/$VAL_DATA")
    else
        echo "validating on held-out pieces of $CORPUS (data/$VAL_DATA.notes.bin not found)"
    fi
    [[ -n "${RUN_NAME:-}" ]] && args+=(--run_name "$RUN_NAME")
    python midi_rvq.py train "${args[@]}"
    ;;
eval)
    : "${CKPT:?set CKPT to ckpt/<run>/ or a .ckpt}"
    C=$(best_ckpt "$CKPT"); EVAL_DATA="${EVAL_DATA:-maestro_test}"
    need "data/$EVAL_DATA.notes.bin" "pack MAESTRO with maestro.sh pack, or set EVAL_DATA"
    echo "checkpoint $C"
    python midi_rvq.py eval --ckpt "$C" --data "data/$EVAL_DATA" --max_pieces "${MAX_PIECES:-50}"
    ;;
export)
    : "${CKPT:?set CKPT to ckpt/<run>/ or a .ckpt}"
    C=$(best_ckpt "$CKPT"); echo "checkpoint $C"
    for s in ${SPLITS:-train validation test}; do
        need "data/maestro_$s.notes.bin" "run maestro.sh pack first"
        python midi_rvq.py export --ckpt "$C" --data "data/maestro_$s"
    done
    ;;
*) echo "unknown stage '$STAGE' (train eval export)" >&2; exit 1 ;;
esac
