#!/usr/bin/env bash
# Pretrain the cp transformer on time-based frames, so that symbolic frames
# line up with MusicGen's EnCodec frames (20 ms) and MAESTRO's performance
# MIDI needs no tempo. See duet/preprocess_time_grid.py for the
# tokenization and duet/cp_transformer_time.py for the model.
#
#   prep    MIDI folders (and optionally MAESTRO's train split) -> one note
#           corpus in milliseconds, duet/data/$CORPUS.*. CPU.
#   stats   for several frame sizes: empty frames, notes per frame, notes
#           dropped by the polyphony budget. CPU, minutes. Read this before
#           fixing FRAME_CODEC and MAX_POLYPHONY.
#   train   the time-grid cp transformer on $CORPUS (its own held-out
#           split validates). Every visible GPU; resubmitting resumes.
#   sample  continue held-out prompts with a checkpoint, write MIDI to listen to.
#
# Knobs:
#   MIDI_DIRS      space-separated MIDI folders (prep). Prefer performance
#                  MIDI (transcribed piano such as Aria-MIDI or
#                  GiantMIDI-Piano): a quantized corpus read in seconds is
#                  metronomic and teaches the wrong timing.
#   MAESTRO_ROOT   also take MAESTRO v3's train split (prep; val/test stay out)
#   CORPUS         corpus name (default time_corpus)
#   FRAME_CODEC    codec frames per symbolic frame (default 6 = 120 ms, with
#                  each note's onset offset inside it; 1 = one frame per 20 ms)
#   MAX_POLYPHONY  notes per frame (default 16)
#   MODEL_SIZE     0-3 (default 1: 12 layers, width 768, as the beat-time backbone)
#   BATCH_SIZE     per GPU (default 8); TRAIN_LENGTH frames per crop (default 384)
#   MAX_STEPS (500000), STRETCH (0.1: random time stretch +-10%), WANDB (0)
#   CKPT, OUT      checkpoint and output folder (sample)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGE="${1:?usage: pretrain_time.sh prep|stats|train|sample}"

CORPUS="${CORPUS:-time_corpus}"
FRAME_CODEC="${FRAME_CODEC:-6}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
need() { [[ -f "$1" ]] || { echo "missing $1: $2" >&2; exit 1; }; }

case "$STAGE" in
prep)
    [[ -n "${MIDI_DIRS:-}${MAESTRO_ROOT:-}" ]] || { echo "set MIDI_DIRS and/or MAESTRO_ROOT" >&2; exit 1; }
    args=(--name "$CORPUS" --pedal "${PEDAL:-1}")
    [[ -n "${MIDI_DIRS:-}" ]] && args+=(--midi_dir $MIDI_DIRS)
    [[ -n "${MAESTRO_ROOT:-}" ]] && args+=(--maestro_root "$MAESTRO_ROOT")
    [[ -n "${MAX_FILES:-}" ]] && args+=(--max_files "$MAX_FILES")
    python preprocess_time_grid.py prep "${args[@]}"
    ;;
stats)
    need "data/$CORPUS.notes.bin" "run 'pretrain_time.sh prep' first"
    python preprocess_time_grid.py stats --name "$CORPUS" --max_polyphony "$MAX_POLYPHONY" \
        --frame_codec ${FRAME_CODECS:-1 3 6 12}
    ;;
train)
    need "data/$CORPUS.notes.bin" "run 'pretrain_time.sh prep' first"
    python cp_transformer_time.py train --data "data/$CORPUS" --frame_codec "$FRAME_CODEC" \
        --max_polyphony "$MAX_POLYPHONY" --size "${MODEL_SIZE:-1}" --batch_size "${BATCH_SIZE:-8}" \
        --train_length "${TRAIN_LENGTH:-384}" --max_steps "${MAX_STEPS:-500000}" --stretch "${STRETCH:-0.1}" \
        --wandb "${WANDB:-0}"
    ;;
sample)
    : "${CKPT:?set CKPT to a checkpoint (ckpt/<run>/*.ckpt)}"
    python cp_transformer_time.py sample --ckpt "$CKPT" --data "data/$CORPUS" \
        --out "${OUT:-temp/time_samples/$(basename "$(dirname "$CKPT")")}" --n "${N:-4}"
    ;;
*) echo "unknown stage '$STAGE' (prep stats train sample)" >&2; exit 1 ;;
esac
