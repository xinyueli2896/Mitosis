#!/usr/bin/env bash
# Pretrain a cp transformer on the time-grid tokenization: one frame per
# 20 ms, the frame of MusicGen's EnCodec 32 kHz, so a symbolic frame and
# a codec frame cover the same time with no tempo involved. Same
# architecture as the beat-time backbone; only the data changes (see
# duet/preprocess_time_grid.py).
#
#   prep   MIDI_DIR (any folder of .mid/.midi, searched recursively)
#          -> duet/data/$CORPUS.notes.pt and friends (sparse; CPU)
#   train  cp_transformer.py on that corpus, its own held-out split
#          (index % 10 == 1) for validation. Uses every visible GPU;
#          resubmitting resumes from the run's last.ckpt.
#
# Knobs: CORPUS (name, default time50_cp16), FRAME_HZ (50), MAX_POLYPHONY
# (16), PEDAL (1), MAX_FILES (all), MODEL_SIZE (1 = 12 layers, width 768,
# the size of the beat-time backbone), VELOCITY (1: the with_velocity
# tokenizer, 16 velocity bins, which rendering audio needs), BATCH_SIZE
# (per GPU, default 4), TRAIN_LENGTH (1500 frames = 30 s, MusicGen's
# context; the model holds at most 1535).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGE="${1:?usage: pretrain_time.sh prep|train}"

CORPUS="${CORPUS:-time50_cp16}"
FRAME_HZ="${FRAME_HZ:-50}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
PEDAL="${PEDAL:-1}"
MODEL_SIZE="${MODEL_SIZE:-1}"
VELOCITY="${VELOCITY:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_LENGTH="${TRAIN_LENGTH:-1500}"

case "$STAGE" in
prep)
    : "${MIDI_DIR:?set MIDI_DIR to the MIDI folder to tokenize}"
    python preprocess_time_grid.py --midi_dir "$MIDI_DIR" --name "$CORPUS" --frame_hz "$FRAME_HZ" \
        --max_polyphony "$MAX_POLYPHONY" --pedal "$PEDAL" ${MAX_FILES:+--max_files "$MAX_FILES"}
    ;;
train)
    need() { [[ -f "$1" ]] || { echo "missing $1: $2" >&2; exit 1; }; }
    need "data/$CORPUS.notes.pt" "run 'pretrain_time.sh prep' first"
    # cp_transformer.py takes a negative size for the velocity tokenizer
    size_arg=$MODEL_SIZE; [[ "$VELOCITY" == 1 ]] && size_arg=$(( -MODEL_SIZE - 1 ))
    export CP_TRAIN_DATA="data/$CORPUS.notes.pt" CP_TRAIN_LENGTH="$TRAIN_LENGTH" CP_RUN_NAME="cp_$CORPUS"
    resume=()
    for run in ckpt/cp_${CORPUS}_size${MODEL_SIZE}*_batch_*/; do
        [[ -f "${run}last.ckpt" ]] && resume=("${run}last.ckpt") && echo "resuming ${run}last.ckpt" && break
    done
    python cp_transformer.py "$BATCH_SIZE" "$size_arg" "${resume[@]}"
    ;;
*) echo "unknown stage '$STAGE' (prep train)" >&2; exit 1 ;;
esac
