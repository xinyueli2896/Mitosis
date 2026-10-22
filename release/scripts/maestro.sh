#!/usr/bin/env bash
# MAESTRO v3 as a paired performance-MIDI + EnCodec-audio corpus, for the
# audio-symbolic work (score-to-performance framing). Three stages, each
# its own SLURM job (slurm/maestro_{download,encode,pack}.sbatch):
#
#   download  the official WAV+MIDI zip (108 GB) into $MAESTRO_ROOT and
#             unzip it. Resumes a partial download. The Hugging Face
#             ZzWaang/music-jepa-maestro set holds 16 kHz log-mel
#             features, not audio, so the codec cannot use it.
#   encode    every WAV -> EnCodec 32 kHz codes, one .npy per piece in
#             $CODES_DIR, sharded over all visible GPUs. Resumable.
#   pack      MIDI -> rolls on a GRID_HZ time grid (cp$MAX_POLYPHONY), cut
#             to the audio's length, packed with the codes into one set
#             of files per MAESTRO split (see duet/maestro_paired.py).
#
# Knobs: MAESTRO_ROOT (default duet/data/maestro-v3.0.0; put it on scratch,
# it needs ~230 GB during unzip), CODES_DIR (data/maestro_encodec32k),
# GRID_HZ (8: a 16th at 120 BPM = 0.125 s, 6.25 codec frames), MAX_POLYPHONY (16),
# PEDAL (1: offsets extended through the sustain pedal).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGE="${1:?usage: maestro.sh download|encode|pack}"

MAESTRO_ROOT="${MAESTRO_ROOT:-$DUET/data/maestro-v3.0.0}"
CODES_DIR="${CODES_DIR:-data/maestro_encodec32k}"
GRID_HZ="${GRID_HZ:-8}"
MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
PEDAL="${PEDAL:-1}"
URL=https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

case "$STAGE" in
download)
    if [[ -f "$MAESTRO_ROOT/maestro-v3.0.0.csv" ]]; then
        echo "already unzipped: $MAESTRO_ROOT"; exit 0
    fi
    parent="$(dirname "$MAESTRO_ROOT")"; mkdir -p "$parent"
    zip="$parent/maestro-v3.0.0.zip"
    wget -c --progress=dot:giga -O "$zip" "$URL"
    want=108445099632
    have=$(stat -c %s "$zip")
    [[ "$have" == "$want" ]] || { echo "zip is $have bytes, expected $want" >&2; exit 1; }
    unzip -q -o "$zip" -d "$parent"
    [[ "$parent/maestro-v3.0.0" == "$MAESTRO_ROOT" ]] || mv "$parent/maestro-v3.0.0" "$MAESTRO_ROOT"
    test -f "$MAESTRO_ROOT/maestro-v3.0.0.csv"
    echo "unzipped to $MAESTRO_ROOT; delete $zip once encode and pack are done"
    ;;
encode)
    n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true); (( n > 0 )) || n=1
    mkdir -p "$ROOT/logs"; pids=()
    for ((i = 0; i < n; i++)); do
        CUDA_VISIBLE_DEVICES=$i python maestro_paired.py encode --root "$MAESTRO_ROOT" \
            --codes_dir "$CODES_DIR" --shard $i --num_shards $n > "$ROOT/logs/maestro_encode_shard$i.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "shard $i failed, see logs/maestro_encode_shard$i.log" >&2; fail=1; }
    done
    exit $fail
    ;;
pack)
    python maestro_paired.py pack --root "$MAESTRO_ROOT" --codes_dir "$CODES_DIR" \
        --grid_hz "$GRID_HZ" --max_polyphony "$MAX_POLYPHONY" --pedal "$PEDAL"
    ;;
*) echo "unknown stage '$STAGE' (download encode pack)" >&2; exit 1 ;;
esac
