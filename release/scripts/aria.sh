#!/usr/bin/env bash
# Aria-MIDI v1 (Bradshaw et al., ICLR 2025): solo-piano MIDI transcribed
# from YouTube recordings, in real time like MAESTRO (no beat grid). Used
# as the large MIDI-only corpus for the time-based models and the MIDI
# RVQ codec. Not paired with audio (the audio is not released).
#
#   download  one subset's .tar.gz from Hugging Face into $ARIA_ROOT and
#             unpack it. Resumes a partial download; a finished one is a no-op.
#
# Knobs:
#   ARIA_ROOT   where to unpack (default duet/data/aria-midi; scratch is better)
#   SUBSET      deduped (default: 371k files, one performance per
#               composition, heavy quality filters), pruned (821k), unique
#               (33k), or full (1.19M)
#   HF_TOKEN    sent if set (the dataset is public)
# Then build the note corpus with scripts/prep_data.sh, MIDI_DIRS=$ARIA_ROOT/<subset>.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGE="${1:?usage: aria.sh download}"

ARIA_ROOT="${ARIA_ROOT:-$DUET/data/aria-midi}"
SUBSET="${SUBSET:-deduped}"
case "$SUBSET" in
    deduped|pruned|unique) name="aria-midi-v1-$SUBSET-ext" ;;
    full) name="aria-midi-v1-ext" ;;
    *) echo "unknown SUBSET '$SUBSET' (deduped pruned unique full)" >&2; exit 1 ;;
esac
URL="https://huggingface.co/datasets/loubb/aria-midi/resolve/main/$name.tar.gz"

case "$STAGE" in
download)
    dst="$ARIA_ROOT/$SUBSET"
    if [[ -f "$dst/.done" ]]; then echo "already unpacked: $dst"; exit 0; fi
    mkdir -p "$dst"
    tgz="$ARIA_ROOT/$name.tar.gz"
    hdr=(); [[ -n "${HF_TOKEN:-}" ]] && hdr=(--header "Authorization: Bearer $HF_TOKEN")
    wget -c --tries=10 --progress=dot:giga "${hdr[@]}" -O "$tgz" "$URL"
    gzip -t "$tgz"   # a cut download fails here, not halfway through prep
    tar -xzf "$tgz" -C "$dst"
    n=$(find "$dst" -name '*.mid' | wc -l)
    (( n > 0 )) || { echo "no .mid files in $dst" >&2; exit 1; }
    echo "$n MIDI files in $dst"; find "$dst" -name 'metadata.json' | head -3
    touch "$dst/.done"
    echo "unpacked to $dst; delete $tgz once the corpus is built"
    ;;
*) echo "unknown stage '$STAGE' (download)" >&2; exit 1 ;;
esac
