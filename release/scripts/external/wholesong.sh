#!/usr/bin/env bash
# External baseline: Whole-Song Gen (Wang, Min and Xia, ICLR 2024),
# prompted lead-sheet continuation.
#
#   bash scripts/external/wholesong.sh setup
#   POP909_RAW=/path/to/POP909-Dataset/POP909 bash scripts/external/wholesong.sh generate
#   DATASET=nottingham bash scripts/external/wholesong.sh generate
#
# Protocol of the paper (row "Whole-Song Gen", system id WSfv4):
#   * the model's own form stage, prompted on the same bars as every other
#     system (WS_FORM=generated), not the oracle form of its paper;
#   * the lead-sheet prompt built from our staged crops, so it is the same
#     prompt every other system sees (needs POP909_RAW for chord_midi.txt);
#   * its chord track then re-voiced into our rendering (bass at 36+pc,
#     tones at 60+pc) and cut to four voices, the representation the duet
#     models live in: temp/E1/WSf -> temp/E1/WSfv4.
#
# Nottingham: the model reads songs in its POP909 layout (melody,
# chord symbols, phrase label), so build_nottingham_wholesong.py writes
# that layout from our staged crops first; the rest of the protocol is
# the same (see that script for the phrase-label heuristic, the one input
# Nottingham does not provide).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WS="$DUET/external/whole_song_gen"

case "${1:?usage: wholesong.sh setup|generate}" in
setup)
    mkdir -p "$DUET/external"
    [[ -d "$WS/.git" ]] || git clone https://github.com/ZZWaang/whole-song-gen "$WS"
    if [[ ! -d "$WS/.venv" ]]; then
        # shares torch and the rest with the main environment
        python -m venv --system-site-packages "$WS/.venv"
    fi
    source "$WS/.venv/bin/activate"
    REQ=$(find "$WS" -maxdepth 2 -iname 'requirements*.txt' | head -1)
    [[ -n "$REQ" ]] && pip install -q -r "$REQ"
    (cd "$WS" && PYTHONPATH="$WS" python "$DUET/wholesong_deps.py")
    cat <<EOF
[wholesong] code and environment ready under $WS.
Download the pretrained stages (form, counterpoint, lead sheet) as the
whole-song-gen README describes, into $WS/pretrained_models/.
EOF
    ;;
generate)
    [[ -d "$WS/.venv" ]] || { echo "run 'wholesong.sh setup' first" >&2; exit 1; }
    bash "$ROOT/scripts/stage_prompts.sh"
    MEL="$(readlink -f "$PROMPTS/mel")"; CHD="$(readlink -f "$PROMPTS/chord")"
    OUT="$(readlink -f "$OUT_ROOT")/WSf"
    if [[ "$DATASET" == nottingham ]]; then
        SONGS="$(readlink -f "$OUT_ROOT")/ws_songs"
        python build_nottingham_wholesong.py --mel-dir "$MEL" --chord-dir "$CHD" --dst "$SONGS"
        IDS=$(for f in "$PROMPTS"/mel/*.mid; do basename "$f" .mid; done | tr '\n' ' ')
        CROPS="$SONGS/crops.tsv"
        # the songs are built from our crops, so the prompt is ours already
        SRC=(--song-dir "$SONGS")
    else
        : "${POP909_RAW:?set POP909_RAW (the lead-sheet prompt reads chord_midi.txt)}"
        IDS=$(for f in "$PROMPTS"/mel/*.mid; do b=$(basename "$f" .mid); echo $((10#$b)); done | tr '\n' ' ')
        CROPS="$(readlink -f "$EVAL_SRC/prompt_crops.tsv")"
        SRC=(--our-melody-dir "$MEL" --our-chord-dir "$CHD" --our-prompt-level all
             --pop909-dir "$POP909_RAW")
    fi
    (
        source "$WS/.venv/bin/activate"
        # its own nested download layout, flattened once
        shopt -s dotglob
        for d in pretrained_models results_default data; do
            if [[ -d "$WS/$d/$d" ]]; then mv "$WS/$d/$d"/* "$WS/$d/"; rmdir "$WS/$d/$d"; fi
        done
        shopt -u dotglob
        cd "$WS"
        export PYTHONPATH="$WS:$PYTHONPATH"
        python "$DUET/wholesong_deps.py"
        python "$DUET/wholesong_prompted.py" --song-ids $IDS \
            --prompt-bars "$((PROMPT_LENGTH / 16))" --n-samples "$N_SAMPLES" --skip-existing \
            --form generated --crops-tsv "$CROPS" "${SRC[@]}" --out-dir "$OUT"
    )
    # re-voice the chord track into our rendering, four voices -> WSfv4
    python wholesong_chord_map.py --out-root "$OUT_ROOT" --system WSf \
        --chord-dir "$PROMPTS/chord" --prompt-frames "$PROMPT_LENGTH" --show 4 \
        --max-voices 4 --apply WSfv4
    echo "[wholesong] -> $DUET/$OUT_ROOT/WSfv4"
    ;;
*) echo "usage: wholesong.sh setup|generate" >&2; exit 1 ;;
esac
