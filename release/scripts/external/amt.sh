#!/usr/bin/env bash
# External baseline: the Anticipatory Music Transformer (Thickstun et al.,
# TMLR 2024), prompted continuation of both streams.
#
#   bash scripts/external/amt.sh setup      # clone + private venv + weights
#   bash scripts/external/amt.sh generate   # -> temp/E1/AMT/
#
# AMT reads the merged, program-tagged prompt (its tokens carry the
# instrument, which is what lets the output split back into two streams).
# Note logits are restricted to the prompt's two instruments so the output
# stays two streams; no anticipation controls are used.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
AMT_DIR="$DUET/external/anticipation"
MODEL="${AMT_MODEL:-stanford-crfm/music-medium-800k}"

case "${1:?usage: amt.sh setup|generate}" in
setup)
    mkdir -p "$DUET/external"
    [[ -d "$AMT_DIR/.git" ]] || git clone https://github.com/jthickstun/anticipation "$AMT_DIR"
    if [[ ! -d "$AMT_DIR/.venv" ]]; then
        python -m venv "$AMT_DIR/.venv"
        source "$AMT_DIR/.venv/bin/activate"
        pip install --upgrade pip
        # the AMT weights ship as .bin, which recent transformers only
        # loads under torch >= 2.6
        pip install "torch>=2.6" ${TORCH_INDEX_URL:+--index-url "$TORCH_INDEX_URL"}
        pip install "$AMT_DIR" transformers accelerate mido pretty_midi
    fi
    source "$AMT_DIR/.venv/bin/activate"
    python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('$MODEL'); print('[amt] weights cached')"
    ;;
generate)
    [[ -d "$AMT_DIR/.venv" ]] || { echo "run 'amt.sh setup' first" >&2; exit 1; }
    bash "$ROOT/scripts/stage_prompts.sh"
    source "$AMT_DIR/.venv/bin/activate"
    python amt_prompted.py --merged-folder "$PROMPTS/merged_tagged" \
        --out-dir "$OUT_ROOT/AMT" --model "$MODEL" \
        --prompt-frames "$PROMPT_LENGTH" --gen-frames "$GEN_LENGTH" \
        --n-samples "$N_SAMPLES" --top-p 0.98 --skip-existing
    echo "[amt] -> $DUET/$OUT_ROOT/AMT"
    ;;
*) echo "usage: amt.sh setup|generate" >&2; exit 1 ;;
esac
