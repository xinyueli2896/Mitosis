#!/usr/bin/env bash
# External baseline: the Anticipatory Music Transformer (Thickstun et al.,
# TMLR 2024), prompted continuation of both streams.
#
#   bash scripts/external/amt.sh setup      # clone + private venv + weights
#   bash scripts/external/amt.sh generate   # -> temp/E1/AMT/
#   SHARD=K/N bash scripts/external/amt.sh generate   # every N-th song, one job per GPU
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
        pip install "$AMT_DIR" accelerate mido pretty_midi
    fi
    source "$AMT_DIR/.venv/bin/activate"
    # Stock transformers 4.x (AMT's GPT-2 checkpoint), outside the if so
    # an existing venv is brought in line too.
    pip install -q "transformers>=4.40,<5"
    # common.sh puts the repo's patched transformers first on PYTHONPATH,
    # for the duet models; AMT must import its own venv's copy.
    unset PYTHONPATH
    python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('$MODEL'); print('[amt] weights cached')"
    ;;
generate)
    [[ -d "$AMT_DIR/.venv" ]] || { echo "run 'amt.sh setup' first" >&2; exit 1; }
    # with SHARD=K/N, run stage_prompts.sh once first: the jobs would
    # otherwise race to stage the prompts
    if [[ -n "${SHARD:-}" ]]; then
        need "$PROMPTS/.ready" "Run 'bash scripts/stage_prompts.sh' once before sharded jobs."
    else
        bash "$ROOT/scripts/stage_prompts.sh"
    fi
    IDS=$(shard $(for f in "$PROMPTS"/merged_tagged/*.mid; do basename "$f" .mid; done))
    echo "[amt] ${SHARD:+shard $SHARD: }$(wc -w <<< "$IDS") songs"
    source "$AMT_DIR/.venv/bin/activate"
    unset PYTHONPATH            # see setup
    python amt_prompted.py --merged-folder "$PROMPTS/merged_tagged" --ids $IDS \
        --out-dir "$OUT_ROOT/AMT" --model "$MODEL" \
        --prompt-frames "$PROMPT_LENGTH" --gen-frames "$GEN_LENGTH" \
        --n-samples "$N_SAMPLES" --top-p 0.98 --skip-existing
    echo "[amt] -> $DUET/$OUT_ROOT/AMT"
    ;;
*) echo "usage: amt.sh setup|generate" >&2; exit 1 ;;
esac
