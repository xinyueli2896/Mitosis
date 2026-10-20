#!/usr/bin/env bash
# Score every complete system under temp/E1/ and build the paper's
# objective results. CPU only.
#
#   bash scripts/evaluate.sh
#
# Writes, under duet/:
#   results/E1_manifest.tsv, results/E1_metrics.csv   per-sample metrics
#   results/E1_pooled.csv                             corpus-pooled JSD
#   results/E1_table_twosided.md                      paired Wilcoxon vs Duet
#   figures/E1_quality.tex        Table 1: general quality (pooled JSD)
#   figures/E1_prompt_fit.pdf     Figure 2: prompt consistency (a-d) and
#                                 melody-chord consistency (e-h)
#   figures/E1_quality_variants.tex   Table 1 with the Duet variants added
#                                 (only when one of them was scored)
#
# A system is scored when every staged song has N_SAMPLES outputs;
# incomplete systems are listed and skipped.
#
# DATASET=nottingham scores temp/E1_nottingham/ and writes the same files
# with E1_nottingham_ in place of E1_.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need "$PROMPTS/.ready" "Generate at least one system first."
SONG_IDS=$(ls "$PROMPTS/mel" | sed 's/\.mid$//' | sort | tr '\n' ' ')

layout() { case "$1" in S1|S-scratch) echo single ;; *) echo duet_multi ;; esac; }
complete() {   # every song has >= N_SAMPLES samples in the system's layout
    local root="$OUT_ROOT/$1" sid n
    [[ -d "$root" ]] || return 1
    for sid in $SONG_IDS; do
        case "$(layout "$1")" in
            duet_multi) n=$(ls "$root/$sid/co/"sample_*.mid 2>/dev/null | wc -l) ;;
            single)     n=$(ls "$root/$sid".mid_temp*_continuation_*.mid 2>/dev/null | wc -l) ;;
        esac
        [[ "$n" -ge "$N_SAMPLES" ]] || return 1
    done
}

SRC=(); SCORED=()
for s in $ALL_SYSTEMS; do
    id=$(system_id "$s")
    if complete "$id"; then SRC+=(--source "$id:$(layout "$id"):$OUT_ROOT/$id"); SCORED+=("$id")
    else echo "[evaluate] $s ($id): incomplete or missing, not scored"; fi
done
[[ ${#SCORED[@]} -gt 0 ]] || { echo "ERROR: no complete system under $OUT_ROOT" >&2; exit 1; }
echo "[evaluate] scoring: ${SCORED[*]}"

python build_eval_manifest.py "${SRC[@]}" --songs $SONG_IDS --modes co --out results/${RESULTS_TAG}_manifest.tsv
python eval_metrics.py --task melchord --manifest results/${RESULTS_TAG}_manifest.tsv \
    --ref-a-dir "$PROMPTS/mel" --ref-b-dir "$PROMPTS/chord" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out results/${RESULTS_TAG}_metrics.csv --pooled-out results/${RESULTS_TAG}_pooled.csv \
    --pooled-boot 2000 --pooled-boot-mode songs --pooled-weight song

BASE=A3ctcaT; [[ " ${SCORED[*]} " == *" $BASE "* ]] || BASE="${SCORED[0]}"
python aggregate_eval_results.py --csv results/${RESULTS_TAG}_metrics.csv --task melchord \
    --baseline "$BASE" --mode co --out-md results/${RESULTS_TAG}_table_twosided.md

# Table 1: each sample index pooled over all songs against the pooled
# reference, mean +- std over the samples; the finetuned single-stream
# model is shown but not ranked
# The Duet variants stay out of Table 1 and Figure 2.
VARIANT_IDS=$(for s in $DUET_VARIANTS; do system_id "$s"; done | tr '\n' ' ')
python e1_quality_table.py --pooled results/${RESULTS_TAG}_pooled.csv --out figures/${RESULTS_TAG}_quality.tex \
    --metrics h3 --per-sample --bold-exclude S1 --group-column --digits 4 --exclude "$VARIANT_IDS"
SCORED_VARIANTS=$(for v in $VARIANT_IDS; do [[ " ${SCORED[*]} " == *" $v "* ]] && echo "$v"; done; true)
if [[ -n "$SCORED_VARIANTS" ]]; then
    echo "[evaluate] variants scored: $SCORED_VARIANTS"
    python e1_quality_table.py --pooled results/${RESULTS_TAG}_pooled.csv --out figures/${RESULTS_TAG}_quality_variants.tex \
        --metrics h3 --per-sample --bold-exclude S1 --group-column --digits 4 \
        --label tab:e1_quality_variants
fi

# Figure 2: solid star = best among Duet, its two ablations and the
# single-stream model trained from scratch (left of the dashed rule);
# hollow star = best of all systems where different
# the stale-draft variant is drawn (and ranked) only when it was scored
RANKED="A3ctcaT D1 A1 S-scratch"; FIG_EXCLUDE="A3stale $VARIANT_IDS"
[[ " ${SCORED[*]} " == *" A3stale "* ]] && { RANKED="A3ctcaT A3stale D1 A1 S-scratch"; FIG_EXCLUDE="$VARIANT_IDS"; }
python plot_e1_box.py --csv results/${RESULTS_TAG}_metrics.csv --out figures/${RESULTS_TAG}_prompt_fit \
    --exclude "$FIG_EXCLUDE" \
    --block "prompt fit" --orient v --ncols 4 --width 7 --panel-height 1.3 \
    --names none --short-titles \
    --ranked "$RANKED" --divider-after S-scratch --best-overall
echo "[evaluate] results under $DUET/results and $DUET/figures"
