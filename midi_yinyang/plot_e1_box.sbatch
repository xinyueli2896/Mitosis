#!/bin/bash
# SLURM (CPU): box plots of an E1 metrics CSV, by system family.
#
# Draws one panel per metric: a standard box plot per system over the
# PER-SONG values (samples averaged within a song first), the box in
# its family's colour, ordered by family. Median line, whiskers at 1.5
# IQR, outliers omitted, the mean as a white diamond, the reference
# level a dashed line, and the best ranked system's mean a solid line
# across the panel in that system's colour. WIDE LAYOUT by default
# (ORIENT=h, 2026-09-17): four metrics side by side across both columns
# (7.0 in), systems down the y-axis, names on the leftmost panel with
# the qualifier line (e.g. "- harmonizers") a point smaller in grey;
# the eight-panel repetition sheet is two such rows. ORIENT=v gives the
# one-column layout with upright boxes. Systems not present
# in the CSV are drawn as hatched "pending" slots rather than dropped,
# so a reader can see what is still missing.
#
# Env knobs (--export=ALL,KEY=val):
#   CSV       metrics CSV from eval_metrics/score_e1
#             (default results/E1_p96_v5b93_metrics.csv)
#   OUT       output path WITHOUT extension; writes .pdf and .png
#             (default: the CSV path with _metrics.csv -> _box)
#   BLOCKS    what to draw, one figure each, suffixed _<block> on OUT.
#             Default is the four paper sheets:
#               quality     harmonic-rhythm, onset-grid and duration JSD,
#                           CORPUS-POOLED with a bootstrap over songs
#                           (needs the _pooled.csv that score_e1 writes)
#               fit         every melody-chord fit metric (H2)
#               repetition  unique beat ratio, i.e. structuredness (R)
#               prompt      every prompt-adherence metric (P)
#             A raw hypothesis block (H1/H2/H3/P/S/R) also works.
#             Empty falls back to METRICS in a single figure.
#             H1 IS ONLY IN THE CSV IF THE RUN WAS SCORED AFTER H1 WENT
#             BACK INTO CSV_GROUPS -- an older CSV has no survival or
#             empty_rate columns and the H1 figure comes out all
#             placeholders. Rescore with score_e1.sbatch if so.
#   METRICS   comma-separated metric names, used only when BLOCKS is
#             empty; default the pre-registered primary endpoints plus
#             coupling_delta
#   PER_SONG=1  force the per-song distribution on a sheet that
#             defaults to pooled. The per-song JSD is biased up by ~0.06
#             at these sample sizes, so do not describe it as unbiased.
#   POOLED_CSV  pooled CSV path; default is CSV with _metrics.csv
#             swapped for _pooled.csv
#   DELTAS_ONLY=1  keep only the _delta panels of each block. P and R
#             carry a raw value and a delta per statistic and only the
#             delta has a reference level, so a P sheet goes from 32
#             panels to 16 without losing a readable comparison.
#   NCOLS     panels per row. Default is the preset's own (4 for the
#             32-panel prompt sheet, 2 elsewhere). At 1 the names are
#             horizontal and two-line with family brackets under the
#             axis; above 1 they turn vertical and the families move
#             into the legend, which is the only way nine of them fit.
#   EXCLUDE_SYSTEMS  internal names to leave off every figure, SPACE-separated
#             (sbatch --export splits on commas, so a comma list loses
#             every item but the first); no column, no placeholder.
#             EXCLUDE_SYSTEMS=A3
#             drops Duet (refine) and keeps Duet (alt. commit).
#   ORIENT    h (default): wide layout, four panels per row, systems on
#             the y-axis, use with figure*; v: one-column layout
#   WIDTH     box-sheet width in inches for ORIENT=v (default 3.4, one
#             column); the wide layout is 7.0 in
#   UBR_WIDTH width of the unique-beat-ratio curves (default 7.0: that
#             figure keeps the full page width)
#   TITLE     optional figure title
#   QUALITY_TABLE=1 (default) also write the quality sheet as a LaTeX
#             table, <OUT>_quality.tex: one row per system, one column
#             per pooled metric, 95% interval under each value, the
#             reference split-half null as the last row
#             (e1_quality_table.py). BOLD_BEST=1 bolds the best system
#             per column (off by default since 2026-09-23: no best marks
#             anywhere); BOLD_EXCLUDE="S1" keeps named systems out of
#             that contest. QUALITY_METRICS=h3 makes the columns the
#             stream-grammar divergences (harmonic rhythm, onset grid,
#             duration) instead of groove/scale consistency;
#             GROUP_COLUMN=1 (default) puts the family in a first
#             column instead of a header row per family.
#             QUALITY_PER_SAMPLE=1 prints the per-sample pooled value
#             (sample j of every song pooled against the pooled
#             reference, mean +- std over j) instead of the song-mixed
#             estimate with its bootstrap interval.
#   MOTIF_SHARE=1 (default) also draw the motif-development shares,
#             <OUT>_motif_share: per stream, one stacked bar per system
#             of the continuation's interval n-grams by relation to the
#             prompt (exact / transposed / inverted / contour / novel),
#             ground truth on top. MOTIF_N picks the n-gram length (3).
#             Needs the P2 columns, i.e. a CSV scored after 2026-09-17.
#   UBR_CURVES=1 (default) also draw the unique-beat-ratio curves --
#             the BEAT paper's figure: ratio against phrase length, one
#             curve per system, SE bands over songs. Two files:
#               _ubr_curves        the ratio itself, ground truth dashed,
#                                  each row's axis tight to its data
#                                  (the main file since 2026-09-16, by
#                                  request)
#               _ubr_curves_delta  system MINUS ground truth, paired per
#                                  song -- the target is the zero line
#             Needs the midis, not the CSV, so it reads:
#   MANIFEST  default CSV with _metrics.csv -> _manifest.tsv
#   EVAL_MEL / EVAL_CHORD   reference dirs, default
#             $OUT_ROOT/prompts/{mel,chord} with OUT_ROOT temp/E1_v5b
#   PROMPT_LENGTH 96   GEN_LENGTH 416
#
# Examples:
#   sbatch --export=ALL,CSV=results/E1_p96_v5b93_metrics.csv \
#       midi_yinyang/plot_e1_box.sbatch
#
#   sbatch --export=ALL,CSV=results/E1_p96_v5b93_metrics.csv,\
# METRICS=onset_grid_jsd_b,duration_jsd_b,voicing_jsd,OUT=results/E1_chord_box \
#       midi_yinyang/plot_e1_box.sbatch
#
#   # just the prompt sheet, deltas only -- 16 panels instead of 32:
#   sbatch --export=ALL,CSV=results/E1_p96_v5b93_metrics.csv,\
# BLOCKS=prompt,DELTAS_ONLY=1 midi_yinyang/plot_e1_box.sbatch

#SBATCH --job-name=plot_e1_box
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:20:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

CSV="${CSV:-results/E1_p96_v5b93_metrics.csv}"
[[ -f "$CSV" ]] || { echo "ERROR: no $CSV -- run score_e1.sbatch first"; exit 1; }
OUT="${OUT:-${CSV%_metrics.csv}_box}"
BLOCKS="${BLOCKS-quality fit repetition prompt}"
METRICS="${METRICS:-}"
NCOLS="${NCOLS:-}"
DELTAS_ONLY="${DELTAS_ONLY:-0}"
PER_SONG="${PER_SONG:-0}"
POOLED_CSV="${POOLED_CSV:-}"
WIDTH="${WIDTH:-3.4}"
ORIENT="${ORIENT:-h}"
UBR_WIDTH="${UBR_WIDTH:-7.0}"
TITLE="${TITLE:-}"
EXCLUDE_SYSTEMS="${EXCLUDE_SYSTEMS:-}"
UBR_CURVES="${UBR_CURVES:-1}"
OUT_ROOT="${OUT_ROOT:-temp/E1_v5b}"
MANIFEST="${MANIFEST:-${CSV%_metrics.csv}_manifest.tsv}"
EVAL_MEL="${EVAL_MEL:-$OUT_ROOT/prompts/mel}"
EVAL_CHORD="${EVAL_CHORD:-$OUT_ROOT/prompts/chord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"

python -c 'import matplotlib' 2>/dev/null || {
    echo "ERROR: matplotlib is not in the $CONDA_ENV env."
    echo "       conda install -n $CONDA_ENV matplotlib   (or pip install)"
    exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CSV   : $CSV"
echo "OUT   : $OUT[_<block>].{pdf,png}"
echo "BLOCKS: ${BLOCKS:-<none: single figure from METRICS>}"
echo "NCOLS : ${NCOLS:-<per preset>}   DELTAS_ONLY=$DELTAS_ONLY"
echo "================================================================"

if [[ -n "${BLOCKS// /}" ]]; then
    for b in $BLOCKS; do
        echo "--- $b ---"
        if [[ "$b" == quality && "${QUALITY_TABLE:-1}" == "1" ]]; then
            # the quality sheet as a LaTeX table too: every panel is a
            # divergence with reference 0, so a table carries it better
            pc="${POOLED_CSV:-${CSV%_metrics.csv}_pooled.csv}"
            [[ -f "$pc" ]] && python e1_quality_table.py --pooled "$pc" \
                --out "${OUT}_quality.tex" ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"} \
                $([[ "${BOLD_BEST:-0}" == "1" ]] || echo --no-bold) \
                ${BOLD_EXCLUDE:+--bold-exclude "$BOLD_EXCLUDE"} \
                --metrics "${QUALITY_METRICS:-gcsc}" \
                $([[ "${QUALITY_PER_SAMPLE:-0}" == "1" ]] && echo --per-sample) \
                $([[ "${GROUP_COLUMN:-1}" == "1" ]] && echo --group-column) \
                || echo "[table] no $pc; quality table skipped"
        fi
        python plot_e1_box.py --csv "$CSV" --out "${OUT}_${b}" \
            --block "$b" ${NCOLS:+--ncols "$NCOLS"} --width "$WIDTH" --orient "$ORIENT" \
            $([[ "$DELTAS_ONLY" == "1" ]] && echo --deltas-only) \
            $([[ "$PER_SONG" == "1" ]] && echo --per-song) \
            ${POOLED_CSV:+--pooled-csv "$POOLED_CSV"} \
            ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"} \
            ${TITLE:+--title "$TITLE"}
    done
else
    python plot_e1_box.py --csv "$CSV" --out "$OUT" --width "$WIDTH" --orient "$ORIENT" \
        ${NCOLS:+--ncols "$NCOLS"} ${METRICS:+--metrics "$METRICS"} \
        ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"} \
        ${TITLE:+--title "$TITLE"}
fi

if [[ "${MOTIF_SHARE:-1}" == "1" ]]; then
    # motif development: stacked shares of the continuation's interval
    # n-grams by relation to the prompt (P2 block; needs a CSV scored
    # with the current eval_metrics, else the script says so)
    echo "--- motif share ---"
    python plot_motif_share.py --csv "$CSV" --out "${OUT}_motif_share" \
        --n "${MOTIF_N:-3}" ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"} \
        || echo "[motif share] skipped (see message above)"
fi

if [[ "$UBR_CURVES" == "1" ]]; then
    if [[ -f "$MANIFEST" && -d "$EVAL_MEL" && -d "$EVAL_CHORD" ]]; then
        echo "--- unique beat ratio curves ---"
        for y in ratio delta; do
            suffix=""; [[ "$y" == delta ]] && suffix="_delta"
            python plot_ubr_curves.py --manifest "$MANIFEST" \
                --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
                --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
                --width "$UBR_WIDTH" --y "$y" --out "${OUT}_ubr_curves$suffix" \
                ${EXCLUDE_SYSTEMS:+--exclude "$EXCLUDE_SYSTEMS"}
        done
    else
        echo "[ubr] skipped: need MANIFEST=$MANIFEST and the reference dirs"
        echo "      $EVAL_MEL / $EVAL_CHORD (UBR_CURVES=0 silences this)"
    fi
fi

echo "================================================================"
echo "Done. The .pdf files are the ones to \\includegraphics."
echo "The quality sheet is CORPUS-POOLED: one histogram per system over"
echo "every song, so each mark is a single value with a bootstrap CI over"
echo "songs, and there is no std to quote. The other sheets are per-song."
echo "Clouds there are densities over SONGS, ticks the quartiles. The ink"
echo "stick is the mean and its"
echo "95% CI: two systems whose sticks miss each other differ at about"
echo "the 95% level. The dashed line is the reference -- the value a"
echo "system scores when it matches the ground-truth continuation -- and"
echo "the solid line is the best ranked system's mean. The unranked column"
echo "has no line of its own."
echo "================================================================"
