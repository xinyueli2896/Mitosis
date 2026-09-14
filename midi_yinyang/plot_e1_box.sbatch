#!/bin/bash
# SLURM (CPU): box plots of an E1 metrics CSV, grouped by system family.
#
# Draws one panel per metric: a box per system over the PER-SONG values
# (samples averaged within a song first), systems ordered and coloured
# by family. The reference level is a dashed line, the mean and its 95%
# CI a dark stick over each box, and the best ranked system's mean a
# solid line across the panel. Systems not present in the CSV are drawn
# as hatched "pending" slots rather than dropped, so a reader can see
# what is still missing.
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
#   WIDTH     figure width in inches (default 7.0, a two-column page)
#   TITLE     optional figure title
#   UBR_CURVES=1 (default) also draw the unique-beat-ratio curves --
#             the BEAT paper's figure: ratio against phrase length, one
#             curve per system, ground truth dashed, SE bands over songs.
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
WIDTH="${WIDTH:-7.0}"
TITLE="${TITLE:-}"
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
        python plot_e1_box.py --csv "$CSV" --out "${OUT}_${b}" \
            --block "$b" ${NCOLS:+--ncols "$NCOLS"} --width "$WIDTH" \
            $([[ "$DELTAS_ONLY" == "1" ]] && echo --deltas-only) \
            $([[ "$PER_SONG" == "1" ]] && echo --per-song) \
            ${POOLED_CSV:+--pooled-csv "$POOLED_CSV"} \
            ${TITLE:+--title "$TITLE"}
    done
else
    python plot_e1_box.py --csv "$CSV" --out "$OUT" --width "$WIDTH" \
        ${NCOLS:+--ncols "$NCOLS"} ${METRICS:+--metrics "$METRICS"} \
        ${TITLE:+--title "$TITLE"}
fi

if [[ "$UBR_CURVES" == "1" ]]; then
    if [[ -f "$MANIFEST" && -d "$EVAL_MEL" && -d "$EVAL_CHORD" ]]; then
        echo "--- unique beat ratio curves ---"
        python plot_ubr_curves.py --manifest "$MANIFEST" \
            --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
            --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
            --width "$WIDTH" --out "${OUT}_ubr_curves"
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
echo "Boxes there are quartiles over SONGS. The dark stick is the mean and its"
echo "95% CI: two systems whose sticks miss each other differ at about"
echo "the 95% level. The dashed line is the reference -- the value a"
echo "system scores when it matches the ground-truth continuation -- and"
echo "the solid line is the best ranked system's mean."
echo "================================================================"
