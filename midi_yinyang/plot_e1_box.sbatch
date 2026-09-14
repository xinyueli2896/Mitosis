#!/bin/bash
# SLURM (CPU): box plots of an E1 metrics CSV, grouped by system family.
#
# Draws one panel per metric: a notched box per system over the PER-SONG
# values (samples averaged within a song first), systems ordered and
# coloured by family, with the reference level marked as a dashed line
# and the 95% CI of each median as the box notch. Systems not present in
# the CSV are drawn as hatched "pending" slots rather than dropped, so a
# reader can see what is still missing.
#
# Env knobs (--export=ALL,KEY=val):
#   CSV       metrics CSV from eval_metrics/score_e1
#             (default results/E1_p96_v5b93_metrics.csv)
#   OUT       output path WITHOUT extension; writes .pdf and .png
#             (default: the CSV path with _metrics.csv -> _box)
#   BLOCKS    hypothesis blocks to draw, one figure each, suffixed
#             _<block> on OUT. Default "H3 H2 H1". Empty falls back to
#             METRICS in a single figure.
#             H1 IS ONLY IN THE CSV IF THE RUN WAS SCORED AFTER H1 WENT
#             BACK INTO CSV_GROUPS -- an older CSV has no survival or
#             empty_rate columns and the H1 figure comes out all
#             placeholders. Rescore with score_e1.sbatch if so.
#   METRICS   comma-separated metric names, used only when BLOCKS is
#             empty; default the pre-registered primary endpoints plus
#             coupling_delta
#   NCOLS     panels per row (default 2). At 1 the system names are
#             horizontal and two-line with family brackets under the
#             axis; above 1 they turn vertical and the families move
#             into the legend, which is the only way nine of them fit.
#   WIDTH     figure width in inches (default 7.0, a two-column page)
#   TITLE     optional figure title
#
# Examples:
#   sbatch --export=ALL,CSV=results/E1_p96_v5b93_metrics.csv \
#       midi_yinyang/plot_e1_box.sbatch
#
#   sbatch --export=ALL,CSV=results/E1_p96_v5b93_metrics.csv,\
# METRICS=onset_grid_jsd_b,duration_jsd_b,voicing_jsd,OUT=results/E1_chord_box \
#       midi_yinyang/plot_e1_box.sbatch

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
BLOCKS="${BLOCKS-H3 H2 H1}"
METRICS="${METRICS:-}"
NCOLS="${NCOLS:-2}"
WIDTH="${WIDTH:-7.0}"
TITLE="${TITLE:-}"

python -c 'import matplotlib' 2>/dev/null || {
    echo "ERROR: matplotlib is not in the $CONDA_ENV env."
    echo "       conda install -n $CONDA_ENV matplotlib   (or pip install)"
    exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CSV   : $CSV"
echo "OUT   : $OUT[_<block>].{pdf,png}"
echo "BLOCKS: ${BLOCKS:-<none: single figure from METRICS>}"
echo "NCOLS : $NCOLS"
echo "================================================================"

if [[ -n "${BLOCKS// /}" ]]; then
    for b in $BLOCKS; do
        echo "--- $b ---"
        python plot_e1_box.py --csv "$CSV" --out "${OUT}_${b}" \
            --block "$b" --ncols "$NCOLS" --width "$WIDTH" \
            ${TITLE:+--title "$TITLE"}
    done
else
    python plot_e1_box.py --csv "$CSV" --out "$OUT" --width "$WIDTH" \
        --ncols "$NCOLS" ${METRICS:+--metrics "$METRICS"} \
        ${TITLE:+--title "$TITLE"}
fi

echo "================================================================"
echo "Done. The .pdf files are the ones to \\includegraphics."
echo "Boxes are quartiles over SONGS; the notch is the 95% CI of the"
echo "median, so two boxes whose notches do not overlap differ at about"
echo "the 95% level. The dashed line is the reference: the value a"
echo "system scores when it matches the ground-truth continuation."
echo "================================================================"
