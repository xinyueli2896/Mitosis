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
#   METRICS   comma-separated metric names; default the three
#             pre-registered primary endpoints plus coupling_delta
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
METRICS="${METRICS:-}"
WIDTH="${WIDTH:-7.0}"
TITLE="${TITLE:-}"

python -c 'import matplotlib' 2>/dev/null || {
    echo "ERROR: matplotlib is not in the $CONDA_ENV env."
    echo "       conda install -n $CONDA_ENV matplotlib   (or pip install)"
    exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CSV   : $CSV"
echo "OUT   : $OUT.{pdf,png}"
echo "METRIC: ${METRICS:-<defaults>}"
echo "================================================================"

python plot_e1_box.py --csv "$CSV" --out "$OUT" --width "$WIDTH" \
    ${METRICS:+--metrics "$METRICS"} \
    ${TITLE:+--title "$TITLE"}

echo "================================================================"
echo "Done. $OUT.pdf is the one to \\includegraphics."
echo "Boxes are quartiles over SONGS; the notch is the 95% CI of the"
echo "median, so two boxes whose notches do not overlap differ at about"
echo "the 95% level. The dashed line is the reference: the value a"
echo "system scores when it matches the ground-truth continuation."
echo "================================================================"
