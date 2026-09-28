#!/usr/bin/env bash
# Analyse the listening test and draw Figure 3. CPU only.
#
#   RATINGS=<ratings.csv> bash scripts/subjective.sh
#
# The CSV is the listening page's export: one row per rating (rater, song,
# system, and one column per axis). Raters whose every rating is the same
# value are dropped. Writes results/subjective_{anova,posthoc}.csv
# (repeated-measures ANOVA, Greenhouse-Geisser corrected, and
# Holm-corrected paired t-tests against Duet) and figures/subjective_bars.pdf.
: "${RATINGS:?set RATINGS=<ratings csv>}"
RATINGS="$(cd "$(dirname "$RATINGS")" && pwd)/$(basename "$RATINGS")"   # before common.sh cds
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python subjective_anova.py --csv "$RATINGS" --out results/subjective --drop-constant
python figures/plot_subjective_bars.py --csv "$RATINGS" --out figures/subjective_bars --drop-constant
