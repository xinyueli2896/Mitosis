#!/usr/bin/env bash
# Build a note corpus: MIDI folders (and optionally MAESTRO v3's train
# split) read into notes in milliseconds, sustain pedal folded into
# durations, written to duet/data/$CORPUS.{notes.bin,pieces.npz,txt,json}
# (duet/preprocess_time_grid.py). CPU only.
#
# Every model that reads MIDI in real time loads this corpus and turns it
# into its own input: the time-grid cp transformer (pretrain_time.sh) and
# the MIDI RVQ codec (midi_rvq.sh). One prep serves both.
#
# The log ends with pieces, hours, and the share of pieces with pedal and
# with flat velocity (a sign of sequenced, not performed, MIDI).
#
# Knobs:
#   CORPUS        corpus name (default time_corpus)
#   MIDI_DIRS     space-separated MIDI folders, searched recursively
#   MAESTRO_ROOT  also take MAESTRO v3's train split (val/test stay out).
#                 Pass MAESTRO_ROOT= to be sure it is off.
#   PEDAL (1), MAX_FILES (all)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CORPUS="${CORPUS:-time_corpus}"
[[ -n "${MIDI_DIRS:-}${MAESTRO_ROOT:-}" ]] || { echo "set MIDI_DIRS and/or MAESTRO_ROOT" >&2; exit 1; }
args=(--name "$CORPUS" --pedal "${PEDAL:-1}")
[[ -n "${MIDI_DIRS:-}" ]] && args+=(--midi_dir $MIDI_DIRS)
[[ -n "${MAESTRO_ROOT:-}" ]] && args+=(--maestro_root "$MAESTRO_ROOT")
[[ -n "${MAX_FILES:-}" ]] && args+=(--max_files "$MAX_FILES")
python preprocess_time_grid.py prep "${args[@]}"
