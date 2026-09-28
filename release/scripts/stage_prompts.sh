#!/usr/bin/env bash
# Stage the held-out prompts once under temp/E1/prompts/ (called by the
# generation scripts; safe to run by hand).
#
#   prompts/mel/<id>.mid            melody stream, also the scoring reference
#   prompts/chord/<id>.mid          chord stream, program 0 (the duet models
#                                   were trained on untagged chords)
#   prompts/merged_tagged/<id>.mid  both streams in one file, chords on
#                                   program 48, for the single-stream models
#
# EXCLUDE_SONGS are left out here, so no system generates them.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -f "$PROMPTS/.ready" ]]; then exit 0; fi
need "$EVAL_SRC/melody" "Run 'prepare_data.sh eval' or download input/heldout_v5."

mkdir -p "$PROMPTS/mel" "$PROMPTS/chord"
cp -f "$EVAL_SRC"/melody/*.mid "$PROMPTS/mel/"
cp -f "$EVAL_SRC"/chord/*.mid "$PROMPTS/chord/"
for s in $EXCLUDE_SONGS; do
    s=$(printf '%03d' "$((10#$s))")
    rm -f "$PROMPTS/mel/$s.mid" "$PROMPTS/chord/$s.mid"
done
[[ "$(ls "$PROMPTS/mel" | wc -l)" -eq "$(ls "$PROMPTS/chord" | wc -l)" ]] || {
    echo "ERROR: melody and chord prompt sets differ in size" >&2; exit 1; }

python - "$PROMPTS/chord" 0 <<'PY'
import glob, os, sys
import mido
folder, prog = sys.argv[1], int(sys.argv[2])
for f in sorted(glob.glob(os.path.join(folder, '*.mid'))):
    m = mido.MidiFile(f)
    hit = False
    for tr in m.tracks:
        for msg in tr:
            if msg.type == 'program_change' and msg.program != prog:
                msg.program = prog
                hit = True
    if hit:
        m.save(f)
PY
python merge_melody_chord.py --melody "$PROMPTS/mel" --chord "$PROMPTS/chord" \
    --dst "$PROMPTS/merged_tagged" --chord-program 48
date > "$PROMPTS/.ready"
echo "[prompts] $(ls "$PROMPTS/mel" | wc -l) songs staged under $DUET/$PROMPTS"
