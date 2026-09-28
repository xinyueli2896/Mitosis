#!/usr/bin/env bash
# Build the training tensors and the evaluation prompts from raw POP909.
#
#   POP909_RAW=/path/to/POP909-Dataset/POP909 bash scripts/prepare_data.sh train
#   POP909_RAW=/path/to/POP909-Dataset/POP909 bash scripts/prepare_data.sh eval
#
# POP909_RAW is the POP909 folder of https://github.com/music-x-lab/POP909-Dataset
# (one sub-folder per song holding <id>.mid, beat_midi.txt, chord_midi.txt).
#
# train  data/pop909_{melody,chord}_cp4_v2.pt       the two Duet streams
#        data/pop909_melchord_tagged_cp16_v2.pt     the single-stream baselines
#        (melody and chords merged into one file, chords on program 48)
# eval   input/heldout_v5/{melody,chord}/            the held-out prompts
#        (every tenth song of the training manifest, re-aligned to a
#        sixteenth-note grid and cropped to start on a clean downbeat)
#
# Run `train` first: `eval` reads the training manifest to know which
# songs were never trained on.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${POP909_RAW:?set POP909_RAW to the POP909 folder of the POP909-Dataset repository}"
STAGE="${1:?usage: prepare_data.sh train|eval}"
# intermediate midi folders, under duet/. Fixed: the tokenizer reads
# POP909-Dataset/POP909-{melody,chord} (settings.py).
WORK=POP909-Dataset
mkdir -p "$WORK"

# Duet streams: max 4 simultaneous notes per frame. The chord stream is a
# bass note plus up to three upper tones; the melody is monophonic.
CP_DUET=4
# Single-stream models: the backbone's own budget.
CP_MERGED=16

case "$STAGE" in
train)
    # 1-3. beat-aligned midis, then the melody track and a chord track
    #      rendered from POP909's chord annotations
    python preprocess_pop909_align.py --src "$POP909_RAW" --dst "$WORK/POP909-aligned"
    python extract_pop909_melody.py --src "$WORK/POP909-aligned" --dst "$WORK/POP909-melody"
    python build_pop909_chord_midi.py --aligned "$WORK/POP909-aligned" \
        --src "$POP909_RAW" --dst "$WORK/POP909-chord"

    # 4. tokenize. Songs 001-005 are left out of the corpus entirely (they
    #    are added to the held-out prompts instead, see `eval`).
    export POP909_EXCLUDE_IDS="001 002 003 004 005"
    python preprocess_large_midi_dataset.py pop909_melody "$CP_DUET"
    python preprocess_large_midi_dataset.py pop909_chord "$CP_DUET"
    python check_paired_dataset.py "data/pop909_melody_cp${CP_DUET}_v2" "data/pop909_chord_cp${CP_DUET}_v2"

    # 5. single-stream corpus: one merged file per song, chords tagged on
    #    program 48 so the single-stream tokenizer keeps the streams apart
    python merge_melody_chord.py --melody "$WORK/POP909-melody" --chord "$WORK/POP909-chord" \
        --dst "$WORK/POP909-melody-chord-tagged" --chord-program 48
    POP909_COMBINED_PATH="$WORK/POP909-melody-chord-tagged" \
    POP909_MELCHORD_DATASET="pop909_melchord_tagged_cp${CP_MERGED}_v2" \
        python preprocess_large_midi_dataset.py pop909_melchord "$CP_MERGED"
    ;;
eval)
    need "data/pop909_melody_cp${CP_DUET}_v2.txt" "Run 'prepare_data.sh train' first."
    # 1. grid-aligned corpus: every note on the sixteenth-note grid at a
    #    fixed 120 bpm, bars starting on POP909's annotated downbeats
    python pop909_align_grid.py --raw "$POP909_RAW" --dst "$WORK/POP909-v5" \
        --sub 4 --tempo 120 --origin bar --chord-program 48
    # 2. held-out prompts: the songs at index 0 mod 10 of the training
    #    manifest (never trained on), plus 001-005, each cropped to start on
    #    a downbeat with four bars of melody before the continuation
    python build_prompt_crops.py \
        --mel-src "$WORK/POP909-v5/melody" --chord-src "$WORK/POP909-v5/chord" \
        --dst "$EVAL_SRC" \
        --dataset "pop909_melody_cp${CP_DUET}_v2" --split-ratio 10 \
        --mel-bars 4 --min-bars 25 --tempo 120 \
        --extra-ids 001 002 003 004 005 --force-ids 004 \
        --align-report "$WORK/POP909-v5/align_grid_report.tsv"
    echo "prompts: $DUET/$EVAL_SRC/{melody,chord}"
    ;;
*) echo "usage: prepare_data.sh train|eval" >&2; exit 1 ;;
esac
