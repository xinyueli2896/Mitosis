#!/bin/bash
# SLURM: END-TO-END E3 (conditional generation, RQ3).
#
# The complete ground-truth partner stream is GIVEN; the model generates
# the remaining stream. Per EXPERIMENTS.md #E3:
#
#   melchord    : A.2 (mel2chord AND chord2mel); optionally B.1's native
#                 conditional (leader given -> follower generated, i.e.
#                 mel2chord).
#   drumnondrum : A.2, B.1, C.1 (rehearsal), C.2 (prefix), all
#                 drum -> nondrum; A.2 additionally nondrum -> drum.
#
# Baselines Y-mc / Y-dn: generate them first with eval_yinyang.sbatch,
# then pass Y_OUT=<its output root> here (adds the Y rows to the same
# table; BASELINE=Y gives paired tests vs the published model). Without
# any baseline present the tables print means only; BASELINE=<internal
# system id> (e.g. A.2) tests the others against it instead.
#
# The no-conditioning FLOOR for interpreting fit scores is the E1
# co-generation table (results/e1_<task>_p<P>_metrics.csv, mode=co):
# the same target-stream metrics scored when the partner was NOT given.
# A conditional system earns its keep by beating those numbers; no extra
# runs are needed for the floor.
#
# DRUM/NONDRUM NAMING ON THE MELCHORD TASK. B.1/C.1/C.2 were written for
# drumnondrum and their CLI still says --drum-folder / --nondrum-folder.
# Those are mod_a (conditioning) and mod_b (generated) SLOTS, not drum
# semantics: nothing downstream of the argument looks at is_drum, and the
# decoder names the output tracks MELODY / CHORD either way, which is
# what eval_metrics splits on. On melchord this script passes
# --mode-name mel2chord (or chord2mel) so the OUTPUT TREE names the
# direction it actually holds instead of writing drum2nondrum folders
# full of melody and chords. Older output trees still say drum2nondrum;
# the manifest's mode column is rewritten after building, so both score
# into the same table section.
#
# PROMPT LENGTH IS THE TARGET TRACK'S SEED. The conditioning stream is
# given in FULL for all GEN_LENGTH frames in every system. PROMPT_LENGTH
# is how much of the GENERATED track the model hears before it has to
# invent -- 64 frames = 4 bars, the E3 default. eval_metrics scores from
# that frame on, so the seeded bars never enter the numbers.
#
# Env knobs (--export=ALL,KEY=val):
#   TASK          melchord (default) | drumnondrum
#   CKPT_A2       A.2 run dir or ckpt file. OPTIONAL, like every other
#                 system: leave it unset to run E3 WITHOUT A.2. At least
#                 one of CKPT_A2/CKPT_B1/CKPT_C1/CKPT_C2/Y_OUT must be
#                 set, and BASELINE must name a system that is in the
#                 run -- with A.2 skipped, BASELINE=A.2 gives a
#                 means-only table with no paired tests.
#   CKPT_B1       optional (its native conditional mode)
#   CKPT_C1       optional, drumnondrum only (rehearsal)
#   CKPT_C2       optional, drumnondrum only (prefix)
#   BASELINE      system id for paired tests; default Y-mc/Y-dn (absent
#                 in the table -> means-only, by design)
#   MEL_FOLDER / CHORD_FOLDER   same task defaults as eval_e1
#   PROMPT_LENGTH frames of the GENERATED track to seed before sampling.
#                 Default 64 = 4 bars. Must be a whole number of bars;
#                 does NOT shorten the conditioning stream, which is
#                 always given in full.
#   GEN_LENGTH    default 384 (TOTAL frames)
#   N_SAMPLES     default 3 -- honoured by A.2 and B.1. C.1/C.2 have no
#                 sampling flag and emit 1 sample each; their per-song
#                 scores are therefore noisier (disclosed, not hidden).
#   RESULT_TAG    extra suffix on result filenames
#   SKIP_INFER=1  reuse existing outputs under OUT_ROOT, score only
#   PIANO_COPIES  1/0; default 1 for melchord (listening copies)
#
# Example (melchord):
#   sbatch --export=ALL,\
# CKPT_A2=ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_melchord_pop909_nottingham_batch_8_schedule/ \
#       midi_yinyang/eval_e3.sbatch
#
# Example (drumnondrum, all four systems, internal baseline A.2):
#   sbatch --export=ALL,TASK=drumnondrum,BASELINE=A.2,\
# CKPT_A2=ckpt/m2c_duet_block_diffusion_v1.0_large_gnl12_K4_drumnondrum_batch_16_schedule/,\
# CKPT_B1=ckpt/m2c_duet_anticipatory_v1.0_large_gnl12_drumnondrum_k16_batch_16_schedule/,\
# CKPT_C1=ckpt/<rehearsal run>/,CKPT_C2=ckpt/<prefix run>/ \
#       midi_yinyang/eval_e3.sbatch

#SBATCH --job-name=mitosis_eval_e3
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

TASK="${TASK:-melchord}"
CKPT_A2="${CKPT_A2:-}"
CKPT_B1="${CKPT_B1:-}"
CKPT_C1="${CKPT_C1:-}"
CKPT_C2="${CKPT_C2:-}"
N_SAMPLES="${N_SAMPLES:-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
SKIP_INFER="${SKIP_INFER:-0}"
RESULT_TAG="${RESULT_TAG:-}"

case "$TASK" in
    melchord)
        MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
        # cp8 is the melchord convention (cp4 retired: seventh chords are
        # 5 notes). Must match the ckpts' training data -- pass
        # MAX_POLYPHONY=4 for pre-migration cp4-trained checkpoints.
        MAX_POLYPHONY="${MAX_POLYPHONY:-8}"
        BASELINE="${BASELINE:-Y-mc}"
        # C.1/C.2 on melchord are valid once their melchord trainings
        # exist (train_duet_{rehearsal,prefix}.sbatch with
        # TASK=melchord_pop909_nottingham). Passing a ckpt asserts you
        # have one; the integrity pre-flight below rejects a wrong path.
        if [[ -n "$CKPT_C1" || -n "$CKPT_C2" ]]; then
            echo "[note] C.1/C.2 on melchord: make sure these are MELCHORD"
            echo "       checkpoints -- a drumnondrum ckpt here would decode"
            echo "       melchord prompts with cp16 assumptions and score garbage."
        fi
        ;;
    drumnondrum)
        MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"
        MAX_POLYPHONY="${MAX_POLYPHONY:-16}"
        BASELINE="${BASELINE:-Y-dn}"
        ;;
    *) echo "ERROR: unknown TASK=$TASK"; exit 1 ;;
esac

OUT_ROOT="${OUT_ROOT:-temp/e3_${TASK}_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

if [[ -z "$CKPT_A2" && -z "$CKPT_C1" && -z "$CKPT_C2" && -z "$CKPT_B1" && -z "${Y_OUT:-}" ]]; then
    echo "ERROR: nothing to evaluate -- set at least one of CKPT_A2,"
    echo "       CKPT_C1, CKPT_C2, CKPT_B1, or Y_OUT."
    exit 1
fi
[[ -z "$CKPT_A2" ]] && echo "[note] CKPT_A2 unset: A.2 SKIPPED (C/Y-only conditional run)"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E3 CONDITIONAL EVALUATION   task=$TASK"
echo "CKPT_A2 = $CKPT_A2"
echo "CKPT_B1 = ${CKPT_B1:-<unset: B.1 SKIPPED>}"
echo "CKPT_C1 = ${CKPT_C1:-<unset: C.1 SKIPPED>}"
echo "CKPT_C2 = ${CKPT_C2:-<unset: C.2 SKIPPED>}"
echo "BASELINE=$BASELINE (absent in table -> means-only)"
echo "prompts  = $MEL_FOLDER / $CHORD_FOLDER"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)  "\
"GEN_LENGTH=$GEN_LENGTH ($((GEN_LENGTH / 16)) bars)  N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================================================"
nvidia-smi | head -n 12 || true

# GPU pre-flight (all generation stages call model.cuda()).
if ! python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"; then
    echo "ERROR: no usable CUDA GPU."
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "  CAUSE: not submitted via sbatch, so --gres never applied."
    else
        echo "  CAUSE: node $(hostname) has no working GPU; resubmit with"
        echo "         sbatch --exclude=$(hostname) ..."
    fi
    exit 1
fi

CKPT_ARGS=()
[[ -n "$CKPT_A2" ]] && CKPT_ARGS+=("$CKPT_A2")
[[ -n "$CKPT_B1" ]] && CKPT_ARGS+=("$CKPT_B1")
[[ -n "$CKPT_C1" ]] && CKPT_ARGS+=("$CKPT_C1")
[[ -n "$CKPT_C2" ]] && CKPT_ARGS+=("$CKPT_C2")
if [[ ${#CKPT_ARGS[@]} -gt 0 ]]; then
    python check_ckpts.py "${CKPT_ARGS[@]}" || {
        echo "ERROR: unreadable checkpoint(s) -- see above."; exit 1; }
fi

for d in "$MEL_FOLDER" "$CHORD_FOLDER"; do
    [[ -d "$d" ]] || { echo "ERROR: $d not found"; exit 1; }
done
N_MEL=$(find "$MEL_FOLDER" -maxdepth 1 \( -name '*.mid' -o -name '*.MID' \) 2>/dev/null | wc -l)
N_CHD=$(find "$CHORD_FOLDER" -maxdepth 1 \( -name '*.mid' -o -name '*.MID' \) 2>/dev/null | wc -l)
if [[ "$N_MEL" -eq 0 || "$N_MEL" -ne "$N_CHD" ]]; then
    echo "ERROR: prompt folders must hold the SAME songs ($N_MEL vs $N_CHD)."
    exit 1
fi
SONG_IDS=$(ls "$MEL_FOLDER" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
echo "[prompts] $N_MEL songs: $SONG_IDS"

# TASK vs prompt content. The B/C CLI says --drum-folder / --nondrum-folder
# on every task, so pointing a melchord run at the RWC drum prompts (or a
# drumnondrum run at POP909) raises no error anywhere -- it just scores
# nonsense. Drums are the one thing that separates the two corpora
# unambiguously, so check for them.
python - "$TASK" "$MEL_FOLDER" "$CHORD_FOLDER" <<'PYEOF'
import glob, os, sys, warnings
import pretty_midi
task, mel_dir, chord_dir = sys.argv[1], sys.argv[2], sys.argv[3]
n_drum = n_files = 0
for d in (mel_dir, chord_dir):
    for f in sorted(glob.glob(os.path.join(d, '*.mid')))[:5]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pm = pretty_midi.PrettyMIDI(f)
        n_files += 1
        if any(i.is_drum and i.notes for i in pm.instruments):
            n_drum += 1
if not n_files:
    sys.exit('no midis sampled from the prompt folders')
print(f'  sampled {n_files} prompt file(s); {n_drum} carry drum tracks')
if task == 'melchord' and n_drum:
    sys.exit(f'TASK=melchord but {n_drum}/{n_files} sampled prompts have '
             f'drum tracks -- MEL_FOLDER/CHORD_FOLDER point at a '
             f'drumnondrum corpus.')
if task == 'drumnondrum' and n_drum == 0:
    sys.exit(f'TASK=drumnondrum but none of the {n_files} sampled prompts '
             f'has a drum track -- the folders look like melchord.')
print(f'  prompt content matches TASK={task}.')
PYEOF

A2_OUT="$OUT_ROOT/A.2"
B1_OUT="$OUT_ROOT/B.1"
C1_OUT="$OUT_ROOT/C.1"
C2_OUT="$OUT_ROOT/C.2"

# ---------------------------------------------------------------------
# Conditional DIRECTION for the B/C systems (melchord):
#   mel2chord (default)  forward ckpts (task melchord_*): mod_a = melody
#   chord2mel            REVERSED ckpts (task melchord_*_rev): mod_a =
#                        chord conditioning, melody generated. Feeds the
#                        chord folder to the mod_a side, swaps the
#                        decoded track names (decode names by stream
#                        position), and relabels the manifest mode.
# A.2 is direction-agnostic (runs both modes natively) and unaffected.
# On drumnondrum the default mapping is the existing behaviour.
# ---------------------------------------------------------------------
DIRECTION="${DIRECTION:-mel2chord}"
case "$DIRECTION" in
    mel2chord)
        MODA_FOLDER="$MEL_FOLDER"; MODB_FOLDER="$CHORD_FOLDER"
        SWAP_ARGS=(); NORM_MODE=mel2chord ;;
    chord2mel)
        MODA_FOLDER="$CHORD_FOLDER"; MODB_FOLDER="$MEL_FOLDER"
        SWAP_ARGS=(--swap-stream-names); NORM_MODE=chord2mel ;;
    *) echo "ERROR: DIRECTION must be mel2chord or chord2mel"; exit 1 ;;
esac

# The B/C scripts were written for drumnondrum and their CLI still says
# --drum-folder / --nondrum-folder. Those names are positional only: the
# argument is the mod_a (conditioning) stream and mod_b (generated)
# stream whatever the task. On melchord they carry melody and chord, and
# --mode-name makes the OUTPUT say so, instead of writing drum2nondrum
# folders full of melody and chords. On drumnondrum the label stays
# drum2nondrum, matching every existing output tree.
if [[ "$TASK" == "melchord" ]]; then
    MODE_NAME="$NORM_MODE"
else
    MODE_NAME=drum2nondrum
fi
echo "DIRECTION=$DIRECTION   conditioning (mod_a, GIVEN in full) = $MODA_FOLDER"
echo "                       target       (mod_b, GENERATED)     = $MODB_FOLDER"
echo "B/C output mode label = $MODE_NAME  (--drum-folder/--nondrum-folder"
echo "                       are mod_a/mod_b slots, not drum semantics)"

# PROMPT_LENGTH seeds the GENERATED stream only -- the conditioning
# stream is given in FULL for all GEN_LENGTH frames, in every system
# (A.2's make_actions_conditional returns ('given', ...) at every frame;
# B.1/C.1/C.2 teacher-force mod_a from the prompt inside the AR loop).
# So this is the number of bars of the target track the model hears
# before it has to invent: 64 frames = 4 bars, the E3 default.
if (( PROMPT_LENGTH % 16 != 0 )); then
    echo "ERROR: PROMPT_LENGTH=$PROMPT_LENGTH is not a whole number of bars"
    echo "       (16 frames/bar). Use 64 for 4 bars."
    exit 1
fi
if (( PROMPT_LENGTH >= GEN_LENGTH )); then
    echo "ERROR: PROMPT_LENGTH ($PROMPT_LENGTH) >= GEN_LENGTH ($GEN_LENGTH):"
    echo "       nothing would be generated, and eval_metrics would score"
    echo "       an empty frame range."
    exit 1
fi
echo "target-track seed = $PROMPT_LENGTH frames ($((PROMPT_LENGTH / 16)) bars);"
echo "                    conditioning track given in full ($GEN_LENGTH frames)"

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing existing outputs"
else
    # -----------------------------------------------------------------
    # A.2 conditional modes. mel2chord = mod_a given, mod_b generated
    # (drum->nondrum on the drumnondrum task); chord2mel the reverse.
    # -----------------------------------------------------------------
    if [[ -n "$CKPT_A2" ]]; then
        echo "--- A.2 mel2chord + chord2mel ---"
        python cp_transformer_m2c_duet_block_diffusion_combined.py \
            --ckpt "$CKPT_A2" \
            --mel-folder "$MEL_FOLDER" --chord-folder "$CHORD_FOLDER" \
            --output-dir "$A2_OUT" \
            --modes mel2chord chord2mel \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY" \
            --model-size large --moe-num-experts 4 --moe-topk 2
    else
        echo "[A.2] SKIPPED (CKPT_A2 unset)"
    fi

    if [[ -n "$CKPT_B1" ]]; then
        if [[ "$DIRECTION" == "chord2mel" ]]; then
            echo "[B.1] SKIPPED: its inference has no reversed mode/naming;"
            echo "      evaluate B.1 in a DIRECTION=mel2chord run."
        else
            echo "--- B.1 drum2nondrum (leader given, its native mode) ---"
            python cp_transformer_m2c_duet_anticipatory_inference.py \
                --ckpt "$CKPT_B1" --mode drum2nondrum \
                --drum-folder "$MODA_FOLDER" --nondrum-folder "$MODB_FOLDER" \
                --output-dir "$B1_OUT" \
                --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
                --temperature 1.0 --n-samples "$N_SAMPLES" \
                --max-polyphony "$MAX_POLYPHONY" \
                --model-size large --moe-num-experts 4 --moe-topk 2 \
                --mode-name "$MODE_NAME"
        fi
    fi

    # C.1/C.2: purpose-built conditional models (mod_a given -> mod_b
    # generated; the ckpt's task decides which stream is which, and
    # DIRECTION must match it).
    if [[ -n "$CKPT_C1" ]]; then
        echo "--- C.1 rehearsal conditional ($DIRECTION, $N_SAMPLES sample(s)/song) ---"
        python cp_transformer_m2c_duet_rehearsal_inference.py \
            --ckpt "$CKPT_C1" \
            --drum-folder "$MODA_FOLDER" --nondrum-folder "$MODB_FOLDER" \
            --output-dir "$C1_OUT" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY" \
            --model-size large --moe-num-experts 4 --moe-topk 2 \
            --mode-name "$MODE_NAME" \
            "${SWAP_ARGS[@]}"
    fi
    if [[ -n "$CKPT_C2" ]]; then
        echo "--- C.2 prefix conditional ($DIRECTION, $N_SAMPLES sample(s)/song) ---"
        python cp_transformer_m2c_duet_prefix_inference.py \
            --ckpt "$CKPT_C2" \
            --drum-folder "$MODA_FOLDER" --nondrum-folder "$MODB_FOLDER" \
            --output-dir "$C2_OUT" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY" \
            --model-size large --moe-num-experts 4 --moe-topk 2 \
            --mode-name "$MODE_NAME" \
            "${SWAP_ARGS[@]}"
    fi
fi

# ---------------------------------------------------------------------
# piano listening copies (melchord only; scored files untouched)
# ---------------------------------------------------------------------
PIANO_COPIES="${PIANO_COPIES:-}"
if [[ -z "$PIANO_COPIES" ]]; then
    [[ "$TASK" == "melchord" ]] && PIANO_COPIES=1 || PIANO_COPIES=0
fi
if [[ "$PIANO_COPIES" == "1" ]]; then
    PIANO_DIRS=()
    # Every duet system, not just A.2/B.1: render_piano_copies walks the
    # tree and keys on track NAME, so the C.1/C.2 layouts work the same.
    # Leaving them out meant a C-only run (A.2 and B.1 both skipped)
    # produced no listening copies at all.
    for d in "$A2_OUT" "$B1_OUT" "$C1_OUT" "$C2_OUT"; do
        [[ -d "$d" ]] && PIANO_DIRS+=("$d")
    done
    [[ ${#PIANO_DIRS[@]} -gt 0 ]] && {
        echo "--- piano listening copies ---"
        python render_piano_copies.py "${PIANO_DIRS[@]}"; }
fi

# ---------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------
if [[ "$N_SAMPLES" -gt 1 ]]; then
    A2_LAYOUT=duet_multi; B1_LAYOUT=duet_multi; C_LAYOUT=duet_multi
else
    A2_LAYOUT=duet;       B1_LAYOUT=cond;       C_LAYOUT=cond
fi

RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
MANIFEST="results/e3_${TASK}_${RSUF}_manifest.tsv"
METRICS="results/e3_${TASK}_${RSUF}_metrics.csv"
TABLE="results/e3_${TASK}_${RSUF}_table.md"

# Optional published-baseline rows: point Y_OUT at a duet_multi root
# produced by eval_yinyang.sbatch / yinyang_e3_driver.py. Its mode
# dirs use the same direction labels, so the rows land in the same
# table sections; BASELINE=Y then gives paired tests vs YinYang.
Y_OUT="${Y_OUT:-}"
SRC_ARGS=()
[[ -d "$A2_OUT" ]] && SRC_ARGS+=(--source "A.2:${A2_LAYOUT}:${A2_OUT}")
[[ -n "$Y_OUT" && -d "$Y_OUT" ]] && SRC_ARGS+=(--source "Y:duet_multi:${Y_OUT}")
[[ -d "$B1_OUT" ]] && SRC_ARGS+=(--source "B.1:${B1_LAYOUT}:${B1_OUT}")
[[ -d "$C1_OUT" ]] && SRC_ARGS+=(--source "C.1:${C_LAYOUT}:${C1_OUT}")
[[ -d "$C2_OUT" ]] && SRC_ARGS+=(--source "C.2:${C_LAYOUT}:${C2_OUT}")

# Every system is optional, so it is possible to reach here with nothing
# to score -- e.g. A.2 skipped and the only other ckpt was B.1 in a
# chord2mel run (B.1 has no reversed mode), or SKIP_INFER=1 pointed at
# an OUT_ROOT that holds no system dirs. build_eval_manifest would fail
# with a bare argparse error; say what actually happened instead.
if [[ ${#SRC_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: no system produced output under $OUT_ROOT -- nothing to score."
    echo "  looked for: $A2_OUT (A.2), $B1_OUT (B.1), $C1_OUT (C.1), $C2_OUT (C.2)"
    [[ -n "$Y_OUT" ]] && echo "              $Y_OUT (Y)"
    echo "  Check that the ckpt(s) you passed match DIRECTION=$DIRECTION:"
    echo "  B.1 is skipped entirely in a chord2mel run, and C.1/C.2 need"
    echo "  reversed (melchord_*_rev) checkpoints for that direction."
    exit 1
fi

SYSTEMS=()
for a in "${SRC_ARGS[@]}"; do
    [[ "$a" == "--source" ]] || SYSTEMS+=("${a%%:*}")
done
echo "[systems] scoring: ${SYSTEMS[*]}"
if [[ ! " ${SYSTEMS[*]} " == *" $BASELINE "* ]]; then
    echo "[baseline] $BASELINE is NOT among the systems above: the table"
    echo "           prints means only, with no paired tests. Set BASELINE"
    echo "           to one of ${SYSTEMS[*]} to get p-values."
fi

python build_eval_manifest.py "${SRC_ARGS[@]}" \
    --songs $SONG_IDS \
    --modes mel2chord chord2mel drum2nondrum \
    --out "$MANIFEST"

# Normalize the direction label so every system's rows land in ONE table
# section. New runs already write the right label (--mode-name above),
# so this is a no-op for them; it still rescues OLDER output trees, and
# SKIP_INFER=1 reruns over them, where B.1/C.1/C.2 wrote 'drum2nondrum'
# regardless of task.
if grep -q $'\tdrum2nondrum\t' "$MANIFEST" && [[ "$NORM_MODE" != drum2nondrum ]]; then
    echo "[manifest] relabeling legacy drum2nondrum rows -> $NORM_MODE"
    sed -i $'s/\tdrum2nondrum\t/\t'"$NORM_MODE"$'\t/g' "$MANIFEST"
fi

python eval_metrics.py --task "$TASK" \
    --manifest "$MANIFEST" \
    --ref-a-dir "$MEL_FOLDER" --ref-b-dir "$CHORD_FOLDER" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline "$BASELINE" \
    --complete-cases \
    --given-stream auto \
    --out-md "$TABLE"

# Why so many rows read the same for every system: the conditioning
# stream is copied verbatim into every system's output, so every metric
# on it is identical by construction. This separates that (expected)
# from two real bugs -- systems sharing output paths, and systems
# generating identical notes.
echo "----------------------------------------------------------------"
python check_e3_identical.py --csv "$METRICS" --manifest "$MANIFEST" || true

echo "================================================================"
echo "E3 evaluation complete."
echo "  generated midis : $OUT_ROOT"
echo "  manifest        : $MANIFEST   (drum2nondrum relabeled mel2chord)"
echo "  per-sample CSV  : $METRICS"
echo "  table           : $TABLE"
echo "Reading guide:"
echo "  * Rows marked (=) measure the GIVEN stream -- the real partner,"
echo "    copied verbatim into every system's output. They are the same"
echo "    music in every column BY CONSTRUCTION (about 9 of the ~20 rows"
echo "    in mel2chord: the whole _a family, mel_stepwise_delta,"
echo "    mel_poly_rate, chord_tone_cov_ref) and are left untested."
echo "    They are not a bug and not evidence about any model."
echo "  * mel2chord section = partner GIVEN, target generated: read the"
echo "    TARGET stream's metrics (_b rows) plus the cross metrics"
echo "    (harmonic_rhythm_jsd, chord_tone_cov_delta) -- fit to a real"
echo "    partner is the whole question. chord2mel: read _a rows."
echo "  * FLOOR: compare against the same metrics in the E1 co table"
echo "    (results/e1_${TASK}_*_metrics.csv, mode=co) -- that is what the"
echo "    system achieves WITHOUT the partner given. Conditioning must"
echo "    beat it to mean anything."
echo "  * The GIVEN stream's rows (_a in mel2chord) are the copied real"
echo "    stream -- reference values, not generation quality."
echo "  * All systems drew $N_SAMPLES sample(s) per song."
echo "================================================================"
