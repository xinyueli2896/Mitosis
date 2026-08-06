#!/bin/bash
# SLURM: END-TO-END E1 (co-generation) evaluation.
#   1. collects the held-aside TEST SPLIT into prompt folders
#   2. runs co-generation inference for every system, saving outputs
#   3. runs the quantitative analysis (manifest -> metrics -> table)
#
# Systems evaluated:
#   A.2  duet block diffusion, co mode, frozen decode schedule
#   B.1  duet anticipatory, co mode (--mode co honours the k-frame
#        anticipatory re-indexing; the leader stream starts generating k
#        steps before the follower, which is the mechanism under test)
#   S1   single-stream in-domain finetune  (melchord: the MATCHED baseline)
#   S0   single-stream pretrained          (unmatched lower anchor)
#
# Env knobs (override via --export=ALL,KEY=val):
#   TASK          melchord (default) | drumnondrum
#   CKPT_A2       A.2 run dir or ckpt file        (required)
#
#   CHECKPOINT SELECTION: every system resolves to the SMALLEST
#   val_loss ckpt when given a run DIRECTORY (or a last.ckpt that
#   has val_loss-tagged siblings). Pass an explicit non-last
#   filename to pin a specific epoch instead. All systems share
#   this rule, so the comparison is best-val vs best-val.
#   CKPT_B1       B.1 run dir or ckpt file        (optional)
#   CKPT_S1       S1 ckpt file                    (melchord; optional)
#   CKPT_S0       S0 pretrained ckpt              (optional)
#   MEL_FOLDER    full melody/drum midis  (prompts AND scoring references)
#   CHORD_FOLDER  full chord/nondrum midis
#                 melchord defaults: input/pop909_split/{melody,chord}
#                 -- the held-aside POP909 test split.
#   SONGS         space-separated song ids to restrict to. DEFAULT EMPTY
#                 = evaluate every song present in the folders, i.e. the
#                 test split itself defines the evaluation set. Set this
#                 only to score a subset.
#   OUT_ROOT      default temp/e1_<TASK>_<JOBID>
#   BASELINE      system id the table is tested against
#                 (default: S1 for melchord, S0 for drumnondrum)
#   N_SAMPLES     samples per song for EVERY system (default 3). Scores
#                 are averaged over samples per song before the paired
#                 test, so raising this shrinks each song's standard
#                 error (~1/sqrt(n)) without changing the song count.
#                 That is the only lever on a fixed, small test set:
#                 it will not raise n, but it makes a real effect far
#                 more likely to hold the same sign on every song.
#   PROMPT_LENGTH prompt frames at 16 frames/bar: 64 = 4 bars (default),
#                 96 = 6 bars. Also sets the scoring window, so the
#                 metrics measure only what was GENERATED, not the prompt.
#   RESULT_TAG    extra suffix on the result filenames. Results are
#                 already scoped by prompt length
#                 (results/e1_<task>_p<PROMPT_LENGTH>_*), so a rerun at a
#                 different prompt length no longer overwrites the last
#                 one; use RESULT_TAG to separate runs that share it.
#   SKIP_INFER=1  reuse existing outputs under OUT_ROOT, score only
#
# Example:
#   sbatch --export=ALL,\
# CKPT_A2=ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_melchord_batch_8_schedule/,\
# CKPT_S1=ckpt/cp_transformer_v0.42_size1_pop909ft_short_batch_48_schedule/last.ckpt \
#       midi_yinyang/eval_e1.sbatch

#SBATCH --job-name=mitosis_eval_e1
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
CKPT_S1="${CKPT_S1:-}"
CKPT_S0="${CKPT_S0:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
N_SAMPLES="${N_SAMPLES:-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
SKIP_INFER="${SKIP_INFER:-0}"

case "$TASK" in
    melchord)
        # The held-aside POP909 TEST SPLIT. These folders define the
        # evaluation set: every song in them is evaluated, so no id list
        # is hardcoded here (a stale list would silently evaluate the
        # wrong songs, or fail on songs the split does not contain).
        MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
        SONGS="${SONGS:-}"          # empty = every song in the folders
        MAX_POLYPHONY_DUET=4
        BASELINE="${BASELINE:-S1}"
        ;;
    drumnondrum)
        MEL_FOLDER="${MEL_FOLDER:-input/rwc_test_prompts_split/drum}"
        CHORD_FOLDER="${CHORD_FOLDER:-input/rwc_test_prompts_split/nondrum}"
        SONGS="${SONGS:-}"          # empty = all songs in the folders
        MAX_POLYPHONY_DUET=16
        BASELINE="${BASELINE:-S0}"
        ;;
    *) echo "ERROR: unknown TASK=$TASK"; exit 1 ;;
esac

OUT_ROOT="${OUT_ROOT:-temp/e1_${TASK}_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E1 CO-GENERATION EVALUATION   task=$TASK"
echo "CKPT_A2 = ${CKPT_A2:-<unset: A.2 will be SKIPPED>}"
echo "CKPT_B1 = ${CKPT_B1:-<unset: B.1 will be SKIPPED>}"
echo "CKPT_S1 = ${CKPT_S1:-<unset: S1 will be SKIPPED>}"
echo "CKPT_S0 = ${CKPT_S0:-<unset: S0 will be SKIPPED>}"
echo "MEL_FOLDER=$MEL_FOLDER  CHORD_FOLDER=$CHORD_FOLDER"
echo "SONGS=${SONGS:-<all>}"
echo "OUT_ROOT=$OUT_ROOT   BASELINE=$BASELINE"
echo "PROMPT_LENGTH=$PROMPT_LENGTH GEN_LENGTH=$GEN_LENGTH N_SAMPLES=$N_SAMPLES"
echo "PROMPT bars: $((PROMPT_LENGTH / 16))  (16 frames/bar)"
echo "================================================================"
nvidia-smi | head -n 12 || true

# ---------------------------------------------------------------------
# 0a. GPU pre-flight.
#     Every generation stage below calls model.cuda(). Without this the
#     absence of a GPU surfaces only once python is deep inside the
#     first inference script ("RuntimeError: No CUDA GPUs are
#     available"), by which point the job has burned its queue slot.
#     nvidia-smi above is NOT sufficient: it is piped through `|| true`
#     so a failure is silent, and a node can list a device that torch
#     still cannot initialise. Ask torch directly -- that is what the
#     generation scripts actually use -- and force a real context init.
# ---------------------------------------------------------------------
if ! python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"; then
    echo "----------------------------------------------------------------"
    echo "ERROR: no usable CUDA GPU in this job; every stage would fail."
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "  CAUSE: SLURM_JOB_ID is unset -- this was NOT run through"
        echo "         sbatch, so the #SBATCH --gres=gpu:1 line never took"
        echo "         effect. Submit it instead of executing it:"
        echo "             sbatch --export=ALL,... midi_yinyang/eval_e1.sbatch"
    else
        echo "  CAUSE: allocated node $(hostname) has no working GPU"
        echo "         (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}')."
        echo "         Resubmit excluding it:"
        echo "             sbatch --exclude=$(hostname) --export=ALL,... \\"
        echo "                 midi_yinyang/eval_e1.sbatch"
    fi
    echo "----------------------------------------------------------------"
    exit 1
fi

# ---------------------------------------------------------------------
# 0. checkpoint integrity pre-flight
#    A truncated/corrupt ckpt otherwise surfaces as an opaque
#    "PytorchStreamReader ... failed finding central directory" stack
#    trace partway through the run. Report it up front instead; the
#    resolver skips unreadable files, so a warning here is survivable
#    but tells you the picked ckpt may not be the best-val one.
# ---------------------------------------------------------------------
for _c in "$CKPT_A2" "$CKPT_B1" "$CKPT_S1" "$CKPT_S0"; do
    [[ -n "$_c" && -e "$_c" ]] && python check_ckpts.py "$_c" || true
done
echo "================================================================"

# ---------------------------------------------------------------------
# 1. eval-subset prompt folders
#    Inference iterates whole folders, so restricting to the held-out
#    songs here is what keeps this from decoding all ~900 POP909 songs.
#    The SAME full midis serve as prompts (first PROMPT_LENGTH frames)
#    and as scoring references (frames PROMPT_LENGTH..GEN_LENGTH).
# ---------------------------------------------------------------------
EVAL_MEL="$OUT_ROOT/prompts/mel"
EVAL_CHORD="$OUT_ROOT/prompts/chord"
EVAL_MERGED="$OUT_ROOT/prompts/merged_tagged"
mkdir -p "$EVAL_MEL" "$EVAL_CHORD"

if [[ -n "$SONGS" ]]; then
    for sid in $SONGS; do
        for pair in "$MEL_FOLDER:$EVAL_MEL" "$CHORD_FOLDER:$EVAL_CHORD"; do
            src_dir="${pair%%:*}"; dst_dir="${pair##*:}"
            src=$(ls "$src_dir/$sid".mid "$src_dir/$sid".MID 2>/dev/null | head -1 || true)
            if [[ -z "$src" ]]; then
                echo "ERROR: song $sid not found in $src_dir"; exit 1
            fi
            cp -f "$src" "$dst_dir/$sid.mid"
        done
    done
else
    # Both extension cases: RWC files are named *.SMF_SYNC.MID (uppercase),
    # POP909 files *.mid -- a lowercase-only glob silently collects NOTHING
    # from the RWC split and the run dies at the count check below.
    cp -f "$MEL_FOLDER"/*.mid "$MEL_FOLDER"/*.MID "$EVAL_MEL"/ 2>/dev/null || true
    cp -f "$CHORD_FOLDER"/*.mid "$CHORD_FOLDER"/*.MID "$EVAL_CHORD"/ 2>/dev/null || true
fi
N_EVAL=$(ls "$EVAL_MEL" 2>/dev/null | wc -l)
N_EVAL_CHORD=$(ls "$EVAL_CHORD" 2>/dev/null | wc -l)
if [[ "$N_EVAL" -eq 0 ]]; then
    echo "ERROR: no midis collected from MEL_FOLDER=$MEL_FOLDER"
    echo "       (does the test split exist? expected .mid files there)"
    exit 1
fi
if [[ "$N_EVAL" -ne "$N_EVAL_CHORD" ]]; then
    echo "ERROR: melody/chord test split sizes differ ($N_EVAL vs"
    echo "       $N_EVAL_CHORD). The two folders must hold the SAME songs"
    echo "       -- scoring pairs them by filename."
    exit 1
fi
echo "[prompts] $N_EVAL eval songs -> $EVAL_MEL , $EVAL_CHORD"
echo "[prompts] songs: $(ls "$EVAL_MEL" | sed 's/\.mid$//' | tr '\n' ' ')"

# merged, program-tagged prompts for the single-stream systems: the
# tokenizer separates streams by PROGRAM only, so the chord stream must
# carry a distinct program or the two parts fuse irreversibly.
if [[ "$TASK" == "melchord" ]]; then
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" \
        --dst "$EVAL_MERGED" --chord-program 48
else
    # drumnondrum: drums are natively program 127, so a plain merge keeps
    # the streams distinguishable.
    python merge_melody_chord.py \
        --melody "$EVAL_MEL" --chord "$EVAL_CHORD" --dst "$EVAL_MERGED"
fi
echo "[prompts] merged+tagged -> $EVAL_MERGED"

# ---------------------------------------------------------------------
# 2. inference
# ---------------------------------------------------------------------
A2_OUT="$OUT_ROOT/A.2"
B1_OUT="$OUT_ROOT/B.1"
S1_OUT_NAME="e1_${TASK}_S1_${SLURM_JOB_ID:-local}"
S0_OUT_NAME="e1_${TASK}_S0_${SLURM_JOB_ID:-local}"
S1_OUT="temp/$S1_OUT_NAME"
S0_OUT="temp/$S0_OUT_NAME"

if [[ "$SKIP_INFER" == "1" ]]; then
    echo "[infer] SKIP_INFER=1 -- reusing existing outputs"
else
    if [[ -n "$CKPT_A2" ]]; then
        echo "--- A.2 co-generation (K=4 schedule from infer defaults) ---"
        export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
        export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
        export A3_TOP_P="${A3_TOP_P:-0.95}"
        export A3_ADAPTIVE="${A3_ADAPTIVE:-1}"
        echo "[A.2] K=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P adaptive=$A3_ADAPTIVE"
        python cp_transformer_m2c_duet_block_diffusion_combined.py \
            --ckpt "$CKPT_A2" \
            --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
            --output-dir "$A2_OUT" \
            --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$MAX_POLYPHONY_DUET" \
            --model-size large --moe-num-experts 4 --moe-topk 2
    else
        echo "[A.2] SKIPPED (CKPT_A2 unset)"
    fi

    if [[ -n "$CKPT_B1" ]]; then
        echo "--- B.1 co-generation (anticipatory, --mode co) ---"
        # --mode co prompts BOTH streams then generates both, honouring
        # the k-frame lead (auto-detected from the ckpt path's _k<N>_).
        python cp_transformer_m2c_duet_anticipatory_inference.py \
            --ckpt "$CKPT_B1" \
            --mode co \
            --drum-folder "$EVAL_MEL" --nondrum-folder "$EVAL_CHORD" \
            --output-dir "$B1_OUT" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 \
            --max-polyphony "$MAX_POLYPHONY_DUET" \
            --model-size large --moe-num-experts 4 --moe-topk 2
    else
        echo "[B.1] SKIPPED (CKPT_B1 unset)"
    fi

    for spec in "S1:$CKPT_S1:$S1_OUT_NAME" "S0:$CKPT_S0:$S0_OUT_NAME"; do
        sys="${spec%%:*}"; rest="${spec#*:}"
        ckpt="${rest%%:*}"; name="${rest##*:}"
        if [[ -z "$ckpt" ]]; then echo "[$sys] SKIPPED (ckpt unset)"; continue; fi
        if [[ ! -e "$ckpt" ]]; then echo "[$sys] SKIPPED (ckpt not found: $ckpt)"; continue; fi
        echo "--- $sys single-stream continuation ---"
        python cp_transformer_inference.py \
            --ckpt "$ckpt" \
            --midi-folder "$EVAL_MERGED" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --save-name "$name"
    done
fi

# ---------------------------------------------------------------------
# 2b. piano listening copies (melchord). The scored files keep the chord
#     stream on program 48 (strings) BY DESIGN -- that program is the
#     stream identity: the tokenizer separates the merged prompt by it
#     and eval_metrics attributes S0/S1 output notes by it
#     (chord_programs={48}). Rewriting programs in the scored files
#     would re-fuse the streams for the scorer, so listening gets
#     parallel <dir>_piano copies instead: every non-drum track set to
#     program 0 (Acoustic Grand -- the GM slot a Steinway soundfont
#     occupies), names/notes/timing preserved. PIANO_COPIES=0 disables.
# ---------------------------------------------------------------------
PIANO_COPIES="${PIANO_COPIES:-}"
if [[ -z "$PIANO_COPIES" ]]; then
    [[ "$TASK" == "melchord" ]] && PIANO_COPIES=1 || PIANO_COPIES=0
fi
if [[ "$PIANO_COPIES" == "1" ]]; then
    PIANO_DIRS=()
    for d in "$A2_OUT" "$B1_OUT" "$S1_OUT" "$S0_OUT"; do
        [[ -d "$d" ]] && PIANO_DIRS+=("$d")
    done
    if [[ ${#PIANO_DIRS[@]} -gt 0 ]]; then
        echo "--- piano listening copies (originals untouched) ---"
        python render_piano_copies.py "${PIANO_DIRS[@]}"
    fi
fi

# ---------------------------------------------------------------------
# 3. quantitative analysis
# ---------------------------------------------------------------------
# Result files are scoped by prompt length (and an optional RESULT_TAG).
# They used to be fixed at results/e1_<task>_{manifest,metrics,table}, so
# a rerun under different settings silently overwrote the previous run's
# numbers -- with no way to tell afterwards which prompt length produced
# a given table. PROMPT_LENGTH is in FRAMES at 16 frames/bar
# (eval_metrics.FRAMES_PER_BAR), i.e. 64 = 4 bars, 96 = 6 bars.
RESULT_TAG="${RESULT_TAG:-}"
RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
MANIFEST="results/e1_${TASK}_${RSUF}_manifest.tsv"
METRICS="results/e1_${TASK}_${RSUF}_metrics.csv"
TABLE="results/e1_${TASK}_${RSUF}_table.md"

# The duet scripts write the 'duet' layout (<song>/<mode>.mid) for a
# single sample and 'duet_multi' (<song>/<mode>/sample_<i>_temp<T>.mid)
# when sampling; the manifest scanner needs to be told which.
DUET_LAYOUT=duet
[[ "$N_SAMPLES" -gt 1 ]] && DUET_LAYOUT=duet_multi

SRC_ARGS=()
[[ -d "$A2_OUT" ]] && SRC_ARGS+=(--source "A.2:$DUET_LAYOUT:$A2_OUT")
[[ -d "$B1_OUT" ]] && SRC_ARGS+=(--source "B.1:$DUET_LAYOUT:$B1_OUT")
[[ -d "$S1_OUT" ]] && SRC_ARGS+=(--source "S1:single:$S1_OUT")
[[ -d "$S0_OUT" ]] && SRC_ARGS+=(--source "S0:single:$S0_OUT")
if [[ ${#SRC_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: no inference outputs found to score."
    exit 1
fi

# Always pass the evaluation song list. --songs does double duty in
# build_eval_manifest: it filters, AND it is what enables the [coverage]
# report that names the songs each system failed to produce. Passing it
# only when SONGS is set (the old behaviour) disabled that report in the
# default configuration -- which is how a run scored 5 songs out of a
# ~90-song test split without a word about the other 85. The list is
# derived from the prompts actually staged, so as a filter it is a no-op.
EVAL_SONG_IDS=$(ls "$EVAL_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
SONG_ARGS=(--songs $EVAL_SONG_IDS)
echo "[coverage] expecting $N_EVAL song(s) per system"

echo "================================================================"
echo "--- building manifest ---"
python build_eval_manifest.py "${SRC_ARGS[@]}" "${SONG_ARGS[@]}" \
    --modes co --out "$MANIFEST"

echo "--- scoring (H1-H3) ---"
python eval_metrics.py --task "$TASK" \
    --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

# Primary analysis. Both flags default ON because they are what makes the
# comparison sound on this test set:
#   ONE_SIDED      H1-H3 predict a direction, and at n=5 the two-sided
#                  exact test floors at p=0.0625 -- above 0.05 at ANY
#                  effect size, so a two-sided-only run cannot produce a
#                  significant result no matter what the models do.
#   COMPLETE_CASES scores every system on the same songs, instead of
#                  grading each on whichever songs it happened to handle.
# A plain two-sided / all-cases table is also written alongside, so the
# conservative reading stays available and nothing is hidden by a default.
ONE_SIDED="${ONE_SIDED:-1}"
COMPLETE_CASES="${COMPLETE_CASES:-1}"
AGG_ARGS=()
[[ "$ONE_SIDED" == "1" ]] && AGG_ARGS+=(--one-sided)
[[ "$COMPLETE_CASES" == "1" ]] && AGG_ARGS+=(--complete-cases)

echo "--- comparison table (paired Wilcoxon vs $BASELINE) ---"
echo "[agg] primary: ${AGG_ARGS[*]:-<two-sided, all cases>}"
python aggregate_eval_results.py --csv "$METRICS" \
    --task "$TASK" --baseline "$BASELINE" --mode co \
    "${AGG_ARGS[@]}" --out-md "$TABLE"

if [[ ${#AGG_ARGS[@]} -gt 0 ]]; then
    TABLE_2S="${TABLE%.md}_twosided.md"
    echo "--- reference table (two-sided, all cases) ---"
    python aggregate_eval_results.py --csv "$METRICS" \
        --task "$TASK" --baseline "$BASELINE" --mode co \
        --out-md "$TABLE_2S"
fi

echo "================================================================"
echo "E1 evaluation complete."
echo "  generated midis : $OUT_ROOT (A.2, B.1), $S1_OUT, $S0_OUT"
[[ "$PIANO_COPIES" == "1" ]] && echo "  piano copies    : same folders with _piano suffix (listening only)"
echo "  manifest        : $MANIFEST"
echo "  per-sample CSV  : $METRICS"
echo "  comparison table: $TABLE   (primary)"
[[ ${#AGG_ARGS[@]} -gt 0 ]] && echo "  two-sided table : ${TABLE%.md}_twosided.md"
echo "Note: B.1 rows should be read PER-STREAM (its design predicts the"
echo "follower improves while the leader stays flat) -- see EXPERIMENTS.md E1."
