#!/bin/bash
# LOCAL (no SLURM): the E1 loop on a laptop -- stage, tag, generate,
# score -- for a handful of songs. This is the Mac counterpart of
# eval_e1.sbatch, not a replacement for it: on the cluster every stage
# is still an sbatch.
#
# Runs from midi_yinyang/ inside a Python env that has torch (arm64
# build on Apple silicon), numpy, scipy, pretty_midi, mido, lightning,
# pytorch_lightning, wandb, joblib, six. The vendored transformers fork
# is picked up from the repo. On Apple silicon the model lands on the
# Metal backend; PYTORCH_ENABLE_MPS_FALLBACK=1 is set below so any op
# it lacks runs on the CPU. MITOSIS_DEVICE=cpu forces the CPU.
#
# Env knobs:
#   SPLIT          melody/chord folders, default input/nottingham_split
#   OUT_ROOT       default temp/E1_local
#   SYSTEMS        default "A12". Known: A12 (cp8, chord tagged 48,
#                  refine with one step), A3 (cp4, tag 0, default
#                  refine), A3ctcaT (cp4, tag 0, alternate commit at
#                  T=1), A3fc / A3fcaT (cp8, tag 0), A1 (causal duet,
#                  cp4, tag 0), S1 / S-scratch (merged single-stream,
#                  cp16, prompts merged with the chord at 48). Each
#                  needs its checkpoint in CKPT_<name> (a run dir or a
#                  .ckpt file; CKPT_A3 serves A3ctcaT, CKPT_A3FC serves
#                  A3fcaT, CKPT_SSCRATCH serves S-scratch). AMT (the
#                  Anticipatory Music Transformer) needs no checkpoint
#                  variable: it runs from its own venv, built once with
#                  `bash setup_amt.sbatch` from the repo root. The
#                  cascades P-mc / P-cm run through pipeline_cogen and
#                  are not wired here.
#   GPU            CUDA device index for this run (sets
#                  CUDA_VISIBLE_DEVICES); run one SYSTEMS list per GPU
#                  in parallel shells on a multi-GPU box.
#   PROMPT_LENGTH  default 80        GEN_LENGTH default 416 -- MUST fit
#                  the shortest song: the reference is the same file,
#                  and frames past its end are silence in the reference
#   N_SAMPLES      default 3
#   EXCLUDE_SONGS  songs left out of generation AND scoring, default
#                  "506 786" as in eval_e1 (chord silent in the prompt
#                  window). EXCLUDE_SONGS="" keeps every song in SPLIT.
#   BASELINE       system the paired table compares against, default
#                  the first in SYSTEMS
#   SKIP_INFER=1   score what is already under OUT_ROOT
#   RESULT_TAG     suffix of the result files; default local-<systems>,
#                  so parallel runs never share a results file
#
# Several runs sharing one OUT_ROOT, one per GPU, is the intended use
# on a multi-GPU box: prompt staging is done once under a lock, every
# system writes its own folder, and each run scores into its own
# results file. Score them all together afterwards with SKIP_INFER=1.
#
# Example (Nottingham, A12 only):
#   CKPT_A12=export/a12k1/epoch=..weights.ckpt GEN_LENGTH=256 bash local_e1.sh
#
# Outputs: <OUT_ROOT>/<system>/<song>/co/sample_<i>_temp1.0.mid and
# results/E1local_p<PROMPT_LENGTH>_{manifest.tsv,metrics.csv,pooled.csv,
# table.md}.

set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

SPLIT="${SPLIT:-input/nottingham_split}"
OUT_ROOT="${OUT_ROOT:-temp/E1_local}"
SYSTEMS="${SYSTEMS:-A12}"
PROMPT_LENGTH="${PROMPT_LENGTH:-80}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
SKIP_INFER="${SKIP_INFER:-0}"
# default tag names the run's systems, so parallel single-system runs
# write results/E1_p80_local-A12_*, results/E1_p80_local-AMT_*, ... and
# the final joint scoring (SKIP_INFER=1, all systems) gets its own file
RESULT_TAG="${RESULT_TAG:-local-$(echo $SYSTEMS | tr ' ' '+')}"
BASELINE="${BASELINE:-${SYSTEMS%% *}}"

EXCLUDE_SONGS="${EXCLUDE_SONGS-506 786}"
MEL_SRC="$SPLIT/melody"; CHORD_SRC="$SPLIT/chord"
[[ -d "$MEL_SRC" && -d "$CHORD_SRC" ]] || { echo "ERROR: no $MEL_SRC / $CHORD_SRC"; exit 1; }
SONG_IDS=""
for s in $(ls "$MEL_SRC" | sed 's/\.[Mm][Ii][Dd]$//' | sort); do
    skip=0
    for x in $EXCLUDE_SONGS; do [[ "$s" == "$x" ]] && skip=1; done
    [[ "$skip" == 0 ]] && SONG_IDS="$SONG_IDS$s "
done
SONG_IDS="${SONG_IDS% }"
echo "songs ($(echo $SONG_IDS | wc -w)): $SONG_IDS"
[[ -n "$EXCLUDE_SONGS" ]] && echo "excluded: $EXCLUDE_SONGS"
echo "prompt $PROMPT_LENGTH frames, total $GEN_LENGTH, $N_SAMPLES samples; systems: $SYSTEMS"

# per-system settings, mirroring eval_e1.sbatch
sys_cfg() {   # -> "cp program schedule steps temp top_p ckpt_var kind"
    case "$1" in
        A12)       echo "8 48 refine 1 1.0 1.0 CKPT_A12 diffusion" ;;
        A3)        echo "4 0 refine - - - CKPT_A3 diffusion" ;;
        A3ctcaT)   echo "4 0 ctc_alt - 1.0 1.0 CKPT_A3 diffusion" ;;
        A3fc)      echo "8 0 ctc_m - - - CKPT_A3FC diffusion" ;;
        A3fcaT)    echo "8 0 ctc_alt - 1.0 1.0 CKPT_A3FC diffusion" ;;
        A1)        echo "4 0 - - - - CKPT_A1 causal" ;;
        S1)        echo "16 48 - - - - CKPT_S1 single" ;;
        S-scratch) echo "16 48 - - - - CKPT_SSCRATCH single" ;;
        AMT)       echo "- 48 - - - - AMT_DIR amt" ;;
        *) echo "ERROR: unknown system $1" >&2; return 1 ;;
    esac
}
sys_layout() { case "$1" in S1|S-scratch) echo single ;; *) echo duet_multi ;; esac; }

# ---- 1. prompts: one staged copy per chord program ------------------
# Several runs may share OUT_ROOT (one system per GPU, in parallel), so
# a stage is built by whoever gets the lock first and the others wait
# for its .ready mark instead of copying over each other.
wait_or_lock() {   # wait_or_lock <dir> -> 0 = caller must build, 1 = ready
    local dir="$1"
    [[ -f "$dir/.ready" ]] && return 1
    if mkdir "$dir.lock" 2>/dev/null; then return 0; fi
    while [[ ! -f "$dir/.ready" ]]; do sleep 2; done
    return 1
}
stage() {   # stage <program> -> folder with mel/ chord/ (chord tagged)
    local prog="$1" dir="$OUT_ROOT/prompts_prog$1"
    if wait_or_lock "$dir"; then
        mkdir -p "$dir/mel" "$dir/chord"
        for s in $SONG_IDS; do
            cp -f "$MEL_SRC/$s.mid" "$dir/mel/"; cp -f "$CHORD_SRC/$s.mid" "$dir/chord/"
        done
        python - "$dir/chord" "$prog" <<'RETAG'
import glob, os, sys
import mido
folder, prog = sys.argv[1], int(sys.argv[2])
n = 0
for f in sorted(glob.glob(os.path.join(folder, '*.mid'))):
    m = mido.MidiFile(f); seen = False
    for tr in m.tracks:
        for msg in tr:
            if msg.type == 'program_change':
                msg.program = prog; seen = True
    if not seen:
        for tr in m.tracks:
            if any(msg.type == 'note_on' for msg in tr):
                tr.insert(0, mido.Message('program_change', program=prog, time=0)); break
    m.save(f); n += 1
print(f'[prompts] chord program -> {prog} on {n} file(s)')
RETAG
        touch "$dir/.ready"; rmdir "$dir.lock"
    fi
    echo "$dir"
}
# the single-stream models read ONE file per song, both streams merged
# with the chord on program 48 -- the only thing that keeps them apart
stage_merged() {
    local dir="$OUT_ROOT/prompts_merged"
    if wait_or_lock "$dir"; then
        python merge_melody_chord.py --melody "$MEL_SRC" --chord "$CHORD_SRC" \
            --dst "$dir" --chord-program 48 --ids $SONG_IDS
        touch "$dir/.ready"; rmdir "$dir.lock"
    fi
    echo "$dir"
}

# ---- 2. generate --------------------------------------------------------
SCORED=()
for name in $SYSTEMS; do
    read -r cp prog sched steps temp topp ckvar kind <<<"$(sys_cfg "$name")"
    ck="${!ckvar:-}"
    dir="$OUT_ROOT/$name"
    if [[ "$SKIP_INFER" == "1" ]]; then
        [[ -d "$dir" ]] && SCORED+=("$name") || echo "[$name] not generated; skipped"
        continue
    fi
    if [[ "$kind" == amt ]]; then
        ck="${AMT_DIR:-external/anticipation}"
        [[ -d "$ck/.venv" ]] || { echo "[$name] SKIPPED: no venv at $ck/.venv -- run: cd .. && bash midi_yinyang/setup_amt.sbatch"; continue; }
    else
        [[ -n "$ck" && -e "$ck" ]] || { echo "[$name] SKIPPED: set $ckvar to its checkpoint"; continue; }
    fi
    [[ -d "$ck" ]] && ckarg="$ck/" || ckarg="$ck"
    echo "================================================================"
    echo "[$name] ckpt=$ck  cp=$cp  chord program=$prog  kind=$kind${sched:+  schedule=$sched}"
    case "$kind" in
    diffusion)
        pdir=$(stage "$prog")
        denv=(A3_SCHEDULE="$sched")
        [[ "$steps" != "-" ]] && denv+=(A3_REFINE_STEPS="$steps")
        [[ "$temp" != "-" ]] && denv+=(A3_FINAL_TEMP="$temp")
        [[ "$topp" != "-" ]] && denv+=(A3_TOP_P="$topp")
        MELCHORD_CP="$cp" env "${denv[@]}" python cp_transformer_m2c_duet_block_diffusion_combined.py \
            --ckpt "$ckarg" \
            --mel-folder "$pdir/mel" --chord-folder "$pdir/chord" \
            --output-dir "$dir" --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$cp" --skip-existing \
            --model-size large --moe-num-experts 4 --moe-topk 2 \
            --min-chord-tokens-before-eos 0 ;;
    causal)
        pdir=$(stage "$prog")
        MELCHORD_CP="$cp" python cp_transformer_m2c_intra_cross_attn_combined.py \
            --ckpt "$ckarg" \
            --mel-folder "$pdir/mel" --chord-folder "$pdir/chord" \
            --output-dir "$dir" --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$cp" --skip-existing \
            --model-size large --moe-num-experts 4 --moe-topk 2 ;;
    amt)
        # their wrapper is portable bash: it merges the two streams with
        # the chord at 48 itself and writes <song>/co/sample_<i>.mid.
        # CUDA_VISIBLE_DEVICES (GPU=) is inherited, so it lands on the
        # same card as this run.
        REPO_DIR="$(cd .. && pwd)" AMT_DIR="$(readlink -f "$ck")" \
        MEL_FOLDER="$(readlink -f "$MEL_SRC")" CHORD_FOLDER="$(readlink -f "$CHORD_SRC")" \
        OUT_DIR="$(readlink -f "$OUT_ROOT")/AMT" IDS="$SONG_IDS" \
        PROMPT_FRAMES="$PROMPT_LENGTH" GEN_FRAMES="$GEN_LENGTH" N_SAMPLES="$N_SAMPLES" \
            bash infer_amt_prompted.sbatch ;;
    single)
        mdir=$(stage_merged)
        # cp_transformer_inference writes temp/<save-name>/ itself, so
        # OUT_ROOT has to live under temp/ for the layout to line up
        [[ "$OUT_ROOT" == temp/* ]] || { echo "[$name] needs OUT_ROOT under temp/"; continue; }
        python cp_transformer_inference.py \
            --ckpt "$ck" \
            --midi-folder "$mdir" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$cp" --skip-existing \
            --save-name "${dir#temp/}" ;;
    esac
    SCORED+=("$name")
done
[[ ${#SCORED[@]} -gt 0 ]] || { echo "ERROR: nothing to score"; exit 1; }

# ---- 3. score -------------------------------------------------------------
mkdir -p results
RSUF="p${PROMPT_LENGTH}_${RESULT_TAG}"
MANIFEST="results/E1_${RSUF}_manifest.tsv"
METRICS="results/E1_${RSUF}_metrics.csv"
POOLED="results/E1_${RSUF}_pooled.csv"
TABLE="results/E1_${RSUF}_table.md"
SRC_ARGS=()
for name in "${SCORED[@]}"; do SRC_ARGS+=(--source "$name:$(sys_layout "$name"):$OUT_ROOT/$name"); done
python build_eval_manifest.py "${SRC_ARGS[@]}" --songs $SONG_IDS --modes co --out "$MANIFEST"
# references: the UNTAGGED split -- the scorer reads all notes of each
# single-stream file, so the program does not enter
python eval_metrics.py --task melchord --manifest "$MANIFEST" \
    --ref-a-dir "$MEL_SRC" --ref-b-dir "$CHORD_SRC" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS" --pooled-out "$POOLED" \
    --pooled-boot-mode songs --pooled-weight song
[[ " ${SCORED[*]} " == *" $BASELINE "* ]] || BASELINE="${SCORED[0]}"
python aggregate_eval_results.py --csv "$METRICS" --task melchord \
    --baseline "$BASELINE" --mode co --one-sided --complete-cases --out-md "$TABLE"
echo "================================================================"
echo "done.  outputs $OUT_ROOT/<system>/   metrics $METRICS   table $TABLE"
echo "With 5 songs the paired tests have no power; read the per-song"
echo "values in $METRICS and listen. The pooled JSD bootstrap over 5"
echo "songs is wide by construction."
