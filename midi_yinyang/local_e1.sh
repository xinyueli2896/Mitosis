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
#                  T=1), A3fc / A3fcaT (cp8, tag 0). Each needs its
#                  checkpoint in CKPT_<name> (a run dir or a .ckpt file;
#                  CKPT_A3 serves A3ctcaT, CKPT_A3FC serves A3fcaT).
#   PROMPT_LENGTH  default 80        GEN_LENGTH default 416 -- MUST fit
#                  the shortest song: the reference is the same file,
#                  and frames past its end are silence in the reference
#   N_SAMPLES      default 3
#   BASELINE       system the paired table compares against, default
#                  the first in SYSTEMS
#   SKIP_INFER=1   score what is already under OUT_ROOT
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

SPLIT="${SPLIT:-input/nottingham_split}"
OUT_ROOT="${OUT_ROOT:-temp/E1_local}"
SYSTEMS="${SYSTEMS:-A12}"
PROMPT_LENGTH="${PROMPT_LENGTH:-80}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
SKIP_INFER="${SKIP_INFER:-0}"
RESULT_TAG="${RESULT_TAG:-local}"
BASELINE="${BASELINE:-${SYSTEMS%% *}}"

MEL_SRC="$SPLIT/melody"; CHORD_SRC="$SPLIT/chord"
[[ -d "$MEL_SRC" && -d "$CHORD_SRC" ]] || { echo "ERROR: no $MEL_SRC / $CHORD_SRC"; exit 1; }
SONG_IDS=$(ls "$MEL_SRC" | sed 's/\.[Mm][Ii][Dd]$//' | sort | tr '\n' ' ')
echo "songs: $SONG_IDS"
echo "prompt $PROMPT_LENGTH frames, total $GEN_LENGTH, $N_SAMPLES samples; systems: $SYSTEMS"

# per-system settings, mirroring eval_e1.sbatch
sys_cfg() {   # -> "cp program schedule steps temp top_p ckpt_var"
    case "$1" in
        A12)     echo "8 48 refine 1 1.0 1.0 CKPT_A12" ;;
        A3)      echo "4 0 refine - - - CKPT_A3" ;;
        A3ctcaT) echo "4 0 ctc_alt - 1.0 1.0 CKPT_A3" ;;
        A3fc)    echo "8 0 ctc_m - - - CKPT_A3FC" ;;
        A3fcaT)  echo "8 0 ctc_alt - 1.0 1.0 CKPT_A3FC" ;;
        *) echo "ERROR: unknown system $1" >&2; return 1 ;;
    esac
}

# ---- 1. prompts: one staged copy per chord program ------------------
stage() {   # stage <program> -> folder with mel/ chord/ (chord tagged)
    local prog="$1" dir="$OUT_ROOT/prompts_prog$1"
    if [[ ! -f "$dir/.ready" ]]; then
        mkdir -p "$dir/mel" "$dir/chord"
        cp -f "$MEL_SRC"/*.mid "$dir/mel/"; cp -f "$CHORD_SRC"/*.mid "$dir/chord/"
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
        touch "$dir/.ready"
    fi
    echo "$dir"
}

# ---- 2. generate --------------------------------------------------------
SCORED=()
for name in $SYSTEMS; do
    read -r cp prog sched steps temp topp ckvar <<<"$(sys_cfg "$name")"
    ck="${!ckvar:-}"
    dir="$OUT_ROOT/$name"
    if [[ "$SKIP_INFER" == "1" ]]; then
        [[ -d "$dir" ]] && SCORED+=("$name") || echo "[$name] not generated; skipped"
        continue
    fi
    [[ -n "$ck" && -e "$ck" ]] || { echo "[$name] SKIPPED: set $ckvar to its checkpoint"; continue; }
    pdir=$(stage "$prog")
    [[ -d "$ck" ]] && ckarg="$ck/" || ckarg="$ck"
    echo "================================================================"
    echo "[$name] ckpt=$ck  cp=$cp  chord program=$prog  schedule=$sched"
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
        --min-chord-tokens-before-eos 0
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
for name in "${SCORED[@]}"; do SRC_ARGS+=(--source "$name:duet_multi:$OUT_ROOT/$name"); done
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
