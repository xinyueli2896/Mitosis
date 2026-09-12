#!/bin/bash
# SLURM (GPU): E1 co-generation evaluation, ONE JOB.
#
#   1. stages the held-aside POP909 test split as prompts
#   2. generates every system's outputs into temp/E1/<system>/
#      -- skipping any system whose folder already holds N_SAMPLES
#         co samples for every test song (FORCE regenerates)
#   3. scores everything (manifest -> metrics -> paired table)
#
# Systems (SYSTEMS knob; default all six):
#   A1    intra-cross-attn co-generation, NO query slots (control)
#   A3    the model: duet block diffusion, K=4 refinement, co mode
#   A3f   A3 + slot_sees_prev_frame (the mask fix; VARIANTS.md A.3f),
#         same decoder and schedule as A3; the flag travels in the ckpt
#   A3K0, A3ctc, A3ctcc, A3ctc2, A3ctca   decode variants on the A3 ckpt (not in
#         the default list): refinement depth 0; commit-then-condition
#         with melody leading / chord leading / both passes / alternating
#         leader (the symmetric one). Diagnostics
#         for the coupling deficit -- see the generate() case.
#   S1    POP909-finetuned single-stream cp transformer (merged
#         streams, chord tagged on program 48), continuation -- the
#         MATCHED single-stream baseline (LA pretrain + POP909 finetune)
#   S-scratch  same architecture and tagged POP909 data, trained from
#         RANDOM init -- the no-pretrain baseline
#   S0    the LA-pretrained single-stream model, same path, never
#         trained on POP909 -- unmatched anchor; not in the default
#         list, add S0 to SYSTEMS to include it
#   P-mc  cascade: S-mel continues melody -> POP909 YinYang (mel2chord,
#         the '_rev' ckpt) harmonizes.  Domain matched.
#   P-cm  cascade: S-chord continues chords -> POP909 YinYang
#         (chord2mel, non-rev) melodizes.  Domain matched.
#   WSf   WS with its own form stage instead of oracle key+phrase over
#         the generated region -- the matched, form-free protocol. Same
#         prompt, same cascade, same checkpoints; only the background of
#         the counterpoint stage changes. Song length stays oracle in
#         both. Run WS and WSf together to show the baseline at its
#         published best AND under the comparison our system faces.
#   WS    external reference: whole-song-gen PROMPTED lead-sheet
#         continuation, same 6-bar prompt, its own inpainting path.
#   AMT   external reference: the Anticipatory Music Transformer
#         (arXiv:2306.08620), prompted continuation of BOTH streams from
#         merged_tagged. The only external arm that CO-generates rather
#         than harmonising one stream given the other, and out of domain
#         (Lakh MIDI, its own tokenization). Needs setup_amt.sbatch once.
#         AMT_RESTRICT=0 runs the published sampler unmasked; the default
#         1 restricts note logits to the prompt's two instruments so the
#         output stays two streams -- say which a table row used.
#         AMT_MODEL picks the size (default music-medium-800k).
#
# Every system: same prompt (PROMPT_LENGTH frames), same total length
# (GEN_LENGTH), N_SAMPLES per song, scored on frames
# PROMPT_LENGTH..GEN_LENGTH only. The cascades and WS run through their
# own wrappers (pipeline_cogen.sbatch, infer_wholesong_prompted.sbatch)
# executed inside this allocation, so the E1 arm is byte-for-byte the
# standalone arm.
#
# Output tree (fixed, rerun-stable):
#   temp/E1/<system>/            the system's folder
#     A1, A3, WS:  <song>/co/sample_<i>[_temp<T>].mid   (duet_multi)
#     S1, S-scratch, S0:  <song>.mid_temp<T>_continuation_<i>.mid (single)
#     P-mc, P-cm:  1_stageA_specialist/ 2_hybrids/ 3_y_conditional/
#                  4_final/<song>/co/sample_<i>_temp<T>.mid  (scored)
#   temp/E1/prompts/{mel,chord,merged_tagged}/  staged test split
#   results/E1_p<PROMPT_LENGTH>[_<RESULT_TAG>]_{manifest.tsv,metrics.csv,
#                                               table.md,table_twosided.md}
#
# Skip rule: a system is DONE when every staged test song has at least
# N_SAMPLES samples in its scored layout. Done systems are not
# regenerated -- so a rerun that adds one system, or a job killed
# mid-way, only generates what is missing. FORCE="A1 P-mc" (or
# FORCE=all) deletes those folders under temp/E1 and regenerates them.
#
# WITHIN a system the generators are also resumable: every arm is
# invoked with --skip-existing, so a job that hits the wall part-way
# through a system keeps what it wrote and continues from there. The
# resumable unit is the sample for A1/A3 and the song for S1/S-scratch/
# S0 and WS/WSf (those draw all their samples in one call). The ONE
# exception is the cascades' stage B: yinyang_e3_driver's reuse cache is
# keyed too loosely to be safe (see its --reuse-cache help), so a P-mc
# or P-cm killed during stage B redoes stage B -- stage A is kept.
#
# Env knobs (--export=ALL,KEY=val):
#   SYSTEMS        default "A1 A3 A3f S1 S-scratch P-mc P-cm WS"; 'all' =
#                  every known name (use with SKIP_INFER=1 / score_e1 to
#                  rescore everything that is complete under OUT_ROOT)
#   FORCE          default empty; space-separated names, or all
#   CKPT_A1        default the melchord cp4tar intra-cross-attn run dir
#   CKPT_A3        default the A3 cp4tar run dir (K4mg_melchord_cp4tar)
#   CKPT_A3F       default the A3f run dir (A3f_melchord_long)
#   CKPT_A3C / CKPT_A3FC   conditional-slot run dirs (systems A3c / A3fc,
#                  decoded with A3_SCHEDULE=${A3C_SCHEDULE:-ctc_m}); no default
#   CKPT_S1        default the pop909ft_tagged_long merged run (cp16)
#   CKPT_SSCRATCH  default the pop909scratch_tagged merged run (cp16)
#   CKPT_S0        default the LA-pretrained cp transformer ckpt
#   CKPT_SMEL      P-mc stage A: default the pop909ft_mel_cp4 specialist
#   CKPT_SCHORD    P-cm stage A: default the pop909ft_chord_cp4 specialist
#   CKPT_Y_M2C     P-mc stage B: default the POP909 YinYang '_rev' (mel2chord)
#   CKPT_Y_C2M     P-cm stage B: default the POP909 YinYang non-rev (chord2mel)
#   MAX_POLYPHONY_DUET   cp of A1/A3/A3f training data (default 4, legacy)
#   MAX_POLYPHONY_CP8    cp of the retrained runs A1cp8/A3c/A3fc (default 8)
#   CKPT_A1CP8     causal duet retrained on cp8, long budget (system A1cp8)
#   CKPT_A10       same-frame-edge model (system A10, cp8); A10_DIRECTION
#                  alt|m|c|random picks the per-frame leader (default alt)
#   MAX_POLYPHONY_SPEC   cp of the cascade specialists (default 4)
#   MAX_POLYPHONY_S1 / MAX_POLYPHONY_SSCRATCH / MAX_POLYPHONY_S0
#                  cp of the merged single-stream models (default 16)
#   PROMPT_LENGTH  default 96 (6 bars).  GEN_LENGTH default 416 (total).
#   N_SAMPLES      default 3
#   BASELINE       system the paired table tests against (default A3)
#   MEL_FOLDER / CHORD_FOLDER   default input/pop909_split/{melody,chord}
#   OUT_ROOT       default temp/E1
#   RESULT_TAG     extra suffix on the result files
#   SKIP_INFER=1   score only, generate nothing (no GPU needed: submit
#                  through score_e1.sbatch, a CPU wrapper that forces it)
#   NUM_GPUS       systems that generate CONCURRENTLY, one per GPU,
#                  INSIDE this allocation. Detected from the allocation
#                  (#SBATCH --gres=gpu:4); lower it to leave a device
#                  free. This is the way to use several GPUs -- splitting
#                  SYSTEMS across several one-GPU jobs makes each of them
#                  queue separately. Lanes are STATIC round-robin over
#                  the systems still to do, so the makespan is the
#                  slowest lane: order SYSTEMS so the expensive ones do
#                  not land on the same lane (same index modulo
#                  NUM_GPUS). Per-system output goes to
#                  OUT_ROOT/genlogs/<system>.log, since four concurrent
#                  arms would otherwise interleave into one stream.
#   PREP_ONLY=1    stage the prompt folders under OUT_ROOT and exit (no
#                  GPU: submit through prep_e1.sbatch). Rarely needed now
#                  that one job drives every GPU; still useful to stage
#                  prompts on a CPU node before a GPU job starts, or to
#                  run disjoint SYSTEMS lists on separate NODES. A GPU
#                  job finding OUT_ROOT/prompts/.ready reuses the staged
#                  prompts instead of rebuilding them.
#   PIANO_COPIES   default 1: <system>_piano listening copies
#   MIN_SONGS      default 2: refuse to run on a smaller staged set (a
#                  1-song stage prints a full table with every +-0.000)
#   RESTAGE=1      delete OUT_ROOT/prompts (and its .ready) before
#                  staging, for when the source folder changed under a
#                  reused OUT_ROOT; pass it to the prep job only
#
# Examples:
#   sbatch midi_yinyang/eval_e1.sbatch          # all six, defaults
#   sbatch --export=ALL,SYSTEMS="A1 A3",FORCE=A1 midi_yinyang/eval_e1.sbatch
#   sbatch --export=ALL,SYSTEMS=A3f midi_yinyang/eval_e1.sbatch   # add one system to an existing tree
#   sbatch --export=ALL,SYSTEMS="A3K0 A3ctc A3ctc2" midi_yinyang/eval_e1.sbatch   # decode diagnostics
#   sbatch --export=ALL,SKIP_INFER=1,BASELINE=S1 midi_yinyang/eval_e1.sbatch

#SBATCH --job-name=eval_e1
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"

conda activate "$CONDA_ENV"
cd "$MY"
mkdir -p /home/xinyue.li/logs results

# ---- knobs ------------------------------------------------------------
SYSTEMS="${SYSTEMS:-A1 A3 A3f S1 S-scratch P-mc P-cm WS}"
# SYSTEMS=all: every system name this script knows. Meant for the
# score-only path (score_e1.sbatch), where it scores whatever is complete
# under OUT_ROOT and skips the rest -- a rescore that names only the
# default eight OVERWRITES the metrics CSV without the decode variants
# (happened 2026-09-10, job 225780). With generation on, 'all' would try
# to generate every system, so keep it to score-only runs.
if [[ "$SYSTEMS" == "all" ]]; then
    SYSTEMS="A11aT A11K4 A1 A1cp8 A3 A3f A3fctcaT A3K0 A3ctc A3ctcc A3ctc2 A3ctca A3ctcaT A3c A3fc A3fcaT A3fcaTar A3fcK4 A10 S1 S-scratch S0 P-mc P-cm WS WSf AMT"
fi
FORCE="${FORCE:-}"
CKPT_A1="${CKPT_A1:-ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_melchord_cp4tar_batch_8_schedule}"
CKPT_A3="${CKPT_A3:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4mg_melchord_cp4tar_batch_8_schedule}"
# A3f: the v1.2 'long' run (A3 + slot_sees_prev_frame; the flag is
# restored from the ckpt, so the decoder call is A3's).
CKPT_A3F="${CKPT_A3F:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A3f_melchord_long_batch_8_schedule}"
# A3c / A3fc: conditional-slot runs (train_duet_block_diffusion COND_SLOT_PROB);
# no default until one is trained -- pass the run dir.
CKPT_A3C="${CKPT_A3C:-}"
CKPT_A3FC="${CKPT_A3FC:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A3fc_melchord_cp8long_batch_8_schedule}"   # 2026-09-10 retrain (cp8, COND_SLOT_PROB=0.8); best_ar/ subdir = AR-selected weights
# A1cp8: the causal duet retrained on cp8 at the long budget; no default.
CKPT_A1CP8="${CKPT_A1CP8:-}"
# A10: the same-frame-edge model (train_intra_cross_attn SAME_FRAME=1); no default.
CKPT_A11="${CKPT_A11:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A11_melchord_cp8long_ag_batch_8_schedule}"   # A.11 agreement head; decodes exactly like A3fc
CKPT_A10="${CKPT_A10:-ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_sf_melchord_cp8long_batch_8_schedule}"   # 2026-09-10 A.10 same-frame (cp8)
export A10_DIRECTION="${A10_DIRECTION:-alt}"
# Merged single-stream baselines: 'tagged' = chord on program 48 in the
# training merge, which is what lets the tokenizer keep the streams
# apart (an untagged merge fuses them irreversibly).
CKPT_S1="${CKPT_S1:-ckpt/cp_transformer_v0.42_size1_pop909ft_tagged_long_batch_36_schedule}"
CKPT_SSCRATCH="${CKPT_SSCRATCH:-ckpt/cp_transformer_v0.42_size1_pop909scratch_tagged_batch_12_schedule}"
CKPT_S0="${CKPT_S0:-ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt}"
# Cascade stage A: the cp4 POP909 specialists (cp4 is in the dir name,
# hence MAX_POLYPHONY_SPEC=4 below). Stage B: the POP909-trained
# YinYang conditionals -- direction is the ckpt's own suffix: '_rev' is
# mel2chord (P-mc), non-rev is chord2mel (P-cm). YinYang restores its
# own cp from the ckpt, so no cp flag reaches stage B.
CKPT_SMEL="${CKPT_SMEL:-ckpt/cp_transformer_v0.42_size1_pop909ft_mel_cp4_batch_24_schedule}"
CKPT_SCHORD="${CKPT_SCHORD:-ckpt/cp_transformer_v0.42_size1_pop909ft_chord_cp4_batch_24_schedule}"
CKPT_Y_M2C="${CKPT_Y_M2C:-ckpt/cp_transformer_yinyang_v5.1_lora_batch_8_pop909_cp8_v2_chord_mel_rev_mask0.25-10-step1}"
CKPT_Y_C2M="${CKPT_Y_C2M:-ckpt/cp_transformer_yinyang_v5.1_lora_batch_8_pop909_cp8_v2_chord_mel_mask0.25-10-step1}"
MAX_POLYPHONY_DUET="${MAX_POLYPHONY_DUET:-4}"
# cp of the retrained duet runs (A1cp8, A3c, A3fc): the certified cp8
# data, full seventh chords. The legacy cp4tar runs (A1, A3, A3f) keep
# MAX_POLYPHONY_DUET=4 -- cp4 dropped the seventh from 13.8% of chord
# frames (tasks.py), a handicap on every H2 row.
MAX_POLYPHONY_CP8="${MAX_POLYPHONY_CP8:-8}"
MAX_POLYPHONY_SPEC="${MAX_POLYPHONY_SPEC:-4}"
MAX_POLYPHONY_S1="${MAX_POLYPHONY_S1:-16}"
MAX_POLYPHONY_SSCRATCH="${MAX_POLYPHONY_SSCRATCH:-16}"
MAX_POLYPHONY_S0="${MAX_POLYPHONY_S0:-16}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
BASELINE="${BASELINE:-A3}"
MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
# CROPS_TSV: build_prompt_crops' prompt_crops.tsv for the test set. Our
# systems read the CROPPED midis, so their prompt is bar 0 of the crop
# by construction. The whole-song baseline does not read our midis at
# all -- it loads the song by POP909 index from its own dataset -- so
# without this it prompts from the ORIGINAL song's bar 0: a different
# prompt from every other row of the table. Default: the TSV beside
# MEL_FOLDER when there is one (a build_prompt_crops set), else none
# (the legacy pop909_split set, which was never cropped).
CROPS_TSV="${CROPS_TSV:-$(dirname "$MEL_FOLDER")/prompt_crops.tsv}"
[[ -f "$CROPS_TSV" ]] || CROPS_TSV=""
# DUET_CHORD_PROGRAM: the midi program the staged CHORD prompts carry,
# which is a TOKEN the duet models read (a frame is (program, pitch-dur)
# pairs, so roughly half of a chord frame's content tokens are it).
# Default 0, because EVERY duet checkpoint that exists today was
# finetuned on a corpus where POP909's MELODY track and the chord
# builder both left the program at 0 -- while POP909-v5's aligner writes
# 48, so an untouched v5 prompt is off-distribution in half the content
# of every chord frame. The single-stream baselines are unaffected
# either way: merged_tagged below force-tags the chord to 48 itself.
# Set 48 to evaluate a checkpoint trained on a tagged corpus, or empty
# to leave the source untouched. See EXPERIMENTS.md E4d.
DUET_CHORD_PROGRAM="${DUET_CHORD_PROGRAM-0}"
# NUM_GPUS: how many systems generate CONCURRENTLY, one per GPU, inside
# this allocation. Detected from the allocation; override to run fewer
# lanes than the GPUs held (e.g. to leave a device free).
# Counted from the allocation itself, never by probing the device --
# the cluster's submit filter rejects scripts carrying nvidia-smi and
# friends as idle/monitoring commands.
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "$NUM_GPUS" && -n "${SLURM_JOB_GPUS:-}" ]]; then
    NUM_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -c .)
fi
if [[ -z "$NUM_GPUS" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
[[ "$NUM_GPUS" -ge 1 ]] 2>/dev/null || NUM_GPUS=1
OUT_ROOT="${OUT_ROOT:-temp/E1}"
RESULT_TAG="${RESULT_TAG:-}"
SKIP_INFER="${SKIP_INFER:-0}"
PREP_ONLY="${PREP_ONLY:-0}"
PIANO_COPIES="${PIANO_COPIES:-1}"
MIN_SONGS="${MIN_SONGS:-2}"
RESTAGE="${RESTAGE:-0}"

# A3's decode schedule (the listening-test winner); identical to
# queryablation so E1's A3 row and the ablation's A3 row are the same
# decoder.
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-0.9}"
export A3_TOP_P="${A3_TOP_P:-0.95}"

[[ "$OUT_ROOT" == temp/* ]] || { echo "ERROR: OUT_ROOT must be under temp/ (got $OUT_ROOT)"; exit 1; }
mkdir -p "$OUT_ROOT"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "E1 CO-GENERATION EVALUATION"
echo "SYSTEMS=$SYSTEMS   FORCE=${FORCE:-<none>}   SKIP_INFER=$SKIP_INFER"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)  GEN_LENGTH=$GEN_LENGTH  N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT=$OUT_ROOT   BASELINE=$BASELINE"
echo "NUM_GPUS=$NUM_GPUS (systems generate concurrently, one per GPU)"
echo "DUET_CHORD_PROGRAM=${DUET_CHORD_PROGRAM:-<source untouched>} (merged_tagged is always 48)"
echo "MEL_FOLDER=$MEL_FOLDER   CROPS_TSV=${CROPS_TSV:-<none: WS/WSf prompt from the original bar 0>}"
echo "A3 decode: REFINE=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P"
echo "cp: duet=$MAX_POLYPHONY_DUET  specialists=$MAX_POLYPHONY_SPEC  S1/S0=$MAX_POLYPHONY_S1/$MAX_POLYPHONY_S0"
echo "================================================================"
# (no nvidia-smi: the submit filter rejects it. The cuda pre-flight
# below already names the devices through torch.)

# ---- 0. pre-flight ----------------------------------------------------
if [[ "$SKIP_INFER" != "1" && "$PREP_ONLY" != "1" ]]; then
    python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"
fi
if [[ "$SKIP_INFER" != "1" && "$PREP_ONLY" != "1" ]]; then
    for _c in "$CKPT_A1" "$CKPT_A3" "$CKPT_A3F" "$CKPT_S1" "$CKPT_SSCRATCH" "$CKPT_S0" "$CKPT_SMEL" "$CKPT_SCHORD" "$CKPT_Y_M2C" "$CKPT_Y_C2M"; do
        [[ -n "$_c" && -e "$_c" ]] && python check_ckpts.py "$_c" || true
    done
fi

# ---- 1. prompts: the test split, staged once -------------------------
# The same full midis serve as prompts (first PROMPT_LENGTH frames) and
# as scoring references (frames PROMPT_LENGTH..GEN_LENGTH).
EVAL_MEL="$OUT_ROOT/prompts/mel"
EVAL_CHORD="$OUT_ROOT/prompts/chord"
EVAL_MERGED="$OUT_ROOT/prompts/merged_tagged"
READY="$OUT_ROOT/prompts/.ready"
if [[ "$RESTAGE" == "1" && "$SKIP_INFER" != "1" ]]; then
    echo "[prompts] RESTAGE=1: removing $OUT_ROOT/prompts and staging afresh from $MEL_FOLDER"
    rm -rf "$OUT_ROOT/prompts"
fi
if [[ -f "$READY" ]]; then
    echo "[prompts] reusing staged prompts ($READY present; delete it to restage)"
else
    mkdir -p "$EVAL_MEL" "$EVAL_CHORD"
    cp -f "$MEL_FOLDER"/*.mid "$MEL_FOLDER"/*.MID "$EVAL_MEL"/ 2>/dev/null || true
    cp -f "$CHORD_FOLDER"/*.mid "$CHORD_FOLDER"/*.MID "$EVAL_CHORD"/ 2>/dev/null || true
    if [[ -n "$DUET_CHORD_PROGRAM" ]]; then
        python - "$EVAL_CHORD" "$DUET_CHORD_PROGRAM" <<'RETAG'
import glob, os, sys
import mido
folder, prog = sys.argv[1], int(sys.argv[2])
changed = seen = 0
for f in sorted(glob.glob(os.path.join(folder, '*.mid'))):
    m = mido.MidiFile(f)
    hit = False
    for tr in m.tracks:
        for msg in tr:
            if msg.type == 'program_change':
                if msg.program != prog:
                    msg.program = prog
                    hit = True
                seen += 1
    # a track with no program_change plays on the channel default (0);
    # only worth inserting one when we want something else
    if not seen and prog != 0:
        m.tracks[-1].insert(0, mido.Message('program_change', program=prog,
                                            time=0))
        hit = True
    if hit:
        m.save(f)
        changed += 1
print(f'[prompts] chord program -> {prog}: rewrote {changed} file(s)')
RETAG
    fi
fi
N_EVAL=$(ls "$EVAL_MEL" | wc -l)
[[ "$N_EVAL" -gt 0 ]] || { echo "ERROR: no midis in $MEL_FOLDER"; exit 1; }
# A one-song stage still produces a full-looking table (every +-0.000,
# job 217364). Refuse below MIN_SONGS unless the caller lowers it.
[[ "$N_EVAL" -ge "$MIN_SONGS" ]] || {
    echo "ERROR: only $N_EVAL song(s) staged under $EVAL_MEL (MIN_SONGS=$MIN_SONGS);"
    echo "       the staging audit dropped the rest -- see its [prompt] block. Delete"
    echo "       $READY and restage from a full MEL_FOLDER, or pass MIN_SONGS=1 on purpose."
    exit 1; }
[[ "$N_EVAL" -eq "$(ls "$EVAL_CHORD" | wc -l)" ]] || {
    echo "ERROR: melody/chord split sizes differ; scoring pairs by filename"; exit 1; }
SONG_IDS=$(ls "$EVAL_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | sort | tr '\n' ' ')
echo "[prompts] $N_EVAL test songs: $SONG_IDS"
# merged, program-tagged copies for the single-stream S0 (the tokenizer
# separates streams by program only)
if [[ ! -f "$READY" ]]; then
    python merge_melody_chord.py --melody "$EVAL_MEL" --chord "$EVAL_CHORD" \
        --dst "$EVAL_MERGED" --chord-program 48
    date > "$READY"
fi
if [[ "$PREP_ONLY" == "1" ]]; then
    echo "[prompts] PREP_ONLY=1: staged $N_EVAL songs under $OUT_ROOT/prompts; exiting"
    exit 0
fi

# ---- system registry --------------------------------------------------
# folder under OUT_ROOT, the scored root inside it, and its layout.
sys_dir()    { echo "$OUT_ROOT/$1"; }
sys_root()   { case "$1" in P-mc|P-cm) echo "$OUT_ROOT/$1/4_final" ;; *) echo "$OUT_ROOT/$1" ;; esac; }
sys_layout() { case "$1" in S0|S1|S-scratch) echo single ;; *) echo duet_multi ;; esac; }

# DONE = every test song has >= N_SAMPLES samples in the scored layout.
is_complete() {
    local root layout sid n
    root=$(sys_root "$1"); layout=$(sys_layout "$1")
    [[ -d "$root" ]] || return 1
    for sid in $SONG_IDS; do
        case "$layout" in
            duet_multi) n=$(ls "$root/$sid/co/"sample_*.mid 2>/dev/null | wc -l) ;;
            single)     n=$(ls "$root/$sid".mid_temp*_continuation_*.mid 2>/dev/null | wc -l) ;;
        esac
        [[ "$n" -ge "$N_SAMPLES" ]] || return 1
    done
    return 0
}

is_forced() {
    [[ "$FORCE" == "all" ]] && return 0
    local f; for f in $FORCE; do [[ "$f" == "$1" ]] && return 0; done
    return 1
}

# ---- 2. generation ----------------------------------------------------
generate() {
    local name="$1" dir; dir=$(sys_dir "$name")
    case "$name" in
    A1|A1cp8|A10)
        # A1cp8: the causal duet retrained on the cp8 data (full seventh
        # chords) at the long budget -- the backbone control for the
        # cp4/budget handicap of the legacy cp4tar runs.
        local ck cp
        case "$name" in
            A1)    ck="$CKPT_A1";    cp="$MAX_POLYPHONY_DUET" ;;
            A1cp8) ck="$CKPT_A1CP8"; cp="$MAX_POLYPHONY_CP8" ;;
            A10)   ck="$CKPT_A10";   cp="$MAX_POLYPHONY_CP8" ;;   # same-frame edge; decode alternates (A10_DIRECTION)
        esac
        [[ -n "$ck" ]] || { echo "[$name] SKIPPED: checkpoint knob unset"; return 0; }
        python cp_transformer_m2c_intra_cross_attn_combined.py \
            --ckpt "$ck/" \
            --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
            --output-dir "$dir" --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$cp" --skip-existing \
            --model-size large --moe-num-experts 4 --moe-topk 2 ;;
    A3|A3f|A3fctcaT|A3K0|A3ctc|A3ctcc|A3ctc2|A3ctca|A3ctcaT|A3c|A3fc|A3fcaT|A3fcK4|A3fcaTar|A11aT|A11K4)
        # Decode variants share the A3 checkpoint and differ only in the
        # env the decoder reads (A3_REFINE_STEPS / A3_SCHEDULE):
        #   A3K0    refinement depth 0: the AR head's draft is the output
        #   A3ctc   commit-then-condition, melody leads (ctc_m)
        #   A3ctcc  commit-then-condition, chord leads (ctc_c)
        #   A3ctc2  two passes: chord | melody, then melody | chord (ctc2_m)
        #   A3c     conditional-slot ckpt (COND_SLOT_PROB), decoded ctc_m
        #   A3fc    same on the A3f mask (diagnostic decode, A3C_SCHEDULE)
        #   A3fcaT  the A3fc ckpt with the PAPER decode (ctc_alt, T=1, top-p 1)
        #   A3fcK4  the A3fc ckpt with refine K=4 (does conditional-slot
        #           training repair refinement's coupling?)
        #   A3fcaTar  A3fcaT on the AR-selected weights (<run>/best_ar/)
        local ck cp="$MAX_POLYPHONY_DUET"
        #   A3fctcaT  the A3f ckpt (slot sees t-1) with the paper decode
        case "$name" in
            A3f|A3fctcaT)  ck="$CKPT_A3F" ;;
            A3c)  ck="$CKPT_A3C";  cp="$MAX_POLYPHONY_CP8" ;;
            A3fc|A3fcaT|A3fcK4) ck="$CKPT_A3FC"; cp="$MAX_POLYPHONY_CP8" ;;
            A3fcaTar) ck="$CKPT_A3FC/best_ar"; cp="$MAX_POLYPHONY_CP8" ;;
            A11aT|A11K4) ck="$CKPT_A11"; cp="$MAX_POLYPHONY_CP8" ;;
            *)    ck="$CKPT_A3" ;;
        esac
        [[ -n "$ck" ]] || { echo "[$name] SKIPPED: checkpoint knob unset"; return 0; }
        [[ -e "$ck" ]] || { echo "[$name] SKIPPED: $ck does not exist (yet)"; return 0; }
        local -a denv=()
        case "$name" in
            A3K0)     denv=(A3_REFINE_STEPS=0) ;;
            A3ctc)    denv=(A3_SCHEDULE=ctc_m) ;;
            A3ctcc)   denv=(A3_SCHEDULE=ctc_c) ;;
            A3ctc2)   denv=(A3_SCHEDULE=ctc2_m) ;;
            A3ctca)   denv=(A3_SCHEDULE=ctc_alt) ;;   # leader alternates per frame: symmetric in expectation
            A3ctcaT)  denv=(A3_SCHEDULE=ctc_alt A3_FINAL_TEMP=1.0 A3_TOP_P=1.0) ;;   # same, follower sampled at T=1, no nucleus cut (repetition check)
            A3c|A3fc) denv=(A3_SCHEDULE="${A3C_SCHEDULE:-ctc_m}") ;;
            A3fcaT|A3fcaTar|A3fctcaT|A11aT) denv=(A3_SCHEDULE=ctc_alt A3_FINAL_TEMP=1.0 A3_TOP_P=1.0) ;;
            A11K4)    denv=(A3_SCHEDULE=refine) ;;
            A3fcK4)   denv=(A3_SCHEDULE=refine) ;;
        esac
        env ${denv[@]+"${denv[@]}"} python cp_transformer_m2c_duet_block_diffusion_combined.py \
            --ckpt "$ck/" \
            --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
            --output-dir "$dir" --modes co \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$cp" --skip-existing \
            --model-size large --moe-num-experts 4 --moe-topk 2 \
            --min-chord-tokens-before-eos 0 ;;
    S0|S1|S-scratch)
        local ck cp
        case "$name" in
            S1)        ck="$CKPT_S1";       cp="$MAX_POLYPHONY_S1" ;;
            S-scratch) ck="$CKPT_SSCRATCH"; cp="$MAX_POLYPHONY_SSCRATCH" ;;
            S0)        ck="$CKPT_S0";       cp="$MAX_POLYPHONY_S0" ;;
        esac
        [[ -n "$ck" ]] || { echo "[$name] SKIPPED: checkpoint knob unset"; return 0; }
        # cp_transformer_inference writes temp/<save-name>/ itself.
        python cp_transformer_inference.py \
            --ckpt "$ck" \
            --midi-folder "$EVAL_MERGED" \
            --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
            --temperature 1.0 --n-samples "$N_SAMPLES" \
            --max-polyphony "$cp" --skip-existing \
            --save-name "${dir#temp/}" ;;
    P-mc|P-cm)
        local spec y
        if [[ "$name" == "P-mc" ]]; then spec="$CKPT_SMEL"; y="$CKPT_Y_M2C"
        else spec="$CKPT_SCHORD"; y="$CKPT_Y_C2M"; fi
        [[ -n "$spec" && -n "$y" ]] || {
            echo "[$name] SKIPPED: needs $([[ "$name" == P-mc ]] && echo CKPT_SMEL and CKPT_Y_M2C || echo CKPT_SCHORD and CKPT_Y_C2M)"
            return 0; }
        # The standalone cascade wrapper, run inside this allocation with
        # E1's prompts and output folder -- so the E1 arm IS the
        # standalone arm. (#SBATCH lines are ignored when run this way;
        # the GPU is this job's.)
        PIPELINE="$name" CKPT_SPEC="$spec" CKPT_Y="$y" STAGE_B=y \
        OUT_ROOT="$dir" MEL_FOLDER="$EVAL_MEL" CHORD_FOLDER="$EVAL_CHORD" \
        PROMPT_LENGTH="$PROMPT_LENGTH" GEN_LENGTH="$GEN_LENGTH" \
        N_SAMPLES="$N_SAMPLES" MAX_POLYPHONY="$MAX_POLYPHONY_SPEC" \
            bash "$MY/pipeline_cogen.sbatch" ;;
    AMT)
        # External SOTA: the Anticipatory Music Transformer (Thickstun et
        # al., arXiv:2306.08620), prompted continuation of BOTH streams.
        # It reads merged_tagged -- the same file S1 reads -- because its
        # tokenizer carries the instrument in the note token, which is
        # what makes the output splittable back into two streams.
        # Anticipation controls are NOT used: controls would make this
        # conditional generation, which is E3's question, not E1's.
        [[ -d "$MY/external/anticipation/.venv" ]] || {
            echo "[AMT] SKIPPED: run setup_amt.sbatch first"; return 0; }
        MERGED_FOLDER="$MY/$EVAL_MERGED" OUT_DIR="$MY/$dir" \
        PROMPT_FRAMES="$PROMPT_LENGTH" GEN_FRAMES="$GEN_LENGTH" \
        N_SAMPLES="$N_SAMPLES" RESTRICT="${AMT_RESTRICT:-1}" \
        MODEL="${AMT_MODEL:-stanford-crfm/music-medium-800k}" \
            bash "$MY/infer_amt_prompted.sbatch" ;;
    WS|WSf)
        # whole-song-gen wants integer ids (its own POP909 index) and an
        # absolute OUT_DIR (it cds into its venv tree). It writes
        # <OUT_DIR>/<id zero-padded to 3>/co/sample_<i>.mid, which is
        # the split's naming.
        local ids; ids=$(for s in $SONG_IDS; do echo $((10#$s)); done | tr '\n' ' ')
        # WS_FORM selects the baseline's protocol: gt = oracle key+phrase
        # over the generated region (their published form-conditioned
        # setting, in the baseline's favour), generated = its own form
        # stage prompted on the same bars (the matched comparison).
        # System name WSf pins `generated` so both can sit in one table.
        local ws_form="${WS_FORM:-gt}"
        [[ "$name" == WSf ]] && ws_form=generated
        # CROPS_TSV absolute: the wrapper cds into the baseline's tree.
        # Without it the baseline is prompted from the song's own bar 0
        # -- see the knob comment -- which is a different prompt from
        # every other system's and cannot go in the same table.
        local crops=""
        [[ -n "$CROPS_TSV" ]] && crops="$(readlink -f "$CROPS_TSV")"
        OUT_DIR="$MY/$dir" SONG_IDS="$ids" N_SAMPLES="$N_SAMPLES" \
        PROMPT_BARS="$((PROMPT_LENGTH / 16))" WS_FORM="$ws_form" \
        CROPS_TSV="$crops" \
            bash "$MY/infer_wholesong_prompted.sbatch" ;;
    *)  echo "ERROR: unknown system $name"; exit 1 ;;
    esac
}

# ---- 2a. decide what needs generating (sequential, cheap) -------------
SCORED=()
TODO=()
for name in $SYSTEMS; do
    dir=$(sys_dir "$name")
    echo "----------------------------------------------------------------"
    if [[ "$SKIP_INFER" == "1" ]]; then
        is_complete "$name" && { echo "[$name] score-only: complete"; SCORED+=("$name"); } \
                            || echo "[$name] score-only: INCOMPLETE or missing -- not scored"
        continue
    fi
    if is_forced "$name" && [[ -d "$dir" ]]; then
        echo "[$name] FORCE: removing $dir"
        rm -rf "$dir"
    fi
    if is_complete "$name"; then
        echo "[$name] SKIP: $(sys_root "$name") already has >= $N_SAMPLES co samples for all $N_EVAL songs"
        SCORED+=("$name"); continue
    fi
    TODO+=("$name")
done

# ---- 2b. generate, one system per GPU, inside THIS allocation ---------
# Systems are dealt round-robin to the visible GPUs and each lane runs
# its share sequentially, so a 4-GPU allocation does four systems at a
# time without submitting four jobs that each have to queue. Every arm
# works unchanged: the lane exports CUDA_VISIBLE_DEVICES and the
# generators all call .cuda(), which then means that lane's device.
#
# Lanes are STATIC round-robin, not a work queue, so the makespan is set
# by the slowest lane -- order SYSTEMS to balance it (put the expensive
# ones first, and avoid landing two of them on the same lane modulo
# NUM_GPUS).
if [[ "$SKIP_INFER" != "1" && ${#TODO[@]} -gt 0 ]]; then
    GENLOG="$OUT_ROOT/genlogs"
    mkdir -p "$GENLOG"
    N_LANES=$NUM_GPUS
    [[ $N_LANES -gt ${#TODO[@]} ]] && N_LANES=${#TODO[@]}
    echo "================================================================"
    echo "GENERATING ${#TODO[@]} system(s) on $N_LANES GPU lane(s): ${TODO[*]}"
    echo "per-system logs under $GENLOG/  (this log keeps only the summary)"
    echo "================================================================"
    lane_pids=()
    for lane in $(seq 0 $((N_LANES - 1))); do
        (
            # EXPORT, not a command prefix: the arms reach the GPU through
            # child python processes (and, for the cascades and WS, through
            # a nested bash), so the variable has to be in the environment
            # the subshell hands down, not a shell-local assignment.
            export CUDA_VISIBLE_DEVICES="$lane"
            for idx in "${!TODO[@]}"; do
                [[ $(( idx % N_LANES )) -eq $lane ]] || continue
                nm="${TODO[$idx]}"
                t0=$(date +%s)
                echo "[lane $lane / gpu $lane] $nm: start $(date +%T)"
                if generate "$nm" > "$GENLOG/$nm.log" 2>&1; then
                    echo "[lane $lane / gpu $lane] $nm: exit 0 after $(( $(date +%s) - t0 ))s"
                else
                    echo "[lane $lane / gpu $lane] $nm: FAILED after $(( $(date +%s) - t0 ))s -- see $GENLOG/$nm.log"
                fi
            done
        ) &
        lane_pids+=($!)
    done
    for pid in "${lane_pids[@]}"; do wait "$pid" || true; done
    echo "================================================================"
    echo "generation lanes finished; completeness below"
    for name in "${TODO[@]}"; do
        if is_complete "$name"; then
            echo "[$name] complete"
            SCORED+=("$name")
        else
            echo "[$name] WARNING: incomplete (missing songs or samples); not scored"
            echo "         why: $GENLOG/$name.log"
        fi
    done
fi

# ---- 2b. piano listening copies (scored files keep chord on prog 48) --
if [[ "$PIANO_COPIES" == "1" && ${#SCORED[@]} -gt 0 && "$SKIP_INFER" != "1" ]]; then
    dirs=(); for name in "${SCORED[@]}"; do dirs+=("$(sys_root "$name")"); done
    python render_piano_copies.py "${dirs[@]}" || echo "[piano] copies failed (non-fatal)"
fi

# ---- 3. scoring -------------------------------------------------------
[[ ${#SCORED[@]} -gt 0 ]] || { echo "ERROR: nothing to score"; exit 1; }
RSUF="p${PROMPT_LENGTH}${RESULT_TAG:+_${RESULT_TAG}}"
MANIFEST="results/E1_${RSUF}_manifest.tsv"
METRICS="results/E1_${RSUF}_metrics.csv"
TABLE="results/E1_${RSUF}_table.md"

SRC_ARGS=()
for name in "${SCORED[@]}"; do
    SRC_ARGS+=(--source "$name:$(sys_layout "$name"):$(sys_root "$name")")
done
echo "================================================================"
echo "[sources] ${SCORED[*]}"
python build_eval_manifest.py "${SRC_ARGS[@]}" --songs $SONG_IDS \
    --modes co --out "$MANIFEST"

echo "--- scoring (H3 > H2 > H1 > S) ---"
python eval_metrics.py --task melchord \
    --manifest "$MANIFEST" \
    --ref-a-dir "$EVAL_MEL" --ref-b-dir "$EVAL_CHORD" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

if [[ " ${SCORED[*]} " != *" $BASELINE "* ]]; then
    echo "WARNING: BASELINE=$BASELINE was not scored; using ${SCORED[0]} for the paired table"
    BASELINE="${SCORED[0]}"
fi
echo "--- comparison table (paired Wilcoxon vs $BASELINE) ---"
python aggregate_eval_results.py --csv "$METRICS" \
    --task melchord --baseline "$BASELINE" --mode co \
    --one-sided --complete-cases --out-md "$TABLE"
python aggregate_eval_results.py --csv "$METRICS" \
    --task melchord --baseline "$BASELINE" --mode co \
    --out-md "${TABLE%.md}_twosided.md"

echo "================================================================"
echo "E1 complete.  systems: ${SCORED[*]}"
echo "  outputs : $OUT_ROOT/<system>/"
echo "  table   : $TABLE   (one-sided, complete cases, vs $BASELINE)"
echo "  2-sided : ${TABLE%.md}_twosided.md"
echo "  metrics : $METRICS"
echo "================================================================"
echo "RESULT TABLE:"
cat "$TABLE"
