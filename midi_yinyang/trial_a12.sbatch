#!/bin/bash
# SLURM (1 GPU): trial decode of an A.12 checkpoint on a few held-out
# v5 songs, prompts TAGGED the way the training corpus was.
#
# Why a separate wrapper and not eval_e1: the E1 tree stages its chord
# prompts with the program rewritten to 0 (DUET_CHORD_PROGRAM's
# default), because every duet checkpoint before the v5 runs was
# finetuned on an untagged corpus. The A.12 run trained on the v5
# corpus, whose aligner writes the chord track at program 48 -- and the
# program is a TOKEN the duet reads, roughly half of every chord
# frame's content tokens. A prompt at program 0 would be off-
# distribution in half the content of every chord frame. This wrapper
# stages the songs from input/heldout_v5 and pins the chord program to
# CHORD_PROGRAM (default 48), then prints the program histogram of the
# staged files so the log shows the tag, not assumes it.
#
# Decode: the A.12 draft-only configuration (DIFFUSION_K=1, MASK_K_PROB=0,
# SYM_K=1, SC_VAL=1) has one state -- every slot holds the AR draft --
# and its k tags match only A3_SCHEDULE=refine A3_REFINE_STEPS=1: a
# masked seed round at k=1 read from the clean AR rows, one draft round
# at k=0. Those are the defaults below; A3_SCHEDULE=ctc_alt is there for
# a checkpoint trained with the commit-then-condition states.
#
# MAX_POLYPHONY CANNOT BE READ OFF THE CHECKPOINT (CLAUDE.md). Default 8,
# the budget the v5 corpus is built at; confirm against the training
# log's "Data for dataset" line, which prints the .pt path with its cp.
# A wrong value loads cleanly and decodes against the wrong tokenization.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT           run directory (smallest val_loss ckpt is picked) or a
#                  ckpt file. Default the A12K1 v5 draft_tagged run.
#                  For a run validated on the draft state (SC_VAL=1),
#                  val_loss is the draft-state loss; CKPT=<run>/best_ar
#                  selects on val_ar_loss_content instead.
#   SONG_IDS       default "126 596 606 696 886 003 726". Numeric ids are
#                  zero-padded to POP909's three digits; a named tune
#                  (ashover_5, from build_nottingham_split) is used as is.
#   MEL_SRC / CHORD_SRC   default input/heldout_v5/{melody,chord}
#   OUT_DIR        default temp/trial_a12_<jobid>
#   CHORD_PROGRAM  program pinned on the staged chord prompts, default 48
#                  (the v5 aligner's tag). Empty leaves the source as is.
#   PROMPT_LENGTH  default 80 (5 bars)     GEN_LENGTH default 416
#   N_SAMPLES      default 3               MAX_POLYPHONY default 8
#   A3_SCHEDULE    default refine          A3_REFINE_STEPS default 1
#   A3_FINAL_TEMP  default 1.0             A3_TOP_P default 1.0
#   MODE_NAME      default co
#
# Example:
#   sbatch --exclude=gpu-51,gpu-05,gpu-54 midi_yinyang/trial_a12.sbatch
#   sbatch --exclude=gpu-51,gpu-05,gpu-54 --export=ALL,CKPT=ckpt/<run>/best_ar \
#       midi_yinyang/trial_a12.sbatch
#
# Output: <OUT_DIR>/<song>/co/sample_<i>_temp1.0.mid plus the staged
# prompts under <OUT_DIR>/prompts/{mel,chord}, so check_prompt_copy can
# be pointed at the tree with MEL_DIR/CHORD_DIR for a prompt-copy check.

#SBATCH --job-name=trial_a12
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=04:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

CKPT="${CKPT:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A12K1_melchord_v5_draft_tagged_batch_8_schedule}"
SONG_IDS="${SONG_IDS:-126 596 606 696 886 003 726}"
MEL_SRC="${MEL_SRC:-input/heldout_v5/melody}"
CHORD_SRC="${CHORD_SRC:-input/heldout_v5/chord}"
OUT_DIR="${OUT_DIR:-temp/trial_a12_${SLURM_JOB_ID:-local}}"
CHORD_PROGRAM="${CHORD_PROGRAM-48}"
PROMPT_LENGTH="${PROMPT_LENGTH:-80}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
MAX_POLYPHONY="${MAX_POLYPHONY:-8}"
A3_SCHEDULE="${A3_SCHEDULE:-refine}"
A3_REFINE_STEPS="${A3_REFINE_STEPS:-1}"
A3_FINAL_TEMP="${A3_FINAL_TEMP:-1.0}"
A3_TOP_P="${A3_TOP_P:-1.0}"
MODE_NAME="${MODE_NAME:-co}"
# task-derived, as every duet wrapper: selects the dataset tokenization
export MAX_POLYPHONY MELCHORD_CP="$MAX_POLYPHONY"

[[ -e "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT"; exit 1; }
[[ -d "$MEL_SRC" && -d "$CHORD_SRC" ]] || {
    echo "ERROR: held-out folders missing: $MEL_SRC / $CHORD_SRC"; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "CKPT=$CKPT"
echo "SONGS=$SONG_IDS"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)  GEN_LENGTH=$GEN_LENGTH  N_SAMPLES=$N_SAMPLES"
echo "MAX_POLYPHONY=$MAX_POLYPHONY  (confirm against the training log's 'Data for dataset' line)"
echo "CHORD_PROGRAM=${CHORD_PROGRAM:-<untouched>}"
echo "A3_SCHEDULE=$A3_SCHEDULE  A3_REFINE_STEPS=$A3_REFINE_STEPS  A3_FINAL_TEMP=$A3_FINAL_TEMP  A3_TOP_P=$A3_TOP_P"
echo "OUT_DIR=$OUT_DIR"
echo "================================================================"

python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"
[[ -d "$CKPT" ]] && python check_ckpts.py "$CKPT" || true

# ---- 1. stage the songs, chord tagged ---------------------------------
EVAL_MEL="$OUT_DIR/prompts/mel"
EVAL_CHORD="$OUT_DIR/prompts/chord"
mkdir -p "$EVAL_MEL" "$EVAL_CHORD"
for s in $SONG_IDS; do
    # POP909 ids are zero-padded to 3; a named tune (ashover_5) is used as is
    if [[ "$s" =~ ^[0-9]+$ ]]; then sid=$(printf '%03d' "$((10#$s))"); else sid="$s"; fi
    for pair in "$MEL_SRC:$EVAL_MEL" "$CHORD_SRC:$EVAL_CHORD"; do
        src="${pair%%:*}"; dst="${pair##*:}"
        f="$src/$sid.mid"; [[ -f "$f" ]] || f="$src/$sid.MID"
        [[ -f "$f" ]] || { echo "ERROR: $sid not in $src"; exit 1; }
        cp -f "$f" "$dst/$sid.mid"
    done
done
echo "[prompts] staged $(ls "$EVAL_MEL" | wc -l) song(s) under $OUT_DIR/prompts"

if [[ -n "$CHORD_PROGRAM" ]]; then
    python - "$EVAL_CHORD" "$CHORD_PROGRAM" <<'RETAG'
import glob, os, sys
import mido
folder, prog = sys.argv[1], int(sys.argv[2])
changed = 0
for f in sorted(glob.glob(os.path.join(folder, '*.mid'))):
    m = mido.MidiFile(f)
    hit = seen = False
    for tr in m.tracks:
        for msg in tr:
            if msg.type == 'program_change':
                seen = True
                if msg.program != prog:
                    msg.program = prog
                    hit = True
    if not seen:
        # no program_change at all: the channel default (0) applies, so
        # insert one on the note track to make the tag explicit
        for tr in m.tracks:
            if any(msg.type == 'note_on' for msg in tr):
                tr.insert(0, mido.Message('program_change', program=prog, time=0))
                hit = True
                break
    if hit:
        m.save(f)
        changed += 1
print(f'[prompts] chord program -> {prog}: rewrote {changed} file(s)')
RETAG
fi

# The tag as the tokenizer will read it: program per note-carrying
# track, and the frame count of each prompt, so a short or untagged
# file is visible here and not two stages later.
python - "$EVAL_MEL" "$EVAL_CHORD" "$PROMPT_LENGTH" <<'SHOW'
import glob, os, sys
import pretty_midi
mel, chd, P = sys.argv[1], sys.argv[2], int(sys.argv[3])
print('[prompts] song   mel prog  chd prog   mel notes<P  chd notes<P')
for f in sorted(glob.glob(os.path.join(mel, '*.mid'))):
    sid = os.path.basename(f)[:-4]
    out = []
    for path in (f, os.path.join(chd, sid + '.mid')):
        pm = pretty_midi.PrettyMIDI(path)
        tpf = pm.resolution / 4
        progs = sorted({i.program for i in pm.instruments if i.notes})
        n = sum(1 for i in pm.instruments for nt in i.notes
                if pm.time_to_tick(nt.start) / tpf < P)
        out.append((progs, n))
    print(f'[prompts] {sid:5s}  {str(out[0][0]):9s} {str(out[1][0]):9s}  '
          f'{out[0][1]:11d}  {out[1][1]:11d}')
SHOW

# ---- 2. decode ---------------------------------------------------------
[[ -d "$CKPT" ]] && ckarg="$CKPT/" || ckarg="$CKPT"
env A3_SCHEDULE="$A3_SCHEDULE" A3_REFINE_STEPS="$A3_REFINE_STEPS" \
    A3_FINAL_TEMP="$A3_FINAL_TEMP" A3_TOP_P="$A3_TOP_P" \
python cp_transformer_m2c_duet_block_diffusion_combined.py \
    --ckpt "$ckarg" \
    --mel-folder "$EVAL_MEL" --chord-folder "$EVAL_CHORD" \
    --output-dir "$OUT_DIR" --modes "$MODE_NAME" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples "$N_SAMPLES" \
    --max-polyphony "$MAX_POLYPHONY" --skip-existing \
    --model-size large --moe-num-experts 4 --moe-topk 2 \
    --min-chord-tokens-before-eos 0

echo "================================================================"
echo "Done -> $OUT_DIR/<song>/$MODE_NAME/sample_<i>_temp1.0.mid"
find "$OUT_DIR" -name 'sample_*.mid' | wc -l | xargs echo "files written:"
echo "Prompt-copy check on this tree:"
echo "  sbatch --export=ALL,OUT_ROOT=$OUT_DIR,SYSTEMS=.,PROMPT_FRAMES=$PROMPT_LENGTH,\\"
echo "MEL_DIR=$EVAL_MEL,CHORD_DIR=$EVAL_CHORD,EXCLUDE_SONGS= midi_yinyang/check_prompt_copy.sbatch"
echo "================================================================"
