#!/usr/bin/env bash
# Generate the evaluation samples of one in-house system.
#
#   bash scripts/generate.sh duet            [CKPT=<run dir or .ckpt>]
#   bash scripts/generate.sh dense | no_draft | ss_finetuned | ss_scratch
#   DATASET=nottingham bash scripts/generate.sh duet   # Nottingham prompts,
#                                     # your Nottingham run, temp/E1_nottingham/
#
# Every system continues the same staged prompts (the held-out songs,
# first PROMPT_LENGTH frames) to GEN_LENGTH frames, N_SAMPLES times, into
# temp/E1/<system id>/. Finished samples are skipped, so an interrupted
# run resumes. One GPU per call; run several systems side by side with
# CUDA_VISIBLE_DEVICES=<i>.
#
# The external baselines have their own scripts under scripts/external/.
# Keep N_SAMPLES >= 2: with a single sample the duet generators write
# <song>/co.mid instead of the <song>/co/sample_<i>.mid layout evaluate.sh reads.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SYSTEM="${1:?usage: generate.sh duet|duet_stale|dense|no_draft|ss_finetuned|ss_scratch|<a duet variant>}"
ID="$(system_id "$SYSTEM")"
# SHARD=K/N: this call takes every N-th song (see slurm/infer_nottingham.sbatch).
# Run stage_prompts.sh once first; parallel calls would race to stage.
if [[ -n "${SHARD:-}" ]]; then
    need "$PROMPTS/.ready" "Run 'bash scripts/stage_prompts.sh' once before sharded calls."
else
    bash "$ROOT/scripts/stage_prompts.sh"
fi

# Default checkpoints: the released files, or your own run directories.
case "$SYSTEM" in
    duet|duet_stale|duet_v2|duet_v2c48|duet_prev|dense|no_draft|ss_finetuned|ss_scratch) ;;
    duet_joint|duet_jointmix|duet_k0|duet_bestar) ;;
    *) echo "generate.sh covers the in-house systems; see scripts/external/ for $SYSTEM" >&2; exit 1 ;;
esac
# The decode variants re-decode the paper's duet checkpoint (duet_bestar:
# its best_ar/ weights, selected on the clean-row loss alone).
CKPT_SYS="$SYSTEM"
case "$SYSTEM" in duet_joint|duet_jointmix|duet_k0|duet_bestar) CKPT_SYS=duet ;; esac
# duet_v2c48: duet_v2's run under the _c48 task, fed chord prompts on program 48
CHORD_SUB=chord
if [[ "$SYSTEM" == duet_v2c48 ]]; then
    CKPT_SYS=duet_v2; TASK="${TASK}_c48"; CHORD_SUB=chord48
    if [[ ! -d "$PROMPTS/chord48" ]]; then
        # built aside and renamed in one step: sharded calls start together
        _tmp=$(mktemp -d "$PROMPTS/chord48.XXXXXX")
        cp -f "$PROMPTS"/chord/*.mid "$_tmp/"
        python - "$_tmp" <<'PY'
import glob, os, sys
import mido
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.mid'))):
    m = mido.MidiFile(f)
    for tr in m.tracks:
        if not any(x.type == 'note_on' for x in tr):
            continue
        tr[:] = [x for x in tr if x.type != 'program_change']
        ch = next(x.channel for x in tr if x.type == 'note_on')
        tr.insert(0, mido.Message('program_change', program=48, channel=ch, time=0))
    m.save(f)
PY
        mv -T "$_tmp" "$PROMPTS/chord48" 2>/dev/null || rm -rf "$_tmp"
    fi
fi
if [[ -z "${CKPT:-}" && "$DATASET" == pop909 && "$CKPT_SYS" == duet && "$SYSTEM" == duet_bestar ]]; then
    echo "ERROR: the released duet checkpoint has no best_ar/; pass CKPT=<run dir>/best_ar" >&2; exit 1
elif [[ -z "${CKPT:-}" && "$DATASET" == pop909 && "$CKPT_SYS" =~ ^(duet|dense|no_draft|ss_finetuned|ss_scratch)$ ]]; then
    CKPT="ckpt/$CKPT_SYS"
elif [[ -z "${CKPT:-}" ]]; then
    # No released Nottingham checkpoints: default to the run directory
    # train.sh wrote for this system (without RUN_TAG; pass CKPT=
    # for a tagged run or when several batch sizes were trained).
    case "$CKPT_SYS" in
        duet|duet_stale|duet_v2|duet_prev|dense) _pat="ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_${CKPT_SYS}_${TASK}_batch_*_schedule" ;;
        no_draft)     _pat="ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_${TASK}_batch_*_schedule" ;;
        ss_finetuned) _pat="ckpt/cp_transformer_v0.42_size*_pop909ft_${SS_RUN_TAG}_batch_*_schedule" ;;
        ss_scratch)   _pat="ckpt/cp_transformer_v0.42_size*_pop909scratch_${SS_RUN_TAG}_batch_*_schedule" ;;
    esac
    mapfile -t _runs < <(compgen -G "$_pat" || true)
    [[ ${#_runs[@]} -eq 1 ]] || {
        echo "ERROR: expected one run directory matching $_pat, found ${#_runs[@]}." \
             "Train it (DATASET=$DATASET scripts/train.sh $SYSTEM) or pass CKPT=<run dir>." >&2; exit 1; }
    CKPT="${_runs[0]}"
    [[ "$SYSTEM" == duet_bestar ]] && CKPT="$CKPT/best_ar"
fi
need "$CKPT" "Download the checkpoints (scripts/download.sh) or pass CKPT=<your run dir>."
# A directory holding a single checkpoint (a released model) is resolved
# to that file; a training run directory is handed over with a trailing
# slash, and the loader picks its best checkpoint by validation loss.
if [[ -d "$CKPT" ]]; then
    mapfile -t _found < <(ls "$CKPT"/*.ckpt 2>/dev/null)
    if [[ ${#_found[@]} -eq 1 ]]; then CKPT="${_found[0]}"; else CKPT="${CKPT%/}/"; fi
fi
OUT="$OUT_ROOT/$ID"

# The generators read whole folders, so a shard is a folder of links to
# its songs, one per system and shard (parallel calls never share one).
# Outputs still go to $OUT; finished songs are skipped there as usual.
PDIR="$PROMPTS"
if [[ -n "${SHARD:-}" ]]; then
    PDIR="$PROMPTS/shards/${SYSTEM}_${SHARD/\//of}"
    mkdir -p "$PDIR"
    for sub in mel $CHORD_SUB merged_tagged; do
        mkdir -p "$PDIR/$sub"
        find "$PDIR/$sub" -maxdepth 1 -name '*.mid' -type l -delete
    done
    SONGS=$(shard $(for f in "$PROMPTS"/mel/*.mid; do basename "$f" .mid; done))
    for s in $SONGS; do
        for sub in mel $CHORD_SUB merged_tagged; do
            ln -sf "$(readlink -f "$PROMPTS/$sub/$s.mid")" "$PDIR/$sub/$s.mid"
        done
    done
    echo "[generate] $SYSTEM shard $SHARD: $(wc -w <<< "$SONGS") songs"
fi

case "$SYSTEM" in
duet|duet_stale|duet_v2|duet_v2c48|duet_prev|dense|duet_joint|duet_jointmix|duet_k0|duet_bestar)
    # Draft-and-denoise, leader alternating by frame (ctc_alt): the
    # leader's draft is sampled from its autoregressive head and written
    # into its draft token, one denoise pass reads out both streams.
    # duet_stale has no leader (joint): one forward drafts both streams
    # from their autoregressive heads, both drafts go into the draft
    # tokens, one refine pass reads out both -- the regime it trains on.
    # Temperature 1, no nucleus cut.
    #
    # Variants (README "Improving Duet's co-generation"), all leaderless:
    #   duet_joint     the paper checkpoint, joint decode: both streams
    #                  drafted by their AR heads, both drafts in the draft
    #                  tokens, one refine forward reads both frames out
    #   duet_jointmix  duet_joint, each stream sampled from the product
    #                  of its refined distribution and its own AR head's
    #                  (the head that sees frame t-1), equal weights
    #   duet_k0        AR heads only, no draft tokens read (control: the
    #                  duet checkpoint's backbone vs w/o draft tokens)
    #   duet_bestar    duet_joint on the best_ar/ weights
    #   duet_v2, duet_v2c48, duet_prev   the retrains, joint decode
    SCHED=ctc_alt; REFINE_STEPS=4; JOINT_MIX=0
    case "$SYSTEM" in
        duet_stale|duet_joint|duet_bestar|duet_v2|duet_v2c48|duet_prev) SCHED=joint ;;
        duet_jointmix) SCHED=joint; JOINT_MIX=0.5 ;;
        duet_k0)  SCHED=refine; REFINE_STEPS=0 ;;
    esac
    SHAPE=(--moe-num-experts 4 --moe-topk 2)
    [[ "$SYSTEM" == dense ]] && SHAPE=(--moe-num-experts 1 --moe-topk 1 --moe-intermediate-size 6144)
    MELCHORD_CP=4 A3_SCHEDULE=$SCHED A3_REFINE_STEPS=$REFINE_STEPS A3_FINAL_TEMP=1.0 A3_TOP_P=1.0 \
    A3_JOINT_MIX=$JOINT_MIX A3_CTC_MIX_MODE=geo \
    python cp_transformer_m2c_duet_block_diffusion_combined.py \
        --ckpt "$CKPT" --mel-folder "$PDIR/mel" --chord-folder "$PDIR/$CHORD_SUB" \
        --output-dir "$OUT" --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony 4 --skip-existing \
        --model-size large "${SHAPE[@]}" --min-chord-tokens-before-eos 0
    ;;
no_draft)
    MELCHORD_CP=4 python cp_transformer_m2c_intra_cross_attn_combined.py \
        --ckpt "$CKPT" --mel-folder "$PDIR/mel" --chord-folder "$PDIR/chord" \
        --output-dir "$OUT" --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony 4 --skip-existing \
        --model-size large --moe-num-experts 4 --moe-topk 2
    ;;
ss_finetuned|ss_scratch)
    # single stream: continues the merged file (chords on program 48) and
    # writes temp/<save-name>/<song>.mid_temp1.0_continuation_<i>.mid
    python cp_transformer_inference.py --ckpt "$CKPT" \
        --midi-folder "$PDIR/merged_tagged" \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony 16 --skip-existing \
        --save-name "${OUT#temp/}"
    ;;
esac
echo "[generate] $SYSTEM -> $DUET/$OUT"
