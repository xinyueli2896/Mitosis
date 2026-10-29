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
    duet|duet_stale|duet_v2|duet_v2c48|duet_prev|duet_v3|duet_v3r|duet_v3h|duet_v3o|duet_jf|duet_jfr|duet_jfm|duet_v3g|duet_jfmg|duet_jfme|duet_jfmeg|duet_jfms|dense|no_draft|ss_finetuned|ss_scratch) ;;
    duet_joint|duet_jointmix|duet_k0|duet_bestar|duet_prevmix|duet_v2mix|duet_prevk0|duet_prevr3|duet_v3rk0|duet_v3rr3|duet_v3g15|duet_v3g2|duet_jfmg15|duet_jfmg2) ;;
    duet_jfmg15s8|duet_jfmg15rnd|duet_jfmg15s8rnd|duet_jfmg15R2|duet_jfmg15s8R2|duet_jfmg15rndR2|duet_jfmg15s8rndR2|duet_jfmeg15|duet_jfmeg15ne|duet_jfms15|duet_jfmssolo) ;;
    *) echo "generate.sh covers the in-house systems; see scripts/external/ for $SYSTEM" >&2; exit 1 ;;
esac
# The decode variants re-decode the paper's duet checkpoint (duet_bestar:
# its best_ar/ weights, selected on the clean-row loss alone).
CKPT_SYS="$SYSTEM"
case "$SYSTEM" in
    duet_joint|duet_jointmix|duet_k0|duet_bestar) CKPT_SYS=duet ;;
    duet_prevmix|duet_prevk0|duet_prevr3) CKPT_SYS=duet_prev ;;
    duet_v2mix)   CKPT_SYS=duet_v2 ;;
    duet_v3rk0|duet_v3rr3) CKPT_SYS=duet_v3r ;;
    duet_v3g15|duet_v3g2) CKPT_SYS=duet_v3g ;;
    duet_jfmg15|duet_jfmg2|duet_jfmg15s8|duet_jfmg15rnd|duet_jfmg15s8rnd|duet_jfmg15R2|duet_jfmg15s8R2|duet_jfmg15rndR2|duet_jfmg15s8rndR2) CKPT_SYS=duet_jfmg ;;
    duet_jfmeg15|duet_jfmeg15ne) CKPT_SYS=duet_jfmeg ;;
    duet_jfms15|duet_jfmssolo) CKPT_SYS=duet_jfms ;;
esac
# duet_v2c48: duet_v2's run under the _c48 task (chord stream on 48)
MEL_DATA="data/${DUET_DATA}_melody_cp4_v2.pt"
CHORD_DATA="data/${DUET_DATA}_chord_cp4_v2.pt"
if [[ "$SYSTEM" == duet_v2c48 ]]; then
    CKPT_SYS=duet_v2; TASK="${TASK}_c48"
    CHORD_DATA="data/nottingham_chord48_cp4_v2.pt"
fi
# duet_jf / duet_jfr: on Nottingham trained with chords on 48 (_c48 task)
if [[ "$SYSTEM" == duet_jf* && "$DATASET" == nottingham ]]; then
    TASK="${TASK}_c48"
    CHORD_DATA="data/nottingham_chord48_cp4_v2.pt"
fi

# Two-stream prompts carry each stream on the MIDI program it was TRAINED
# on: the models keep the real program as a token (preserve_program), so
# a model trained with chords on 48 must be prompted with chords on 48.
# The program is the most common one in the training tensor (cached
# beside it); MEL_PROGRAM / CHORD_PROGRAM override it, e.g. for a
# checkpoint trained on other data than the local files.
train_program() {   # <.pt> -> the most common program among its notes
    local pt="$1" cache="$1.program"
    if [[ ! -s "$cache" ]]; then
        need "$pt" "No training tensor to read the program from; set MEL_PROGRAM / CHORD_PROGRAM."
        python - "$pt" > "$cache.$$" <<'PY'
import collections, sys, torch
p = torch.load(sys.argv[1]).view(-1, 4)[:, 0]
print(collections.Counter(p[p < 128].tolist()).most_common(1)[0][0])
PY
        mv -f "$cache.$$" "$cache"
    fi
    cat "$cache"
}
retag_prompts() {   # <src sub> <program> -> prompts/<src>_p<program>, built once
    local src="$1" prog="$2" dst="$PROMPTS/${1}_p$2"
    if [[ ! -d "$dst" ]]; then
        # built aside and renamed in one step: sharded calls start together
        local tmp; tmp=$(mktemp -d "$dst.XXXXXX")
        cp -f "$PROMPTS/$src"/*.mid "$tmp/"
        python - "$tmp" "$prog" <<'PY'
import glob, os, sys
import mido
prog = int(sys.argv[2])
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.mid'))):
    m = mido.MidiFile(f)
    for tr in m.tracks:
        if not any(x.type == 'note_on' for x in tr):
            continue
        tr[:] = [x for x in tr if x.type != 'program_change']
        ch = next(x.channel for x in tr if x.type == 'note_on')
        tr.insert(0, mido.Message('program_change', program=prog, channel=ch, time=0))
    m.save(f)
PY
        mv -T "$tmp" "$dst" 2>/dev/null || rm -rf "$tmp"
    fi
    echo "${1}_p$2"
}
MEL_SUB=mel; CHORD_SUB=chord
case "$SYSTEM" in ss_finetuned|ss_scratch) ;; *)
    MEL_PROGRAM="${MEL_PROGRAM:-$(train_program "$MEL_DATA")}"
    CHORD_PROGRAM="${CHORD_PROGRAM:-$(train_program "$CHORD_DATA")}"
    MEL_SUB=$(retag_prompts mel "$MEL_PROGRAM")
    CHORD_SUB=$(retag_prompts chord "$CHORD_PROGRAM")
    echo "[generate] $SYSTEM: prompts on the trained programs, melody $MEL_PROGRAM, chord $CHORD_PROGRAM"
    ;;
esac
if [[ -z "${CKPT:-}" && "$DATASET" == pop909 && "$CKPT_SYS" == duet && "$SYSTEM" == duet_bestar ]]; then
    echo "ERROR: the released duet checkpoint has no best_ar/; pass CKPT=<run dir>/best_ar" >&2; exit 1
elif [[ -z "${CKPT:-}" && "$DATASET" == pop909 && "$CKPT_SYS" =~ ^(duet|dense|no_draft|ss_finetuned|ss_scratch)$ ]]; then
    CKPT="ckpt/$CKPT_SYS"
elif [[ -z "${CKPT:-}" ]]; then
    # No released Nottingham checkpoints: default to the run directory
    # train.sh wrote for this system (without RUN_TAG; pass CKPT=
    # for a tagged run or when several batch sizes were trained).
    case "$CKPT_SYS" in
        duet|duet_stale|duet_v2|duet_prev|duet_v3|duet_v3r|duet_v3h|duet_v3o|duet_jf|duet_jfr|duet_jfm|duet_v3g|duet_jfmg|duet_jfme|duet_jfmeg|duet_jfms|dense) _pat="ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_${CKPT_SYS}_${TASK}_batch_*_schedule" ;;
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
    for sub in mel $MEL_SUB $CHORD_SUB merged_tagged; do
        mkdir -p "$PDIR/$sub"
        find "$PDIR/$sub" -maxdepth 1 -name '*.mid' -type l -delete
    done
    SONGS=$(shard $(for f in "$PROMPTS"/mel/*.mid; do basename "$f" .mid; done))
    for s in $SONGS; do
        for sub in mel $MEL_SUB $CHORD_SUB merged_tagged; do
            ln -sf "$(readlink -f "$PROMPTS/$sub/$s.mid")" "$PDIR/$sub/$s.mid"
        done
    done
    echo "[generate] $SYSTEM shard $SHARD: $(wc -w <<< "$SONGS") songs"
fi

case "$SYSTEM" in
duet|duet_stale|duet_v2|duet_v2c48|duet_prev|duet_v3|duet_v3r|duet_v3h|duet_v3o|duet_jf|duet_jfr|duet_jfm|duet_v3g|duet_jfmg|dense|duet_joint|duet_jointmix|duet_k0|duet_bestar|duet_prevmix|duet_v2mix|duet_prevk0|duet_prevr3|duet_v3rk0|duet_v3rr3|duet_v3g15|duet_v3g2|duet_jfmg15|duet_jfmg2|duet_jfmg15s8|duet_jfmg15rnd|duet_jfmg15s8rnd|duet_jfmg15R2|duet_jfmg15s8R2|duet_jfmg15rndR2|duet_jfmg15s8rndR2|duet_jfme|duet_jfmeg|duet_jfmeg15|duet_jfmeg15ne|duet_jfms|duet_jfms15|duet_jfmssolo)
    # Draft-and-denoise, leader alternating by frame (ctc_alt): the
    # leader's draft is sampled from its autoregressive head and written
    # into its draft token, one denoise pass reads out both streams.
    # duet_stale has no leader (joint): one forward drafts both streams
    # from their autoregressive heads, both drafts go into the draft
    # tokens, one refine pass reads out both -- the regime it trains on.
    # Temperature 1, no nucleus cut. Each stream samples only the program
    # its prompt carries (LOCK_PROGRAM=0 lifts that), so a program-tagged
    # model keeps chords on 48 and an untagged one keeps both on theirs.
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
    #   duet_v2, duet_v2c48, duet_prev, duet_v3   the retrains, joint decode
    #   duet_prevmix, duet_v2mix         duet_prev / duet_v2 with
    #                  duet_jointmix's AR-head mix
    #   duet_prevk0    duet_prev with duet_k0's AR heads only
    #   duet_prevr3    duet_prev, joint refine repeated 3 rounds
    #   duet_v3h       held-note-token retrain, joint decode (the decode
    #                  builds each frame's held notes from the history)
    #   duet_jf, duet_jfr  joint-frame retrains, joint decode: each frame
    #                  of both streams is one draw from the joint head
    #   duet_v3r       residual-refine retrain, joint decode; duet_v3rk0
    #                  its AR heads only, duet_v3rr3 its refine x3
    #   duet_jfm       masked joint frame, joint decode: each frame's note
    #                  slots unmasked over 4 rounds, most confident first
    #   duet_v3g, duet_jfmg  partner-dropout retrains, joint decode, no
    #                  guidance; duet_v3g15/2, duet_jfmg15/2 the same
    #                  checkpoints with partner guidance w = 1.5 / 2
    #   duet_jfmg15s8, _rnd, _R2 and their combinations: duet_jfmg15
    #                  with 8 unmasking rounds (one note each), a random
    #                  unmasking order, and/or 2 refine rounds that
    #                  re-mask one whole stream and fill it again given
    #                  the other (alternating, random first per frame)
    #   duet_jfme, duet_jfmeg  self-correcting masked-frame retrains:
    #                  visible notes may be edited during the unmasking
    #                  rounds, then 2 edit-only refine rounds;
    #                  duet_jfmeg15 the same at guidance 1.5, and
    #                  duet_jfmeg15ne at 1.5 with edits off (the
    #                  ablation: duet_jfmg15's decode on this checkpoint)
    #   duet_jfms      separable masked-frame retrain, joint decode;
    #                  duet_jfms15 at guidance 1.5 (its uncond pass has no
    #                  partner at either level); duet_jfmssolo each stream
    #                  generated alone (A3_SOLO=1), from its own prompt only
    SCHED=ctc_alt; REFINE_STEPS=4; JOINT_MIX=0; JOINT_ROUNDS=1; PARTNER_CFG=1
    M_STEPS=4; M_ORDER=conf; M_REFINE=0; M_MODE=stream; M_EDIT=""   # masked frame
    SOLO=0
    case "$SYSTEM" in
        duet_stale|duet_joint|duet_bestar|duet_v2|duet_v2c48|duet_prev|duet_v3|duet_v3r|duet_v3h|duet_v3o|duet_jf|duet_jfr|duet_jfm|duet_v3g|duet_jfmg) SCHED=joint ;;
        duet_v3g15|duet_jfmg15) SCHED=joint; PARTNER_CFG=1.5 ;;
        duet_v3g2|duet_jfmg2) SCHED=joint; PARTNER_CFG=2 ;;
        duet_jfmg15?*) SCHED=joint; PARTNER_CFG=1.5
            [[ "$SYSTEM" == *s8* ]] && M_STEPS=8
            [[ "$SYSTEM" == *rnd* ]] && M_ORDER=random
            [[ "$SYSTEM" == *R2 ]] && M_REFINE=2
            ;;
        duet_jfme|duet_jfmeg) SCHED=joint; M_REFINE=2; M_MODE=edit ;;
        duet_jfmeg15) SCHED=joint; PARTNER_CFG=1.5; M_REFINE=2; M_MODE=edit ;;
        duet_jfmeg15ne) SCHED=joint; PARTNER_CFG=1.5; M_EDIT=0 ;;
        duet_jfms) SCHED=joint ;;
        duet_jfms15) SCHED=joint; PARTNER_CFG=1.5 ;;
        duet_jfmssolo) SCHED=joint; SOLO=1 ;;
        duet_prevr3|duet_v3rr3) SCHED=joint; JOINT_ROUNDS=3 ;;
        duet_jointmix|duet_prevmix|duet_v2mix) SCHED=joint; JOINT_MIX=0.5 ;;
        duet_k0|duet_prevk0|duet_v3rk0) SCHED=refine; REFINE_STEPS=0 ;;
    esac
    SHAPE=(--moe-num-experts 4 --moe-topk 2)
    [[ "$SYSTEM" == dense ]] && SHAPE=(--moe-num-experts 1 --moe-topk 1 --moe-intermediate-size 6144)
    MELCHORD_CP=4 A3_SCHEDULE=$SCHED A3_REFINE_STEPS=$REFINE_STEPS A3_FINAL_TEMP=1.0 A3_TOP_P=1.0 \
    A3_JOINT_MIX=$JOINT_MIX A3_JOINT_ROUNDS=$JOINT_ROUNDS A3_CTC_MIX_MODE=geo \
    A3_PARTNER_CFG=$PARTNER_CFG A3_MASK_STEPS="${MASK_STEPS:-$M_STEPS}" \
    A3_MASK_ORDER="${MASK_ORDER:-$M_ORDER}" A3_MASK_REFINE="${MASK_REFINE:-$M_REFINE}" \
    A3_MASK_REFINE_MODE="${MASK_REFINE_MODE:-$M_MODE}" A3_MASK_EDIT="${MASK_EDIT:-$M_EDIT}" \
    A3_MASK_EDIT_MARGIN="${MASK_EDIT_MARGIN:-1.0}" A3_SOLO=$SOLO \
    A3_LOCK_PROGRAM="${LOCK_PROGRAM:-1}" \
    python cp_transformer_m2c_duet_block_diffusion_combined.py \
        --ckpt "$CKPT" --mel-folder "$PDIR/$MEL_SUB" --chord-folder "$PDIR/$CHORD_SUB" \
        --output-dir "$OUT" --modes co \
        --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
        --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony 4 --skip-existing \
        --model-size large "${SHAPE[@]}" --min-chord-tokens-before-eos 0
    ;;
no_draft)
    MELCHORD_CP=4 python cp_transformer_m2c_intra_cross_attn_combined.py \
        --ckpt "$CKPT" --mel-folder "$PDIR/$MEL_SUB" --chord-folder "$PDIR/$CHORD_SUB" \
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
