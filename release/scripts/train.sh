#!/usr/bin/env bash
# Train one of the five in-house systems of the paper.
#
#   bash scripts/train.sh duet            # Duet (the proposed model)
#   bash scripts/train.sh duet_stale      # Duet, drafts from a stale snapshot
#   bash scripts/train.sh duet_v2         # Duet trained for the leaderless joint decode (see README)
#   bash scripts/train.sh duet_v2c48      # duet_v2 with the chord stream on program 48
#   bash scripts/train.sh duet_prev       # Duet, query slots also read frame t-1
#   bash scripts/train.sh duet_v3         # Duet trained so the leaderless refine can beat its AR heads
#   bash scripts/train.sh duet_v3r        # duet_v3 with the residual refine (slot = AR head + correction)
#   bash scripts/train.sh duet_v3h        # duet_v3 with held-note tokens (every frame states what is sounding)
#   bash scripts/train.sh duet_v3o        # duet_v3 with query frames drawn mostly where a note starts
#   bash scripts/train.sh duet_jf         # duet_v3 with one local decoder pass per frame for both streams
#   bash scripts/train.sh duet_jfr        # duet_jf with a random note order inside each frame
#   bash scripts/train.sh duet_jfm        # duet_v3 with the masked joint frame (note slots unmasked in rounds)
#   bash scripts/train.sh duet_v3g        # duet_v3 trained with partner dropout, for partner guidance
#   bash scripts/train.sh duet_jfmg       # duet_jfm trained with partner dropout
#   bash scripts/train.sh duet_jfme       # duet_jfm trained to fix its own visible notes
#   bash scripts/train.sh duet_jfmeg      # duet_jfme trained with partner dropout
#   bash scripts/train.sh dense           # Duet w/o MoE: one dense FFN of width 6144
#   bash scripts/train.sh no_draft        # Duet w/o draft tokens
#   bash scripts/train.sh ss_finetuned    # single-stream backbone, finetuned
#   bash scripts/train.sh ss_scratch      # single-stream backbone, random init
#
# DATASET=nottingham trains the same systems on the Nottingham tensors
# (prepare_data.sh with DATASET=nottingham); run directories carry the
# task name, so they never collide with the POP909 runs.
#
# Uses every GPU in CUDA_VISIBLE_DEVICES (torchrun when there are
# several). The learning rate scales with the GPU count as in the paper
# runs unless MAX_LR is set. Re-running resumes from the run's last.ckpt.
#
# Knobs (environment): MAX_LR, LR_TOTAL_STEPS, BATCH_SIZE (per GPU),
# RUN_TAG (run-name suffix), WANDB=1 (log to Weights & Biases).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SYSTEM="${1:?usage: train.sh duet|duet_stale|duet_v2|duet_v2c48|duet_prev|duet_v3|duet_v3r|duet_v3h|duet_v3o|duet_jf|duet_jfr|duet_jfm|duet_v3g|duet_jfmg|duet_jfme|duet_jfmeg|dense|no_draft|ss_finetuned|ss_scratch}"
system_id "$SYSTEM" >/dev/null

NGPUS="$(n_gpus)"
if [[ -z "${MAX_LR:-}" ]]; then
    case "$NGPUS" in 1) MAX_LR=5e-5 ;; 2) MAX_LR=7e-5 ;; 4) MAX_LR=1e-4 ;; 8) MAX_LR=1.5e-4 ;; *) MAX_LR=1e-4 ;; esac
fi
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
launch() {   # launch <script.py> args...: python on one GPU, torchrun on several
    if [[ "$NGPUS" -le 1 ]]; then python "$@"
    else python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPUS" "$@"; fi
}
wandb_flag() { [[ "${WANDB:-0}" == "1" ]] && echo "$1" || echo "$2"; }

case "$SYSTEM" in
duet|duet_stale|duet_v2|duet_v2c48|duet_prev|duet_v3|duet_v3r|duet_v3h|duet_v3o|duet_jf|duet_jfr|duet_jfm|duet_v3g|duet_jfmg|duet_jfme|duet_jfmeg|dense)
    # Both Duet streams read cp4 tensors: MAX_POLYPHONY is the tokenizer's
    # budget and MELCHORD_CP selects data/${DUET_DATA}_*_cp${MELCHORD_CP}_v2.pt.
    # The checkpoint does not record it, so keep the two in step.
    MAX_POLYPHONY=4
    export MELCHORD_CP="$MAX_POLYPHONY"
    need "data/${DUET_DATA}_melody_cp${MELCHORD_CP}_v2.pt" "Run 'prepare_data.sh train' or download the data."
    need "$BACKBONE" "Download the pretrained backbone (scripts/download.sh)."
    if [[ "$SYSTEM" == duet_v2c48 ]]; then
        # same recipe as duet_v2, chord stream on program 48; the task
        # name carries it into the run directory
        [[ "$DATASET" == nottingham ]] || { echo "ERROR: duet_v2c48 is built for DATASET=nottingham" >&2; exit 1; }
        need "data/nottingham_chord48_cp${MELCHORD_CP}_v2.pt" "Run 'DATASET=nottingham prepare_data.sh chord48'."
        TASK="${TASK}_c48"
    fi
    if [[ "$SYSTEM" == duet_jf* && "$DATASET" == nottingham ]]; then
        # the joint frame tells the streams apart by program, and both
        # Nottingham tracks are on 0: train on the program-48 chord
        # tensor, as duet_v2c48 and the single-stream models do
        need "data/nottingham_chord48_cp${MELCHORD_CP}_v2.pt" "Run 'DATASET=nottingham prepare_data.sh chord48'."
        TASK="${TASK}_c48"
    fi
    # Draft slots: the paper Duet fills a slot with the live model's
    # query-head argmax for half the items. duet_stale fills BOTH slots of
    # every item with a draft sampled from the AR head the way the decode
    # does (local_sampling, temperature 1), drawn from a frozen copy of
    # the model refreshed every STALE_EVERY steps; no drafts before the
    # first refresh.
    DRAFT=(--self_cond_prob 0.5)
    MRQL=0      # --mask_revealed_query_loss
    QP=1        # --query_pairs
    if [[ "$SYSTEM" != dense ]]; then
        EXPERTS=(--moe_num_experts 4 --moe_topk 2)
        ABBR=$SYSTEM; STEPS="${LR_TOTAL_STEPS:-50000}"; LIMIT_VAL=25
        case "$SYSTEM" in
        duet_stale)
            DRAFT=(--self_cond_prob 1.0 --sc_ar_frac 1.0 --sc_ar_free_run
                   --sc_draft_temp 1.0 --stale_draft_every "${STALE_EVERY:-1000}") ;;
        duet_prev)
            # the paper recipe with one change: a query slot also reads
            # frame t-1 of both streams, which the AR head that drafts
            # into it always could (without it the slot sees up to t-2)
            DRAFT=(--self_cond_prob 0.5 --slot_sees_prev_frame) ;;
        duet_v2|duet_v2c48)
            # trained for the leaderless joint decode it is sampled with:
            # the slot reads frame t-1; every query pair is the decode's
            # state -- both draft tokens hold a draft sampled from the AR
            # heads the way the decode samples it (free-running,
            # temperature 1), both tagged k=0, both slots scored against
            # their true frames; validation in that same state (drafts
            # at argmax), so best-val selects on what the decode reads.
            DRAFT=(--slot_sees_prev_frame --self_cond_prob 1.0
                   --sc_ar_frac 1.0 --sc_ar_free_run --sc_draft_temp 1.0
                   --draft_k0 --sc_val) ;;
        duet_v3|duet_v3r|duet_v3h|duet_v3o|duet_jf|duet_jfr|duet_jfm|duet_v3g|duet_jfmg|duet_jfme|duet_jfmeg)
            # duet_v2's fixes without its k=0-only states: the slot reads
            # t-1; a draft token never holds the true frame -- unmasked it
            # always carries an AR-sampled draft (decode's sampler, temp
            # 1), so the refine has nothing to copy and must correct; one
            # commitment level per pair for both slots (the joint decode
            # moves them together), masked only at k=K, a whole draft at
            # every k<K (--sc_k_consistent); 8 query pairs per sequence
            # instead of 1. Both losses are token means, so Q does not
            # shift the AR/query balance (query_loss_weight 2 either
            # way); it averages the query term over 8x the frames.
            DRAFT=(--slot_sees_prev_frame --self_cond_prob 1.0
                   --sc_ar_frac 1.0 --sc_ar_free_run --sc_draft_temp 1.0
                   --sym_k --sc_k_consistent --sc_val)
            QP=8
            # duet_v3r: each slot outputs its stream's AR row for the
            # same frame plus a zero-initialised correction, so the
            # refine starts exactly at AR-heads-only quality; the AR row
            # is detached there, so only the AR loss trains it
            if [[ "$SYSTEM" == duet_v3r ]]; then DRAFT+=(--residual_refine); fi
            # duet_v3h: up to 4 held notes in front of every frame of both
            # streams, built from the onsets on the fly (no new data)
            if [[ "$SYSTEM" == duet_v3h ]]; then DRAFT+=(--hold_tokens 4); fi
            # duet_v3o: query frames drawn with weight 1 + 4 x (share of
            # items with an onset there), not uniformly
            if [[ "$SYSTEM" == duet_v3o ]]; then DRAFT+=(--query_onset_weight 4); fi
            # duet_jf / duet_jfr: frame t of both streams comes out of
            # one local decoder pass, [melody row, chord row, merged
            # notes], as the pretrained local decoder saw music; the
            # notes in the corpus order (jf) or a fresh random one (jfr)
            if [[ "$SYSTEM" == duet_jf ]]; then DRAFT+=(--joint_frame s1); fi
            if [[ "$SYSTEM" == duet_jfr ]]; then DRAFT+=(--joint_frame random); fi
            # duet_jfm / duet_jfmg: the masked joint frame -- a fixed grid
            # of note slots per stream, filled by masked prediction over
            # A3_MASK_STEPS rounds at decode, most confident first
            if [[ "$SYSTEM" == duet_jfm* ]]; then DRAFT+=(--joint_frame masked); fi
            # duet_jfme / duet_jfmeg: the masked joint frame trained to
            # correct itself -- half the masked slots take the model's own
            # sample, every slot is scored (fill the masked, keep or fix
            # the visible), so the decode can edit committed notes
            if [[ "$SYSTEM" == duet_jfme* ]]; then DRAFT+=(--jm_edit_prob "${JM_EDIT_PROB:-0.5}"); fi
            # duet_v3g / duet_jfmg / duet_jfmeg: 10% of items train with
            # the other stream cut off (every cross-stream gate 0), which
            # partner guidance (A3_PARTNER_CFG) contrasts at decode
            if [[ "$SYSTEM" == duet_v3g || "$SYSTEM" == duet_jfmg || "$SYSTEM" == duet_jfmeg ]]; then
                DRAFT+=(--partner_drop_prob 0.1)
            fi ;;
        esac
    else
        # compute-matched dense arm: one expert of width 6144 = two active 3072-wide experts
        EXPERTS=(--moe_num_experts 1 --moe_topk 1 --moe_intermediate_size 6144)
        ABBR=dense; STEPS="${LR_TOTAL_STEPS:-100000}"; LIMIT_VAL=100
    fi
    BATCH_SIZE="${BATCH_SIZE:-4}"
    [[ "$SYSTEM" == duet_v2c48 ]] && ABBR=duet_v2   # the task name tells the two apart
    INIT="ckpt/init/${ABBR%%_*}_init.ckpt"   # every duet_* variant starts from duet's init
    RUN_DIR="ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_${ABBR}_${TASK}${RUN_TAG:+_$RUN_TAG}_batch_$((BATCH_SIZE * NGPUS))_schedule"
    if [[ -f "$RUN_DIR/last.ckpt" ]]; then
        START="$RUN_DIR/last.ckpt"; echo "[train] resuming $START"
    else
        # copy the backbone into both streams: per-stream attention, four
        # experts initialised from its feed-forward layer, cross-stream
        # gates closed (bias -10) so step 0 is the backbone on each stream
        # duet and duet_stale share this file, and train_all.sh may start
        # them together: build it under a lock
        mkdir -p "$(dirname "$INIT")"
        ( flock 9
          [[ -f "$INIT" ]] || python init_pretrained_into_duet_block_diffusion.py \
              --pretrained "$BACKBONE" --output "$INIT" \
              --model_size large --global_num_layers 12 "${EXPERTS[@]}" \
              --gate_init_bias -10.0 --query_loss_weight 2.0 --diffusion_K 4
        ) 9>"$INIT.lock"
        START="$INIT"
    fi
    launch cp_transformer_m2c_duet_block_diffusion.py \
        --task "$TASK" --max_polyphony "$MAX_POLYPHONY" \
        --batch_size "$BATCH_SIZE" --model_size large "${EXPERTS[@]}" \
        --max_lr "$MAX_LR" --lr_total_steps "$STEPS" \
        --val_check_interval 500 --limit_val_batches "$LIMIT_VAL" \
        --gradient_clip_val 1.0 --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 --dump_samples_dir "temp/train_samples_$ABBR$DS_SUFFIX" \
        --gate_init_bias -10.0 --query_loss_weight 2.0 "${DRAFT[@]}" \
        --time_rope_aligned 1 --moe_modality_bias 0 --moe_modality_gates 1 \
        --moe_modality_hard_route 0 --token_level_mask 0 --mask_revealed_query_loss "$MRQL" \
        --query_pairs "$QP" --query_block 1 --decoy_corruption 0 \
        --decoy_mask_residual 0.25 --decoy_lag_bins 1:2,3:11,12:19 \
        --model_abbr "$ABBR" --diffusion_K 4 --step_ckpt_every 5000 \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} $(wandb_flag --wandb "") \
        --checkpoint_path "$START"
    echo "[train] checkpoints under $DUET/$RUN_DIR"
    ;;
no_draft)
    # The same two-stream model without draft tokens: per-stream attention
    # with gated cross-stream reads and the shared experts, decoded
    # autoregressively.
    MAX_POLYPHONY=4
    export MELCHORD_CP="$MAX_POLYPHONY"
    need "data/${DUET_DATA}_melody_cp${MELCHORD_CP}_v2.pt" "Run 'prepare_data.sh train' or download the data."
    need "$BACKBONE" "Download the pretrained backbone (scripts/download.sh)."
    INIT="ckpt/init/no_draft_init.ckpt"
    [[ -f "$INIT" ]] || python init_pretrained_into_intra_cross_attn.py \
        --pretrained "$BACKBONE" --output "$INIT" \
        --model_size large --global_num_layers 12 \
        --moe_num_experts 4 --moe_topk 2 --gate_init_bias -10.0
    launch cp_transformer_m2c_intra_cross_attn.py \
        --task "$TASK" --max_polyphony "$MAX_POLYPHONY" \
        --batch_size "${BATCH_SIZE:-4}" --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" --lr_total_steps "${LR_TOTAL_STEPS:-50000}" \
        --val_check_interval 500 --ctx_corrupt_prob 0.0 --ctx_corrupt_len 8 \
        --time_rope_aligned 1 --gradient_clip_val 1.0 --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 --dump_samples_dir "temp/train_samples_no_draft$DS_SUFFIX" \
        --gate_init_bias -10.0 --fresh_schedule \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} $(wandb_flag --wandb "") \
        --checkpoint_path "$INIT"
    ;;
ss_finetuned|ss_scratch)
    # The backbone as a single stream over the merged, program-tagged
    # sequence: finetuned from the pretrained weights, or from random init.
    DATA="$SS_DATA"
    need "$DATA" "Run 'prepare_data.sh train' or download the data."
    EXTRA=()
    if [[ "$SYSTEM" == ss_scratch ]]; then
        EXTRA+=(--from_scratch --run_tag "${RUN_TAG:-$SS_RUN_TAG}")
    else
        need "$BACKBONE" "Download the pretrained backbone (scripts/download.sh)."
        EXTRA+=(--run_tag "${RUN_TAG:-$SS_RUN_TAG}")
    fi
    [[ "${WANDB:-0}" == "1" ]] || EXTRA+=(--no_wandb)
    launch finetune_pop909.py --data "$DATA" --pretrained "$BACKBONE" \
        --batch_size "${BATCH_SIZE:-12}" --max_lr "${MAX_LR_SS:-1e-5}" \
        --lr_total_steps "${LR_TOTAL_STEPS:-2000}" --val_check_interval 25 \
        "${EXTRA[@]}"
    ;;
esac
