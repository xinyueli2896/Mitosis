#!/usr/bin/env bash
# Train one of the five in-house systems of the paper.
#
#   bash scripts/train.sh duet            # Duet (the proposed model)
#   bash scripts/train.sh dense           # Duet w/o MoE: one dense FFN of width 6144
#   bash scripts/train.sh no_draft        # Duet w/o draft tokens
#   bash scripts/train.sh ss_finetuned    # single-stream backbone, finetuned
#   bash scripts/train.sh ss_scratch      # single-stream backbone, random init
#
# Uses every GPU in CUDA_VISIBLE_DEVICES (torchrun when there are
# several). The learning rate scales with the GPU count as in the paper
# runs unless MAX_LR is set. Re-running resumes from the run's last.ckpt.
#
# Knobs (environment): MAX_LR, LR_TOTAL_STEPS, BATCH_SIZE (per GPU),
# RUN_TAG (run-name suffix), WANDB=1 (log to Weights & Biases).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SYSTEM="${1:?usage: train.sh duet|dense|no_draft|ss_finetuned|ss_scratch}"
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
duet|dense)
    # Both Duet streams read cp4 tensors: MAX_POLYPHONY is the tokenizer's
    # budget and MELCHORD_CP selects data/pop909_*_cp${MELCHORD_CP}_v2.pt.
    # The checkpoint does not record it, so keep the two in step.
    MAX_POLYPHONY=4
    export MELCHORD_CP="$MAX_POLYPHONY"
    need "data/pop909_melody_cp${MELCHORD_CP}_v2.pt" "Run 'prepare_data.sh train' or download the data."
    need "$BACKBONE" "Download the pretrained backbone (scripts/download.sh)."
    if [[ "$SYSTEM" == duet ]]; then
        EXPERTS=(--moe_num_experts 4 --moe_topk 2)
        ABBR=duet; STEPS="${LR_TOTAL_STEPS:-50000}"; LIMIT_VAL=25
    else
        # compute-matched dense arm: one expert of width 6144 = two active 3072-wide experts
        EXPERTS=(--moe_num_experts 1 --moe_topk 1 --moe_intermediate_size 6144)
        ABBR=dense; STEPS="${LR_TOTAL_STEPS:-100000}"; LIMIT_VAL=100
    fi
    BATCH_SIZE="${BATCH_SIZE:-4}"
    INIT="ckpt/init/${ABBR}_init.ckpt"
    RUN_DIR="ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_${ABBR}_melchord${RUN_TAG:+_$RUN_TAG}_batch_$((BATCH_SIZE * NGPUS))_schedule"
    if [[ -f "$RUN_DIR/last.ckpt" ]]; then
        START="$RUN_DIR/last.ckpt"; echo "[train] resuming $START"
    else
        # copy the backbone into both streams: per-stream attention, four
        # experts initialised from its feed-forward layer, cross-stream
        # gates closed (bias -10) so step 0 is the backbone on each stream
        [[ -f "$INIT" ]] || python init_pretrained_into_duet_block_diffusion.py \
            --pretrained "$BACKBONE" --output "$INIT" \
            --model_size large --global_num_layers 12 "${EXPERTS[@]}" \
            --gate_init_bias -10.0 --query_loss_weight 2.0 --diffusion_K 4
        START="$INIT"
    fi
    launch cp_transformer_m2c_duet_block_diffusion.py \
        --task melchord --max_polyphony "$MAX_POLYPHONY" \
        --batch_size "$BATCH_SIZE" --model_size large "${EXPERTS[@]}" \
        --max_lr "$MAX_LR" --lr_total_steps "$STEPS" \
        --val_check_interval 500 --limit_val_batches "$LIMIT_VAL" \
        --gradient_clip_val 1.0 --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 --dump_samples_dir "temp/train_samples_$ABBR" \
        --gate_init_bias -10.0 --query_loss_weight 2.0 --self_cond_prob 0.5 \
        --time_rope_aligned 1 --moe_modality_bias 0 --moe_modality_gates 1 \
        --moe_modality_hard_route 0 --token_level_mask 0 --mask_revealed_query_loss 0 \
        --query_pairs 1 --query_block 1 --decoy_corruption 0 \
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
    need "data/pop909_melody_cp${MELCHORD_CP}_v2.pt" "Run 'prepare_data.sh train' or download the data."
    need "$BACKBONE" "Download the pretrained backbone (scripts/download.sh)."
    INIT="ckpt/init/no_draft_init.ckpt"
    [[ -f "$INIT" ]] || python init_pretrained_into_intra_cross_attn.py \
        --pretrained "$BACKBONE" --output "$INIT" \
        --model_size large --global_num_layers 12 \
        --moe_num_experts 4 --moe_topk 2 --gate_init_bias -10.0
    launch cp_transformer_m2c_intra_cross_attn.py \
        --task melchord --max_polyphony "$MAX_POLYPHONY" \
        --batch_size "${BATCH_SIZE:-4}" --model_size large \
        --moe_num_experts 4 --moe_topk 2 \
        --max_lr "$MAX_LR" --lr_total_steps "${LR_TOTAL_STEPS:-50000}" \
        --val_check_interval 500 --ctx_corrupt_prob 0.0 --ctx_corrupt_len 8 \
        --time_rope_aligned 1 --gradient_clip_val 1.0 --aux_loss_weight 0.01 \
        --moe_monitor_every_n_steps 500 --dump_samples_dir temp/train_samples_no_draft \
        --gate_init_bias -10.0 --fresh_schedule \
        ${RUN_TAG:+--run_tag "$RUN_TAG"} $(wandb_flag --wandb "") \
        --checkpoint_path "$INIT"
    ;;
ss_finetuned|ss_scratch)
    # The backbone as a single stream over the merged, program-tagged
    # sequence: finetuned from the pretrained weights, or from random init.
    DATA=data/pop909_melchord_tagged_cp16_v2.pt
    need "$DATA" "Run 'prepare_data.sh train' or download the data."
    EXTRA=()
    if [[ "$SYSTEM" == ss_scratch ]]; then
        EXTRA+=(--from_scratch --run_tag "${RUN_TAG:-tagged}")
    else
        need "$BACKBONE" "Download the pretrained backbone (scripts/download.sh)."
        EXTRA+=(--run_tag "${RUN_TAG:-tagged}")
    fi
    [[ "${WANDB:-0}" == "1" ]] || EXTRA+=(--no_wandb)
    launch finetune_pop909.py --data "$DATA" --pretrained "$BACKBONE" \
        --batch_size "${BATCH_SIZE:-12}" --max_lr "${MAX_LR_SS:-1e-5}" \
        --lr_total_steps "${LR_TOTAL_STEPS:-2000}" --val_check_interval 25 \
        "${EXTRA[@]}"
    ;;
esac
