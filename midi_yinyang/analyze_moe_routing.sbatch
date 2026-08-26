#!/bin/bash
# SLURM (CPU): MoE router collapse analysis on a TRAINED checkpoint.
#
# Answers two questions MoERoutingMonitor cannot, because it logs mean
# expert PROBABILITY rather than load:
#
#   1. Is the router COLLAPSED? A router can show near-uniform mean
#      probs while its top-k always lands on the same experts -- the
#      mean is over a softmax that never has to commit. Collapse lives
#      in the argmax, so this reports per-expert top-1 LOAD, the worst
#      max_load, and any DEAD experts (never in any token's top-k, so
#      receiving no gradient and unable to recover).
#   2. Does it SPECIALIZE BY MODALITY? Reports the L1 distance between
#      the mod_a and mod_b routing distributions. ~0 means the same
#      experts serve both streams -- worth knowing, since the router has
#      no modality INPUT: it is a linear layer on the hidden state, and
#      token_type_embeddings is zeroed and frozen in the duet variants.
#
# A normalized entropy near 1.0 is the quiet failure: the router is
# uniform, every token gets the same blend, and the MoE is a dense FFN
# with E x the parameters. Load balance looks perfect precisely BECAUSE
# nothing is being decided -- which is why max_load alone is not enough.
#
# CPU only, so it will not compete with a training job for GPUs. One
# forward over a real batch; a few minutes at most.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT      REQUIRED. Run dir (best-val auto-resolved) or ckpt file.
#   VARIANT   REQUIRED. a1 | a2 | b1 | c1 | c2 -- picks the right
#             load_model AND the sequence layout used to split the
#             positions by modality. A wrong value mislabels that split.
#   TASK      REQUIRED. Must match what the ckpt was TRAINED on: the
#             batch comes from that task's own dataset.
#   PROBE     none (default) | identical | swap. The stamp-vs-content
#             probe: nothing in the ENCODING separates the streams (same
#             local encoder, zeroed type embeddings, constant programs on
#             melchord), but the trained per-modality attention
#             projections stamp slot parity onto the hidden before the
#             router reads it. 'identical' duplicates melody into the
#             chord slots -- any parity separation that survives IS the
#             stamp. 'swap' exchanges the contents -- whichever the
#             experts follow (slot or content) wins. Both run a baseline
#             pass first and print a per-layer comparison + verdict.
#             'within' is the WITHIN-STREAM content probe: per layer and
#             stream, does routing vary with the frame's note density /
#             register / mean duration, against a shuffle null? The
#             right content question for A.2.moe_permod ckpts (where
#             swap's slot/content dichotomy is undefined); also runs on
#             shared-router ckpts for a directly comparable baseline.
#   BATCH_SIZE     default 2 (one forward is plenty; this is not a
#                  statistical estimate over the corpus, it is a read of
#                  the router's behaviour).
#   SEED      default 0. Seeds the batch draw (songs + windows), so runs
#             on DIFFERENT ckpts with the same SEED analyze the same
#             batch -- required for fair cross-ckpt comparisons (e.g.
#             PROBE=within on mg vs baseline).
#   MAX_POLYPHONY  melchord ckpts are cp8; sets MELCHORD_CP so tasks.py
#                  resolves the matching data files.
#
# Example (A.2 on the combined melchord corpus):
#   sbatch --export=ALL,VARIANT=a2,TASK=melchord_pop909_nottingham,\
# CKPT=ckpt/m2c_duet_block_diffusion_v1.1_large_gnl12_K4_melchord_pop909_nottingham_batch_8_schedule/ \
#       midi_yinyang/analyze_moe_routing.sbatch

#SBATCH --job-name=moe_routing
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

CKPT="${CKPT:-}"
VARIANT="${VARIANT:-}"
TASK="${TASK:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_TOPK="${MOE_TOPK:-2}"
PROBE="${PROBE:-none}"
SEED="${SEED:-0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

for v in CKPT VARIANT TASK; do
    if [[ -z "${!v}" ]]; then
        echo "ERROR: $v is required."
        echo "  sbatch --export=ALL,VARIANT=a2,TASK=melchord_pop909_nottingham,\\"
        echo "CKPT=ckpt/<run>/ midi_yinyang/analyze_moe_routing.sbatch"
        exit 1
    fi
done
[[ -e "$CKPT" ]] || { echo "ERROR: ckpt not found: $CKPT"; exit 1; }

# Same cp rule as the trainers: the batch must be tokenized the way the
# ckpt was trained, and MELCHORD_CP is what selects the data file.
if [[ -z "${MAX_POLYPHONY:-}" ]]; then
    case "$TASK" in
        melchord*) MAX_POLYPHONY=8 ;;
        *)         MAX_POLYPHONY=16 ;;
    esac
fi
case "$TASK" in melchord*) export MELCHORD_CP="$MAX_POLYPHONY" ;; esac

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "MoE ROUTER COLLAPSE ANALYSIS"
echo "VARIANT=$VARIANT  TASK=$TASK  MAX_POLYPHONY=$MAX_POLYPHONY  PROBE=$PROBE"
echo "CKPT=$CKPT"
echo "================================================================"

python analyze_moe_routing.py \
    --variant "$VARIANT" \
    --ckpt "$CKPT" \
    --task "$TASK" \
    --batch-size "$BATCH_SIZE" \
    --moe-num-experts "$MOE_NUM_EXPERTS" \
    --moe-topk "$MOE_TOPK" \
    --probe "$PROBE" \
    --seed "$SEED"

echo "================================================================"
echo "Done. Read the VERDICT block above."
echo "  dead experts / max_load high -> raise aux_loss_weight"
echo "  entropy ~1.0                 -> router is uniform; the MoE is a"
echo "                                  dense FFN with ${MOE_NUM_EXPERTS}x params"
echo "  mod L1 ~0                    -> experts do not separate the streams"
echo "================================================================"
