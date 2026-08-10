#!/bin/bash
# SLURM: is B.1's chord stream dying because of the MECHANISM or a BUG?
#
# Symptom: in co-generation the chord (follower) thins out and stops well
# before the end -- E1 measured survival_b = 0.580 and density_drift_b =
# -0.132, the most negative drift in the table. The output is NOT
# truncated: co_generate returns gen_length frames for both streams with
# an equal number generated (verified), so the content is decaying, not
# being cut.
#
# The hypothesis this script tests: under `co` the follower harmonizes
# against a SAMPLED leader future, so drift in the leader compounds into
# the follower. Under `drum2nondrum` the leader is GIVEN (real), so the
# follower harmonizes against a true future.
#
#   follower healthy with a real leader, sick with a sampled one
#       -> compounding drift; a property of B.1 under co-generation,
#          and a finding to report rather than a bug to fix.
#   follower sick in BOTH
#       -> not about the leader; look at the model or the decode path.
#
# Both modes are run on the same songs, same prompt, same gen length, so
# the only thing that differs is where the leader comes from.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT_B1       B.1 run dir or ckpt file  (required)
#   MEL_FOLDER    default input/pop909_split/melody   (leader)
#   CHORD_FOLDER  default input/pop909_split/chord    (follower)
#   PROMPT_LENGTH default 64  (frames; 16 frames/bar, so 64 = 4 bars)
#   GEN_LENGTH    default 384 (TOTAL frames incl. prompt = 24 bars)
#   N_SAMPLES     default 3
#   OUT_ROOT      default temp/b1_follower_<JOBID>
#
# Example:
#   sbatch --export=ALL,CKPT_B1=ckpt/m2c_duet_anticipatory_v1.0_large_gnl12_melchord_pop909_nottingham_k16_batch_16_schedule/ \
#       midi_yinyang/diagnose_b1_follower.sbatch

#SBATCH --job-name=b1_follower_diag
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=6:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

CKPT_B1="${CKPT_B1:-}"
MEL_FOLDER="${MEL_FOLDER:-input/pop909_split/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-input/pop909_split/chord}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GEN_LENGTH="${GEN_LENGTH:-384}"
N_SAMPLES="${N_SAMPLES:-3}"
# Melchord polyphony budget: cp8 is the convention (cp4 retired). Must
# match the ckpt's training data -- pass MAX_POLYPHONY=4 to diagnose the
# pre-migration cp4-trained checkpoints.
MAX_POLYPHONY="${MAX_POLYPHONY:-8}"
# Anticipation override. k is auto-detected from the ckpt path pattern
# _k<N>_, with DEFAULT 16 when absent -- so pointing this diagnostic at
# an A.1 checkpoint (whose dir has no _k<N>_) would silently run A.1
# with a k=16 shift it was never trained for. A.1 IS B.1 with k=0 and
# their state dicts are identical, so ANTICIPATION_FRAMES=0 makes this
# script the architecture-matched control run: same decode, same songs,
# only the shift differs.
ANTICIPATION_FRAMES="${ANTICIPATION_FRAMES:-}"
OUT_ROOT="${OUT_ROOT:-temp/b1_follower_${SLURM_JOB_ID:-local}}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs "$OUT_ROOT" results

if [[ -z "$CKPT_B1" ]]; then
    echo "ERROR: CKPT_B1 is required (B.1 run dir or ckpt file)."
    exit 1
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "B.1 FOLLOWER-DECAY DIAGNOSTIC"
echo "CKPT_B1  = $CKPT_B1"
echo "ANTICIPATION_FRAMES=${ANTICIPATION_FRAMES:-<auto from ckpt path; default 16>}"
echo "prompts  = $MEL_FOLDER (leader) / $CHORD_FOLDER (follower)"
echo "PROMPT_LENGTH=$PROMPT_LENGTH ($((PROMPT_LENGTH / 16)) bars)  "\
"GEN_LENGTH=$GEN_LENGTH ($((GEN_LENGTH / 16)) bars)  N_SAMPLES=$N_SAMPLES"
echo "OUT_ROOT = $OUT_ROOT"
echo "================================================================"
nvidia-smi | head -n 12 || true

# GPU pre-flight: both generation stages call model.cuda(); fail here with
# a clear cause rather than deep inside a traceback after taking the slot.
if ! python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"; then
    echo "ERROR: no usable CUDA GPU."
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "  CAUSE: not submitted via sbatch, so --gres never applied."
    else
        echo "  CAUSE: node $(hostname) has no working GPU; resubmit with"
        echo "         sbatch --exclude=$(hostname) ..."
    fi
    exit 1
fi

python check_ckpts.py "$CKPT_B1" || {
    echo "ERROR: B.1 checkpoint(s) unreadable -- see above."; exit 1; }

for d in "$MEL_FOLDER" "$CHORD_FOLDER"; do
    [[ -d "$d" ]] || { echo "ERROR: $d not found"; exit 1; }
done
N_MEL=$(ls "$MEL_FOLDER"/*.mid "$MEL_FOLDER"/*.MID 2>/dev/null | wc -l)
N_CHD=$(ls "$CHORD_FOLDER"/*.mid "$CHORD_FOLDER"/*.MID 2>/dev/null | wc -l)
if [[ "$N_MEL" -eq 0 || "$N_MEL" -ne "$N_CHD" ]]; then
    echo "ERROR: leader/follower folders must hold the SAME songs "\
"(got $N_MEL vs $N_CHD) -- scoring pairs them by filename."
    exit 1
fi
SONG_IDS=$(ls "$MEL_FOLDER" | sed 's/\.[Mm][Ii][Dd]$//' | tr '\n' ' ')
echo "[prompts] $N_MEL songs: $SONG_IDS"

CO_OUT="$OUT_ROOT/co"
CD_OUT="$OUT_ROOT/drum2nondrum"

# ---------------------------------------------------------------------
# 1. co-generation: leader is SAMPLED past the prompt
# ---------------------------------------------------------------------
echo "--- [1/2] mode=co (leader sampled) ---"
python cp_transformer_m2c_duet_anticipatory_inference.py \
    --ckpt "$CKPT_B1" --mode co \
    --drum-folder "$MEL_FOLDER" --nondrum-folder "$CHORD_FOLDER" \
    --output-dir "$CO_OUT" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony "$MAX_POLYPHONY" \
    --model-size large --moe-num-experts 4 --moe-topk 2 \
    ${ANTICIPATION_FRAMES:+--anticipation-frames "$ANTICIPATION_FRAMES"}

# ---------------------------------------------------------------------
# 2. conditional: leader is GIVEN (real) for the whole piece
# ---------------------------------------------------------------------
echo "--- [2/2] mode=drum2nondrum (leader given) ---"
python cp_transformer_m2c_duet_anticipatory_inference.py \
    --ckpt "$CKPT_B1" --mode drum2nondrum \
    --drum-folder "$MEL_FOLDER" --nondrum-folder "$CHORD_FOLDER" \
    --output-dir "$CD_OUT" \
    --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH" \
    --temperature 1.0 --n-samples "$N_SAMPLES" --max-polyphony "$MAX_POLYPHONY" \
    --model-size large --moe-num-experts 4 --moe-topk 2 \
    ${ANTICIPATION_FRAMES:+--anticipation-frames "$ANTICIPATION_FRAMES"}

# ---------------------------------------------------------------------
# 3. score both
# ---------------------------------------------------------------------
# Layout depends on sample count: >1 writes <song>/<mode>/sample_i_temp<T>.mid
# ('duet_multi', mode taken from the dir name) for BOTH modes; at 1 sample
# co writes <song>/co.mid ('duet') and drum2nondrum writes
# <song>/drum2nondrum_temp<T>.mid ('cond').
if [[ "$N_SAMPLES" -gt 1 ]]; then
    CO_LAYOUT=duet_multi; CD_LAYOUT=duet_multi
else
    CO_LAYOUT=duet;       CD_LAYOUT=cond
fi

MANIFEST="results/b1_follower_${SLURM_JOB_ID:-local}_manifest.tsv"
METRICS="results/b1_follower_${SLURM_JOB_ID:-local}_metrics.csv"

python build_eval_manifest.py \
    --source "B.1:${CO_LAYOUT}:${CO_OUT}" \
    --source "B.1:${CD_LAYOUT}:${CD_OUT}" \
    --songs $SONG_IDS \
    --modes co drum2nondrum \
    --out "$MANIFEST"

python eval_metrics.py --task melchord \
    --manifest "$MANIFEST" \
    --ref-a-dir "$MEL_FOLDER" --ref-b-dir "$CHORD_FOLDER" \
    --prompt-frames "$PROMPT_LENGTH" --total-frames "$GEN_LENGTH" \
    --out "$METRICS"

# ---------------------------------------------------------------------
# 4. verdict: follower health, sampled leader vs given leader
# ---------------------------------------------------------------------
python - "$METRICS" <<'PY'
import csv, math, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
# follower-health metrics; survival/density near reference = healthy,
# empty_rate high = silent, density_drift negative = fading out.
KEYS = ['survival_b', 'density_ratio_b', 'density_drift_b',
        'empty_rate_b', 'survival_a']
acc = defaultdict(lambda: defaultdict(list))
for r in rows:
    for k in KEYS:
        try:
            v = float(r.get(k, ''))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            acc[r.get('mode', '?')][k].append(v)

modes = [m for m in ('co', 'drum2nondrum') if m in acc]
print('\n' + '=' * 64)
print('FOLLOWER HEALTH: sampled leader (co) vs given leader (drum2nondrum)')
print('=' * 64)
print(f'{"metric":<20}' + ''.join(f'{m:>16}' for m in modes) + f'{"delta":>12}')
print('-' * 64)
for k in KEYS:
    vals = [sum(acc[m][k]) / len(acc[m][k]) if acc[m].get(k) else None
            for m in modes]
    cells = ''.join(f'{v:>16.4f}' if v is not None else f'{"--":>16}'
                    for v in vals)
    d = (f'{vals[1] - vals[0]:>+12.4f}'
         if len(vals) == 2 and None not in vals else f'{"--":>12}')
    print(f'{k:<20}{cells}{d}')

print("""
Reading it:
  survival_b much HIGHER with a given leader -> the follower is fine when
    it harmonizes against a REAL future, and decays only when the leader
    is sampled. That is compounding drift: a property of B.1 under
    co-generation, to report rather than fix.
  survival_b LOW in BOTH -> the leader is not the cause; the follower
    decays regardless, so look at the model or the decode path.
  density_drift_b is the direct 'fading out' signal: strongly negative
    means the follower thins as generation proceeds.""")
PY

echo "================================================================"
echo "Done."
echo "  midis   : $CO_OUT (co), $CD_OUT (drum2nondrum)"
echo "  metrics : $METRICS"
echo "Listen to a co/drum2nondrum pair for the same song to hear the"
echo "difference the leader makes."
