#!/bin/bash
# Co-generation with ONE duet checkpoint over MANY GPUs: the prompt songs
# are split round-robin into one shard per GPU and each shard is decoded
# by its own process, all writing into the same output tree. Same
# staging, flags and decode settings as eval_e1's duet arm, so the tree
# it leaves can be scored there (layout <song>/co/sample_<i>_temp1.0.mid).
#
# Runs under SLURM on the cluster (--gres=gpu:N picks the lane count) or
# as a plain `bash` script on a machine without SLURM, where the #SBATCH
# lines are comments and the lane count comes from CUDA_VISIBLE_DEVICES
# or, failing that, from every GPU torch can see.
#
# Env knobs (--export=ALL,KEY=val under sbatch, KEY=val bash ... elsewhere):
#   CKPT           REQUIRED: a checkpoint FILE, or a run directory
#                  (resolve_best_ckpt then picks the best-val file in it)
#   SYSTEM         name of the output folder under OUT_ROOT (default D1)
#   OUT_ROOT       default temp/E1_v5_multigpu; prompts are staged under
#                  OUT_ROOT/prompts/{mel,chord} exactly as eval_e1 does,
#                  samples go to OUT_ROOT/SYSTEM/<song>/co/
#   MEL_FOLDER / CHORD_FOLDER   prompt sources; default the held-out v5
#                  set input/heldout_v5/{melody,chord} (not in git: copy
#                  it from the cluster or download it, see below)
#   DUET_CHORD_PROGRAM  midi program the staged chord prompts are re-tagged
#                  to (default 0: every checkpoint trained on the untagged
#                  corpus, which includes the *_melchord_long family, saw
#                  chord program 0; heldout_v5 as built carries 48). Empty
#                  leaves the prompts untouched; 48 for a *_tagged ckpt.
#   MAX_POLYPHONY  prompt tokenisation cap; MUST match the training data
#                  (default 4: the melchord_cp4tar and melchord_long runs;
#                  8 for the *_cp8long runs). Not recoverable from the
#                  checkpoint -- a mismatch decodes garbage without error.
#                  Confirm from the training log's "Data for dataset"
#                  line, which names the .pt file and its cp.
#   MOE_NUM_EXPERTS / MOE_TOPK / MOE_INTERMEDIATE_SIZE   model shape;
#                  default 4 / 2 / <unset> (Duet). The dense ablation arm
#                  is 1 / 1 / 6144.
#   PROMPT_LENGTH / GEN_LENGTH / N_SAMPLES   default 96 / 416 / 3 (E1)
#   A3_SCHEDULE / A3_REFINE_STEPS / A3_FINAL_TEMP / A3_TOP_P
#                  decode; default ctc_alt / 4 / 1.0 / 1.0, the paper's
#                  draft-and-denoise decode (eval_e1's A3ctcaT arm).
#                  A3_ADAPTIVE and A3_CTC_DENOISE_LEADER pass through.
#   NUM_GPUS       lanes; default: the allocation (SLURM), else the
#                  CUDA_VISIBLE_DEVICES list, else torch's device count.
#   GPU_IDS        explicit device list, e.g. "0 1 2 3" (overrides NUM_GPUS)
#   MAX_SONGS      decode only the first N staged songs (smoke test)
#   RESTAGE=1      re-copy the prompts even if OUT_ROOT/prompts/.ready exists
#   DRY_RUN=1      stage, shard, print the per-lane commands, run nothing
#   CONDA_ENV      default mitosis; CONDA_ENV= (empty) skips activation
#   REPO_DIR       default: the submit/current directory, which must hold
#                  midi_yinyang/ (or the cluster checkout)
#
# Re-running resumes: every lane passes --skip-existing, so a finished
# sample is never redrawn. The job fails if any lane fails or the tree
# is incomplete at the end, so a chained score step does not run on a
# partial set.
#
# Cluster, dense arm on the held-out v5 prompts with 4 GPUs:
#   sbatch --gres=gpu:4 --exclude=gpu-51,gpu-05,gpu-54 --export=ALL,\
#   CKPT=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_D1A3_melchord_long_batch_8_schedule,\
#   MOE_NUM_EXPERTS=1,MOE_TOPK=1,MOE_INTERMEDIATE_SIZE=6144,SYSTEM=D1,OUT_ROOT=temp/E1_v5_dense \
#       midi_yinyang/infer_duet_multigpu.sbatch
# Machine without SLURM, 8 GPUs, checkpoint downloaded by download_hf.sbatch:
#   CKPT=ckpt/<file>.ckpt MOE_NUM_EXPERTS=1 MOE_TOPK=1 MOE_INTERMEDIATE_SIZE=6144 \
#   SYSTEM=D1 OUT_ROOT=temp/E1_v5_dense CONDA_ENV= \
#       bash midi_yinyang/infer_duet_multigpu.sbatch 2>&1 | tee infer_dense.log

#SBATCH --job-name=infer_duet_multigpu
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=24:00:00

set -euo pipefail

# ---- where we are ------------------------------------------------------
REPO_DIR="${REPO_DIR:-}"
if [[ -z "$REPO_DIR" ]]; then
    for cand in "${SLURM_SUBMIT_DIR:-}" "$PWD" /home/xinyue.li/Mitosis; do
        [[ -n "$cand" && -d "$cand/midi_yinyang" ]] && { REPO_DIR="$cand"; break; }
    done
fi
[[ -n "$REPO_DIR" && -d "$REPO_DIR/midi_yinyang" ]] || {
    echo "ERROR: no midi_yinyang/ under the submit/current directory; run from the repo root or set REPO_DIR"; exit 1; }
# conda env: skipped when CONDA_ENV is empty or that env is already the
# active one (a machine whose ~/.bashrc has no conda hook, run from an
# activated shell); otherwise activated through the hook of the conda
# on PATH, not ~/.bashrc, so 'conda init' is not required.
CONDA_ENV="${CONDA_ENV-mitosis}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH; activate the env yourself and pass CONDA_ENV="; exit 1; }
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
cd "$REPO_DIR/midi_yinyang"
# the cluster log dir (its #SBATCH --output); nothing to do elsewhere
mkdir -p /home/xinyue.li/logs 2>/dev/null || true

# ---- knobs ---------------------------------------------------------------
CKPT="${CKPT:?set CKPT=<checkpoint file or run dir>}"
SYSTEM="${SYSTEM:-D1}"
OUT_ROOT="${OUT_ROOT:-temp/E1_v5_multigpu}"
MEL_FOLDER="${MEL_FOLDER:-input/heldout_v5/melody}"
CHORD_FOLDER="${CHORD_FOLDER:-input/heldout_v5/chord}"
DUET_CHORD_PROGRAM="${DUET_CHORD_PROGRAM-0}"
MAX_POLYPHONY="${MAX_POLYPHONY:-4}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_TOPK="${MOE_TOPK:-2}"
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-}"
PROMPT_LENGTH="${PROMPT_LENGTH:-96}"
GEN_LENGTH="${GEN_LENGTH:-416}"
N_SAMPLES="${N_SAMPLES:-3}"
MAX_SONGS="${MAX_SONGS:-}"
RESTAGE="${RESTAGE:-0}"
DRY_RUN="${DRY_RUN:-0}"
export A3_SCHEDULE="${A3_SCHEDULE:-ctc_alt}"
export A3_REFINE_STEPS="${A3_REFINE_STEPS:-4}"
export A3_FINAL_TEMP="${A3_FINAL_TEMP:-1.0}"
export A3_TOP_P="${A3_TOP_P:-1.0}"

[[ "$OUT_ROOT" == temp/* ]] || { echo "ERROR: OUT_ROOT must be under temp/ (got $OUT_ROOT)"; exit 1; }
[[ -e "$CKPT" ]] || { echo "ERROR: CKPT=$CKPT does not exist"; exit 1; }
[[ -d "$MEL_FOLDER" && -d "$CHORD_FOLDER" ]] || {
    echo "ERROR: prompt folders missing: $MEL_FOLDER / $CHORD_FOLDER"
    echo "       input/ is not in git. Copy the held-out set from the machine that has it, e.g."
    echo "       rsync -av <host>:/home/xinyue.li/Mitosis/midi_yinyang/input/heldout_v5/ midi_yinyang/input/heldout_v5/"
    echo "       or fetch it from the Hub with download_hf.sbatch (INCLUDE='heldout_v5/*' DEST=input)."
    exit 1; }

# ---- GPUs ------------------------------------------------------------------
# Lane devices: an explicit GPU_IDS list; else the ids CUDA_VISIBLE_DEVICES
# names (a SLURM allocation sets it); else 0..N-1 for N from the
# allocation or from torch. No nvidia-smi: the cluster's submit filter
# rejects it.
if [[ -n "${GPU_IDS:-}" ]]; then
    read -r -a GPUS <<< "$GPU_IDS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
    [[ -n "${NUM_GPUS:-}" ]] && GPUS=("${GPUS[@]:0:$NUM_GPUS}")
else
    n="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
    if [[ -z "$n" ]]; then
        n=$(python -c "import torch; print(torch.cuda.device_count())")
    fi
    [[ "$n" -ge 1 ]] 2>/dev/null || { echo "ERROR: no GPU found (torch.cuda.device_count()=$n)"; exit 1; }
    GPUS=($(seq 0 $((n - 1))))
fi
NUM_LANES=${#GPUS[@]}
if [[ "$DRY_RUN" != "1" ]]; then
    python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch.cuda.is_available() == False')
torch.zeros(1).cuda()
print(f'[gpu] {torch.cuda.device_count()} visible; {torch.cuda.get_device_name(0)}')
"
    python check_ckpts.py "$CKPT"
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "MULTI-GPU CO-GENERATION  system=$SYSTEM  ckpt=$CKPT"
echo "lanes=$NUM_LANES on GPUs [${GPUS[*]}]"
echo "prompts: $MEL_FOLDER / $CHORD_FOLDER  chord program -> ${DUET_CHORD_PROGRAM:-<untouched>}"
echo "PROMPT_LENGTH=$PROMPT_LENGTH  GEN_LENGTH=$GEN_LENGTH  N_SAMPLES=$N_SAMPLES  cp=$MAX_POLYPHONY"
echo "model: experts=$MOE_NUM_EXPERTS topk=$MOE_TOPK inter=${MOE_INTERMEDIATE_SIZE:-<default>}"
echo "decode: schedule=$A3_SCHEDULE refine=$A3_REFINE_STEPS temp=$A3_FINAL_TEMP top_p=$A3_TOP_P adaptive=${A3_ADAPTIVE:-<unset>} denoise_leader=${A3_CTC_DENOISE_LEADER:-<default 1>}"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================================================"

# ---- 1. prompts, staged once (eval_e1's layout and re-tagging) ------------
EVAL_MEL="$OUT_ROOT/prompts/mel"
EVAL_CHORD="$OUT_ROOT/prompts/chord"
READY="$OUT_ROOT/prompts/.ready"
[[ "$RESTAGE" == "1" ]] && rm -rf "$OUT_ROOT/prompts"
if [[ -f "$READY" ]]; then
    echo "[prompts] reusing staged prompts ($READY present; RESTAGE=1 to restage)"
else
    mkdir -p "$EVAL_MEL" "$EVAL_CHORD"
    cp -f "$MEL_FOLDER"/*.mid "$MEL_FOLDER"/*.MID "$EVAL_MEL"/ 2>/dev/null || true
    cp -f "$CHORD_FOLDER"/*.mid "$CHORD_FOLDER"/*.MID "$EVAL_CHORD"/ 2>/dev/null || true
    if [[ -n "$DUET_CHORD_PROGRAM" ]]; then
        python - "$EVAL_CHORD" "$DUET_CHORD_PROGRAM" <<'RETAG'
import glob, os, sys
import mido
folder, prog = sys.argv[1], int(sys.argv[2])
changed = 0
for f in sorted(glob.glob(os.path.join(folder, '*.mid'))):
    m = mido.MidiFile(f)
    hit, seen = False, 0
    for tr in m.tracks:
        for msg in tr:
            if msg.type == 'program_change':
                if msg.program != prog:
                    msg.program = prog
                    hit = True
                seen += 1
    if not seen and prog != 0:
        m.tracks[-1].insert(0, mido.Message('program_change', program=prog, time=0))
        hit = True
    if hit:
        m.save(f)
        changed += 1
print(f'[prompts] chord program -> {prog}: rewrote {changed} file(s)')
RETAG
    fi
    touch "$READY"
fi
N_MEL=$(ls "$EVAL_MEL" | wc -l); N_CHORD=$(ls "$EVAL_CHORD" | wc -l)
[[ "$N_MEL" -gt 0 ]] || { echo "ERROR: no midis staged from $MEL_FOLDER"; exit 1; }
[[ "$N_MEL" -eq "$N_CHORD" ]] || { echo "ERROR: $N_MEL melody vs $N_CHORD chord prompts; the pairing is by file name"; exit 1; }
SONGS=$(ls "$EVAL_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | sort)
[[ -n "$MAX_SONGS" ]] && SONGS=$(echo "$SONGS" | head -n "$MAX_SONGS")
N_SONGS=$(echo "$SONGS" | grep -c .)
echo "[prompts] $N_SONGS songs"

# ---- 2. shards: one symlink folder pair per lane, round-robin -------------
SHARDS="$OUT_ROOT/shards/$SYSTEM"
rm -rf "$SHARDS"
ABS_MEL=$(cd "$EVAL_MEL" && pwd); ABS_CHORD=$(cd "$EVAL_CHORD" && pwd)
i=0
for sid in $SONGS; do
    lane=$((i % NUM_LANES))
    mkdir -p "$SHARDS/lane_$lane/mel" "$SHARDS/lane_$lane/chord"
    ln -sf "$ABS_MEL/$sid.mid" "$SHARDS/lane_$lane/mel/$sid.mid"
    ln -sf "$ABS_CHORD/$sid.mid" "$SHARDS/lane_$lane/chord/$sid.mid"
    i=$((i + 1))
done
for ((lane = 0; lane < NUM_LANES; lane++)); do
    [[ -d "$SHARDS/lane_$lane/mel" ]] || continue
    echo "[shard] lane $lane (GPU ${GPUS[$lane]}): $(ls "$SHARDS/lane_$lane/mel" | wc -l) songs"
done

# ---- 3. one decoder per GPU, all into OUT_ROOT/SYSTEM ---------------------
OUT_DIR="$OUT_ROOT/$SYSTEM"
LOGS="$OUT_ROOT/logs"
mkdir -p "$OUT_DIR" "$LOGS"
EXTRA=()
[[ -n "$MOE_INTERMEDIATE_SIZE" ]] && EXTRA=(--moe-intermediate-size "$MOE_INTERMEDIATE_SIZE")
# a run directory needs the trailing slash resolve_best_ckpt expects; a file is passed as is
CKPT_ARG="$CKPT"; [[ -d "$CKPT" ]] && CKPT_ARG="${CKPT%/}/"

declare -a PIDS=() LANES=()
for ((lane = 0; lane < NUM_LANES; lane++)); do
    [[ -d "$SHARDS/lane_$lane/mel" ]] || continue
    gpu="${GPUS[$lane]}"
    log="$LOGS/${SYSTEM}_lane${lane}_gpu${gpu}.log"
    cmd=(python cp_transformer_m2c_duet_block_diffusion_combined.py
         --ckpt "$CKPT_ARG"
         --mel-folder "$SHARDS/lane_$lane/mel" --chord-folder "$SHARDS/lane_$lane/chord"
         --output-dir "$OUT_DIR" --modes co
         --prompt-length "$PROMPT_LENGTH" --gen-length "$GEN_LENGTH"
         --temperature 1.0 --n-samples "$N_SAMPLES"
         --max-polyphony "$MAX_POLYPHONY" --skip-existing
         --model-size large --moe-num-experts "$MOE_NUM_EXPERTS" --moe-topk "$MOE_TOPK"
         ${EXTRA[@]+"${EXTRA[@]}"} --min-chord-tokens-before-eos 0)
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}  > $log"
        continue
    fi
    echo "[lane $lane] GPU $gpu -> $log"
    # per-lane routing-stats file, if the caller asked for one
    stats_env=()
    [[ -n "${MOE_ROUTING_STATS:-}" ]] && stats_env=(MOE_ROUTING_STATS="${MOE_ROUTING_STATS%.json}_lane${lane}.json")
    CUDA_VISIBLE_DEVICES="$gpu" env ${stats_env[@]+"${stats_env[@]}"} "${cmd[@]}" > "$log" 2>&1 &
    PIDS+=($!); LANES+=("$lane")
done
[[ "$DRY_RUN" == "1" ]] && { echo "[dry-run] nothing launched"; exit 0; }

FAILED=0
for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
        echo "[lane ${LANES[$idx]}] finished"
    else
        echo "[lane ${LANES[$idx]}] FAILED (see $LOGS/${SYSTEM}_lane${LANES[$idx]}_gpu${GPUS[${LANES[$idx]}]}.log)"
        FAILED=$((FAILED + 1))
    fi
done

# ---- 4. completeness: every song has N_SAMPLES samples -------------------
MISSING=0
for sid in $SONGS; do
    n=$(ls "$OUT_DIR/$sid/co/"sample_*.mid 2>/dev/null | wc -l)
    [[ "$n" -ge "$N_SAMPLES" ]] || { echo "[incomplete] $sid: $n/$N_SAMPLES samples"; MISSING=$((MISSING + 1)); }
done
echo "================================================================"
echo "$SYSTEM: $((N_SONGS - MISSING))/$N_SONGS songs complete under $OUT_DIR  (lanes failed: $FAILED)"
echo "logs: $LOGS/${SYSTEM}_lane*.log"
[[ "$FAILED" -eq 0 && "$MISSING" -eq 0 ]] || { echo "re-run the same command to resume (--skip-existing keeps finished samples)"; exit 1; }
