#!/bin/bash
# SLURM (GPU): T3' -- whole-song-gen PROMPTED lead-sheet continuation
# (block (i) of the E1 external comparison; EXPERIMENTS.md 2.6).
#
# Drives wholesong_prompted.py inside the whole_song_gen venv: per test
# song, ground-truth form (key+phrase channels, oracle IN THE BASELINE'S
# FAVOR), 6-bar ground-truth prompts at the counterpoint and lead-sheet
# levels through their own create_canvas(prompt=...) inpainting path,
# their generation loop unchanged. Output: duet_multi layout
# <OUT_DIR>/<songid>/co/sample_<i>.mid with MELODY/CHORD tracks
# (programs 0/48) -- directly scoreable by the eval harness alongside
# our co decodes (same 6-bar prompt, frames 96..416 scored).
#
# Env knobs (--export=ALL,KEY=val):
#   PROMPT_BARS default 6        N_SAMPLES default 3
#   SONG_IDS    default "1 2 3 4 5"
#   OUT_DIR     default temp/wholesong_prompted
#
# Example:
#   sbatch midi_yinyang/infer_wholesong_prompted.sbatch

#SBATCH --job-name=infer_wholesong_prompted
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=12:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"
WS="$MY/external/whole_song_gen"

conda activate "$CONDA_ENV"
mkdir -p /home/xinyue.li/logs

PROMPT_BARS="${PROMPT_BARS:-6}"
N_SAMPLES="${N_SAMPLES:-3}"
SONG_IDS="${SONG_IDS:-1 2 3 4 5}"
OUT_DIR="${OUT_DIR:-$MY/temp/wholesong_prompted}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "T3': whole-song PROMPTED continuation"
echo "PROMPT_BARS=$PROMPT_BARS  N_SAMPLES=$N_SAMPLES  SONG_IDS=$SONG_IDS"
echo "OUT_DIR=$OUT_DIR"
echo "================================================================"

[[ -d "$WS/.venv" ]] || { echo "ERROR: run setup_wholesong.sbatch first"; exit 1; }
# shellcheck disable=SC1091
source "$WS/.venv/bin/activate"

# Flatten the doubly-nested zip payloads (idempotent; same block as the
# unconditional job -- a no-op once done).
shopt -s dotglob
for d in pretrained_models results_default data; do
    if [[ -d "$WS/$d/$d" ]]; then
        echo "[layout] flattening $d/$d -> $d"
        mv "$WS/$d/$d"/* "$WS/$d/"
        rmdir "$WS/$d/$d"
    fi
done
shopt -u dotglob

# torchvision: their losses/lpips.py imports it at module-import time
# (training-only LPIPS; inference never instantiates it). Install the
# wheel PAIRED to the venv's torch with --no-deps so torch itself is
# never touched or upgraded.
if ! python -c "import torchvision" 2>/dev/null; then
    TV=$(python - <<'PY'
import torch
maj = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
m = {'2.0': '0.15.2', '2.1': '0.16.2', '2.2': '0.17.2', '2.3': '0.18.1',
     '2.4': '0.19.1', '2.5': '0.20.1', '2.6': '0.21.0', '2.7': '0.22.0'}
print(m.get(maj, ''))
PY
)
    if [[ -z "$TV" ]]; then
        echo "ERROR: no torchvision pairing known for this torch version;"
        echo "       pip install a matching torchvision into $WS/.venv manually."
        exit 1
    fi
    echo "[deps] installing torchvision==$TV pillow (--no-deps; torch untouched)"
    pip install -q --no-deps "torchvision==$TV" pillow
    python -c "import torchvision; print('[deps] torchvision', torchvision.__version__)"
fi

cd "$WS"
export PYTHONPATH="$WS${PYTHONPATH:+:$PYTHONPATH}"
python "$MY/wholesong_prompted.py" \
    --song-ids $SONG_IDS \
    --prompt-bars "$PROMPT_BARS" \
    --n-samples "$N_SAMPLES" \
    --out-dir "$OUT_DIR"

echo "================================================================"
echo "Done. Outputs: $OUT_DIR/<songid>/co/sample_<i>.mid (duet_multi)."
echo "Score alongside our A3 co decodes with build_eval_manifest"
echo "(--source WS:duet_multi:$OUT_DIR) + eval_metrics, same window."
echo "================================================================"
