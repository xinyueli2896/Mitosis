#!/bin/bash
# SLURM (CPU): one-time setup for the Anticipatory Music Transformer
# external baseline (Thickstun et al., arXiv:2306.08620).
#
# Clones the package into midi_yinyang/external/anticipation, installs it
# into an isolated venv, and pre-downloads the pretrained weights so the
# GPU job never waits on the network (and works if compute nodes have no
# egress). Same shape as setup_wholesong.sbatch.
#
# The venv is isolated on purpose: the package pins its own transformers,
# and this repo runs a VENDORED transformers fork
# (transformers_roformer_moe) that must not be disturbed.
#
# Env knobs (--export=ALL,KEY=val):
#   AMT_DIR    checkout root (default midi_yinyang/external/anticipation)
#   MODEL      HF id to pre-fetch (default stanford-crfm/music-medium-800k;
#              music-small-800k and music-large-800k also exist)
#   FORCE=1    re-clone and re-install even if the venv is there
#
#   sbatch midi_yinyang/setup_amt.sbatch
#
# PORTABLE: this script is meant to run on a cluster that is NOT the one
# the rest of the repo assumes. It derives the repo root from where
# sbatch was invoked, does not require the `mitosis` conda env, and
# names no partition -- pass -p/-q yourself if your site needs them:
#   sbatch -p <partition> -q <qos> midi_yinyang/<this script>
# Logs land in the submission directory rather than ~/logs.

#SBATCH --job-name=setup_amt
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
# ~/.bashrc and conda are this-cluster conventions; tolerate their
# absence. Everything below runs out of the package's own venv.
[[ -f ~/.bashrc ]] && source ~/.bashrc || true

# Repo root: where sbatch was invoked, so a clone anywhere works.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f "$REPO_DIR/midi_yinyang/amt_prompted.py" ]]; then
    MY="$REPO_DIR/midi_yinyang"
elif [[ -f "$REPO_DIR/amt_prompted.py" ]]; then
    MY="$REPO_DIR"
else
    echo "ERROR: cannot find amt_prompted.py from REPO_DIR=$REPO_DIR."
    echo "       Submit from the repo root, or pass REPO_DIR=<clone path>."
    exit 1
fi
AMT_DIR="${AMT_DIR:-$MY/external/anticipation}"
MODEL="${MODEL:-stanford-crfm/music-medium-800k}"
FORCE="${FORCE:-0}"

mkdir -p "$MY/external"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "AMT_DIR=$AMT_DIR"
echo "MODEL=$MODEL"
echo "================================================================"

if [[ ! -d "$AMT_DIR/.git" ]]; then
    echo "[clone] https://github.com/jthickstun/anticipation"
    git clone https://github.com/jthickstun/anticipation "$AMT_DIR"
else
    echo "[clone] $AMT_DIR already present"
fi

if [[ "$FORCE" == "1" || ! -d "$AMT_DIR/.venv" ]]; then
    echo "[venv] creating $AMT_DIR/.venv"
    rm -rf "$AMT_DIR/.venv"
    python -m venv "$AMT_DIR/.venv"
    # shellcheck disable=SC1091
    source "$AMT_DIR/.venv/bin/activate"
    pip install --upgrade pip
    # torch first, so the package's own resolution does not pull a build
    # that disagrees with the driver on this cluster
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install "$AMT_DIR"
    pip install transformers accelerate mido pretty_midi
else
    echo "[venv] $AMT_DIR/.venv present (FORCE=1 to rebuild)"
    # shellcheck disable=SC1091
    source "$AMT_DIR/.venv/bin/activate"
fi

echo "[check] imports + vocab geometry"
python - <<'EOF'
import torch, transformers
from anticipation import ops
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.sample import generate, instr_logits
from anticipation.config import TIME_RESOLUTION, DELTA, MAX_INSTR, MAX_PITCH
from anticipation.vocab import NOTE_OFFSET
print(f'  torch {torch.__version__}  transformers {transformers.__version__}')
print(f'  TIME_RESOLUTION={TIME_RESOLUTION} (expect 100 = 10ms bins)')
print(f'  DELTA={DELTA}s anticipation interval')
print(f'  MAX_INSTR={MAX_INSTR} MAX_PITCH={MAX_PITCH} NOTE_OFFSET={NOTE_OFFSET}')
# amt_prompted patches sample.instr_logits; add_token must resolve it
# through the module global for that to take effect.
import inspect, anticipation.sample as s
src = inspect.getsource(s.add_token)
assert 'instr_logits(' in src, 'add_token no longer calls instr_logits'
print('  add_token calls instr_logits: patch hook OK')
EOF

echo "[fetch] $MODEL"
python - "$MODEL" <<'EOF'
import sys
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(sys.argv[1])
n = sum(p.numel() for p in m.parameters())
print(f'  cached; {n/1e6:.0f}M parameters')
EOF

echo "================================================================"
echo "Ready. Run the baseline with:"
echo "  sbatch midi_yinyang/infer_amt_prompted.sbatch"
echo "or as an E1 system:  SYSTEMS=\"... AMT\""
echo "================================================================"
