#!/bin/bash
# SLURM (CPU): T7 of the external-baseline plan (EXPERIMENTS.md 2.6).
#
# AccoMontage2 (Yi, Hu, Zhao & Xia, ISMIR'22, MIT) is the melody-given
# harmonization baseline: melody midi + phrase segmentation string
# (e.g. 'A8B8A8B8') + tonic key in, chord_gen.mid (melody+chords) out.
# The documented interface is a GUI, but the repo exposes a scriptable
# path (demo.py; chorderator.generate_save(); Core class in
# chorderator/core.py) -- this job clones it, installs deps into a venv
# inheriting the mitosis env, and VERIFIES the headless path by
# importing chorderator and introspecting the generate/save API.
#
# It also reports the input requirements we must supply per test song
# (phrase string + key), which decides whether the POP909
# phrase-annotation dataset needs to be staged next.
#
# Report-oriented: failures print and the job continues to the verdict.
#
# Example:
#   sbatch midi_yinyang/setup_accomontage2.sbatch

#SBATCH --job-name=setup_accomontage2
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=02:00:00

set -uo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
EXT="$REPO_DIR/midi_yinyang/external"
AM="$EXT/accomontage2"

conda activate "$CONDA_ENV"
mkdir -p "$EXT" /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "T7: AccoMontage2 setup + headless-path verification"
echo "================================================================"

if [[ -d "$AM/.git" ]]; then
    echo "[clone] already present; pulling latest"
    git -C "$AM" pull --ff-only || echo "[clone] pull failed; using existing tree"
else
    git clone https://github.com/billyblu2000/accomontage2 "$AM" \
        || { echo "[FATAL] clone failed -- no network from this node?"; exit 1; }
fi
echo "[clone] HEAD: $(git -C "$AM" log -1 --format='%h %s' 2>/dev/null)"

if [[ ! -d "$AM/.venv" ]]; then
    python -m venv --system-site-packages "$AM/.venv"
fi
# shellcheck disable=SC1091
source "$AM/.venv/bin/activate"
REQ=$(find "$AM" -maxdepth 2 -iname 'requirements*.txt' | head -1)
if [[ -n "$REQ" ]]; then
    echo "[deps] installing from $REQ"
    pip install --quiet -r "$REQ" || echo "[deps] WARNING: some installs failed (see above)"
else
    echo "[deps] no requirements file found"
fi

echo "----------------------------------------------------------------"
echo "[headless] demo.py entry (first 60 lines):"
sed -n 1,60p "$AM/demo.py" 2>/dev/null || echo "  demo.py not found"
echo "----------------------------------------------------------------"
echo "[headless] import + API introspection:"
cd "$AM"
python - <<'PY'
import inspect, sys
try:
    import chorderator as cdt
except Exception as e:                                   # noqa: BLE001
    print(f'[FAIL] import chorderator: {e!r}')
    sys.exit(0)
print('[ok ] import chorderator')
for name in ('set_melody', 'set_phrase', 'set_segmentation', 'set_meta',
             'set_output_style', 'generate', 'generate_save', 'set_key',
             'set_mode', 'set_meter'):
    fn = getattr(cdt, name, None)
    if fn is None:
        continue
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        sig = '(...)'
    print(f'  chorderator.{name}{sig}')
try:
    from chorderator.core import Core
    print('[ok ] chorderator.core.Core importable; methods:')
    for m in sorted(dir(Core)):
        if not m.startswith('_'):
            print(f'    Core.{m}')
except Exception as e:                                   # noqa: BLE001
    print(f'[warn] Core import: {e!r}')
PY

echo "----------------------------------------------------------------"
echo "[inputs] what each test melody needs (from the repo docs):"
grep -rn "segmentation\|A8B8\|tonic\|set_key\|set_meta" "$AM"/*.py "$AM"/README* 2>/dev/null | head -15
echo "----------------------------------------------------------------"
echo "[data] bundled dataset/resources:"
du -sh "$AM" 2>/dev/null
find "$AM" -maxdepth 2 -type d | sed "s|$AM|.|" | head -25

echo "================================================================"
echo "VERDICT: if 'import chorderator' printed [ok] and generate/save"
echo "functions listed above, the headless path (T8) is GO. Remaining"
echo "per-song inputs: melody midi (we have), phrase segmentation"
echo "string, tonic key -- stage POP909 phrase annotations next if the"
echo "grep above confirms the segmentation string is mandatory."
echo "================================================================"
