#!/bin/bash
# SLURM (CPU): T1+T2 of the external-baseline plan (EXPERIMENTS.md 2.6).
#
# T1: clone ZZWaang/whole-song-gen (ICLR'24 cascaded-diffusion whole-song
#     model, MIT) into external/whole_song_gen, set up a venv that
#     INHERITS the mitosis env's torch (--system-site-packages, so only
#     the repo's extra deps are installed), and attempt the checkpoint
#     download the repo documents.
# T2: report CHECKPOINT COVERAGE -- which of the four stage models
#     (frm=form, ctp=reduced lead sheet, lsh=lead sheet, acc=accomp.)
#     are present under results_default/. We need frm+ctp+lsh; acc is
#     NOT needed (accompaniment is out of scope). The repo releases only
#     a subset "sufficient for testing", so this verdict decides whether
#     local regeneration works or we fall back to released samples.
#     Also fetches the data/ payload: the prompted-continuation protocol
#     (T3') encodes prompts with the authors' own data pipeline, which
#     needs their processed POP909 data.
#
# The job is REPORT-ORIENTED: download failures do not kill it; every
# step prints what it found so the log is the deliverable.
#
# Example:
#   sbatch midi_yinyang/setup_wholesong.sbatch

#SBATCH --job-name=setup_wholesong
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=04:00:00

set -uo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
EXT="$REPO_DIR/midi_yinyang/external"
WS="$EXT/whole_song_gen"

conda activate "$CONDA_ENV"
mkdir -p "$EXT" /home/xinyue.li/logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "T1+T2: whole-song-gen setup + checkpoint coverage"
echo "================================================================"

# ---- T1a: clone --------------------------------------------------------
if [[ -d "$WS/.git" ]]; then
    echo "[clone] already present; pulling latest"
    git -C "$WS" pull --ff-only || echo "[clone] pull failed (offline tag?); using existing tree"
else
    git clone https://github.com/ZZWaang/whole-song-gen "$WS" \
        || { echo "[FATAL] clone failed -- no network from this node?"; exit 1; }
fi
echo "[clone] HEAD: $(git -C "$WS" log -1 --format='%h %s' 2>/dev/null)"

# ---- T1b: venv inheriting mitosis torch --------------------------------
if [[ ! -d "$WS/.venv" ]]; then
    python -m venv --system-site-packages "$WS/.venv"
fi
# shellcheck disable=SC1091
source "$WS/.venv/bin/activate"
REQ=$(find "$WS" -maxdepth 2 -iname 'requirements*.txt' | head -1)
if [[ -n "$REQ" ]]; then
    echo "[deps] installing from $REQ"
    pip install --quiet -r "$REQ" || echo "[deps] WARNING: some installs failed (see above)"
else
    echo "[deps] no requirements file found; listing imports the scripts use:"
    grep -rhoE '^(import|from) [a-zA-Z0-9_]+' "$WS"/*.py 2>/dev/null | sort -u | head -30
fi
# Force a CURRENT gdown: job 196788 failed because an old gdown in the
# venv rejected --fuzzy (the error went to the .err file). We now
# upgrade explicitly, extract the bare file ID ourselves (works on any
# gdown version), and show gdown's own output in this log.
pip install -q --upgrade gdown || true
echo "[deps] gdown version: $(gdown --version 2>&1 | head -1)"

# ---- T1c: checkpoint download ------------------------------------------
echo "----------------------------------------------------------------"
echo "[ckpt] looking for the documented download pointers..."
# Each download_link.txt sits in the directory its payload belongs to
# (pretrained_models/, results_default/, data/). data/ WAS skipped when
# only unconditional inference was planned; the prompted-continuation
# protocol (2.6 T3', decision (c) rev. 2026-09-01) encodes our prompt
# bars with THEIR data pipeline (read_pop909_data + tonal_reduction),
# which reads the processed POP909 payload under data/ -- so fetch it.
for f in $(find "$WS" -maxdepth 3 -iname '*download*' -type f); do
    tgt=$(dirname "$f")
    echo "[ckpt] ---- $f  (target: $tgt) ----"
    cat "$f"; echo
    grep -oE 'https://drive\.google\.com[^ )"[:space:]]*' "$f" | while read -r url; do
        id=$(echo "$url" | sed -E 's#.*/d/([A-Za-z0-9_-]+).*#\1#')
        if [[ "$id" == "$url" ]]; then
            id=$(echo "$url" | grep -oE 'id=[A-Za-z0-9_-]+' | cut -d= -f2)
        fi
        echo "[ckpt] file id: $id"
        out="$tgt/gdown_$id.bin"
        if [[ -s "$out" ]]; then
            echo "[ckpt] already downloaded: $out ($(du -h "$out" | cut -f1))"
        else
            gdown "$id" -O "$out" 2>&1 || { echo "[ckpt] FAILED for $id"; rm -f "$out"; continue; }
        fi
        # Unzip archives into the target directory.
        case "$(file -b "$out" 2>/dev/null)" in
            Zip*)  echo "[ckpt] unzipping into $tgt"; unzip -o -q "$out" -d "$tgt" ;;
            gzip*) echo "[ckpt] untarring into $tgt"; tar xzf "$out" -C "$tgt" ;;
            *)     echo "[ckpt] payload type: $(file -b "$out" 2>/dev/null)" ;;
        esac
    done
done

# ---- T2: coverage verdict ----------------------------------------------
echo "================================================================"
echo "T2 CHECKPOINT COVERAGE (need frm + ctp + lsh; acc NOT needed)"
echo "================================================================"
MISSING=0
for stage in frm ctp lsh acc; do
    hit=$(find "$WS" -maxdepth 4 -type d -name "${stage}*" | head -1)
    n=$( [[ -n "$hit" ]] && find "$hit" -type f \( -name '*.pt' -o -name '*.ckpt' -o -name '*.pth' \) | wc -l || echo 0 )
    if [[ -n "$hit" && "$n" -gt 0 ]]; then
        echo "  [ok ] $stage: $hit  ($n weight file(s))"
    else
        echo "  [MISS] $stage: no weights found"
        [[ "$stage" != "acc" ]] && MISSING=1
    fi
done
echo "----------------------------------------------------------------"
if [[ "$MISSING" == "0" ]]; then
    echo "VERDICT: levels frm+ctp+lsh covered -> local regeneration (T3) is GO."
else
    echo "VERDICT: required level(s) missing. If the download needed manual"
    echo "steps, fetch per the printed instructions and re-run this job"
    echo "(idempotent). Otherwise fall back to released samples (2.6 (d))."
fi
echo "================================================================"
echo "Tree summary:"
du -sh "$WS" 2>/dev/null
find "$WS" -maxdepth 2 -type d | sed "s|$WS|.|" | head -30
