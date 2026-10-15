#!/bin/bash
# SLURM (CPU): listening samples from music-jepa-maestro. Prints the .h5
# layout, calibrates the mel filterbank against the piano roll, then
# renders N clips (Griffin-Lim wav, MIDI, MIDI synth wav, overview png)
# with the calibrated settings.
#
#   sbatch maestro_jepa/render_maestro.sbatch
#   sbatch --export=ALL,N=10,SPLIT=val maestro_jepa/render_maestro.sbatch
#
# Env knobs: DATA (default /home/xinyue.li/music-jepa/data/maestro),
# OUT (default /home/xinyue.li/music-jepa/samples_maestro), SPLIT (test), N (5)

#SBATCH --job-name=maestro_render
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc || true
command -v conda >/dev/null && eval "$(conda shell.bash hook)"
conda activate mitosis

S="${SLURM_SUBMIT_DIR:-$(pwd)}/maestro_jepa/render_samples.py"
[[ -f "$S" ]] || S=/home/xinyue.li/Mitosis/maestro_jepa/render_samples.py
DATA="${DATA:-/home/xinyue.li/music-jepa/data/maestro}"
OUT="${OUT:-/home/xinyue.li/music-jepa/samples_maestro}"
SPLIT="${SPLIT:-test}"
N="${N:-5}"
mkdir -p "$OUT" /home/xinyue.li/logs

python "$S" inspect --data-dir "$DATA"
python "$S" calibrate --data-dir "$DATA" --split "$SPLIT" --n 3 | tee "$OUT/calibrate.txt"
FLAGS=$(grep '^best:' "$OUT/calibrate.txt" | sed 's/^best: //')
echo "using: $FLAGS"
python "$S" render --data-dir "$DATA" --split "$SPLIT" --n "$N" --out "$OUT" $FLAGS
