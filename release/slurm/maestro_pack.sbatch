#!/bin/bash
# MAESTRO v3 stage 3 of 3: tokenize the MIDI on the time grid (default
# FRAME_HZ=50: one frame per 20 ms codec frame; cp16, sustain pedal applied), cut it to the audio, and
# pack rolls + codes per split into duet/data/maestro_t<FRAME_HZ>_cp<P>_*.
# CPU only. The log prints, per split, hours and the share of notes
# dropped over the polyphony budget; raise MAX_POLYPHONY if it is not small.
#
#   sbatch --export=ALL,MAESTRO_ROOT=...,FRAME_HZ=25 slurm/maestro_pack.sbatch
#SBATCH --job-name=maestro_pack
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=deploy
#SBATCH --qos=deploy
set -eo pipefail
source ~/.bashrc || true
# many ~/.bashrc files return early in batch shells, before conda init
command -v conda >/dev/null && eval "$(conda shell.bash hook)"
conda activate mitosis
set -u
ROOT="${SLURM_SUBMIT_DIR:?submit from release/}"
cd "$ROOT"; mkdir -p logs
bash scripts/maestro.sh pack
