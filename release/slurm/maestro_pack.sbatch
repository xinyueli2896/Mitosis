#!/bin/bash
# MAESTRO v3 stage 3 of 3: read each piece's MIDI into a note corpus in
# milliseconds (sustain pedal folded in), cut it to its audio, and pack it
# with the codes per official split into duet/data/maestro_<split>.*. CPU
# only. Tokenization happens at load time (MaestroPairedDataset), with
# the time-grid cp transformer's own frame size and polyphony budget, so
# one pack serves every tokenizer setting.
#
#   sbatch --export=ALL,MAESTRO_ROOT=... slurm/maestro_pack.sbatch
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
