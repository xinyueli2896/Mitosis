#!/bin/bash
# Time-grid pretraining, stage 1 of 2: tokenize a MIDI corpus at one
# frame per 20 ms (the EnCodec 32 kHz frame) into duet/data/$CORPUS.*.
# CPU only. The log ends with pieces, hours and the share of notes
# dropped over cp$MAX_POLYPHONY. From release/:
#
#   J1=$(sbatch --parsable --export=ALL,MIDI_DIR=/path/to/midis,CORPUS=time50_cp16 slurm/pretrain_time_prep.sbatch)
#   sbatch --dependency=afterok:$J1 --export=ALL,CORPUS=time50_cp16 slurm/pretrain_time_train.sbatch
#
# Knobs (see scripts/pretrain_time.sh): MIDI_DIR, CORPUS, FRAME_HZ, MAX_POLYPHONY, PEDAL, MAX_FILES.
#SBATCH --job-name=pretrain_time_prep
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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
bash scripts/pretrain_time.sh prep
