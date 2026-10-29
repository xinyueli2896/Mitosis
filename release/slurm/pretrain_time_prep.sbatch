#!/bin/bash
# Time-grid pretraining 1/3: read MIDI folders (and MAESTRO's train split)
# into one note corpus in milliseconds, duet/data/$CORPUS.*. CPU. The log
# ends with pieces, hours, and the share of pieces with pedal and with
# flat velocity (a sign of sequenced, not performed, MIDI).
#
# The chain, from release/ (each stage a plain sbatch; afterok holds the
# rest if one fails):
#
#   C="--export=ALL,CORPUS=time_piano,MIDI_DIRS=/data/aria-midi /data/giantmidi,MAESTRO_ROOT=$R"
#   J1=$(sbatch --parsable "$C" slurm/pretrain_time_prep.sbatch)
#   sbatch --dependency=afterok:$J1 "$C" slurm/pretrain_time_stats.sbatch
#
# read duet/data/time_piano.stats_cp16.json, pick FRAME_CODEC / MAX_POLYPHONY, then
#   sbatch --export=ALL,CORPUS=time_piano,FRAME_CODEC=6 slurm/pretrain_time_train.sbatch
# and, on a checkpoint (paths are relative to duet/):
#   sbatch --export=ALL,CORPUS=time_piano,CKPT=ckpt/<run>/last.ckpt slurm/pretrain_time_sample.sbatch
#SBATCH --job-name=pretrain_time_prep
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
