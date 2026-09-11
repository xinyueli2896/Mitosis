#!/bin/bash
# SLURM (CPU): gate for A.12 (AR-head self-conditioning drafts) before
# training. See audit_sc_ar_draft.py for the eight checks.
#
#   sbatch midi_yinyang/audit_sc_ar_draft.sbatch

#SBATCH --job-name=audit_sc_ar_draft
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:20:00

set -euo pipefail
source ~/.bashrc
conda activate mitosis
cd /home/xinyue.li/Mitosis/midi_yinyang
mkdir -p /home/xinyue.li/logs
python audit_sc_ar_draft.py
