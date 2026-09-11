#!/bin/bash
# SLURM (CPU): gate for A.11 (partner-agreement discrimination head)
# before training. See audit_agree_head.py for the five checks.
#
#   sbatch midi_yinyang/audit_agree_head.sbatch

#SBATCH --job-name=audit_agree_head
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
python audit_agree_head.py
