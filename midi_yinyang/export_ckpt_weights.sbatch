#!/bin/bash
# SLURM (CPU): strip a Lightning checkpoint to its weights for upload.
#
# A training checkpoint carries the optimizer state and the scheduler
# alongside the weights, about three times the size of the weights
# alone. Inference reads only state_dict (plus hyper_parameters, which
# load_model may use to size the model), so the stripped file loads
# exactly as the full one through the same resolve_best_ckpt path and
# is the one to put on Hugging Face.
#
# Env knobs (--export=ALL,KEY=val):
#   CKPT   run directory (the smallest val_loss ckpt is picked, as at
#          inference) or a ckpt file. Default the A12K1 v5 run.
#   OUT    default: the source file's name with .weights.ckpt, under
#          export/<run dir name>/
#
# Example:
#   sbatch --export=ALL,CKPT=ckpt/<run>/ midi_yinyang/export_ckpt_weights.sbatch

#SBATCH --job-name=export_ckpt_weights
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

CKPT="${CKPT:-ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_A12K1_melchord_v5_draft_tagged_batch_8_schedule}"
OUT="${OUT:-}"

python - "$CKPT" "$OUT" <<'EOF'
import os, sys
import torch
sys.path.insert(0, '.')
from ckpt_utils import resolve_best_ckpt
src = resolve_best_ckpt(sys.argv[1])
out = sys.argv[2]
if not out:
    run = os.path.basename(os.path.dirname(src.rstrip('/')))
    out = os.path.join('export', run,
                       os.path.basename(src).replace('.ckpt', '') + '.weights.ckpt')
os.makedirs(os.path.dirname(out), exist_ok=True)
ck = torch.load(src, map_location='cpu', weights_only=False)
keep = {k: ck[k] for k in ('state_dict', 'hyper_parameters', 'epoch',
                           'global_step', 'pytorch-lightning_version')
        if k in ck}
torch.save(keep, out)
mb = lambda p: os.path.getsize(p) / 2**20
print(f'source  {src}  {mb(src):.0f} MB')
print(f'weights {out}  {mb(out):.0f} MB')
print(f'keys kept: {sorted(keep)}; tensors: {len(keep["state_dict"])}')
print('The training run name is NOT in the file: note MAX_POLYPHONY '
      '(cp) and the run tag beside it when you upload.')
EOF
