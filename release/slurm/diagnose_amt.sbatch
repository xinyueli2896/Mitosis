#!/bin/bash
# One tune through AMT with their generate(debug=True): the prompt
# tokens and every generated event are printed, to see why a
# continuation comes out (nearly) empty.
#
#   sbatch slurm/diagnose_amt.sbatch                     # ashover19
#   sbatch --export=ALL,TUNE=hpps31 slurm/diagnose_amt.sbatch
#
# Needs the staged prompts and AMT's venv (infer_external_nottingham
# makes both). Writes to duet/temp/diagnose_amt/, never to the AMT
# results.
#SBATCH --job-name=diagnose_amt
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=deploy
#SBATCH --qos=deploy
set -eo pipefail
source ~/.bashrc || true
command -v conda >/dev/null && eval "$(conda shell.bash hook)"
conda activate mitosis
set -u
ROOT="${SLURM_SUBMIT_DIR:?submit from release/}"
cd "$ROOT"; mkdir -p logs
export DATASET=nottingham
source scripts/common.sh                    # cd's into duet/
AMT_DIR="$DUET/external/anticipation"
[[ -d "$AMT_DIR/.venv" ]] || { echo "run 'amt.sh setup' first" >&2; exit 1; }
need "$PROMPTS/.ready" "Run 'DATASET=nottingham bash scripts/stage_prompts.sh' first."
source "$AMT_DIR/.venv/bin/activate"
unset PYTHONPATH
python -c "import torch, transformers; print('[diag] torch', torch.__version__, 'transformers', transformers.__version__)"
rm -rf temp/diagnose_amt
python amt_prompted.py --merged-folder "$PROMPTS/merged_tagged" --ids "${TUNE:-ashover19}" \
    --out-dir temp/diagnose_amt --model "${AMT_MODEL:-stanford-crfm/music-medium-800k}" \
    --prompt-frames "$PROMPT_LENGTH" --gen-frames "$GEN_LENGTH" \
    --n-samples 1 --top-p 0.98 --debug
