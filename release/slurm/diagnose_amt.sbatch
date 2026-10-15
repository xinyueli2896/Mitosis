#!/bin/bash
# One tune through AMT with their generate(debug=True): the prompt
# tokens and every generated event are printed, to see why a
# continuation comes out (nearly) empty.
#
#   sbatch slurm/diagnose_amt.sbatch                     # ashover19
#   sbatch --export=ALL,TUNE=hpps31 slurm/diagnose_amt.sbatch
#
# Controls against the POP909 run that worked (same driver, same code):
#   AMT_DIR=<dir>   use another checkout's venv, e.g. the one the POP909
#                   run used (midi_yinyang/external/anticipation)
#   MERGED=<dir>    another merged prompt folder, e.g. a POP909
#                   prompts/merged_tagged, with TUNE=<its song id>
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
AMT_DIR="${AMT_DIR:-$DUET/external/anticipation}"
MERGED="${MERGED:-$PROMPTS/merged_tagged}"
[[ -d "$AMT_DIR/.venv" ]] || { echo "run 'amt.sh setup' first" >&2; exit 1; }
need "$MERGED/${TUNE:-ashover19}.mid" "Stage the prompts first (DATASET=nottingham bash scripts/stage_prompts.sh) or set MERGED/TUNE."
echo "[diag] venv $AMT_DIR/.venv  prompt $MERGED/${TUNE:-ashover19}.mid"
source "$AMT_DIR/.venv/bin/activate"
unset PYTHONPATH
python -c "import torch, transformers; print('[diag] torch', torch.__version__, 'transformers', transformers.__version__)"
rm -rf temp/diagnose_amt
python amt_prompted.py --merged-folder "$MERGED" --ids "${TUNE:-ashover19}" \
    --out-dir temp/diagnose_amt --model "${AMT_MODEL:-stanford-crfm/music-medium-800k}" \
    --prompt-frames "$PROMPT_LENGTH" --gen-frames "$GEN_LENGTH" \
    --n-samples 1 --top-p 0.98 --debug
