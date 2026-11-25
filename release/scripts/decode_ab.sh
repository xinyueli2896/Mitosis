#!/usr/bin/env bash
# Decode A/B: run generate.sh for one system once per variant, one after
# the other in a single job (the QOS caps how many jobs can be queued).
# Each variant writes to its own duet/temp/dbg_<name>/ and reuses the
# staged prompts of the dataset's E1 folder; CKPT, SHARD, DATASET come
# from the environment as for generate.sh.
#
#   sbatch [res] --export=ALL,DATASET=pop909,SHARD=0/48,CKPT=<ckpt> \
#       slurm/run.sbatch decode_ab.sh duet_diffusion_gatedjoint ema0:A3_EMA=0 conf:MASK_ORDER=conf
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SYSTEM="${1:?usage: decode_ab.sh <system> <name>:<VAR=value> ...}"; shift
E1=$(bash -c "source scripts/common.sh >/dev/null; echo \$OUT_ROOT")
export PYTHONUNBUFFERED=1
for v in "$@"; do
    name=${v%%:*}; kv=${v#*:}
    mkdir -p "duet/temp/dbg_$name"
    ln -sfn "$(cd duet && realpath "$E1")/prompts" "duet/temp/dbg_$name/prompts"
    echo "=== [decode_ab] $name: $kv"
    env OUT_ROOT="temp/dbg_$name" "$kv" bash scripts/generate.sh "$SYSTEM"
done
