#!/usr/bin/env bash
# Build the patched transformers package the models import.
#
# The MoE RoFormer lives in a small fork of HuggingFace transformers
# 4.49.0: seven files differ from the release (third_party/
# transformers_roformer_overlay). The model code puts
# duet/transformers_roformer_moe/src on sys.path and imports
# `transformers` from there, so this script materialises that directory:
# the official 4.49.0 wheel, unpacked, with the overlay copied on top.
#
#   bash scripts/setup_transformers.sh          # idempotent
#   FORCE=1 bash scripts/setup_transformers.sh  # rebuild
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$ROOT/duet/transformers_roformer_moe/src"
OVERLAY="$ROOT/third_party/transformers_roformer_overlay/transformers"
VERSION=4.49.0

if [[ -f "$DST/transformers/models/roformer/grouped_gemm_util.py" && "${FORCE:-0}" != "1" ]]; then
    echo "[setup_transformers] $DST already built (FORCE=1 to rebuild)"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python -m pip download "transformers==$VERSION" --no-deps -d "$TMP" -q
python -m zipfile -e "$TMP"/transformers-"$VERSION"-py3-none-any.whl "$TMP/unpacked"

rm -rf "$DST/transformers"
mkdir -p "$DST"
cp -r "$TMP/unpacked/transformers" "$DST/transformers"
cp -r "$OVERLAY/." "$DST/transformers/"
echo "[setup_transformers] transformers $VERSION + RoFormer-MoE overlay -> $DST"
