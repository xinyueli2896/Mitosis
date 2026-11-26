#!/usr/bin/env bash
# Put the MIDI codec and the AR model on the Hugging Face Hub, with the
# code and the MAESTRO eval corpora they need, so eval / listen / sample
# can run anywhere (e.g. a Colab GPU) without the cluster.
#
# Writes slim checkpoints (weights + hyperparameters, no optimizer state)
# to temp/hf_export/ next to midi_rvq.py, midi_ar.py, preprocess_time_grid.py
# and data/<EVAL_DATA>.*, then uploads that folder to HF_REPO (private).
# CPU only.
#
# Knobs:
#   HF_TOKEN   write token (required; pass it on --export, never write it down)
#   HF_REPO    e.g. c0smic1atte/midi-codec-ar (required)
#   CODEC      codec ckpt or run folder (highest val_note_f1 in it)
#   AR         AR ckpt or run folder (lowest val_loss in it); empty = codec only
#   EVAL_DATA  note corpora to include (default "maestro_test_midi maestro_val_midi")
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${HF_TOKEN:?set HF_TOKEN on --export}"; : "${HF_REPO:?set HF_REPO=<user>/<repo>}"
: "${CODEC:?set CODEC to the codec run folder or a .ckpt}"

pick_ckpt() {
    if [[ -f "$1" ]]; then echo "$1"; return; fi
    local order=-g best; [[ "$3" == max ]] || order=-gr
    best=$(ls "$1"/*"$2"=*.ckpt 2>/dev/null \
        | sed "s/.*$2=\([0-9.]*\)\.ckpt$/\1 &/" | sort $order | tail -1 | cut -d' ' -f2-)
    [[ -n "$best" ]] || { echo "ERROR: no *$2=*.ckpt in $1" >&2; exit 1; }
    echo "$best"
}

OUT=temp/hf_export; rm -rf "$OUT"; mkdir -p "$OUT/data"
C=$(pick_ckpt "$CODEC" val_note_f1 max); echo "codec $C"
A=""; [[ -n "${AR:-}" ]] && { A=$(pick_ckpt "$AR" val_loss min); echo "AR $A"; }
cp midi_rvq.py midi_ar.py preprocess_time_grid.py "$OUT/"
for d in ${EVAL_DATA:-maestro_test_midi maestro_val_midi}; do
    need "data/$d.notes.bin" "set EVAL_DATA to note corpora under data/"
    cp data/"$d".* "$OUT/data/"
done
python - "$C" "$A" "$OUT" <<'PY'
import sys, torch
codec, ar, out = sys.argv[1:]
for src, name in ((codec, 'codec.ckpt'), (ar, 'ar.ckpt')):
    if not src:
        continue
    ck = torch.load(src, map_location='cpu', weights_only=False)
    slim = {k: ck[k] for k in ('state_dict', 'hyper_parameters', 'pytorch-lightning_version', 'global_step', 'epoch') if k in ck}
    slim['source'] = src
    torch.save(slim, f'{out}/{name}')
    print(f'{name}: {src}, step {ck.get("global_step")}')
PY
python - "$HF_REPO" "$OUT" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1:]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=folder, commit_message='codec + AR for eval')
print(f'uploaded {folder} to https://huggingface.co/{repo} (private)')
PY
du -sh "$OUT"
