#!/bin/bash
# SLURM (CPU): data-overlap audit for the external harmonization
# baseline (EXPERIMENTS.md 2.6 T8).
#
# Question: our 5 test melodies are held out from OUR training only --
# is AccoMontage2's retrieval library allowed to have seen them?
# chorderator's process_raw/ scripts show its progression library was
# built partly from POP909 phrase-split data, so in the worst case the
# DP could retrieve the ground-truth progression of a test song.
#
# This job reports, from the shipped pickles and our saved run logs:
#   1. library composition -- meta['source'] tally of every progression
#      the DP can retrieve (POP909-like vs external pack), and whether
#      any library entry originates from songs 001-005 specifically;
#   2. per-song picks -- for each generated test song, the sources of
#      the progressions the DP actually chose (reconstructed from
#      chord_gen_log.json), flagging POP909-derived picks.
#
# Read-only: touches nothing under temp/, writes only its own log.
#
# Example:
#   sbatch midi_yinyang/audit_accomontage2_overlap.sbatch

#SBATCH --job-name=audit_accomontage2_overlap
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=01:00:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis
MY="$REPO_DIR/midi_yinyang"
AM="$MY/external/accomontage2"
OUT_DIR="${OUT_DIR:-$MY/temp/accomontage2_pop909}"

conda activate "$CONDA_ENV"
mkdir -p /home/xinyue.li/logs
[[ -d "$AM/.venv" ]] || { echo "ERROR: run setup_accomontage2.sbatch first"; exit 1; }
# shellcheck disable=SC1091
source "$AM/.venv/bin/activate"
cd "$AM"
export PYTHONPATH="$AM${PYTHONPATH:+:$PYTHONPATH}"
export AUDIT_OUT_DIR="$OUT_DIR"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "AccoMontage2 data-overlap audit   OUT_DIR=$OUT_DIR"
echo "================================================================"

python - <<'PY'
import json
import os
import re

from chorderator.chords.ChordProgression import read_progressions
from chorderator.utils.utils import pickle_read

TEST_IDS = {'001', '002', '003', '004', '005'}


def classify(source):
    """'pop909' when the source name looks like a POP909 song
    ('114-a-...' style: leading digit id), else 'external-pack'."""
    if source is None:
        return 'unknown'
    s = str(source)
    if 'pop909' in s.lower() or re.match(r'^\d{1,3}[-_]', s):
        return 'pop909'
    return 'external-pack'


# ---- 1. library composition -------------------------------------------
print('---- library composition ' + '-' * 39)
id2src, tally, pop_hits, test_hits = {}, {}, [], []
for name in ('rep', 'lib'):
    try:
        progs = read_progressions(name)
    except Exception as e:                        # noqa: BLE001
        print(f'[warn] read_progressions({name!r}) failed: {e!r}')
        continue
    kinds = {}
    for p in progs:
        src = p.meta.get('source') if isinstance(getattr(p, 'meta', None),
                                                 dict) else None
        kind = classify(src)
        kinds[kind] = kinds.get(kind, 0) + 1
        pid = getattr(p, 'id', None)
        if pid is not None and pid not in id2src:
            id2src[pid] = src
        if kind == 'pop909':
            pop_hits.append(str(src))
            m = re.match(r'^(\d{1,3})', str(src))
            if m and m.group(1).zfill(3) in TEST_IDS:
                test_hits.append(str(src))
    tally[name] = kinds
    print(f'  {name:4s}: {len(progs)} progressions  {kinds}')
if pop_hits:
    uniq = sorted(set(pop_hits))
    print(f'  POP909-derived sources in library: {len(uniq)} unique; '
          f'sample: {uniq[:10]}')
if test_hits:
    print(f'  !! LEAKAGE: library entries from OUR TEST SONGS: '
          f'{sorted(set(test_hits))}')
else:
    print('  no library entry traces to songs 001-005')

# ---- 2. what the DP actually picked per test song ---------------------
print('---- per-song picked progressions ' + '-' * 30)
out_dir = os.environ['AUDIT_OUT_DIR']
verdict_bad = []
for sid in sorted(os.listdir(out_dir)):
    logf = os.path.join(out_dir, sid, 'chord_gen_log.json')
    if not os.path.isfile(logf):
        print(f'  {sid}: no chord_gen_log.json (skipped)')
        continue
    report = json.load(open(logf))
    try:
        best = max(report[-1], key=lambda e: e['cumulative'])
        path = best['path'] or [report[0].index(best)]
        picked_ids = [pid for layer, idx in zip(report, path)
                      for pid in layer[idx]['progression_ids']]
    except Exception as e:                        # noqa: BLE001
        print(f'  {sid}: could not reconstruct pick path ({e!r})')
        continue
    srcs = [id2src.get(pid, f'<id {pid} not in rep/lib>')
            for pid in picked_ids]
    kinds = {classify(s) for s in srcs}
    flag = ''
    if 'pop909' in kinds:
        flag = '  <-- POP909-derived pick'
        verdict_bad.append(sid)
    print(f'  {sid}: {len(srcs)} picked progressions, '
          f'source kinds {sorted(kinds)}{flag}')
    for s in sorted(set(str(x) for x in srcs)):
        print(f'      {s}')

print('================ VERDICT ================')
if verdict_bad:
    print(f'POP909-derived progressions were retrieved for: {verdict_bad}')
    print('-> report library-level overlap in the paper; bias favors the')
    print('   baseline (conservative for us), but check none is the test')
    print("   song's own progression (list above).")
else:
    print('All picked progressions come from external packs; no direct')
    print('POP909 leakage in the scored outputs. Library-level overlap')
    print('(if any, see above) still worth one sentence in the paper.')
PY

echo "================================================================"
echo "Done."
echo "================================================================"
