#!/bin/bash
# SLURM (CPU): what is actually IN a staged prompt's scored window?
#
# Answers the question a cascade failure raises: a conditional stage
# needs its conditioning stream to exist in the prompt window, and when
# it does not, the failure surfaces two stages later as
# "'NoneType' object is not subscriptable" -- pretty_midi drops an
# instrument that holds no note, so a hybrid built from an empty
# conditioning window comes back as a one-track file and the driver's
# request for track-1 returns None.
#
# For each song it prints, per stream: notes in the file, notes whose
# ONSET falls inside the window, the first and last onset frame, and the
# file's tick resolution and tempo map. Frames are computed the way the
# tokenizer computes them -- tick / (resolution/4) -- not from seconds,
# so a file with a bogus initial tempo reads correctly here.
#
# Env knobs (--export=ALL,KEY=val):
#   MEL_DIR / CHORD_DIR  staged prompt folders. Default the E1 tree's,
#                        via E1_ROOT.
#   E1_ROOT              default temp/E1_v5b; only used to default the
#                        two folders above.
#   SONG_IDS             required, space separated: "506 786"
#   PROMPT_FRAMES        window end, default 96 (6 bars at 16/bar)
#
# Example:
#   sbatch --export=ALL,E1_ROOT=temp/E1_v5b,SONG_IDS="506 786" \
#       midi_yinyang/diagnose_prompt_window.sbatch

#SBATCH --job-name=diagnose_prompt_window
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:15:00

set -euo pipefail
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

E1_ROOT="${E1_ROOT:-temp/E1_v5b}"
MEL_DIR="${MEL_DIR:-$E1_ROOT/prompts/mel}"
CHORD_DIR="${CHORD_DIR:-$E1_ROOT/prompts/chord}"
PROMPT_FRAMES="${PROMPT_FRAMES:-96}"
SONG_IDS="${SONG_IDS:-}"
[[ -n "${SONG_IDS// /}" ]] || {
    echo "ERROR: SONG_IDS is required, e.g. SONG_IDS=\"506 786\""; exit 1; }

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "mel  : $MEL_DIR"
echo "chord: $CHORD_DIR"
echo "window: frames 0..$PROMPT_FRAMES  ($((PROMPT_FRAMES / 16)) bars)"
echo "songs : $SONG_IDS"
echo "================================================================"

MEL_DIR="$MEL_DIR" CHORD_DIR="$CHORD_DIR" \
PROMPT_FRAMES="$PROMPT_FRAMES" SONG_IDS="$SONG_IDS" python - <<'EOF'
import os
import warnings

import pretty_midi

MEL_DIR = os.environ['MEL_DIR']
CHORD_DIR = os.environ['CHORD_DIR']
WIN = int(os.environ['PROMPT_FRAMES'])
SONGS = os.environ['SONG_IDS'].split()


def find(folder, song):
    for ext in ('.mid', '.MID'):
        p = os.path.join(folder, song + ext)
        if os.path.exists(p):
            return p
    return None


def report(label, path):
    if path is None:
        print(f'  {label:6s} FILE MISSING')
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pm = pretty_midi.PrettyMIDI(path)
    # the tokenizer's grid: frame = tick / (resolution / 4)
    tpf = pm.resolution / 4.0
    notes = [n for inst in pm.instruments if not inst.is_drum
             for n in inst.notes]
    frames = sorted(pm.time_to_tick(n.start) / tpf for n in notes)
    inwin = [f for f in frames if f < WIN]
    _, tempi = pm.get_tempo_changes()
    if not frames:
        print(f'  {label:6s} {len(notes):4d} notes  -- FILE HOLDS NO NOTE')
        return
    print(f'  {label:6s} {len(notes):4d} notes, {len(inwin):4d} in window'
          f'   onset frames {frames[0]:.2f} .. {frames[-1]:.2f}')
    print(f'         resolution={pm.resolution}'
          f'   tempi={[round(float(t), 2) for t in tempi[:3]]}')
    if notes and not inwin:
        print(f'         *** NOTHING IN THE WINDOW: the first note starts '
              f'at frame {frames[0]:.1f}, past {WIN}. A conditional stage '
              f'given this as its conditioning stream has nothing to read, '
              f'and a hybrid built from it loses the track entirely.')


for song in SONGS:
    song = f'{int(song):03d}' if song.isdigit() else song
    print(f'--- {song} ---')
    report('mel', find(MEL_DIR, song))
    report('chord', find(CHORD_DIR, song))
EOF

echo "================================================================"
echo "A stream with notes but NONE in the window is the cascade's"
echo "'NoneType' failure: stage B writes a hybrid whose conditioning"
echo "track is empty, pretty_midi drops it on read-back, and stage C"
echo "asks for a track that is not there."
echo "================================================================"
