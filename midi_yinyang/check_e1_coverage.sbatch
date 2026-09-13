#!/bin/bash
# SLURM: what did an E1 tree actually generate? On CPU.
#
# Answers "did the run finish, and if not, where did it stop" without
# regenerating or scoring anything. Reads only directory listings, so it
# is safe against a tree a job is still writing into -- a song counted
# mid-write shows as partial, which is the honest answer.
#
# It applies the SAME completeness rule as eval_e1.sbatch's is_complete:
# a song counts as done when it has >= N_SAMPLES files in the layout its
# system uses. Three layouts, and mixing them up is the usual way a
# coverage count lies:
#   duet_multi (A*, WS*, AMT, ...)  <sys>/<song>/co/sample_<i>.mid
#   single     (S0, S1, S-scratch)  <sys>/<song>.mid_temp*_continuation_*.mid
#   pipeline   (P-mc, P-cm)         <sys>/4_final/<song>/co/sample_<i>.mid
#                                   -- the earlier stages are NOT scored,
#                                   so counting every .mid under P-* is
#                                   an overcount, not a coverage figure.
#
# The song list comes from the tree's own staged prompts
# (<OUT_ROOT>/prompts/mel), not from the held-out folder, so it reports
# against what the job was actually asked to do.
#
# Env knobs (--export=ALL,KEY=val):
#   OUT_ROOT    the E1 tree, relative to midi_yinyang (default temp/E1).
#               Must match the OUT_ROOT= line in the job's log.
#   N_SAMPLES   samples per song that count as complete (default 3).
#               Pass the value the generating job used.
#   SYSTEMS     restrict to these system names; default every directory
#               in the tree apart from prompts/ and genlogs/.
#   VERBOSE=1   list every short song with its count, not just the first
#               few. Long output on a tree that barely started.
#
# Examples:
#   sbatch --export=ALL,OUT_ROOT=temp/E1_v5b midi_yinyang/check_e1_coverage.sbatch
#   sbatch --export=ALL,OUT_ROOT=temp/E1_v5b,SYSTEMS="A1 A3",VERBOSE=1 \
#       midi_yinyang/check_e1_coverage.sbatch
#
# A short system resumes by resubmitting the generating job unchanged:
# every arm runs with --skip-existing and a complete system is skipped
# outright, so only the gaps cost GPU time.

#SBATCH --job-name=check_e1_coverage
#SBATCH --output=/home/xinyue.li/logs/%x_%j.out
#SBATCH --error=/home/xinyue.li/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=00:20:00

set -euo pipefail
# A missing song folder is the thing this script REPORTS, so globbing
# must not treat it as an error: `ls ... | wc -l` returns ls's failure
# under pipefail and kills the run on the first gap. Arrays with
# nullglob count the same thing and expand to empty instead.
shopt -s nullglob
source ~/.bashrc

CONDA_ENV=mitosis
REPO_DIR=/home/xinyue.li/Mitosis

OUT_ROOT="${OUT_ROOT:-temp/E1}"
N_SAMPLES="${N_SAMPLES:-3}"
SYSTEMS="${SYSTEMS:-}"
VERBOSE="${VERBOSE:-0}"

conda activate "$CONDA_ENV"
cd "$REPO_DIR/midi_yinyang"
mkdir -p /home/xinyue.li/logs

[[ -d "$OUT_ROOT" ]] || {
    echo "ERROR: no tree at $OUT_ROOT (relative to $REPO_DIR/midi_yinyang)."
    echo "       The generating job's log prints the OUT_ROOT it used."
    exit 1; }

PROMPT_MEL="$OUT_ROOT/prompts/mel"
[[ -d "$PROMPT_MEL" ]] || {
    echo "ERROR: $PROMPT_MEL missing -- this tree never finished staging"
    echo "       prompts, so nothing was generated against a song list."
    exit 1; }

SONG_IDS=$(ls "$PROMPT_MEL" | sed 's/\.[Mm][Ii][Dd]$//' | sort)
N_EVAL=$(echo "$SONG_IDS" | wc -l)

if [[ -z "$SYSTEMS" ]]; then
    # `|| true`: grep exits 1 when every directory is filtered out,
    # which is a legitimate answer (a tree with prompts and nothing
    # else), not a failure.
    SYSTEMS=$(find "$OUT_ROOT" -maxdepth 1 -mindepth 1 -type d \
              -printf '%f\n' | { grep -vx 'prompts\|genlogs' || true; } \
              | sort | tr '\n' ' ')
fi

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "OUT_ROOT=$OUT_ROOT   N_SAMPLES=$N_SAMPLES"
echo "songs staged: $N_EVAL   (complete system = ${N_EVAL} x ${N_SAMPLES}"
echo "              = $((N_EVAL * N_SAMPLES)) files)"
echo "systems: ${SYSTEMS:-<none found>}"
echo "================================================================"

[[ -n "${SYSTEMS// /}" ]] || {
    echo "Nothing generated: $OUT_ROOT holds prompts and no system folder."
    exit 0; }

sys_root()   { case "$1" in P-mc|P-cm) echo "$OUT_ROOT/$1/4_final" ;;
                            *)         echo "$OUT_ROOT/$1" ;; esac; }
sys_layout() { case "$1" in S0|S1|S-scratch) echo single ;;
                            *)               echo duet_multi ;; esac; }

printf '%-14s %8s %8s %8s %9s  %s\n' \
       SYSTEM COMPLETE PARTIAL MISSING FILES STATUS

incomplete=""
for name in $SYSTEMS; do
    root=$(sys_root "$name"); layout=$(sys_layout "$name")
    full=0; part=0; none=0; files=0; short=""
    if [[ -d "$root" ]]; then
        for sid in $SONG_IDS; do
            case "$layout" in
              duet_multi) got=( "$root/$sid/co/"sample_*.mid ) ;;
              single)     got=( "$root/$sid".mid_temp*_continuation_*.mid ) ;;
            esac
            n=${#got[@]}
            files=$((files + n))
            if   [[ "$n" -ge "$N_SAMPLES" ]]; then full=$((full + 1))
            elif [[ "$n" -gt 0 ]]; then part=$((part + 1)); short="$short $sid($n)"
            else none=$((none + 1)); short="$short $sid(0)"
            fi
        done
    else
        none=$N_EVAL
    fi

    if [[ ! -d "$root" ]];            then status="NOT STARTED"
    elif [[ "$full" -eq "$N_EVAL" ]]; then status="complete"
    elif [[ "$full" -eq 0 ]];         then status="NOTHING USABLE"
    else                                   status="INCOMPLETE"
    fi
    printf '%-14s %8d %8d %8d %9d  %s\n' \
           "$name" "$full" "$part" "$none" "$files" "$status"

    if [[ "$full" -ne "$N_EVAL" ]]; then
        incomplete="$incomplete $name"
        if [[ -z "${short// /}" ]]; then
            :   # NOT STARTED: no per-song detail to give
        elif [[ "$VERBOSE" == "1" ]]; then
            echo "               short:$short"
        else
            # First few are enough to tell "died on one song" from
            # "never really ran"; VERBOSE=1 prints the rest.
            echo "               short:$(echo $short | cut -d' ' -f1-8) ..."
        fi
    fi
done

echo "================================================================"
if [[ -z "${incomplete// /}" ]]; then
    echo "All systems complete for all $N_EVAL songs."
    echo "Score without regenerating:"
    echo "  sbatch --export=ALL,OUT_ROOT=$OUT_ROOT,SYSTEMS=all \\"
    echo "      midi_yinyang/score_e1.sbatch"
else
    echo "Incomplete:$incomplete"
    echo
    echo "Resubmit the generating job unchanged to fill the gaps -- every"
    echo "arm runs with --skip-existing and a complete system is skipped,"
    echo "so finished work is not redone. Why a system produced nothing"
    echo "is in $OUT_ROOT/genlogs/<system>.log."
    echo
    echo "A song listed at the full N_SAMPLES but audibly truncated was"
    echo "counted by filename, not by content: delete that song's folder"
    echo "and it regenerates."
fi
echo "================================================================"
