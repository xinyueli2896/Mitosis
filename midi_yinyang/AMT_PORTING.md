# Running the AMT baseline on another cluster

The Anticipatory Music Transformer arm (E1 system `AMT`) is written to
be portable: `setup_amt.sbatch` and `infer_amt_prompted.sbatch` derive
the repo root from where `sbatch` was invoked, need no `mitosis` conda
env, name no partition, and write logs to the submission directory.
Everything runs out of the package's own venv.

Nothing else in this repo is portable. Do not expect the training or
the other evaluation wrappers to run elsewhere without work.

## 1. What a clone does NOT bring

`midi_yinyang/input/` is gitignored, so **the held-out prompts are not
in the repo**. Copy them across:

```bash
# from the machine that has them
rsync -av --include='*/' --include='*.mid' --include='prompt_crops.tsv' \
      --exclude='*' \
      <host>:/home/xinyue.li/Mitosis/midi_yinyang/input/heldout_v5/ \
      <clone>/midi_yinyang/input/heldout_v5/
```

That is 95 songs x 2 files plus the TSV, a few hundred KB. Nothing else
is needed: this baseline reads no checkpoint of ours, no `.pt` dataset
and no POP909 source.

Verify after copying:

```bash
ls <clone>/midi_yinyang/input/heldout_v5/melody/*.mid | wc -l   # 95
ls <clone>/midi_yinyang/input/heldout_v5/chord/*.mid  | wc -l   # 95
```

The two counts must match: the merge pairs by filename and the wrapper
refuses to run if they differ.

## 1b. If that cluster has no SLURM

The `#SBATCH` lines are comments, so both wrappers run as ordinary bash
scripts with no changes. Run them from the repo root -- `REPO_DIR`
falls back to `$PWD` -- and pass the knobs as plain environment
variables instead of `--export`:

```bash
cd <clone>
bash midi_yinyang/setup_amt.sbatch
SELFTEST=1 bash midi_yinyang/infer_amt_prompted.sbatch
OUT_DIR=$PWD/midi_yinyang/temp/E1_v5b/AMT bash midi_yinyang/infer_amt_prompted.sbatch
```

Two differences from the batch path. Output goes to your terminal
rather than `%x_%j.out`, so redirect anything long
(`... 2>&1 | tee amt.log`), and run it under `tmux`/`nohup` if the
session can drop. And nothing allocates a GPU for you: set
`CUDA_VISIBLE_DEVICES=<n>` yourself if the machine has more than one
and you do not want device 0.

Everything below applies unchanged -- read `sbatch ... --export=ALL,K=V`
as `K=V bash ...`.

## 2. Setup, once

```bash
cd <clone>
sbatch [-p <partition> -q <qos>] midi_yinyang/setup_amt.sbatch
```

Clones `jthickstun/anticipation` into `midi_yinyang/external/`, builds
an isolated venv, and pre-caches `stanford-crfm/music-medium-800k`.

Two things it needs that a compute node may not have: **network
egress** (git clone, pip, HuggingFace) and a **CUDA 12.1-compatible
torch** -- the pip line pins that index. If your site's torch differs,
edit that one line before submitting. If compute nodes have no egress,
run this step on a login node's interactive session instead; it is the
only step that touches the network.

The `[check]` block prints the vocab geometry and asserts that
`add_token` still calls `instr_logits` through the module global, which
is the hook `amt_prompted.py` patches. If that assert fires, the
package has changed and the instrument restriction would silently do
nothing.

## 3. Pre-flight, no GPU

```bash
sbatch --gres=none [-p <cpu partition>] \
  --export=ALL,SELFTEST=1 midi_yinyang/infer_amt_prompted.sbatch
```

Tokenizes one song, clips it to the prompt boundary, round-trips it
back to MIDI and reports. Read four lines:

| line | wanted | if not |
|---|---|---|
| `clip to 10s: ... max_time <= 10.000` | the boundary is in SECONDS | the `seconds=` flag or the units are wrong |
| `ticks_per_beat=50` | 2 beats/s = 120 BPM, E1's grid | the output is on a different grid than every other system |
| `notes N -> N  OK` | round-trip is not lossy | notes are hitting the duration or polyphony caps |
| `split: ... off-stream 0  OK` | the program tag survives tokenization | the whole split-by-program approach fails and the arm needs another separation rule |

The driver was written against the package's public API without the
package installed, so this step is what catches a signature or unit
mismatch for the price of a CPU minute.

## 4. Smoke run, then the full set

```bash
sbatch [-p <gpu partition>] \
  --export=ALL,MAX_SONGS=3,N_SAMPLES=1,OUT_DIR=$PWD/midi_yinyang/temp/amt_smoke \
  midi_yinyang/infer_amt_prompted.sbatch

sbatch --time=12:00:00 [-p <gpu partition>] \
  --export=ALL,OUT_DIR=$PWD/midi_yinyang/temp/E1_v5b/AMT \
  midi_yinyang/infer_amt_prompted.sbatch
```

`OUT_DIR` must be ABSOLUTE. Output is
`<OUT_DIR>/<song>/co/sample_<i>.mid` with instruments named `MELODY`
and `CHORD` -- the layout E1 scores -- plus `amt_notes.tsv` recording
per-sample note counts and anything that landed off-stream.

## 5. Bringing the results back

Copy the whole `AMT/` folder into the E1 tree on the main cluster:

```bash
rsync -av <clone>/midi_yinyang/temp/E1_v5b/AMT/ \
      <host>:/home/xinyue.li/Mitosis/midi_yinyang/temp/E1_v5b/AMT/
```

then score without regenerating anything:

```bash
sbatch --export=ALL,OUT_ROOT=temp/E1_v5b,RESULT_TAG=v5b,PROMPT_LENGTH=80,\
GEN_LENGTH=416,N_SAMPLES=3,SYSTEMS=all midi_yinyang/score_e1.sbatch
```

`score_e1` forces `SKIP_INFER=1`, so it scores whatever is complete
under `OUT_ROOT` and skips the rest. `SYSTEMS=all` rather than a subset,
or the metrics CSV is overwritten without the systems you did not name.

## 6. Two things to decide once results exist

**`RESTRICT`.** The default masks note logits to the prompt's two
instruments, because the package's own `instr_logits` only restricts
above 15 instruments and a two-instrument prompt can otherwise acquire
a third whose notes belong to neither stream. Run once with
`RESTRICT=0` on a few songs and read the off-stream share in
`amt_notes.tsv`: if it is small the restriction is cosmetic and the
published behaviour can be reported; if it is large the restriction is
load-bearing and the paper must say so.

**Domain.** This model is trained on Lakh MIDI with its own
tokenization, so it is the only OUT-OF-DOMAIN reference in E1. It
answers "what does a large general-purpose symbolic model do here",
which is a different question from "what does the best POP909-native
system do". Keep the two apart in the table.
