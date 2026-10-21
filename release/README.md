# Duet: Coordinating Pretrained Language Models for Joint Generation of Melody and Harmony

Code for the paper by Xinyue Li, Gus Xia and Ziyu Wang.
Demos: <https://xinyueli2896.github.io/duet_demo/>

Duet composes two pretrained autoregressive music models into one Transformer
that generates a melody and its chords together, frame by frame. Each layer
keeps a copy of the pretrained block per stream, lets the streams read each
other through gated cross-stream attention, routes both over a shared pool of
experts with stream-specific routers, and resolves each frame's two
components jointly with draft-and-denoise decoding.

This repository contains exactly what the paper's evaluation needs: data
preparation, training of the five in-house systems, generation for all seven
systems, and the scripts that produce Table 1, Figure 2 and Figure 3.

## Repository layout

```
duet/            model, data, generation and evaluation code (every command runs from here)
  figures/       plotting scripts for the listening test
scripts/         one script per stage; these are the entry points
  external/      the two external baselines (AMT, Whole-Song Gen)
slurm/run.sbatch run any stage under SLURM
third_party/     the seven files of our transformers fork (MoE RoFormer), Apache-2.0
```

Data, checkpoints and outputs are written under `duet/` (`data/`, `ckpt/`,
`input/`, `temp/`, `results/`, `figures/`) and are not tracked.

## Setup

```bash
conda create -n duet python=3.11 && conda activate duet
pip install -r requirements.txt
bash scripts/setup_transformers.sh
```

The MoE models import a fork of HuggingFace transformers 4.49.0 that differs
from the release in seven files. `setup_transformers.sh` downloads the
official 4.49.0 wheel, applies `third_party/transformers_roformer_overlay`,
and places the result in `duet/transformers_roformer_moe/src`, which every
script puts first on `PYTHONPATH`. With MoE disabled the fork's RoFormer is
identical to upstream, so the single-stream models use the same package.

## The seven systems

| Paper row | Script name | Output id | What it is | Decoding |
|---|---|---|---|---|
| Duet (ours) | `duet` | `A3ctcaT` | backbone copied into two streams, 4 shared experts (top-2), draft tokens | draft-and-denoise, leader alternating per frame, T=1 |
| Duet w/o MoE (dense) | `dense` | `D1` | same, one dense feed-forward block of width 6144 | as Duet |
| Duet w/o draft tokens | `no_draft` | `A1` | same attention and experts, no draft tokens | autoregressive, T=1 |
| Single-stream (scratch) | `ss_scratch` | `S-scratch` | backbone architecture on the merged sequence, random init | autoregressive, T=1 |
| Single-stream (finetuned) | `ss_finetuned` | `S1` | the backbone finetuned on the merged sequence | autoregressive, T=1 |
| Whole-Song Gen | `wholesong` | `WSfv4` | Wang et al., ICLR 2024 | its own sampler |
| AMT | `amt` | `AMT` | Thickstun et al., TMLR 2024, `music-medium-800k` | top-p 0.98 |

Every system continues the same prompts: the first 80 frames (five bars of
sixteenth notes) of each of 92 held-out POP909 songs, to 416 frames in total,
three samples per song. The output ids are the folder names under
`duet/temp/E1/` and the keys the tables and figures use.

## Reproduce the results from released checkpoints

Checkpoints and prepared data are hosted on Hugging Face in the same layout
as `duet/`:

```
ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt   pretrained backbone
ckpt/{duet,dense,no_draft,ss_finetuned,ss_scratch}/<file>.ckpt         the five in-house systems
input/heldout_v5/{melody,chord}/, input/heldout_v5/prompt_crops.tsv    held-out prompts
data/pop909_{melody,chord}_cp4_v2.*, data/pop909_melchord_tagged_cp16_v2.*   training tensors
```

```bash
HF_REPO=<owner>/<repo> bash scripts/download.sh

# in-house systems (one GPU each; run them in parallel with CUDA_VISIBLE_DEVICES)
for s in duet dense no_draft ss_finetuned ss_scratch; do bash scripts/generate.sh $s; done

# external baselines (each sets up its own repository and environment once)
bash scripts/external/amt.sh setup && bash scripts/external/amt.sh generate
bash scripts/external/wholesong.sh setup
POP909_RAW=/path/to/POP909-Dataset/POP909 bash scripts/external/wholesong.sh generate

# Table 1, Figure 2 and the per-metric significance table
bash scripts/evaluate.sh

# Figure 3, from the listening-test export
RATINGS=/path/to/ratings.csv bash scripts/subjective.sh
```

`evaluate.sh` scores every system that is complete and skips the rest, so it
can be run after any subset of the generation steps. Generation for the
single-stream baselines needs a CUDA GPU; training, Duet generation and all
scoring also run on CPU, slowly.

## Train from scratch

Clone <https://github.com/music-x-lab/POP909-Dataset> and download the
pretrained backbone (`HF_REPO=... bash scripts/download.sh 'ckpt/cp_transformer_*'`).

```bash
export POP909_RAW=/path/to/POP909-Dataset/POP909
bash scripts/prepare_data.sh train     # training tensors
bash scripts/prepare_data.sh eval      # held-out prompts (reads the training manifest)

bash scripts/train.sh duet             # also: dense, no_draft, ss_finetuned, ss_scratch
CKPT=ckpt/<run dir> bash scripts/generate.sh duet
```

`train.sh` uses every GPU in `CUDA_VISIBLE_DEVICES` and resumes from the run's
`last.ckpt` when it is re-run. The recipes of the paper's runs:

| System | Data | Steps | Per-GPU batch | Notes |
|---|---|---|---|---|
| Duet | `pop909_{melody,chord}_cp4_v2` | 50k | 4 | 2 GPUs, lr 7e-5, one-cycle; experts and attention copied from the backbone, cross-stream gates start closed (bias -10) |
| dense | same | 100k | 4 | `--moe_num_experts 1 --moe_intermediate_size 6144` |
| w/o draft tokens | same | 50k | 4 | |
| SS finetuned | `pop909_melchord_tagged_cp16_v2` | see note | | global batch 36 |
| SS scratch | same | see note | 12 | `--from_scratch`, global batch 12 |

The exact schedule of each released checkpoint is stored in its
`hyper_parameters`. With a different GPU count the learning rate is scaled
as in the table's runs (`MAX_LR` overrides it).

## Variant: drafts from a stale snapshot

`train.sh duet_stale` trains Duet with one change in how the draft tokens
are filled during training. In the paper's Duet, half of the slots are
filled with the live model's query-head argmax. In `duet_stale`, both the
melody and the chord slot of every item get a draft. The draft is sampled
from the AR head the way the decode samples (`local_sampling`,
temperature 1). It comes from a frozen copy of the model that is
refreshed from the live weights every `STALE_EVERY` steps (default
1000). There are no drafts before the first refresh. The masking
schedule (`k`) is unchanged, so a masked slot stays masked. The copy
costs one more model's weights in GPU memory. Scoring treats it like
Duet (`generate.sh duet_stale`, output id `A3stale`), and it starts from
Duet's init. Its decode has no leader (`A3_SCHEDULE=joint`): one forward
drafts both streams from their AR heads, both drafts are written into
the draft tokens, and one more forward reads both refined frames out.

## Improving Duet's co-generation

Duet's generations fall short of its own ablations and of the finetuned
single-stream model, and they lose on every rhythm row (harmonic rhythm,
onset grid, duration). Three things in the code bear on it:

- **A draft token never sees the previous frame.** A query slot for
  frame t reads clean history through t-2 (`_build_masks`). The AR
  heads, which write the drafts, see t-1, and so do both ablations. But
  every committed frame is read out of a slot. Whether a note was just
  struck or is still sounding is exactly what onset and duration
  statistics measure.
- **The decode reads the slots in a state training barely visits.** In
  the leaderless joint decode, both draft tokens hold AR samples tagged
  k=0. The paper's training puts a draft in a slot a quarter of the time
  (a coin at rate k/K masks half of them), and that draft is a
  query-head argmax rather than an AR sample.
- **Best-val scores both slots masked**, a state no decode reads from.

Every variant below decodes without a leader: one forward drafts both
streams from their AR heads, both drafts go into the draft tokens, and
one more forward reads both refined frames out (`A3_SCHEDULE=joint`).

| Script name | Output id | What changes |
|---|---|---|
| `duet_joint` | `A3jt` | the paper's duet checkpoint, joint decode |
| `duet_jointmix` | `A3jm` | as `duet_joint`, each stream sampled from the product of its refined distribution and its own AR head's (`A3_JOINT_MIX=0.5`, geo), so both see frame t-1 |
| `duet_k0` | `A3K0` | AR heads only, draft tokens unused (control) |
| `duet_bestar` | `A3ar` | `duet_joint` on `<run>/best_ar/`, the weights best on the clean-row loss |
| `duet_prev` | `A3f` | retrain: paper recipe + `--slot_sees_prev_frame` |
| `duet_v2` | `A3v2` | retrain for the joint decode: `--slot_sees_prev_frame`; both slots drafted every pair, from the AR heads the way the decode samples (`--self_cond_prob 1 --sc_ar_frac 1 --sc_ar_free_run --sc_draft_temp 1`), tagged k=0 and scored (`--draft_k0`); validation in the same state (`--sc_val`) |
| `duet_v2c48` | `A3v2c48` | `duet_v2` with the chord stream on program 48, the single-stream models' tag (`prepare_data.sh chord48`, task `melchord_nottingham_c48`); the same notes, only the program token differs |
| `duet_v3` | `A3v3` | retrain so the leaderless refine has something to learn: `--slot_sees_prev_frame`; an unmasked draft token always holds an AR-sampled draft, never the true frame (`--self_cond_prob 1 --sc_ar_frac 1 --sc_ar_free_run --sc_draft_temp 1`); one commitment level for both slots, masked only at k=K (`--sym_k --sc_k_consistent`); 8 query pairs per sequence (`--query_pairs 8`); validation on drafts (`--sc_val`) |
| `duet_prevmix` | `A3fm` | `duet_prev`'s checkpoint with `duet_jointmix`'s decode |
| `duet_v2mix` | `A3v2m` | `duet_v2`'s checkpoint with `duet_jointmix`'s decode |
| `duet_prevk0` | `A3fK0` | `duet_prev`'s checkpoint, AR heads only (control for `A3f`'s refine pass) |
| `duet_prevr3` | `A3fr3` | `duet_prev`'s checkpoint, joint refine repeated 3 rounds (`A3_JOINT_ROUNDS=3`), each reading the last round's frames |

`duet_stale` (above) is the closest earlier attempt at the joint
regime. It differs from `duet_v2` in the t-1 mask, drafts from a frozen
copy, k tags drawn uniformly with half the drafts masked, and validation
on masked slots.

Two whole-node jobs run them (8 GPUs; `DATASET` defaults to nottingham):

```bash
# decodes of the existing checkpoint, every GPU taking an eighth of the songs
sbatch slurm/infer_duet_variants.sbatch
# the retrains, two GPUs each, then their samples
T=$(sbatch --parsable slurm/train_duet_variants.sbatch)
sbatch --dependency=afterok:$T --export=ALL,SYSTEMS="duet_v2 duet_v2c48 duet_prev" slurm/infer_duet_variants.sbatch
```

With `--export=ALL,DATASET=pop909` the same jobs run on POP909: the
decodes use the released checkpoint and leave out `duet_bestar` (it has
no best_ar/; `CKPT_duet_bestar=<run dir>/best_ar` adds it), and the
retrains are `duet_v2` and `duet_prev`.
`evaluate.sh` scores every variant that is complete. Table 1 and
Figure 2 leave them out; `figures/<tag>_quality_variants.tex` is Table 1
with them added, and `results/<tag>_table_twosided.md` tests each one
against Duet.

## Training everything on one machine

`scripts/train_all.sh` trains all six systems on one multi-GPU machine
without SLURM. Each run gets the GPU count of its paper recipe, so its
learning rate and global batch match the paper. Runs are packed onto the
free GPUs with the longest first. On 8 GPUs, dense, duet, duet_stale and
no_draft start together on two GPUs each. The single-stream runs take the
first GPUs that free up, and dense (100k steps) finishes last.

```bash
DATASET=nottingham NOTTINGHAM_RAW=/path/to/nottingham-dataset/MIDI \
    bash scripts/train_all.sh 2>&1 | tee logs/train_all_nottingham.log
```

It prepares the data first if the tensors are missing, writes one log per
run to `logs/train_<dataset>_<system>.log`, and ends with each run's exit
code. Re-running it resumes. `GPU_IDS`, `SYSTEMS` and `GPUS_<system>`
(for example `GPUS_dense=4`) change the packing.

## The same task on Nottingham

`DATASET=nottingham` runs every in-house system on the Nottingham folk
tunes (<https://github.com/jukedeck/nottingham-dataset>) instead of
POP909. Every stage script reads it, and everything it writes is kept
apart from the POP909 files: `data/nottingham_*`, `input/nottingham_heldout/`,
run directories named with the task `melchord_nottingham`,
`temp/E1_nottingham/`, and `results/E1_nottingham_*`.

```bash
export DATASET=nottingham NOTTINGHAM_RAW=/path/to/nottingham-dataset/MIDI
bash scripts/prepare_data.sh train     # data/nottingham_{melody,chord}_cp4_v2.pt, data/nottingham_melchord_tagged_cp16_v2.pt
bash scripts/prepare_data.sh eval      # input/nottingham_heldout/{melody,chord}/

for s in duet dense no_draft ss_finetuned ss_scratch; do bash scripts/train.sh $s; done
for s in duet dense no_draft ss_finetuned ss_scratch; do bash scripts/generate.sh $s; done
bash scripts/evaluate.sh
```

The recipes are the POP909 ones (same steps, batch sizes and learning
rates). What differs:

- **Corpus.** Melody is each MIDI's first track and chords its second,
  both on program 0 as for POP909's duet streams; the single-stream copy
  re-tags the chords to program 48. The 13 tunes without a chord track
  are dropped, leaving 1,021; the 153 shorter than 384 frames sit
  outside training and validation (`FramedDataset`), leaving 868.
- **Held-out prompts.** Index 0 mod 10 of the training manifest, as for
  POP909. Only tunes in 4/4 or 2/2 are kept, because the scoring and the
  80/416-frame protocol assume 16 frames per bar, and only tunes whose
  crop reaches `GEN_LENGTH`: 49 of the 87 held-out tunes (the other 38
  are in 3/4, 6/8 and similar metres). Each is cropped so its first full downbeat
  is at frame 32; the pickup is found from the chord track, which starts
  on the downbeat. `prompt_crops.tsv` lists every held-out tune, kept or
  dropped and why. Training uses every metre.
- **Checkpoints.** No Nottingham checkpoints are released, so
  `generate.sh` defaults to the run directory `train.sh` wrote for the
  system; pass `CKPT=` for any other.
- **External baselines.** Both run on the same prompts, with the same
  prompt length, continuation length and sample count:
  `DATASET=nottingham bash scripts/external/amt.sh generate` and
  `DATASET=nottingham bash scripts/external/wholesong.sh generate`.
  AMT needs nothing new: the prompts are at 120 BPM, its time base.
  Whole-Song Gen reads songs in its POP909 layout, which
  `build_nottingham_wholesong.py` writes from the staged crops: melody,
  chord symbols named from our chord track (every held-out chord matches
  a template), and a phrase label. Nottingham has no phrase annotation,
  so the label is a heuristic (an optional pickup bar as intro, then
  8-bar phrases lettered by melodic repetition), and it reaches the
  model only through the prompt bars. As for POP909's padded crops, the
  baseline's prompt drops our silent lead-in bars (one for pickup tunes,
  two otherwise) and its output is shifted back by them. Its chords are
  re-voiced into Nottingham's close-position voicing (bass at 36+pc,
  tones stacked on it), which `wholesong_chord_map.py` now fits
  alongside POP909's fixed-octave rendering.
  `evaluate.sh` scores whichever systems are complete.

## Running under SLURM

`slurm/run.sbatch` runs any stage script; resources go on the command line:

```bash
J1=$(sbatch --parsable --gres=gpu:1 --time=12:00:00 slurm/run.sbatch generate.sh duet)
sbatch --dependency=afterok:$J1 slurm/run.sbatch evaluate.sh
```

Environment knobs such as `DATASET` travel with `--export=ALL,...`.

### Running the Nottingham pipeline on SLURM

Two whole-node jobs cover training and the external baselines (they
carry `--account`/`--qos` for our cluster; override on the command
line elsewhere):

```bash
sbatch slurm/train_all_nottingham.sbatch        # every in-house system, scripts/train_all.sh on 8 GPUs
sbatch slurm/infer_external_nottingham.sbatch   # AMT + Whole-Song Gen, 8 slices each
sbatch slurm/infer_nottingham.sbatch            # the six in-house systems, 8 slices each
```

Or stage by stage:

```bash
E=ALL,DATASET=nottingham,NOTTINGHAM_RAW=/path/to/nottingham-dataset/MIDI
P=$(sbatch --parsable --export=$E --cpus-per-task=16 --time=02:00:00 slurm/run.sbatch prepare_data.sh train)
V=$(sbatch --parsable --export=$E --dependency=afterok:$P slurm/run.sbatch prepare_data.sh eval)
V=$(sbatch --parsable --export=$E --dependency=afterok:$V slurm/run.sbatch stage_prompts.sh)  # once, before the parallel generators
for s in duet dense no_draft ss_finetuned ss_scratch; do
    g=1; [[ $s == duet || $s == dense || $s == no_draft ]] && g=2
    T=$(sbatch --parsable --export=$E --dependency=afterok:$P --gres=gpu:$g --time=48:00:00 slurm/run.sbatch train.sh $s)
    G=$(sbatch --parsable --export=$E --dependency=afterok:$T:$V --gres=gpu:1 --time=12:00:00 slurm/run.sbatch generate.sh $s)
    GEN="$GEN:$G"
done
sbatch --export=$E --dependency=afterok$GEN slurm/run.sbatch evaluate.sh
```

## Things that fail silently if changed

- **Polyphony.** The duet models read tensors tokenized with at most four
  notes per frame (`MAX_POLYPHONY=4`, which also selects
  `data/pop909_*_cp4_v2.pt` through `MELCHORD_CP`). The single-stream models
  use 16. Neither is stored in the checkpoint; a mismatch loads and runs but
  degrades the music. The scripts set both.
- **Chord program.** The duet models were trained with the chord stream on
  program 0, the single-stream models with chords on program 48 in a merged
  file. `stage_prompts.sh` prepares both versions of every prompt.
- **Excluded songs.** Songs 506 and 786 have no chord inside the prompt
  window and 456 has no Whole-Song Gen output; all systems skip them
  (`EXCLUDE_SONGS`).

## Citation

```bibtex
@inproceedings{li2026duet,
  title  = {Duet: Coordinating Pretrained Language Models for Joint Generation of Melody and Harmony},
  author = {Li, Xinyue and Xia, Gus and Wang, Ziyu},
  year   = {2026}
}
```

## Acknowledgements

The pretrained backbone is the symbolic-music foundation model of Jiang et
al., "Versatile Symbolic Music-for-Music Modeling via Function Alignment"
(ISMIR 2025). The data is POP909 (Wang et al., ISMIR 2020). The MoE RoFormer
is a modification of HuggingFace transformers (Apache-2.0; see
`third_party/transformers_roformer_overlay/LICENSE`). The external baselines
are cloned from their authors' repositories at setup time.
