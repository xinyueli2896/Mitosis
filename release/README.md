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

## Running under SLURM

`slurm/run.sbatch` runs any stage script; resources go on the command line:

```bash
J1=$(sbatch --parsable --gres=gpu:1 --time=12:00:00 slurm/run.sbatch generate.sh duet)
sbatch --dependency=afterok:$J1 slurm/run.sbatch evaluate.sh
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
