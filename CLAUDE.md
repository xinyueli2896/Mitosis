# Mitosis — working conventions

## Always sbatch, never bare `python`

Every training, inference, preprocessing and evaluation command in this
repo runs on SLURM. **Do not hand the user a `python script.py ...`
command line**, not even for a quick one-off or a "just to listen" run.
If the thing you want to run has no sbatch wrapper, write one — that is
the fix, not a raw invocation.

The wrappers are not optional packaging. They carry setup the bare
command silently loses:

- `source ~/.bashrc` + `conda activate mitosis`
- `cd $REPO_DIR/midi_yinyang` (every relative path in the repo assumes it)
- `set -euo pipefail`, GPU health pre-flight, logs to `~/logs/%x_%j.out`
- task-derived `MAX_POLYPHONY` and the matching `MELCHORD_CP` export,
  which is what selects the dataset file — get this wrong and the run
  trains or decodes against the wrong tokenization without erroring

Existing wrappers, by stage:

| stage | scripts |
|---|---|
| preprocess | `preprocess_*.sbatch`, `combine_*.sbatch`, `build_*.sbatch` |
| train | `train_duet_{rehearsal,prefix,block,block_diffusion,anticipatory}.sbatch`, `train_intra_cross_attn.sbatch`, `finetune_pop909.sbatch` |
| inference | `infer_duet_{rehearsal,prefix,anticipatory,block_diffusion}.sbatch`, `infer_singlestream_e1.sbatch`, `infer_all_rwc.sbatch` |
| evaluate | `eval_e1.sbatch`, `eval_e2.sbatch`, `eval_e3.sbatch`, `eval_yinyang.sbatch`, `eval_a2_rope_ablation.sbatch` |
| diagnose | `diagnose_*.sbatch`, `check_*.sbatch`, `cp_capacity_check.sbatch` |

Knobs go through `--export=ALL,KEY=val`; sbatch flags (`--gres=gpu:2`,
`--exclude=gpu-50`) go before the script path and override the
`#SBATCH` directives inside it.

**This applies to submission too — no `bash run_everything.sh` helper
wrappers.** Multi-stage pipelines chain with sbatch's own dependency
mechanism, so every stage is still a plain `sbatch` the user can see,
reorder, or resubmit alone:

```bash
J1=$(sbatch --parsable <flags> midi_yinyang/preprocess_x.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 <flags> midi_yinyang/combine_x.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J2 <flags> midi_yinyang/train_x.sbatch)
```

`afterok` starts a stage only if the previous exited 0; since every
wrapper runs `set -euo pipefail`, a failure holds the rest of the chain
(they show `(DependencyNeverSatisfied)` in `squeue`) instead of running
on a broken artifact.

## Conventions that bite

- **`MAX_POLYPHONY` cannot be recovered from a checkpoint.** The local
  transformer uses a fixed `max_position_embeddings=4096`; nothing is
  sized by `subseq_len`. A cp mismatch loads cleanly and just degrades
  the music. Confirm from the training log's `Data for dataset ...`
  line, which prints the actual `.pt` path.
- **`--drum-folder` / `--nondrum-folder` are mod_a/mod_b SLOTS**, not
  drum semantics. On melchord they carry melody and chord; pass
  `MODE_NAME=mel2chord` so the output tree names the direction it holds.
- **Conditional models are direction-locked.** mod_a is fixed at
  training time, so chord→mel needs a `*_rev` task checkpoint. A forward
  ckpt run in reverse produces garbage without erroring.
- **`resolve_best_ckpt` scans a directory** and picks the smallest
  metric-tagged value. Keep one run per directory (`RUN_TAG=`), and note
  it refuses a directory mixing two monitored metrics.
