# Experiment plan — generation-mode comparison vs baselines

Goal: quantify what the two-stream (duet) architectures buy over
single-stream modeling, separately for each **generation mode**, on both
tasks. Every mode has its own natural baseline; comparing a mode against
the wrong baseline (e.g. co-generation against a conditional model)
answers nothing, so the matrix below keeps them apart.

Research questions:

- **RQ1 (co-generation)**: does explicit two-stream joint AR beat a
  single merged-stream model at generating both parts together?
  Does A.3's within-frame refinement add anything over plain joint AR?
- **RQ2 (single-stream / marginal)**: does joint training degrade or
  improve the quality of ONE stream generated alone, vs a model that
  only ever saw that kind of material?
- **RQ3 (conditional)**: given the full partner stream, do the
  purpose-built conditional architectures (C.1/C.2) beat the symmetric
  models run in conditional mode, and do all of them beat the
  single-stream baseline given the same information in-context?
- **RQ4 (ablations)**: refinement depth K, adaptive skipping, decode
  schedule — already partially answered on drumnondrum; replicate the
  winner settings on melchord to test generality.

---

## 1. Tasks, data, splits

| | drumnondrum | melchord |
|---|---|---|
| streams (mod_a / mod_b) | drum / nondrum | melody / chord |
| training corpus | LA (la_*_cp16_v2) | POP909 (pop909_*_cp4_v2) |
| test prompts | RWC (`input/rwc_test_prompts_split/{drum,nondrum}`) — fully held-out corpus | POP909 held-out songs (below) |
| polyphony | 16 | 4 (duet) / 16 (single-stream combined) |

**POP909 held-out set.** Split conventions differ between codebases:
the duet `FramedDataset` (cp_transformer_m2c_moe) holds out
`idx % 10 == 0`; the base `FramedDataset` (cp_transformer) uses
`% 10 == 1` (val) and `% 10 == 0` (test). Songs with sorted index
`% 10 == 0` — ids 001, 011, 021, … — are unseen by BOTH the duet
training and the single-stream finetune. **Use only those ids for all
POP909 evaluation.** Suggested fixed eval list (10 songs):
`001 011 021 031 041 051 061 071 081 091`.

**Prompt protocol (all experiments).** 4-bar prompt = 64 frames,
384 total frames (24 bars), i.e. 20 bars of continuation.
n = 3 samples per (song, system, mode). Temperature 1.0 unless a
system's tuned decode schedule says otherwise (§4).

---

## 2. Systems inventory

Trained / available:

| id | system | task coverage | ckpt |
|---|---|---|---|
| S0 | single-stream pretrained (v0.42 size1) | both (zero-shot) | `ckpt/cp_transformer_v0.42_..._schedule.epoch.00.fin.ckpt` |
| S1 | single-stream POP909-finetune | melchord | `ckpt/cp_transformer_v0.42_size1_pop909ft_batch_48_schedule/` (finetune sbatch) |
| A1 | M2CIntraCrossAttn (DuetAttn) | drumnondrum (trained); melchord (would need training — optional) | `ckpt/m2c_intra_cross_attn_v1.0_...` |
| A3 | M2CDuetBlockDiffusion v1.1 | drumnondrum (98k); melchord (in training) | `ckpt/m2c_duet_block_diffusion_v1.1_..._{drumnondrum,melchord}_...` |
| B1 | M2CDuetAnticipatory | drumnondrum | existing run dir |
| C1 | M2CDuetRehearsal | drumnondrum | existing run dir |
| C2 | M2CDuetPrefix | drumnondrum | existing run dir |

Baseline preparation notes:

- **S0/S1 need merged prompts with distinct stream programs** (melody 0 /
  chord tag 48; drums are natively 127) or the streams fuse. Prompt
  folders: RWC unsplit prompts for drumnondrum; for melchord either the
  tagged combined folder or same-program files via the inference
  script's automatic CHORD-track tagging. Outputs are split back into
  streams with `split_melody_chord.py` for stream-level metrics.
- S1 is the *matched* single-stream baseline for melchord (saw POP909);
  S0 is the *unmatched* lower anchor. Report both.

---

## 3. Experiment matrix

### E1 — Co-generation (RQ1)

Both streams generated jointly from a 4-bar prompt of both.

| task | systems | mode |
|---|---|---|
| drumnondrum | S0 (merged) vs A1 vs A3(K=0) vs A3(K=1) vs B1 | `co` |
| melchord | S0 vs S1 (merged) vs A3(K=0) vs A3(K=1) [vs A1-melchord if trained] | `co` |

Key contrasts: S* vs A* (does stream separation help at all);
A3(K=0) vs A1 (is the A.3 backbone as good as the reference when run
as plain AR); A3(K≥1) vs A3(K=0) (does refinement help — expected to
matter most when both streams are active; melchord always has both
active, so this is the cleanest test of refinement yet).

### E2 — Single-stream / marginal generation (RQ2)

Only one stream generated; the other absent (silence).

| task | systems | modes |
|---|---|---|
| drumnondrum | A1, A3 (`mel_only`, `chord_only`) vs S0 prompted with drum-only / nondrum-only files | one stream each |
| melchord | A3 (`mel_only`, `chord_only`) vs S1 prompted with melody-only / chord-only files | one stream each |

Question: is the duet model's marginal P(one stream) intact, or did
joint training siphon capacity? Baseline = single-stream model prompted
with exactly the same one-stream material.

### E3 — Conditional generation (RQ3)

Full ground-truth partner stream given; generate the other stream.

| task | systems | direction |
|---|---|---|
| drumnondrum | C1, C2, B1, A1(`mel2chord`), A3(`mel2chord`) | drum → nondrum |
| melchord | A3 `mel2chord` and `chord2mel` vs S1-conditional (below) | both directions |

**S1-conditional baseline**: the single-stream model cannot condition
on the future of the partner stream; give it the fairest equivalent —
merged prompt where the conditioning stream continues (spliced in) and
generation is constrained to... not expressible. So instead report the
honest version: S1 co-generates from the same prompt and we *discard*
its partner stream, scoring only the generated stream against the
conditional systems' outputs. Document this asymmetry explicitly — the
conditional systems see strictly more information; the gap measures the
value of that information + architecture.

Conditional-specific scoring (in addition to §5): agreement between the
generated stream and the *ground-truth* stream it replaces
(the reference continuation is known for every eval song).

### E4 — Ablations (RQ4, A.3 only)

Already run on drumnondrum (listening): decode schedule winner
K=1 / final_temp 0.9 / top_p 0.95; K=4 > K=0 with drums present,
K=4 ≤ K=0 without; A3_ADAPTIVE implemented in response. Replicate on
melchord:

1. K ∈ {0, 1, 4} at the winner schedule, `co` mode, eval list above.
2. A3_ADAPTIVE on/off at K=4 (melchord rarely has silent frames —
   expect no effect; a null result here confirms the mechanism is
   silence-specific and not a confound).
3. (optional) draft-temp piecewise schedule vs linear anneal.

---

## 4. Decode settings (frozen before any scoring)

| system | settings |
|---|---|
| S0/S1 | temperature 1.0, `--prompt-length 64 --gen-length 384 --n-samples 3` |
| A1 | temperature 1.0 (its tuned default) |
| A3 | `A3_REFINE_STEPS=1 A3_FINAL_TEMP=0.9 A3_TOP_P=0.95` (drumnondrum winner) — used for ALL headline A3 rows; K-sweep only inside E4 |
| B1/C1/C2 | temperature 1.0, existing script defaults |

Freeze these; no per-song or per-metric cherry-picking. Any change ⇒
rerun the whole grid (it's cheap: ≤ ~15 songs × ~6 systems × 3 samples).

---

## 5. Metrics

### 5.1 Objective — needs a small new module (`eval_metrics.py`, TODO)

Computed per generated continuation (frames 64..384), against the
ground-truth continuation of the same song where applicable. All
stream-level metrics run on split streams (`split_melody_chord.py` for
S* outputs; duet outputs are natively split).

Per-stream:
- pitch-class histogram overlap vs reference continuation (per song)
- note density / polyphony trajectory error vs reference
- duration distribution distance
- repetition: max 4-bar self-similarity over the continuation
  (catches the known repetition failure mode)
- silence/EOS collapse rate (fraction of empty frames — catches the
  known A.2-style collapse mode)

Joint / cross-stream:
- melchord: chord-tone coverage — fraction of melody notes whose pitch
  class ∈ concurrent chord's pc-set (score both generated-vs-generated
  and generated-vs-ground-truth partner)
- drumnondrum: onset synchrony — fraction of nondrum onsets within
  1 frame of a drum onset, compared to the reference's rate
- conditional (E3): frame-wise pc-set Jaccard between generated stream
  and the ground-truth stream it replaces

Model-based:
- teacher-forced NLL of ground-truth continuations. Comparable ONLY
  within a family (A1 vs A3 share tokenization+layout; S0 vs S1 share
  theirs). Never compare NLL across families — different token spaces.

### 5.2 Subjective — listening test

- Material: 8 POP909 eval songs × {E1 systems} + 5 RWC songs ×
  {E1, E3 systems}; 1 sample per cell (pre-drawn at fixed seed order,
  not hand-picked).
- Protocol: blind pairwise A/B per axis, randomized order, axes:
  (a) overall musicality, (b) inter-stream coherence (do the parts fit
  each other), (c) structure over 24 bars (repetition vs development).
- Raters: ≥3 (you + labmates); report per-axis win rates with a sign
  test. Pairwise beats MOS at this scale (few raters, few items).

### 5.3 Statistics

Per-song paired comparisons (same prompts across systems). Report
mean ± std over songs; significance via Wilcoxon signed-rank on
per-song means (3 samples averaged first). With ~10-15 songs only
large effects will clear p<0.05 — treat objective metrics as
directional evidence and let the listening test carry the headline.

---

## 6. Execution order

Phase 0 — prerequisites
1. A.3 melchord training to ≥50k steps; pick best-val ckpt.
2. Single-stream POP909 finetune (`finetune_cp_transformer_pop909.sbatch`).
3. Build POP909 eval prompt folders for ids `001 011 ... 091`:
   split melody/chord folders (already exist) + tagged combined folder
   (for S0/S1). RWC prompt folders already exist.
4. Write `eval_metrics.py` (§5.1) + a results collator (one CSV row per
   sample: task, mode, system, song, sample, metrics...).

Phase 1 — generation sweeps (all sbatch, mostly existing scripts)
- E1/E2/E3 duet lanes: `infer_all_rwc.sbatch` (drumnondrum) and
  `infer_duet_block_diffusion.sbatch` with melchord ckpt +
  `MEL_FOLDER/CHORD_FOLDER/MAX_POLYPHONY=4` (melchord).
- E1/E2 baselines: `cp_transformer_inference.py` runs per prompt folder
  (S0 and S1 ckpts).
- E4: three more A3 melchord runs (K sweep + adaptive toggle).

Phase 2 — scoring: run `eval_metrics.py` over Phase-1 outputs; produce
the per-mode comparison tables.

Phase 3 — listening test on the E1/E3 shortlist; collate votes.

Phase 4 — write-up: one table per E-block (rows = systems, columns =
metrics + listening win-rate), plus the E4 ablation figure (metric vs K).

---

## 7. Output layout convention

```
temp/eval_<task>/
    <system>/<mode>/<song>/sample<i>.mid          (+ split/ for S* outputs)
results/
    metrics_<task>.csv
    listening_votes.csv
```

Keep every generated file until the write-up is done — re-listening
beats re-generating.
