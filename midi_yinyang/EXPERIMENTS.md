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

## 2. Model nomenclature

All experiments refer to systems by the identifiers below. The
identifier is used verbatim in result tables, output directories, and
the write-up; no other aliases are permitted.

### 2.1 Baseline systems (single-stream)

| ID | Model | Architecture | Training data | Role |
|---|---|---|---|---|
| **S0** | `RoFormerSymbolicTransformer` v0.42 (size 1) | Single-stream CP transformer; one merged token stream; stream identity carried only by the MIDI program token | LA (large scraped MIDI corpus) | Unmatched lower anchor: no stream separation, no in-domain data |
| **S1** | S0 + POP909 finetune | Identical architecture; weights finetuned on merged, program-tagged POP909 melody+chord | LA → POP909 (20k steps) | Matched single-stream baseline for melchord: in-domain data, still no architectural stream separation |

### 2.2 Two-stream (duet) systems

| ID | Model | Architecture | Generation modes | Training status |
|---|---|---|---|---|
| **A.1** | `M2CIntraCrossAttn` (DuetAttn) | Interleaved two-stream joint AR; strict causal attention; intra- and cross-stream SDPA passes | co, mel2chord, chord2mel, mel_only, chord_only | drumnondrum: trained. melchord: optional (train only if E1-melchord needs the reference point) |
| **A.3** | `M2CDuetBlockDiffusion` v1.1 | A.1 + two appended next-frame query slots trained as discrete denoisers across noise levels K; inference runs K+1 parallel-refinement passes per frame | same five modes; refinement depth K is a decode-time knob | drumnondrum: trained (98k). melchord: in training |
| **B.1** | `M2CDuetAnticipatory` | A.1 with the drum stream re-indexed k frames ahead, giving the partner stream k frames of future context | co, conditional | drumnondrum: trained |
| **C.1** | `M2CDuetRehearsal` | Bidirectional conditioning-stream prefix ("rehearsal") + interleaved suffix; purpose-built conditional model | drum→nondrum only | drumnondrum: trained |
| **C.2** | `M2CDuetPrefix` | Prefix-LM: bidirectional conditioning stream, causal target stream; purpose-built conditional model | drum→nondrum only | drumnondrum: trained |

Decode-variant notation: **A.3(K=n)** denotes A.3 decoded with n
refinement rounds; K=0 reduces to plain joint AR on the A.3 backbone
(architecture ablation of the refinement mechanism at inference time).

### 2.3 Baseline preparation requirements

- **S0/S1 prompts must carry distinct stream programs** (melody 0 /
  chord 48; drums are natively 127); otherwise the streams fuse
  irreversibly in token space. Prompt sources: RWC unsplit prompts
  (drumnondrum); the program-tagged combined POP909 folder, or
  same-program files via `cp_transformer_inference.py`'s automatic
  CHORD-track tagging (melchord).
- S0/S1 outputs are separated into streams with
  `split_melody_chord.py` before any stream-level metric is computed.
- Report S0 and S1 side by side on melchord: the S0→S1 gap isolates
  the value of in-domain data; the S1→A.3 gap isolates the value of
  architectural stream separation.

---

## 3. Experiment matrix

### E1 — Co-generation (RQ1)

Both streams are generated jointly, conditioned on a 4-bar prompt of
both streams.

| Task | Systems under test | Baseline | Mode |
|---|---|---|---|
| drumnondrum | A.1, A.3(K=0), A.3(K=1), B.1 | S0 (merged stream) | `co` |
| melchord | A.3(K=0), A.3(K=1); A.1-melchord if trained | S0, S1 (merged stream) | `co` |

Planned contrasts:

1. **S\* vs duet systems** — value of architectural stream separation.
2. **A.3(K=0) vs A.1** — parity check: the A.3 backbone decoded as
   plain joint AR should match the reference architecture; a deficit
   indicates the diffusion objective taxed the AR pathway.
3. **A.3(K≥1) vs A.3(K=0)** — contribution of within-frame iterative
   refinement. Prior drumnondrum listening results indicate refinement
   helps only when both streams are active; melchord has both streams
   active throughout, making it the cleanest test of this mechanism.

### E2 — Marginal (single-stream) generation (RQ2)

Exactly one stream is generated; the partner stream is absent.

| Task | Systems under test | Baseline | Modes |
|---|---|---|---|
| drumnondrum | A.1, A.3 | S0 prompted with the corresponding single-stream files | `mel_only`, `chord_only` |
| melchord | A.3 | S1 prompted with the corresponding single-stream files | `mel_only`, `chord_only` |

Hypothesis under test: joint two-stream training does not degrade the
marginal distribution of a single stream relative to a model prompted
with identical single-stream material. A deficit would indicate
capacity interference from the joint objective.

### E3 — Conditional generation (RQ3)

The complete ground-truth partner stream is provided; the model
generates the remaining stream.

| Task | Systems under test | Baseline | Direction |
|---|---|---|---|
| drumnondrum | C.1, C.2, B.1, A.1, A.3 (conditional mode) | S0 (see protocol note) | drum → nondrum |
| melchord | A.3 (`mel2chord`, `chord2mel`) | S1 (see protocol note) | both directions |

**Baseline protocol note.** A single-stream AR model cannot condition
on the future of the partner stream; no faithful conditional decoding
exists for S0/S1. The reported baseline is therefore: S\* co-generates
from the same prompt, the partner stream is discarded, and only the
generated target stream is scored. This asymmetry is disclosed in the
write-up: the conditional systems observe strictly more information,
and the measured gap quantifies the combined value of that information
and of the conditioning architecture.

Conditional-specific scoring (in addition to §5): agreement between
the generated stream and the ground-truth stream it replaces (the
reference continuation is available for every evaluation song).

### E4 — Decode-time ablations (RQ4; A.3 only)

Completed on drumnondrum via listening tests: the selected decode
schedule is K=1, final temperature 0.9, nucleus 0.95; refinement at
K=4 outperformed K=0 when both streams were active and matched or
underperformed it otherwise, motivating the A3_ADAPTIVE early-exit.
Replication on melchord:

1. **Refinement depth**: K ∈ {0, 1, 4} at the selected schedule,
   `co` mode, full evaluation list.
2. **Adaptive early-exit**: A3_ADAPTIVE ∈ {off, on} at K=4. Melchord
   material contains few silent frames, so the expected result is a
   null effect; observing one confirms the mechanism is
   silence-specific rather than a general confound.
3. (Optional) **Temperature schedule**: piecewise draft/commit schedule
   vs linear annealing.

---

## 4. Decode settings (frozen before any scoring)

| system | settings |
|---|---|
| S0/S1 | temperature 1.0, `--prompt-length 64 --gen-length 384 --n-samples 3` |
| A.1 | temperature 1.0 (its tuned default) |
| A.3 | `A3_REFINE_STEPS=1 A3_FINAL_TEMP=0.9 A3_TOP_P=0.95` (schedule selected on drumnondrum) — used for ALL headline A.3 rows; the K-sweep is confined to E4 |
| B.1/C.1/C.2 | temperature 1.0, existing script defaults |

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
  within a family (A.1 vs A.3 share tokenization+layout; S0 vs S1 share
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
- E4: three additional A.3 melchord runs (K sweep + adaptive toggle).

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
