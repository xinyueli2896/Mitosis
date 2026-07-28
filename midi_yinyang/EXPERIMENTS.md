# Experiment plan — generation-mode comparison vs baselines

Goal: quantify what the two-stream (duet) architectures buy over
single-stream modeling, separately for each **generation mode**, on both
tasks. Every mode has its own natural baseline; comparing a mode against
the wrong baseline (e.g. co-generation against a conditional model)
answers nothing, so the matrix below keeps them apart.

Research questions:

- **RQ1 (co-generation)**: does explicit two-stream joint modeling beat
  a single merged-stream model at generating both parts together?
- **RQ2 (single-stream / marginal)**: does joint training degrade or
  improve the quality of ONE stream generated alone, vs a model that
  only ever saw that kind of material?
- **RQ3 (conditional)**: given the full partner stream, do the
  purpose-built conditional architectures (C.1/C.2) beat the symmetric
  models run in conditional mode, and how does the conditioning
  horizon (none / bounded / unbounded future) shape quality?
- **RQ4 (ablations)**: what does the simultaneous mutual within-frame
  conditioning strategy (同步看, realized by A.2's block-diffusion
  refinement) contribute — isolated at decode time, at training time,
  and combined — plus decode-schedule ablations.

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

### 2.2 Two-stream (duet) systems — main experiments

| ID | Model | Architecture | Generation modes | Training status |
|---|---|---|---|---|
| **A.2** | `M2CDuetBlockDiffusion` v1.1 | Interleaved two-stream joint AR backbone + two appended next-frame query slots trained as discrete denoisers across noise levels (diffusion_K = 4); inference runs K+1 parallel-refinement passes per frame, giving the two streams simultaneous mutual within-frame conditioning (同步看) | co, mel2chord, chord2mel, mel_only, chord_only; refinement depth K is a decode-time knob | drumnondrum: trained (98k). melchord: in training |
| **B.1** | `M2CDuetAnticipatory` | Joint-AR duet with the drum stream re-indexed k frames ahead, giving the partner stream k frames of future context | co, conditional (drum→nondrum) | drumnondrum: trained |
| **C.1** | `M2CDuetRehearsal` | Bidirectional conditioning-stream prefix ("rehearsal") + interleaved suffix; purpose-built conditional model | drum→nondrum only | drumnondrum: trained |
| **C.2** | `M2CDuetPrefix` | Prefix-LM: bidirectional conditioning stream, causal target stream; purpose-built conditional model | drum→nondrum only | drumnondrum: trained |

### 2.3 Ablation-only system

| ID | Model | Architecture | Role |
|---|---|---|---|
| **A.1** | `M2CIntraCrossAttn` (DuetAttn) | Interleaved two-stream joint AR; strict causal attention; NO query slots, NO refinement — each stream sees the partner only up to the previous position | Excluded from the main experiments. Used exclusively in the E4 ablation to isolate what A.2's 同步看 mechanism adds over plain sequential joint AR |

**Paper-ID ↔ code-name mapping (important).** The repository's
historical numbering differs from the paper IDs used here:

| Paper ID | Code / repo name | Repo-historical label |
|---|---|---|
| A.2 | `M2CDuetBlockDiffusion` (`A3_*` env knobs, `m2c_duet_block_diffusion*` ckpt dirs and scripts) | "A.3" |
| A.1 | `M2CIntraCrossAttn` (`m2c_intra_cross_attn*`) | "A.1" (unchanged) |
| — | `M2CDuetBlockAttn` (retired; query slots trained only fully-masked, collapsed at inference) | "A.2" — plays NO role in any experiment; the label is reused here for the diffusion model |

Filenames, environment variables (`A3_REFINE_STEPS`, …), and checkpoint
directories keep their historical names; only the experiment/paper
nomenclature changes.

Decode-variant notation: **A.2(K=n)** denotes A.2 decoded with n
refinement rounds; K=0 commits the AR-head draft with no slot
refinement (plain sequential joint AR on the A.2 backbone).

Disambiguation — the symbol K is overloaded and the two meanings must
not be conflated:

- **Training `diffusion_K` = 4** (fixed, a property of every A.2
  checkpoint): the number of noise-level bins the query slots were
  trained across (k ∈ {0..4}, independently per slot).
- **Inference K = `A3_REFINE_STEPS`** (free decode-time knob, the K in
  A.2(K=n)): how many refinement rounds the decoder runs per frame.
  Because training covered every noise level, any inference K in
  {0..diffusion_K} is in-distribution. All A.2(K=n) rows in this plan
  are the SAME checkpoint at different decode compute.

### 2.4 Baseline preparation requirements

- **S0/S1 prompts must carry distinct stream programs** (melody 0 /
  chord 48; drums are natively 127); otherwise the streams fuse
  irreversibly in token space. Prompt sources: RWC unsplit prompts
  (drumnondrum); the program-tagged combined POP909 folder, or
  same-program files via `cp_transformer_inference.py`'s automatic
  CHORD-track tagging (melchord).
- S0/S1 outputs are separated into streams with
  `split_melody_chord.py` before any stream-level metric is computed.
- Report S0 and S1 side by side on melchord: the S0→S1 gap isolates
  the value of in-domain data; the S1→A.2 gap isolates the value of
  architectural stream separation.

---

## 3. Experiment matrix

### E1 — Co-generation (RQ1)

Both streams are generated jointly, conditioned on a 4-bar prompt of
both streams.

| Task | Systems under test | Matched baseline | Mode |
|---|---|---|---|
| drumnondrum | A.2(K=1), B.1 | S0 (merged stream; same LA training corpus) | `co` |
| melchord | A.2(K=1) | S1 (merged stream; same POP909 data); S0 as unmatched anchor | `co` |

**Claim under test.** The duet architecture — per-stream Q/K/V/O
projections, dedicated cross-stream attention, and shared-MoE routing
over two interleaved streams — models two symbolic streams that follow
different musical grammars (drum vs pitched; melody vs chord) better
than a single-stream model that treats them as one homogeneous token
sequence distinguished only by program tokens. The baseline pairing is
MATCHED within each row (same training corpus), so the measured gap is
attributable to architecture alone.

**Pre-registered expectations** — how the co-generation advantage
should manifest, mapped to §5 metrics. The single-stream model carries
three burdens the duet removes; each removal predicts a specific
observable:

The three hypotheses partition cleanly: H1 judges WHAT each stream
plays (role identity), H2 judges HOW MUCH it plays over time (budget
stability), H3 judges how well the two streams FIT EACH OTHER. H1 and
H2 are independent failure modes — a merged model can keep every role
correct while the texture gradually thickens (H2 fails, H1 holds), or
keep counts stable while the melody turns polyphonic or drifts into
chord register (H1 fails, H2 holds); only outright stream death
violates both.

| # | Burden removed | Predicted observable | Metric |
|---|---|---|---|
| H1 | stream identity inferable only from program tokens (duet: identity is structural — the interleave position IS the stream) | role fidelity: melody stays monophonic, chords stay block-voiced, registers stay separated, neither stream dies; S\* shows role leakage or a stream vanishing mid-piece | per-stream polyphony profile; register overlap; silence/collapse rate; stream-survival length |
| H2 | both streams' notes compete inside one shared per-frame token budget (duet: each stream gets its own frame slot at every time step) | quantity stability: per-stream note density stays near the reference across all 24 bars; S\* thickens or thins with continuation length even when roles stay correct | density trajectory error as a function of bar index |
| H3 | partner notes are ordinary tokens scattered in one undifferentiated sequence — attention must LEARN to find them (duet: a dedicated cross-stream attention pass makes the partner architecturally addressable) | inter-stream fit: generated chords are consonant with the simultaneous melody; nondrum onsets lock to drum onsets — each stream can be individually well-formed and still fail this | chord-tone coverage of concurrent melody notes (melchord); onset synchrony (drumnondrum) |
| H4 | one set of weights multiplexing two grammars | stream-appropriate statistics (harmonic rhythm ≈ 1 chord change/bar; melodic contour smoothness); supporting evidence: MoE expert usage separating by stream | duration / harmonic-rhythm distributions; expert-routing side analysis |
| H5 | (net perceptual effect) | listening wins concentrated on the inter-stream-coherence axis rather than raw musicality | pairwise A/B axis (b) |

**Pre-registered counter-expectations** (stated now so results are not
cherry-picked later): S\* is expected to TIE on local plausibility and
pitch-class histograms (S1 saw the same data; merged AR is strong
locally) — the claim lives in the structural/cross-stream measures
H1–H3, not the local ones. A.2 may LOSE on the structure/development
listening axis due to its known repetition tendency; that axis is
reported but is not the claim under test.

Planned contrasts:

1. **S\* vs duet systems** — value of architectural stream separation
   (the headline RQ1 comparison), read through H1–H5 above.
2. **B.1 vs A.2(K=1)** — bounded-lookahead leader/follower design vs
   symmetric refinement design, as the two competing duet strategies.
   B.1's hypothesis is stream-asymmetric: the follower (nondrum)
   improves because it always harmonizes against k frames of committed
   drum future, while the leader (drum) is generated with
   k-frames-stale partner context. B.1's results are therefore ALWAYS
   reported per-stream; a joint aggregate would average away the
   effect under test. (The strict architecture-matched ablation pair
   for B.1 — A.1, which is exactly B.1 with k=0 — is reported in the
   E4 appendix, not here.)

### E2 — Marginal (single-stream) generation (RQ2)

Exactly one stream is generated; the partner stream is absent.

| Task | Systems under test | Baseline | Modes |
|---|---|---|---|
| drumnondrum | A.2 | S0 prompted with the corresponding single-stream files | `mel_only`, `chord_only` |
| melchord | A.2 | S1 prompted with the corresponding single-stream files | `mel_only`, `chord_only` |

Hypothesis under test: joint two-stream training does not degrade the
marginal distribution of a single stream relative to a model prompted
with identical single-stream material. A deficit would indicate
capacity interference from the joint objective.

### E3 — Conditional generation (RQ3)

The complete ground-truth partner stream is provided; the model
generates the remaining stream.

| Task | Systems under test | Baseline | Direction |
|---|---|---|---|
| drumnondrum | A.2 (conditional mode), B.1, C.1, C.2 | S0 (see protocol note) | drum → nondrum |
| melchord | A.2 (`mel2chord`, `chord2mel`) | S1 (see protocol note) | both directions |

**Conditioning-horizon spectrum (drum → nondrum).** The systems form
an ordered spectrum in how much FUTURE of the conditioning stream the
target stream can attend:

| System | Future drum context visible to nondrum_t | Streamable |
|---|---|---|
| A.2 (conditional mode) | 0 frames (same-frame only; the given stream is committed frame-by-frame) | yes |
| B.1 | k frames (bounded lookahead; k = 16 ≈ 1 bar) | yes, with k-frame latency |
| C.1 / C.2 | entire sequence (bidirectional) | no |

This ordering turns E3 from an unordered system comparison into a
dose–response study of conditioning horizon. The headline question:
does B.1's bounded lookahead recover most of the C.2−A.2 gap? If yes,
bounded (hence real-time-capable) conditioning suffices and the
offline bidirectional architectures buy little; if no, full-sequence
conditioning is genuinely load-bearing.

**B.1 direction restriction.** The reverse direction
(nondrum → drum) is EXCLUDED for B.1 by design: under the anticipatory
layout, drum_{t+k} attends nondrum only up to t−1, so even with the
full ground-truth nondrum available the model conditions on
k-frames-stale partner context — an expected-negative by construction,
not a fair capability test. (At most, report it once as an asymmetry
control, clearly labeled.) B.1 is likewise excluded from E2: its
marginals are not its design question, and the follower-only mode is
structurally handicapped by the missing leader stream.

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

### E4 — Ablations (RQ4)

#### E4a — Mutual within-frame conditioning (同步看) study: A.1 vs A.2

The central mechanism ablation. A.2 differs from A.1 in exactly two
coupled respects: (i) the training objective adds denoising query
slots, and (ii) decoding can run iterative refinement in which both
streams condition on each other's estimate of the SAME frame. The
study factorizes the total effect into those parts:

| Contrast | Isolates | Held fixed |
|---|---|---|
| A.2(K≥1) vs A.2(K=0) | the decode-time refinement mechanism | weights, tokenization, prompts |
| A.2(K=0) vs A.1 | the training-objective effect on the plain-AR pathway (parity check: a deficit means the diffusion objective taxed the backbone) | decode procedure (both plain joint AR) |
| A.1 vs A.2(K=1) | the combined package | task, prompts |

Tasks: drumnondrum (A.1 trained). Melchord requires training
A.1-melchord — decide after the drumnondrum ablation: if the combined
effect is large, the melchord replication is worth the GPU-time;
otherwise report drumnondrum only and run just A.2(K≥1) vs A.2(K=0)
(weights-shared, free) on melchord.

Also reported here: **B.1 vs A.1** (strict anticipation ablation —
A.1 is exactly B.1 with k=0), per-stream, supporting the E1 B.1
narrative.

#### E4b — Decode-schedule ablations (A.2 only)

Completed on drumnondrum via listening tests: the selected decode
schedule is K=1, final temperature 0.9, nucleus 0.95; refinement at
K=4 outperformed K=0 when both streams were active and matched or
underperformed it otherwise, motivating the A3_ADAPTIVE early-exit.
Replication on melchord:

1. **Refinement depth**: K ∈ {0, 1, 4} at the selected schedule,
   `co` mode, full evaluation list (doubles as the E4a first-row data).
2. **Adaptive early-exit**: A3_ADAPTIVE ∈ {off, on} at K=4. Melchord
   material contains few silent frames, so the expected result is a
   null effect; observing one confirms the mechanism is
   silence-specific rather than a general confound.
3. (Optional) **Temperature schedule**: piecewise draft/commit schedule
   vs linear annealing.

### E5 — (Optional) Anticipation-horizon sweep (B.1 only)

Run only if the E3 horizon-spectrum result is positive (B.1 recovers a
substantial fraction of the C.2−A.2 gap). Train B.1 at
k ∈ {8, 16, 32} (`ANTICIPATION_FRAMES` knob in
`train_duet_anticipatory.sbatch`; k=16 already trained) and plot the
E3 conditional metrics against k, with A.1 (ablation system) as the
k=0 point and C.2 as the k=∞ asymptote. Deliverable: one dose–response
figure locating the knee of the lookahead curve.

---

## 4. Decode settings (frozen before any scoring)

| System | Settings |
|---|---|
| S0/S1 | temperature 1.0, `--prompt-length 64 --gen-length 384 --n-samples 3` |
| A.2 | `A3_REFINE_STEPS=1 A3_FINAL_TEMP=0.9 A3_TOP_P=0.95` (schedule selected on drumnondrum) — used for ALL headline A.2 rows; the K-sweep is confined to E4 |
| A.1 (ablation only) | temperature 1.0 (its tuned default) |
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
  collapse mode of the retired `M2CDuetBlockAttn`)

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
  within a family (A.1 vs A.2 share tokenization+layout; S0 vs S1 share
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
1. A.2 melchord training to ≥50k steps; pick best-val ckpt.
2. Single-stream POP909 finetune (`finetune_pop909.sbatch`) → S1.
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
- E4a: A.1 drumnondrum runs (co + conditional, existing ckpt);
  A.1-melchord training deferred until the drumnondrum result is in.
- E4b: three additional A.2 melchord runs (K sweep + adaptive toggle).

Phase 2 — scoring: run `eval_metrics.py` over Phase-1 outputs; produce
the per-mode comparison tables.

Phase 3 — listening test on the E1/E3 shortlist; collate votes.

Phase 4 — write-up: one table per E-block (rows = systems, columns =
metrics + listening win-rate), plus the E4 figures (metric vs K; the
A.1/A.2 factorization bar chart).

---

## 7. Output layout convention

```
temp/eval_<task>/
    <system>/<mode>/<song>/sample<i>.mid          (+ split/ for S* outputs)
results/
    metrics_<task>.csv
    listening_votes.csv
```

System directory names use the PAPER IDs (S0, S1, A.2, B.1, C.1, C.2,
and A.1 only under an `ablation/` prefix), regardless of the historical
script/ckpt names.

Keep every generated file until the write-up is done — re-listening
beats re-generating.
