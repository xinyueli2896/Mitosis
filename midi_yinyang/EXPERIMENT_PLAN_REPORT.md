# Evaluating Two-Stream Symbolic Music Generation: Experiment Plan Report

*Companion document to `EXPERIMENTS.md` (the operational protocol).*

---

## 1. Setup

**Tasks** — generation of symbolic music with two paired streams:

1. Melody and chord pair. **Training**: POP909 (~900 songs) + Nottingham (~1,000 two-track tunes) combined, roughly doubling the melody/chord training data. **Evaluation**: POP909-only held-aside test split (`input/pop909_split/{melody,chord}`), so metrics stay in the target domain.
2. Drum and non-drum pair (LA corpus; evaluated on held-out RWC)

**Generation modes:**

1. Co-generation (co-gen)
2. Marginal generation (marg. gen)
3. Conditional generation (cond. gen)

**Baselines:**

| ID | Construction |
|---|---|
| **S0** | Pretrained single-stream CP transformer, trained on the LA dataset |
| **S1** | S0 finetuned on the COMBINED POP909+Nottingham melody–chord data (merged streams) — the same corpus the duet systems train on |
| **S-mel / S-chord** | S0 finetuned on ONE stream of the combined melody–chord data |
| **Y** | YinYang: frozen S0 + low-rank cross-attention adapters + LoRA; conditional generation |

**Our models:**

| ID | Model |
|---|---|
| **A.1** | DuetAttn (`M2CIntraCrossAttn`) |
| **A.2** | DuetBlockDiffusion (`M2CDuetBlockDiffusion`) |
| **B.1** | DuetAnticipatory (`M2CDuetAnticipatory`) |
| **C.1** | `M2CDuetRehearsal` |
| **C.2** | `M2CDuetPrefix` |

---

## 2. Research questions

- **RQ1 — Co-generation.** Does explicit two-stream joint modeling
  beat a merged-stream model at generating both parts together?
- **RQ2 — Marginals.** Does joint training *retain* the ability to
  generate each stream alone (a non-inferiority question, not a
  superiority one)?
- **RQ3 — Conditional generation.** Given the full partner stream, how
  does the *conditioning horizon* (no future / bounded future /
  unbounded future) shape quality, and do fully-trained conditional
  duets beat parameter-efficient adapter conditioning?
- **RQ4 — Mechanisms.** Which architectural components are causally
  load-bearing: the within-frame refinement mechanism, the training
  objective behind it, and the MoE routing?

---

## 3. Experiment blocks

Per-experiment training requirements (checkpoints are reused across
experiments; only the first introducing row carries a training cost;
**bold** = new runs currently owed):

| Exp. | System | Training data | Status |
|---|---|---|---|
| E1 | A.2 (drum/non-drum) | LA drum + non-drum pair (`la_drum_cp16_v2` / `la_nondrum_cp16_v2`) | trained (98k) |
| E1 | A.2 (melody/chord) | COMBINED melody + chord pair (POP909 + Nottingham, cp4) | trained |
| E1 | B.1 (drum/non-drum) | LA drum + non-drum pair | trained |
| E1 | B.1 (melody/chord) | COMBINED melody + chord pair, k=16 | trained |
| E1 | S0 | LA merged single stream (pretraining) | trained |
| E1 | S1 | COMBINED merged, program-tagged (cp16); finetune of S0 | trained (short recipe) |
| E2 | A.2 | — (reuses the E1 checkpoints) | — |
| E2 | S-mel | COMBINED melody-only stream; finetune of S0, short recipe | **to train** |
| E2 | S-chord | COMBINED chord-only stream; finetune of S0, short recipe | **to train** |
| E2 | S-drum / S-nondrum | LA drum-only / non-drum-only; finetunes of S0 | gated on melchord E2 |
| E3 | A.2, B.1 | — (reuse the E1 checkpoints) | — |
| E3 | C.1, C.2 | LA drum + non-drum pair | trained |
| E3 | Y-dn | LA drum-subset, 31k songs (adapters + LoRA on frozen S0) | existing paper ckpt |
| E3 | Y-mc | Nottingham, 1,020 songs (adapters + LoRA on frozen S0) | existing paper ckpt |
| E3 | (optional domain match) | a POP909-matched YinYang finetune, or a Nottingham-only eval slice; Nottingham is now in the duet training mix, so Y-mc is no longer the only system seeing folk data | optional |
| E4a | A.1 (drum/non-drum) | LA drum + non-drum pair | trained |
| E4a | A.1 (melody/chord) | COMBINED melody + chord pair | gated on drum/non-drum E4a |
| E4b | — | decode-time only; reuses the A.2-melchord checkpoint | — |
| E4c | A.2-dense | COMBINED melody + chord pair (identical recipe to A.2-melchord) | **to train** |
| E5 | B.1 at k ∈ {8, 32} | LA drum + non-drum pair (k=16 exists) | gated on E3 |


### E1 — Co-generation (headline)

Both streams are generated jointly from a 4-bar prompt of both.
Corpus-matched pairing: A.2-drumnondrum vs **S0** (both LA-trained);
A.2-melchord vs **S1** (both trained on the combined POP909+Nottingham corpus) — so the measured gap is
attributable to architecture alone. B.1 participates per-stream as the
competing duet strategy.

Four pre-registered hypotheses map each burden the single-stream model
carries to a specific observable:

| # | Judges | Prediction | Primary metric |
|---|---|---|---|
| H1 | each stream alone | consistent role & texture across 24 bars (melody monophonic, chords block-voiced, stable density, no stream death); S\* shows role leakage, density drift, or stream death | `survival_min` |
| H2 | mutual fit | chords consonant with the simultaneous melody; non-drum onsets lock to drums | `chord_tone_cov_delta` / `onset_sync_delta` |
| H3 | grammar | stream-appropriate statistics (harmonic rhythm ≈ 1 change/bar, contour smoothness); MoE experts separate by stream | `harmonic_rhythm_jsd` / `onset_grid_jsd` |
| H4 | listeners | an ORDERING of listening margins: coherence > musicality ≥ 0 ≈ structure — a uniform profile would indicate generic quality, not the mechanism | pairwise A/B per axis |

Pre-registered counter-expectations: ties on local plausibility and
pitch-class histograms (the matched baseline saw the same data);
possible loss on the structure axis (A.2's known repetition tendency).

### E2 — Marginal generation (guard; retention, not superiority)

Claim: joint training did not cost the model its marginals. Opponent:
the per-stream **specialist** — same pretrained initialization, same
finetune recipe, same songs, single-modality data — the strictest fair
opponent (each devotes full capacity to one marginal). Success
criterion: non-inferiority within a pre-registered margin (δ = 0.05 on
primary JSD grammar metrics). Outside-margin results are reported as
the *quantified marginal cost* of joint training; a better-than-
specialist result is reported as positive cross-stream transfer.

Mechanism note (pre-registered): in marginal mode the absent stream is
**clamped** to explicit silence (never sampled), and under the frozen
adaptive decode the refinement loop exits every step — marginal-mode
A.2 is therefore its AR pathway. Disclosed risk: an all-silent partner
is out-of-distribution for the melody/chord model — chords are
essentially always active in BOTH corpora of the combined mix
(Nottingham tunes carry a rendered chord track throughout, and the ~13
melody-only tunes were excluded in preprocessing so the streams stayed
paired), so adding Nottingham does NOT relieve this risk — while it is
in-distribution for drum/non-drum, where drum-less songs occur
naturally. The two-task contrast remains the built-in diagnostic
separating capacity interference from distribution shift.

### E3 — Conditional generation (horizon dose–response)

The full ground-truth partner is given; the model generates the other
stream. The systems form an **ordered conditioning-horizon spectrum**:

| Horizon of visible partner future | System | Streamable |
|---|---|---|
| 0 frames (same-frame) | A.2 conditional mode | yes |
| k = 16 frames (1 bar) | B.1 | yes, k-frame latency |
| unbounded (bidirectional) | C.1, C.2, Y | no |

Scoring separates two notions of quality: **fit to the given stream**
(primary — what conditioning is for; computed by the existing metric
module on the pair ⟨given ground truth, generated⟩) and agreement with
the original target (secondary only — many accompaniments are valid
for one partner). Hypotheses: **H-E3.1**, a within-family sanity gate
(each system must beat its own co-generation output scored against the
ground-truth partner — same model, same decode, only the conditioning
differs); **H-E3.2**, monotone improvement along the horizon spectrum,
with the headline sub-question being the fraction of the C.2−A.2 gap
B.1 recovers (a large fraction ⇒ bounded, streamable conditioning
suffices); **H-E3.3**, fully-trained duets vs adapters at fixed
unbounded horizon — deliberately registered without a directional
prior, as both outcomes are informative.

### E4 — Ablations

**E4a — mechanism (同步看) factorization.** A.2 differs from A.1 in a
coupled objective + decode change; three contrasts separate them:
A.2(K≥1) vs A.2(K=0) isolates the decode-time refinement (weights
shared — free); A.2(K=0) vs A.1 is the training-objective parity check
(expected parity; deficit = objective tax, surplus = beneficial
auxiliary task); A.1 vs A.2(K=4) is the combined package (expected ≈
additive). The refinement gain is predicted to be **metric-selective**
(landing on H2, the fit block) — a uniform gain would indicate generic
decode compute rather than the claimed mechanism. Disclosed confound:
A.1 and A.2 are separate training runs; their step counts are recorded
and any imbalance stated with its bias direction.

**E4b — decode schedule (melody/chord replication).** K ∈ {0, 1, 4}
(expected: clearer monotone H2 benefit than on drum/non-drum, since
both streams are always active; a flat curve would weaken the K=4
headline choice); adaptive early-exit on/off at K=4 (expected null on
melody/chord — confirming the mechanism is silence-specific); optional
temperature-schedule comparison.

**E4c — MoE ablation.** A.2 vs **A.2-dense** (compute-matched dense
FFN), identical otherwise. Predicted deficit profile is the mirror
image of E4a's: landing on H3 (grammar) and secondarily H1, with H2
least affected — the coupling is carried by the attention structure
both variants share. Combined with the observational expert-routing
analysis, a positive result closes the specialization story from both
ends; a null is reported as MoE-not-load-bearing.

### E5 — Anticipation-horizon sweep (optional, gated)

Only if E3 shows B.1 recovering a substantial fraction of the
unbounded-horizon gap: train B.1 at k ∈ {8, 16, 32} and plot the
conditional fit against k, with A.1 as the k=0 point and C.2 as the
k=∞ asymptote — one dose–response figure locating the knee of the
lookahead curve.

---

## 4. Evaluation methodology

### 4.1 Objective metrics (implemented: `eval_metrics.py`)

All metrics are **reference-calibrated**: distributional metrics are
Jensen–Shannon divergences against the ground-truth continuation's
distributions; fit metrics are deltas against the same statistic
computed on the reference pair; density is a ratio plus a drift slope.
Scoring is confined to the continuation window (frames 64–384).
Reporting follows the pre-registered priority ordering **H3 > H2 >
H1**, with one primary endpoint per hypothesis; the remaining metrics
are supporting diagnostics. The module was validated on synthetic
fixtures in which each engineered failure mode (grammar chaos, stream
death) trips exactly its own hypothesis block.

### 4.2 Listening test

Blind pairwise A/B comparisons on three axes — (a) overall musicality,
(b) inter-stream coherence, (c) 24-bar structure — with ≥3 raters,
randomized order, and samples pre-drawn at fixed seed order (no
curation). Win rates per axis with a sign test. The H4 prediction is
about the *profile* across axes, not the overall win rate.

### 4.3 Frozen decode settings

One configuration per system, frozen before scoring: A.2 at
`K=4, final-temp 0.9, top-p 0.95, adaptive on` (K=4 exercises the full
trained refinement depth; the temperature schedule is the prior
listening-test selection; the adaptive guard covers the known
silent-frame regression and is itself ablated in E4b). All other
systems at temperature 1.0 with their script defaults. Any change to
these settings invalidates and reruns the affected grid.

### 4.4 Statistics

Paired per-song comparisons on identical prompts; three samples
averaged per song before testing; Wilcoxon signed-rank on per-song
means. With 10–15 evaluation songs, only large effects reach
conventional significance — objective metrics are treated as
directional evidence and the listening test carries the headline
perceptual claims.

---

## 5. Threats to validity (all pre-registered)

1. **Y-mc domain mismatch** — the melody/chord YinYang is
   Nottingham-trained while evaluation is POP909; it is reported as an
   external reference, never as a matched baseline. A POP909-matched
   YinYang finetune is an identified optional upgrade.
2. **E4a training-budget imbalance** — A.1 and A.2 are separate runs;
   step counts are recorded and any imbalance disclosed with its bias
   direction.
3. **E2 architecture non-constancy** — there is no non-degenerate
   "A.2 trained marginally," so E2 compares the joint system's
   marginal against the single-stream pipeline's best specialist; the
   specialist also sees half the tokens per song (an asymmetry that
   makes an A.2 win evidence of positive transfer, not a confound).
4. **Multiple valid accompaniments** — ground-truth agreement is never
   a primary conditional endpoint; fit-to-given is.
5. **Small evaluation set** — 10 POP909 + ~5–15 RWC songs bounds
   statistical power; mitigated by paired design, three samples per
   cell, and the listening test as the perceptual arbiter.
6. **Marginal-mode decode path** — under the frozen adaptive setting,
   E2's A.2 outputs come from the AR pathway (refinement exits on the
   clamped-silent partner); stated so results are attributed to the
   right mechanism.

---

## 6. Execution roadmap

**Phase 0 — prerequisites** (current phase)
1. A.2 melody/chord training to best-validation checkpoint *(in
   progress)*.
2. S1 merged finetune; S-mel and S-chord specialist finetunes
   (existing script, distinct run tags). *Status note: the first S1
   run (20k steps, max-LR 2e-5, val every 250 steps) overfit
   immediately — POP909 train is ~20 steps/epoch at this batch, so the
   first val measurement landed ~12 epochs in, past the minimum, and
   val loss rose monotonically 0.37 → 0.74 while train loss fell to
   0.10. The recipe was retuned (2k-step schedule, max-LR 1e-5, val
   every 25 steps) to capture the early domain-adaptation dip; the
   same short recipe applies to the specialists. **Rerun outcome
   (batch 48): val descended 0.39 → ~0.358 over ~1k steps and
   plateaued with no overfitting upturn — a genuine ~8% in-domain NLL
   gain. S1 := the best-val checkpoint of the `short` run (record its
   exact step and val loss from the ckpt filename).* 
3. E4c: A.2-dense melody/chord training run (sbatch knobs in place).
4. Evaluation prompt folders for the ten held-out POP909 ids; RWC
   folders already exist.
5. Locate the Y-dn / Y-mc checkpoints.
   *(Metric module: complete.)*

**Phase 1 — generation sweeps.** All inference via existing sbatch
scripts: the duet lanes, the S\*/specialist lanes, the YinYang lanes,
the E4a A.1 runs (existing checkpoint), and the E4b decode grid.
A.1-melchord training and the drum/non-drum specialists are
**gated** on the corresponding first-task results.

**Phase 2 — scoring.** Manifest construction; `eval_metrics.py` over
all outputs; per-mode comparison tables in the pre-registered priority
order.

**Phase 3 — listening test** on the E1/E3 shortlist; vote collation.

**Phase 4 — write-up.** One results table per experiment block; the
E4b metric-vs-K figure; the E4a factorization chart; the E5
dose–response figure if gated in.

Approximate additional training cost beyond runs already underway:
three short finetunes (S1, S-mel, S-chord ≈ a few GPU-hours each) plus
one full A.2-dense run (≈ one A.2-melchord budget). All evaluation
sweeps are single-GPU inference jobs.

---

## 7. Planned deliverables

- **T1**: E1 co-generation table (systems × H1–H3 primaries +
  listening win rates), per task.
- **T2**: E2 retention table (A.2 marginal vs specialist, margin
  verdict per stream).
- **T3**: E3 horizon table ordered by conditioning horizon, with the
  B.1 gap-recovery fraction highlighted.
- **F1**: E4a factorization bar chart (decode effect / objective
  effect / combined).
- **F2**: E4b metric-vs-K curve.
- **F3**: expert-routing by stream (observational companion to E4c).
- **T4**: E4c MoE table (A.2 vs A.2-dense across hypothesis blocks).
- Listening-test protocol sheet and anonymized vote record.
