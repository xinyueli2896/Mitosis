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
  models run in conditional mode and the published adapter baseline
  (Y), and how does the conditioning horizon (none / bounded /
  unbounded future) shape quality?
- **RQ4 (ablations)**: what does the simultaneous mutual within-frame
  conditioning strategy (同步看, realized by A.2's block-diffusion
  refinement) contribute — isolated at decode time, at training time,
  and combined; does the MoE FFN causally contribute to per-stream
  grammar specialization; plus decode-schedule ablations.

---

## Paper scope (registered 2026-08-27)

The publication target, for now, is **co-generation plus two ablation
families**. Everything below stays in the plan, but these three carry
the paper:

1. **Co-generation (E1)** — lead-sheet (melody+chord) co-generation:
   duet systems vs single-stream baselines vs the external SOTA
   references of §2.6 (whole-song model's lead-sheet stage;
   AccoMontage via the future accompaniment arm). Headline claim: what
   explicit two-stream joint modeling buys.
2. **MoE ablation (E6)** — interpretability-first: dense / shared /
   hard-route / per-modality-gates spanning imposed-vs-learned
   modality separation, with the stamp-probe diagnosis, the falsified
   bias variant, and the emergent specialists+integrators result.
3. **Alignment ablation** — the "one musical event, one coordinate"
   principle, shown twice independently: A.2's rope-geometry ablation
   (v1.1 parity RoPE vs v1.2 time-aligned,
   `eval_a2_rope_ablation.sbatch`) and C.1's prefix-geometry study
   (legacy vs `prefix_stride2`, reconstruction 0.830 → 0.999) — the
   same alignment idea at two different places in the architecture.

Second-degree (recorded, not in the paper for now): conditional
generation as a headline (E3 stays supporting material), the
conditioning-balance knob (E3 side note), non-aligned modalities
(§2.7), the lyrics / audio-symbolic task, E5.

---

## Paper writing decisions (ICASSP, registered 2026-08-30)

Target venue ICASSP; the paper stays **inside symbolic music** and makes
no multimodal claim. Framing, motivation and the three decisions below
are settled — revisit only with a reason.

### Framing

The gap is not that prior work fails to distinguish tracks; it is that
it distinguishes them **at the input only**. MMM, MMT, FIGARO, MuseNet
and similar systems put every track through one shared network and
separate them with an instrument or track-identity token. Say
"predominantly", not "most", and locate the gap in *parameter* sharing:
prior work differentiates what the model reads, not how it computes.

The motivating claim is that tracks carry distinct **functional roles**
— melody, harmony, accompaniment — each with its own grammar (melodic
contour and phrase structure; voice-leading and harmonic rhythm;
textural patterning). One shared parameter set gives role-specific
structure nowhere to form.

### Decision 0 — per-part gates are the DEFAULT, not an arm (2026-08-31)

The per-modality-gate router (`MOE_MODALITY_GATES=1`, run-dir marker
`mg`) is the system's standard configuration — the model the paper
presents — and `train_duet_block_diffusion.sbatch` now defaults to it.
Stop calling it "the D3 arm" in prose and conversation: it is simply
the model. E6's ablation arms are the DEPARTURES from it:

| ablation arm | flags | what it removes |
|---|---|---|
| shared router | `MOE_MODALITY_GATES=0` | the per-part gates |
| dense (no MoE) | `MOE_NUM_EXPERTS=1,MOE_TOPK=1,MOE_INTERMEDIATE_SIZE=6144` | routing entirely |
| hard route | `MOE_MODALITY_HARD_ROUTE=1` | the learned (vs imposed) split |

The historical codenames (A2/D1/D2/D3) remain in older notes and run
dirs; going forward the table above is the naming.

### Decision 1 — never write "modality" in the paper

Internally the arm is `moe_modality_gates` (D.3). In the prose it is a
**per-part** or **role-specific router**. "Modality-specific" in a
symbolic-music paper either confuses the reader or reads as the
multimodal claim we are explicitly not making yet. Code names stay in
the repo; "part" / "role" goes in the paper. This also leaves the
multimodal extension available as future work without pre-claiming it.

### Decision 2 — the "unassigned pool" clause is load-bearing

The router is per-part but the expert pool is **shared and unassigned**:
which experts a part recruits, and whether any expert serves both, is
LEARNED, not imposed. That is what separates this from MoMa / VL-MoE
style architectures that partition experts by track a priori. Keep the
clause in the abstract. D.2 (hard route, disjoint pools) is the
ablation that backs it; if D.2 does not land in time the clause is
still a defensible design claim, just weaker.

### Decision 3 — what the interpretability result must say

When the MoE experiments complete, `[INTERPRETABILITY RESULT]` has to
make TWO moves in one sentence, or the claim collapses into "we added
MoE and it helped":

1. under the **shared** router, a large share of expert separation is
   explained by stream identity alone (the stamp-share probes
   178945/178946 gave ~69% — re-measure on the final config, do not
   reuse the old number blindly);
2. **per-part** routers reallocate that capacity to content (chord
   frames route by register in 9 of 12 layers versus 5 for the shared
   router).

If the final numbers move, preserve the two-part shape. It is what
makes the claim falsifiable.

### Decision 4 — E1 fields ONE system; A.3-A.6 are engineering, not content (2026-09-01)

The A.3/A.4/A.5/A.6 series is internal exploration of the query-slot
training recipe — pure engineering. None of it appears in the paper as
an ablation. E1 co-generation fields ONE system: the best-performing
(checkpoint, decode schedule) combo, and the Methods section describes
that winner's recipe plainly as *the* training scheme (one paragraph,
one D3PM/MaskGIT-practice citation, no variant names, no alternatives).

Selection protocol (so "how did you choose?" has a clean answer):
pick on DEV evidence — best-val CE (directly comparable across all
four), a ~20-song co-generation decode at REFINE_STEPS 4 and 0, the
eval-harness numbers on those, ears — then FREEZE the choice and run
the full E1 against the baselines exactly once. Never select on the
E1 test comparison itself.

Distinct and still in: the decode-time mechanism ablation
(REFINE_STEPS=0 vs 4 on the SAME final checkpoint, RQ4) — it ablates
the same-instant-conditioning claim, not the recipe history. One row.

The frozen winner gets recorded here when the bake-off concludes.

### Abstract draft (placeholders marked)

Single paragraph, ICASSP register. ~230 words. REVISED 2026-09-04:
repositioned from a multi-track critique to the joint-generation
framing (below), with a lead-in that narrows field -> dominant framing
(distant pairs) -> the neglected family (shared timeline, continuous
agreement) -> our case, so "agreement, not translation" lands as a
contrast the reader has been set up for.

> Generative models are increasingly asked to produce several
> coordinated streams at once, and most of that effort has gone to
> semantically distant pairs such as image and text, where the
> difficulty is bridging representations. Signals that instead share one
> timeline and must agree continuously - speech with its transcript,
> melody with harmony - have drawn less attention. Melody and harmony
> are a demanding case: two domains of a single modality, aligned frame
> by frame, each obeying its own grammar, from contour and phrase
> structure to voice leading and harmonic rhythm. Agreement, not
> translation, is the problem here, and the bar is set by unified
> single-stream music models. Existing systems share all parameters
> across tracks and separate them by an identity token alone, or chain
> two specialists, fixing one stream before the other is written. We
> generate both with a single network that keeps their computation
> distinct and their decisions simultaneous. Specialization arises from
> a sparsely gated mixture-of-experts layer with a per-stream router
> over a shared, unassigned pool: which experts a stream recruits, and
> whether any serves both, is learned rather than declared. Three
> attention passes - within-stream, cross-stream causal, and
> same-instant - restore the mutual conditioning that a causally
> interleaved sequence destroys. With the backbone pretrained on
> unpaired single-stream music, the scarce aligned corpora are spent on
> coordination alone. On melody-chord co-generation over POP909,
> Nottingham and Pop1K7, [RESULT]; routing analysis indicates the
> specialization is content-driven rather than an artifact of stream
> identity: [INTERPRETABILITY RESULT].

**Decision 1 AMENDED (2026-09-04) - the modality/domain rule.** The
earlier rule was "never write 'modality'". It is now: melody and chord
are two DOMAINS of one MODALITY (symbolic music), and the word
"modality" appears exactly once, in the positioning clause that places
this work beside speech-text rather than image-text. Everywhere else
the streams are parts / domains / streams. The distinction is load
bearing for the paper's framing: it is what makes "agreement, not
translation" the honest statement of the problem, and it is what
licenses both the scarce-paired-data argument (unpaired single-stream
music pretrains the backbone) and the strong-baseline argument
(unified single-stream models set the bar).

If the venue's word limit bites, the two cheapest cuts that keep the
structure are the second half of "speech with its transcript, melody
with harmony" and the clause "from contour and phrase structure to
voice leading and harmonic rhythm" (~20 words).

Open placeholders: `[RESULT]` (E1 co-generation vs the SOTA lead-sheet
references of 2.6) and `[INTERPRETABILITY RESULT]` (E6, per Decision 3).

---

## 1. Tasks, data, splits

| | drumnondrum | melchord |
|---|---|---|
| streams (mod_a / mod_b) | drum / nondrum | melody / chord |
| training corpus | LA (la_*_cp16_v2) | POP909 (pop909_*_cp8_v2) |
| test prompts | RWC (`input/rwc_test_prompts_split/{drum,nondrum}`) — fully held-out corpus | POP909 held-out songs (below) |
| polyphony | 16 | 8 (duet) / 16 (single-stream combined) |

**Melchord polyphony = cp8, certified.** The chord renderer emits 1 bass
+ up to 4 upper tones, so seventh chords are FIVE simultaneous notes;
the original cp4 budget silently dropped the topmost tone — the seventh —
from 13.8% of chord frames (3.3% of chord notes on the eval split,
measured by `cp_capacity_check.sbatch`). cp8 covers the observed maximum
(5) with margin at half of cp16's padding. Certified over ALL melchord
corpora, training included: full POP909 chords peak at 5 onsets/frame
(12,061 five-note frames = 10.1% of chord frames — under cp4 the models
TRAINED on targets missing those sevenths, ~2.5% of all chord notes);
full POP909 melody peaks at 6 (four ornament frames corpus-wide);
Nottingham peaks at 7 even with melody+chord pooled. Zero clipping at
cp8 everywhere. Budgets are certified against
every corpus used (`cp_capacity_check.py` exits 1 on clipping); the
remaining known clip is on the drumnondrum side and runs AGAINST the
merged baseline: duet cp16/stream drops 0.69% of nondrum notes (worst
frame 22) while S0's pooled cp16 drops 1.76% of all notes (worst 23).
Pre-migration cp4-trained ckpts remain evaluable via the
`MAX_POLYPHONY*=4` knobs on the eval/infer sbatches (`MELCHORD_CP=4`
selects the legacy datasets in `tasks.py`).

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
| **S1** | S0 + POP909 finetune | Identical architecture; weights finetuned on merged, program-tagged POP909 melody+chord | LA → POP909 (20k steps) | Matched single-stream baseline for melchord (E1 headline opponent) |
| **S-mel** / **S-chord** | S0 + single-modality POP909 finetune | Identical architecture; same finetune recipe as S1, but on ONE stream's data only | LA → POP909 melody-only / chord-only | Per-stream marginal specialists: the fair E2 opponents for A.2's `mel_only` / `chord_only` modes (H-E2.1); pairing is per-stream and never crossed (§E2) |
| **S-drum** / **S-nondrum** | S0 + single-modality LA finetune | As above, on `la_drum_cp16_v2.pt` / `la_nondrum_cp16_v2.pt` | LA → LA drum-only / nondrum-only | Drumnondrum marginal specialists — GATED on the melchord E2 outcome |

### 2.2 Two-stream (duet) systems — main experiments

| ID | Model | Architecture | Generation modes | Training status |
|---|---|---|---|---|
| **A.2** | `M2CDuetBlockDiffusion` v1.1 | Interleaved two-stream joint AR backbone + two appended next-frame query slots trained as discrete denoisers across noise levels (diffusion_K = 4); inference runs K+1 parallel-refinement passes per frame, giving the two streams simultaneous mutual within-frame conditioning (同步看) | co, mel2chord, chord2mel, mel_only, chord_only; refinement depth K is a decode-time knob | drumnondrum: trained (98k). melchord: in training |
| **B.1** | `M2CDuetAnticipatory` | Joint-AR duet with the drum stream re-indexed k frames ahead, giving the partner stream k frames of future context | co, conditional (drum→nondrum) | drumnondrum: trained |
| **C.1** | `M2CDuetRehearsal` | Bidirectional conditioning-stream prefix ("rehearsal") + interleaved suffix; purpose-built conditional model | drum→nondrum only | drumnondrum: trained |
| **C.2** | `M2CDuetPrefix` | Prefix-LM: bidirectional conditioning stream, causal target stream; purpose-built conditional model | drum→nondrum only | drumnondrum: trained |

### 2.3 Ablation-only systems

| ID | Model | Architecture | Role |
|---|---|---|---|
| **A.1** | `M2CIntraCrossAttn` (DuetAttn) | Interleaved two-stream joint AR; strict causal attention; NO query slots, NO refinement — each stream sees the partner only up to the previous position | Excluded from the main experiments. Used exclusively in the E4 ablation to isolate what A.2's 同步看 mechanism adds over plain sequential joint AR |

**The D family — E6 MoE-ablation variants of A.2.** A.2 itself (shared
gate) is the base arm; the D-numbered systems differ from it ONLY in
the FFN's routing design, in spectrum order (none → imposed → learned):

| ID | Routing design | Repo knob / ckpt family | Role in E6 |
|---|---|---|---|
| **D.1** | dense — no MoE (`moe_num_experts=1`, inter 6144, compute-matched to the activated 2-of-4 × 3072) | `RUN_TAG=densecm`; `CKPT_DENSE` in the harness | the floor: is routing load-bearing at all? |
| **D.2** | hard route — disjoint per-modality pools, separation imposed | `MOE_MODALITY_HARD_ROUTE=1`, `K4hr` dirs; `CKPT_HR` | the imposed-separation control; integrators unrepresentable |
| **D.3** | per-modality gates — per-stream routers over a shared, unassigned pool | `MOE_MODALITY_GATES=1`, `K4mg` dirs; `CKPT_MG` | separation available but learned; the candidate design |

(The falsified per-modality **bias** variant, `A.2.moe_improved`/mb,
keeps its historical name — it is a documented negative result, not an
E6 arm.)

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

### 2.4 Published conditional baseline (YinYang)

| ID | Model | Architecture | Task coverage | Training data |
|---|---|---|---|---|
| **Y** | `RoformerYinyang` (`cp_transformer_yinyang.py`; arXiv:2506.15548) | FROZEN pretrained CP transformer run twice — one pass fully encodes the conditioning stream, a second pass generates the target stream AR while low-rank cross-attention adapters (every `n_skip` layers, + LoRA on Q/V) inject the conditioning stream's hidden states at ALL positions. Parameter-efficient conditional adaptation of S0 | both directions per task: mel→chord, chord→mel; drum→others, others→drum | **Y-mc**: Nottingham (1,020 songs). **Y-dn**: LA subset with drums (31k songs) |

Role: the external, published conditional baseline — unlike S0/S1 it
genuinely conditions on the full partner stream, so it is the honest
conditional comparison point. It also contrasts architecturally with
C.1/C.2 at the SAME conditioning horizon: fully-trained duet
architectures vs frozen-backbone + adapters.

**Domain caveat (must be disclosed wherever Y-mc appears).** Y-mc is
finetuned on Nottingham (folk); our melchord evaluation is POP909
(pop). The Y-mc vs A.2-melchord comparison therefore confounds
architecture with training domain. Y-mc is reported as an external
reference point, NOT as a matched baseline. Optional upgrade if the
comparison matters to the write-up, two symmetric options now exist:
(a) finetune a POP909-matched YinYang (`cp_transformer_yinyang.py` on
the POP909 melody/chord data), or (b) evaluate on NOTTINGHAM instead,
where Y-mc is the matched system — the Nottingham melody/chord
tokenization pipeline exists (`preprocess_nottingham_melchord.sbatch`
→ `data/nottingham_{melody,chord}_cp8_v2.pt`, POP909-compatible cp8
format, plus a `melchord_nottingham` task entry for duet training).
Record the decision either way.

**Decision (2026-08-25): option (a) — train the POP909-matched YinYang
("Y-909"), both directions.** Pipeline:
`preprocess_pop909_yinyang.sbatch` builds the paired within-frame
dataset `data/pop909_cp8_v2_chord_mel.pt` from the tagged merged folder
(same `ins_ids=['track-1','track-0']` build and `_chord_mel` naming as
the published Y-mc data, so all direction auto-detection carries over);
`train_yinyang_pop909.sbatch` trains one direction per submission
(`DIRECTION=chord2mel` → no `_rev`, `DIRECTION=mel2chord` → `_rev`)
with the published recipe defaults (mask 0.25/10, LoRA r16, n_skip 2).
Y-mc remains reported as the external Nottingham-domain reference;
Y-909 is the matched conditional baseline for the POP909 melchord E3. Y-dn has no such caveat: it is
LA-trained like the duet drumnondrum systems and evaluated on held-out
RWC.

### 2.5 Baseline preparation requirements

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

### 2.6 External SOTA baseline — lead-sheet framing (PLANNED 2026-09-01)

Status: **planned** — decisions resolved, facts verified against the
paper/repo (sources at the end of this section); integration tasks
T1-T6 below, nothing downloaded or run yet.

Our melchord co-generation IS lead-sheet generation (melody + chord
jointly), which gives us a recognized task name and two natural
external baselines, both POP909-native (Xia lab), i.e. domain-matched
to our main corpus:

| system | task shape | maps onto |
|---|---|---|
| **Whole-song hierarchical generation** (cascaded diffusion; Wang / Min / Xia, ICLR 2024 — verify) | unconditional full-song generation: form → **lead sheet (melody+chord)** → accompaniment, POP909-trained | its lead-sheet stage is a direct external baseline for **E1 co-generation**; its form stage makes it the strongest available reference for the long-term-structure metrics |
| **AccoMontage** (Zhao & Xia, ISMIR 2021; later versions exist — verify which) | conditional: lead sheet → piano accompaniment (phrase retrieval + texture style transfer over POP909) | no direct melchord arm — its slot is a future **accompaniment task**: Pop1K7's dropped `piano` track gives us paired (melody, chord, accompaniment) data to train our own arm against it |

**Integration work (why this is its own pass, not a checkbox):**

1. *Output conversion*: both emit their own representations; we need
   converters into the split melody/chord midi layout the eval chain
   consumes (`build_eval_manifest.py` expects per-stream files on our
   16th-note 4/4 grid; both systems are beat-quantized 4/4, so this is
   bookkeeping, not resampling).
2. *Protocol matching*: the whole-song model is natively
   unconditional-with-form; our E1 rows are prompt-continuations. Fair
   comparison needs either its prompted/infilling mode (verify one
   exists) or an unconditional comparison row where OUR systems also
   run unprompted. Do not mix the two protocols in one table.
3. *Contamination*: both baselines TRAINED on POP909 — our held-out
   ids were in their training data. POP909 rows carry that caveat in
   their favor; the clean generalization comparison is on Nottingham /
   Pop1K7-held-out prompts, which are out-of-domain for them (caveat
   then runs the other way). Report both, labeled.
4. *Decode budget parity*: sample counts, lengths and any cherry-pick
   policy matched to §4's frozen settings.

**Verified facts (2026-09-01).** Whole-song model: Wang, Min & Xia,
"Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded
Diffusion Models," ICLR 2024. Four cascaded levels: Form -> Reduced
Lead Sheet -> Lead Sheet -> Accompaniment; levels are image-diffusion
models with scopes full-song(<=256 bars) / 32 bars / 8 bars / 8 bars;
inference is quasi-autoregressive segment inpainting. Repo
github.com/ZZWaang/whole-song-gen, MIT license, PARTIAL checkpoints
released ("sufficient for testing"), inference via
`inference_whole_song.py` with two modes: unconditional, and
conditioned on a specified form+key. **Prompted generation (first-N-
measures continuation) is explicitly NOT released.** Trained on POP909.

**Decisions (a)-(d), resolved:**

- (a)/(b) REVISED 2026-09-01: **AccoMontage2 is IN — as the
  harmonization-row baseline.** AccoMontage2 (Yi, Hu, Zhao & Xia,
  ISMIR 2022; repo github.com/billyblu2000/accomontage2, code+dataset
  released) adds a harmonization module: given a lead MELODY it
  generates a full-length chord progression (micro dissonance + meso
  phrase-template + macro coherency losses), then accompaniment. It
  cannot co-generate melody, so it enters a third, MELODY-GIVEN block:
  our `mel2chord` mode vs AccoMontage2's harmonization, both fed the
  same test melodies. The accompaniment task itself stays out
  (second-degree); original AccoMontage (ISMIR 2021) stays related
  work only.
- (c) **Protocol = three blocks, each system in its native mode,
  never mixed:** (i) prompted continuation — ours vs S0/S1/S-scratch
  AND the whole-song cascade (see rev. below); (ii) unconditional —
  ours vs the whole-song cascade; (iii) melody-given harmonization —
  ours (mel2chord) vs AccoMontage2.
  Our one checkpoint serves all three blocks (co, unconditional co,
  mel2chord are decode modes of the same model — worth one sentence in
  the paper, since neither baseline can do that).
  **REVISED 2026-09-01 — whole-song enters block (i) too.** The
  original rejection ("hacking its inpainting into continuation") was
  based on the README's "prompted generation not released". Code
  inspection shows continuation is IMPLEMENTED in the released
  inference library, only unexposed: every level's `create_canvas`
  takes `prompt=` (canvas written, mask=1 over the prompt region) and
  the sampler's `generate` takes `orig_x`/`mask` as first-class
  inpainting inputs — the segment loop already generates conditioned
  on known regions; only the thin `WholeSongGeneration.main()` driver
  hard-codes `prompt=None`. Enabling it = calling their functions with
  the argument they accept, prompts encoded by THEIR data pipeline
  (`read_pop909_data`, `tonal_reduction_algo`, `specify_form` with
  ground-truth form+key). No custom sampling logic — fair adaptation,
  disclosed in the paper as such. AccoMontage stays out of block (i):
  nothing in that series models melody generation at all (capability
  gap, not interface gap).
  **Contamination note (block i/ii):** their POP909 split is
  deterministic (seed=1234, 9:1); reproducing it puts our test songs
  001/002/003/005 in their TRAIN set, 004 in their valid set (assumes
  index i = song i+1; verify against their split.npz once data/ is
  downloaded). For continuation, the ground-truth continuations of
  four prompts are in their training data: if we match/beat them the
  claim is maximally conservative; a loss on those four is
  memorization-ambiguous. Report 004 separately as the clean point.
- (d) **Regenerate locally** for both externals. Whole-song: MIT +
  released ckpts. AccoMontage2: code+dataset released; verify it runs
  headless (it ships a GUI — a scriptable path must exist or be
  extracted). Fallbacks: demo/released samples, clearly labeled.

**Integration tasks:**

- **T1 setup** (CPU sbatch): clone into `external/whole_song_gen`,
  resolve deps (own env if the mitosis env clashes), pull ckpts via
  the repo's download_link.txt.
- **T2 coverage check**: confirm the released ckpts include levels
  1-3 (Form, Reduced LS, Lead Sheet). We do NOT need level 4
  (accompaniment). If missing -> (d) fallback.
- **T3 generate** (GPU sbatch): N unconditional songs matching Sec. 4's
  frozen sample budget; determine their output format and export to
  MIDI if the repo does not already.
- **T4 convert**: `convert_wholesong_outputs.py` -> split melody/chord
  midi on our 16th-note 4/4 grid (they are beat-quantized 4/4:
  bookkeeping, not resampling). Score the first 384 frames (24 bars)
  of each song, same window as ours.
- **T5 ours-unconditional**: verify/enable prompt-free co generation
  in our inference (minimal seed), same 384-frame budget.
- **T3' whole-song PROMPTED continuation** (block (i) external; after
  T2 GO + data/ payload): per test song — ground-truth form string
  (Dai labels, same source we use) + key via `specify_form`; prompt
  bars encoded to ctp/lsh languages with their data pipeline; run
  ctp/lsh generation with `create_canvas(..., prompt=...)` (their
  generation loop unchanged); export the lsh level; convert with the
  T4 converter. Match our block-(i) prompt length; skip acc. Also
  verify the split-contamination note against their shipped
  split.npz.
- **T7 AccoMontage2 setup**: clone, deps, verify headless/scriptable
  harmonization (GUI is the documented interface); check whether its
  harmonizer needs phrase annotations for the input melody (its meso
  loss is phrase-template based -- POP909's phrase-annotation dataset
  covers our test songs if so).
- **T8 harmonization row**: feed both systems the same POP909 test
  melodies; convert AccoMontage2's chord output to our chord-track
  midi layout; score with the harness's conditional machinery
  (GIVEN_STREAM_BY_MODE already handles melody-given rows: chord-side
  and inter-stream metrics only, melody rows excluded as copied
  input).
  **T8 status 2026-09-01: generation + conversion DONE (5/5 test
  songs, `temp/accomontage2_pop909_eval/`).** Integration notes, all
  uniformly applied and to be stated as baseline-interface
  limitations: phrase labels snapped to chorderator's {4..32}
  grid; on solver failure (a phrase length with zero library
  templates: 'no matched length' -> max()-of-empty crash) the song is
  retried once with phrases decomposed to <=8 bars (hit song 005);
  chord-only task (its texture stage needs unshipped reference data
  and is not scored anyway).
  **Data-overlap audit (job 197554): the harmonization row is
  POP909-clean.** The DP's retrieval space (`rep`, 537 representative
  progressions) contains zero POP909-derived entries — all commercial
  chord-pack sources — so no output can retrieve a test song's own
  progression; the POP909-derived material in their repo feeds only
  the texture stage we bypass. Contamination sentence is therefore
  needed only for the whole-song model (POP909-trained; unconditional
  corpus-level block; exposure favors the baseline, so reported gaps
  are conservative).
- **T6 eval-harness corpus-reference mode**: unconditional outputs
  have no paired reference song, so the *_ref/*_delta metrics need
  corpus-level reference statistics (distributions pooled over the
  POP909 test set) instead of per-song pairs. coupling / ctnctr / pcs
  / mctd absolutes are intra-output and need no change. Also run
  structure_metrics on both systems -- their form stage should win
  long-term structure, and reporting that honestly motivates the
  future-work paragraph.

### 2.7 Tabled: non-aligned modalities (second-degree)

The duet family assumes one shared frame clock with pairing by
arithmetic — the interleaved layout, A.2's frame pass and frame gate,
C.1's stride-2 prefix coordinates, and even the MoE modality ids
(slot parity) all key on it. The question of alignment-FREE operation
(streams on their own clocks, content-addressed cross-attention) is
**deliberately tabled as second-degree**: the C.1 study showed
position-addressed retrieval is what makes the rehearsal mechanism
work (0.830 → 0.999), and giving that up is a research project, not a
feature. Working policy until revisited: **buy alignment in
preprocessing** and keep the proven architecture — beat-track/quantize
unaligned clocks onto the grid (the Pop1K7/madmom route), force-align
weakly-aligned modalities (the planned lyrics phase 1), and treat
timeless conditioning (tags, descriptions) as global conditioning, not
a stream. The one change any of these need regardless: MoE modality
ids passed explicitly instead of derived from parity.

---

## 3. Experiment matrix

### E1 — Co-generation (RQ1)

Both streams are generated jointly, conditioned on a 4-bar prompt of
both streams.

| Task | Systems under test | Matched baseline | Mode |
|---|---|---|---|
| drumnondrum | A.2(K=4), B.1 | S0 (merged stream; same LA training corpus) | `co` |
| melchord | A.2(K=4) | S1 (merged stream; same POP909 data); S0 as unmatched anchor | `co` |

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

The hypotheses partition cleanly: H1 judges each stream ON ITS OWN
(consistent role and texture), H2 judges how well the two streams FIT
EACH OTHER, H3 judges whether the statistics are stream-appropriate,
H4 is the net perceptual effect.

| # | Burden removed | Predicted observable | Metric |
|---|---|---|---|
| H1 | stream identity inferable only from program tokens, and both streams' notes competing inside one shared per-frame token budget (duet: the interleave position IS the stream, and each stream gets its own frame slot at every time step) | consistent role and texture per stream across all 24 bars: melody stays monophonic and in register, chords stay block-voiced, per-stream density stays near the reference, neither stream dies; S\* shows role leakage, texture thickening/thinning with continuation length, or a stream vanishing mid-piece | per-stream polyphony profile; register overlap; density trajectory as a function of bar index; silence/collapse rate; stream-survival length |
| H2 | partner notes are ordinary tokens scattered in one undifferentiated sequence — attention must LEARN to find them (duet: a dedicated cross-stream attention pass makes the partner architecturally addressable) | inter-stream fit: generated chords are consonant with the simultaneous melody; nondrum onsets lock to drum onsets — each stream can be individually well-formed and still fail this | chord-tone coverage of concurrent melody notes (melchord); onset synchrony (drumnondrum) |
| H3 | one set of weights multiplexing two grammars | stream-appropriate statistics (harmonic rhythm ≈ 1 chord change/bar; melodic contour smoothness); supporting evidence: MoE expert usage separating by stream | duration / harmonic-rhythm distributions; expert-routing side analysis |
| H4 | (net perceptual effect) | an ORDERING of listening margins: coherence (b) > musicality (a) ≥ 0, structure (c) ≈ 0. Rationale: the architecture changes only the cross-stream joint distribution, so (b) moves directly; (a) moves only derivatively, when H1-type integrity failures in the baseline drag down its perceived musicality; (c) is untouched by either architecture. A roughly UNIFORM win profile would instead indicate a generic quality difference (capacity, tuning) rather than the stream-differentiation mechanism | pairwise A/B margins per axis; the (b) vs (a) margin comparison is the test |

**Pre-registered counter-expectations** (stated now so results are not
cherry-picked later): S\* is expected to TIE on local plausibility and
pitch-class histograms (S1 saw the same data; merged AR is strong
locally) — the claim lives in the structural/cross-stream measures
H1–H2, not the local ones. A.2 may LOSE on the structure/development
listening axis due to its known repetition tendency; that axis is
reported but is not the claim under test.

Planned contrasts:

1. **S\* vs duet systems** — value of architectural stream separation
   (the headline RQ1 comparison), read through H1–H4 above.
2. **B.1 vs A.2(K=4)** — bounded-lookahead leader/follower design vs
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

**Mechanism note (pre-registered, not a post-hoc discovery).** In
A.2's `mel_only`/`chord_only` modes the absent stream is CLAMPED, not
filtered: its frame is resolved to explicit silence before the model
runs, held as a committed given (k=0) across all refinement rounds,
and never sampled — so unwanted-modality content cannot exist, and the
committed context contains explicit `[stream_t, silence_t]` pairs.
Consequence of the frozen decode config: with `A3_ADAPTIVE=1`, the
silent partner probe triggers the early-exit at EVERY step, so
marginal-mode A.2 effectively decodes at K≈0 (AR-draft commit, no
refinement). This is the adaptive mechanism working as designed —
refinement is mutual negotiation and a clamped-silent partner leaves
nothing to negotiate — but it must be stated in the write-up: E2's
A.2 outputs come from the AR pathway, not the refinement machinery.

**Claim under test — RETENTION, not superiority.** E2 does not argue
that A.2 generates single streams better than anything; it argues that
joint training did not COST the model its marginal ability — a jointly
trained model can still generate each modality alone at
specialist-comparable quality. The experiment is a guard against the
capacity-interference failure mode, and its success criterion is a
pre-registered non-inferiority margin:

- **Margin δ**: A.2 retains the marginal if, on each primary H3 grammar
  metric of the generated stream, its mean is within δ = 0.05 absolute
  (JSD metrics) / within the between-song standard deviation (delta
  metrics) of the matching specialist's mean. Within margin → "joint
  training retains clean marginals." Outside margin → the deficit is
  reported as the QUANTIFIED marginal cost of joint training — an
  informative result, not a failure to be explained away.
- If A.2 lands BETTER than the specialist, that is a bonus finding
  (positive cross-stream transfer, see §2.1 note) and is reported as
  such — but E2's design and framing do not depend on it.

The baseline is the **S-specialist**: the fair opponent for the
marginal question, constructed to match A.2's provenance in everything
but the objective — warm-started from the SAME LA-pretrained ckpt that
A.2's backbone warm-started from, finetuned with the SAME recipe/steps
as S1, on the SAME POP909 songs, but on single-modality data only
(S-mel / S-chord from `pop909_melody_cp8_v2.pt` /
`pop909_chord_cp8_v2.pt`; optionally S-drum / S-nondrum from
`la_drum_cp16_v2.pt` / `la_nondrum_cp16_v2.pt`).

Rejected baseline alternatives, for the record:

- **S0 alone** confounds objective with in-domain exposure.
- **A from-scratch single-modality model** is denied the pretraining
  inheritance A.2 received and would lose for data-scale reasons — a
  strawman.
- **S\*-merged prompted with single-stream files** cannot express a
  single-stream request at all: its training distribution always
  contains both streams, so it tends to spawn the missing partner in
  the continuation (and the prompt is OOD for it besides). This is
  the one-line justification for building specialists rather than
  reusing S1 — optionally verified by an informal spot-check, but NOT
  run as a scored E2 lane or carried as a hypothesis.

Two owned asymmetries of the specialist comparison: architecture
cannot be held constant (there is no non-degenerate "A.2 trained
marginally"), so this compares the joint system's marginal against the
single-stream pipeline's best specialist; and the specialist sees half
the tokens per song — if A.2 nevertheless WINS, that is evidence of
positive cross-stream transfer, a reportable finding rather than a
confound.

| Task | Systems under test | Baseline | Modes |
|---|---|---|---|
| melchord | A.2 | S-mel / S-chord specialists | `mel_only`, `chord_only` |
| drumnondrum | A.2 | S-drum / S-nondrum specialists GATED — train only if the melchord E2 result is surprising | `mel_only`, `chord_only` |

**Specialist pairing is per-stream and never crossed** — each A.2
marginal mode is compared only against the specialist of the SAME
stream:

| A.2 mode | Opponent | Specialist training data |
|---|---|---|
| melody marginal (`mel_only`) | S-mel | `pop909_melody_cp8_v2.pt` |
| chord marginal (`chord_only`) | S-chord | `pop909_chord_cp8_v2.pt` |
| drum marginal | S-drum (gated) | `la_drum_cp16_v2.pt` |
| nondrum marginal | S-nondrum (gated) | `la_nondrum_cp16_v2.pt` |

Design note: separate per-stream specialists are DELIBERATELY the
harshest opponents — each devotes its full capacity to one marginal,
while A.2 holds both marginals plus their coupling in one set of
weights. The asymmetry runs against A.2, which is the correct
direction for a non-inferiority claim. (A capacity-matched alternative
— one union model finetuned on the mixture of melody-only and
chord-only sequences — exists as a softer fallback but is not the
primary design.)

Specialist training is cheap: the single-stream tokenized data already
exists and each specialist is one `finetune_pop909.py --data <file>`
run with the same recipe as S1.

**Pre-registered hypotheses:**

- **H-E2.1 (non-inferiority — the capacity-interference test).** On
  the stream actually generated, A.2's marginal grammar quality (H3
  block: onset-grid JSD, duration JSD, stepwise motion, harmonic
  rhythm) is not worse than the SPECIALIST's by more than the margin δ
  above. Deliberately a non-inferiority claim, not superiority.
  Expectation: parity, because per-stream projections and MoE routing
  partially decouple the streams' capacity.
- **H-E2.2 (pre-registered risk — partner silence may be OOD for
  A.2-melchord).** On drumnondrum, silent-partner stretches occur
  naturally in training (songs without drums), so marginal generation
  is in-distribution. On POP909 the chord stream is essentially always
  active and training used no silence augmentation, so an entirely
  silent partner is a regime A.2-melchord never saw. Possible outcome:
  degraded A.2 marginals on melchord specifically (drift or
  EOS-collapse of the generated stream). If observed, this is an
  informative limitation of joint training without silence
  augmentation — registered now so the result cannot be reframed
  post-hoc in either direction.

### E3 — Conditional generation (RQ3)

The complete ground-truth partner stream is provided; the model
generates the remaining stream.

| Task | Systems under test | Baselines | Direction |
|---|---|---|---|
| drumnondrum | A.2 (conditional mode), B.1, C.1, C.2 | **Y-dn** (matched external conditional baseline) | drum → nondrum (Y-dn additionally: nondrum → drum vs A.2 `chord2mel`-equivalent) |
| melchord | A.2 (`mel2chord`, `chord2mel`) | **Y-mc** (external reference, domain caveat §2.4) | both directions |

The no-conditioning FLOOR needed to interpret fit scores is taken from
the E1 co-generation outputs (already produced, no extra runs): score
a system's co-mode target stream against the ground-truth partner. No
single-stream anchor participates in E3 — a merged-stream model has no
faithful conditional decoding, and scoring its co-generation against a
partner it never saw is a cross-architecture construct that adds noise
without information.

**Reading an E3 table: which rows can discriminate.** Every conditional
decoder writes the conditioning stream into its output *verbatim* (A.2
returns `('given', condition[:, t, :])` at every frame; C.1/C.2 set
`m_tokens = drum_tokens[:, t, :]` inside the AR loop). So every metric
computed on the given stream is the same copied ground truth in every
system's output, identical across columns **by construction** — about
nine of the twenty rows in `mel2chord` (the whole `_a` family plus
`mel_stepwise_delta`, `mel_poly_rate`, `chord_tone_cov_ref`). This is
the experiment's design, not a bug in the eval;
`aggregate_eval_results.py --given-stream auto` marks those rows `(=)`
and leaves them untested, and `check_e3_identical.py` separates them
from the two failure modes that look the same in the table (systems
sharing output paths; systems generating identical notes).

The consequence for hypothesis reading: **in `chord2mel` the H3 primary
`harmonic_rhythm_jsd` is computed on the GIVEN chord stream and cannot
discriminate.** Read H3 for that direction from the melody-side rows
(`onset_grid_jsd_a`, `duration_jsd_a`, `mel_stepwise_delta`), or take
H3 from the `mel2chord` direction, where the chord stream is the one
generated.

**Conditioning-horizon spectrum (drum → nondrum).** The systems form
an ordered spectrum in how much FUTURE of the conditioning stream the
target stream can attend:

| System | Future drum context visible to nondrum_t | Streamable | Parameter profile |
|---|---|---|---|
| A.2 (conditional mode) | 0 frames (same-frame only; the given stream is committed frame-by-frame) | yes | fully-trained duet |
| B.1 | k frames (bounded lookahead; k = 16 ≈ 1 bar) | yes, with k-frame latency | fully-trained duet |
| C.1 / C.2 | entire sequence (bidirectional) | no | fully-trained duet |
| Y | entire sequence (cross-attention into the fully-encoded conditioning stream) | no | frozen backbone + LoRA/adapters |

This ordering turns E3 from an unordered system comparison into a
dose–response study of conditioning horizon. The headline question:
does B.1's bounded lookahead recover most of the C.2−A.2 gap? If yes,
bounded (hence real-time-capable) conditioning suffices and the
offline bidirectional architectures buy little; if no, full-sequence
conditioning is genuinely load-bearing.

A second contrast now available at FIXED (unbounded) horizon:
**C.1/C.2 vs Y** — fully-trained duet conditional architectures vs
parameter-efficient adaptation of the frozen single-stream backbone.
If Y matches C.\*, full conditional training buys little over adapters;
if C.\* wins clearly, the duet training is load-bearing, not just the
conditioning pathway.

**B.1 direction restriction.** The reverse direction
(nondrum → drum) is EXCLUDED for B.1 by design: under the anticipatory
layout, drum_{t+k} attends nondrum only up to t−1, so even with the
full ground-truth nondrum available the model conditions on
k-frames-stale partner context — an expected-negative by construction,
not a fair capability test. (At most, report it once as an asymmetry
control, clearly labeled.) B.1 is likewise excluded from E2: its
marginals are not its design question, and the follower-only mode is
structurally handicapped by the missing leader stream.


**Scoring — fit-to-given is primary, target agreement is secondary.**
Two notions of conditional quality must not be conflated:

1. **Fit to the given stream** (PRIMARY): does the generated stream
   follow the ground-truth partner it was handed? Measured with the H2
   block on the pair (given ground-truth stream, generated stream) —
   chord-tone coverage against the given melody / onset synchrony
   against the given drums — calibrated against the fully-ground-truth
   pair. This is what conditioning is FOR, and `eval_metrics.py`
   computes it as-is.
2. **Agreement with the ground-truth target** (SECONDARY,
   corroborating only): similarity between the generated stream and
   the original stream it replaces (frame-wise pc-set Jaccard). Weaker
   by construction — many different accompaniments are musically valid
   for the same partner, so mismatch is not failure. Never used as a
   primary endpoint.

**Pre-registered hypotheses:**

- **H-E3.1 (sanity gate — within-family).** Each conditional system
  exceeds its own NO-CONDITIONING floor on the primary fit metric: the
  same family's E1 co-generation output scored against the
  ground-truth partner (A.2-cond vs A.2-co; B.1-cond vs B.1-co — same
  model, same decode, the ONLY difference is whether the partner was
  given). For C.\*/Y, which have no co mode, the A.2-co floor serves
  as the common reference. A system that ties its floor is not using
  the conditioning information; its remaining E3 numbers are then
  reported but flagged as uninterpretable.
- **H-E3.2 (horizon dose–response — the headline).** Primary fit
  improves monotonically with conditioning horizon:
  A.2-cond (0 future) ≤ B.1 (k = 16) ≤ C.\*/Y (unbounded).
  Sub-question: the fraction of the C.2−A.2 gap that B.1 recovers —
  a large fraction means bounded (streamable, k-frame-latency)
  conditioning suffices and offline bidirectional architectures buy
  little.
- **H-E3.3 (training vs adapters at fixed horizon — deliberately
  open).** C.\* vs Y-dn at unbounded horizon. No directional prior is
  registered: full conditional training may beat parameter-efficient
  adaptation, or the frozen backbone's intact generative quality may
  win. Both outcomes are informative; registering a fake prediction
  here would be rigor theater. (Y-mc participates in the melchord rows
  as an external reference only — domain caveat §2.4.)

**E3 side note (second-degree, not a headline): conditioning-balance
knob.** C.1's cross gates take a pre-activation offset, so
`infer_duet_rehearsal.sbatch COND_GATE_OFFSET=<v>` shifts, at decode
time only, how much the generating slot listens to the conditioning
stream (cross path: prefix + interleaved mod_a) versus its own
generated history (intra path). Positive opens the gate toward the
condition; negative leans on own context; 0.0 is the trained operating
point and the model was never trained elsewhere — a sweep (±1, ±2) is
an out-of-distribution probe of the gate mechanism, not a calibrated
control. If it ever graduates from side note, the honest version is a
small sweep scored with the E3 conditional metrics (pc-set Jaccard vs
the condition on one axis, own-stream grammar on the other) to draw the
listen-to-condition / self-coherence trade-off curve. Deliberately
tabled: conditional generation is not the main task.

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
| A.1 vs A.2(K=4) | the combined package | task, prompts |

**Pre-registered expectations per contrast:**

1. **A.2(K≥1) vs A.2(K=0)** — the refinement gain should be
   metric-SELECTIVE, mirroring E1's H4 logic: the mechanism is mutual
   within-frame negotiation, so the gain should land on the H2 block
   (inter-stream fit: chord-tone coverage / onset synchrony), with at
   most secondary movement on H1 and little on H3 (refinement does not
   teach new grammar). A uniform improvement across all blocks would
   suggest the extra decode compute helps generically rather than via
   the claimed mechanism.
2. **A.2(K=0) vs A.1** — expected PARITY within the E2-style margin
   (δ = 0.05 on JSD metrics). A deficit on A.2's side means the
   diffusion objective taxed the AR pathway; a surplus would suggest
   the query-slot objective acted as a beneficial auxiliary task. Both
   deviations are findings.
3. **A.1 vs A.2(K=4)** — the combined effect should approximately
   decompose as (1) + (2); a large interaction term (combined ≫ sum of
   parts) would itself be interesting and reported.

**Disclosed confound for contrast 2**: A.1 and A.2 are different
training runs; their checkpoints may differ in total training steps
and data passes. Record both ckpts' global_step in the write-up and,
if they differ substantially, state the direction of the resulting
bias (the longer-trained side is favored). This cannot be fixed
without retraining and is disclosed rather than hidden.

Tasks: drumnondrum (A.1 trained). Melchord requires training
A.1-melchord — decide after the drumnondrum ablation: if the combined
effect is large, the melchord replication is worth the GPU-time;
otherwise report drumnondrum only and run just A.2(K≥1) vs A.2(K=0)
(weights-shared, free) on melchord.

Also reported here: **B.1 vs A.1** (strict anticipation ablation —
A.1 is exactly B.1 with k=0), per-stream, with the E1 stream-asymmetric
expectation (follower improves, leader flat or slightly degraded).

#### E4b — Decode-schedule ablations (A.2 only)

Drumnondrum listening tests established the temperature/nucleus
schedule (final temperature 0.9, nucleus 0.95) and found refinement at
K=4 outperformed K=0 when both streams were active but matched or
underperformed it on silent-stream material — motivating the
A3_ADAPTIVE early-exit that guards the headline K=4 configuration
(§4). Replication on melchord:

1. **Refinement depth**: K ∈ {0, 1, 4} at the selected schedule,
   `co` mode, full evaluation list (doubles as the E4a first-row data).
   Pre-registered expectation: the K benefit should be CLEARER on
   melchord than it was on drumnondrum — both streams are active in
   every frame, which is exactly the regime where refinement helped in
   the drumnondrum listening tests; there is no silent-stream material
   to dilute it. Expected profile: monotone improvement on the H2
   block from K=0 → 4 with diminishing returns; a flat curve would
   weaken the E1 headline's reliance on K=4.
2. **Adaptive early-exit**: A3_ADAPTIVE ∈ {off, on} at K=4. Melchord
   material contains few silent frames, so the expected result is a
   null effect; observing one confirms the mechanism is
   silence-specific rather than a general confound.
3. (Optional) **Temperature schedule**: piecewise draft/commit schedule
   vs linear annealing.

#### E4c — MoE ablation: A.2 vs A.2-dense

**Promoted out of E4.** The dense arm is now one of the four arms of
**E6 — MoE ablation**, its own experiment. The design, the
pre-registered expectations and the training command live there; this
entry is kept so older references to "E4c" resolve. E4a/E4b are
unaffected.

### E5 — (Optional) Anticipation-horizon sweep (B.1 only)

Run only if the E3 horizon-spectrum result is positive (B.1 recovers a
substantial fraction of the C.2−A.2 gap). Train B.1 at
k ∈ {8, 16, 32} (`ANTICIPATION_FRAMES` knob in
`train_duet_anticipatory.sbatch`; k=16 already trained) and plot the
E3 conditional metrics against k, with A.1 (ablation system) as the
k=0 point and C.2 as the k=∞ asymptote. Deliverable: one dose–response
figure locating the knee of the lookahead curve.

---

### E6 — MoE ablation (RQ5): does *how* the experts are routed matter?

Its own experiment, not a sub-ablation. E4a asks whether the
attention-side coupling is load-bearing; E6 asks a complete question
about the capacity side: is expert routing load-bearing at all, and if
so, does the ROUTING DESIGN change what the model learns? The four arms
form a spectrum of how much modality separation the architecture
imposes versus permits.

**This experiment is interpretability-first.** Its headline deliverable
is a mechanism account backed by probes — what the router actually
keys on — with generation quality as the accompanying check that the
mechanism change costs nothing. Arms are therefore chosen to make
mechanisms *distinguishable*, not merely to win a metric.

| arm | FFN | who may reach which expert | what it isolates |
|---|---|---|---|
| **D1 (dense)** | one dense FFN (`moe_num_experts=1`, inter 6144) | n/a — no routing | the floor: is routing load-bearing at all? cross-stream interaction survives only in attention |
| **A2 (shared)** | 4 experts, top-2, ONE router | any token → any expert | the status quo, whose apparent specialisation the probes traced to the attention stamp |
| **D2 (hard route)** | 4 experts, top-2, **disjoint pools** | mod_a → {0,1}, mod_b → {2,3}, enforced | separation IMPOSED (MoMa / VL-MoE / Uni-MoE style): an integrator expert is unrepresentable |
| **D3 (per-mod gates)** | 4 experts, top-2, per-modality routers | any token → any expert, scored by its own gate | separation AVAILABLE but not imposed: specialists and integrators both representable, the split is learned |

**The load-bearing contrast is D2 vs D3.** Both give each stream its
own routing decision; they differ ONLY in whether the model may also
learn to let an expert serve both streams. A2 and D1 bracket
them from below. This is the experimental form of the design intent:
*experts may specialise by modality, the modalities must not be fully
separated, and some experts should be able to integrate the two* —
with which experts do what left to training rather than assigned.

**Fairness of the hard-route arm (matters, or the contrast is rigged).**
D2 (hard route) keeps the same parameters and the same activated compute (2
experts per token) as the shared-pool arms — the only removed
capability is cross-stream expert sharing. Its load-balancing aux loss
is computed WITHIN each pool: the standard all-expert form would demand
a uniform load over 4 experts that hard routing cannot produce, taxing
the arm for its own architecture. Verified by `audit_moe_hard_route.py`
(pool disjointness, distribution validity, aux fairness, parameter
parity, checkpoint detectability, unrepresentability of integrators).
Note the E=4 default makes within-pool routing a weighting rather than
a selection (top-2 of a 2-expert pool is both); run `MOE_NUM_EXPERTS=8`
if within-pool sparsity must be preserved, and report the 2× expert
parameters when doing so.

**Established mechanism results (A2 (shared) vs D3 (per-mod gates)).** The shared gate's
apparent modality specialisation is ~69% architectural stamp: content
equalisation removes only ~31% of the routing separation, and expert
preferences follow moved content in 0/11 layers. A per-modality bias
offered as a free shortcut went unused — falsified against a
pre-registered metric, and explained: the bias enters the logits
identically to the stamp term, adding no representable function. Under
per-modality gates, specialists and integrators emerged over the
unassigned pool; chord-side within-stream content-routing rose 5→9 of
12 layers while melody held its ceiling (9→10); zero dead experts
teacher-forced AND free-running; deadliness 0.246/0.308 (mg) vs
0.248/0.364 (shared). Figures in `midi_yinyang/figures/`
(`expert_purity`, `content_probe`, `expert_load*`, `load_distribution`).

**Pre-registered predictions for the new arms.**
1. *D2 (hard route) routing*: within-pool content-responsiveness should be
   comparable to D3 (per-mod gates)'s (both removed the stamp from the routing
   decision), so a per-modality router is not by itself the gain.
2. *D2 (hard route) quality*: at parity or slightly below D3 (per-mod gates). If D3 (per-mod gates) wins, the
   integrator experts it learned are doing work that disjoint pools
   cannot express — the design intent confirmed. If D2 (hard route) matches D3 (per-mod gates),
   the honest conclusion is that cross-stream expert sharing is NOT
   load-bearing here and the simpler imposed split suffices; the
   interpretability result (what the shared router keyed on) stands
   either way.
3. *D1 (dense)*: deficit should be metric-selective on H3 stream-grammar
   metrics; a uniform deficit means MoE capacity helps generically.

**Analysis protocol per arm.** `analyze_moe_routing.sbatch` with
`PROBE=within` (the only probe meaningful on all four arms — the
stamp/swap probes are refused on D2 (hard route), where slot identity is a wall
rather than a preference, and the analyzer says so instead of printing
a confidently wrong verdict); free-running utilization via
`MOE_ROUTING_STATS`; purity tables read as learned structure ONLY on
the shared-pool arms — on D2 (hard route) purity is 0/100 by construction and load
is bounded by 1/pool, so utilization is compared within pools.

**Output-side evaluation.** `eval_a2_moe_ablation.sbatch` runs every
arm on identical prompts and decode settings, SAVES the sample midis
per (song, arm) as listening material, records per-arm free-running
routing stats, then scores structure metrics plus the standard E1 chain
with `BASELINE=A2`. Arm identity is verified from `load_model`'s
own auto-detect lines (`moe_modality_gates`, `moe_modality_hard_route`)
so a swapped `CKPT_*` aborts before any number is produced.

**Training the new arms.** Match the D3 long-run config exactly
(cp4 family, 100k-step schedule, 2 GPUs) so the four arms are
budget-comparable; `hr` and `densecm` run-dir families cannot collide
with the existing ones:

    # D2 -- hard route (disjoint pools)
    sbatch --exclude=gpu-50,gpu-51 --export=ALL,TASK=melchord,\
    MAX_POLYPHONY=4,MOE_MODALITY_HARD_ROUTE=1,RUN_TAG=long,\
    LR_TOTAL_STEPS=100000 \
        midi_yinyang/train_duet_block_diffusion.sbatch

    # D1 -- dense (no MoE), compute-matched
    sbatch --exclude=gpu-50,gpu-51 --export=ALL,TASK=melchord,\
    MAX_POLYPHONY=4,MOE_NUM_EXPERTS=1,MOE_TOPK=1,\
    MOE_INTERMEDIATE_SIZE=6144,RUN_TAG=densecm_long,LR_TOTAL_STEPS=100000 \
        midi_yinyang/train_duet_block_diffusion.sbatch

Budget-parity note: the A2 (shared) reference checkpoint is the 50k
cp4tar run; a budget-matched shared long run is still an open decision
for the final table.

**Cost.** Two additional melchord training runs (hr, dense) at the same
budget as the existing pair; evaluation is one GPU job for all four
arms. The shared and mg runs already exist.

---

## 4. Decode settings (frozen before any scoring)

| System | Settings |
|---|---|
| S0/S1/S-specialists | temperature 1.0, `--prompt-length 64 --gen-length 384 --n-samples 3` |
| A.2 | `A3_REFINE_STEPS=4 A3_FINAL_TEMP=0.9 A3_TOP_P=0.95 A3_ADAPTIVE=1` — used for ALL headline A.2 rows; the K-sweep is confined to E4. Rationale: K=4 exercises the full trained refinement depth (K = diffusion_K), so the headline tests the complete 同步看 mechanism rather than a compute-trimmed variant; the temperature/nucleus schedule is the drumnondrum listening-test selection; A3_ADAPTIVE guards the known silent-frame regression of deep refinement (drumnondrum listening: K=4 ≤ K=0 on silent-stream material) and is itself ablated in E4b |
| A.1 (ablation only) | temperature 1.0 (its tuned default) |
| B.1/C.1/C.2 | temperature 1.0, existing script defaults |
| Y | temperature 1.0, `cp_transformer_yinyang_inference.py` defaults; the paper's released/finetuned ckpts (Y-dn, Y-mc) as-is |

Freeze these; no per-song or per-metric cherry-picking. Any change ⇒
rerun the whole grid (it's cheap: ≤ ~15 songs × ~6 systems × 3 samples).

---

## 5. Metrics

### 5.1 Objective — implemented in `eval_metrics.py`

The module scores H1–H3 only (H4 is the listening test) and reports in
the pre-registered priority order **H3 > H2 > H1**, with one primary
endpoint per hypothesis (marked `*` in the summary): H3
`harmonic_rhythm_jsd` (melchord) / `onset_grid_jsd_b` (drumnondrum);
H2 `chord_tone_cov_delta` / `onset_sync_delta`; H1 `survival_min`.
Reference-calibrated throughout: distribution metrics are
Jensen–Shannon divergences against the ground-truth continuation's
distributions, fit metrics are deltas against the same statistic
computed on the reference pair. Batch interface: a TSV manifest
(system, mode, song, sample, path) + reference stream folders → CSV +
priority-ordered summary. Validated on synthetic fixtures where each
failure mode (grammar chaos, stream death) trips exactly its own
hypothesis block.

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
3. Marginal specialists: `finetune_pop909.py --data
   data/pop909_melody_cp8_v2.pt` → S-mel and `--data
   data/pop909_chord_cp8_v2.pt` → S-chord (same recipe as S1; use
   `--run_tag mel` / `--run_tag chord` so the run dirs stay distinct).
   S-drum / S-nondrum deferred until the melchord E2 gate decides.
4. Build POP909 eval prompt folders for ids `001 011 ... 091`:
   split melody/chord folders (already exist) + tagged combined folder
   (for S0/S1). RWC prompt folders already exist.
5. Locate the Y-dn / Y-mc checkpoints on the cluster.
   (`eval_metrics.py` is done — §5.1.)

Phase 1 — generation sweeps (all sbatch, mostly existing scripts)
- E1/E2/E3 duet lanes: `infer_all_rwc.sbatch` (drumnondrum) and
  `infer_duet_block_diffusion.sbatch` with melchord ckpt +
  `MEL_FOLDER/CHORD_FOLDER/MAX_POLYPHONY=4` (melchord).
- E1 baselines: `cp_transformer_inference.py` on the tagged combined
  prompts (S0 and S1 ckpts).
- E2 baselines: `cp_transformer_inference.py` on the single-stream
  prompt folders with the matching specialist ckpt per stream (S-mel
  on melody prompts, S-chord on chord prompts; never crossed).
- E3 conditional baseline: `cp_transformer_yinyang_inference.py` with
  the Y-dn ckpt on the RWC prompt pairs (both directions) and the Y-mc
  ckpt on the POP909 eval pairs (both directions, domain caveat noted).
- E4a: A.1 drumnondrum runs (co + conditional, existing ckpt);
  A.1-melchord training deferred until the drumnondrum result is in.
- E4b: three additional A.2 melchord runs (K sweep + adaptive toggle).
- E6 (MoE ablation): two additional melchord TRAINING runs — hard route
  (`MOE_MODALITY_HARD_ROUTE=1`) and dense (`MOE_NUM_EXPERTS=1`,
  inter 6144, the former E4c arm); the shared and mg runs exist. Then
  ONE `eval_a2_moe_ablation.sbatch` job covering all four arms at the
  frozen decode settings, plus `analyze_moe_routing.sbatch PROBE=within`
  per arm for the interpretability half.

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
