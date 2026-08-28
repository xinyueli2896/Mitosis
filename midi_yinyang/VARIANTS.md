# Variant lineup

The model variants in this directory fall into a small number of
conceptual roles relative to each other. This document is the canonical
notes file for the framing.

All variants share: CP tokenization, the same single-stream pretrained
backbone for warm-start, per-modality Q/K/V/O projections, RoPE inside
attention, and a shared MoE FFN (4 experts, top-2 routing). They differ
in **sequence layout**, **attention mask**, and **loss**.

For the task `drumnondrum`: mod_a = drum, mod_b = nondrum.

## Lineup

| # | Class | File | Sequence layout | Attention | Loss | Direction |
|---|---|---|---|---|---|---|
| 2 | `M2CIntraCrossAttn` | `cp_transformer_m2c_intra_cross_attn.py` | interleaved `[a_0, b_0, a_1, b_1, …]` | 2 SDPA passes, both causal | CE on every token | symmetric |
| 3 | `M2CDuetRehearsal` | `cp_transformer_m2c_duet_rehearsal.py` | drum prefix `[drum_0..drum_{T-1}]` + standard DuetAttn shifted interleaved suffix (length 3T total) | 2 SDPA passes; prefix bidirectional within, suffix sees all prefix + causal within suffix | CE on entire 2T suffix + Brier-MSE recon on drum logits (drum-side both collapse fast; useful signal in nondrum CE) | symmetric joint, drum-conditioned |
| 4 | `M2CDuetBlockAttn` | `cp_transformer_m2c_duet_block.py` | interleaved + 2 appended query slots | 3 SDPA passes (intra / cross-strict-past / frame-bidirectional) | AR-CE + query-CE | symmetric |
| 5 | `M2CDuetPrefix` | `cp_transformer_m2c_duet_prefix.py` | `[drum_0, …, drum_{T-1}, sos_n, nondrum_0, …, nondrum_{T-2}]` | 2 SDPA passes; drum-drum bidirectional, nondrum-nondrum causal, nondrum→drum cross | CE on nondrum positions only | one-way drum→nondrum |
| 6 | `M2CDuetAnticipatory` | `cp_transformer_m2c_duet_anticipatory.py` | interleaved with drum shifted ahead by k frames: `[drum_k, nondrum_0, drum_{k+1}, nondrum_1, …]` | same as #2 (architecture identical to DuetAttn) | standard CE | symmetric joint, drum-anticipated |

Variant #1 (`M2CJointAttn`) is conceptually retired — superseded by
DuetAttn — but its file stays in the repo because newer variants
import shared helpers (`_rope_freqs`, `_apply_rope`, `SimpleMoEFFN`)
from it.

## Conceptual roles

### DuetAttn (#2) — base / reference

The reference architecture. Symmetric joint AR over both modalities on
the interleaved sequence. Everything else is measured against this.

### DuetRehearsal (#3) — conditioning baseline, joint generation with drum-rehearsal prefix

The genuine "rehearsal" design that the retired
M2CIntraCrossAttnRecon was originally trying (and failing) to
implement via loss alone. Architecturally:

- **Drum prefix** (T positions): the entire drum stream prepended to
  the global stack input, bidirectional within itself.
- **Interleaved suffix** (2T positions): standard DuetAttn-style
  shifted causal AR sequence (drum, nondrum interleaved), with full
  visibility into the prefix.
- **Total sequence length**: 3T.

Loss is standard CE on the entire 2T suffix plus a **Brier-style MSE
recon term on drum logits** (`||softmax(drum_logits) −
one_hot(drum_target)||²` over non-pad suffix-drum slots), controlled
by `recon_weight` (default 1.0). With the prefix giving full drum
visibility, both the drum-side CE and the recon term collapse fast —
the model can trivially copy `drum_k` from the prefix to its
corresponding suffix slot. That's fine: the recon term provides the
explicit "match the drum you just saw in rehearsal" gradient that
the original recon variant was trying to capture, and the useful
signal for ablation lives in the **nondrum CE**, where each
`nondrum_k` prediction now sees the **entire drum stream** (past and
future) via the prefix, plus causal nondrum past, plus the suffix's
drum past (which itself is conditioned on the prefix).

Why this is different from #5 DuetPrefix: #5 is **one-way drum→nondrum**
(only nondrum is predicted; drum is pure conditioning). #3 keeps
**symmetric joint generation** in the suffix — both modalities are
predicted, sharing the same architecture as #2 — while making drum
available as a forward-visible prefix. So #3 sits between #2 (no
rehearsal) and #5 (one-way only): joint AR like #2, with drum
conditioning like #5.

Architecture: 2 SDPA passes per block (intra + cross), per-modality
Q/K/V/O, per-modality cross gate (bias = −10 at init), shared MoE FFN —
same machinery as #2; only the mask shape and input layout differ.

Warm-start: per-modality projections from the single-stream
pretrained backbone. Not exact equivalence with #2 (the prefix
changes the key/value distribution even with cross gate silent), but
close — the model needs a brief adaptation phase before it leverages
the prefix usefully.

### DuetRehearsal splits into C.1.A and C.1.B

The rehearsal model has TWO variants, differing only in how the
interleaved suffix is shifted. They are not two settings of one model:
the shift decides what the ungated intra path carries, which cross-gate
init is coherent, and how the prefix must be indexed. Run dirs are named
`m2c_duet_rehearsal_C1A_...` / `_C1B_...` so `resolve_best_ckpt` (which
scans a directory) can never confuse them.

|  | **C.1.A** (shift-2) | **C.1.B** (shift-1) |
|---|---|---|
| suffix | `[sos_m, sos_c, x_0 …]`, slot *i* holds x_{i-2} | `[sos_m, x_0 …]`, slot *i* holds x_{i-1} |
| slot predicting `b_k` holds | `b_{k-1}` | **`a_k`** |
| intra path (ungated) carries | mod_b's own history | mod_a |
| mod_b's own history reached via | intra | **cross (gated)** |
| `a_k` reaches `b_k` via | prefix only | prefix **and** the query slot |
| coherent `GATE_INIT_BIAS` | **-10** (warm start intact) | **0.0** (else no AR history at init) |
| prefix rotary index | `2j+2` | `2j+1` |
| rotary offset, `b_k` query → prefix `a_k` | -1 | 0 |
| `SUFFIX_SHIFT1` | `0` | `1` (the default) |

C.1.A is the conservative reading: it keeps the co-generation shift the
DuetAttn lineage uses, so the warm start lands exactly as intended and
the prefix is the sole route to same-frame and future mod_a. C.1.B makes
`a_k` locally available inside the interleave, at the cost of moving
mod_b's AR history behind the gate.

Both share the corrections that are not optional:

- **`prefix_stride2`** — the prefix copy of frame *j* takes the same
  rotary index as the suffix slot holding it, so one musical event has
  one coordinate. The offset tracks the shift (see table). Without it,
  the distance from `b_k`'s query to `a_k` grew as *k*+1, and so did the
  distance for the Brier retrieval term.
- **`target_only_loss`** — no token-level CE on mod_a. It is trivially
  satisfiable and diluted `val_loss`.
- **`recon_weight` > 0** — the Brier term is KEPT. It is not a copy: the
  slot predicting `a_k` never holds `a_k`, so it must retrieve it from
  the prefix by position, over the same `k_m`/`v_m` a mod_b query reads
  on the cross path. It trains the shared half of the conditioning
  mechanism.
- Checkpoints select on `val_ce_loss_nondrum`, since `val_loss` carries
  the Brier term.

The prefix itself carries no loss under either variant: logits are read
only from suffix positions.

### C.1 reconstruction study — results (2026-08-24)

**Conclusion.** The rehearsal mechanism works: with the corrected
prefix geometry, C.1 recovers the conditioning modality from its prefix
essentially perfectly even under free-running sampling (Jaccard 0.999),
where the legacy geometry manages only 0.830 despite looking identical
teacher-forced — proof that the old model was coasting on local
continuation while the fixed one genuinely retrieves. The two design
axes decouple cleanly: prefix geometry governs how well the conditioning
modality can be reconstructed, while the suffix shift governs the
generation modality's quality — shift-1 (C.1.B) proves pure retrieval is
learnable but collapses generation from 0.95 to 0.65 by exiling the
generated stream's own history to the gated cross path. C.1.A — fixed
geometry, co-generation shift — is therefore the canonical C.1, the only
configuration strong in both columns. The mel2chord replication
(2026-08-25, below) confirms retrieval is direction-independent at both
the teacher-forced and free-running levels.

Terminology: the **conditioning modality** is the stream given in full
(mod_a, the rehearsal prefix); the **generation modality** is the stream
the model produces (mod_b). On the melchord testbed both are symbolic
streams simulating the cross-modal setting.

Setup: chord→melody direction (`melchord_nottingham_rev`), so the
conditioning modality is chord. Two tests per checkpoint
(`analyze_rehearsal_recon.sbatch` teacher-forced + position-resolved;
`infer_duet_rehearsal.sbatch MODE=reconstruct` free-running, scored as
per-frame pitch-duration Jaccard). SLURM jobs 178861–178864 and
178877–178878.

| checkpoint | geometry | shift | cond. recon (TF) | cond. recon (free-run) | gen. prediction (TF) |
|---|---|---|---|---|---|
| legacy v1.0 | growing | 2 | 0.998, flat | **0.830** | 0.961 |
| **C.1.A** | fixed | 2 | 1.000, flat | **0.999** | **0.952** |
| C.1.B | fixed | 1 | 0.950, flat | 0.847 | 0.647 |

Findings:

1. **Geometry governs reconstruction of the conditioning modality.**
   At the same shift, fixing the prefix rotary indexing moves
   free-running reconstruction from 0.830 to 0.999. Teacher-forced
   scores could not see this (legacy 0.998): with clean context, a
   shift-2 model can *continue* a sustained chord instead of retrieving
   it, and only free-running — where its own errors must be re-anchored
   against the prefix — exposes the broken addressing.
2. **Shift governs the generation modality's quality.** C.1.B's shift-1
   puts the conditioning frame in the predicting slot but exiles the
   generation modality's own history to the gated, wrong-branch cross
   path; its generation prediction collapses 0.952 → 0.647 (audibly
   chaotic output) while C.1.A's is unharmed. C.1.B's value is as the
   diagnostic: with continuation impossible, its 0.95-flat TF curve and
   0.847 free-running score are *pure prefix retrieval* — the proof the
   rehearsal mechanism works.
3. **C.1.A is the canonical C.1**: the only configuration weak in
   neither column. Legacy matched it on generation but fails the
   retrieval test; C.1.B proves retrieval but breaks generation.

**mel2chord replication (2026-08-25, jobs 179919/179920).** Forward
direction (`melchord_nottingham`): the conditioning modality is melody —
sparse, non-repetitive, no sustain to hide behind. Teacher-forced,
position-resolved:

| checkpoint | shift | cond. recon (TF) | error vs gen. floor | curve | cond. recon (free-run) |
|---|---|---|---|---|---|
| C.1.A fwd | 2 | 0.998 | 13× smaller (0.0015 vs 0.0201) | flat (+0.003) | **1.000** (every song, every quartile) |
| C.1.B fwd | 1 | 0.999 | 7× smaller (0.0010 vs 0.0066) | flat (+0.004) | 0.997 (first-quartile 0.988, then 1.000) |

Free-running: jobs 179956/179957 (`MODE=reconstruct`, `not_split`
Nottingham melody/chord, 3 songs). Retrieval holds at both levels, in
both variants — **the rev-only caveat is retired.** Notable direction
asymmetry, consistent across levels: C.1.B free-runs at 0.997 on sparse
melody where it managed only 0.847 on dense cp8 chords (TF: 0.999 vs
0.950). Since C.1.B's accuracy is pure retrieval (shift-1 makes
continuation impossible), the asymmetry lives in the content difficulty
— a chord frame is a many-token exact match, a melody frame nearly
monophonic — not in the mechanism. Conclusions unchanged: C.1.A remains
canonical (its generation quality is what C.1.B lacks; both retrieve).
Note the first forward level-3 attempt (jobs 179922/179923) ran on the
default RWC drumnondrum prompt folders — out-of-domain drum content
into a melchord ckpt, Jaccards 0.35/0.30, not comparable, discarded.

Caveats: legacy vs C.1.A is not a single-variable ablation (loss form, gate
schedule and LR horizon co-vary with geometry; a `PREFIX_STRIDE2=0`
C.1.A run would isolate it). Token accuracies are EOS-inflated on
sparse streams, and each run drew its own val batches, so only large
gaps are meaningful.

### DuetPrefix (#5) — conditioning baseline, architecture-side

Different conditioning route: drum is a **hard prefix** that the
nondrum block reads via full cross attention. Conditioning enters
through the **architecture itself** (mask shape), not the loss. One-way
drum→nondrum by construction.

#3 (DuetRehearsal) and #5 (DuetPrefix) together form the
**conditioning-baseline pair**. #3 keeps symmetric joint generation
(same target structure as #2) while making drum visible as a
rehearsal prefix; #5 commits to one-way drum→nondrum entirely. #4 is
the symmetric same-instant fix that the lineup is testing against
both.

### DuetAttn-Block (#4) — the fair-looking fix

Targets the **asymmetry problem** baked into the base. Under strict
causal attention in the interleaved layout, `b_t` sees `a_t` (same
frame) but `a_t` does not see `b_t` — same-instant interaction is
one-directional and direction-dependent. Swap the modality labels and
the model's outputs change non-trivially.

DuetAttn-Block introduces **appended next-frame query slots** with
bidirectional within-frame attention between them, restoring **symmetric
same-instant coupling** while preserving streaming + per-frame causal AR
across frames. The base's 2 SDPA passes become 3 (intra / cross-strict /
frame); the base's single cross gate becomes 2 gates (cross + frame),
both bias=-10 at init for warm-start equivalence on the AR forward path.

### Terminology: k is a COMMITMENT level, not a noise level (2026-08-28)

Decision recorded for all A.2 (`M2CDuetBlockDiffusion`) writing. The
diffusion vocabulary the variant inherits ("noise level" for k) is not
what the mechanism does: a query slot is one frame VECTOR (the
local-encoder bottleneck), so corruption is all-or-nothing masking per
slot — no intermediate corruption state exists, and for a masked slot
the target is statistically independent of k. What k actually carries
is coordination: each slot reads the partner's k through the frame
pass (committed vs guessing), k indexes the refinement round at
inference (drafts improve as k falls), and the (k_m, k_c)
configuration is how a decode schedule — parallel refinement vs
MaskGIT commit-then-condition — is expressed to one checkpoint. The
honest one-line description of the decode mechanism is **symmetric
iterative negotiation over a two-slot block, with a commitment tag**.
Prose and figures say "commitment level"; parameter names
(`k_emb_m/c`) are frozen — renaming them would orphan every existing
checkpoint. Full rationale: the TERMINOLOGY note in
`cp_transformer_m2c_duet_block_diffusion.py`. The variant that restores
genuinely graded corruption is now implemented as **A.4** (next
section).

### A.4 — token-level commitment (graded within-frame corruption, 2026-08-28)

The naturalness fix that follows from the terminology note above: if
the corruption is all-or-nothing only because a slot is one frame
VECTOR, corrupt at the TOKEN level instead. Opt-in
(`--token_level_mask` / sbatch `TOKEN_LEVEL_MASK=1`), run-dir tag
`tk`, carried in ckpts by the `token_level_mask_flag` buffer (stored
by VALUE -- plain ckpts carry it as 0; detection must read the value).

**Training.** At commitment level k, each non-pad token of the target
frame is masked independently with prob k/K and the local encoder
embeds the partially-masked frame: intermediate k are genuinely
intermediate states ("chord root known, upper voices open").
Self-conditioning applies at the token level (the draft's tokens are
corrupted, not its encoding). Endpoints are preserved exactly -- an
all-masked draw falls back to the learned `mask_*_emb` (deterministic
at k=K) and k=0 encodes the clean frame -- so the fully-unknown and
fully-known states remain the trained ones and the variant is a strict
generalisation of the plain schedule.

**The mask id is free.** With `with_velocity=False` the vocabulary
pads the program range to 256 with only 128 real programs and caps
pitch-dur ids at 128·25−1 = 3199, so ids 3200..3327 are unreachable in
real data and excluded by `local_sampling`'s valid-token masks. A.4
takes `n_normal_tokens − 1` (3327): no vocabulary change, no embedding
surgery, warm starts load unchanged. The velocity vocabulary has no
free id, so the constructor refuses `with_velocity` + A.4.

**Decoding.** Inference auto-detects the flag. By default the
refinement loop feeds each next round a PARTIALLY re-masked draft: the
(r−1)/K lowest-confidence tokens (teacher-forced probability of the
sampled token, pads exempt) are replaced by the mask id before
re-encoding -- per-frame MaskGIT, matching the graded corruption the
model trained on. `A3_TOKEN_REMASK=0` falls back to full-draft
re-embedding.

**Audited** by `audit_a4_token_mask.py` / `.sbatch`: mask-id safety,
bit-exact endpoint equivalence with the plain variant at k=K and k=0,
masked-fraction statistics, token-level self-conditioning, value-based
flag detection, the velocity guard, and the re-masking helpers.

**Status.** Implemented; audit not yet run on-cluster; no training run
yet. Second-degree relative to the E6/paper scope -- registered so the
naturalness argument has a concrete, costed counterpart.

### A.2.moe_improved — modality-bias MoE router (2026-08-24)

An opt-in change to A.2 (`M2CDuetBlockDiffusion`), motivated directly by
the stamp-vs-content probes (jobs 178945/178946, MOE_ROUTING_REPORT.md):
~69% of the router's melody/chord separation survives content
equalisation, and expert preferences follow moved content in 0 of 11
layers — the router mostly re-derives slot parity from the per-modality
attention projections' imprint on the hidden state. That bit is
redundant (the architecture encodes modality twice already) and it costs
routing capacity: the 2+2 expert partition means each token effectively
chooses between 2 experts, not 4.

The fix hands the router the bit for free: a learned per-modality
additive bias `[2, E]` on the router logits (`SimpleMoEFFN
modality_bias`), zero-initialised so warm-start behaviour is
bit-identical (audited: `audit_moe_bias.sbatch`). The input-driven part
of the routing decision is then free to specialise on within-modality
structure (texture, density, harmonic rhythm) — or, if the modality
partition really was the optimum, the bias simply learns it explicitly
and we learn that instead.

Mechanics:

- `train_duet_block_diffusion.sbatch` knob `MOE_MODALITY_BIAS=1`; run
  dirs get an `mb` marker after the K tag (`..._K4mb_...`) so plain and
  bias runs never auto-resume into each other.
- The bias parameter's presence in the state dict IS the flag: inference
  and the routing analyzer auto-detect it, nothing to pass at decode.
- **Success metric** (falsifiable, fixed before training):
  `analyze_moe_routing.sbatch PROBE=identical` on the new ckpt, where
  the probe automatically runs on the CONTENT pathway (softmax of the
  unbiased logits). The content-pathway stamp share should fall toward
  zero (baseline without the bias: ~69%), and PROBE=swap layers should
  shift from "neither" toward content-following.

**Result (2026-08-25, run `..._K4mb_melchord_cp4tar_batch_8_schedule`,
probes 179683/179684): NEGATIVE — the bias was offered and not used.**
Bias deltas trained to only 0.019–0.056 logits (softmax-negligible),
full and content L1 differ by ≤0.02 everywhere, the content-pathway
stamp share came out ~93% (vs ~69% baseline), and the swap probe found
0/12 content-following layers. The gate kept reading the parity stamp
off the hidden state because nothing in the loss made that read costly
— gradient descent does not migrate a function between redundant
pathways without pressure. val_loss was unaffected (0.340 vs baseline
0.365 — run variance; the change is performance-neutral). Follow-up
options are listed at the end of MOE_ROUTING_REPORT.md: add an
invariance penalty on the unbiased logits (the differentiable success
metric), or a hard 2+2 per-modality expert partition ablation, or close
the line of work with the negative result recorded. The chosen
follow-up is A.2.moe_permod (next section).

### A.2.moe_permod — per-modality router gates (codename D.3, 2026-08-25)

The constructive successor to A.2.moe_improved, built after its negative
result. Instead of *offering* the router the modality label (a bias it
ignored), the router is *restructured* so the label is no longer its
job: two router matrices, `gate_m` for melody slots and `gate_c` for
chord slots, replacing the single shared gate — the same
per-modality-parameters move the duet family applies to Q/K/V/O,
applied to the one per-token module that stayed shared. Each gate only
ever scores its own stream's tokens, so the parity stamp is a
near-constant input component within that population and cannot
influence how one token routes differently from another: within-stream
routing is content-driven by construction, with no loss change.

**The expert pool stays fully shared and unassigned.** No expert is
designated melody, chord, or "integrator" — the design goal (per-stream
specialists coexisting with experts that serve both streams) is left
for training to discover, and read off the analyzer's purity tables: an
expert near 0%/100% purity is a learned specialist, one near the base
rate that both gates keep using is a learned integrator. Precedent:
modality-aware routing is standard in multimodal MoE (MoMa, VL-MoE,
Uni-MoE route per modality but over hard disjoint pools; MoIIE hybrid);
the shared-unassigned-pool version is the departure that makes this an
experiment — melody/chord are far more alike than image/text, so how
much sharing the model wants is genuinely open.

Mechanics:

- `train_duet_block_diffusion.sbatch` knob `MOE_MODALITY_GATES=1`; run
  dirs get an `mg` marker after the K tag (`..._K4mg_...`).
- Warm start: a ckpt carrying the shared `gate.weight` (the init ckpt
  or any trained shared-gate run) seeds BOTH gates with it — step 0 is
  bit-identical to the shared-router model (audited:
  `audit_moe_gates.sbatch`); the gates diverge only from their streams'
  gradients, the q_m/q_c convention.
- The `gate_m/gate_c` keys' presence in the ckpt IS the flag; inference
  and the analyzer auto-detect.
- Readout: `analyze_moe_routing.sbatch` — the mod-L1 column is by
  construction here (two different gates) and is NOT the finding; the
  purity table is. Probe semantics change for this variant (the
  analyzer prints gates-specific verdicts): the identical probe
  measures gate_m-vs-gate_c divergence (expected), and swap's
  SLOT/CONTENT dichotomy does not apply — moved content is scored by
  the other stream's gate, so "neither" is the expected outcome for a
  content-responsive gate; the meaningful split is prefs-changed
  (input-sensitive) vs prefs-unchanged (weight-prior routing).

**Preliminary result (2026-08-26, mid-extension snapshot at best-val
0.368, probes 182634/182635): the specialists+integrators structure
EMERGED without assignment.** Typical layer: one melody specialist,
one chord specialist, and mixed experts winning real traffic from both
streams — e.g. L0: e0 1.7%/e1 93.5% (specialist pair) with e2 at 41%
(integrator); L7: e2 at 51%; L3/L10/L11 near-pure specialist pairs
(0.0%/100%). Loads MORE balanced than the shared-router baseline
(worst max_load 0.36 vs 0.43–0.52), no dead experts. Input
sensitivity: swap changed expert preferences in 9/12 layers; layers
0/4/5 kept theirs (weight-prior routing there). val_loss at the
snapshot ≈ baseline (0.368 vs 0.365; mb run 0.340); final numbers and
final-ckpt probes pending the extended run.

**Within-stream content-responsiveness (PROBE=within, seeded batch,
jobs 182680/182681): the freed routing capacity went to the chord
stream.** On the identical batch (SEED=0), routing follows frame
register in — melody: mg 10/12 layers vs baseline 9/12 (tied; the
shared router's content channel already served melody); **chord: mg
9/12 vs baseline 5/12**, with mg's strongest effects mid-stack (L8
0.245 vs null 0.110). Direction replicates on a second independent
batch. Interpretation: in a shared router, chord content competes with
the parity signal and melody structure for the same weight rows and
loses; a dedicated gate_c starts tracking chord register. (Density and
mean-duration terciles are degenerate on this corpus/batch and skip —
register is the live feature.) Snapshot-level; re-run at the final
extended ckpt.

### A.2.moe_hardroute — disjoint per-modality expert pools (codename D.2, 2026-08-27)

The **control** for A.2.moe_permod, added when the MoE work was
promoted to its own experiment (EXPERIMENTS.md, E6). Not a candidate
design: it exists to make the permod claim falsifiable.

**What it does.** The expert list is split in half by index — mod_a may
only reach experts `[0, E/2)`, mod_b only `[E/2, E)` — by masking the
other pool's logits to `-inf` before the softmax. This is what the
multimodal MoE literature calls modality-specific experts (MoMa,
VL-MoE, Uni-MoE): separation IMPOSED rather than learned. No new
parameters, so the variant is carried by a persistent buffer
`ffn.hard_route_flag` that inference auto-detects; run dirs are tagged
`hr` after the K tag.

**Why it is the right control.** A.2.moe_permod changes two things at
once relative to the shared router: each stream gets its own routing
decision, AND the pool stays shared so an expert may serve both
streams. Hard route keeps the first and removes the second. So

| | own routing decision per stream | expert may serve both streams |
|---|---|---|
| A2 (shared gate) | no | yes |
| D2 (hard route) | yes (by wall) | **no** |
| D3 (per-mod gates) | yes (by gate) | yes |

and D2 vs D3 isolates exactly the property the design intent names:
*modalities specialise but are not fully separated, and some experts
integrate the two.* If D3 > D2, the learned integrators are doing
work disjoint pools cannot express. If they tie, the honest reading is
that cross-stream sharing is not load-bearing on this corpus and the
simpler imposed split suffices — the interpretability result (the
shared router keys on the attention stamp, ~69%) stands either way.

**Fairness, deliberately engineered.** Same parameters, same activated
compute (2 experts per token). The load-balancing aux loss is computed
WITHIN each pool and averaged: the standard all-expert Switch form is
minimised by a uniform load over all E experts, which hard routing
CANNOT produce, so it would act as a constant penalty on the arm's own
architecture rather than as balancing pressure. Verified end to end by
`audit_moe_hard_route.py` / `.sbatch` — pool disjointness at FFN and
layer level (arbitrary gate weights, query slots included), rows still
summing to 1 with finite outputs and live gate gradients through the
`-inf` mask, aux fairness (pool-uniform load scores the same ~1.0 floor
the shared arm gets at ITS optimum, not 2.0), parameter parity with the
shared arm (state dicts differ by the flag buffer alone), flag
persistence for auto-detection, loud misuse, and — over 25 random
routers — that no expert ever wins tokens from both streams.

**Reading its diagnostics (the trap).** On this arm, parity separation
is maximal and expert purity is 0%/100% **by construction**; neither is
a finding. Per-expert load is bounded by `1/pool`, so utilization must
be compared within pools, never against the 1/E line. The stamp/swap
probes are meaningless — slot identity is a wall, not a preference —
and `analyze_moe_routing.py` now REFUSES them on a hard-route
checkpoint with an explanation rather than printing a confidently wrong
"100% stamp" verdict. The meaningful measurement is `PROBE=within`:
inside a pool, does routing still follow the music?

**Geometry knob.** At the default E=4 the pools are 2 wide and top-2
selects both, so within-pool routing is a weighting rather than a
selection — acceptable for the primary comparison because parameters
and activated compute stay matched. `MOE_NUM_EXPERTS=8` preserves
sparsity inside the pools (2 of 4) at 2× expert parameters; the eval
harness takes `HR_EXPERTS`/`HR_TOPK` and the table must then report the
parameter difference. The constructor rejects an odd expert count or
`topk > E/2` rather than silently degrading.

**Status.** Implemented and audited; training run not yet launched.

## Experimental story

The variants bracket the question:

> Does **symmetric same-instant coupling** — achieved without committing
> to a hard conditioning direction — beat both **conditioning baselines**
> (#3 loss-side, #5 architecture-side) and the **asymmetric base** (#2)?

The four-way comparison reads as:

- **#2 vs #4**: does fixing the asymmetry help at all?
- **#3 vs #4**: is the symmetry fix doing something the rehearsal-style
  conditioning baseline (joint generation + drum-prefix context) can't
  reach?
- **#5 vs #4**: is the symmetry fix doing something the one-way
  conditioning baseline can't reach?
- **#3 vs #5**: secondary — which conditioning baseline is stronger,
  joint-with-prefix or one-way-prefix?

If #4 cleanly beats both #3 and #5 while matching or beating #2, the
"symmetry fix without hard conditioning" framing is doing real work and
isn't reducible to "just condition on drum better."

## Shared infra notes

- Warm-start: every variant's init script copies the single-stream
  pretrained backbone into per-modality projections (same backbone →
  identical mod-a and mod-b weights at init, then diverge during
  training). MoE experts are duplicated from the dense FFN. Gates start
  silent (bias = -10) where applicable.
- `M2CIntraCrossAttn` (DuetAttn) and `M2CIntraCrossAttnRecon` share a
  ckpt format — the recon variant adds no new parameters, only a loss
  term — so the existing intra-cross-attn inference script
  (`cp_transformer_m2c_intra_cross_attn_combined.py`) loads recon ckpts
  directly.
- `M2CDuetBlockAttn` and `M2CDuetPrefix` change the sequence shape going
  into the global stack, so their inference loops are not drop-in
  replacements for the base — they need dedicated inference scripts
  (TBD).
