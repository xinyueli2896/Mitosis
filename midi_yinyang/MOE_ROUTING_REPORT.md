# MoE router analysis — A.2 (melchord, POP909-only)

**Task 6: analyse router usage (make sure no collapse) and improve the
MoE architecture.** This report covers the first half and sets up the
second.

- **Checkpoint** `m2c_duet_block_diffusion_v1.2_large_gnl12_K4_melchord_cp4tar_batch_8_schedule`
  (best-val, `val_loss=0.36469`), v1.2 time-aligned RoPE, K=4, cp4,
  POP909-only melchord.
- **Method** `analyze_moe_routing.sbatch` — offline, CPU-only: load the
  checkpoint, run one val batch through the model's own `loss()`, read
  each `SimpleMoEFFN`'s cached routing probabilities (`[B, L, E]`).
  Batch = 2 songs × 384 frames, interleaved → ~1.5k routed tokens per
  layer, ~770 per modality.
- **Evidence** SLURM jobs **178410** and **178528** (independent val
  batches, same checkpoint); stamp-vs-content probes **178945**
  (identical) and **178946** (swap). Logs: `~/logs/moe_routing_*.out`.

## Verdict

1. **No collapse.** Zero dead experts across all 12 layers in both
   batches; worst top-1 load 0.454 / 0.516 (jobs 178528 / 178410)
   against a 0.25 balanced ideal. `aux_loss_weight=0.01` is doing its
   job; do not raise it.
2. **Routing is soft.** Normalized entropy 0.83–0.93 (mean 0.87):
   the top-2 experts are blended with similar weights rather than
   selected sharply.
3. **Experts route by modality — but the probes show the router mostly
   reads the ARCHITECTURE'S IMPRINT, not the music.** Behaviourally the
   partition is strong (mean per-layer specialisation 0.82; layers 0
   and 11 near-pure). Mechanistically, equalising the content across
   parities removes only ~31% of the separation (mean L1 0.589 → 0.408,
   **stamp share ~69%**), and when the contents are swapped, expert
   preferences follow the moved content in **0 of 11** layers. The
   diverged per-modality projections (`o_m` vs `o_c`) stamp slot parity
   onto every hidden, and the router piggybacks on that stamp. The
   correct claim is therefore *"the architecture's modality signal
   propagates into the FFN routing"*, not *"the router discovers the
   streams unsupervised"*.
4. **The stamp is installed immediately; content routing accumulates
   with depth.** Layers 0–3 are ~94% stamp — which RESOLVES the layer-0
   caveat: the 0.97 purity there was never density-reading. Content's
   share grows in the middle and late layers (layer 9 is
   majority-content at 56%), so the genuine content-based routing the
   MoE does happens on top of, and after, the parity signal.

## Why three different metrics disagreed

Each measurement lens hid what the next revealed. This is the
methodological finding, and it generalises:

| lens | value on this model | what it suggests | why it misleads |
|---|---|---|---|
| mean routing probability (what `MoERoutingMonitor` logs) | mod L1 = 0.58 / 2.0 | "mild separation" | soft routing (entropy 0.87) flattens the mean even when the argmax is decisive |
| top-1 load per expert | 0.25 ± 0.2, no dead experts | "balanced, nothing special" | a clean 2-experts-per-stream partition IS balanced — specialisation does not show up as load skew when E = 2 × streams |
| **expert purity** P(modality \| expert) — share of the tokens an expert *wins* that are melody | **mean spec 0.82; layers 0, 10, 11 at 0.97 / 0.93 / 1.00** | strong implicit partition | — (this is the direct question) |

Load and mean-prob are P(expert | modality)-flavoured views; the
specialisation question is P(modality | expert). They come apart exactly
when routing is soft and the partition is balanced — both true here.

## Data

Job 178528 (primary; job 178410 replicates every pattern below on an
independent batch — mean L1 0.590 vs 0.576, the same layers peak, and
layer 6's heavy expert is e2 in both).

Per-layer routing health:

| layer | max_load | entropy | dead | top-1 load e0–e3 | mod L1 |
|---|---|---|---|---|---|
| 0 | 0.341 | 0.826 | 0 | 0.281 0.341 0.197 0.181 | 0.386 |
| 1 | 0.314 | 0.931 | 0 | 0.242 0.250 0.195 0.314 | 0.478 |
| 2 | 0.297 | 0.926 | 0 | 0.256 0.297 0.264 0.183 | 0.392 |
| 3 | 0.311 | 0.883 | 0 | 0.311 0.278 0.232 0.179 | 0.546 |
| 4 | 0.395 | 0.875 | 0 | 0.273 0.147 0.395 0.184 | 0.711 |
| 5 | 0.368 | 0.910 | 0 | 0.234 0.186 0.212 0.368 | 0.413 |
| 6 | 0.454 | 0.834 | 0 | 0.181 0.167 0.454 0.198 | 0.696 |
| 7 | 0.333 | 0.841 | 0 | 0.117 0.290 0.333 0.260 | 0.702 |
| 8 | 0.318 | 0.839 | 0 | 0.195 0.297 0.318 0.189 | 0.569 |
| 9 | 0.338 | 0.844 | 0 | 0.217 0.338 0.206 0.239 | 0.560 |
| 10 | 0.308 | 0.856 | 0 | 0.232 0.283 0.177 0.308 | 0.573 |
| 11 | 0.316 | 0.848 | 0 | 0.249 0.279 0.316 0.156 | 0.881 |

Expert purity — of the tokens each expert wins, the % that are melody
(base rate 50%; 0% = pure chord expert, 100% = pure melody expert):

| layer | e0 | e1 | e2 | e3 | spec |
|---|---|---|---|---|---|
| 0 | 1.6 | 58.5 | 93.4 | 62.0 | 0.97 |
| 1 | 85.5 | 9.4 | 32.3 | 66.0 | 0.81 |
| 2 | 6.8 | 83.8 | 53.4 | 50.7 | 0.86 |
| 3 | 83.7 | 23.1 | 62.5 | 17.0 | 0.67 |
| 4 | 20.0 | 86.3 | 77.8 | 6.0 | 0.88 |
| 5 | 46.1 | 47.0 | 8.6 | 77.8 | 0.83 |
| 6 | 49.5 | 24.9 | 78.4 | 6.6 | 0.87 |
| 7 | 20.6 | 72.7 | 66.7 | 16.5 | 0.67 |
| 8 | 37.5 | 66.2 | 65.7 | 11.0 | 0.78 |
| 9 | 50.0 | 77.7 | 26.4 | 31.2 | 0.55 |
| 10 | 3.4 | 60.6 | 54.4 | 72.8 | 0.93 |
| 11 | 100.0 | 82.3 | 3.3 | 6.7 | 1.00 |

Stamp-vs-content probe (job 178945): parity L1 with real content vs
with melody duplicated into the chord slots. What survives equalisation
is the architectural stamp:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 real | .36 | .47 | .38 | .53 | .80 | .42 | .74 | .69 | .62 | .59 | .59 | .88 |
| L1 identical | .33 | .44 | .39 | .49 | .52 | .25 | .41 | .36 | .43 | .26 | .40 | .63 |
| stamp share | 91% | 93% | 100% | 92% | 65% | 59% | 55% | 52% | 69% | 44% | 68% | 72% |

Swap probe (job 178946): slot preserved exactly in 2/11 usable layers,
content followed in 0/11, "neither" in 9/11 — swapped inputs are
out-of-distribution for parity-specialised projections, itself evidence
of the stamp.

The layer-11 partition read together with load: melody pair e0+e1 takes
0.249+0.279 = 0.528 of routing, chord pair e2+e3 takes 0.316+0.156 =
0.472 — a near-even 2+2 split, which is why the load view saw nothing.

## Caveats

- **RESOLVED: the stamp-vs-content question.** The two candidate
  mechanisms (router reads content vs router reads the parity imprint
  of the diverged `o_m`/`o_c` projections) were separated by the
  probes: **stamp-driven, ~69%**, with the content component (~31%)
  concentrated in middle/late layers. Both mechanisms are learned —
  the stamp is absent at the warm start, whose branches are identical
  copies — but the discovery lives in the attention projections, not
  the router. The originally proposed drumnondrum control was retracted
  as confounded (drums carry program 127, leaking modality).
- Purity attributes each token to its argmax winner, but routing is
  top-2 with similar weights (entropy 0.87), so purity overstates
  commitment just as the mean-prob L1 understates it. The truth is
  between 0.58/2.0 and 0.82/1.0 — still clearly a partition.
- One checkpoint, one task, one seed, ~1.5k tokens per layer per batch
  (two independent batches agree on every headline pattern).

## Recommendations

1. **Do not raise `aux_loss_weight`.** There is no collapse to fix, and
   a stronger balancing pressure would fight the (healthy) 2+2 split.
2. **Probes done (jobs 178945/178946): stamp-driven.** Results in the
   Data section; the verdict above is restated accordingly.
3. **For "improve MoE" the case for a modality-informed router is now
   at its strongest.** The router spends its capacity relaying a parity
   signal the architecture computes explicitly anyway. Hand it over for
   free — a per-modality bias added to the router logits (no new
   module; the modality of every position is known) — and the learned
   routing is released for the content signal that today only reaches
   ~31% and only in the deeper layers. Success metrics: the content
   share of routing (re-run the identical probe on the new ckpt — it
   should FALL toward 0 as the bias absorbs the parity job) and
   within-modality routing structure emerging (register, density,
   harmonic function).

   **BUILT (2026-08-24) as A.2.moe_improved** — see VARIANTS.md for the
   full description. `SimpleMoEFFN` gains an opt-in zero-init
   `modality_bias [2, E]` on the router logits, wired through the A.2
   layer stack only; `train_duet_block_diffusion.sbatch` knob
   `MOE_MODALITY_BIAS=1` (run dirs marked `K4mb`). Inference and
   `analyze_moe_routing.py` auto-detect the bias from the ckpt; on a
   bias ckpt the analyzer separates the explicit-bias vs content
   pathways and runs the probes on the content pathway, which is where
   the success metric above lives. Correctness (warm-start equivalence,
   layout, probe-cache isolation) is audited by
   `audit_moe_bias.sbatch`.
4. **Extend the same analysis to the other MoE checkpoints** (A.1, B.1,
   C.1, C.2, and A.2-drumnondrum) — the analyzer takes any of them via
   `VARIANT=`.

## Reproduction

```
sbatch --export=ALL,VARIANT=a2,TASK=melchord,MAX_POLYPHONY=4,\
CKPT=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4_melchord_cp4tar_batch_8_schedule/ \
    midi_yinyang/analyze_moe_routing.sbatch
```

Drumnondrum control:

```
sbatch --export=ALL,VARIANT=a2,TASK=drumnondrum,\
CKPT=ckpt/<the drumnondrum A.2 run>/ \
    midi_yinyang/analyze_moe_routing.sbatch
```
