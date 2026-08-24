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
  batches, same checkpoint). Logs: `~/logs/moe_routing_1784{10,528}.out`.

## Verdict

1. **No collapse.** Zero dead experts across all 12 layers in both
   batches; worst top-1 load 0.454 / 0.516 (jobs 178528 / 178410)
   against a 0.25 balanced ideal. `aux_loss_weight=0.01` is doing its
   job; do not raise it.
2. **Routing is soft.** Normalized entropy 0.83–0.93 (mean 0.87):
   the top-2 experts are blended with similar weights rather than
   selected sharply.
3. **Experts HAVE specialised by modality — strongly, and without ever
   being told the modality.** Mean per-layer specialisation 0.82
   (1.00 = a pure per-stream expert); layer 11 reaches 1.00, layer 0
   reaches 0.97. The router is a bias-free `Linear(hidden, 4)` with
   `token_type_embeddings` zeroed and frozen, so modality reaches it
   only through the per-modality attention output — and it recovers the
   stream boundary anyway.

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

The layer-11 partition read together with load: melody pair e0+e1 takes
0.249+0.279 = 0.528 of routing, chord pair e2+e3 takes 0.316+0.156 =
0.472 — a near-even 2+2 split, which is why the load view saw nothing.

## Caveats

- **Two candidate mechanisms, and the purity number does not separate
  them.** In the ENCODING nothing distinguishes the streams (same local
  encoder, `token_type_embeddings` zeroed, programs constant 0 on
  melchord) — but the router reads the hidden AFTER the per-modality
  attention projections, and once training diverges `o_m` from `o_c`,
  every position is STAMPED with its slot parity before the first
  router sees it. So high purity is either **content-reading** (the
  router recognises sparse-monophonic vs dense frames — the layer-0
  score of 0.97 makes density the prime suspect) or **stamp-reading**
  (the attention projections specialised and the router merely reads
  their imprint). Both are learned — the stamp is absent at the warm
  start, whose branches are identical copies — but they are different
  claims about what the MoE is doing.
- **The probe that separates them needs no retraining**:
  `--probe identical` duplicates the melody content into the chord
  slots, so any parity separation that survives IS the stamp;
  `--probe swap` exchanges the contents and asks whether expert
  preferences follow the slot or the content. This supersedes the
  drumnondrum control originally proposed here, which turns out to be
  confounded: drums are encoded as program 127, so the program token
  would leak modality to the router there.
- Purity attributes each token to its argmax winner, but routing is
  top-2 with similar weights (entropy 0.87), so purity overstates
  commitment just as the mean-prob L1 understates it. The truth is
  between 0.58/2.0 and 0.82/1.0 — still clearly a partition.
- One checkpoint, one task, one seed, ~1.5k tokens per layer per batch
  (two independent batches agree on every headline pattern).

## Recommendations

1. **Do not raise `aux_loss_weight`.** There is no collapse to fix, and
   a stronger balancing pressure would fight the (healthy) 2+2 split.
2. **Run the stamp-vs-content probes** (one CPU sbatch each, ~minutes)
   before putting the "implicit modality routing" claim in any writeup:

   ```
   sbatch --export=ALL,VARIANT=a2,TASK=melchord,MAX_POLYPHONY=4,PROBE=identical,\
   CKPT=ckpt/m2c_duet_block_diffusion_v1.2_large_gnl12_K4_melchord_cp4tar_batch_8_schedule/ \
       midi_yinyang/analyze_moe_routing.sbatch
   ```

   and the same with `PROBE=swap`. They print a per-layer baseline
   comparison and a verdict (content-driven / stamp-driven / mixed).
3. **For "improve MoE": a hard modality→expert route is now pointless**
   — the router already found that partition unaided. The interesting
   variant is the opposite: *give* modality to the router for free (a
   per-modality bias added to the router logits — no new module; the
   modality of every position is known) and see whether the experts then
   differentiate on something else: register, density, harmonic
   function. Measure success as within-modality routing structure
   emerging, not as loss.
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
