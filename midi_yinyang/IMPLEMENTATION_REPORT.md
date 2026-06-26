# Implementation report — model variants

Status of each variant in the lineup, the files that implement it, and
what's currently runnable end-to-end. For the conceptual framing
(what each variant is *for* in the experimental story), see
`VARIANTS.md`.

## Variants at a glance

| # | Class | Conceptual role | Train | Infer |
|---|---|---|---|---|
| 2 | `M2CIntraCrossAttn` (DuetAttn) | base / reference | ✅ trained, ckpt on cluster | ✅ |
| 3 | `M2CDuetRehearsal` | conditioning baseline — drum-prefix rehearsal + joint interleaved AR suffix | ⏳ not yet trained | ❌ not implemented |
| 4 | `M2CDuetBlockAttn` (DuetAttn-Block) | the fair-looking fix | 🟡 in progress (12k+ steps as of last check) | ✅ Option B implemented (untested) |
| 5 | `M2CDuetPrefix` | conditioning baseline — one-way drum→nondrum | ⏳ not yet trained | ❌ not implemented |
| 6 | `M2CDuetAnticipatory` | anticipatory conditioning baseline — drum stream shifted ahead by k frames in the interleaved input (DuetAttn architecture, data-side reindex) | ⏳ not yet trained | uses #2's inference (state-dict-compatible) with the same shift applied internally |

The previous occupant of slot #3 (`M2CIntraCrossAttnRecon`) has been
retired and its code files deleted. Existing ckpts from that variant
remain valid as a standalone loss-shape ablation but no longer have a
slot in the lineup.

Retired: `M2CJointAttn` (variant #1) — superseded by DuetAttn. File
remains for shared imports.

## Variant #2 — `M2CIntraCrossAttn` (DuetAttn)

The base model. Per-modality Q/K/V/O projections + two key-masked SDPA
passes per block (intra: same-modality causal, cross: other-modality
causal) + per-block cross gate (bias = −10 at init) + shared MoE FFN.
Standard AR cross-entropy loss. Symmetric joint generation over both
modalities on the interleaved sequence.

**Files**:
- `cp_transformer_m2c_intra_cross_attn.py` — model + training script.
- `cp_transformer_m2c_intra_cross_attn_inference.py` — inference helpers (loads ckpt, patches jointattn's `main`).
- `cp_transformer_m2c_intra_cross_attn_combined.py` — folder/mode dispatch.
- `init_pretrained_into_intra_cross_attn.py` — warm-start from single-stream pretrained ckpt.
- `train_intra_cross_attn.sbatch` — SLURM launcher.
- `infer_intra_cross_attn_combined.sbatch` — SLURM inference launcher (5 modes).

**Loss**: `L_CE + 0.01 · L_moe_aux`. Default per-frame and per-token-type weights are 1.0 → effectively plain mean CE over non-pad tokens.

**Training status**: long run in progress / completed. Default ckpt at `ckpt/m2c_intra_cross_attn_v1.0_large_gnl12_drumnondrum_batch_16_schedule/`.

**Inference**: working. Standard run:
```bash
sbatch --export=ALL,CKPT=<path> midi_yinyang/infer_intra_cross_attn_combined.sbatch
```

## Variant #3 — `M2CDuetRehearsal`

The genuine "rehearsal" design. Drum stream is prepended as a
bidirectional prefix to the global stack input; standard DuetAttn-style
shifted interleaved AR runs on the suffix with full visibility into the
prefix.

**Files**:
- `cp_transformer_m2c_duet_rehearsal.py` — model + training script. New
  attention masks; per-modality Q/K/V/O routing reused from DuetAttn;
  per-modality cross gate (bias = −10 at init).
- `init_pretrained_into_duet_rehearsal.py` — warm-start.
- `train_duet_rehearsal.sbatch` — SLURM launcher.

**Sequence layout (length 3T)**:
```
[ drum_0, …, drum_{T-1},   sos_m, sos_c, drum_0, nondrum_0, …, drum_{T-2}, nondrum_{T-2} ]
  └─── drum prefix (T) ─┘  └─────────── shifted interleaved suffix (2T) ────────────┘
  bidirectional within     standard DuetAttn AR; sees all prefix + causal-within-suffix
  invisible to nothing in suffix
```

**Loss**:
```
L_total = CE_suffix + recon_weight · MSE_drum + lambda_aux · L_moe_aux
```
where `MSE_drum = mean ||softmax(drum_logits) − one_hot(drum_target)||²`
over non-pad suffix-drum slots. Default `recon_weight = 1.0`.

Drum-side CE and MSE_drum both collapse fast because the model can copy
`drum_k` from the prefix to the matching suffix slot — that's expected
and is the explicit supervision pinning the rehearsal copy. The useful
gradient signal for the experimental story is on **nondrum-side CE**,
where each nondrum_k now sees the **entire** drum stream via the prefix
(not just past drum), plus standard causal past.

Diagnostic logging splits CE into drum-side vs nondrum-side
(`train_ce_loss_drum`, `train_ce_loss_nondrum`) and logs the recon term
separately (`train_recon_loss`), so the collapse pattern is observable.

**Training status**: not started yet. Run via:
```bash
sbatch midi_yinyang/train_duet_rehearsal.sbatch
```

**Inference**: not implemented. Needs a custom loop maintaining a prefix
buffer of committed drum frames separately from the suffix's standard
shift-trick buffer. Deferred to outstanding work.

**Replaces**: the retired `M2CIntraCrossAttnRecon`. The recon variant's
loss-shape ablation (Brier MSE on drum logits, redundant with CE) didn't
implement actual conditioning — see the "design discussion" thread in
git history. The new #3 implements the rehearsal idea architecturally.

**Files**:
- `cp_transformer_m2c_intra_cross_attn_recon.py` — model (subclasses #2's class, overrides `loss()`).
- `init_pretrained_into_intra_cross_attn_recon.py` — warm-start.
- `train_intra_cross_attn_recon.sbatch` — SLURM launcher.

**Loss**: `L_CE + recon_weight · L_Brier_drum + 0.01 · L_moe_aux`. With `recon_weight = 1.0` by default, the drum reconstruction term is on the same scale as CE.

**Training status**: trained, ckpt on cluster.

**Inference**: **no separate script needed**. The recon variant adds zero new parameters — only a loss term — so its state dict is identical to #2's and loads directly into `M2CIntraCrossAttn` for inference. Use #2's script with a recon ckpt path:
```bash
sbatch --export=ALL,CKPT=ckpt/m2c_intra_cross_attn_recon_..._last.ckpt \
       midi_yinyang/infer_intra_cross_attn_combined.sbatch
```

## Variant #4 — `M2CDuetBlockAttn` (DuetAttn-Block)

The fair-looking fix. Targets the asymmetry in DuetAttn (`b_t` sees
`a_t` but not vice versa) by introducing **2 appended next-frame
query slots** with bidirectional within-frame attention. Three SDPA
passes per block (intra / cross-strict-past / frame-bidirectional)
and two gates per modality (cross + frame). At training, the query
slots predict a randomly-sampled `T_query` frame jointly via CE; at
inference, the query slot mechanism is used to sample frames with
bidirectional mutual conditioning.

**Files**:
- `cp_transformer_m2c_duet_block.py` — model class + training script. Adds `gate_fm`, `gate_fc` per layer; `mask_m_emb`, `mask_c_emb` at the top level.
- `init_pretrained_into_duet_block.py` — warm-start init.
- `train_duet_block.sbatch` — SLURM training launcher.
- `cp_transformer_m2c_duet_block_inference.py` — Option B inference module (custom decode loop using query slots).
- `cp_transformer_m2c_duet_block_combined.py` — folder/mode dispatch wrapper.

**Loss**: `L_AR + λ_query · L_query + 0.01 · L_moe_aux` where:
- `L_AR` = standard CE on the clean shifted stream (same as #2).
- `L_query` = CE on the 2 appended query slots vs the target frame's tokens. `T_query ~ Uniform[1, T_full−1]` per batch.
- Default `λ_query = 1.0`.

**Training status**: in progress. Crashed once at step 12414 on a DDP
"unused parameters" error (fixed by setting
`find_unused_parameters=True`). Continuing.

**Inference**: Option B (query-slot decoding) implemented. At each
generation step `t`: append 2 query slots, run forward with `T_query
= max(t, 1)`, read the last 2 hiddens, decode via `local_sampling`.

**Known limitation in inference**: the query slot's strict-less-than
mask (`frame(q) < T_query`) means at step `t` the query slot does
**not** see the clean position containing `m_{t−1}` / `c_{t−1}` (the
most recent committed frame). This is consistent with training but
means Option B has 1 frame less recent context than a hypothetical
Option A (standard AR clean-stream decoding) would have. Fix
requires changing `<` to `≤` in the mask and retraining.

## Variant #5 — `M2CDuetPrefix`

Architecture-side conditioning baseline. Drops the interleaved layout
in favor of a **drum prefix + nondrum suffix**:

```
[drum_0, …, drum_{T−1},  sos_n, nondrum_0, …, nondrum_{T−2}]
```

Drum block runs **bidirectional** within itself (full context, no
causality). Nondrum block runs **strict causal** within itself and
reads **all drum** via cross attention. Per-modality cross gate
(nondrum side) controls drum-conditioning strength. One-way
drum→nondrum by construction.

**Files**:
- `cp_transformer_m2c_duet_prefix.py` — model + training script.
- `init_pretrained_into_duet_prefix.py` — warm-start init.
- `train_duet_prefix.sbatch` — SLURM training launcher.

**Loss**: `L_CE_nondrum + 0.01 · L_moe_aux`. CE only on the nondrum
positions; drum is pure conditioning, no loss on it.

**Training status**: not started yet.

**Inference**: not implemented. Once trained, will need a dedicated
inference script (single-direction: prompt drum → sample nondrum).
The existing intra-cross-attn inference can't be reused because the
sequence layout and predict-vs-condition semantics differ.

## Shared infrastructure

- **Tokenizer / data layer**: `cp_transformer_m2c_moe.py` (`RoFormerSymbolicTransformer`, `FramedDataset`, `preprocess`).
- **Per-modality projection + RoPE + SimpleMoEFFN helpers**: `cp_transformer_m2c_jointattn.py`.
- **Pretrained backbone**: `ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt`. Same single-stream model used as init source for all variants.
- **Warm-start remap utilities**: `_count_pretrained_layers`, `_map_global_key`, `assert_vocab_matches` in `init_pretrained_into_jointattn.py`.
- **Dataset prep**: `split_drum_nondrum.py` (split MIDIs by `is_drum`), `preprocess_large_midi_dataset.py` (LA-MD CP tokenization).
- **MoE diagnostics**: `moe_routing_monitor.py` (callback, logs expert utilization).
- **Train-sample dumping**: `dump_train_samples.py` (writes first training batch as .mid for sanity-check listening).

## RWC inference pipeline

Unified SLURM job that runs inference for variants #2, #3, #4 on the
RWC prompt folder:

- `infer_all_rwc.sbatch` — single launcher; defaults to
  `input/rwc_test_prompts_split/{drum,nondrum}` and writes per-variant
  subdirectories under `temp/rwc_infer_<JOBID>/`.

Variant #5 not included yet (no inference script).

## Outstanding work

| Item | Priority | Notes |
|---|---|---|
| Train #3 (M2CDuetRehearsal) | high | New rehearsal-style conditioning baseline. |
| Train #5 (M2CDuetPrefix) | high | One-way conditioning baseline. |
| Write inference for #3 | medium | Needs a custom loop maintaining a prefix buffer + suffix buffer per step. Adapt from the duet_block inference template. |
| Write inference for #5 | medium | Single-direction (drum→nondrum); can adapt the jointattn inference machinery. |
| Fix #4's off-by-one query mask + retrain | medium | `<` → `≤` in strict-past-frame check; may improve Option B sample quality materially. |
| Add Option A (AR clean-stream) inference for #4 | low | Useful as a baseline against Option B. ~50 lines. |
| Stage 3 of #4 (proper denoising inference) | low | Multi-step denoise within a frame (currently 1-pass). |
| Refactor shared helpers out of jointattn into a common module | low | Lets us actually retire `cp_transformer_m2c_jointattn.py`. |

## Compute footprint per variant (large config, 12 layers)

All four active variants share the same hidden size / layer count /
MoE expert config, so per-step compute differs only in the attention:

- **#2, #3**: 2 SDPA passes per block (intra + cross). Same compute.
- **#4**: 3 SDPA passes per block (intra + cross + frame); sequence length 2T+2 instead of 2T (negligible). ~50% more attention compute, ~20% slower per step in practice.
- **#5**: 2 SDPA passes per block but with different masks; total compute comparable to #2.

Memory footprint is dominated by MoE FFN activations (the same
across all variants); attention activations are a smaller share, so
the per-variant memory ranking is roughly equal at the
batch_size=4-per-GPU setting we use.
