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
| 3 | `M2CDuetRehearsal` | `cp_transformer_m2c_duet_rehearsal.py` | drum prefix `[drum_0..drum_{T-1}]` + standard DuetAttn shifted interleaved suffix (length 3T total) | 2 SDPA passes; prefix bidirectional within, suffix sees all prefix + causal within suffix | CE on entire 2T suffix (drum-side CE collapses fast; useful signal in nondrum CE) | symmetric joint, drum-conditioned |
| 4 | `M2CDuetBlockAttn` | `cp_transformer_m2c_duet_block.py` | interleaved + 2 appended query slots | 3 SDPA passes (intra / cross-strict-past / frame-bidirectional) | AR-CE + query-CE | symmetric |
| 5 | `M2CDuetPrefix` | `cp_transformer_m2c_duet_prefix.py` | `[drum_0, …, drum_{T-1}, sos_n, nondrum_0, …, nondrum_{T-2}]` | 2 SDPA passes; drum-drum bidirectional, nondrum-nondrum causal, nondrum→drum cross | CE on nondrum positions only | one-way drum→nondrum |

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

Loss is standard CE on the entire 2T suffix (drum-side and
nondrum-side). The drum-side CE collapses fast — the model can
trivially copy `drum_k` from the prefix to its corresponding suffix
slot — but that's fine: the useful signal lives in the **nondrum CE**,
where each `nondrum_k` prediction now sees the **entire drum stream**
(past and future) via the prefix, plus causal nondrum past, plus the
suffix's drum past (which itself is conditioned on the prefix).

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
