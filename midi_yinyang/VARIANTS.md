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
| 3 | `M2CIntraCrossAttnRecon` | `cp_transformer_m2c_intra_cross_attn_recon.py` | same as #2 | same as #2 | CE + Brier-MSE on drum logits | symmetric |
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

### DuetAttn-Recon (#3) — loss-shape ablation (not a true conditioning baseline)

Originally framed as a loss-side conditioning baseline: same architecture
as DuetAttn (#2), **adds a Brier-style MSE term on the drum logits** on
top of CE. The intent was that the extra drum supervision would behave
like a "rehearsal" — the model trains as if it had drum context.

**Caveat (discovered post-implementation)**: the Brier MSE term operates
on the *same* drum logits that CE already supervises and uses the *same*
causal past as conditioning. Mathematically, the two terms just push the
softmax toward the one-hot target with different gradient shapes — they
don't introduce any new information flow. A true rehearsal-style
conditioning would require the drum stream to be **architecturally
visible** as context (full drum prefix bidirectional, then nondrum
predicted with that context available), not just better-supervised.

So #3 is best understood as a **loss-shape ablation** on top of #2 (does
the Brier gradient profile do anything CE doesn't?), **not** as a
distinct conditioning baseline. Variant #5 (DuetPrefix) is the
architecture-side conditioning baseline that the original framing of #3
was trying to be a loss-side counterpart to — but the counterpart was
never realized in the loss alone, because conditioning can't be
implemented purely by changing the loss on a model that already
sees the same context.

If a true loss-side conditioning baseline is wanted, see the "future
work" note in `IMPLEMENTATION_REPORT.md` about a hypothetical
M2CDuetRehearsal (#6) that prepends the entire drum stream as a
bidirectional prefix and then runs interleaved AR on the suffix.

### DuetPrefix (#5) — conditioning baseline, architecture-side

Different conditioning route: drum is a **hard prefix** that the
nondrum block reads via full cross attention. Conditioning enters
through the **architecture itself** (mask shape), not the loss. One-way
drum→nondrum by construction.

**Note**: as discussed above, #3 does not actually function as a
conditioning baseline (the Brier MSE acts on the same predictions CE
already supervises, with the same context). The architecture-side
conditioning baseline #5 is what currently stands for "conditioning"
in the lineup. A new variant #6 (DuetRehearsal) would be the
genuine loss-side / hybrid conditioning baseline; see
`IMPLEMENTATION_REPORT.md`.

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
- **#5 vs #4**: is the symmetry fix doing something the
  architecture-side conditioning baseline can't reach?
- **#3 vs #2**: pure ablation — does the Brier MSE gradient profile
  on drum logits do anything CE doesn't? (Loss-shape only; not a
  conditioning test.)

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
