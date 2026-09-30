# Duet beyond symbolic: architecture and engineering directions

Living notes for the extended project: coordinating pretrained models over
more distant modalities (audio + symbolic first). One section per open
design question. Each section records (1) what the ICASSP Duet does now,
with the flag that controls it, (2) what changes when the streams are
distant, (3) the options, (4) the current position, (5) the experiment
that decides it. The decision log at the end is updated as we go.

Status key: `open` (undecided), `leaning` (default chosen, untested),
`decided` (experiment done).

## 0. What is different when the streams are audio and symbolic

Everything in Duet leans on the two streams sharing a backbone, a
tokenizer, a frame rate and a width. None of those hold for audio.

| Property | melody + chord (now) | audio + symbolic |
|---|---|---|
| backbone | one, copied into both streams | two different models (or one audio LM and one symbolic LM) |
| width / depth | identical (large, 12 global layers) | differ; shared experts need a common width |
| frame | one sixteenth-note frame, up to 4 (program, pitch-dur) pairs + EOS | audio: codec frames at 50-75 Hz with 4-8 RVQ codebooks; one symbolic frame spans 6-10 audio frames, each 4-8 tokens |
| "frame vector" for the draft slot | local encoder bottleneck over ~10 tokens | would compress 30-80 tokens; may need a two-level local encoder |
| corruption | whole-frame mask (all or nothing) | token-level masking is natural and is what audio masked LMs already train on |
| loss scale | two CE terms of similar magnitude | audio CE per frame is 5-10x larger in token count; must be normalised per token per stream |
| conditioning direction | symmetric | asymmetric in practice: symbolic->audio is well posed, audio->symbolic is transcription |
| RoPE time alignment | `--time_rope_aligned 1`, both streams on the same grid | needs a rate-ratio mapping so a symbolic frame and its audio block share a rotary phase |

Two consequences shape every section below. First, "copy the backbone into
two streams" becomes "bring two backbones into one model", so init and
parameter efficiency become real problems. Second, the draft token becomes a
block, so the draft-and-denoise process needs a within-block corruption
design, which is where discrete diffusion actually applies.

## 1. Draft-and-denoise: training and decoding

### Now

- **Slots.** One draft slot per stream per supervised frame
  (`--query_pairs 1`, `--query_block 1`). Total: 2 draft slots. The code
  supports Q pairs (`--query_pairs Q`, each with its own visibility
  window and (k_m, k_c) draw) and Q=-1 (a pair on every frame, variant
  A.9); the paper run used 1.
- **Corruption.** Per slot, a commitment level k ~ Uniform{0..K} with
  K=4 (`--diffusion_K 4`); the slot is fed the ground-truth frame
  embedding with prob (K-k)/K, else the learned mask embedding. Because a
  slot is one frame vector, corruption is all or nothing: k is
  coordination metadata ("is my partner committed?"), not a noise level.
  This is *not* discrete diffusion. The opt-in token-level variant
  (`--token_level_mask 1`, A.4) masks pitch-duration positions inside the
  frame and needs the two fixes in its docstring (trained mask row,
  structural tokens exempt); it was not used in the paper.
- **Self-conditioning.** `--self_cond_prob 0.5`: an unmasked slot is fed
  the model's own no-grad draft instead of ground truth half the time.
- **Loss.** `total = ar_loss + 2.0 * query_loss + 0.01 * moe_aux`
  (`--query_loss_weight 2.0`, `--aux_loss_weight 0.01`,
  `--mel_loss_weight 1.0 --acc_loss_weight 1.0`). The 2.0 compensates for
  the draft pathway receiving 1/T of the gradient (one supervised frame
  per forward vs. all frames for AR). It was set by hand, not swept.
- **Decoding** (`A3_SCHEDULE=ctc_alt`): two forwards per frame. Forward
  1 samples both streams' drafts from the AR heads (`seed_from_ar`);
  the leader (alternating by frame parity) writes its draft into its
  slot at k=0, the follower's slot stays masked at k=K. Forward 2 reads
  out both: follower conditioned on the leader's draft, leader
  re-predicted from its own draft (`A3_CTC_DENOISE_LEADER=1`). The
  parallel schedule (`A3_SCHEDULE=refine`, K+1 rounds) is available on
  the same checkpoint. Setting `refine` with `A3_REFINE_STEPS=0` gives
  plain AR decoding.
- **Known deficit.** Under the uniform k draw a slot predicts with its
  partner masked half the time and learns its stream's marginal
  (coupling 0.053 vs. 0.088 reference before the ctc schedule).
  `--cond_slot_prob` exists to train the commit-then-condition regime
  directly; the paper run kept it at 0 and fixed this at decode time.

### What changes

The audio "frame" is a block of tokens. Whole-block masking throws away
the graded corruption that makes iterative refinement worth its cost;
token-level absorbing corruption inside the block (MaskGIT, MDLM, MAGNeT
on audio codecs) is the standard recipe and the first real use of
discrete diffusion in this line. The RVQ hierarchy adds a second axis:
codebook level is a natural corruption order (coarse committed, fine
masked), and the delay pattern of MusicGen is a third choice.

### Options

**Corruption inside the block**

| option | pros | cons |
|---|---|---|
| a. whole-block mask (as now), 2 slots | proven; cheapest; one k per stream | draft is a single vector; refinement rounds have nothing graded to work on |
| b. token-level absorbing mask, MDLM-style, mask ratio from a schedule | literal discrete diffusion; matches audio masked-LM pretraining; any number of rounds at decode | needs trained [MASK] rows per stream, structural-token exemption, and a per-token time signal; 8x the slot tokens |
| c. codebook-ordered masking (level 1 committed first, levels 2..n masked) | aligns with RVQ semantics; cheap per round | audio only; symbolic side still needs a. or b. |
| d. b. + cond_slot regime (leader block clean, follower block masked at a random ratio) | trains exactly the decode-time conditional; keeps the ctc schedule | drops the both-masked regime unless mixed in |

**Number of draft slots**

- 2 (one block per stream) as now, or Q pairs over Q frames, or a pair on
  every frame (A.9). With blocks, Q>1 costs Q x block length in sequence.
  Position: Q pairs with Q around 4-8 for the symbolic frame count,
  since the draft pathway otherwise starves, and audio blocks make each
  forward expensive enough that supervising one frame per forward is
  wasteful.

**Loss balancing**

- Normalise every CE per token *within* its stream, then weight streams
  (`mel_loss_weight`/`acc_loss_weight` generalised to per-stream
  weights). Otherwise the audio stream's 30-80 tokens per frame dominate.
- The draft-vs-AR weight (2.0 now) should scale with the gradient share:
  weight ~ T / Q so the draft pathway sees a comparable share, then
  sweep {0.5, 1, 2}x around it.
- With token-level masking, weight masked-token CE by 1/(mask ratio)
  (MDLM's ELBO weighting) so low-ratio and high-ratio draws contribute
  equally.
- Uncertainty weighting (Kendall) or GradNorm across the four terms is
  an option if hand tuning fails; keep it as a fallback, not a default.

### Position: `leaning`

Option d. for the audio block (token-level absorbing mask with
per-stream trained [MASK], leader-clean/follower-masked draws mixed with
uniform draws), option a. kept for the symbolic block as a first step so
the symbolic side is unchanged from the paper. Q pairs. Per-stream
per-token loss normalisation with stream weights 1.0 and a draft weight
of T/Q.

### Deciding experiment

Melody + chord first, on the existing data and metrics, so the decision
is made before audio is in the loop: A.4 (token-level) vs. A.3
(whole-frame) at K=4 with the fixes in place, both decoded with ctc_alt
and with `refine` at 4 rounds. Success criterion: coupling and the
listening-test coordination axis; if A.4 does not beat A.3 on symbolic
frames, the audio block still gets token-level masking (its block is too
large for a single vector) but the symbolic block stays whole-frame.

## 2. LoRA or full finetuning

### Now

Full finetuning of everything except `token_type_embeddings`, which is
frozen. Two low-rank hooks exist and were 0 in the paper:
`cross_lora_rank` (low-rank Q/K/V for the cross-stream reads over frozen
intra projections) and `moe_expert_lora_rank` (experts as low-rank deltas
over one shared, optionally frozen, FFN; `moe_freeze_base_ffn`).
`A3_DISABLE_LORA` at decode zeroes either set for ablation.

### What changes

Two backbones instead of one, and the audio one is large (MusicGen-scale
models are 1.5-3.3B). Full finetuning both on 8 x H200 is feasible in
memory but the data is small relative to the audio backbone, so
forgetting is the risk, and "keep the backbones" is the whole premise of
the method.

### Options

| option | pros | cons |
|---|---|---|
| a. full finetune both (as now) | simplest; best fit | forgetting; each experiment retrains billions of params |
| b. freeze both backbones, train only coordination parameters (cross projections, gates, routers, draft embeddings, k tables, [MASK] rows) | preserves backbones exactly; cheap; makes the "coordination layer" the object of study | may under-fit: Duet's experts and attention did move during training |
| c. b. + LoRA on backbone attention/FFN | standard middle ground; forgetting bounded by rank | two more hyper-parameters (rank, which matrices) |
| d. asymmetric: freeze/LoRA the large audio backbone, full-tune the small symbolic one | matches where the data is; audio prior stays intact | asymmetric optimisation dynamics, gates may open one way only |

### Position: `leaning` d.

Full tuning of the symbolic stream (it worked in the paper), LoRA (rank
16-64 on Q/V and FFN) on the audio stream, full training of every new
parameter. Report backbone forgetting as the audio backbone's NLL on a
held-out slice of its own pretraining-like data before and after.

### Deciding experiment

On melody + chord, where the answer is cheap: rerun Duet with the
backbone frozen and (i) only coordination parameters trainable, (ii)
plus `cross_lora_rank 32` and `moe_expert_lora_rank 32`. If (ii) is
within noise of the paper's full-finetune Duet on Table 1 and coupling,
the audio project starts from c./d.; if not, the coordination parameters
alone are not enough and the budget goes to a. for the symbolic stream.

## 3. MoE design and expert initialisation

### Now

- Four experts, top-2, shared across the two streams
  (`--moe_num_experts 4 --moe_topk 2`), one pool per global layer,
  applied over the flat concatenated sequence.
- Init: every expert is a copy of the backbone's FFN (sparse upcycling,
  no perturbation); `init_pretrained_into_duet_block_diffusion.py`.
  Symmetry is broken only by the routers and the data.
- Routers: one gate matrix per stream (`--moe_modality_gates 1`), no
  per-stream bias (`--moe_modality_bias 0`), soft top-2 routing, no hard
  modality partition (`--moe_modality_hard_route 0`). Balance loss
  0.01, computed over all tokens (the `moe_aux_clean_only` option
  restricts it to clean tokens).
- The compute-matched dense arm (one expert of width 6144) was close on
  objective metrics and worse on the listening test's coordination axis.
- `MOE_ROUTING_REPORT.md` has the routing statistics of the paper
  checkpoint: experts do specialise by stream but not fully.

### What changes

There is no single FFN to copy, and the two backbones' widths differ.
A shared expert pool needs either a common width or per-stream
projections into and out of the pool. Whether the pool should be shared
at all is the question the dense/MoE ablation answered only for
same-backbone streams.

### Options

**Where the shared computation lives**

| option | pros | cons |
|---|---|---|
| a. shared pool at a common width, per-stream in/out projections (adapter into the pool) | keeps the Duet block; pool is the only place the streams mix besides attention | projections start random; adds a bottleneck |
| b. no shared FFN: each stream keeps its own FFN, coordination only through gated cross-attention | no width problem; cheapest | gives up the component the listening test credited for coordination |
| c. shared experts only in the top n layers, own FFN below | mixes where representations are most abstract; bounds cost | one more hyper-parameter (n) |

**Expert initialisation given two backbones**

| option | pros | cons |
|---|---|---|
| i. branch-train-mix: pool = {copies of audio FFN, copies of symbolic FFN}, routers learn to mix | each stream starts with its own expert available; cross-use emerges | requires common width (only under a.); experts from the wrong stream are initially useless to the other |
| ii. copies of each stream's own FFN + n_shared fresh experts (DeepSeek-style shared + routed) | dedicated per-stream experts keep the backbone behaviour; shared ones absorb coordination | fresh experts start random and may never be routed to without a bias |
| iii. upcycle with perturbation (noise or split-and-scale) | breaks symmetry at step 0 | perturbation size is a new knob; the paper's unperturbed copies worked |

**Router details worth carrying over or adding**

- Per-stream gate matrices (keep).
- Router z-loss and expert-choice or capacity-free routing: not needed
  at 4 experts; revisit at 8+.
- A hard "own-stream expert always on" path (one always-active expert
  per stream plus top-k over the rest) reproduces the DeepSeek shared
  expert and guarantees the backbone FFN is never routed away from.

### Position: `leaning` a. + ii. in the top half of the layers (c.)

Common pool width = the symbolic backbone's width (smaller), audio
tokens projected down and up; per-stream dedicated expert initialised
from that stream's FFN and always active; 2-4 routed experts initialised
as copies of the symbolic FFN (the smaller, cheaper one) plus one fresh
expert; balance loss over routed experts only.

### Deciding experiment

Same-backbone first: on melody + chord compare (1) the paper's 4 shared
copies, (2) 2 dedicated always-on + 2 routed, (3) top-6-layers-only
pool. Then the first audio run compares a. vs. b. only, since b. is the
fallback if projections into a shared pool do not train.

## 4. One stage or several

### Now

One stage. The copied init (`ckpt/init/*_init.ckpt`) sets the
cross-stream gate bias to -10 so step 0 is the backbone on each stream;
AR loss, draft loss and balance loss are all on from step 0; one-cycle
schedule, 50k steps, global batch 8 for the released Duet. No freezing
schedule and no curriculum on k.

### What changes

With two backbones and random cross projections, turning everything on
at once risks the gates opening onto noise and the draft slots learning
from an uncoordinated model. Staging costs wall-clock but each stage has
its own diagnostic.

### Options

| stage plan | what each stage trains | diagnostic that ends the stage |
|---|---|---|
| A. single stage (as now) | everything | val loss |
| B. two stages: (1) backbones frozen, cross reads + projections + gates, AR loss only; (2) unfreeze/LoRA, add draft slots and draft loss | (1) does coordination emerge through attention alone? (2) does the draft pathway add to it? | (1) gate means and coupling on samples; (2) draft loss and ctc-decoded coupling |
| C. three stages: B. + (3) draft-only finetune, AR head frozen | isolates the denoiser; matches the "denoise pass is optional at inference" story | query loss and refinement gain (K=4 vs. K=0) |
| D. curriculum on k inside one stage: start with both-committed draws, move to uniform | drafts learn from clean partners first | draft loss slope per k bin |

### Position: `leaning` B.

Stage 1 is also the natural place for the alignment experiments (rate
mapping, RoPE phase) since nothing else moves. Stage 3 of C. is kept as
an optional tail if the draft loss is still falling when stage 2 ends.
The "w/o draft tokens" arm of the paper is exactly stage 1's model, so
the ablation comes for free.

### Deciding experiment

Melody + chord: run B. with stage 1 = the existing no-draft trainer
(`cp_transformer_m2c_intra_cross_attn.py`) and stage 2 warm-started
from it into the draft trainer (the init script already zero-inits the k
tables so a warm start reproduces stage 1 at k=K). Compare with the
single-stage paper run at equal total steps.

## 5. Classifier-free guidance

### Now

None. Neither trainer nor generator implements guidance; the only
occurrence in the tree is HuggingFace's generic logits processor in the
transformers fork, which our decode loop does not use.

### Why it is nearly free in Duet

The draft slot is trained at k=0 (partner committed) and k=K (partner
masked) by the uniform draw. Those are exactly the conditional and
unconditional branches guidance needs: in the denoise forward, read the
follower's logits with the leader's slot committed and, in a second
forward (or a doubled batch), with the leader's slot masked, then sample
from `l_cond + w (l_cond - l_uncond)`. No retraining. The coupling
deficit noted in section 1 (slots learned the marginal) is the case
guidance is designed to fix. A second, retraining-free variant is
guidance on the cross-stream gates: scale the gate output by w at the
follower's layers (w=0 is the unconditional stream).

### What changes

Symbolic -> audio is the direction guidance matters most in practice
(MusicGen decodes at guidance 3.0), and the audio prior is strong enough
that the follower will ignore a weak partner without it. The
audio -> symbolic direction probably needs little.

### Options

| option | cost | risk |
|---|---|---|
| a. slot-mask guidance (above), per frame | +1 forward per frame or batch doubling | over-sharpening within the frame lowers diversity; interacts with top-p |
| b. gate-scale guidance | none | the gate is not a clean conditional; effect may be nonlinear in w |
| c. train with condition dropout (leader slot masked with p=0.1) and guide at decode | standard; cleanest branches | the uniform k draw already does this at p=0.5, which may be too much; `--cond_slot_prob` plus a small mask prob is the tunable version |
| d. asymmetric guidance: w>1 for the audio follower only | targets where it is needed | one more decode knob per stream |

### Position: `open`, a. first because it is a decode-only change

### Deciding experiment

On the released Duet checkpoint, add a. to the ctc_alt decoder behind an
env var (`A3_CFG_W`), sweep w in {0, 1, 2, 3} on the 92 held-out songs,
and score coupling, pitch consonance and pitch-class JSD. If coupling
rises without the prompt metrics falling, guidance becomes a default
decode knob and c. is revisited when the audio trainer exists.

## Order of experiments

All on melody + chord unless stated, so decisions land before the audio
pipeline is built. Each is one sbatch chain on the existing wrappers.

1. **CFG at decode** (section 5): no training, one afternoon on the
   released checkpoint.
2. **Frozen-backbone / LoRA Duet** (section 2): two training runs.
3. **A.4 token-level vs. A.3 whole-frame** (section 1): two runs, both
   decoded two ways.
4. **Two-stage training** (section 4): reuses the no-draft checkpoint.
5. **Expert layout** (section 3): three runs.
6. First audio run: symbolic backbone (ours) + a small audio LM, stage 1
   only, to settle the rate mapping and the shared-pool width before any
   draft-slot work.

## Decision log

| date | question | decision | evidence |
|---|---|---|---|
| 2026-09-30 | all five | positions recorded above; nothing decided yet | paper checkpoint and code as of commit f02148a |
