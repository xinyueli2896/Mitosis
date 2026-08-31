"""M2CDuetBlockDiffusion (variant A.3): DuetBlock with discrete-diffusion-style
training at the query slots.

Motivation
----------
Plain DuetBlock (A.2) trains the two query slots only in the "both fully
masked" regime: their inputs are always mask_m_emb / mask_c_emb. At
inference, each query slot predicts its target frame conditionally
independently of the other slot, given the past. This is "equalize by
removing" -- both slots have the same conditioning surface (past only),
neither sees the other's current-frame value. To get mutual conditioning
within a frame you need either:

(a) iterative refinement (block diffusion): K passes per frame, with
    progressively-committed slot inputs, so round k+1 sees round k's
    estimates. The model needs to handle intermediate noise levels at
    the query slots, not just the fully-masked regime.

(b) MaskGIT-style commit-then-condition: round 1 commits one slot's
    prediction, round 2 predicts the other conditioned on the committed
    one. The model needs to handle "one slot committed, one slot masked"
    inputs -- never seen in A.2 training.

This variant trains the model to handle ANY noise combination at the two
query slots. At training, per item, per slot, we sample a noise level
k in {0, 1, ..., K} independently. With prob k/K the slot is fed
mask_*_emb (parent's behaviour); with prob (K-k)/K it is fed the actual
ground-truth frame embedding _encode_frame(target, mod). A learned
k-embedding is added to each slot so the model knows the noise level
(analogous to the timestep embedding in diffusion models).

TERMINOLOGY -- k is a COMMITMENT level, not a noise level. The name
"noise level" is inherited from the diffusion analogy, but because a
slot is a single frame VECTOR (the local-encoder bottleneck), the
corruption here is all-or-nothing masking per slot: no intermediate
corruption state exists, and for a masked slot the target is
statistically independent of k. What k actually carries is
COORDINATION metadata: (a) each slot reads the partner's k through the
frame pass, so k is the signal distinguishing "the partner is
committed -- harmonize with it" (k=0) from "the partner is guessing --
negotiate" (k=K); (b) at inference k indexes the refinement round, the
hook for round-aware behaviour (drafts improve as k falls); (c) the
(k_m, k_c) configuration is how a decode schedule -- parallel
diffusion, MaskGIT commit-then-condition -- is communicated to one
checkpoint. Prose and figures should say "commitment level"; the
k_emb_* parameter names stay (renaming them would orphan every
existing checkpoint). The opt-in --token_level_mask variant (codename
A.4, run-dir tag 'tk') restores genuinely graded corruption -- per-token
masking inside the frame -- and makes the diffusion reading literal;
see the A.4 note in __init__.

A.4 FAILURE ANALYSIS (first run, tag 'mgtk')
--------------------------------------------
The first A.4 run decoded to dense, messy music -- worse than A.2, and
worse in a specific way (note density inflated, phrase boundaries gone).
Per-token absorbing corruption is the standard recipe (D3PM, MaskGIT,
MDLM), so the variant did not fail because the idea is wrong; it failed
because that recipe has two prerequisites this implementation violated.

  FIX 1 -- the [MASK] embedding must be trained, not borrowed. In the
  literature [MASK] is a first-class vocabulary entry whose row is
  learned from initialisation. A.4 instead reused a dead id in the
  instrument-padding range (3327), so its embedding row arrived from the
  pretrained checkpoint untouched by any gradient -- an arbitrary vector
  pushed through a local encoder that had never seen it. Every corrupted
  frame was therefore encoded through an out-of-distribution input.
  Fixed by _init_frame_mask_row(): set the row to the mean of the real
  token embeddings once, on_train_start, after the warm-start load.

  FIX 2 -- structural tokens must be exempt from corruption. A cp frame
  is (program, pitch-dur) pairs terminated by EOS: even positions are
  programs and the EOS, odd positions are pitch-durs. The first run
  masked all of them, so EOS itself could be masked -- and a frame whose
  terminator is unknown reads as unfinished, which biases the next
  refinement round toward more notes and compounds over rounds. That is
  the density inflation that was actually heard. Fixed in
  _token_level_slot(): only odd (pitch-dur) positions are maskable, so
  corruption destroys WHICH notes while preserving HOW MANY and on which
  instrument -- also the musically meaningful partial state.

Both endpoints remain exact under the fixes (k=0 clean, k=K the learned
whole-frame mask embedding, silent frames included), so a warm start
from an A.2 checkpoint still reproduces A.2 at k=K.

QUERY-PAIR COUNT (Q, --query_pairs)
-----------------------------------
Training supervises the query slots on ONE frame per forward: T_query is
a single index, shared by the whole batch because it defines the [L, L]
attention mask. The AR loss meanwhile scores all 2*T_full positions. So
the frame pass -- the same-instant symmetric conditioning that is the
only mechanism the decode loop actually uses -- receives roughly 1/T of
the gradient, while the inherited AR pathway receives all of it.

That is not how the literature does it. D3PM, MDLM and MaskGIT draw one
noise level per sample but reconstruct EVERY corrupted position; BERT
masks 15% of positions rather than one, for exactly this reason.

--query_pairs Q appends Q query PAIRS instead of one, for Q distinct
frames, each with its own visibility window (frames < T_j), its own
(k_m, k_c) draw, and its own loss. Pairs are blind to each other -- a
slot may read its own partner's draft, never another pair's, which
inference could not supply. The clean stream is untouched and still
blind to every slot.

Cost: L goes from 2*T_full + 2 to 2*T_full + 2Q. At TRAIN_LENGTH=384
that is 770 -> 784 for Q=8, i.e. +2% sequence and +3.7% attention for
8x the query gradient and 8x the coverage of the (k_m, k_c) grid. No
rotary extrapolation: each pair takes the rotary phase of its OWN
frame, so no position is visited that Q=1 did not already visit.

Q is training-only. Inference decodes one frame at a time regardless,
so the parameters, the checkpoint and the decode path are unchanged,
and validation keeps Q=1 so val_loss stays comparable across settings.

Both schedules (parallel diffusion, MaskGIT) become valid inference
strategies on the same trained checkpoint -- the user can experiment
with either without retraining.

Architecture
------------
Identical to M2CDuetBlockAttn. The mask_frame attention pass already
lets the two query slots see each other, which is the channel through
which mutual conditioning happens at inference. The new k_emb_* tables
are the only extra parameters; they are zero-initialised so a warm-start
from an A.2 ckpt behaves identically at k_m = k_c = K (fully masked).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_transformer_m2c_moe import (
    RoFormerSymbolicTransformer, FramedDataset, TRAIN_LENGTH, MAX_STEPS,
)
from cp_transformer_m2c_duet_block import M2CDuetBlockAttn, normalize_T_query
from cp_transformer_m2c_jointattn import _rope_freqs
from tasks import get_task, TASKS


class M2CDuetBlockDiffusion(M2CDuetBlockAttn):
    """DuetBlock with discrete-diffusion training at the query slots.

    See module docstring. Only forward() and loss() are overridden; the
    layer stack, gates, mask construction, AR loss, and inference-time
    shape are inherited unchanged.
    """

    def __init__(self, *args, diffusion_K=4, slot_rope_aligned=True,
                 time_rope_aligned=False, self_cond_prob=0.5,
                 token_level_mask=False, mask_revealed_query_loss=False,
                 query_pairs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_K = int(diffusion_K)
        # --- A.4: token-level masking inside the frame ----------------
        # Restores genuinely graded corruption: at commitment level k,
        # each TOKEN of the target frame is masked independently with
        # prob k/K and the local encoder embeds the partially-masked
        # frame -- intermediate k become real intermediate states
        # ("chord root known, upper voices open") instead of a biased
        # coin between fully-masked and fully-clean. Endpoints preserved
        # exactly: an all-masked draw falls back to mask_*_emb and k=0
        # is the clean frame, so a warm start from an A.2/A.3 ckpt keeps
        # its trained behaviour at both ends of the trajectory.
        # The MASK id needs NO vocab change: with_velocity=False pads
        # the program range to 256 with only 128 real programs and caps
        # pitch-dur ids at 128*25-1 = 3199, so ids 3200..3327 are
        # unreachable in real data AND excluded by local_sampling's
        # valid-token masks. We take the last one.
        self.register_buffer(
            'token_level_mask_flag',
            torch.tensor(1 if token_level_mask else 0, dtype=torch.long),
        )
        if token_level_mask:
            if self.tokenizer.with_velocity:
                raise ValueError(
                    'token_level_mask needs a free token id, and the '
                    'with_velocity vocabulary has none (128 instruments '
                    'x 16 velocities fills the padded range).'
                )
            self.frame_mask_token = self.tokenizer.n_normal_tokens - 1
        # A.4 FIX 1 (see the FAILURE ANALYSIS note below): the MASK id's
        # embedding row must not start as an untrained random vector.
        # Set once, AFTER weights load (on_train_start) -- doing it here
        # would be overwritten by the warm-start state_dict, which
        # carries the whole local_embedding table.
        self.register_buffer(
            'frame_mask_row_init_flag',
            torch.tensor(0, dtype=torch.long),
        )
        # Score the query loss only where the slot did NOT hand the
        # model its own target -- see _query_loss_keep_mask. Stored as a
        # buffer so the objective a checkpoint was trained under travels
        # inside it. NOTE: validation pins k=K, where nothing is
        # revealed and the keep mask degenerates to the non-pad mask,
        # so val_loss stays computed identically across the flag --
        # only the TRAINING objective differs.
        self.register_buffer(
            'mask_revealed_query_loss_flag',
            torch.tensor(1 if mask_revealed_query_loss else 0,
                         dtype=torch.long),
        )
        # Per-modality commitment-level embeddings (the diffusion
        # "timestep" analogue -- see the TERMINOLOGY note in the module
        # docstring), indexed by k in {0, ..., K}. Zero-init: a
        # warmstart from an A.2 ckpt then reproduces parent behaviour
        # at k = K (fully masked) on step 0.
        self.k_emb_m = nn.Embedding(self.diffusion_K + 1, self.hidden_size)
        self.k_emb_c = nn.Embedding(self.diffusion_K + 1, self.hidden_size)
        with torch.no_grad():
            self.k_emb_m.weight.zero_()
            self.k_emb_c.weight.zero_()

        # --- v1.1 training-scheme flags -------------------------------
        # slot_rope_aligned: apply RoPE to the two query slots at rotary
        # index 2*T_query+2 / 2*T_query+3 -- the phase they naturally
        # occupy at inference (right after the committed pairs of frame
        # T_query) -- instead of their physical end-of-sequence position
        # 2*T_full{, +1}. v1.0 ckpts trained the slots at a CONSTANT
        # phase ~2*T_full, which mismatches inference for every t <
        # T_full and required a zero-padding workaround at decode time.
        # Stored as a buffer so the scheme travels inside the ckpt and
        # inference can auto-detect it (legacy ckpts lack the key).
        self.register_buffer(
            'slot_rope_aligned_flag',
            torch.tensor(1 if slot_rope_aligned else 0, dtype=torch.long),
        )
        # --- v1.2 training-scheme flag --------------------------------
        # time_rope_aligned: rotary index = physical index // 2 for the
        # whole sequence, so m_t and c_t share rotary position t and the
        # SOS pair sits at 0. Musical distance == rotary distance again
        # (the legacy parity scheme DOUBLES every musical distance
        # relative to the single-stream pretrain and pushes a 384-frame
        # sample to rotary 0..767, half of it untrained in the warm
        # start) -- the candidate fix for the duet family's long-term-
        # structure deficit, which A.2 exhibits despite 43k steps and
        # full stream survival. Subsumes v1.1: the slot remap to
        # 2*T_query+2/+3 then halves to T_query+1 for BOTH slots --
        # exactly the rotary phase frame T_query's content occupies in
        # the SOS-shifted clean stream, at any t, so decode needs no
        # padding. Stream identity is carried by content (mask_*_emb,
        # k_emb_*, token types), not position parity. Stored as a buffer
        # so the scheme travels inside the ckpt and inference auto-
        # detects it (legacy ckpts lack the key). Same D.1 scheme as
        # M2CIntraCrossAttn's time_rope_aligned_flag.
        self.register_buffer(
            'time_rope_aligned_flag',
            torch.tensor(1 if time_rope_aligned else 0, dtype=torch.long),
        )
        # self_cond_prob: per-item probability that an UNMASKED slot is
        # fed the model's own (no-grad) prediction of the target frame
        # instead of the ground-truth embedding. Closes the exposure gap
        # between training (gt-or-mask) and inference (self-samples fed
        # back across refinement rounds).
        self.self_cond_prob = float(self_cond_prob)
        # QUERY-PAIR COUNT (Q). How many DISTINCT frames each training
        # forward supervises -- see the note in the module docstring.
        # Training-only: inference always decodes one frame at a time,
        # so Q leaves the parameters, the checkpoint and the decode path
        # untouched, and Q=1 is bit-for-bit the historical behaviour.
        self.query_pairs = max(int(query_pairs), 1)

    @property
    def slot_rope_aligned(self):
        return bool(self.slot_rope_aligned_flag.item())

    @property
    def time_rope_aligned(self):
        return bool(self.time_rope_aligned_flag.item())

    @property
    def token_level_mask(self):
        return bool(self.token_level_mask_flag.item())

    @property
    def mask_revealed_query_loss(self):
        return bool(self.mask_revealed_query_loss_flag.item())

    def on_load_checkpoint(self, checkpoint):
        """Let a checkpoint that predates a scheme buffer still resume.

        Every training-scheme flag here is a registered buffer, so it
        travels inside the ckpt -- but that also means a ckpt written
        before a flag existed is MISSING that key, and Lightning's
        resume path loads the state_dict strictly. Fill any absent
        buffer with this run's own value, which is the right default:
        the flag then comes from the CLI, exactly as it would on a cold
        start. (frame_mask_row_init_flag=0 in particular means a legacy
        A.4 ckpt gets its never-initialised mask row fixed on resume.)
        """
        sd = checkpoint.get('state_dict')
        if isinstance(sd, dict):
            for name, buf in self.named_buffers():
                if name not in sd:
                    sd[name] = buf.detach().clone()
                    print(f'[compat] checkpoint predates buffer {name!r}; '
                          f'filling in with this run\'s value '
                          f'{buf.tolist()}')
        super().on_load_checkpoint(checkpoint)

    def _init_frame_mask_row(self):
        """A.4 FIX 1: give the MASK id a trained-model-like embedding.

        Standard absorbing-state discrete diffusion (D3PM, MaskGIT,
        MDLM) carries [MASK] as a first-class vocabulary entry whose
        embedding is learned from the start. A.4 instead repurposed a
        dead id in the instrument-padding range, so its row arrived at
        training as the pretrained checkpoint left it: never updated by
        any real data, i.e. effectively random -- a stray vector shoved
        into a PRETRAINED local encoder that had never seen it. That is
        the leading explanation for the first A.4 run's collapse.

        Fix: initialise the row to the MEAN of the real token
        embeddings (programs 0..127 and pitch-durs 128..3199), so the
        mask starts as a neutral, in-distribution "average token"
        rather than noise, and learns away from there. Runs once, from
        on_train_start (after the warm-start state_dict has loaded --
        doing it in __init__ would be overwritten), and records itself
        in a buffer so a resumed run does not re-initialise a row that
        has since trained.
        """
        if not self.token_level_mask:
            return
        if bool(self.frame_mask_row_init_flag.item()):
            return
        with torch.no_grad():
            emb = self.local_embedding.weight
            real = emb[:128 * 25]        # programs + pitch-durs, no padding
            emb[self.frame_mask_token] = real.mean(dim=0)
            self.frame_mask_row_init_flag.fill_(1)
        print(f'[A.4] frame mask row {self.frame_mask_token} initialised to '
              f'the mean of {real.shape[0]} real token embeddings')

    def on_train_start(self):
        hook = getattr(super(), 'on_train_start', None)
        if callable(hook):
            hook()
        self._init_frame_mask_row()

    def _token_level_slot(self, content_tokens, sc_mask, sc_toks, k_t, mod):
        """A.4 slot construction: per-token absorbing corruption.

        content_tokens: [B, S] the target frame's tokens (or, where
            sc_mask is set, the self-conditioning draft's tokens).
        k_t: LongTensor[B] commitment levels.

        A.4 FIX 2: only PITCH-DURATION tokens are maskable. A cp frame
        is (program, pitch-dur) pairs terminated by EOS, so programs sit
        at even positions and EOS at the even position after the last
        pair. The first A.4 run masked those too, and a frame whose EOS
        is masked reads as "not finished" -- which biases the next
        refinement round toward more notes and compounds across rounds.
        That is the density inflation heard on that run. Masking only
        the odd (pitch-dur) positions preserves each frame's INSTRUMENT
        and LENGTH while corrupting its CONTENT, which is both the
        musically meaningful partial state ("how many notes is settled,
        which notes is not") and the standard practice of leaving
        structural tokens out of the corruption process.

        Endpoints stay exact: k=K masks every pitch-dur token and falls
        back to the learned whole-frame mask_*_emb (silent frames
        included, so a silent frame cannot leak its silence at k=K);
        k=0 masks nothing and encodes the clean frame.

        Returns (slot [B, 1, H] WITHOUT the k-embedding -- the caller
        adds it -- and revealed [B, S] bool, the positions whose GROUND
        TRUTH was handed to the model inside the slot; see
        _query_loss_keep_mask for why the loss must drop those).
        """
        B, S = content_tokens.shape
        is_sc = (torch.zeros(B, dtype=torch.bool, device=content_tokens.device)
                 if sc_mask is None else sc_mask)
        if sc_mask is not None and sc_toks is not None:
            content_tokens = torch.where(
                sc_mask.view(B, 1), sc_toks, content_tokens)
        corrupted, fully, drawn = self._corrupt_frame_tokens(
            content_tokens, k_t)
        enc = self._encode_frame(corrupted, mod)               # [B, 1, H]
        mask_emb = (self.mask_m_emb if mod == 0 else self.mask_c_emb)
        mask_emb = mask_emb.view(1, 1, -1).expand(B, 1, -1).to(enc.dtype)
        slot = torch.where(fully.view(B, 1, 1), mask_emb, enc)
        # A token is REVEALED when it survived the draw, the slot was
        # not replaced wholesale by the mask embedding, and the content
        # is ground truth rather than a self-conditioning draft (a draft
        # token may be wrong, so predicting it is not free).
        revealed = (~drawn) & (~fully).view(B, 1) & (~is_sc).view(B, 1)
        return slot, revealed.expand(B, S)

    def _corrupt_frame_tokens(self, content_tokens, k_t):
        """Draw the per-token absorbing corruption for one slot.

        Split out of _token_level_slot so the audit can exercise the
        real draw instead of a copy of it.

        Returns (corrupted [B, S], fully [B] bool, drawn [B, S] bool)
        where `fully` marks the items whose slot must fall back to the
        whole-frame mask embedding rather than the encoding of
        `corrupted`, and `drawn` marks the individual tokens that were
        replaced by the mask id.
        """
        B, S = content_tokens.shape
        device = content_tokens.device
        denom = max(self.diffusion_K, 1)
        p = (k_t.float() / denom).view(B, 1)
        # Odd positions are pitch-dur; even positions carry programs and
        # the terminating EOS and are never corrupted (FIX 2).
        is_pd = (torch.arange(S, device=device) % 2 == 1).view(1, S)
        maskable = is_pd & (content_tokens != self.tokenizer.pad_token)
        drawn = (torch.rand(B, S, device=device) < p) & maskable
        corrupted = torch.where(
            drawn,
            torch.full_like(content_tokens, self.frame_mask_token),
            content_tokens,
        )
        # Fully-unknown fallback -> the learned whole-frame mask
        # embedding, in three cases:
        #   k = K            endpoint, forced, so it is exact for EVERY
        #                    frame including silent ones;
        #   every pd drawn   the frame carries no content any more;
        #   silent frame     a frame with no pitch-dur token at all has
        #                    nothing to corrupt, so encoding it cleanly
        #                    would reveal its silence for free at every
        #                    k < K. Corrupt it all-or-nothing with the
        #                    same probability p instead, which keeps the
        #                    reveal rate monotone in k and both
        #                    endpoints exact (p=0 at k=0, p=1 at k=K).
        n_maskable = maskable.sum(dim=1)
        all_drawn = (n_maskable > 0) & (drawn.sum(dim=1) == n_maskable)
        silent_drawn = (n_maskable == 0) & (
            torch.rand(B, device=device) < p.view(B))
        fully = (k_t == self.diffusion_K) | all_drawn | silent_drawn
        return corrupted, fully, drawn

    def _query_loss_keep_mask(self, non_pad_q):
        """Which query-slot positions the query loss may score.

        The query slot is BOTH the model's conditioning input for the
        target frame AND the thing whose target it predicts. Wherever
        the slot carries un-corrupted ground truth, predicting that
        target is a free copy: at k=0 the whole frame is handed over,
        and under A.4 every token that survived the draw is handed over
        individually.

        Scoring those positions is not just a diluted average. The copy
        path and the "infer it from the partner's draft" path compete
        for the same gradient, and only the second one exists at
        inference -- where the slot holds the model's own draft, never
        the answer. D3PM, MDLM and MaskGIT all score the denoising loss
        on corrupted positions only, for exactly this reason.

        So: drop the revealed positions. Self-conditioned items are
        NOT dropped -- their slot holds a model draft that may be wrong,
        so correcting it is real signal, and arguably the most valuable
        signal in the objective.

        We take the plain mean over the kept positions (MaskGIT) rather
        than the 1/p importance weighting of the MDLM/D3PM ELBO: we want
        a training signal, not a likelihood bound, and the reweighting
        adds variance at small k for no benefit here.

        Off by default. Validation is unaffected either way -- eval
        pins k=K, nothing is revealed there, and the keep mask
        degenerates to the non-pad mask, so val_loss stays directly
        comparable across the flag. What the flag changes is the
        TRAINING objective, so for the E6 arm comparison turn it on for
        a WHOLE arm-set (e.g. all four arms) or none -- otherwise arms
        differ by more than their router.
        """
        if not self.mask_revealed_query_loss:
            return non_pad_q
        revealed = getattr(self, '_last_query_revealed', None)
        if revealed is None:
            return non_pad_q
        return non_pad_q * (~revealed).to(non_pad_q.dtype)

    def _run_global_stack(self, h, T_query):
        """Override: slot-aligned (v1.1) / time-aligned (v1.2) RoPE.

        v1.1 (slot_rope_aligned): clean positions keep rotary index ==
        physical index (0..L-3); the two slots get index 2*T_query+2 and
        2*T_query+3, matching where inference naturally places them
        after the committed pairs of frame T_query.

        v1.2 (time_rope_aligned): the same position vector is then
        HALVED (// 2), so m_t and c_t share rotary position t and both
        slots land on T_query+1 -- the rotary phase frame T_query's
        content occupies in the SOS-shifted clean stream. Musical
        distance == rotary distance; within-stream geometry matches the
        single-stream pretrain exactly.

        Legacy scheme (v1.0 ckpts) falls through to the parent
        implementation (contiguous 0..L-1).

        Note the slots' rotary index may coincide with clean positions
        holding frame T_query{+1}'s content at training time. Duplicate
        rotary phases are benign: attention stays well-defined, the
        slots never attend those rows (frame >= T_query is masked for
        slot queries), and clean rows never attend the slots.
        """
        if not (self.slot_rope_aligned or self.time_rope_aligned):
            return super()._run_global_stack(h, T_query)
        B, L, H = h.shape
        tq = normalize_T_query(T_query)
        clean_len = L - 2 * len(tq)
        head_dim = H // self.num_attention_heads
        positions = torch.arange(L, device=h.device)
        for j, t_j in enumerate(tq):
            positions[clean_len + 2 * j] = 2 * t_j + 2
            positions[clean_len + 2 * j + 1] = 2 * t_j + 3
        if self.time_rope_aligned:
            positions = torch.div(positions, 2, rounding_mode='floor')
        max_pos = int(positions.max().item()) + 1
        cos_b, sin_b = _rope_freqs(max_pos, head_dim,
                                    device=h.device, dtype=h.dtype)
        cos = cos_b[:, :, positions]
        sin = sin_b[:, :, positions]
        total_aux = torch.zeros((), device=h.device, dtype=h.dtype)
        for layer in self.global_layers:
            h, aux = layer(h, T_query, cos, sin, clean_len)
            total_aux = total_aux + aux
        return h, total_aux / max(len(self.global_layers), 1)

    # ------------------------------------------------------------------
    # forward: same as parent except the query-slot inputs.
    # ------------------------------------------------------------------
    def forward(self, x, T_query=None, k_m=None, k_c=None,
                sc_mask_m=None, sc_emb_m=None,
                sc_mask_c=None, sc_emb_c=None,
                sc_toks_m=None, sc_toks_c=None):
        """x: [B, 2*T_full, subseq_len] interleaved ground-truth sequence.

        T_query (int, optional): frame the query slots predict.
        k_m, k_c (int OR LongTensor[B], optional): noise level per slot.
            k = K (default): the slot is masked (parent behaviour).
            k = 0: the slot is the ground-truth frame embedding.
            0 < k < K: per-item Bernoulli mask with prob k/K.
        sc_mask_m / sc_emb_m (optional): self-conditioning override for
            the m slot. sc_mask_m: BoolTensor[B]; sc_emb_m: [B, 1, H].
            Items where the mask is True use sc_emb (a model-generated
            frame embedding) instead of the ground-truth embedding as
            the slot's unmasked content. Only affects the Bernoulli
            "content" branch; masked items still get mask_*_emb.
            Same for the c slot.

        Returns the same triple as parent: (ar_logits, query_logits, aux_loss).
        """
        batch_size, seq_len, subseq_len = x.shape
        assert seq_len % 2 == 0
        T_full = seq_len // 2

        if T_query is None:
            T_query = T_full - 1
        tq = normalize_T_query(T_query)
        n_pairs = len(tq)
        for t_j in tq:
            assert 1 <= t_j < T_full, (
                f'T_query entry {t_j} out of valid range [1, {T_full})'
            )
        assert len(set(tq)) == n_pairs, (
            f'T_query must list DISTINCT frames, got {tq}'
        )

        K = self.diffusion_K
        k_m_t = self._coerce_k_pairs(k_m, batch_size, n_pairs, x.device)
        k_c_t = self._coerce_k_pairs(k_c, batch_size, n_pairs, x.device)
        sc_mask_m = self._coerce_sc_mask(sc_mask_m, batch_size, n_pairs)
        sc_mask_c = self._coerce_sc_mask(sc_mask_c, batch_size, n_pairs)

        # Local encode + token type ids (identical to parent).
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2).long()
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, seq_len, subseq_len
        )
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, seq_len, 1
        )
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)

        h, emb = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)
        H = h.shape[-1]

        # Standard shift for clean stream (identical to parent).
        sos = self._assemble_sos(batch_size, h.device, h.dtype)
        h_clean = torch.cat([sos, h[:, :-2]], dim=1)   # [B, 2T_full, H]

        # --- query-slot construction (the new part) ---
        # One (m, c) pair per entry of tq, built in pair order so that
        # Q=1 consumes the RNG in exactly the historical order and
        # reproduces the old behaviour bit-for-bit.
        slots, revealed = [], []
        mask_m_expand = self.mask_m_emb.view(1, 1, -1).expand(batch_size, 1, -1)
        mask_c_expand = self.mask_c_emb.view(1, 1, -1).expand(batch_size, 1, -1)
        denom = max(K, 1)
        for j, t_j in enumerate(tq):
            sc_m_j = None if sc_mask_m is None else sc_mask_m[:, j]
            sc_c_j = None if sc_mask_c is None else sc_mask_c[:, j]
            if self.token_level_mask:
                # A.4: per-token absorbing corruption of the target
                # frame, locally encoded -- intermediate k are genuinely
                # partial frames. Self-conditioning at the TOKEN level.
                slot_m, rev_m = self._token_level_slot(
                    x[:, 2 * t_j], sc_m_j,
                    self._sc_tok_slice(sc_toks_m, j), k_m_t[:, j], 0,
                )
                slot_c, rev_c = self._token_level_slot(
                    x[:, 2 * t_j + 1], sc_c_j,
                    self._sc_tok_slice(sc_toks_c, j), k_c_t[:, j], 1,
                )
                slot_m = slot_m.to(h.dtype)
                slot_c = slot_c.to(h.dtype)
            else:
                # Ground-truth frame embeddings at t_j, with optional
                # self-conditioning override (model-generated frame
                # embeddings replacing gt for the flagged items).
                gt_m = h[:, 2 * t_j:2 * t_j + 1]           # [B, 1, H]
                gt_c = h[:, 2 * t_j + 1:2 * t_j + 2]       # [B, 1, H]
                if sc_m_j is not None:
                    gt_m = torch.where(
                        sc_m_j.view(batch_size, 1, 1),
                        self._sc_emb_slice(sc_emb_m, j).to(dtype=gt_m.dtype),
                        gt_m)
                if sc_c_j is not None:
                    gt_c = torch.where(
                        sc_c_j.view(batch_size, 1, 1),
                        self._sc_emb_slice(sc_emb_c, j).to(dtype=gt_c.dtype),
                        gt_c)

                # Per-item Bernoulli mask draws with prob k[i] / K (=0
                # if K==0). Using max(K, 1) is purely a divide-by-zero
                # guard; K==0 would mean "never mask," degenerate but
                # well-defined.
                u_m = torch.rand(batch_size, device=h.device)
                u_c = torch.rand(batch_size, device=h.device)
                is_masked_m = (u_m < (k_m_t[:, j].float() / denom)).to(h.dtype)
                is_masked_c = (u_c < (k_c_t[:, j].float() / denom)).to(h.dtype)
                # [B] -> [B, 1, 1] for broadcasting.
                is_masked_m = is_masked_m.view(batch_size, 1, 1)
                is_masked_c = is_masked_c.view(batch_size, 1, 1)

                slot_m = is_masked_m * mask_m_expand \
                    + (1.0 - is_masked_m) * gt_m
                slot_c = is_masked_c * mask_c_expand \
                    + (1.0 - is_masked_c) * gt_c

                # Frame-level corruption is all-or-nothing, so an
                # unmasked slot reveals the WHOLE target frame -- unless
                # it was overridden by a self-conditioning draft, which
                # may be wrong. [B] -> [B, S].
                unmasked_m = is_masked_m.view(batch_size) == 0
                unmasked_c = is_masked_c.view(batch_size) == 0
                if sc_m_j is not None:
                    unmasked_m = unmasked_m & ~sc_m_j
                if sc_c_j is not None:
                    unmasked_c = unmasked_c & ~sc_c_j
                rev_m = unmasked_m.view(batch_size, 1).expand(-1, subseq_len)
                rev_c = unmasked_c.view(batch_size, 1).expand(-1, subseq_len)

            # Add per-item k-embeddings -- the commitment tag. Crucial
            # for iterative refinement at inference, where the same slot
            # input can mean very different things depending on where in
            # the K-step trajectory we are, and it is what the partner
            # slot reads (via the frame pass) to tell a committed frame
            # from a tentative draft.
            slot_m = slot_m + self.k_emb_m(
                k_m_t[:, j]).view(batch_size, 1, -1).to(h.dtype)
            slot_c = slot_c + self.k_emb_c(
                k_c_t[:, j]).view(batch_size, 1, -1).to(h.dtype)
            slots.extend([slot_m, slot_c])
            revealed.extend([rev_m, rev_c])

        # Stash for the loss. Set on EVERY forward, so the no-grad
        # self-conditioning forward's value is overwritten by the real
        # one that follows it -- loss() reads it after that second call.
        self._last_query_revealed = torch.stack(
            revealed, dim=1)                              # [B, 2Q, S]

        h_full = torch.cat([h_clean] + slots, dim=1)
        # h_full: [B, 2*T_full + 2*Q, H]

        h_global, aux_loss = self._run_global_stack(h_full, T_query=tq)

        # Split outputs (identical to parent at Q=1).
        h_clean_global = h_global[:, :seq_len]
        h_query_global = h_global[:, seq_len:]             # [B, 2Q, H]

        ar_logits = self.local_decode(h_clean_global, emb)

        emb_reshape = emb.view(batch_size, seq_len, subseq_len, -1)
        emb_query = torch.cat(
            [emb_reshape[:, 2 * t_j:2 * t_j + 2] for t_j in tq], dim=1,
        )                                                  # [B, 2Q, S, D]
        emb_query_flat = emb_query.reshape(
            batch_size * 2 * n_pairs, subseq_len, -1)
        query_logits = self.local_decode(h_query_global, emb_query_flat)

        return ar_logits, query_logits, aux_loss

    def _coerce_k(self, k, batch_size, device):
        """Accept int / None / LongTensor[B] and return LongTensor[B]."""
        K = self.diffusion_K
        if k is None:
            # Default: fully masked (parent behaviour). Useful for
            # warmstart sanity and for inference at the first round.
            return torch.full((batch_size,), K, device=device, dtype=torch.long)
        if isinstance(k, int):
            return torch.full((batch_size,), int(k), device=device,
                              dtype=torch.long)
        k = k.to(device=device, dtype=torch.long)
        if k.dim() == 0:
            return k.view(1).expand(batch_size).clone()
        assert k.shape == (batch_size,), (
            f'k shape {tuple(k.shape)} != ({batch_size},)'
        )
        return k

    def _coerce_k_pairs(self, k, batch_size, n_pairs, device):
        """Accept int / None / LongTensor[B] / LongTensor[B, Q].

        Returns LongTensor[B, Q]. A [B] tensor (the historical shape,
        and what inference passes) is broadcast to every query pair.
        """
        if k is not None and torch.is_tensor(k) and k.dim() == 2:
            k = k.to(device=device, dtype=torch.long)
            assert k.shape == (batch_size, n_pairs), (
                f'k shape {tuple(k.shape)} != ({batch_size}, {n_pairs})'
            )
            return k
        return self._coerce_k(k, batch_size, device).view(
            batch_size, 1).expand(batch_size, n_pairs)

    @staticmethod
    def _coerce_sc_mask(sc_mask, batch_size, n_pairs):
        """None / BoolTensor[B] / BoolTensor[B, Q] -> None or [B, Q]."""
        if sc_mask is None:
            return None
        if sc_mask.dim() == 1:
            return sc_mask.view(batch_size, 1).expand(batch_size, n_pairs)
        assert sc_mask.shape == (batch_size, n_pairs), (
            f'sc_mask shape {tuple(sc_mask.shape)} != '
            f'({batch_size}, {n_pairs})'
        )
        return sc_mask

    @staticmethod
    def _sc_tok_slice(t, j):
        """Query pair j's draft TOKENS -> [B, S].

        Accepts the historical un-paired [B, S] and the paired
        [B, Q, S]. Kept separate from the embedding slicer because
        [B, 1, S] and [B, 1, H] are indistinguishable by shape alone.
        """
        return t if (t is None or t.dim() == 2) else t[:, j]

    @staticmethod
    def _sc_emb_slice(t, j):
        """Query pair j's draft EMBEDDING -> [B, 1, H].

        Accepts the historical un-paired [B, 1, H] and the paired
        [B, Q, 1, H].
        """
        return t if (t is None or t.dim() == 3) else t[:, j]

    # ------------------------------------------------------------------
    # loss: sample k_m, k_c per item per batch and call forward.
    # ------------------------------------------------------------------
    def loss(self, x_mel, x_acc, batch_pitch_shift):
        # Preprocess + interleave (identical to parent).
        x_mel, x_acc = self.preprocess(x_mel, batch_pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape

        stacked = torch.stack([x_mel, x_acc], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)
        T_full = seq_len
        full_seq_len = seq_len * 2

        # Sample the query frames. Q = query_pairs distinct frames per
        # forward (see the QUERY-PAIR COUNT note in the module
        # docstring); Q=1 reproduces the historical single-frame draw.
        Q = min(max(int(self.query_pairs), 1), T_full - 1)
        if self.training:
            if Q == 1:
                tq = (int(torch.randint(
                    low=1, high=T_full, size=(1,), device=x.device,
                ).item()),)
            else:
                # Distinct frames, sorted so the run-to-run layout is
                # deterministic given the draw.
                perm = torch.randperm(T_full - 1, device=x.device)[:Q] + 1
                tq = tuple(sorted(int(t) for t in perm.tolist()))
        else:
            # Eval keeps the historical single frame, so val_loss stays
            # comparable across Q.
            tq = (T_full - 1,)
        n_pairs = len(tq)
        T_query = tq[0] if n_pairs == 1 else tq

        K = self.diffusion_K
        if self.training:
            # Per-item, per-slot noise levels in {0, ..., K}. Sampling
            # independently across slots covers BOTH inference schedules:
            #   parallel diffusion (k_m == k_c per round) AND MaskGIT
            #   (one slot at k=0, the other at k=K). The model has to
            #   handle every (k_m, k_c) combination at train time.
            # Per pair as well as per item: one forward then covers
            # n_pairs points of the (k_m, k_c) grid instead of one.
            k_m = torch.randint(0, K + 1, (batch_size, n_pairs),
                                device=x.device)
            k_c = torch.randint(0, K + 1, (batch_size, n_pairs),
                                device=x.device)
        else:
            # Eval: fully-masked (most informative single-pass setting).
            k_m = torch.full((batch_size, n_pairs), K, device=x.device,
                             dtype=torch.long)
            k_c = torch.full((batch_size, n_pairs), K, device=x.device,
                             dtype=torch.long)

        # --- self-conditioning (exposure-gap closing) -----------------
        # At inference the slots carry the model's own previous-round
        # samples, never ground truth. Train for that regime: with prob
        # self_cond_prob per item per slot, replace the slot's unmasked
        # content with the model's OWN prediction of the target frame,
        # produced by a no-grad forward at fully-masked slots (round-one
        # conditions). Token choice is the teacher-forced argmax of the
        # query logits -- a cheap approximation of true AR sampling that
        # still yields a realistic "plausible but imperfect" frame
        # embedding. No gradient flows through the override content.
        sc_mask_m = sc_emb_m = sc_mask_c = sc_emb_c = None
        sc_toks_m = sc_toks_c = None
        self._last_selfcond_frac = torch.zeros((), device=x.device)
        if self.training and self.self_cond_prob > 0:
            sc_mask_m = torch.rand(batch_size, n_pairs,
                                   device=x.device) < self.self_cond_prob
            sc_mask_c = torch.rand(batch_size, n_pairs,
                                   device=x.device) < self.self_cond_prob
            if bool(sc_mask_m.any()) or bool(sc_mask_c.any()):
                with torch.no_grad():
                    k_full = torch.full((batch_size, n_pairs), K,
                                        device=x.device, dtype=torch.long)
                    _, q_logits_sc, _ = self.forward(
                        x, T_query=T_query, k_m=k_full, k_c=k_full,
                    )
                    V = self.tokenizer.n_tokens
                    toks = q_logits_sc.view(
                        batch_size, n_pairs, 2, subseq_len, V,
                    ).argmax(dim=-1)                    # [B, Q, 2, S]
                    sc_toks_m = toks[:, :, 0]           # [B, Q, S]
                    sc_toks_c = toks[:, :, 1]
                    # A.4 corrupts at the TOKEN level, so it needs the
                    # draft tokens themselves, not their encoding; the
                    # frame-level branch needs the encoding.
                    sc_emb_m = torch.stack([
                        self._encode_frame(sc_toks_m[:, j], 0)
                        for j in range(n_pairs)
                    ], dim=1)                           # [B, Q, 1, H]
                    sc_emb_c = torch.stack([
                        self._encode_frame(sc_toks_c[:, j], 1)
                        for j in range(n_pairs)
                    ], dim=1)
                    if n_pairs == 1:
                        # Historical shapes, so a Q=1 run is unchanged.
                        sc_toks_m, sc_toks_c = sc_toks_m[:, 0], sc_toks_c[:, 0]
                        sc_emb_m, sc_emb_c = sc_emb_m[:, 0], sc_emb_c[:, 0]
                self._last_selfcond_frac = (
                    (sc_mask_m.float().sum() + sc_mask_c.float().sum())
                    / (2 * batch_size * n_pairs)
                ).detach()
            else:
                sc_mask_m = sc_mask_c = None

        ar_logits, query_logits, aux_loss = self.forward(
            x, T_query=T_query, k_m=k_m, k_c=k_c,
            sc_mask_m=sc_mask_m, sc_emb_m=sc_emb_m,
            sc_mask_c=sc_mask_c, sc_emb_c=sc_emb_c,
            sc_toks_m=sc_toks_m, sc_toks_c=sc_toks_c,
        )
        targets_ar = x
        targets_query = torch.cat(
            [x[:, 2 * t_j:2 * t_j + 2] for t_j in tq], dim=1,
        )                                                  # [B, 2Q, S]

        # --- AR loss (unchanged from parent) ---
        per_token_ar = F.cross_entropy(
            ar_logits.reshape(-1, self.tokenizer.n_tokens),
            targets_ar.reshape(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none',
        ).view(batch_size, full_seq_len, subseq_len)

        non_pad_ar = (targets_ar != self.tokenizer.pad_token).float()
        is_eos_ar = (targets_ar == self.tokenizer.eos_token).float() * non_pad_ar
        is_content_ar = non_pad_ar * (1.0 - is_eos_ar)

        frame_idx = torch.arange(full_seq_len, device=x.device)
        frame_w = torch.where(
            frame_idx % 2 == 0,
            torch.as_tensor(self.mel_loss_weight, device=x.device),
            torch.as_tensor(self.acc_loss_weight, device=x.device),
        )
        w_ar = frame_w.view(1, full_seq_len, 1).expand(batch_size, -1, subseq_len)
        ttw_ar = 1.0 + (self.eos_loss_weight - 1.0) * is_eos_ar
        weighted_ar = per_token_ar * w_ar * ttw_ar * non_pad_ar
        norm_ar = (w_ar * ttw_ar * non_pad_ar).sum().clamp_min(1.0)
        ar_loss = weighted_ar.sum() / norm_ar

        content_n_ar = is_content_ar.sum().clamp_min(1.0)
        eos_n_ar = is_eos_ar.sum().clamp_min(1.0)
        ar_loss_content = (per_token_ar * is_content_ar).sum() / content_n_ar
        ar_loss_eos = (per_token_ar * is_eos_ar).sum() / eos_n_ar

        # --- Query loss (CE on the 2 appended slots, parent's shape) ---
        per_token_q = F.cross_entropy(
            query_logits.reshape(-1, self.tokenizer.n_tokens),
            targets_query.reshape(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none',
        ).view(batch_size, 2 * n_pairs, subseq_len)
        non_pad_q = (targets_query != self.tokenizer.pad_token).float()
        keep_q = self._query_loss_keep_mask(non_pad_q)
        norm_q = keep_q.sum().clamp_min(1.0)
        query_loss = (per_token_q * keep_q).sum() / norm_q
        self._last_query_kept_frac = (
            keep_q.sum() / non_pad_q.sum().clamp_min(1.0)).detach()

        # Diagnostic split: average query CE by noise-level bin per slot.
        # Useful for spotting "model only learns at k=0 / k=K" failure modes.
        with torch.no_grad():
            q_loss_per_item = (per_token_q * non_pad_q).sum(dim=(1, 2)) / \
                non_pad_q.sum(dim=(1, 2)).clamp_min(1.0)   # [B]
            # Mean k across the batch (cheap proxy for the distribution).
            mean_k_m = k_m.float().mean()
            mean_k_c = k_c.float().mean()

        self._last_ar_loss = ar_loss.detach()
        self._last_ar_loss_content = ar_loss_content.detach()
        self._last_ar_loss_eos = ar_loss_eos.detach()
        self._last_query_loss = query_loss.detach()
        self._last_T_query = tq[0]
        self._last_n_pairs = n_pairs
        self._last_mean_k_m = mean_k_m.detach()
        self._last_mean_k_c = mean_k_c.detach()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
        else:
            aux_loss = ar_loss.new_zeros(())

        total_loss = (
            ar_loss
            + self.query_loss_weight * query_loss
            + self.aux_loss_weight * aux_loss
        )
        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_ar_loss', self._last_ar_loss)
        self.log('train_ar_loss_content', self._last_ar_loss_content)
        self.log('train_ar_loss_eos', self._last_ar_loss_eos)
        self.log('train_query_loss', self._last_query_loss)
        self.log('train_moe_aux_loss', aux_loss.detach())
        self.log('train_T_query', float(self._last_T_query))
        self.log('train_mean_k_m', self._last_mean_k_m)
        self.log('train_mean_k_c', self._last_mean_k_c)
        self.log('train_selfcond_frac', self._last_selfcond_frac)
        self.log('train_query_kept_frac', self._last_query_kept_frac)
        self.log('train_query_pairs', float(self._last_n_pairs))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ar_loss', self._last_ar_loss)
        self.log('val_ar_loss_content', self._last_ar_loss_content)
        self.log('val_ar_loss_eos', self._last_ar_loss_eos)
        self.log('val_query_loss', self._last_query_loss)
        self.log('val_query_kept_frac', self._last_query_kept_frac)
        self.log('val_moe_aux_loss', aux_loss.detach())
        return loss


# ---------------------------------------------------------------------------
# Training entry point. Mirrors duet_block but adds --diffusion_K.
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    try:
        import lightning as L
        from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
    except ImportError:
        import pytorch_lightning as L
        from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

    parser = argparse.ArgumentParser(
        description='Train M2CDuetBlockDiffusion (DuetBlock + discrete-diffusion '
                    'training at the query slots; supports both parallel and '
                    'MaskGIT-style refinement at inference).',
    )
    parser.add_argument('--task', type=str, required=True,
                        choices=sorted(TASKS))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_size', type=str, default='large',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str, default=None)
    parser.add_argument('--mod_a_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--model_abbr', type=str, default=None,
                        help='Model abbreviation used in the run-dir name '
                             '(A3/A4/A5/A6, arm-prefixed for '
                             'departures from the per-part-gate default: '
                             'A2=shared router, D1=dense, D2=hard route). '
                             'Default: derived from the flags, so '
                             'mismatched configs never share a dir. '
                             'Override only to pin a legacy name.')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--global_num_layers', type=int, default=None)
    parser.add_argument('--mel_loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_loss_weight', type=float, default=1.0)
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--preserve_program', action='store_true', default=True)
    parser.add_argument('--hardcode_program', dest='preserve_program',
                        action='store_false')
    parser.add_argument('--wandb_dir', type=str, default='/tmp/wandb')
    parser.add_argument('--save_top_k', type=int, default=2)
    parser.add_argument('--limit_val_batches', type=int, default=25,
                        help='validation batches per check. Validation is '
                             'now DETERMINISTIC (fixed sample order and '
                             'crop offsets -- see FramedDataset.__iter__), '
                             'so this sets how much of the val split the '
                             'metric covers rather than how noisy it is. '
                             'Raise it (100+) for a metric stable enough to '
                             'compare ACROSS runs; the cost is per check, '
                             'so pair a raise with a larger '
                             '--val_check_interval.')
    parser.add_argument('--val_check_interval', type=int, default=500,
                        help='steps between val evaluations. On the small '
                             'melchord corpora the val minimum can arrive '
                             'within the first ~1k steps, which 500 '
                             'resolves with only one or two points -- too '
                             'coarse to tell a real minimum from a '
                             'monotonic rise.')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--lr_total_steps', type=int, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)
    parser.add_argument('--eos_loss_weight', type=float, default=1.0)
    parser.add_argument('--silence_augment_prob', type=float, default=0.0)
    parser.add_argument('--moe_monitor_every_n_steps', type=int, default=0)
    parser.add_argument('--moe_monitor_n_samples', type=int, default=4)
    parser.add_argument('--dump_samples_dir', type=str, default=None)
    parser.add_argument('--dump_samples_n', type=int, default=4)
    parser.add_argument('--dump_samples_every_n_epochs', type=int, default=None)
    parser.add_argument('--max_polyphony', type=int, default=16)
    parser.add_argument('--gate_init_bias', type=float, default=-10.0)
    parser.add_argument('--query_loss_weight', type=float, default=1.0,
                        help='Weight on the query-slot CE term. Lower it '
                             'if the AR stream regresses while the model is '
                             'learning the diffusion task; 1.0 is fine for '
                             'warmstart-from-A.2.')
    parser.add_argument('--diffusion_K', type=int, default=4,
                        help='Number of noise-level bins. K=4 means each '
                             'slot is sampled in {0,1,2,3,4}: 0=fully '
                             'committed (sees ground truth), K=fully masked '
                             '(parent behaviour). At inference, K is also '
                             'the number of refinement steps you can run. '
                             'Larger K = finer schedule, larger embedding '
                             'table, more train-time noise diversity.')
    parser.add_argument('--self_cond_prob', type=float, default=0.5,
                        help='Per-item, per-slot probability that an '
                             'unmasked query slot is fed the model\'s own '
                             '(no-grad, teacher-forced-argmax) prediction '
                             'instead of the ground-truth embedding. '
                             'Closes the train/inference exposure gap. '
                             '0 disables (v1.0 behaviour). Costs one extra '
                             'no-grad forward per step when active.')
    parser.add_argument('--legacy_slot_rope', action='store_true', default=False,
                        help='Train with the v1.0 slot RoPE scheme (slots '
                             'at constant end-of-sequence phase) instead '
                             'of the v1.1 aligned scheme. Ablation only.')
    parser.add_argument('--time_rope_aligned', type=int, default=0,
                        help='1 = v1.2 scheme: rotary index = physical '
                             'index // 2, so m_t and c_t share rotary '
                             'position t and musical distance == rotary '
                             'distance (restores the pretrain positional '
                             'geometry; candidate fix for the long-term-'
                             'structure deficit). Subsumes v1.1 slot '
                             'alignment. Baked into the ckpt as a buffer; '
                             'inference auto-detects. Incompatible with '
                             '--legacy_slot_rope.')
    parser.add_argument('--moe_modality_bias', type=int, default=0,
                        help='1 = A.2.moe_improved: learned per-modality '
                             'additive bias [2, E] on the router logits, '
                             'zero-init. Hands the router the slot-parity '
                             'bit the per-modality attention projections '
                             'already imprint on the hidden state (probes '
                             'measured ~69%% of routing separation as that '
                             'stamp), freeing the input-driven pathway for '
                             'within-modality structure. Baked into the '
                             'ckpt as the ffn.modality_bias parameter; '
                             'inference auto-detects. Success metric: the '
                             'identical-content probe\'s stamp share on '
                             'the CONTENT pathway falls toward zero '
                             '(analyze_moe_routing.sbatch PROBE=identical).')
    parser.add_argument('--moe_modality_gates', type=int, default=0,
                        help='1 = A.2.moe_permod: per-modality router '
                             'matrices gate_m/gate_c replacing the single '
                             'shared gate -- the q_m/q_c move applied to '
                             'the router. Each gate only scores its own '
                             'stream, so the parity stamp becomes a '
                             'constant offset it cannot route on, and '
                             'within-stream routing is content-driven by '
                             'construction. The expert pool stays fully '
                             'shared and unassigned: which experts each '
                             'stream uses, and whether any serves both '
                             '(an integrator), is learned -- read it off '
                             'analyze_moe_routing\'s purity tables. '
                             'Presence of gate_m/gate_c in the ckpt is '
                             'the flag; inference auto-detects. A '
                             'warm-start ckpt with only the shared '
                             'gate.weight seeds BOTH gates with it.')
    parser.add_argument('--moe_modality_hard_route', type=int, default=0,
                        help='1 = A.2.moe_hardroute: DISJOINT expert '
                             'pools. mod_a may only reach experts '
                             '[0, E/2), mod_b only [E/2, E), enforced by '
                             'masking the other pool out of the softmax. '
                             'This is the imposed-separation control '
                             '(MoMa / VL-MoE / Uni-MoE style) that the '
                             'learned per-modality gates are argued '
                             'against: same parameters, same activated '
                             'compute, but an integrator expert serving '
                             'both streams is no longer representable. '
                             'The load-balancing aux loss is computed '
                             'WITHIN each pool so the arm is not '
                             'penalised for its own architecture. Expert '
                             'purity is 0/100 BY CONSTRUCTION -- read '
                             'within-pool content-responsiveness and '
                             'downstream quality instead. Requires an '
                             'even --moe_num_experts and topk <= E/2; '
                             'carried in the ckpt as the '
                             'ffn.hard_route_flag buffer, which '
                             'inference auto-detects.')
    parser.add_argument('--token_level_mask', type=int, default=0,
                        help='1 = A.4: per-token absorbing corruption of '
                             'the query-slot frame. At commitment level '
                             'k each non-pad token of the target frame '
                             'is masked independently with prob k/K and '
                             'the local encoder embeds the partial '
                             'frame, so intermediate k are genuinely '
                             'intermediate states (the plain variant '
                             'is all-or-nothing per slot). Endpoints '
                             'match the plain variant exactly '
                             '(all-masked -> mask_*_emb; k=0 -> clean), '
                             'so shared/mg ckpts warm-start cleanly. '
                             'Uses free token id n_normal_tokens-1 '
                             '(instrument-padding range; unreachable in '
                             'data and excluded from sampling) -- no '
                             'vocab change. Carried in the ckpt as the '
                             'token_level_mask_flag buffer; inference '
                             'auto-detects and enables confidence-based '
                             'per-token re-masking across rounds. '
                             'melchord (with_velocity=False) only.')
    parser.add_argument('--mask_revealed_query_loss', type=int, default=0,
                        help='1 = score the query loss ONLY where the '
                             'query slot did not already hand the model '
                             'its own target. The slot is both the '
                             'conditioning input and the thing being '
                             'predicted, so at k=0 (and, under A.4, at '
                             'every token that survived the draw) the '
                             'target is a free copy. D3PM / MDLM / '
                             'MaskGIT all score corrupted positions '
                             'only; we did not, which lets the copy '
                             'path compete for gradient with the '
                             '"infer it from the partner draft" path -- '
                             'the only one that exists at inference. '
                             'Self-conditioned items are kept (their '
                             'slot holds a draft that may be wrong). '
                             'OFF by default. val_loss stays '
                             'comparable across the flag (eval pins '
                             'k=K, where nothing is revealed), but the '
                             'TRAINING objective differs -- so enable '
                             'it for a WHOLE arm-set or none. Run dirs '
                             'get a "qm" marker; carried in the ckpt '
                             'as the mask_revealed_query_loss_flag '
                             'buffer.')
    parser.add_argument('--query_pairs', type=int, default=1,
                        help='Q: how many DISTINCT frames each training '
                             'forward supervises at the query slots. '
                             'Q=1 (default) is the historical behaviour '
                             'and is reproduced bit-for-bit. Q>1 '
                             'appends Q query pairs, each with its own '
                             'visibility window, its own (k_m, k_c) '
                             'draw and its own loss, so the frame pass '
                             'gets Qx the gradient for a few percent '
                             'more attention (L: 2T+2 -> 2T+2Q; at '
                             'TRAIN_LENGTH=384, Q=8 is +2% sequence, '
                             '+3.7% attention). Pairs are blind to each '
                             'other. Training-only: inference decodes '
                             'one frame at a time, so parameters, the '
                             'ckpt and the decode path are unchanged, '
                             'and validation stays at Q=1 so val_loss '
                             'remains comparable. Run dirs get a "qN" '
                             'marker for Q>1.')
    parser.add_argument('--fresh_schedule', action='store_true', default=False)
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    task = get_task(args.task)
    mod_a_path = args.mod_a_path if args.mod_a_path is not None else task.mod_a_path
    mod_b_path = args.path_to_dataset if args.path_to_dataset is not None else task.mod_b_path

    def derive_model_abbr(a):
        """ONE abbreviation per model configuration -- the run-dir name.

        Names come from the codename ledger in VARIANTS.md. The old
        concatenated flag markers (mg/tk/qm/qN) are ABOLISHED
        (2026-08-31): a run dir carries its model's name, and the name
        is DERIVED from the flags so mismatched configs still can never
        auto-resume into each other. Suffixes appear only for
        non-default settings (K != 4; A.6 at Q != 8).
        """
        if a.token_level_mask and a.mask_revealed_query_loss:
            fam = 'A4'                         # A.4 = A.5 + token corruption
        elif a.token_level_mask:
            fam = 'A4legacy'                   # deprecated: token corruption
                                               # without A.5's loss -- the
                                               # bugged first run's config
        elif a.mask_revealed_query_loss and a.query_pairs > 1:
            fam = 'A6'
        elif a.mask_revealed_query_loss:
            fam = 'A5'
        elif a.query_pairs > 1:
            fam = f'A3q{a.query_pairs}'        # unnamed combo, kept unique
        else:
            fam = 'A3'
        # E6 ablation arms = departures from the per-part-gate default.
        if a.moe_num_experts == 1:
            arm = 'D1'                         # dense / no MoE
        elif a.moe_modality_hard_route:
            arm = 'D2'                         # imposed split
        elif not a.moe_modality_gates:
            arm = 'A2'                         # shared router
        else:
            arm = ''                           # the default model
        if a.moe_modality_bias:
            arm += 'mb'
        abbr = arm + fam
        if a.diffusion_K != 4:
            abbr += f'K{a.diffusion_K}'
        if fam == 'A6' and a.query_pairs != 8:
            abbr += f'q{a.query_pairs}'
        return abbr

    tag = f'_{args.run_tag}' if args.run_tag else ''
    if args.time_rope_aligned and args.legacy_slot_rope:
        raise SystemExit('--time_rope_aligned and --legacy_slot_rope are '
                         'mutually exclusive (v1.2 vs v1.0).')
    scheme_version = ('v1.2' if args.time_rope_aligned
                      else 'v1.0' if args.legacy_slot_rope else 'v1.1')
    model_abbr = args.model_abbr or derive_model_abbr(args)
    default_name = (f"m2c_duet_block_diffusion_{scheme_version}_{args.model_size}_"
                    f"gnl{gnl}_{model_abbr}_{task.name}{tag}_"
                    f"batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    print(f'[task] {task.name}  mod_a={task.mod_a_label}  mod_b={task.mod_b_label}')

    net = M2CDuetBlockDiffusion(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
        preserve_program=args.preserve_program,
        max_lr=args.max_lr,
        lr_total_steps=args.lr_total_steps,
        aux_loss_weight=args.aux_loss_weight,
        silence_augment_prob=args.silence_augment_prob,
        eos_loss_weight=args.eos_loss_weight,
        gate_init_bias=args.gate_init_bias,
        query_loss_weight=args.query_loss_weight,
        diffusion_K=args.diffusion_K,
        slot_rope_aligned=(not args.legacy_slot_rope),
        time_rope_aligned=bool(args.time_rope_aligned),
        self_cond_prob=args.self_cond_prob,
        moe_modality_bias=bool(args.moe_modality_bias),
        moe_modality_gates=bool(args.moe_modality_gates),
        moe_modality_hard_route=bool(args.moe_modality_hard_route),
        token_level_mask=bool(args.token_level_mask),
        mask_revealed_query_loss=bool(args.mask_revealed_query_loss),
        query_pairs=args.query_pairs,
    )
    print(f'[scheme] {scheme_version}: slot_rope_aligned={not args.legacy_slot_rope}  '
          f'time_rope_aligned={bool(args.time_rope_aligned)}  '
          f'self_cond_prob={args.self_cond_prob}  '
          f'moe_modality_bias={bool(args.moe_modality_bias)}'
          f'{" (A.2.moe_improved)" if args.moe_modality_bias else ""}  '
          f'moe_modality_gates={bool(args.moe_modality_gates)}'
          f'{" (A.2.moe_permod)" if args.moe_modality_gates else ""}  '
          f'moe_modality_hard_route={bool(args.moe_modality_hard_route)}'
          f'{" (A.2.moe_hardroute)" if args.moe_modality_hard_route else ""}  '
          f'token_level_mask={bool(args.token_level_mask)}'
          f'{" (A.4)" if args.token_level_mask else ""}\n'
          f'mask_revealed_query_loss='
          f'{bool(args.mask_revealed_query_loss)}  '
          f'query_pairs={args.query_pairs}')
    print(f'Architecture: M2CDuetBlockDiffusion (A.3)  K={args.diffusion_K}  '
          f'3-pass (intra/cross/frame) + 2 gates + query slots with per-item '
          f'noise levels + k-embedding')
    print(f'Global depth: {gnl}   gate_init_bias: {args.gate_init_bias}   '
          f'query_loss_weight: {args.query_loss_weight}')

    train_set = FramedDataset(mod_b_path, TRAIN_LENGTH,
                              args.batch_size, split='train',
                              mel_path=mod_a_path)
    val_set = FramedDataset(mod_b_path, TRAIN_LENGTH,
                            args.batch_size, split='val',
                            mel_path=mod_a_path)
    train_set_loader = DataLoader(train_set, batch_size=None, num_workers=0)
    val_set_loader = DataLoader(val_set, batch_size=None, num_workers=0)

    global_batch = args.batch_size * n_gpus
    steps_per_epoch = max(1, train_set.valid_song_count // global_batch)
    if args.lr_total_steps is not None:
        implied_epochs = args.lr_total_steps / max(1, steps_per_epoch)
        print(f'[lr] valid_train_songs={train_set.valid_song_count}  '
              f'global_batch={global_batch}  steps_per_epoch={steps_per_epoch}  '
              f'lr_total_steps={args.lr_total_steps}  '
              f'implied_epochs={implied_epochs:.2f}')

    ckpt_dir = args.ckpt_dir or f'ckpt/{model_name}'
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss', save_top_k=args.save_top_k, save_last=True,
        enable_version_counter=False,
        dirpath=ckpt_dir,
        filename=model_name + '.{epoch:02d}.{val_loss:.5f}',
    )

    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(
            timeout=datetime.timedelta(hours=2),
            find_unused_parameters=True,
        )
    else:
        strategy = 'auto'

    extra_callbacks = []
    if args.moe_monitor_every_n_steps > 0:
        from moe_routing_monitor import MoERoutingMonitor
        extra_callbacks.append(
            MoERoutingMonitor(
                every_n_steps=args.moe_monitor_every_n_steps,
                n_samples=args.moe_monitor_n_samples,
            ).as_callback()
        )
    if args.dump_samples_dir is not None:
        from dump_train_samples import DumpInputSamplesCallback
        extra_callbacks.append(
            DumpInputSamplesCallback(
                out_dir=args.dump_samples_dir,
                n_samples=args.dump_samples_n,
                max_polyphony=args.max_polyphony,
                every_n_epochs=args.dump_samples_every_n_epochs,
            ).as_callback()
        )

    trainer = L.Trainer(
        devices=n_gpus,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=(args.lr_total_steps if args.lr_total_steps is not None else MAX_STEPS),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback] + extra_callbacks,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=None,
        gradient_clip_val=(args.gradient_clip_val if args.gradient_clip_val > 0 else None),
        logger=(
            WandbLogger(
                name=model_name, project='MusicMOE',
                save_dir=args.wandb_dir,
                config={
                    'batch_size': args.batch_size,
                    'model_size': args.model_size,
                    'train_length': TRAIN_LENGTH,
                    'variant': 'm2c_duet_block_diffusion',
                    'task': task.name,
                    'mod_a_label': task.mod_a_label,
                    'mod_b_label': task.mod_b_label,
                    'global_num_layers': gnl,
                    'moe_num_experts': args.moe_num_experts,
                    'moe_topk': args.moe_topk,
                    'gate_init_bias': args.gate_init_bias,
                    'query_loss_weight': args.query_loss_weight,
                    'diffusion_K': args.diffusion_K,
                    'slot_rope_aligned': not args.legacy_slot_rope,
                    'self_cond_prob': args.self_cond_prob,
                    'moe_modality_bias': bool(args.moe_modality_bias),
                    'moe_modality_gates': bool(args.moe_modality_gates),
                    'moe_modality_hard_route': bool(
                        args.moe_modality_hard_route),
                    'token_level_mask': bool(args.token_level_mask),
                    'mask_revealed_query_loss':
                        bool(args.mask_revealed_query_loss),
                    'query_pairs': args.query_pairs,
                    'run_tag': args.run_tag,
                    'model_abbr': model_abbr,
                },
            ) if args.wandb else TensorBoardLogger('tb_logs', name=model_name)
        ),
        num_sanity_val_steps=0 if args.checkpoint_path is not None else 2,
        strategy=strategy,
    )
    ckpt_path_for_resume = None
    if args.checkpoint_path is not None:
        loaded = torch.load(args.checkpoint_path, map_location='cpu',
                             weights_only=False)
        has_lightning_meta = (
            isinstance(loaded, dict)
            and 'pytorch-lightning_version' in loaded
        )
        if has_lightning_meta and not args.fresh_schedule:
            print(f'[resume] full Lightning ckpt at {args.checkpoint_path}')
            ckpt_path_for_resume = args.checkpoint_path
        else:
            if has_lightning_meta and args.fresh_schedule:
                print(f'[fresh-schedule] loading model weights only from '
                       f'{args.checkpoint_path}')
            else:
                print(f'[init] bare warm-start ckpt at {args.checkpoint_path}')
            sd = loaded['state_dict'] if isinstance(loaded, dict) and 'state_dict' in loaded else loaded
            # Drop incoming scheme-flag buffers: warm-starting from a
            # ckpt trained under another rope scheme (e.g. an A.1 ckpt,
            # or a v1.1 A.2 run via --fresh_schedule) must not silently
            # override the scheme this run's CLI declared -- the
            # [scheme] line above prints BEFORE this load and would lie.
            sd = dict(sd)
            sd.pop('time_rope_aligned_flag', None)
            sd.pop('slot_rope_aligned_flag', None)
            # A.2.moe_permod warm start: a ckpt carrying only the shared
            # gate.weight (the init ckpt, or a trained shared-gate run)
            # seeds BOTH per-modality gates with it, so the gates start
            # identical and diverge only from their streams' gradients
            # (the q_m/q_c warm-start convention).
            if args.moe_modality_gates:
                n_remap = 0
                for k in [k for k in sd if k.endswith('ffn.gate.weight')]:
                    base = k[: -len('gate.weight')]
                    for tgt in ('gate_m.weight', 'gate_c.weight'):
                        if base + tgt not in sd:
                            sd[base + tgt] = sd[k].clone()
                            n_remap += 1
                    del sd[k]
                if n_remap:
                    print(f'[init] moe_modality_gates: seeded {n_remap} '
                          f'gate_m/gate_c weights from shared gate.weight')
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                # Expected: k_emb_m.weight, k_emb_c.weight (zero-init).
                print(f'[init] {len(missing)} missing keys (first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected keys (first few: {unexpected[:3]})')
    print(f'[scheme] effective: slot_rope_aligned={net.slot_rope_aligned} '
          f'time_rope_aligned={net.time_rope_aligned} '
          '(after any warm-start load)')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
