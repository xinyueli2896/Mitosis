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
from cp_transformer_m2c_duet_block import M2CDuetBlockAttn
from cp_transformer_m2c_jointattn import _rope_freqs
from tasks import get_task, TASKS


class M2CDuetBlockDiffusion(M2CDuetBlockAttn):
    """DuetBlock with discrete-diffusion training at the query slots.

    See module docstring. Only forward() and loss() are overridden; the
    layer stack, gates, mask construction, AR loss, and inference-time
    shape are inherited unchanged.
    """

    def __init__(self, *args, diffusion_K=4, slot_rope_aligned=True,
                 time_rope_aligned=False, self_cond_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_K = int(diffusion_K)
        # Per-modality noise-level (timestep) embeddings, indexed by
        # k in {0, ..., K}. Zero-init: a warmstart from an A.2 ckpt then
        # reproduces parent behaviour at k = K (fully masked) on step 0.
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

    @property
    def slot_rope_aligned(self):
        return bool(self.slot_rope_aligned_flag.item())

    @property
    def time_rope_aligned(self):
        return bool(self.time_rope_aligned_flag.item())

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
        clean_len = L - 2
        head_dim = H // self.num_attention_heads
        positions = torch.arange(L, device=h.device)
        positions[clean_len] = 2 * int(T_query) + 2
        positions[clean_len + 1] = 2 * int(T_query) + 3
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
                sc_mask_c=None, sc_emb_c=None):
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
        T_query = int(T_query)
        assert 1 <= T_query < T_full, (
            f'T_query={T_query} out of valid range [1, {T_full})'
        )

        K = self.diffusion_K
        k_m_t = self._coerce_k(k_m, batch_size, x.device)
        k_c_t = self._coerce_k(k_c, batch_size, x.device)

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
        # Ground-truth frame embeddings at T_query, with optional
        # self-conditioning override (model-generated frame embeddings
        # replacing gt for the flagged items).
        gt_m = h[:, 2 * T_query:2 * T_query + 1]       # [B, 1, H]
        gt_c = h[:, 2 * T_query + 1:2 * T_query + 2]   # [B, 1, H]
        if sc_mask_m is not None:
            gt_m = torch.where(sc_mask_m.view(batch_size, 1, 1),
                                sc_emb_m.to(dtype=gt_m.dtype), gt_m)
        if sc_mask_c is not None:
            gt_c = torch.where(sc_mask_c.view(batch_size, 1, 1),
                                sc_emb_c.to(dtype=gt_c.dtype), gt_c)
        mask_m_expand = self.mask_m_emb.view(1, 1, -1).expand(batch_size, 1, -1)
        mask_c_expand = self.mask_c_emb.view(1, 1, -1).expand(batch_size, 1, -1)

        # Per-item Bernoulli mask draws with prob k_m[i] / K (=0 if K==0).
        # Using max(K, 1) is purely a divide-by-zero guard; K==0 would
        # mean "never mask," which is degenerate but well-defined.
        denom = max(K, 1)
        u_m = torch.rand(batch_size, device=h.device)
        u_c = torch.rand(batch_size, device=h.device)
        is_masked_m = (u_m < (k_m_t.float() / denom)).to(h.dtype)
        is_masked_c = (u_c < (k_c_t.float() / denom)).to(h.dtype)
        # [B] -> [B, 1, 1] for broadcasting.
        is_masked_m = is_masked_m.view(batch_size, 1, 1)
        is_masked_c = is_masked_c.view(batch_size, 1, 1)

        slot_m = is_masked_m * mask_m_expand + (1.0 - is_masked_m) * gt_m
        slot_c = is_masked_c * mask_c_expand + (1.0 - is_masked_c) * gt_c

        # Add per-item k-embeddings. The model learns to interpret the
        # noise level -- crucial for iterative refinement at inference
        # where the same slot input statistically can mean very different
        # things depending on where in the K-step trajectory we are.
        k_m_e = self.k_emb_m(k_m_t).view(batch_size, 1, -1).to(h.dtype)
        k_c_e = self.k_emb_c(k_c_t).view(batch_size, 1, -1).to(h.dtype)
        slot_m = slot_m + k_m_e
        slot_c = slot_c + k_c_e

        h_full = torch.cat([h_clean, slot_m, slot_c], dim=1)
        # h_full: [B, 2T_full + 2, H]   (same shape as parent)

        h_global, aux_loss = self._run_global_stack(h_full, T_query=T_query)

        # Split outputs (identical to parent).
        h_clean_global = h_global[:, :seq_len]
        h_query_global = h_global[:, seq_len:seq_len + 2]

        ar_logits = self.local_decode(h_clean_global, emb)

        emb_reshape = emb.view(batch_size, seq_len, subseq_len, -1)
        emb_query_m = emb_reshape[:, 2 * T_query:2 * T_query + 1]
        emb_query_c = emb_reshape[:, 2 * T_query + 1:2 * T_query + 2]
        emb_query = torch.cat([emb_query_m, emb_query_c], dim=1)
        emb_query_flat = emb_query.view(batch_size * 2, subseq_len, -1)
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

        # Sample T_query (parent's behaviour).
        if self.training:
            T_query = int(torch.randint(
                low=1, high=T_full, size=(1,), device=x.device,
            ).item())
        else:
            T_query = T_full - 1

        K = self.diffusion_K
        if self.training:
            # Per-item, per-slot noise levels in {0, ..., K}. Sampling
            # independently across slots covers BOTH inference schedules:
            #   parallel diffusion (k_m == k_c per round) AND MaskGIT
            #   (one slot at k=0, the other at k=K). The model has to
            #   handle every (k_m, k_c) combination at train time.
            k_m = torch.randint(0, K + 1, (batch_size,), device=x.device)
            k_c = torch.randint(0, K + 1, (batch_size,), device=x.device)
        else:
            # Eval: fully-masked (most informative single-pass setting).
            k_m = torch.full((batch_size,), K, device=x.device, dtype=torch.long)
            k_c = torch.full((batch_size,), K, device=x.device, dtype=torch.long)

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
        self._last_selfcond_frac = torch.zeros((), device=x.device)
        if self.training and self.self_cond_prob > 0:
            sc_mask_m = torch.rand(batch_size, device=x.device) < self.self_cond_prob
            sc_mask_c = torch.rand(batch_size, device=x.device) < self.self_cond_prob
            if bool(sc_mask_m.any()) or bool(sc_mask_c.any()):
                with torch.no_grad():
                    k_full = torch.full((batch_size,), K, device=x.device,
                                         dtype=torch.long)
                    _, q_logits_sc, _ = self.forward(
                        x, T_query=T_query, k_m=k_full, k_c=k_full,
                    )
                    V = self.tokenizer.n_tokens
                    toks = q_logits_sc.view(
                        batch_size, 2, subseq_len, V,
                    ).argmax(dim=-1)                       # [B, 2, S]
                    sc_emb_m = self._encode_frame(toks[:, 0], 0)  # [B, 1, H]
                    sc_emb_c = self._encode_frame(toks[:, 1], 1)
                self._last_selfcond_frac = (
                    (sc_mask_m.float().sum() + sc_mask_c.float().sum())
                    / (2 * batch_size)
                ).detach()
            else:
                sc_mask_m = sc_mask_c = None

        ar_logits, query_logits, aux_loss = self.forward(
            x, T_query=T_query, k_m=k_m, k_c=k_c,
            sc_mask_m=sc_mask_m, sc_emb_m=sc_emb_m,
            sc_mask_c=sc_mask_c, sc_emb_c=sc_emb_c,
        )
        targets_ar = x
        targets_query = torch.stack([
            x[:, 2 * T_query],
            x[:, 2 * T_query + 1],
        ], dim=1)

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
        ).view(batch_size, 2, subseq_len)
        non_pad_q = (targets_query != self.tokenizer.pad_token).float()
        norm_q = non_pad_q.sum().clamp_min(1.0)
        query_loss = (per_token_q * non_pad_q).sum() / norm_q

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
        self._last_T_query = T_query
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
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ar_loss', self._last_ar_loss)
        self.log('val_ar_loss_content', self._last_ar_loss_content)
        self.log('val_ar_loss_eos', self._last_ar_loss_eos)
        self.log('val_query_loss', self._last_query_loss)
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
    parser.add_argument('--fresh_schedule', action='store_true', default=False)
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    task = get_task(args.task)
    mod_a_path = args.mod_a_path if args.mod_a_path is not None else task.mod_a_path
    mod_b_path = args.path_to_dataset if args.path_to_dataset is not None else task.mod_b_path

    tag = f'_{args.run_tag}' if args.run_tag else ''
    if args.time_rope_aligned and args.legacy_slot_rope:
        raise SystemExit('--time_rope_aligned and --legacy_slot_rope are '
                         'mutually exclusive (v1.2 vs v1.0).')
    scheme_version = ('v1.2' if args.time_rope_aligned
                      else 'v1.0' if args.legacy_slot_rope else 'v1.1')
    default_name = (f"m2c_duet_block_diffusion_{scheme_version}_{args.model_size}_"
                    f"gnl{gnl}_K{args.diffusion_K}_{task.name}{tag}_"
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
    )
    print(f'[scheme] {scheme_version}: slot_rope_aligned={not args.legacy_slot_rope}  '
          f'time_rope_aligned={bool(args.time_rope_aligned)}  '
          f'self_cond_prob={args.self_cond_prob}')
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
        limit_val_batches=25,
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
                    'run_tag': args.run_tag,
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
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                # Expected: k_emb_m.weight, k_emb_c.weight (zero-init).
                print(f'[init] {len(missing)} missing keys (first few: {missing[:3]})')
            if unexpected:
                print(f'[init] {len(unexpected)} unexpected keys (first few: {unexpected[:3]})')

    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=ckpt_path_for_resume)
