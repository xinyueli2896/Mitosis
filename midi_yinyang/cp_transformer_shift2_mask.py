"""Random-masking / MaskGIT-style joint model over interleaved modalities.

Same backbone as DualRoFormerShift2 (CP local encoder/decoder + block-causal
global RoFormer with within-block bidirectional attention), but trained
order-free: at every block t, each modality slot is independently masked with
probability p ~ U(0, 1) per sample. The model predicts only the masked slots
from the visible ones plus history. This captures intra-block correlation
without privileging any modality ordering.

  * If only m_t is masked: learns P(m_t | history, c_t).
  * If only c_t is masked: learns P(c_t | history, m_t).
  * If both masked: learns the marginals P(m_t | history), P(c_t | history).

Inference is iterative MaskGIT-style refinement, block-by-block: at block t,
start both slots as MASK, run forward, sample the more confident slot, refix,
re-run, sample the other. n_refine_steps controls K (K=1 collapses to the
fully parallel dual sampler; K=2 = Gibbs/round-robin).

Generalises to N modalities by stacking more streams; only the masking and
the refinement loop need to be re-indexed. The architectural footprint (the
global block-causal transformer) is unchanged.

Run from midi_yinyang/:

    python cp_transformer_shift2_mask.py <bs> <size> \
        data/pop909_melody_cp4_v2.pt data/pop909_chord_cp8_v2.pt [resume]
"""

import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from cp_transformer_shift2 import (
    RoFormerSymbolicTransformer,
    TRAIN_LENGTH,
    MAX_STEPS,
)
from cp_transformer_shift2_dual import DualFramedDataset


class MaskPredictDualRoFormer(RoFormerSymbolicTransformer):
    """Two-modality MaskGIT-style joint model.

    Extends the CP vocab by one [MASK] id and adds two learned global mask
    embeddings (one per modality). The forward signature takes per-block masks
    and replaces the corresponding subseq with [MASK] tokens before local
    encoding, then replaces the encoded block summary at the global axis with
    the learned mask embedding. The block-causal pair mask inherited from the
    base class plays its intended role here: a masked slot reads from its
    visible same-block sibling AND from all earlier blocks of both modalities.
    """

    def __init__(self, size=1, max_position_embeddings=1536, with_velocity=False, max_lr=None,
                 melody_loss_weight=1.0, chord_loss_weight=1.0):
        super().__init__(size=size, max_position_embeddings=max_position_embeddings,
                         with_velocity=with_velocity, max_lr=max_lr)
        self.save_hyperparameters()
        # Per-modality loss weights (see dual model for rationale).
        self.melody_loss_weight = melody_loss_weight
        self.chord_loss_weight = chord_loss_weight

        # Extend the CP vocab by 1 to accommodate the [MASK] token used at the
        # local level for masked blocks.
        old_n = self.tokenizer.n_tokens
        self.mask_token_id = old_n
        new_n = old_n + 1

        new_emb = nn.Embedding(new_n, self.hidden_size)
        with torch.no_grad():
            new_emb.weight[:old_n] = self.local_embedding.weight
            nn.init.normal_(new_emb.weight[old_n:], std=0.02)
        self.local_embedding = new_emb

        new_dec = nn.Linear(self.hidden_size, new_n)
        with torch.no_grad():
            new_dec.weight[:old_n] = self.final_decoder.weight
            new_dec.bias[:old_n] = self.final_decoder.bias
            nn.init.normal_(new_dec.weight[old_n:], std=0.02)
            nn.init.zeros_(new_dec.bias[old_n:])
        self.final_decoder = new_dec
        self.vocab_size = new_n

        # Per-modality learned mask embedding used at the GLOBAL axis when the
        # corresponding block is masked.
        self.mask_emb_m = nn.Parameter(torch.randn(self.hidden_size))
        self.mask_emb_c = nn.Parameter(torch.randn(self.hidden_size))

    def _apply_token_mask(self, x, mask):
        """Replace tokens in masked blocks with the [MASK] token id.
        x: [B, S, subseq]; mask: [B, S] bool."""
        if mask is None:
            return x
        return torch.where(
            mask.unsqueeze(-1),
            torch.full_like(x, self.mask_token_id),
            x,
        )

    def forward(self, x_m, x_c, mask_m=None, mask_c=None):
        """x_m, x_c: [B, S, subseq_*] (preprocessed). mask_m, mask_c: [B, S]
        bool. True at (b, t) means slot is masked for that block.

        Returns y_m, y_c: [B*S, subseq_*, vocab] logits. Cross-entropy at the
        masked-block positions is the training signal."""
        B, S = x_m.shape[:2]

        x_m_in = self._apply_token_mask(x_m, mask_m)
        x_c_in = self._apply_token_mask(x_c, mask_c)

        h_m, emb_m = self.local_encode(x_m_in)
        h_c, emb_c = self.local_encode(x_c_in)
        h_m = h_m.view(B, S, -1)
        h_c = h_c.view(B, S, -1)

        if mask_m is not None:
            h_m = torch.where(mask_m.unsqueeze(-1), self.mask_emb_m.view(1, 1, -1), h_m)
        if mask_c is not None:
            h_c = torch.where(mask_c.unsqueeze(-1), self.mask_emb_c.view(1, 1, -1), h_c)

        # Interleave on the global axis. No shift-by-2 here: input position p
        # corresponds to target slot p directly. The block-causal mask
        # (within-block bidirectional, cross-block causal) is the same as in
        # the dual model so the global trunk is interchangeable.
        h_pairs = torch.stack([h_m, h_c], dim=2).reshape(B, 2 * S, -1)
        h_out = self.model(
            h_pairs, attention_mask=self.buffered_pair_causal_mask(h_pairs)
        )[0]
        h_out = h_out.view(B, S, 2, -1)

        y_m = self.local_decode(h_out[:, :, 0], emb_m)
        y_c = self.local_decode(h_out[:, :, 1], emb_c)
        return y_m, y_c

    def loss(self, x_m_raw, x_c_raw, pitch_shift):
        x_m = self.preprocess(x_m_raw, pitch_shift)
        x_c = self.preprocess(x_c_raw, pitch_shift)
        B, S, subseq_m = x_m.shape
        subseq_c = x_c.shape[2]

        # Per-sample mask ratio ~ U(0, 1), then independent Bernoulli per block
        # per modality. Guarantees order-free training across the whole grid.
        p = torch.rand(B, device=x_m.device)
        mask_m = torch.rand(B, S, device=x_m.device) < p.unsqueeze(1)
        mask_c = torch.rand(B, S, device=x_m.device) < p.unsqueeze(1)
        # Guarantee at least one masked position (otherwise loss has no signal).
        if not (mask_m.any() or mask_c.any()):
            mask_m[0, 0] = True

        y_m, y_c = self(x_m, x_c, mask_m, mask_c)

        vocab = self.vocab_size
        pad = self.tokenizer.pad_token

        loss_m_tok = F.cross_entropy(
            y_m.reshape(-1, vocab), x_m.reshape(-1),
            ignore_index=pad, reduction='none',
        ).view(B, S, subseq_m)
        loss_c_tok = F.cross_entropy(
            y_c.reshape(-1, vocab), x_c.reshape(-1),
            ignore_index=pad, reduction='none',
        ).view(B, S, subseq_c)

        block_m = mask_m.unsqueeze(-1).float()
        block_c = mask_c.unsqueeze(-1).float()
        nonpad_m = (x_m != pad).float()
        nonpad_c = (x_c != pad).float()

        denom_m = (block_m * nonpad_m).sum().clamp(min=1)
        denom_c = (block_c * nonpad_c).sum().clamp(min=1)
        loss_m = (loss_m_tok * block_m).sum() / denom_m
        loss_c = (loss_c_tok * block_c).sum() / denom_c
        return loss_m, loss_c

    def _weighted_loss(self, loss_m, loss_c):
        mw = self.melody_loss_weight
        cw = self.chord_loss_weight
        return (mw * loss_m + cw * loss_c) / max(mw + cw, 1e-8)

    def training_step(self, batch, batch_idx):
        loss_m, loss_c = self.loss(*batch)
        loss = self._weighted_loss(loss_m, loss_c)
        self.log('train_loss', loss)
        self.log('train_loss_melody', loss_m)
        self.log('train_loss_chord', loss_c)
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log('training/lr', scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss_m, loss_c = self.loss(*batch)
        loss = self._weighted_loss(loss_m, loss_c)
        self.log('val_loss', loss)
        self.log('val_loss_melody', loss_m)
        self.log('val_loss_chord', loss_c)
        return loss

    @torch.no_grad()
    def _sample_tokens(self, logits, temperature):
        """logits: [..., vocab]. Sample / argmax, never returning [MASK]."""
        logits = logits.clone()
        logits[..., self.mask_token_id] = float('-inf')
        if temperature == 0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        return torch.multinomial(flat, 1).squeeze(-1).view(*logits.shape[:-1])

    @torch.no_grad()
    def mask_predict_sampling(self, x_m_prompt, x_c_prompt, max_seq_len=384,
                              n_refine_steps=2, temperature=1.0):
        """Block-by-block generation with within-block MaskGIT refinement.

        x_m_prompt, x_c_prompt: [B, S0, subseq_*] preprocessed prompts.
        n_refine_steps (K):
          1  -> pure parallel: both slots sampled simultaneously each block.
          2  -> Gibbs: sample the more-confident slot first, fix, then sample
                the other conditioned on it.
          >2 -> Same as K=2 here since N=2 (only two slots per block); extra
                iterations are no-ops once both are filled. Future N>2 versions
                generalize with a top-confidence schedule.
        Returns (y_m, y_c) as lists (per timestep) matching the dual sampler."""
        B, S0, subseq_m = x_m_prompt.shape
        subseq_c = x_c_prompt.shape[2]
        device = x_m_prompt.device

        x_m_full = x_m_prompt.clone()
        x_c_full = x_c_prompt.clone()
        mask_m_full = torch.zeros(B, S0, dtype=torch.bool, device=device)
        mask_c_full = torch.zeros(B, S0, dtype=torch.bool, device=device)

        for t in range(S0, max_seq_len):
            if t % 10 == 0:
                print(f'Sampling block {t} / {max_seq_len}')

            # Extend with a fully-masked block at position t.
            x_m_full = torch.cat([
                x_m_full,
                torch.full((B, 1, subseq_m), self.mask_token_id,
                           dtype=torch.long, device=device),
            ], dim=1)
            x_c_full = torch.cat([
                x_c_full,
                torch.full((B, 1, subseq_c), self.mask_token_id,
                           dtype=torch.long, device=device),
            ], dim=1)
            mask_m_full = torch.cat([
                mask_m_full, torch.ones(B, 1, dtype=torch.bool, device=device),
            ], dim=1)
            mask_c_full = torch.cat([
                mask_c_full, torch.ones(B, 1, dtype=torch.bool, device=device),
            ], dim=1)

            for k in range(n_refine_steps):
                y_m, y_c = self(x_m_full, x_c_full, mask_m_full, mask_c_full)
                S = t + 1
                y_m_block = y_m.view(B, S, subseq_m, -1)[:, t]
                y_c_block = y_c.view(B, S, subseq_c, -1)[:, t]

                m_sample = self._sample_tokens(y_m_block, temperature)
                c_sample = self._sample_tokens(y_c_block, temperature)

                last = (k == n_refine_steps - 1)
                if last:
                    # Final pass: fill whatever is still masked.
                    m_active = mask_m_full[:, t].unsqueeze(-1)
                    c_active = mask_c_full[:, t].unsqueeze(-1)
                    x_m_full[:, t] = torch.where(m_active, m_sample, x_m_full[:, t])
                    x_c_full[:, t] = torch.where(c_active, c_sample, x_c_full[:, t])
                    mask_m_full[:, t] = False
                    mask_c_full[:, t] = False
                else:
                    # Pick the more-confident still-masked slot per batch elt.
                    conf_m = F.softmax(y_m_block.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                    conf_c = F.softmax(y_c_block.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                    pick_m = (conf_m >= conf_c) & mask_m_full[:, t]
                    pick_c = (conf_m <  conf_c) & mask_c_full[:, t]
                    # If only one is still masked, pick that one.
                    only_m = mask_m_full[:, t] & ~mask_c_full[:, t]
                    only_c = mask_c_full[:, t] & ~mask_m_full[:, t]
                    pick_m = pick_m | only_m
                    pick_c = pick_c | only_c

                    x_m_full[:, t] = torch.where(pick_m.unsqueeze(-1), m_sample, x_m_full[:, t])
                    x_c_full[:, t] = torch.where(pick_c.unsqueeze(-1), c_sample, x_c_full[:, t])
                    mask_m_full[:, t] = mask_m_full[:, t] & ~pick_m
                    mask_c_full[:, t] = mask_c_full[:, t] & ~pick_c

        y_m_list = [x_m_full[:, t, :] for t in range(x_m_full.shape[1])]
        y_c_list = [x_c_full[:, t, :] for t in range(x_c_full.shape[1])]
        return y_m_list, y_c_list


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python cp_transformer_shift2_mask.py '
              '<batch_size> <model_size> <melody_data.pt> <chord_data.pt> '
              '[<resume_ckpt>]')
        sys.exit(1)
    batch_size = int(sys.argv[1])
    model_size = int(sys.argv[2])
    melody_data = sys.argv[3]
    chord_data = sys.argv[4]
    checkpoint_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    suffix = sys.argv[6] if len(sys.argv) > 6 else '0.42'

    with_velocity = False
    if model_size < 0:
        model_size = -model_size - 1
        with_velocity = True
    assert model_size in [0, 1, 2, 3]
    gradient_clip = 1.0 if model_size >= 2 else None
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    n_gpus = max(torch.cuda.device_count(), 1)
    model_name = (f'cp_transformer_shift2_mask_v{suffix}'
                  f'_size{model_size}_batch_{batch_size * n_gpus}_schedule')

    melody_loss_weight = float(os.environ.get('MELODY_LOSS_WEIGHT', 1.0))
    chord_loss_weight = float(os.environ.get('CHORD_LOSS_WEIGHT', 1.0))
    print(f'Loss weights: melody={melody_loss_weight}, chord={chord_loss_weight}')
    net = MaskPredictDualRoFormer(
        size=model_size, max_lr=max_lr, with_velocity=with_velocity,
        melody_loss_weight=melody_loss_weight, chord_loss_weight=chord_loss_weight,
    )
    train_loader = DataLoader(
        DualFramedDataset(melody_data, chord_data, TRAIN_LENGTH, batch_size, split='train'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    val_loader = DataLoader(
        DualFramedDataset(melody_data, chord_data, TRAIN_LENGTH, batch_size, split='val'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss', save_top_k=10, save_last=True,
        enable_version_counter=False,
        dirpath=f'ckpt/{model_name}',
        filename=model_name + '.{epoch:02d}.{val_loss:.5f}',
    )

    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        checkpoint_path = None

    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'

    trainer = L.Trainer(
        devices=-1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        max_steps=MAX_STEPS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        val_check_interval=500,
        limit_val_batches=25,
        check_val_every_n_epoch=None,
        gradient_clip_val=gradient_clip,
        logger=[
            TensorBoardLogger("tb_logs", name=model_name),
            WandbLogger(
                project=os.environ.get("WANDB_PROJECT", "mitosis"),
                name=model_name,
            ),
        ],
        num_sanity_val_steps=0 if checkpoint_path is not None else 2,
        strategy=strategy,
    )
    trainer.fit(net, train_loader, val_loader, ckpt_path=checkpoint_path)
    torch.save(net.state_dict(), f'ckpt/{model_name}.pt')
