"""M2C MoE model with MaskGIT-style mask-predict training and inference.

Built on top of cp_transformer_m2c_samestep.M2CSameStep (which already gives
symmetric same-timestep cross-stream attention via mask + RoPE share-
position). This variant adds:

  - A [MASK] token in the vocab and two learned global-axis mask
    embeddings (mask_emb_m, mask_emb_c).
  - A masked training objective: for every batch a global mask ratio
    p ~ U(0, 1) is sampled; for every (block, modality) slot an
    independent Bernoulli(p) decides whether it's masked. Tokens at
    masked slots become [MASK] before local-encoding, and the
    encoded block summary is replaced by the modality-specific
    mask_emb at the global axis. Cross-entropy is computed only on
    masked positions.
  - NO shift-by-2 on the global axis. The mask itself is the
    information-hiding mechanism; positions align with targets.
  - The pair-causal mask (cross-block causal + within-block
    bidirectional) inherited from M2CSameStep is used at the global
    level.

Why this is what you asked for:

  - At training, m_t and c_t can each be either visible or masked.
    When only m_t is masked, the model is learning P(m_t | history,
    c_t) -- chord at the SAME timestep is part of the input. When
    only c_t is masked, P(c_t | history, m_t). When both are masked,
    the joint marginal P(m_t | history) * P(c_t | history).
  - At inference, block-by-block generation with K refinement passes
    per block. Both slots start as [MASK]; each refinement pass runs
    the model, samples the more-confident slot's tokens, fixes it,
    and re-runs to refine the other slot. The other slot's logits
    NOW reflect "given the just-sampled partner, plus history" --
    exactly the conditional the model learned with masked training.
  - The mask is fair: every modality is equally often visible /
    hidden / conditioned upon.

Run from midi_yinyang/:

    python cp_transformer_m2c_moe_mask.py \
        --batch_size 8 \
        --model_size small \
        --path_to_dataset data/pop909_chord_cp4_v2.pt \
        --moe_num_experts 4 --moe_topk 2 \
        --wandb
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from cp_transformer_m2c_moe import (
    FramedDataset,
    TRAIN_LENGTH,
    MAX_STEPS,
)
from cp_transformer_m2c_samestep import M2CSameStep


class M2CMaskMoE(M2CSameStep):
    """Mask-predict variant: random per-block-per-modality masking at training,
    iterative refinement at inference. Inherits the symmetric-RoPE global
    attention from M2CSameStep."""

    def __init__(self, *args, mel_loss_weight=1.0, acc_loss_weight=1.0, **kwargs):
        # Default equal weights for mask-predict (random masking gives both
        # modalities equal training signal). Override via CLI if desired.
        super().__init__(*args, mel_loss_weight=mel_loss_weight,
                         acc_loss_weight=acc_loss_weight, **kwargs)

        # Extend the CP vocab by 1 to add a [MASK] token.
        old_n = self.tokenizer.n_tokens
        self.mask_token_id = old_n
        self.vocab_size = old_n + 1

        new_emb = nn.Embedding(self.vocab_size, self.hidden_size)
        with torch.no_grad():
            new_emb.weight[:old_n] = self.local_embedding.weight
            nn.init.normal_(new_emb.weight[old_n:], std=0.02)
        self.local_embedding = new_emb

        new_dec = nn.Linear(self.hidden_size, self.vocab_size)
        with torch.no_grad():
            new_dec.weight[:old_n] = self.final_decoder.weight
            new_dec.bias[:old_n] = self.final_decoder.bias
            nn.init.normal_(new_dec.weight[old_n:], std=0.02)
            nn.init.zeros_(new_dec.bias[old_n:])
        self.final_decoder = new_dec

        # Learned global-axis mask embeddings, one per modality.
        self.mask_emb_m = nn.Parameter(torch.randn(self.hidden_size))
        self.mask_emb_c = nn.Parameter(torch.randn(self.hidden_size))

    def _apply_token_mask(self, x, mask):
        """Replace tokens at masked blocks with the [MASK] id.
        x: [B, T, subseq]; mask: [B, T] bool."""
        if mask is None:
            return x
        return torch.where(
            mask.unsqueeze(-1),
            torch.full_like(x, self.mask_token_id),
            x,
        )

    def forward(self, x_m, x_c, mask_m=None, mask_c=None):
        """No shift-by-2. Positions align with targets; masking is what hides
        the answer. Returns (logits_m, logits_c, aux_loss).

        x_m, x_c: [B, T, subseq] preprocessed tokens.
        mask_m, mask_c: [B, T] bool (True = slot is masked).
        """
        B, T, subseq = x_m.shape
        device = x_m.device

        x_m_in = self._apply_token_mask(x_m, mask_m)
        x_c_in = self._apply_token_mask(x_c, mask_c)

        # token_type_ids: mel=0 throughout the m subseq, chord=1 throughout c.
        # local_encode prepends an SOS so we need length subseq+1 here.
        type_m = torch.zeros(B, T, subseq + 1, dtype=torch.long, device=device)
        type_c = torch.ones(B, T, subseq + 1, dtype=torch.long, device=device)

        h_m, emb_m = self.local_encode(x_m_in, type_m)
        h_c, emb_c = self.local_encode(x_c_in, type_c)
        h_m = h_m.view(B, T, -1)
        h_c = h_c.view(B, T, -1)

        # Override block summary at masked positions with the learned global
        # mask embedding. The local emb (which the local decoder will see)
        # still contains the mask token embeddings -- consistent with how
        # mask blocks look at inference.
        if mask_m is not None:
            h_m = torch.where(
                mask_m.unsqueeze(-1),
                self.mask_emb_m.view(1, 1, -1), h_m,
            )
        if mask_c is not None:
            h_c = torch.where(
                mask_c.unsqueeze(-1),
                self.mask_emb_c.view(1, 1, -1), h_c,
            )

        # Interleave: [B, 2T, H] = [m_0, c_0, m_1, c_1, ...].
        h_pairs = torch.stack([h_m, h_c], dim=2).reshape(B, 2 * T, -1)

        # Pair-causal mask via M2CSameStep._global_interaction:
        #   cross-block causal + within-block bidirectional + symmetric RoPE.
        h_global, aux_loss = self._global_interaction(h_pairs)
        h_global = h_global.view(B, T, 2, -1)
        h_m_out = h_global[:, :, 0]
        h_c_out = h_global[:, :, 1]

        logits_m = self.local_decode(h_m_out, emb_m)  # [B*T, subseq, vocab]
        logits_c = self.local_decode(h_c_out, emb_c)
        logits_m = logits_m.view(B, T, subseq, -1)
        logits_c = logits_c.view(B, T, subseq, -1)
        return logits_m, logits_c, aux_loss

    def loss(self, x_m_raw, x_c_raw, pitch_shift):
        x_m, x_c = self.preprocess(x_m_raw, pitch_shift, y=x_c_raw)
        B, T, subseq = x_m.shape

        # Per-sample mask ratio uniform in [0, 1]; per-(block, modality)
        # Bernoulli(p) independently. Guarantee at least one masked position.
        p = torch.rand(B, device=x_m.device)
        mask_m = torch.rand(B, T, device=x_m.device) < p.unsqueeze(1)
        mask_c = torch.rand(B, T, device=x_m.device) < p.unsqueeze(1)
        if not (mask_m.any() or mask_c.any()):
            mask_m[0, 0] = True

        logits_m, logits_c, aux_loss = self(x_m, x_c, mask_m, mask_c)

        pad = self.tokenizer.pad_token
        loss_m_per_tok = F.cross_entropy(
            logits_m.reshape(-1, self.vocab_size),
            x_m.reshape(-1),
            ignore_index=pad,
            reduction='none',
        ).view(B, T, subseq)
        loss_c_per_tok = F.cross_entropy(
            logits_c.reshape(-1, self.vocab_size),
            x_c.reshape(-1),
            ignore_index=pad,
            reduction='none',
        ).view(B, T, subseq)

        block_m = mask_m.unsqueeze(-1).float()
        block_c = mask_c.unsqueeze(-1).float()
        nonpad_m = (x_m != pad).float()
        nonpad_c = (x_c != pad).float()

        denom_m = (block_m * nonpad_m).sum().clamp_min(1.0)
        denom_c = (block_c * nonpad_c).sum().clamp_min(1.0)
        loss_m = (loss_m_per_tok * block_m).sum() / denom_m
        loss_c = (loss_c_per_tok * block_c).sum() / denom_c

        mw = self.mel_loss_weight
        cw = self.acc_loss_weight
        ce_loss = (mw * loss_m + cw * loss_c) / max(mw + cw, 1e-8)

        if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
            aux_loss = aux_loss.mean()
            total_loss = ce_loss + 0.01 * aux_loss
        else:
            aux_loss = ce_loss.new_zeros(())
            total_loss = ce_loss

        # Logging-friendly: also expose per-modality losses
        self._last_loss_m = loss_m.detach()
        self._last_loss_c = loss_c.detach()
        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_loss_melody', self._last_loss_m, on_step=True)
        self.log('train_loss_chord', self._last_loss_c, on_step=True)
        self.log('moe_aux_loss', aux_loss, on_step=True, on_epoch=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('training/lr', lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_loss_melody', self._last_loss_m)
        self.log('val_loss_chord', self._last_loss_c)
        self.log('val_moe_aux_loss', aux_loss, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_tokens(self, logits, temperature):
        """logits: [..., vocab]. Returns sampled token ids of shape logits[...:-1].
        [MASK] is excluded from sampling."""
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
        """Block-by-block generation with K within-block refinement passes.

        x_m_prompt, x_c_prompt: [B, S0, subseq] preprocessed prompts.
        n_refine_steps:
          1 = pure parallel (m_t and c_t sampled independently each block).
          2 = Gibbs / MaskGIT: sample the more-confident slot first; refine
              the other conditioned on it. This is what gives bidirectional
              same-timestep dependency at inference.
          >2 = no extra benefit for N=2 modalities; both fill in <=2 passes.

        Returns (mel_frames, chord_frames), lists of length max_seq_len of
        [B, subseq] long tensors."""
        B, S0, subseq = x_m_prompt.shape
        device = x_m_prompt.device

        x_m_full = x_m_prompt.clone()
        x_c_full = x_c_prompt.clone()
        mask_m_full = torch.zeros(B, S0, dtype=torch.bool, device=device)
        mask_c_full = torch.zeros(B, S0, dtype=torch.bool, device=device)

        for t in range(S0, max_seq_len):
            if t % 10 == 0:
                print(f'[mask-predict] block {t}/{max_seq_len}')

            # Append a fully-masked block at position t.
            x_m_full = torch.cat([
                x_m_full,
                torch.full((B, 1, subseq), self.mask_token_id,
                           dtype=torch.long, device=device),
            ], dim=1)
            x_c_full = torch.cat([
                x_c_full,
                torch.full((B, 1, subseq), self.mask_token_id,
                           dtype=torch.long, device=device),
            ], dim=1)
            mask_m_full = torch.cat([
                mask_m_full,
                torch.ones(B, 1, dtype=torch.bool, device=device),
            ], dim=1)
            mask_c_full = torch.cat([
                mask_c_full,
                torch.ones(B, 1, dtype=torch.bool, device=device),
            ], dim=1)

            for k in range(n_refine_steps):
                logits_m, logits_c, _ = self(
                    x_m_full, x_c_full, mask_m_full, mask_c_full,
                )
                # logits_m: [B, t+1, subseq, vocab]
                S = t + 1
                m_block_logits = logits_m[:, t]  # [B, subseq, vocab]
                c_block_logits = logits_c[:, t]

                m_sample = self._sample_tokens(m_block_logits, temperature)
                c_sample = self._sample_tokens(c_block_logits, temperature)

                last = (k == n_refine_steps - 1)
                if last:
                    # Fill anything still masked.
                    m_active = mask_m_full[:, t].unsqueeze(-1)
                    c_active = mask_c_full[:, t].unsqueeze(-1)
                    x_m_full[:, t] = torch.where(m_active, m_sample, x_m_full[:, t])
                    x_c_full[:, t] = torch.where(c_active, c_sample, x_c_full[:, t])
                    mask_m_full[:, t] = False
                    mask_c_full[:, t] = False
                else:
                    # Confidence = mean of max softmax probability across subseq positions.
                    conf_m = F.softmax(m_block_logits.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                    conf_c = F.softmax(c_block_logits.float(), dim=-1).max(dim=-1).values.mean(dim=-1)
                    pick_m = (conf_m >= conf_c) & mask_m_full[:, t]
                    pick_c = (conf_m < conf_c) & mask_c_full[:, t]
                    # If only one is still masked, force-pick that one.
                    only_m = mask_m_full[:, t] & ~mask_c_full[:, t]
                    only_c = mask_c_full[:, t] & ~mask_m_full[:, t]
                    pick_m = pick_m | only_m
                    pick_c = pick_c | only_c

                    x_m_full[:, t] = torch.where(pick_m.unsqueeze(-1), m_sample, x_m_full[:, t])
                    x_c_full[:, t] = torch.where(pick_c.unsqueeze(-1), c_sample, x_c_full[:, t])
                    mask_m_full[:, t] = mask_m_full[:, t] & ~pick_m
                    mask_c_full[:, t] = mask_c_full[:, t] & ~pick_c

        mel_frames = [x_m_full[:, t, :] for t in range(x_m_full.shape[1])]
        chord_frames = [x_c_full[:, t, :] for t in range(x_c_full.shape[1])]
        return mel_frames, chord_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train M2C-MoE mask-predict variant. Random per-block-'
                    'per-modality masking; loss only on masked positions; '
                    'training and inference both use the pair-causal + '
                    'symmetric-RoPE backbone, so m_t and c_t at the same '
                    'timestep can listen to each other in both phases.',
    )
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str,
                        help='path to the chord .pt; melody .pt found by '
                             'replacing "chord" -> "melody"')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--mel_loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_loss_weight', type=float, default=1.0)
    args = parser.parse_args()

    batch_size = args.batch_size
    model_size = args.model_size
    dataset = args.path_to_dataset
    checkpoint_path = args.checkpoint_path

    with_velocity = False
    n_gpus = max(torch.cuda.device_count(), 1)

    default_name = (f"m2c_mask_moe_v1.0_{model_size}"
                    f"_batch_{batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CMaskMoE(
        large=(model_size == 'large'),
        with_velocity=with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
    )
    print(f'MoE enabled: {net.global_roformer.config.moe}')
    print('Variant: mask-predict + symmetric-RoPE pair-causal global '
          'attention. Same-step cross-stream dependency is captured at '
          'BOTH training and inference (iterative refinement).')
    print(f'Loss weights: mel={args.mel_loss_weight}, '
          f'chord={args.acc_loss_weight}')

    train_set_loader = DataLoader(
        FramedDataset(dataset, TRAIN_LENGTH, batch_size, split='train'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    val_set_loader = DataLoader(
        FramedDataset(dataset, TRAIN_LENGTH, batch_size, split='val'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss', save_top_k=5, save_last=True,
        enable_version_counter=False,
        dirpath=f'ckpt/{model_name}',
        filename=model_name + '.{epoch:02d}.{val_loss:.5f}',
    )

    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'

    trainer = L.Trainer(
        devices=n_gpus,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        max_steps=MAX_STEPS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        val_check_interval=500,
        limit_val_batches=25,
        check_val_every_n_epoch=None,
        logger=(
            WandbLogger(
                name=model_name, project='MusicMOE',
                config={
                    'batch_size': batch_size,
                    'model_size': model_size,
                    'train_length': TRAIN_LENGTH,
                    'variant': 'mask_predict_symmetric_rope',
                    'mel_loss_weight': args.mel_loss_weight,
                    'acc_loss_weight': args.acc_loss_weight,
                },
            ) if args.wandb else TensorBoardLogger('tb_logs', name=model_name)
        ),
        num_sanity_val_steps=0 if checkpoint_path is not None else 2,
        strategy=strategy,
    )
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    torch.save(
        {
            'state_dict': net.state_dict(),
            'hyper_parameters': {
                'model_size': model_size,
                'with_velocity': with_velocity,
                'moe_num_experts': args.moe_num_experts,
                'moe_topk': args.moe_topk,
                'moe_intermediate_size': args.moe_intermediate_size,
                'mel_loss_weight': args.mel_loss_weight,
                'acc_loss_weight': args.acc_loss_weight,
                'variant': 'mask_predict_symmetric_rope',
            },
        },
        f'ckpt/{model_name}.pt',
    )
