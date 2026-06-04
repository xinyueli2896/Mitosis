"""Per-layer-fusion + joint mixture head: 'two models in concept'.

The architecture below the head is the EXISTING shift-by-2 interleaved
per-layer fusion model (cp_transformer_m2c_per_layer_fusion.M2CPerLayerFusion):
one shared backbone called twice with different attention masks, gated
cross-attention per fusion block.

What changes here is ONLY the output layer:

  Instead of a single per-modality final projection (which produces
  independent per-stream logits and lets m_t and c_t be sampled
  independently from each other), we attach a JOINT MIXTURE HEAD that:

    1. Computes a per-timestep mixture prior pi(k | h_m_t, h_c_t).
    2. Produces K per-modality vocab projections per token.

  Joint distribution per timestep:

      p(m_t, c_t | h_m_t, h_c_t)
        = sum_k pi_k(h_m_t, h_c_t)
              * prod_i p_k(m_t[i] | h_m_decoded[t, i])
              * prod_j p_k(c_t[j] | h_c_decoded[t, j])

  The mixture index k carries the cross-modal correlation. Sample k once,
  sample both modalities from p_k -- single forward, no iteration, true
  joint sampling. m_t and c_t are CONDITIONALLY INDEPENDENT given k but
  JOINTLY CORRELATED through the mixing.

  "Two models in concept": the per-k decoders p_k(m | h_m) and
  p_k(c | h_c) are modality-specific projections; the rest of the model
  is shift-by-2 interleaved as before. For a future audio+symbolic
  experiment, swap the per-modality local encoders/decoders and the
  per-k decoder heads for modality-specific ones -- everything else
  transfers.

Inference modes (in the companion inference script):
  co        : sample k from pi, then m_t and c_t from p_k in parallel.
  mel2chord : posterior p(k | m_t obs) proportional to pi_k * p_k(m_t obs);
              sample k, then c_t from p_k.
  chord2mel : symmetric.
  mel_only  : sample k from prior pi, m_t from p_k. Chord unused.
  chord_only: symmetric.

Run:
    python cp_transformer_m2c_mixture_head.py \\
        --batch_size 8 --model_size large \\
        --path_to_dataset data/pop909_chord_cp4_v2.pt \\
        --mixture_K 8 \\
        --wandb
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

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
from cp_transformer_m2c_per_layer_fusion import M2CPerLayerFusion


# ---------------------------------------------------------------------------
# Joint mixture head
# ---------------------------------------------------------------------------

class JointMixtureHead(nn.Module):
    """Per-timestep K-component joint mixture.

    Inputs (during training / a single forward pass):
      h_m_t  [B, T, H]   -- per-timestep mel-slot hidden state from the global
                            tower (the position that predicts m_t).
      h_c_t  [B, T, H]   -- symmetric for chord.
      h_m_dec [B, T, subseq, H] -- per-token mel hidden states from
                                   local_decode_to_hidden (pre-vocab projection).
      h_c_dec [B, T, subseq, H] -- symmetric for chord.

    Outputs:
      log_pi              [B, T, K]
      logits_m_per_k      [B, T, subseq, K, V_m]
      logits_c_per_k      [B, T, subseq, K, V_c]
    """

    def __init__(self, hidden_size, vocab_size_m, vocab_size_c, K=8):
        super().__init__()
        self.K = K
        self.hidden_size = hidden_size
        self.vocab_size_m = vocab_size_m
        self.vocab_size_c = vocab_size_c

        # Mixture router: conditioned on joint (h_m_t, h_c_t).
        self.mixture_router = nn.Linear(2 * hidden_size, K)

        # K-component vocab projections. Per-modality so audio+symbolic later
        # can pass different vocab sizes here.
        self.final_m = nn.Linear(hidden_size, K * vocab_size_m)
        self.final_c = nn.Linear(hidden_size, K * vocab_size_c)

    def mixture_log_pi(self, h_m_t, h_c_t):
        joint = torch.cat([h_m_t, h_c_t], dim=-1)
        return F.log_softmax(self.mixture_router(joint), dim=-1)

    def per_k_logits_m(self, h_dec):
        """h_dec: [..., H] -> [..., K, V_m]."""
        logits = self.final_m(h_dec)
        new_shape = logits.shape[:-1] + (self.K, self.vocab_size_m)
        return logits.view(*new_shape)

    def per_k_logits_c(self, h_dec):
        logits = self.final_c(h_dec)
        new_shape = logits.shape[:-1] + (self.K, self.vocab_size_c)
        return logits.view(*new_shape)


# ---------------------------------------------------------------------------
# Per-layer-fusion + joint mixture head
# ---------------------------------------------------------------------------

class M2CMixtureHead(M2CPerLayerFusion):
    """Per-layer fusion with the joint mixture head at the output.

    Inherits:
      - The interleaved shift-by-2 input layout from the m2c MoE family.
      - The per-layer fusion stack (per-block dual-pass + per-block gates).
    Adds:
      - JointMixtureHead at the output (replaces per-stream independent
        final projections with K-component joint mixing).
    """

    def __init__(self, *args, mixture_K=8, **kwargs):
        super().__init__(*args, **kwargs)

        V = self.tokenizer.n_tokens
        self.mixture_K = mixture_K
        self.joint_mixture_head = JointMixtureHead(
            hidden_size=self.hidden_size,
            vocab_size_m=V,
            vocab_size_c=V,
            K=mixture_K,
        )

        # The parent's final_decoder is no longer the output. We keep it
        # allocated so inherited methods that reference it stay valid; the
        # mixture-head FORWARD path uses local_decode_to_hidden + per-k
        # projections inside the joint_mixture_head.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def local_decode_to_hidden(self, h, emb):
        """Variant of local_decode that returns the local_decoder hidden
        states BEFORE the final vocab projection. The mixture head's
        per-k Linears then project these to per-k vocab logits.

        h: [B*T, H] per-block summary.
        emb: [B*T, subseq, H] from local_encode.
        Returns: [B*T, subseq, H].
        """
        batch_size, subseq_len, _ = emb.shape
        h = h.contiguous().view(batch_size, 1, -1)
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        out = self.local_decoder(
            emb, attention_mask=self.buffered_future_mask(emb),
        )[0]
        return out

    def _interleave_type_ids(self, B, T, subseq, device):
        """[B, 2T, subseq+1] with mel=0 at even positions, chord=1 at odd."""
        alt = torch.tensor([0, 1], dtype=torch.long, device=device).repeat(T)
        return alt[None, :, None].expand(B, 2 * T, subseq + 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """x: [B, 2T, subseq] interleaved [m_0, c_0, m_1, c_1, ...].

        Returns:
          log_pi:            [B, T, K]
          logits_m_per_k:    [B, T, subseq, K, V_m]
          logits_c_per_k:    [B, T, subseq, K, V_c]
          aux_loss:          MoE balance loss (from per-layer fusion)
        """
        B, seq_len, subseq_len = x.shape
        assert seq_len % 2 == 0
        T = seq_len // 2
        device = x.device

        type_ids = self._interleave_type_ids(B, T, subseq_len, device)
        h, emb = self.local_encode(x, type_ids)
        h = h.view(B, seq_len, -1)

        # Shift-by-2: prepend SOS pair, drop last 2.
        sos = self.global_sos.view(1, 1, -1).repeat(B, 2, 1)
        h = torch.cat([sos, h[:, :-2]], dim=1)

        # Per-layer fusion's _global_interaction (inherited from M2CPerLayerFusion).
        h_global, aux_loss = self._global_interaction(h)
        # h_global: [B, 2T, H]. Mel-prediction positions are 0::2, chord are 1::2.

        h_m = h_global[:, 0::2]  # [B, T, H]
        h_c = h_global[:, 1::2]

        # Slice emb to mel/chord positions for per-modality local decoding.
        emb = emb.view(B, seq_len, subseq_len, -1)
        emb_m = emb[:, 0::2].reshape(B * T, subseq_len, -1)
        emb_c = emb[:, 1::2].reshape(B * T, subseq_len, -1)

        h_m_dec = self.local_decode_to_hidden(
            h_m.reshape(B * T, -1), emb_m,
        )  # [B*T, subseq, H]
        h_c_dec = self.local_decode_to_hidden(
            h_c.reshape(B * T, -1), emb_c,
        )
        h_m_dec = h_m_dec.view(B, T, subseq_len, -1)
        h_c_dec = h_c_dec.view(B, T, subseq_len, -1)

        log_pi = self.joint_mixture_head.mixture_log_pi(h_m, h_c)
        logits_m_per_k = self.joint_mixture_head.per_k_logits_m(h_m_dec)
        logits_c_per_k = self.joint_mixture_head.per_k_logits_c(h_c_dec)

        return log_pi, logits_m_per_k, logits_c_per_k, aux_loss

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, x_mel, x_acc, pitch_shift):
        x = self.preprocess(x_mel, pitch_shift, y=x_acc)
        # x: interleaved [B, 2T, subseq]
        B, seq_len, subseq = x.shape
        T = seq_len // 2

        log_pi, logits_m_per_k, logits_c_per_k, aux_loss = self(x)

        targets_m = x[:, 0::2]  # [B, T, subseq]
        targets_c = x[:, 1::2]

        # Per-component log-probs.
        log_probs_m = F.log_softmax(logits_m_per_k.float(), dim=-1)
        log_probs_c = F.log_softmax(logits_c_per_k.float(), dim=-1)

        K = self.mixture_K
        tgt_m = targets_m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, K, 1)
        tgt_c = targets_c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, K, 1)
        log_p_target_m = log_probs_m.gather(-1, tgt_m).squeeze(-1)  # [B, T, subseq, K]
        log_p_target_c = log_probs_c.gather(-1, tgt_c).squeeze(-1)

        pad = self.tokenizer.pad_token
        mask_m = (targets_m != pad).float().unsqueeze(-1)
        mask_c = (targets_c != pad).float().unsqueeze(-1)
        log_p_target_m = log_p_target_m * mask_m
        log_p_target_c = log_p_target_c * mask_c

        # Sum over within-block tokens -> per-component block log-likelihood.
        log_p_m_block = log_p_target_m.sum(dim=2)  # [B, T, K]
        log_p_c_block = log_p_target_c.sum(dim=2)

        # Mixture log-likelihood per timestep:
        # log sum_k pi_k(h_m, h_c) * p_k(m_block | h_m) * p_k(c_block | h_c)
        mixture_logp = torch.logsumexp(
            log_pi + log_p_m_block + log_p_c_block, dim=-1,
        )  # [B, T]

        nll = -mixture_logp.mean()

        # Light entropy regularization on pi to discourage early collapse.
        pi_entropy = -(log_pi.exp() * log_pi).sum(dim=-1).mean()
        entropy_coef = 0.01
        total_loss = nll - entropy_coef * pi_entropy

        # Aux loss from MoE balance (inherited from per-layer fusion).
        if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
            aux_loss = aux_loss.mean()
            total_loss = total_loss + 0.01 * aux_loss
        else:
            aux_loss = nll.new_zeros(())

        self._last_nll = nll.detach()
        self._last_pi_entropy = pi_entropy.detach()
        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_nll', self._last_nll, on_step=True)
        self.log('train_pi_entropy', self._last_pi_entropy, on_step=True)
        self.log('moe_aux_loss', aux_loss, on_step=True, on_epoch=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('training/lr', lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_nll', self._last_nll)
        self.log('val_pi_entropy', self._last_pi_entropy)
        return loss


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train per-layer fusion + joint mixture head.',
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument('--path_to_dataset', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--moe_intermediate_size', type=int, default=None)
    parser.add_argument('--global_num_layers', type=int, default=None,
                        help='Default: 12 if large else 6.')
    parser.add_argument('--mixture_K', type=int, default=8,
                        help='Joint mixture component count. K=1 -> '
                             'independent predictions per modality, no coupling.')
    parser.add_argument('--mel_loss_weight', type=float, default=1.0,
                        help='Reserved; current loss is joint mixture NLL '
                             '(no per-modality weighting). Kept for compat.')
    parser.add_argument('--acc_loss_weight', type=float, default=3.0,
                        help='Reserved; see --mel_loss_weight.')
    parser.add_argument('--run_tag', type=str, default=None)
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if args.model_size == 'large' else 6

    tag = f'_{args.run_tag}' if args.run_tag else ''
    default_name = (f"m2c_mixture_head_v1.0_{args.model_size}_"
                    f"gnl{gnl}_K{args.mixture_K}{tag}"
                    f"_batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CMixtureHead(
        large=(args.model_size == 'large'),
        with_velocity=False,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        mel_loss_weight=args.mel_loss_weight,
        acc_loss_weight=args.acc_loss_weight,
        mixture_K=args.mixture_K,
    )
    print(f'Architecture: per-layer fusion (shift-by-2 interleaved) '
          f'+ joint mixture head')
    print(f'Global depth: {gnl} fusion blocks')
    print(f'Mixture K: {args.mixture_K}  '
          f'(K=1 -> no joint coupling; K>1 -> joint mixture sampling)')

    train_set_loader = DataLoader(
        FramedDataset(args.path_to_dataset, TRAIN_LENGTH, args.batch_size, split='train'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    val_set_loader = DataLoader(
        FramedDataset(args.path_to_dataset, TRAIN_LENGTH, args.batch_size, split='val'),
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
                    'batch_size': args.batch_size,
                    'model_size': args.model_size,
                    'train_length': TRAIN_LENGTH,
                    'variant': 'm2c_mixture_head',
                    'global_num_layers': gnl,
                    'mixture_K': args.mixture_K,
                    'run_tag': args.run_tag,
                },
            ) if args.wandb else TensorBoardLogger('tb_logs', name=model_name)
        ),
        num_sanity_val_steps=0 if args.checkpoint_path is not None else 2,
        strategy=strategy,
    )
    trainer.fit(net, train_set_loader, val_set_loader,
                ckpt_path=args.checkpoint_path)
    torch.save(
        {
            'state_dict': net.state_dict(),
            'hyper_parameters': {
                'model_size': args.model_size,
                'moe_num_experts': args.moe_num_experts,
                'moe_topk': args.moe_topk,
                'moe_intermediate_size': args.moe_intermediate_size,
                'global_num_layers': gnl,
                'mixture_K': args.mixture_K,
                'variant': 'm2c_mixture_head',
                'run_tag': args.run_tag,
            },
        },
        f'ckpt/{model_name}.pt',
    )
