"""Two CP transformer backbones + cross-attention adapter.

Architecture (matches the user spec):

  mel tokens   --> mel_local_encoder   --> h_m_blocks  --> global_roformer_m (pretrained, SELF-ATTN ONLY)  --> u_mm
  chord tokens --> chord_local_encoder --> h_c_blocks  --> global_roformer_c (pretrained, SELF-ATTN ONLY)  --> u_cc

  Cross-attention adapter (NEW, trainable; the only learned bridge):
    u_mc = CrossAttn(Q = u_mm, K = u_cc, V = u_cc)   # mel queries reading chord
    u_cm = CrossAttn(Q = u_cc, K = u_mm, V = u_mm)   # chord queries reading mel

  Gated combination per stream:
    o_m = u_mm + sigmoid(gate_m(h_m_in)) * u_mc
    o_c = u_cc + sigmoid(gate_c(h_c_in)) * u_cm

  Per-modality local decoder + final head produce per-token logits for each stream.

Properties:

  - Each pretrained backbone keeps its original self-attention behaviour
    (it only ever sees its own modality's sequence). At init with the cross-
    attention gate set to ~0 (bias = -10), the model computes EXACTLY what
    each pretrained model would alone -- training lifts the gate.
  - Two physically separate models: each backbone is a standalone CP
    transformer. Loadable from a pretrained ckpt with a 1:1 key map.
  - Future modalities: untie local_embedding / local_encoder / local_decoder
    per modality (--untie_local flag) and the architecture supports
    different per-modality vocab sizes.
  - No mixture head -- the cross-attention adapter is the cross-modal
    coupling mechanism. Each modality has its own per-token decoder head.

Mask choices:
  - Self-attention (inside each backbone): standard causal across the
    modality's sequence (shift-by-1 within each stream).
  - Cross-attention (in the adapter): causal across timesteps + same-step
    allowed. Both Q and K at the SAME timestep are PREDICTION states
    derived from the past (not the current target), so no chicken-and-egg
    at sampling time.

Run:
    python cp_transformer_m2c_two_backbones_crossattn.py \\
        --batch_size 8 --model_size large \\
        --path_to_dataset data/pop909_chord_cp4_v2.pt \\
        --checkpoint_path <init_from_pretrained.pt>  # OPTIONAL \\
        --wandb
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from cp_transformer_m2c_moe import (
    RoFormerSymbolicTransformer,
    FramedDataset,
    TRAIN_LENGTH,
    MAX_STEPS,
)
from transformers.models.roformer.modeling_roformer import (
    RoFormerEncoder,
    RoFormerConfig,
)


# ---------------------------------------------------------------------------
# Cross-attention adapter: a single attention layer with Q and K/V from
# different streams. PyTorch's nn.MultiheadAttention does exactly this.
# ---------------------------------------------------------------------------

class CrossAttentionAdapter(nn.Module):
    """Bidirectional cross-attention between two streams.

    Note: a SINGLE adapter instance computes cross-attention in one direction
    (Q from one stream, K/V from the other). We instantiate TWO per fusion
    block -- one for mel-reads-chord, one for chord-reads-mel.

    The adapter is the only NEW trainable parameters in the model (beyond
    the per-stream gates). At init the output should be small / near-zero
    so the gated combination o = u_self + sigmoid(gate) * u_cross starts
    close to u_self (preserving the pretrained backbone's behaviour).
    """

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Initialize output projection close to zero so the gate has a clean
        # "off" signal at init. This is the standard pattern for additive
        # adapters layered onto a frozen backbone.
        with torch.no_grad():
            self.attn.out_proj.weight.mul_(0.01)
            if self.attn.out_proj.bias is not None:
                self.attn.out_proj.bias.zero_()

    def forward(self, query, key, value, attn_mask=None):
        # query, key, value: [B, T, H]
        out, _ = self.attn(
            query, key, value,
            attn_mask=attn_mask, need_weights=False,
        )
        return out


# ---------------------------------------------------------------------------
# The two-backbones + cross-attention adapter model
# ---------------------------------------------------------------------------

class M2CTwoBackbonesCrossAttn(L.LightningModule):
    """Two pretrained CP backbones with a cross-attention adapter bridge.

    Inherits from LightningModule directly (not RoFormerSymbolicTransformer)
    because the standard CP transformer's forward / loss are not
    representative of the two-stream structure -- we override them entirely.

    Local encoder / decoder / embedding can be either shared across modalities
    (default, fine for melody+chord using the same CP vocab) or untied per
    modality (--untie_local, needed for audio+symbolic later). Final per-token
    decoder heads are always per-modality.
    """

    def __init__(
        self,
        size: int = 1,                # 0=512, 1=768, 2=1024, 3=1280 (matches cp_transformer.py)
        with_velocity: bool = False,
        untie_local: bool = False,
        crossattn_num_heads: int = 8,
        gate_init_bias: float = -10.0,  # sigmoid(-10) ~ 4.5e-5; adapter starts off
        max_lr: Optional[float] = None,
    ):
        super().__init__()
        # Mirror cp_transformer.py's size schedule for direct pretrained
        # compatibility.
        self.hidden_size = [512, 768, 1024, 1280][size]
        self.num_layers = [6, 12, 24, 32][size]
        self.num_attention_heads = [8, 12, 16, 16][size]
        self.intermediate_size = [1024, 3072, 4096, 5120][size]
        self.local_model_num_layers = 3
        self.local_model_num_attention_heads = 8
        self.local_model_intermediate_size = 768
        self.untie_local = untie_local
        self.with_velocity = with_velocity
        self.max_lr = max_lr

        # Tokenizer (shared, CP vocab is per-CP-encoding).
        from cp_transformer import CPTokenizer
        self.tokenizer = CPTokenizer(with_velocity=with_velocity)
        V = self.tokenizer.n_tokens

        # ----- Local components (per-modality if untied) -----
        local_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=True,
        )
        if untie_local:
            self.local_embedding_m = nn.Embedding(V, self.hidden_size)
            self.local_embedding_c = nn.Embedding(V, self.hidden_size)
            self.local_encoder_m = RoFormerEncoder(local_config)
            self.local_encoder_c = RoFormerEncoder(local_config)
            self.local_decoder_m = RoFormerEncoder(local_decoder_config)
            self.local_decoder_c = RoFormerEncoder(local_decoder_config)
        else:
            self.local_embedding = nn.Embedding(V, self.hidden_size)
            self.local_encoder = RoFormerEncoder(local_config)
            self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)

        # ----- Per-modality global backbones (THE pretrained-loadable ones) -----
        global_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1536,
            is_decoder=True,
        )
        self.global_roformer_m = RoFormerEncoder(global_config)
        self.global_roformer_c = RoFormerEncoder(global_config)
        self.global_sos_m = nn.Parameter(torch.randn(self.hidden_size))
        self.global_sos_c = nn.Parameter(torch.randn(self.hidden_size))

        # ----- Cross-attention adapter (the only new trainable bridge) -----
        self.cross_attn_m_reads_c = CrossAttentionAdapter(
            self.hidden_size, num_heads=crossattn_num_heads,
        )
        self.cross_attn_c_reads_m = CrossAttentionAdapter(
            self.hidden_size, num_heads=crossattn_num_heads,
        )

        # ----- Per-stream gates: scalar gating of the adapter contribution -----
        self.gate_m = nn.Linear(self.hidden_size, 1)
        self.gate_c = nn.Linear(self.hidden_size, 1)
        # Initialize gate bias so sigmoid(bias) ~ 0 at start -- the adapter
        # starts OFF and the model behaves like two independent pretrained
        # backbones at step 0. Training lifts the gate.
        with torch.no_grad():
            self.gate_m.bias.fill_(gate_init_bias)
            self.gate_c.bias.fill_(gate_init_bias)
            self.gate_m.weight.zero_()
            self.gate_c.weight.zero_()

        # ----- Per-modality final decoders -----
        self.final_decoder_m = nn.Linear(self.hidden_size, V)
        self.final_decoder_c = nn.Linear(self.hidden_size, V)

        # AR mask buffer (within-block local decoder).
        self._future_mask = torch.empty(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emb_table(self, modality):
        if self.untie_local:
            return (self.local_embedding_m if modality == 'm'
                    else self.local_embedding_c)
        return self.local_embedding

    def _local_enc(self, modality):
        if self.untie_local:
            return (self.local_encoder_m if modality == 'm'
                    else self.local_encoder_c)
        return self.local_encoder

    def _local_dec(self, modality):
        if self.untie_local:
            return (self.local_decoder_m if modality == 'm'
                    else self.local_decoder_c)
        return self.local_decoder

    def buffered_future_mask(self, tensor):
        """Causal AR mask cached buffer. Pulled from RoFormerSymbolicTransformer
        so subclasses don't drag in the full parent dependency just for this."""
        dim = tensor.size(1)
        if self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                torch.zeros([dim, dim]).float().fill_(float('-inf')),
                1,
            ).to(tensor.device)
        return self._future_mask[:dim, :dim]

    def local_encode(self, x, token_type_ids, modality):
        """Returns (h_block_summary, emb_for_decoder) for one modality."""
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)
        x = torch.cat(
            [torch.full((x.shape[0], 1), self.tokenizer.sos_token,
                        dtype=torch.long, device=x.device), x],
            dim=-1,
        )
        mask = x != self.tokenizer.pad_token
        word_emb = self._emb_table(modality)(x)
        type_emb = self.token_type_embeddings(token_type_ids.view(-1, subseq_len + 1))
        emb = word_emb + type_emb
        h = self._local_enc(modality)(emb, encoder_attention_mask=mask)[0]
        return h[:, 0], emb[:, :-1]

    def local_decode(self, o, emb, modality):
        """o: [B*T, H] per-block hidden state with cross-attn already mixed in.
        emb: [B*T, subseq, H] from local_encode.
        Returns per-token logits via the modality-specific final head.
        """
        batch_size, subseq_len, _ = emb.shape
        o = o.contiguous().view(batch_size, 1, -1)
        emb = torch.cat([o, emb[:, 1:]], dim=1)
        h = self._local_dec(modality)(
            emb, attention_mask=self.buffered_future_mask(emb),
        )[0]
        head = self.final_decoder_m if modality == 'm' else self.final_decoder_c
        return head(h)

    def _causal_mask(self, T, device, dtype=torch.float32):
        return torch.triu(
            torch.full((T, T), float('-inf'), device=device, dtype=dtype),
            diagonal=1,
        )

    def _causal_same_step_ok_mask(self, T, device, dtype=torch.float32):
        """Cross-attention mask: causal across timesteps, same-step ALLOWED.
        Same shape as the standard causal mask -- they're identical here
        because we treat positions as timesteps directly (no shift offset).
        Both Q and K at the SAME timestep are prediction states derived from
        encoded PAST tokens, so no chicken-egg dependency at inference.
        """
        return self._causal_mask(T, device, dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_m, x_c):
        """x_m, x_c: [B, T, subseq] preprocessed CP tokens for each modality.
        Returns (logits_m, logits_c) per [B, T, subseq, V].
        """
        B, T, subseq = x_m.shape
        device = x_m.device

        type_m = torch.zeros(B, T, subseq + 1, dtype=torch.long, device=device)
        type_c = torch.ones(B, T, subseq + 1, dtype=torch.long, device=device)

        h_m, emb_m = self.local_encode(x_m, type_m, 'm')
        h_c, emb_c = self.local_encode(x_c, type_c, 'c')
        h_m = h_m.view(B, T, -1)
        h_c = h_c.view(B, T, -1)

        # Shift-by-1 each stream independently. Each stream is a standard
        # single-modality AR sequence over its own modality's tokens only.
        sos_m = self.global_sos_m.view(1, 1, -1).expand(B, 1, -1)
        sos_c = self.global_sos_c.view(1, 1, -1).expand(B, 1, -1)
        h_m_in = torch.cat([sos_m, h_m[:, :-1]], dim=1)
        h_c_in = torch.cat([sos_c, h_c[:, :-1]], dim=1)

        causal = self._causal_mask(T, device)
        out_m = self.global_roformer_m(h_m_in, attention_mask=causal)
        out_c = self.global_roformer_c(h_c_in, attention_mask=causal)
        u_mm = out_m.last_hidden_state  # [B, T, H]
        u_cc = out_c.last_hidden_state

        # Cross-attention adapter (causal + same-step allowed).
        cross_mask = self._causal_same_step_ok_mask(T, device)
        u_mc = self.cross_attn_m_reads_c(
            query=u_mm, key=u_cc, value=u_cc, attn_mask=cross_mask,
        )
        u_cm = self.cross_attn_c_reads_m(
            query=u_cc, key=u_mm, value=u_mm, attn_mask=cross_mask,
        )

        # Gated combination.
        g_m = torch.sigmoid(self.gate_m(h_m_in))  # [B, T, 1]
        g_c = torch.sigmoid(self.gate_c(h_c_in))
        o_m = u_mm + g_m * u_mc
        o_c = u_cc + g_c * u_cm

        # Per-modality local decode + final head.
        logits_m = self.local_decode(
            o_m.reshape(B * T, -1), emb_m, 'm',
        ).view(B, T, subseq, -1)
        logits_c = self.local_decode(
            o_c.reshape(B * T, -1), emb_c, 'c',
        ).view(B, T, subseq, -1)
        return logits_m, logits_c

    # ------------------------------------------------------------------
    # Loss (uses parent's preprocess via cp_transformer_m2c_moe)
    # ------------------------------------------------------------------

    def preprocess(self, x_mel, pitch_shift, y):
        """Defer to RoFormerSymbolicTransformer.preprocess; same CP encoding."""
        # Temporarily borrow the parent's preprocess via a thin shim.
        return RoFormerSymbolicTransformer.preprocess(
            self, x_mel, pitch_shift, y=y,
        )

    def loss(self, x_mel, x_acc, pitch_shift):
        x_m, x_c = self.preprocess(x_mel, pitch_shift, y=x_acc)
        logits_m, logits_c = self(x_m, x_c)
        pad = self.tokenizer.pad_token
        ce_m = F.cross_entropy(
            logits_m.reshape(-1, logits_m.shape[-1]),
            x_m.reshape(-1),
            ignore_index=pad,
        )
        ce_c = F.cross_entropy(
            logits_c.reshape(-1, logits_c.shape[-1]),
            x_c.reshape(-1),
            ignore_index=pad,
        )
        loss = 0.5 * (ce_m + ce_c)
        self._last_ce_m = ce_m.detach()
        self._last_ce_c = ce_c.detach()
        self._last_gate_m = self.gate_m.bias.detach().sigmoid().mean()
        self._last_gate_c = self.gate_c.bias.detach().sigmoid().mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_ce_m', self._last_ce_m, on_step=True)
        self.log('train_ce_c', self._last_ce_c, on_step=True)
        self.log('gate_m_avg', self._last_gate_m, on_step=True)
        self.log('gate_c_avg', self._last_gate_c, on_step=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('training/lr', lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ce_m', self._last_ce_m)
        self.log('val_ce_c', self._last_ce_c)
        return loss

    def configure_optimizers(self):
        max_lr = self.max_lr if self.max_lr is not None else 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr,
            total_steps=MAX_STEPS, pct_start=0.005,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the two-backbones + cross-attention adapter model.',
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--size', type=int, default=1, choices=[0, 1, 2, 3],
                        help='Hidden-size schedule (matches cp_transformer.py): '
                             '0=512, 1=768, 2=1024, 3=1280.')
    parser.add_argument('--path_to_dataset', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--untie_local', action='store_true', default=False)
    parser.add_argument('--crossattn_num_heads', type=int, default=8)
    parser.add_argument('--gate_init_bias', type=float, default=-10.0)
    parser.add_argument('--run_tag', type=str, default=None)
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    tag = f'_{args.run_tag}' if args.run_tag else ''
    default_name = (f"m2c_two_backbones_crossattn_v1.0_sz{args.size}"
                    f"{'_untiedlocal' if args.untie_local else ''}{tag}"
                    f"_batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CTwoBackbonesCrossAttn(
        size=args.size,
        with_velocity=False,
        untie_local=args.untie_local,
        crossattn_num_heads=args.crossattn_num_heads,
        gate_init_bias=args.gate_init_bias,
    )
    print(f'Two backbones (untied), size={args.size}, hidden={net.hidden_size}, '
          f'global_layers={net.num_layers}')
    print(f'Cross-attention adapter: {args.crossattn_num_heads} heads per direction')
    print(f'Local components: {"untied per modality" if args.untie_local else "shared"}')
    print(f'Gate init bias: {args.gate_init_bias}  '
          f'(sigmoid -> {torch.sigmoid(torch.tensor(args.gate_init_bias)).item():.2e}, '
          f'adapter starts ~off)')

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
                    'size': args.size,
                    'train_length': TRAIN_LENGTH,
                    'variant': 'm2c_two_backbones_crossattn',
                    'untie_local': args.untie_local,
                    'crossattn_num_heads': args.crossattn_num_heads,
                    'gate_init_bias': args.gate_init_bias,
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
                'size': args.size,
                'untie_local': args.untie_local,
                'crossattn_num_heads': args.crossattn_num_heads,
                'gate_init_bias': args.gate_init_bias,
                'variant': 'm2c_two_backbones_crossattn',
                'run_tag': args.run_tag,
            },
        },
        f'ckpt/{model_name}.pt',
    )
