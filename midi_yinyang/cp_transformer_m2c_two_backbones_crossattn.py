"""Two CP transformer backbones + per-layer cross-attention adapter.

Architecture (matches the user spec, corrected):

  Input: SAME interleaved 2T layout as M2CMixtureHead / m2c MoE
         x = [m_0, c_0, m_1, c_1, ..., m_{T-1}, c_{T-1}]   shape [B, 2T, subseq]

  Internally the input gets routed: mel-position tokens flow into the mel
  backbone, chord-position tokens flow into the chord backbone. Each
  backbone runs SELF-ATTENTION ONLY on its modality's single-stream
  sequence (length T).

  PER LAYER:
    u_mm^i = mel_layer[i](h_m^i)         # self-attn on mel stream, layer i
    u_cc^i = chord_layer[i](h_c^i)       # self-attn on chord stream, layer i

    Cross-attention adapter at layer i (NEW, trainable):
      u_mc^i = CrossAttn(Q = u_mm^i, K = u_cc^i, V = u_cc^i)
      u_cm^i = CrossAttn(Q = u_cc^i, K = u_mm^i, V = u_mm^i)

    Gated combination per stream:
      h_m^{i+1} = u_mm^i + sigmoid(gate_m^i(h_m^i)) * u_mc^i
      h_c^{i+1} = u_cc^i + sigmoid(gate_c^i(h_c^i)) * u_cm^i

  After L layers, per-modality local decoders + final heads produce
  per-token logits for each stream.

Compared to M2CMixtureHead (per-layer fusion with one shared backbone):

  - Two PHYSICALLY SEPARATE backbones (untied). Each gets its own pretrained
    init at step 0.
  - Each backbone sees ONLY its own modality (no interleaved 2T sequence
    hitting one backbone). In-distribution for pretrained CP transformers
    from step 0.
  - Cross-modal coupling is per-LAYER (not end-of-stack), via the dedicated
    cross-attention adapter.
  - The cross-attention adapter has its OWN Q/K/V projections (separate
    from the backbones' self-attention projections). The backbones'
    pretrained weights are never used for cross-attention.

Init from pretrained:
  - global_layers_m[i] <-- pretrained model.layer[i]
  - global_layers_c[i] <-- pretrained model.layer[i] (same ckpt or different)
  - cross-attention adapters / gates: fresh init, gate bias = -10 so
    sigmoid ~ 0 at step 0 -> adapter starts ~off -> model behaves like
    two independent pretrained backbones at init.

Run:
    python cp_transformer_m2c_two_backbones_crossattn.py \\
        --batch_size 8 --size 1 \\
        --path_to_dataset data/pop909_chord_cp4_v2.pt \\
        --checkpoint_path pretrained/two_backbones_crossattn_init.pt \\
        --wandb
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import copy
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
# Cross-attention adapter
# ---------------------------------------------------------------------------

class CrossAttentionAdapter(nn.Module):
    """Single direction of cross-attention (Q from one stream, K/V from the
    other). One instance per direction per layer.

    Output projection is init'd small so the gate has a clean "off" signal
    at step 0; combined with gate bias = -10 the adapter contributes
    ~0 to its host stream at init.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        with torch.no_grad():
            self.attn.out_proj.weight.mul_(0.01)
            if self.attn.out_proj.bias is not None:
                self.attn.out_proj.bias.zero_()

    def forward(self, query, key, value, attn_mask=None):
        out, _ = self.attn(
            query, key, value,
            attn_mask=attn_mask, need_weights=False,
        )
        return out


# ---------------------------------------------------------------------------
# Two-backbones + per-layer cross-attention adapter
# ---------------------------------------------------------------------------

class M2CTwoBackbonesCrossAttn(L.LightningModule):
    """Two pretrained CP backbones, per-layer cross-attention adapter,
    per-layer gates. Input is interleaved 2T but each backbone only ever
    sees its own modality's single-stream sequence."""

    def __init__(
        self,
        size: int = 1,                # 0=512, 1=768, 2=1024, 3=1280
        with_velocity: bool = False,
        untie_local: bool = False,
        crossattn_num_heads: int = 8,
        gate_init_bias: float = -10.0,
        max_lr: Optional[float] = None,
        max_position_embeddings: int = 1536,
    ):
        super().__init__()
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

        from cp_transformer import CPTokenizer
        self.tokenizer = CPTokenizer(with_velocity=with_velocity)
        V = self.tokenizer.n_tokens

        # ----- Local components (shared by default; untie for cross-modality) -----
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

        # ----- Per-modality global backbones AS LISTS of single-layer encoders -----
        # Single-layer wrappers let us intercept between layers to apply the
        # cross-attention adapter. Pretrained loading copies each pretrained
        # layer i into global_layers_m[i].layer[0] and global_layers_c[i].layer[0].
        single_layer_global_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=1,                         # ONE layer per wrapper
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=max_position_embeddings,
            is_decoder=True,
        )
        self.global_layers_m = nn.ModuleList([
            RoFormerEncoder(single_layer_global_config)
            for _ in range(self.num_layers)
        ])
        self.global_layers_c = nn.ModuleList([
            RoFormerEncoder(single_layer_global_config)
            for _ in range(self.num_layers)
        ])
        self.global_sos_m = nn.Parameter(torch.randn(self.hidden_size))
        self.global_sos_c = nn.Parameter(torch.randn(self.hidden_size))

        # ----- PER-LAYER cross-attention adapters (the new trainable bridges) -----
        self.cross_attn_m_reads_c = nn.ModuleList([
            CrossAttentionAdapter(self.hidden_size, num_heads=crossattn_num_heads)
            for _ in range(self.num_layers)
        ])
        self.cross_attn_c_reads_m = nn.ModuleList([
            CrossAttentionAdapter(self.hidden_size, num_heads=crossattn_num_heads)
            for _ in range(self.num_layers)
        ])

        # ----- PER-LAYER gates -----
        self.gates_m = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(self.num_layers)
        ])
        self.gates_c = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(self.num_layers)
        ])
        with torch.no_grad():
            for gate in list(self.gates_m) + list(self.gates_c):
                gate.bias.fill_(gate_init_bias)
                gate.weight.zero_()

        # ----- Per-modality final decoders -----
        self.final_decoder_m = nn.Linear(self.hidden_size, V)
        self.final_decoder_c = nn.Linear(self.hidden_size, V)

        self._future_mask = torch.empty(0)

    # ------------------------------------------------------------------
    # Local component getters (handles shared / untied)
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
        dim = tensor.size(1)
        if self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                torch.zeros([dim, dim]).float().fill_(float('-inf')), 1,
            ).to(tensor.device)
        return self._future_mask[:dim, :dim]

    def _causal_mask(self, T, device, dtype=torch.float32):
        return torch.triu(
            torch.full((T, T), float('-inf'), device=device, dtype=dtype),
            diagonal=1,
        )

    # ------------------------------------------------------------------
    # Local encode / decode (same as standard CP)
    # ------------------------------------------------------------------

    def local_encode_modality(self, x_blocks, modality):
        """x_blocks: [B, T, subseq] for ONE modality.
        Returns (h_block_summary, emb_for_decoder).
        """
        B, T, subseq = x_blocks.shape
        flat = x_blocks.view(-1, subseq)
        flat = torch.cat(
            [torch.full((flat.shape[0], 1), self.tokenizer.sos_token,
                        dtype=torch.long, device=flat.device), flat],
            dim=-1,
        )
        mask = flat != self.tokenizer.pad_token
        word_emb = self._emb_table(modality)(flat)
        token_type_id = 0 if modality == 'm' else 1
        types = torch.full_like(flat, token_type_id)
        type_emb = self.token_type_embeddings(types)
        emb = word_emb + type_emb
        h = self._local_enc(modality)(emb, encoder_attention_mask=mask)[0]
        return h[:, 0], emb[:, :-1]  # h: [B*T, H], emb: [B*T, subseq, H]

    def local_decode_modality(self, o, emb, modality):
        """o: [B*T, H] -- per-block state with cross-attn already mixed in.
        emb: [B*T, subseq, H] from local_encode_modality.
        """
        batch_size, subseq_len, _ = emb.shape
        o = o.contiguous().view(batch_size, 1, -1)
        emb = torch.cat([o, emb[:, 1:]], dim=1)
        h = self._local_dec(modality)(
            emb, attention_mask=self.buffered_future_mask(emb),
        )[0]
        head = self.final_decoder_m if modality == 'm' else self.final_decoder_c
        return head(h)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """x: interleaved [B, 2T, subseq] in [m_0, c_0, m_1, c_1, ...] order.

        Internally:
          - Split x at the position-parity boundary: mel = x[:, 0::2], chord = x[:, 1::2].
          - Each modality's tokens go ONLY into its own backbone.
          - Per-layer cross-attention adapter exchanges info between the two
            backbones at every layer.

        Returns (logits_m, logits_c), each [B, T, subseq, V_m or V_c].
        """
        B, seq_len, subseq = x.shape
        assert seq_len % 2 == 0, 'input must have an even number of slots (m+c interleaved)'
        T = seq_len // 2
        device = x.device

        # Split the interleaved input. Each modality's tokens go into its own
        # backbone -- this is the "different tokens go into different backbones"
        # constraint, expressed at the input boundary.
        x_m = x[:, 0::2].contiguous()  # [B, T, subseq]
        x_c = x[:, 1::2].contiguous()

        h_m, emb_m = self.local_encode_modality(x_m, 'm')
        h_c, emb_c = self.local_encode_modality(x_c, 'c')
        h_m = h_m.view(B, T, -1)
        h_c = h_c.view(B, T, -1)

        # Shift-by-1 each single-stream backbone (standard CP transformer AR).
        sos_m = self.global_sos_m.view(1, 1, -1).expand(B, 1, -1)
        sos_c = self.global_sos_c.view(1, 1, -1).expand(B, 1, -1)
        h_m_seq = torch.cat([sos_m, h_m[:, :-1]], dim=1)  # [B, T, H]
        h_c_seq = torch.cat([sos_c, h_c[:, :-1]], dim=1)

        # Self-attention causal mask for each backbone's layers.
        causal = self._causal_mask(T, device)

        u_m, u_c = h_m_seq, h_c_seq
        for i in range(self.num_layers):
            # Self-attention only -- each backbone sees only its modality.
            u_mm_i = self.global_layers_m[i](
                u_m, attention_mask=causal,
            ).last_hidden_state
            u_cc_i = self.global_layers_c[i](
                u_c, attention_mask=causal,
            ).last_hidden_state

            # Per-layer cross-attention adapter (the only cross-modal flow).
            # Causal across timesteps; same-step allowed (Q at t reads K at t
            # too, both are prediction states derived from past tokens).
            u_mc_i = self.cross_attn_m_reads_c[i](
                u_mm_i, u_cc_i, u_cc_i, attn_mask=causal,
            )
            u_cm_i = self.cross_attn_c_reads_m[i](
                u_cc_i, u_mm_i, u_mm_i, attn_mask=causal,
            )

            # Gated combination -> next-layer input.
            g_m_i = torch.sigmoid(self.gates_m[i](u_m))  # [B, T, 1]
            g_c_i = torch.sigmoid(self.gates_c[i](u_c))
            u_m = u_mm_i + g_m_i * u_mc_i
            u_c = u_cc_i + g_c_i * u_cm_i

        # Per-modality local decode + final head.
        logits_m = self.local_decode_modality(
            u_m.reshape(B * T, -1), emb_m, 'm',
        ).view(B, T, subseq, -1)
        logits_c = self.local_decode_modality(
            u_c.reshape(B * T, -1), emb_c, 'c',
        ).view(B, T, subseq, -1)
        return logits_m, logits_c

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def preprocess(self, x_mel, pitch_shift, y):
        return RoFormerSymbolicTransformer.preprocess(
            self, x_mel, pitch_shift, y=y,
        )

    def loss(self, x_mel, x_acc, pitch_shift):
        x_m, x_c = self.preprocess(x_mel, pitch_shift, y=x_acc)
        B, T, subseq = x_m.shape
        # Build interleaved [m_0, c_0, m_1, c_1, ...] for the forward.
        stacked = torch.stack([x_m, x_c], dim=2)
        x = stacked.view(B, T * 2, subseq)

        logits_m, logits_c = self(x)
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
        # Average per-layer gate value (sanity for adapter activation).
        gates = torch.stack(
            [g.bias for g in self.gates_m] + [g.bias for g in self.gates_c],
        ).detach().sigmoid().mean()
        self._last_gate_avg = gates
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('train_ce_m', self._last_ce_m, on_step=True)
        self.log('train_ce_c', self._last_ce_c, on_step=True)
        self.log('gate_avg', self._last_gate_avg, on_step=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('training/lr', lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_ce_m', self._last_ce_m)
        self.log('val_ce_c', self._last_ce_c)
        return loss

    # ------------------------------------------------------------------
    # Two-phase finetuning support
    # ------------------------------------------------------------------

    def _adapter_param_prefixes(self):
        """Substring prefixes for parameters that constitute the
        cross-attention adapter and other 'fusion-only' bits NOT present
        in the pretrained CP transformer ckpts. Phase 1 trains only these."""
        return (
            'cross_attn_m_reads_c.',
            'cross_attn_c_reads_m.',
            'gates_m.',
            'gates_c.',
            'token_type_embeddings.',
        )

    def freeze_backbones(self):
        """Phase 1: freeze both pretrained backbones (global_layers_m/c) and
        the local components (local_embedding/encoder/decoder, final_decoder_*,
        global_sos_*). Train ONLY the cross-attention adapter, per-layer gates,
        and token_type_embeddings.

        Per-modality final decoders and global SOS are kept FROZEN here because
        they were loaded from pretrained too -- if you want them trainable in
        Phase 1, override this method or call .requires_grad = True on them
        after this method.
        """
        keep = self._adapter_param_prefixes()
        n_frozen = n_trained = 0
        for name, p in self.named_parameters():
            is_adapter = any(name.startswith(prefix) for prefix in keep)
            p.requires_grad = is_adapter
            if is_adapter:
                n_trained += p.numel()
            else:
                n_frozen += p.numel()
        print(f'[freeze_backbones] adapter-only training: '
              f'frozen={n_frozen:,}  trainable={n_trained:,}  '
              f'({100*n_trained/(n_frozen+n_trained):.2f}% trainable)')

    def unfreeze_all(self):
        """Phase 2: turn requires_grad back on for every parameter."""
        for p in self.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        max_lr = self.max_lr if self.max_lr is not None else 1e-4
        # Filter to requires_grad=True so --freeze_backbones doesn't waste
        # AdamW optimizer state on frozen backbone parameters.
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=max_lr)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--size', type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument('--path_to_dataset', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--untie_local', action='store_true', default=False)
    parser.add_argument('--crossattn_num_heads', type=int, default=8)
    parser.add_argument('--gate_init_bias', type=float, default=-10.0)
    parser.add_argument(
        '--freeze_backbones', action='store_true', default=False,
        help='Phase 1 finetuning: freeze both pretrained backbones + local '
             'components; train only the cross-attention adapter, gates, '
             'and token_type_embeddings. For Phase 2, drop this flag and '
             'pass --checkpoint_path pointing at the Phase 1 checkpoint to '
             'unfreeze everything for joint training.',
    )
    parser.add_argument('--run_tag', type=str, default=None)
    args = parser.parse_args()

    n_gpus = max(torch.cuda.device_count(), 1)
    tag = f'_{args.run_tag}' if args.run_tag else ''
    phase_tag = '_phase1adapter' if args.freeze_backbones else ''
    default_name = (f"m2c_two_backbones_crossattn_v2.0_perlayer_sz{args.size}"
                    f"{'_untiedlocal' if args.untie_local else ''}"
                    f"{phase_tag}{tag}"
                    f"_batch_{args.batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name

    net = M2CTwoBackbonesCrossAttn(
        size=args.size,
        with_velocity=False,
        untie_local=args.untie_local,
        crossattn_num_heads=args.crossattn_num_heads,
        gate_init_bias=args.gate_init_bias,
    )
    print(f'Two backbones (untied per modality), size={args.size}, '
          f'hidden={net.hidden_size}, layers={net.num_layers}')
    print(f'Per-LAYER cross-attention adapter: '
          f'{args.crossattn_num_heads} heads per direction, '
          f'{net.num_layers} layers')
    print(f'Local: {"untied per modality" if args.untie_local else "shared"}')
    print(f'Gate init bias: {args.gate_init_bias} '
          f'(sigmoid -> {torch.sigmoid(torch.tensor(args.gate_init_bias)).item():.2e})')

    if args.freeze_backbones:
        # Phase 1: only the adapter (and gates, token_type_embeddings) trains.
        # The pretrained backbones and local components stay fixed -- their
        # behaviour at this stage matches "two independent pretrained CP
        # models", and the adapter learns where cross-modal info helps.
        net.freeze_backbones()

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
                    'variant': 'm2c_two_backbones_crossattn_perlayer',
                    'untie_local': args.untie_local,
                    'crossattn_num_heads': args.crossattn_num_heads,
                    'gate_init_bias': args.gate_init_bias,
                    'freeze_backbones': args.freeze_backbones,
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
                'freeze_backbones': args.freeze_backbones,
                'variant': 'm2c_two_backbones_crossattn_perlayer',
                'run_tag': args.run_tag,
            },
        },
        f'ckpt/{model_name}.pt',
    )
