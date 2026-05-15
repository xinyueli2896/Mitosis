"""Shift-by-2 cp_transformer that takes parallel melody and chord streams and
interleaves them along the global axis.

The two streams are loaded from preprocess_large_midi_dataset.py outputs. At
each global timestep t the model sees one melody subseq and one chord subseq;
they are interleaved as [m_0, c_0, m_1, c_1, ..., m_{S-1}, c_{S-1}] and the
existing pair-causal mask is applied globally so the predictions for m_t and
c_t are produced in parallel, each attending to all m_{<t} and c_{<t}.

Within each subseq the local CP decoding still uses the shift-by-2 / pair
mechanism inherited from cp_transformer_shift2 (the (instrument, pitch+dur)
token pair per note).

Run from the midi_yinyang/ directory:

    python cp_transformer_shift2_dual.py <bs> <size> \
        data/pop909_melody_cp4_v2.pt data/pop909_chord_cp8_v2.pt [resume_ckpt]
"""

import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from cp_transformer_shift2 import (
    RoFormerSymbolicTransformer,
    TRAIN_LENGTH,
    MAX_STEPS,
)


class DualFramedDataset(IterableDataset):
    """Yields aligned (melody_window, chord_window, pitch_shift) batches from
    two preprocessed datasets that share the same songs and time axis. The
    per-song length used for sampling is min(melody_length, chord_length)."""

    def __init__(self, melody_path, chord_path, target_length, batch_size,
                 split='all', split_ratio=10, sample_step=1,
                 random_order=True, repeat=True):
        self.melody_path = melody_path
        self.chord_path = chord_path
        self.length_m = torch.load(melody_path[:-3] + '.length.pt', weights_only=True)
        self.length_c = torch.load(chord_path[:-3] + '.length.pt', weights_only=True)
        if len(self.length_m) != len(self.length_c):
            raise ValueError(
                f"melody and chord must have same #songs "
                f"({len(self.length_m)} vs {len(self.length_c)})"
            )
        self.length = torch.minimum(self.length_m, self.length_c)
        self.start_m = torch.cumsum(self.length_m, dim=0) - self.length_m
        self.start_c = torch.cumsum(self.length_c, dim=0) - self.length_c
        is_valid = self.length >= target_length
        ids = torch.arange(len(self.length))
        if split == 'all':
            self.valid_indices = ids[is_valid]
        elif split == 'train':
            self.valid_indices = ids[torch.logical_and(ids % split_ratio > 1, is_valid)]
        elif split == 'val':
            self.valid_indices = ids[torch.logical_and(ids % split_ratio == 1, is_valid)]
        elif split == 'test':
            self.valid_indices = ids[torch.logical_and(ids % split_ratio == 0, is_valid)]
        else:
            raise ValueError(split)
        self.split = split
        self.target_length = target_length
        self.batch_size = batch_size
        self.sample_step = sample_step
        self.random_order = random_order
        self.repeat = repeat
        self.data_m = None
        self.data_c = None
        self.pitch_shift_range = None
        print(f'DualFramedDataset {melody_path} + {chord_path} '
              f'split={split}: {len(self.valid_indices)} valid songs')

    def __iter__(self):
        if self.data_m is None:
            self.data_m = torch.load(self.melody_path, weights_only=True)
            self.data_c = torch.load(self.chord_path, weights_only=True)
            psr = torch.load(self.melody_path[:-3] + '.pitch_shift_range.pt',
                             weights_only=True).reshape(-1, 2)
            psr[psr[:, 0] < -5, 0] = -5
            psr[psr[:, 1] > 6, 1] = 6
            if self.split in ('val', 'test'):
                psr = torch.zeros_like(psr)
            self.pitch_shift_range = psr
        while True:
            if self.random_order:
                indices = torch.randperm(len(self.valid_indices))
            else:
                indices = torch.arange(len(self.valid_indices))
            for i in range(0, len(self.valid_indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                raw_ids = self.valid_indices[batch]
                psr = self.pitch_shift_range[raw_ids]
                offsets = (
                    torch.floor(
                        torch.rand(len(raw_ids))
                        * (self.length[raw_ids] - self.target_length)
                        / self.sample_step
                    ).long() * self.sample_step
                )
                grid = torch.arange(self.target_length).view(1, -1)
                idx_m = grid + (offsets + self.start_m[raw_ids]).view(-1, 1)
                idx_c = grid + (offsets + self.start_c[raw_ids]).view(-1, 1)
                pitch_shift = (
                    torch.floor(torch.rand(len(raw_ids)) * (psr[:, 1] - psr[:, 0] + 1)).long()
                    + psr[:, 0]
                )
                yield self.data_m[idx_m], self.data_c[idx_c], pitch_shift
            if not self.repeat:
                break


class DualRoFormerShift2(RoFormerSymbolicTransformer):
    """RoFormerSymbolicTransformer extended to accept paired melody/chord
    inputs. Local CP decoding remains unchanged. The global axis is built as
    an interleaved [m_t, c_t] sequence and decoded with the pair-causal mask,
    so (m_t, c_t) are predicted in parallel from {m_{<t}, c_{<t}}."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_sos_m = nn.Parameter(torch.randn(self.hidden_size))
        self.global_sos_c = nn.Parameter(torch.randn(self.hidden_size))

    def forward(self, x_m, x_c):
        B, S = x_m.shape[:2]
        h_m, emb_m = self.local_encode(x_m)
        h_c, emb_c = self.local_encode(x_c)
        h_m = h_m.view(B, S, -1)
        h_c = h_c.view(B, S, -1)
        # Interleave melody/chord on the global axis: [B, 2S, H]
        h_pairs = torch.stack([h_m, h_c], dim=2).reshape(B, 2 * S, -1)
        sos_m = self.global_sos_m.view(1, 1, -1).expand(B, 1, -1)
        sos_c = self.global_sos_c.view(1, 1, -1).expand(B, 1, -1)
        h_in = torch.cat([sos_m, sos_c, h_pairs[:, :-2]], dim=1)
        h_out = self.model(
            h_in, attention_mask=self.buffered_pair_causal_mask(h_in)
        )[0]
        h_out = h_out.view(B, S, 2, -1)
        y_m = self.local_decode(h_out[:, :, 0], emb_m)
        y_c = self.local_decode(h_out[:, :, 1], emb_c)
        return y_m, y_c

    def loss(self, x_m_raw, x_c_raw, pitch_shift):
        x_m = self.preprocess(x_m_raw, pitch_shift)
        x_c = self.preprocess(x_c_raw, pitch_shift)
        y_m, y_c = self(x_m, x_c)
        n = self.tokenizer.n_tokens
        pad = self.tokenizer.pad_token
        loss_m = F.cross_entropy(y_m.reshape(-1, n), x_m.reshape(-1), ignore_index=pad)
        loss_c = F.cross_entropy(y_c.reshape(-1, n), x_c.reshape(-1), ignore_index=pad)
        return loss_m, loss_c

    def dual_global_sampling(self, x_m, x_c, max_seq_len=384, temperature=1.0,
                             sampling_func=None):
        """Autoregressively generate melody and chord in parallel.

        x_m, x_c: prompt tensors as produced by self.preprocess(), shapes
        [B, S0, subseq_m] and [B, S0, subseq_c]. At each step t >= S0 the model
        predicts the pair (m_t, c_t) jointly from all prior melody/chord
        subseqs. Returns (y_m, y_c) where each is a list of length max_seq_len
        of [B, subseq_*] long-tensors covering the full sequence (prompt
        included)."""
        B, S0, subseq_m = x_m.shape
        subseq_c = x_c.shape[2]
        h_m, _ = self.local_encode(x_m)
        h_c, _ = self.local_encode(x_c)
        h_m = h_m.view(B, S0, -1)
        h_c = h_c.view(B, S0, -1)
        sos_m = self.global_sos_m.view(1, 1, -1).expand(B, 1, -1)
        sos_c = self.global_sos_c.view(1, 1, -1).expand(B, 1, -1)
        pairs = torch.stack([h_m, h_c], dim=2).reshape(B, 2 * S0, -1)
        # h_buffer at the start of step t has length 2 + 2*t. Output positions
        # 2*t and 2*t+1 then predict m_t and c_t respectively.
        h_buffer = torch.cat([sos_m, sos_c, pairs], dim=1)

        y_m_list = [x_m[:, i, :] for i in range(S0)]
        y_c_list = [x_c[:, i, :] for i in range(S0)]
        for t in range(S0, max_seq_len):
            if t % 10 == 0:
                print(f'Sampling {t} / {max_seq_len}')
            mask = self.buffered_pair_causal_mask(h_buffer)
            h_out = self.model(h_buffer, attention_mask=mask)[0]
            h_m_pred = h_out[:, 2 * t]
            h_c_pred = h_out[:, 2 * t + 1]
            y_m_next = self.local_sampling(
                h_m_pred, max_subseq_len=subseq_m, temperature=temperature,
                global_step=t, sampling_func=sampling_func,
            )
            y_c_next = self.local_sampling(
                h_c_pred, max_subseq_len=subseq_c, temperature=temperature,
                global_step=t, sampling_func=sampling_func,
            )
            y_m_list.append(y_m_next)
            y_c_list.append(y_c_next)
            new_h_m = self.local_encode(y_m_next.unsqueeze(1))[0].unsqueeze(1)
            new_h_c = self.local_encode(y_c_next.unsqueeze(1))[0].unsqueeze(1)
            h_buffer = torch.cat([h_buffer, new_h_m, new_h_c], dim=1)
        return y_m_list, y_c_list

    def training_step(self, batch, batch_idx):
        loss_m, loss_c = self.loss(*batch)
        loss = 0.5 * (loss_m + loss_c)
        self.log('train_loss', loss)
        self.log('train_loss_melody', loss_m)
        self.log('train_loss_chord', loss_c)
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log('training/lr', scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss_m, loss_c = self.loss(*batch)
        loss = 0.5 * (loss_m + loss_c)
        self.log('val_loss', loss)
        self.log('val_loss_melody', loss_m)
        self.log('val_loss_chord', loss_c)
        return loss


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python cp_transformer_shift2_dual.py '
              '<batch_size> <model_size> <melody_data.pt> <chord_data.pt> '
              '[<resume_ckpt>]')
        sys.exit(1)
    batch_size = int(sys.argv[1])
    model_size = int(sys.argv[2])
    melody_data = sys.argv[3]
    chord_data = sys.argv[4]
    checkpoint_path = sys.argv[5] if len(sys.argv) > 5 else None

    with_velocity = False
    if model_size < 0:
        model_size = -model_size - 1
        with_velocity = True
    assert model_size in [0, 1, 2, 3]
    gradient_clip = 1.0 if model_size >= 2 else None
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    n_gpus = max(torch.cuda.device_count(), 1)
    suffix = 'vel' if with_velocity else ''
    model_name = (f'cp_transformer_shift2_dual_v0.42{suffix}'
                  f'_size{model_size}_batch_{batch_size * n_gpus}_schedule')

    net = DualRoFormerShift2(size=model_size, max_lr=max_lr, with_velocity=with_velocity)
    train_loader = DataLoader(
        DualFramedDataset(melody_data, chord_data, TRAIN_LENGTH, batch_size, split='train'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )
    val_loader = DataLoader(
        DualFramedDataset(melody_data, chord_data, TRAIN_LENGTH, batch_size, split='val'),
        batch_size=None, num_workers=1, persistent_workers=True,
    )

    # Sanity dump: take one training batch, run it through preprocess, decode
    # each example back to a midi (MELODY + CHORD tracks) so you can listen and
    # confirm the data fed to the model is what you expect.
    print('Dumping 10 training-input samples to temp/<model_name>/inputs/...')
    from cp_transformer_shift2_dual_inference import decode_output_dual
    sample_iter = iter(DataLoader(
        DualFramedDataset(melody_data, chord_data, TRAIN_LENGTH,
                          batch_size=10, split='train', random_order=False, repeat=False),
        batch_size=None, num_workers=0,
    ))
    x_m_raw, x_c_raw, pitch_shift_dump = next(sample_iter)
    with torch.no_grad():
        x_m_dump = net.preprocess(x_m_raw, pitch_shift_dump)
        x_c_dump = net.preprocess(x_c_raw, pitch_shift_dump)
    n_dump = min(10, x_m_dump.shape[0])
    for i in range(n_dump):
        m_steps = [x_m_dump[i:i + 1, t, :] for t in range(x_m_dump.shape[1])]
        c_steps = [x_c_dump[i:i + 1, t, :] for t in range(x_c_dump.shape[1])]
        decode_output_dual(
            m_steps, c_steps,
            save_path=f'temp/{model_name}/inputs/sample_{i}.mid',
        )
    print(f'  wrote {n_dump} samples.')
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
