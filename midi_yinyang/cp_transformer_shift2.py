import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerModel, RoFormerConfig, RoFormerEncoder
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import sys

TRAIN_LENGTH = 384
MAX_STEPS = 2000000

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class CPTokenizer(object):
    def __init__(self, with_velocity=False):
        if with_velocity:
            self.n_normal_tokens = 24 * 128 + 128 * 16  # 24 durations, 128 pitches, 128 instruments, 16 velocities
        else:
            self.n_normal_tokens = 24 * 128 + 256  # 24 durations, 128 pitches, 128 instruments (padded to 256)
        self.n_tokens = self.n_normal_tokens + 3
        self.sos_token = self.n_normal_tokens
        self.eos_token = self.n_normal_tokens + 1
        self.pad_token = self.n_normal_tokens + 2
        self.with_velocity = with_velocity

class RoFormerSymbolicTransformer(L.LightningModule):

    def __init__(self, size=1, max_position_embeddings=1536, with_velocity=False, max_lr=None):
        super().__init__()
        self.hidden_size = [512, 768, 1024, 1280][size]
        self.num_layers = [6, 12, 24, 32][size]
        self.num_attention_heads = [8, 12, 16, 16][size]
        self.intermediate_size = [1024, 3072, 4096, 5120][size]
        self.local_model_num_layers = 3
        self.local_model_num_attention_heads = 8
        self.local_model_intermediate_size = 768
        main_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=max_position_embeddings,
            is_decoder=True,
        )
        self.model = self.get_base_model(main_roformer_config)
        local_encoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=True
        )
        self.tokenizer = CPTokenizer(with_velocity=with_velocity)
        self.local_embedding = nn.Embedding(self.tokenizer.n_tokens, self.hidden_size)
        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, self.tokenizer.n_tokens)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)
        self._pair_future_mask = torch.empty(0)
        self.with_velocity = with_velocity
        self.max_lr = max_lr

    def get_base_model(self, config):
        return RoFormerEncoder(config)

    def local_encode(self, x):
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)
        # prepend a <sos> token
        x = torch.cat([torch.full((x.shape[0], 1), self.tokenizer.sos_token, dtype=torch.long, device=x.device), x], dim=-1)
        mask = x != self.tokenizer.pad_token
        emb = self.local_embedding(x)
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]
        # h[:, 0]: pooled representation; emb[:, 1:-2]: token embeddings of a_0,b_0,...,a_{N-2},b_{N-2}
        # for shift-by-2 teacher forcing (two seed slots are filled with the global h in local_decode)
        return h[:, 0], emb[:, 1:-2]

    def local_decode(self, h, emb):
        # emb: [B*S, 2N-2, H] holding a_0,b_0,...,a_{N-2},b_{N-2}; h: global context [B*S, H]
        # Build [h, h, a_0, b_0, ..., a_{N-2}, b_{N-2}] of length 2N (shift-by-2)
        bs_seq = emb.shape[0]
        h = h.reshape(bs_seq, 1, -1)
        emb = torch.cat([h, h, emb], dim=1)
        h = self.local_decoder(emb, attention_mask=self.buffered_pair_causal_mask(emb))[0]
        return self.final_decoder(h)

    def _build_slot_logit_mask(self):
        """Per-slot additive logit mask. The CP vocab is shared across both
        slots of a (program, pitch+duration) pair, but each slot has a
        disjoint set of legal token ids; an undertrained model otherwise
        leaks tokens across slots and decode_output rejects them. Apply
        this mask before softmax/argmax to constrain sampling."""
        n = self.tokenizer.n_tokens
        a_mask = torch.full((n,), float("-inf"))
        b_mask = torch.full((n,), float("-inf"))
        if self.with_velocity:
            # a-slot: program + 128 * velocity_bin in [0, 128*16)
            a_mask[: 128 * 16] = 0
            # b-slot: pitch + (duration + 16) * 128 in [128*16, 128*16 + 24*128)
            b_mask[128 * 16 : 128 * 16 + 24 * 128] = 0
        else:
            # a-slot: program in [0, 128)
            a_mask[:128] = 0
            # b-slot: pitch + (duration + 1) * 128 in [128, 128 + 24*128) = [128, 3200)
            b_mask[128 : 128 + 24 * 128] = 0
        # eos terminates the local subseq from the a-slot; pad fills both slots.
        a_mask[self.tokenizer.eos_token] = 0
        a_mask[self.tokenizer.pad_token] = 0
        b_mask[self.tokenizer.pad_token] = 0
        return torch.stack([a_mask, b_mask], dim=0)  # [2, n_tokens]

    def local_sampling(self, h, max_subseq_len=32, temperature=1.0, global_step=None, sampling_func=None,
                       constrain_to_slot=True):
        # Pair-parallel sampling: at step i, predict a_i and b_i jointly, conditioned on a_{<i}, b_{<i}.
        # max_subseq_len must be even (interleaved a/b tokens).
        assert max_subseq_len % 2 == 0
        batch_size, hidden = h.shape
        n_pairs = max_subseq_len // 2
        # seed both modality slots with the global context h
        cur_emb = h[:, None, :].expand(batch_size, 2, hidden).contiguous()
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)
        slot_mask = self._build_slot_logit_mask().to(h.device) if constrain_to_slot else None
        for i in range(n_pairs):
            out = self.local_decoder(cur_emb, attention_mask=self.buffered_pair_causal_mask(cur_emb))[0]
            logits = self.final_decoder(out[:, -2:])  # [B, 2, n_tokens]
            if slot_mask is not None:
                logits = logits + slot_mask[None]
            if sampling_func is not None:
                logits = torch.stack([
                    sampling_func(global_step, 2 * i, logits[:, 0]),
                    sampling_func(global_step, 2 * i + 1, logits[:, 1]),
                ], dim=1)
            if temperature == 0:
                sampled = logits.argmax(dim=-1)  # [B, 2]
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(
                    probs.reshape(-1, self.tokenizer.n_tokens), 1
                ).view(batch_size, 2)
            sampled[eos_triggered] = self.tokenizer.pad_token
            eos_triggered = eos_triggered | (sampled == self.tokenizer.eos_token).any(dim=-1)
            y = torch.cat([y, sampled], dim=1)
            if torch.all(eos_triggered):
                remaining = max_subseq_len - 2 * (i + 1)
                if remaining > 0:
                    y = torch.cat([y, torch.full((batch_size, remaining), self.tokenizer.pad_token, dtype=torch.long, device=h.device)], dim=1)
                break
            cur_emb = torch.cat([cur_emb, self.local_embedding(sampled)], dim=1)
        return y

    def global_sampling(self, x, max_seq_len=384, temperature=1.0, sampling_func=None):
        # import time
        # start_time = time.time()
        batch_size, seq_len, subseq_len = x.shape
        h, _ = self.local_encode(x)
        h = h.view(batch_size, seq_len, self.hidden_size)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        past_key_values = None
        h_next = h
        for i in range(seq_len, max_seq_len):
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)
                # print('Time passed', time.time() - start_time)
            attention_mask = self.buffered_future_mask(h) if past_key_values is None else None
            h_out, past_key_values = self.model(h_next, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=False)
            # h_out_ref = self.model(h, attention_mask=self.buffered_future_mask(h))[0]
            # print('assert')
            # assert torch.allclose(h_out[:, -1:], h_out_ref[:, -1:], rtol=1e-3, atol=1e-5)
            # h_out = h_out_ref
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature, global_step=i, sampling_func=sampling_func)
            y.append(y_next)
            h_next = self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
            # h = torch.cat([h, self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)], dim=1)
        return y

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def buffered_pair_causal_mask(self, tensor):
        # Block-causal mask over pairs: position i may attend to j iff (j // 2) <= (i // 2).
        # Within a pair {2k, 2k+1} both slots attend to all earlier pairs and to each other,
        # implementing parallel decoding of a_k and b_k from a_{<k}, b_{<k}.
        dim = tensor.size(1)
        if (
                self._pair_future_mask.size(0) == 0
                or (not self._pair_future_mask.device == tensor.device)
                or self._pair_future_mask.size(0) < dim
        ):
            rows = torch.arange(dim).unsqueeze(1) // 2
            cols = torch.arange(dim).unsqueeze(0) // 2
            mask = torch.zeros([dim, dim])
            mask.masked_fill_(cols > rows, float("-inf"))
            self._pair_future_mask = mask
        self._pair_future_mask = self._pair_future_mask.to(tensor)
        return self._pair_future_mask[:dim, :dim]

    def forward(self, x):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        batch_size, seq_len, subseq_len = x.shape
        h, emb = self.local_encode(x)
        h = h.view(batch_size, seq_len, -1)
        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)
        # Use global transformer to decode
        h = self.model(h, attention_mask=self.buffered_future_mask(h))[0]
        return self.local_decode(h, emb)


    def preprocess(self, x, pitch_shift, tuple_size=4):
        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // tuple_size, tuple_size)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // tuple_size, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255
        eos_indices = x[:, :, :, 0] == 254
        is_not_drum = x[:, :, :, 0] != 127
        if self.with_velocity:
            x_processed[:, :, :, 0] = x[:, :, :, 0] + 128 * (x[:, :, :, 3] // 8)  # there are 128 instruments
            x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 16) * 128 + pitch_shift[:, None, None] * is_not_drum
        else:
            x_processed[:, :, :, 0] = x[:, :, :, 0]
            x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 1) * 128 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = self.tokenizer.pad_token
        x_processed[:, :, :, 0][eos_indices] = self.tokenizer.eos_token
        return x_processed.view(batch_size, seq_length, subseq_length // tuple_size * 2)
    def loss(self, x, pitch_shift):
        x = self.preprocess(x, pitch_shift)
        y = self(x)
        return F.cross_entropy(y.view(-1, self.tokenizer.n_tokens), x.view(-1), ignore_index=self.tokenizer.pad_token)

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        # scheduler step
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log('training/lr', scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        max_lr = self.max_lr
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005)
        return [optimizer], [scheduler]

    def inference_perplexity(self, x):
        x = self.preprocess(x, torch.zeros(x.shape[0], device=x.device, dtype=torch.long))
        y = self(x)
        result = F.cross_entropy(y.view(-1, self.tokenizer.n_tokens), x.view(-1), ignore_index=self.tokenizer.pad_token, reduction='none')
        batch_size = x.shape[0]
        result = result.view(batch_size, -1).sum(dim=1) / torch.count_nonzero(x.view(batch_size, -1) != self.tokenizer.pad_token, dim=1)
        # Calculate mean and variance along the batch dimension
        return result.mean().item(), result.std().item() * (batch_size / (batch_size - 1)) ** 0.5

class FramedDataset(IterableDataset):

    def __init__(self, file_path, target_length, batch_size, split='all', split_ratio=10, sample_step=1, random_order=True,
                 repeat=True):
        self.file_path = file_path
        self.length = torch.load(file_path[:-3] + '.length.pt', weights_only=True)
        self.start = torch.cumsum(self.length, dim=0) - self.length
        # Invalid samples are those whose length is less than min_length
        is_valid = self.length >= target_length
        self.song_indices = torch.arange(len(self.start))
        # Get training or validation split
        if split == 'all':
            self.valid_indices = self.song_indices[is_valid]
        elif split == 'train':
            self.valid_indices = self.song_indices[torch.logical_and(self.song_indices % split_ratio > 1, is_valid)]
        elif split == 'val':
            self.valid_indices = self.song_indices[torch.logical_and(self.song_indices % split_ratio == 1, is_valid)]
        elif split == 'test':
            self.valid_indices = self.song_indices[torch.logical_and(self.song_indices % split_ratio == 0, is_valid)]
        self.split = split
        self.valid_song_count = len(self.valid_indices)
        self.target_length = target_length
        self.batch_size = batch_size
        self.sample_step = sample_step
        self.random_order = random_order
        self.repeat = repeat
        print('Metadata for dataset', file_path, 'split', split, 'loaded. Number of valid songs:', self.valid_song_count, 'first 20:', self.valid_indices[:20])
        self.data = None
        self.pitch_shift_range = None
    def __iter__(self):
        if self.data is None:
            self.data = torch.load(self.file_path, weights_only=True)
            self.pitch_shift_range = torch.load(self.file_path[:-3] + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2)
            self.pitch_shift_range[self.pitch_shift_range[:, 0] < -5, 0] = -5
            self.pitch_shift_range[self.pitch_shift_range[:, 1] > 6, 1] = 6
            if self.split == 'val' or self.split == 'test':
                self.pitch_shift_range = torch.zeros_like(self.pitch_shift_range)  # No pitch shift for validation
            print('Data for dataset', self.file_path, 'loaded.')
        while True:
            if self.random_order:
                indices = torch.randperm(len(self.valid_indices))
            else:
                indices = torch.arange(len(self.valid_indices))
            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_pitch_shift_range = self.pitch_shift_range[self.valid_indices[batch_indices]]
                raw_ids = self.valid_indices[batch_indices]
                # starts = torch.randint(self.start[raw_ids], self.start[raw_ids] + self.length[raw_ids] - self.target_length)
                starts = torch.floor(torch.rand(len(raw_ids)) * (self.length[raw_ids] - self.target_length) / self.sample_step).long() * self.sample_step + self.start[raw_ids]
                index_matrix = torch.arange(self.target_length).view(1, -1) + starts.view(-1, 1)
                # Shift the pitch in range [min, max], inclusive
                batch_pitch_shift = torch.floor(torch.rand(len(raw_ids)) * (batch_pitch_shift_range[:, 1] - batch_pitch_shift_range[:, 0] + 1)).long() + batch_pitch_shift_range[:, 0]
                yield self.data[index_matrix], batch_pitch_shift
            if not self.repeat:
                break


if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    model_size = int(sys.argv[2])
    with_velocity = False
    if model_size < 0:  # with velocity
        model_size = -model_size - 1
        with_velocity = True
    assert model_size in [0, 1, 2, 3]
    gradient_clip = 1.0 if model_size >= 2 else None
    max_lr = 5e-5 if model_size >= 2 else 1e-4
    n_gpus = max(torch.cuda.device_count(), 1)
    suffix = 'vel' if with_velocity else ''
    model_name = f'cp_transformer_shift2_v0.42{suffix}_size{model_size}_batch_{batch_size * n_gpus}_schedule'
    net = RoFormerSymbolicTransformer(size=model_size, max_lr=max_lr, with_velocity=with_velocity)
    train_set_loader = DataLoader(FramedDataset('data/la_cp16_v2.pt', TRAIN_LENGTH, batch_size), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset('data/rwc_cp16_v2.pt', TRAIN_LENGTH, batch_size), batch_size=None, num_workers=1, persistent_workers=True)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_loss',
                                                      save_top_k=10,
                                                      save_last=True,
                                                      enable_version_counter=False,
                                                      dirpath=f'ckpt/{model_name}',
                                                      filename=model_name + '.{epoch:02d}.{val_loss:.5f}')

    # load from checkpoint
    checkpoint_path = None
    if len(sys.argv) > 3:
        checkpoint_path = sys.argv[3]
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'
    trainer = L.Trainer(devices=-1,
                        precision="bf16-mixed" if torch.cuda.is_available() else 32,
                        max_steps=MAX_STEPS,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        callbacks=[checkpoint_callback],
                        val_check_interval=2500,
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
                        strategy=strategy)
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    # save the model (parameters only)
    torch.save(net.state_dict(), f'ckpt/{model_name}.pt')