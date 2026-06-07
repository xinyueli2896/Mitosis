"""Melody -> chord MoE model with the two-pass / gated cross-stream
attention (mel-only-keys pass + chord-only-keys pass, mixed via per-token
sigmoid gate). The global transformer layer is the MoE-enabled RoFormer from
the vendored fork at transformers_roformer_moe/.

Run from midi_yinyang/, e.g.:

    python cp_transformer_m2c_moe.py \
        --batch_size 8 \
        --model_size small \
        --path_to_dataset data/pop909_chord_cp4_v2.pt \
        --model_name m2c_moe_v1 \
        --moe_num_experts 4 --moe_topk 2

(The dataset path is the *chord* .pt file; the DataLoader also loads the
matching melody .pt by string-replacing 'chord' -> 'melody', so the two
files must be named consistently.)
"""

import numpy as np
import pickle
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_TRANSFORMERS_MOE_ROOT = os.path.join(os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _TRANSFORMERS_MOE_ROOT not in sys.path:
    sys.path.insert(0, _TRANSFORMERS_MOE_ROOT)

from transformers.models.roformer.modeling_roformer import RoFormerModel, RoFormerConfig, RoFormerEncoder, RoFormerMoE
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pdb
import wandb
from pytorch_lightning.loggers import WandbLogger
from typing import Optional
import argparse
import pretty_midi
# DURATION_TEMPLATES is defined in the existing preprocess script.
from preprocess_large_midi_dataset import DURATION_TEMPLATES

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
def decode_output(outputs, save_path, tempo=120.0, prompt=True, single=False, tokenizer=None):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4
    if tokenizer is None:
        tokenizer = CPTokenizer()
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    for output in outputs:
        instrument_map = {}
        for time_step, data in enumerate(output):
            content = data.squeeze(0)
            time_step = time_step if single else time_step//2
            start_time = time_step * time_step_length
            for i in range(0, len(content), 2):
                program = int(content[i].item())
                if program == tokenizer.eos_token:
                    break
                if i + 1 >= len(content):
                    print('Incomplete note @', time_step, i)
                    break
                pitch_duration = int(content[i+1].item())
                pitch_duration = pitch_duration - 2
                pitch = pitch_duration % 128
                duration = pitch_duration // 128
                if program != 0 and program != 1:
                    print('Invalid program:', program, '@', time_step, i)
                    break
                if pitch < 0 or pitch >= 128:
                    print('Invalid pitch:', pitch, '@', time_step, i)
                    break
                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print('Invalid duration:', duration, '@', time_step, i)
                    break
                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
                if program not in instrument_map:
                    if program == 0:
                        inst = pretty_midi.Instrument(program=24, name='Guitar')
                    elif program == 1:  # program == 1
                        inst = pretty_midi.Instrument(program=0, name='Piano')
                    instrument_map[program] = inst
                    midi.instruments.append(inst)

                inst = instrument_map[program]
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)

def decode_tokenized_batch(
    outputs: torch.Tensor,
    save_path: str,
    tokenizer: "CPTokenizer",
    with_velocity: bool,
    pitch_shift: Optional[torch.Tensor] = None,
    tempo: float = 120.0,
):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4
    instrument_map = {}
    if outputs.dim() != 2:
        raise ValueError("Expected outputs with shape [seq_len, subseq_len].")
    for time_step, data in enumerate(outputs):
        content = data.squeeze(0)
        start_time = time_step * time_step_length
        for i in range(0, len(content), 2):
            program_token = int(content[i].item())
            if program_token == tokenizer.eos_token:
                break
            if program_token == tokenizer.pad_token:
                continue
            if i + 1 >= len(content):
                break
            pitch_duration = int(content[i + 1].item())
            if pitch_duration == tokenizer.pad_token:
                continue
            program = program_token
            shift = int(pitch_shift.item()) if pitch_shift is not None else 0
            if pitch_shift is not None and program != 127:
                pitch_duration -= shift
            pitch = pitch_duration % 128
            duration = pitch_duration // 128
            duration -= 16 if with_velocity else 1
            if pitch < 0 or pitch >= 128:
                continue
            if duration < 0 or duration >= len(DURATION_TEMPLATES):
                continue
            if program < 0 or program >= 128:
                continue
            is_drum = program == 127
            if program not in instrument_map:
                instrument_map[program] = pretty_midi.Instrument(
                    program=program,
                    is_drum=is_drum,
                )
                midi.instruments.append(instrument_map[program])
            end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
            instrument_map[program].notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
            )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)

TRAIN_LENGTH = 384
MAX_STEPS = 1000000

# Indicator: 0
# pitch+duration*2: 3200 (25*128)
N_NORMAL_TOKENS = 3202
N_TOKENS = N_NORMAL_TOKENS + 3
SOS_TOKEN = N_NORMAL_TOKENS
EOS_TOKEN = N_NORMAL_TOKENS + 1
PAD_TOKEN = N_NORMAL_TOKENS + 2

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _robust_load(path):
    """Attempt to load tensors with weights_only, fallback to standard load on failure."""
    with open(path, 'rb') as f:
        prefix = f.read(64)
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise RuntimeError(
            f"File {path} appears to be a Git LFS pointer. Fetch actual data via `git lfs pull`."
        )
    try:
        return torch.load(path, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError, EOFError):
        return torch.load(path, weights_only=False)

class RoFormerSymbolicTransformer(L.LightningModule):

    def __init__(
        self,
        large: bool = False,
        with_velocity: bool = False,
        mel_loss_weight: float = 1.0,
        acc_loss_weight: float = 3.0,
        moe_num_experts: int = 4,
        moe_topk: int = 2,
        moe_intermediate_size: Optional[int] = None,
        global_num_layers: Optional[int] = None,
        global_dropout: float = 0.0,
        preserve_program: bool = False,
        min_acc_tokens_before_eos: int = 0,
    ):
        super().__init__()
        # When True, preprocess() keeps the actual per-note program in the
        # a-slot token instead of hardcoding 24 (mel) / 0 (chord). Needed
        # for LAMD-style multi-instrument streams; for POP909-style single-
        # instrument mel + chord the hardcode is harmless. Default False
        # for backward compat with existing trained ckpts.
        self.preserve_program = preserve_program

        # ------------------------------------------------------------
        # Core sizes
        # ------------------------------------------------------------
        self.with_velocity = with_velocity
        self.hidden_size = 768 if large else 512
        self.num_attention_heads = 12 if large else 8
        self.intermediate_size = 3072 if large else 1024
        # Mirror the depth schedule of cp_transformer.py: size 0 (small) -> 6,
        # size 1 (large) -> 12. Override via the constructor for non-defaults.
        if global_num_layers is None:
            global_num_layers = 12 if large else 6
        self.global_num_layers = global_num_layers

        # Local encoder/decoder sizes
        self.local_model_num_layers = 3
        self.local_model_num_attention_heads = 8
        self.local_model_intermediate_size = 768

        # ------------------------------------------------------------
        # Tokenizer & embeddings
        # ------------------------------------------------------------
        self.tokenizer = CPTokenizer(with_velocity=with_velocity)

        self.local_embedding = nn.Embedding(
            self.tokenizer.n_tokens,
            self.hidden_size
        )

        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)

        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)

        # ------------------------------------------------------------
        # Local RoFormer encoder/decoder (NO MoE here)
        # ------------------------------------------------------------
        local_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            moe=False,  # local model is dense
        )

        self.local_encoder = RoFormerEncoder(local_config)
        self.local_decoder = RoFormerEncoder(local_config)

        self.final_decoder = nn.Linear(
            self.hidden_size,
            self.tokenizer.n_tokens
        )

        # ------------------------------------------------------------
        # Global RoFormer (MoE INSIDE transformer layers)
        # ------------------------------------------------------------
        global_moe_intermediate = moe_intermediate_size or self.intermediate_size
        global_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=global_num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=global_moe_intermediate,
            hidden_act="gelu",
            hidden_dropout_prob=global_dropout,
            attention_probs_dropout_prob=global_dropout,
            moe=True,
            num_experts=moe_num_experts,
            topk=moe_topk,
            expert_capacity_factor=1.0,
            log_moe_stats=True,
            max_position_embeddings=4096,
        )

        self.global_roformer = RoFormerEncoder(global_config)

        # ------------------------------------------------------------
        # Global SOS token
        # ------------------------------------------------------------
        self.global_sos = nn.Parameter(
            torch.randn(self.hidden_size)
        )

        # ------------------------------------------------------------
        # Cross-stream gating
        # ------------------------------------------------------------
        self.gate_m = nn.Linear(self.hidden_size, 1)
        self.gate_c = nn.Linear(self.hidden_size, 1)

        # ------------------------------------------------------------
        # Autoregressive mask buffer (local decoder)
        # ------------------------------------------------------------
        self._future_mask = torch.empty(0)

        # ------------------------------------------------------------
        # Loss weights
        # ------------------------------------------------------------
        self.mel_loss_weight = mel_loss_weight
        self.acc_loss_weight = acc_loss_weight

        # ------------------------------------------------------------
        # Optional MIDI dumping
        # ------------------------------------------------------------
        self.sample_dump_dir = os.environ.get("DUMP_TRAIN_SAMPLES_DIR")
        self.sample_dump_every = int(os.environ.get("DUMP_TRAIN_SAMPLES_EVERY", "500"))
        self.sample_dump_max = int(os.environ.get("DUMP_TRAIN_SAMPLES_MAX", "4"))
        self._sample_dumped = 0
        self._last_dump_step = -1

        # Prevent chord sampler from ending too early
        # Prevents the chord/non-drum sampler from emitting EOS in the first
        # N tokens of a frame. Default 0 = no constraint. Used to be hardcoded
        # to 64, which was a silent no-op at cp16 (subseq_len <= 64 disables
        # it) and could surprise you at cp32+ by forcing overlong frames.
        # Pass a positive int via the model ctor if you want the legacy
        # behavior; otherwise the model is free to emit EOS as soon as it
        # learns to.
        self.min_acc_tokens_before_eos = min_acc_tokens_before_eos
    def _timestep_causal_mask_interleaved(self, L, device, dtype=torch.float32):
        """
        L = 2T interleaved tokens: [m0,c0,m1,c1,...]
        Allow attending to keys whose timestep <= query timestep.
        timestep(i) = i//2
        """
        idx = torch.arange(L, device=device)
        t = idx // 2
        mask_bool = t[None, :] > t[:, None]
        mask = mask_bool.to(dtype) * float("-inf")
        # mask = torch.where(
        #     t[None, :] > t[:, None],
        #     torch.tensor(float("-inf"), device=device, dtype=dtype),
        #     torch.tensor(0.0, device=device, dtype=dtype),
        # )
        return mask  # [L, L] additive mask


    def _restrict_keys(self, base_mask, keep_keys_bool):
        """
        base_mask: [L, L] additive
        keep_keys_bool: [L] bool mask. Keys where keep_keys_bool=False are blocked.
        """
        m = base_mask.clone()
        m[:, ~keep_keys_bool] = float("-inf")
        return m
    def get_base_model(self, config):
        return RoFormerEncoder(config)

    def local_encode(self, x, token_type_ids):

        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)

        # prepend SOS:
        x = torch.cat(
            [
                torch.full((x.shape[0], 1), self.tokenizer.sos_token, dtype=torch.long, device=x.device),
                x,
            ],
            dim=-1,
        )  # now [B*seq_len, subseq_len+1]

        mask = x != self.tokenizer.pad_token     # [B*seq_len, subseq_len+1]
        word_emb = self.local_embedding(x)  # → [B*seq_len, subseq_len+1, H]

        type_emb = self.token_type_embeddings(token_type_ids)      # [B*seq_len, subseq_len+1, H]
        type_emb = type_emb.view(batch_size*seq_len, word_emb.shape[1], -1)

        emb = word_emb + type_emb
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]

        return h[:, 0], emb[:, :-1]

    def local_decode(self, h, emb):
        batch_size, subseq_len, _ = emb.shape
        # Add h as the first token of emb
        h = h.contiguous().view(batch_size, 1, -1)
        # print(emb.shape) #3840 8 512
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        # Create an autoregressive mask
        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)


    def local_sampling(
        self,
        h,
        max_subseq_len: int = 32,
        temperature: float = 1.0,
        token_type_id: int = 1,   # 0=melody, 1=chord
    ):
        batch_size, _ = h.shape
        device = h.device
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        emb = h[:, None, :]   # [B, 1, H]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=device)

        min_tokens_before_eos = (
            self.min_acc_tokens_before_eos if token_type_id == 1 else 0
        )
        if token_type_id == 1 and min_tokens_before_eos >= max_subseq_len:
            min_tokens_before_eos = 0

        for step in range(max_subseq_len):

            h_dec = self.local_decoder(
                emb,
                attention_mask=self.buffered_future_mask(emb),
            )[0]

            logits = self.final_decoder(h_dec[:, -1])  # [B, vocab]

            vocab_ids = torch.arange(self.tokenizer.n_tokens, device=device)

            is_program_step = (y.size(1) % 2 == 0)

            valid = torch.zeros_like(logits, dtype=torch.bool)

            if is_program_step:
                if not self.with_velocity:
                    if getattr(self, 'preserve_program', False):
                        # All 128 programs are valid a-slot tokens (the
                        # tokenizer's a-slot range is 0..127, padded out to
                        # 256 by n_normal_tokens but only 0..127 are real
                        # MIDI programs). Drum side typically samples 127;
                        # non-drum side samples 0..126. Constraining
                        # further would re-introduce the squash.
                        valid[:, 0:128] = True
                    elif token_type_id == 0:      # melody
                        valid[:, 24] = True
                    elif token_type_id == 1:    # chord
                        valid[:, 0] = True
                else:
                    valid |= (vocab_ids <= 128 * 16 - 1)
                valid[:, self.tokenizer.eos_token] = True
                if token_type_id == 1 and step < min_tokens_before_eos:
                    valid[:, self.tokenizer.eos_token] = False

            else:
                if self.with_velocity:
                    pitch_min = 128 * 16
                    pitch_max = 128 * (16 + 24) - 1
                else:
                    pitch_min = 128
                    pitch_max = 128 * 25 - 1

                valid |= (vocab_ids >= pitch_min) & (vocab_ids <= pitch_max)

            valid[:, self.tokenizer.pad_token] = False
            logits = logits.masked_fill(~valid, float("-inf"))

            if temperature == 0:
                y_next = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                probs_sum = probs.sum(dim=-1, keepdim=True)
                fallback = probs_sum.squeeze(-1) == 0
                if fallback.any():
                    probs[fallback, self.tokenizer.eos_token] = 1.0
                    probs_sum = probs.sum(dim=-1, keepdim=True)

                probs = probs / probs_sum
                y_next = torch.multinomial(probs, 1)

            y_next[eos_triggered] = self.tokenizer.pad_token
            eos_triggered = eos_triggered | (
                y_next.squeeze(1) == self.tokenizer.eos_token
            )
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                break
            type_ids = torch.full_like(y_next, token_type_id)
            emb = torch.cat(
                [
                    emb,
                    self.local_embedding(y_next)
                    + self.token_type_embeddings(type_ids),
                ],
                dim=1,
            )

        if y.size(1) < max_subseq_len:
            pad_len = max_subseq_len - y.size(1)
            y = F.pad(
                y,
                (0, pad_len),
                value=self.tokenizer.pad_token,
            )

        return y
    def _frame_types(self, seq_len, device):
        return (torch.arange(seq_len, device=device) % 2).long()

    def _key_mask(self, key_mask, dtype, device):
        seq_len = key_mask.numel()
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        mask[:, ~key_mask] = float("-inf")
        return mask

    def _causal_mask(self, T, device, dtype):
        """
        Strict timestep-level mask:
        timestep t can only attend to timesteps < t.
        """
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device, dtype=dtype),
            diagonal=1
        )

    def _global_interaction(self, h):
        """
        h: [B, 2T, H], interleaved [m0,c0,m1,c1,...]

        Returns:
            out: [B, 2T, H]
            aux_loss: scalar MoE load balancing loss
        """

        B, L, H = h.shape
        assert L % 2 == 0
        device = h.device

        idx = torch.arange(L, device=device)
        t = idx // 2

        mask_bool = t[None, :] > t[:, None]
        base_mask = torch.zeros((L, L), device=device, dtype=torch.float32)
        base_mask[mask_bool] = float("-inf")
        # Disallow same-timestep cross-stream attention; allow only self on diagonal.
        same_step = t[None, :] == t[:, None]
        diag = torch.eye(L, device=device, dtype=torch.bool)
        base_mask[same_step & ~diag] = float("-inf")

        mel_keys = (idx % 2 == 0)
        chord_keys = ~mel_keys

        mask_mel = base_mask.clone()
        mask_mel[:, ~mel_keys] = float("-inf")

        mask_chord = base_mask.clone()
        mask_chord[:, ~chord_keys] = float("-inf")
        diag_idx = torch.arange(L, device=device)
        mask_mel[diag_idx, diag_idx] = 0.0
        mask_chord[diag_idx, diag_idx] = 0.0

        # The vendored fork's RoFormerEncoder.forward doesn't accept
        # return_aux_loss and its MoE block discards the router balance
        # loss silently. Drop the kwarg and synthesize a zero aux loss;
        # the load-balancing term can be wired in later if/when the fork
        # exposes it.
        out_mel = self.global_roformer(h, attention_mask=mask_mel)
        out_chord = self.global_roformer(h, attention_mask=mask_chord)
        aux_mel = h.new_zeros(())
        aux_chord = h.new_zeros(())

        out_mel_states = out_mel.last_hidden_state
        out_chord_states = out_chord.last_hidden_state
        u_mm = out_mel_states[:, 0::2, :]
        u_cm = out_mel_states[:, 1::2, :]
        u_mc = out_chord_states[:, 0::2, :]
        u_cc = out_chord_states[:, 1::2, :]

        h_m_in = h[:, 0::2, :]
        h_c_in = h[:, 1::2, :]

        o_m = u_mm + torch.sigmoid(self.gate_m(h_m_in)) * u_mc
        o_c = u_cc + torch.sigmoid(self.gate_c(h_c_in)) * u_cm

        out = torch.zeros_like(h)
        out[:, 0::2, :] = o_m
        out[:, 1::2, :] = o_c

        aux_loss = aux_mel + aux_chord
        return out, aux_loss

    def _encode_frame(self, frame_tokens, frame_type):
        batch_size, subseq_len = frame_tokens.shape
        token_type_ids = torch.full(
            (batch_size, 1, subseq_len + 1),
            frame_type,
            dtype=torch.long,
            device=frame_tokens.device,
        )
        h, _ = self.local_encode(frame_tokens.unsqueeze(1), token_type_ids=token_type_ids)
        return h.view(batch_size, 1, -1)

    def global_sampling(self, x, max_seq_len=384, temperature=1.0):

        batch_size, seq_len, subseq_len = x.shape
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2).long()  # → [seq_len], 0 at even idx (mel), 1 at odd idx (acc)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        # print(token_type_ids)
        h, _= self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)
        y = [x[:, i, :] for i in range(seq_len)]  # running list of frames: mel0, acc0, mel1, acc1, ...
        total_frames_target = seq_len + max_seq_len

        for step in range(0, max_seq_len, 2):
            if step % 10 == 0:
                print('Sampling', step, '/', max_seq_len)

            # Align with training shift: predict frame t from context up to t-2.
            sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 2, 1)
            h_pad = torch.cat([h, torch.zeros_like(h[:, :2])], dim=1)
            h_in = torch.cat([sos, h_pad[:, :-2]], dim=1)
            # h_out = self._global_interaction(h_in)
            h_out, _ = self._global_interaction(h_in)
            mel_tokens = self.local_sampling(
                h_out[:, -2],
                max_subseq_len=subseq_len,
                temperature=temperature,
                token_type_id=0
            )
            chord_tokens = self.local_sampling(
                h_out[:, -1],
                max_subseq_len=subseq_len,
                temperature=temperature,
                token_type_id=1
            )

            mel_h = self._encode_frame(mel_tokens, 0)
            chord_h = self._encode_frame(chord_tokens, 1)
            h = torch.cat([h, mel_h, chord_h], dim=1)
            y.append(mel_tokens)
            y.append(chord_tokens)

            if len(y) >= total_frames_target:
                break
        return y[:total_frames_target]



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

    def forward(self, x):
        """
        Returns:
            logits: [B, 2T, subseq_len, vocab]
            aux_loss: MoE auxiliary loss
        """

        batch_size, seq_len, subseq_len = x.shape
        assert seq_len % 2 == 0

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

        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 2, 1)
        h = torch.cat([sos, h[:, :-2]], dim=1)

        h_global, aux_loss = self._global_interaction(h)

        logits = self.local_decode(h_global, emb)

        return logits, aux_loss


    def preprocess(
        self,
        x: torch.LongTensor, # melody
        pitch_shift: torch.LongTensor,
        tuple_size=4,
        y: Optional[torch.LongTensor] = None, # accompaniment
    ):

        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // tuple_size, tuple_size)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // tuple_size, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255 #pitch is 255 that need to be pad
        eos_indices = x[:, :, :, 0] == 254 #program is 254
        is_not_drum = x[:, :, :, 0] != 127
        if self.with_velocity:
            x_processed[:, :, :, 0] = x[:, :, :, 0] + 128 * (x[:, :, :, 3] // 8)  # there are 128 instruments
            x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 16) * 128 + pitch_shift[:, None, None] * is_not_drum
        else:
            # POP909 default: hardcode a-slot to 24 (mel) so the model treats
            # everything as Classical Guitar. LAMD-style multi-instrument
            # streams want the real program preserved instead.
            x_processed[:, :, :, 0] = x[:, :, :, 0] if self.preserve_program else 24
            x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 1) * 128 + pitch_shift[:, None, None] * is_not_drum
        # x_processed[:, :, :, 0] = 0 # program 不变
        # x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = self.tokenizer.pad_token
        x_processed[:, :, :, 0][eos_indices] = self.tokenizer.eos_token
        x_processed[:, :, :, 1][eos_indices] = self.tokenizer.pad_token

        if y==None:
            return x_processed.view(batch_size, seq_length, subseq_length // tuple_size * 2)
        else:
            batch_size_y, seq_length_y, subseq_length_y = y.shape
            y = y.long().view(batch_size_y, seq_length_y, subseq_length_y // tuple_size, tuple_size)
            y_processed = torch.zeros(batch_size_y, seq_length_y, subseq_length_y // tuple_size, 2, dtype=torch.long, device=y.device)
            pad_indices_y = y[:, :, :, 1] == 255 #pitch is 255 that need to be pad
            eos_indices_y = y[:, :, :, 0] == 254 #program is 254
            is_not_drum_y = y[:, :, :, 0] != 127
            if self.with_velocity:
                y_processed[:, :, :, 0] = y[:, :, :, 0] + 128 * (y[:, :, :, 3] // 8)  # there are 128 instruments
                y_processed[:, :, :, 1] = y[:, :, :, 1] + (y[:, :, :, 2] + 16) * 128 + pitch_shift[:, None, None] * is_not_drum_y
            else:
                # POP909 default: hardcode a-slot to 0 (chord). LAMD wants the
                # real program preserved -- see preserve_program note above.
                y_processed[:, :, :, 0] = y[:, :, :, 0] if self.preserve_program else 0
                y_processed[:, :, :, 1] = y[:, :, :, 1] + (y[:, :, :, 2] + 1) * 128 + pitch_shift[:, None, None] * is_not_drum_y

            y_processed[pad_indices_y] = self.tokenizer.pad_token
            y_processed[:, :, :, 0][eos_indices_y] = self.tokenizer.eos_token
            y_processed[:, :, :, 1][eos_indices_y] = self.tokenizer.pad_token

            return x_processed.view(batch_size, seq_length, subseq_length // tuple_size * 2), y_processed.view(batch_size_y, seq_length_y, subseq_length_y // tuple_size * 2)



    def loss(self, x_mel, x_acc, pitch_shift):

        x_mel, x_acc = self.preprocess(x_mel, pitch_shift, y=x_acc)

        batch_size, seq_len, subseq_len = x_mel.shape

        stacked = torch.stack([x_mel, x_acc], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        logits, aux_loss = self(x)
        targets = x

        frame_idx = torch.arange(seq_len * 2, device=targets.device)

        frame_weights = torch.where(
            frame_idx % 2 == 0,
            torch.as_tensor(self.mel_loss_weight, device=targets.device),
            torch.as_tensor(self.acc_loss_weight, device=targets.device),
        )

        weights = frame_weights.view(1, seq_len * 2, 1).expand(
            batch_size, -1, subseq_len
        )

        per_token_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.n_tokens),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token,
            reduction='none'
        ).view(batch_size, seq_len * 2, subseq_len)

        non_pad_mask = (targets != self.tokenizer.pad_token).float()
        weighted_loss = per_token_loss * weights * non_pad_mask

        normalizer = (weights * non_pad_mask).sum().clamp_min(1.0)
        ce_loss = weighted_loss.sum() / normalizer

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
            total_loss = ce_loss + 0.01 * aux_loss   # 0.01 typical MoE weight
        else:
            aux_loss = ce_loss.new_zeros(())
            total_loss = ce_loss

        return total_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('train_loss', loss)
        self.log('moe_aux_loss', aux_loss, prog_bar=False, on_step=True, on_epoch=True)
        # scheduler step
        # scheduler = self.lr_schedulers()
        # scheduler.step()
        # self.log('training/lr', scheduler.get_last_lr()[0])
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("training/lr", lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss = self.loss(*batch)
        self.log('val_loss', loss)
        self.log('val_moe_aux_loss', aux_loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        max_lr = 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step"
        }
    }

class FramedDataset(IterableDataset):

    def __init__(
        self,
        file_path,
        target_length,
        batch_size,
        split='all',
        split_ratio=10,
    ):
        self.file_path = file_path
        self.mel_path = file_path.replace('chord', 'melody')
        # Per-song lengths for both streams. Chord and melody can differ in
        # length per song (e.g. chord_midi.txt segments may extend slightly
        # past the melody track's last note), so we use min() per song as
        # the effective length and keep separate cumulative-start offsets
        # for each stream when indexing.
        self.length_chord = _robust_load(file_path[:-3] + '.length.pt')
        self.length_mel = _robust_load(self.mel_path[:-3] + '.length.pt')
        if len(self.length_chord) != len(self.length_mel):
            raise RuntimeError(
                f"chord ({len(self.length_chord)} songs) and melody "
                f"({len(self.length_mel)} songs) length files disagree on "
                "song count -- re-preprocess so they match."
            )
        self.length = torch.minimum(self.length_chord, self.length_mel)
        self.start_chord = torch.cumsum(self.length_chord, dim=0) - self.length_chord
        self.start_mel = torch.cumsum(self.length_mel, dim=0) - self.length_mel
        # Invalid samples are those whose length is less than min_length
        is_valid = self.length >= target_length
        self.valid_indices = torch.arange(len(self.length))[is_valid]
        # Get training or validation split
        if split == 'all':
            pass
        elif split == 'train':
            self.valid_indices = self.valid_indices[self.valid_indices % split_ratio != 0]
        elif split == 'val':
            self.valid_indices = self.valid_indices[self.valid_indices % split_ratio == 0]
        self.split = split
        self.valid_song_count = len(self.valid_indices)
        self.target_length = target_length
        self.batch_size = batch_size
        print('Metadata for dataset', file_path, 'loaded. Number of valid songs:', self.valid_song_count)

    def __iter__(self):
        data = _robust_load(self.file_path)        # chord stream
        data_c = _robust_load(self.mel_path)       # melody stream (yes, named data_c)
        pitch_shift_range = _robust_load(self.file_path[:-3] + '.pitch_shift_range.pt').reshape(-1, 2)
        pitch_shift_range[pitch_shift_range[:, 0] < -5, 0] = -5
        pitch_shift_range[pitch_shift_range[:, 1] > 6, 1] = 6
        pitch_shift_range_c = _robust_load(self.mel_path[:-3] + '.pitch_shift_range.pt').reshape(-1, 2)
        pitch_shift_range_c[pitch_shift_range_c[:, 0] < -5, 0] = -5
        pitch_shift_range_c[pitch_shift_range_c[:, 1] > 6, 1] = 6
        if self.split == 'val':
            pitch_shift_range = torch.zeros_like(pitch_shift_range)  # No pitch shift for validation
            pitch_shift_range_c = torch.zeros_like(pitch_shift_range_c)  # No pitch shift for validation
        print('Data for dataset', self.file_path, 'loaded.')
        while True:
            indices = torch.randperm(len(self.valid_indices))
            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                raw_ids = self.valid_indices[batch_indices]
                # Sample a window offset that fits in BOTH streams (use min length).
                to_be_added = torch.floor(torch.rand(len(raw_ids)) * (self.length[raw_ids] - self.target_length)).long()
                to_be_added -= to_be_added % 16
                # Per-stream absolute starts: same window offset, different cumulative bases.
                starts_chord = to_be_added + self.start_chord[raw_ids]
                starts_mel = to_be_added + self.start_mel[raw_ids]
                offset_grid = torch.arange(self.target_length).view(1, -1)
                idx_chord = offset_grid + starts_chord.view(-1, 1)
                idx_mel = offset_grid + starts_mel.view(-1, 1)
                batch_pitch_shift_range = pitch_shift_range[raw_ids]
                batch_pitch_shift_range_c = pitch_shift_range_c[raw_ids]
                minmax = torch.minimum(batch_pitch_shift_range_c[:, 1], batch_pitch_shift_range[:, 1])
                maxmin = torch.maximum(batch_pitch_shift_range[:, 0], batch_pitch_shift_range_c[:, 0])
                valid = minmax >= maxmin
                span = (minmax - maxmin + 1).clamp_min(1)
                batch_pitch_shift = torch.floor(torch.rand(len(raw_ids)) * span).long() + maxmin
                batch_pitch_shift = torch.where(valid, batch_pitch_shift, torch.zeros_like(batch_pitch_shift))
                yield data_c[idx_mel], data[idx_chord], batch_pitch_shift



def sanity_check():
    # Do some sanity check
    model = RoFormerSymbolicTransformer()
    tokenizer = model.tokenizer
    x = [
        [
            [0 , 34, 12, tokenizer.eos_token, tokenizer.pad_token, tokenizer.pad_token],
            [12, 34, 52, 34, 12, tokenizer.eos_token],
            [12, 34, 52, 34, 12, 52]
        ],
        [
            [52, 34, 12, tokenizer.eos_token, tokenizer.pad_token, tokenizer.pad_token],
            [12, 34, 52, 34, 12, tokenizer.eos_token],
            [tokenizer.eos_token, tokenizer.pad_token, tokenizer.pad_token, tokenizer.pad_token, tokenizer.pad_token, tokenizer.pad_token]
        ]
    ]
    y = [
        [
            [42, 34, 12, tokenizer.eos_token, tokenizer.pad_token, tokenizer.pad_token],
            [22, 34, 52, 34, 12, tokenizer.eos_token],
            [22, 34, 52, 34, 12, 52]
        ],
        [
            [42, 34, 12, tokenizer.eos_token, tokenizer.pad_token, tokenizer.pad_token],
            [22, 34, 52, 34, 12, tokenizer.eos_token],
            [tokenizer.eos_token, tokenizer.pad_token, tokenizer.pad_token, tokenizer.pad_token, tokenizer.pad_token, tokenizer.pad_token]
        ]
    ]
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    loss = model.loss(x, y, pitch_shift = torch.zeros(1, dtype=torch.int8))
    # print(loss)

if __name__ == '__main__':
    # sanity_check()

    parser = argparse.ArgumentParser(description="process midi folder(s) into usable tensors for the task")

    parser.add_argument("--batch_size",type=int, default=10, help="batch size")
    parser.add_argument("--model_size", type=str, default='small', help="large or small")
    parser.add_argument("--path_to_dataset",type=str,help="name to your accompaniment .pt file")
    parser.add_argument("--model_name",type=str,default=None, help="name to your model")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="continue training from checkpoint")
    parser.add_argument("--wandb", action="store_true",  default=False, help="enable wand logging")
    parser.add_argument("--moe_num_experts", type=int, default=4, help="number of experts for interaction MoE")
    parser.add_argument("--moe_topk", type=int, default=2, help="top-k experts per token for interaction MoE")
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=None,
        help="override MoE intermediate size (defaults to global intermediate size)",
    )
    parser.add_argument(
        "--global_num_layers",
        type=int,
        default=None,
        help="number of global transformer layers. Default: 12 if model_size=large "
             "else 6 (mirrors cp_transformer.py's depth schedule). Pass "
             "--global_num_layers 1 to load a legacy single-layer checkpoint.",
    )
    parser.add_argument(
        "--preserve_program", action="store_true", default=False,
        help="Preserve actual per-note program in the a-slot token. Use for "
             "LAMD-style multi-instrument streams. Default False (POP909-style "
             "hardcode of program 24 / 0).",
    )


    args = parser.parse_args()

    batch_size = args.batch_size
    model_size = args.model_size
    dataset = args.path_to_dataset
    checkpoint_path = args.checkpoint_path

    with_velocity = False
    # if model_size < 0:  # with velocity
    #     model_size = -model_size - 1
    #     with_velocity = True
    # assert model_size in [0, 1, 2, 3]
    # gradient_clip = 1.0 if model_size >= 2 else None
    # max_lr = 5e-5 if model_size >= 2 else 1e-4
    n_gpus = max(torch.cuda.device_count(), 1)
    suffix = 'vel' if with_velocity else ''

    gnl = args.global_num_layers
    if gnl is None:
        gnl = 12 if model_size == 'large' else 6

    default_name = (f"m2c_transformer_moe_v4.0_{model_size}_gnl{gnl}"
                    f"_batch_{batch_size * n_gpus}_schedule")
    model_name = args.model_name if args.model_name is not None else default_name
    net = RoFormerSymbolicTransformer(
        model_size == 'large',
        with_velocity=with_velocity,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=gnl,
        preserve_program=args.preserve_program,
    )
    print(f"MoE enabled: {net.global_roformer.config.moe}")
    train_set_loader = DataLoader(FramedDataset(dataset, TRAIN_LENGTH, batch_size, split = 'train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(dataset, TRAIN_LENGTH, batch_size, split = 'val'), batch_size=None, num_workers=1, persistent_workers=True)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_loss',
                                                      save_top_k=5,
                                                      save_last=True,
                                                      enable_version_counter=False,
                                                      dirpath=f'ckpt/{model_name}',
                                                      filename=model_name + '.{epoch:02d}.{val_loss:.5f}')
    if n_gpus > 1:
        import pytorch_lightning.strategies as strategies
        import datetime
        strategy = strategies.DDPStrategy(timeout=datetime.timedelta(hours=2))
    else:
        strategy = 'auto'
    trainer = L.Trainer(devices=n_gpus,
                        precision="bf16-mixed" if torch.cuda.is_available() else 32,
                        max_steps=MAX_STEPS,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        callbacks=[checkpoint_callback],
                        val_check_interval=500,
                        limit_val_batches=25,
                        check_val_every_n_epoch=None,
                        logger= WandbLogger(name=model_name, project="MusicMOE",
                                config={
                                    "batch_size": batch_size,
                                    "model_size": model_size,
                                    "train_length": TRAIN_LENGTH
                                }) if args.wandb else TensorBoardLogger("tb_logs", name=model_name),
                        num_sanity_val_steps=0 if checkpoint_path is not None else 2,
                        strategy=strategy)
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "hyper_parameters": {
                "model_size": model_size,
                "with_velocity": with_velocity,
                "moe_num_experts": args.moe_num_experts,
                "moe_topk": args.moe_topk,
                "moe_intermediate_size": args.moe_intermediate_size,
                "global_num_layers": gnl,
            },
        },
        f'ckpt/{model_name}.pt',
    )
