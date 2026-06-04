"""Inference for M2CMixtureHead: per-layer fusion + joint mixture head.

5 modes via marginal / posterior on the mixture index k:

  co        : sample k from prior pi(h_m, h_c), then m_block and c_block
              from the k-th component, in parallel.
  mel2chord : posterior p(k | observed m_block) proportional to
              pi_k * p_k(m_block); sample k from posterior, then c_block.
  chord2mel : symmetric.
  mel_only  : sample k from prior, then m_block from p_k. Chord ignored.
  chord_only: symmetric.

Within-block sampling is AR with CP-structure constraints (program / pitch
alternation, valid ranges) -- same machinery as the working AR inference,
just routed through the k-th component's final projection in the joint
mixture head.

Run:
    python cp_transformer_m2c_mixture_head_inference.py \\
        --mode co \\
        --ckpt ckpt/<run>/last.ckpt \\
        --melody POP909-Dataset/POP909-melody/001.mid \\
        --chord  POP909-Dataset/POP909-chord/001.mid \\
        --prompt-length 100 --gen-length 384 \\
        --model-size large
"""

import os as _os
import sys as _sys
_MOE_ROOT = _os.path.join(_os.path.dirname(__file__), "transformers_roformer_moe", "src")
if _MOE_ROOT not in _sys.path:
    _sys.path.insert(0, _MOE_ROOT)

import argparse
import os
import re
from glob import glob

import torch
import torch.nn.functional as F

from cp_transformer_m2c_mixture_head import M2CMixtureHead


MODES = ['co', 'mel2chord', 'chord2mel', 'mel_only', 'chord_only']


# ---------------------------------------------------------------------------
# Run global tower on current buffer; get h_m_pred, h_c_pred at next slot
# ---------------------------------------------------------------------------

@torch.no_grad()
def step_global(model, h_buffer):
    """Run the per-layer fusion stack on (SOS_pair + h_buffer) and return
    the LAST two positions, which are the prediction states for the next
    (m_t, c_t) under shift-by-2.

    h_buffer: [B, 2t, H] encoded past blocks (interleaved).
    Returns: h_m_pred [B, H], h_c_pred [B, H].
    """
    B = h_buffer.shape[0]
    sos = model.global_sos.view(1, 1, -1).expand(B, 2, -1)
    h_in = torch.cat([sos, h_buffer], dim=1)  # [B, 2(t+1), H]
    h_global, _ = model._global_interaction(h_in)
    return h_global[:, -2], h_global[:, -1]


# ---------------------------------------------------------------------------
# Per-component log-likelihood for an observed block (used in posterior modes)
# ---------------------------------------------------------------------------

@torch.no_grad()
def block_logp_per_k(model, h_pred, block_tokens, modality):
    """Compute log p_k(block_tokens | h_pred) for every mixture component k.

    h_pred: [B, H] -- global prediction state for this block.
    block_tokens: [B, subseq] -- observed CP tokens.
    modality: 'm' or 'c'.
    Returns: [B, K].
    """
    B = h_pred.shape[0]
    subseq = block_tokens.shape[1]
    device = h_pred.device
    token_type_id = 0 if modality == 'm' else 1

    type_ids = torch.full(
        (B, 1, subseq + 1), token_type_id,
        dtype=torch.long, device=device,
    )
    _, emb = model.local_encode(block_tokens.unsqueeze(1), type_ids)
    # emb: [B, subseq, H]

    h_dec = model.local_decode_to_hidden(h_pred, emb)  # [B, subseq, H]
    if modality == 'm':
        logits_per_k = model.joint_mixture_head.per_k_logits_m(h_dec)
    else:
        logits_per_k = model.joint_mixture_head.per_k_logits_c(h_dec)
    # logits_per_k: [B, subseq, K, V]

    log_probs = F.log_softmax(logits_per_k.float(), dim=-1)
    target = block_tokens.unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, model.mixture_K, 1,
    )
    log_p_target = log_probs.gather(-1, target).squeeze(-1)  # [B, subseq, K]

    pad = model.tokenizer.pad_token
    mask = (block_tokens != pad).float().unsqueeze(-1)
    log_p_target = log_p_target * mask
    return log_p_target.sum(dim=1)  # [B, K]


# ---------------------------------------------------------------------------
# Within-block AR sampling using a specific mixture component k per batch item
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_block_given_k(model, h_pred, k_per_batch, modality, temperature,
                          max_subseq_len):
    """Within-block CP-structured AR sampling using the k-th component's
    final projection. Same CP token-type constraints as the AR baseline's
    local_sampling.

    h_pred: [B, H]
    k_per_batch: [B] -- which mixture component per batch element.
    Returns: [B, max_subseq_len] padded with pad_token after EOS.
    """
    B = h_pred.shape[0]
    device = h_pred.device
    token_type_id = 0 if modality == 'm' else 1
    head = (model.joint_mixture_head.per_k_logits_m if modality == 'm'
            else model.joint_mixture_head.per_k_logits_c)
    vocab_size = (model.joint_mixture_head.vocab_size_m if modality == 'm'
                  else model.joint_mixture_head.vocab_size_c)

    y = torch.zeros((B, 0), dtype=torch.long, device=device)
    emb = h_pred[:, None, :]
    eos_triggered = torch.zeros(B, dtype=torch.bool, device=device)

    min_tokens_before_eos = (
        model.min_acc_tokens_before_eos if token_type_id == 1 else 0
    )
    if token_type_id == 1 and min_tokens_before_eos >= max_subseq_len:
        min_tokens_before_eos = 0

    for step in range(max_subseq_len):
        h_dec = model.local_decoder(
            emb, attention_mask=model.buffered_future_mask(emb),
        )[0]
        last_hidden = h_dec[:, -1]  # [B, H]
        all_k_logits = head(last_hidden)  # [B, K, V]
        k_idx = k_per_batch[:, None, None].expand(-1, 1, vocab_size)
        logits = all_k_logits.gather(1, k_idx).squeeze(1)  # [B, V]

        vocab_ids = torch.arange(logits.shape[-1], device=device)
        is_program_step = (y.size(1) % 2 == 0)
        valid = torch.zeros_like(logits, dtype=torch.bool)

        if is_program_step:
            if not model.with_velocity:
                if token_type_id == 0:
                    valid[:, 24] = True
                elif token_type_id == 1:
                    valid[:, 0] = True
            else:
                valid |= (vocab_ids <= 128 * 16 - 1)
            valid[:, model.tokenizer.eos_token] = True
            if token_type_id == 1 and step < min_tokens_before_eos:
                valid[:, model.tokenizer.eos_token] = False
        else:
            if model.with_velocity:
                pitch_min = 128 * 16
                pitch_max = 128 * (16 + 24) - 1
            else:
                pitch_min = 128
                pitch_max = 128 * 25 - 1
            valid |= (vocab_ids >= pitch_min) & (vocab_ids <= pitch_max)

        valid[:, model.tokenizer.pad_token] = False
        logits = logits.masked_fill(~valid, float('-inf'))

        if temperature == 0:
            y_next = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            fallback = probs_sum.squeeze(-1) == 0
            if fallback.any():
                probs[fallback, model.tokenizer.eos_token] = 1.0
                probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / probs_sum
            y_next = torch.multinomial(probs, 1)

        y_next[eos_triggered] = model.tokenizer.pad_token
        eos_triggered = eos_triggered | (
            y_next.squeeze(1) == model.tokenizer.eos_token
        )
        y = torch.cat([y, y_next], dim=1)
        if torch.all(eos_triggered):
            break
        type_ids = torch.full_like(y_next, token_type_id)
        emb = torch.cat(
            [
                emb,
                model.local_embedding(y_next)
                + model.token_type_embeddings(type_ids),
            ],
            dim=1,
        )

    if y.size(1) < max_subseq_len:
        pad_len = max_subseq_len - y.size(1)
        y = F.pad(y, (0, pad_len), value=model.tokenizer.pad_token)
    return y


# ---------------------------------------------------------------------------
# Block-by-block AR generation across all 5 modes
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_modes(model, mode, mel_prompt, chord_prompt, prompt_length,
                   gen_length, temperature):
    """Returns (mel_frames, chord_frames), each a list of [B, subseq] tensors.

    Length conventions:
      co        : total = gen_length (prompt for first prompt_length, sample after)
      mel2chord : total = prompt_length + gen_length (mel given throughout,
                  chord prompt for first prompt_length then sampled gen_length)
      chord2mel : symmetric
      mel_only  : total = gen_length (mel prompt for prompt_length, sample after)
      chord_only: symmetric
    """
    device = next(model.parameters()).device
    H = model.hidden_size
    subseq = (mel_prompt.shape[2] if mel_prompt is not None
              else chord_prompt.shape[2])
    B = (mel_prompt.shape[0] if mel_prompt is not None
         else chord_prompt.shape[0])

    if mode == 'co':
        assert mel_prompt is not None and chord_prompt is not None
        total_T = gen_length
        usable_mel = min(prompt_length, mel_prompt.shape[1])
        usable_chord = min(prompt_length, chord_prompt.shape[1])
    elif mode == 'mel2chord':
        assert mel_prompt is not None
        total_T = prompt_length + gen_length
        total_T = min(total_T, mel_prompt.shape[1])
        usable_mel = total_T
        usable_chord = (min(prompt_length, chord_prompt.shape[1])
                        if chord_prompt is not None else 0)
    elif mode == 'chord2mel':
        assert chord_prompt is not None
        total_T = prompt_length + gen_length
        total_T = min(total_T, chord_prompt.shape[1])
        usable_chord = total_T
        usable_mel = (min(prompt_length, mel_prompt.shape[1])
                      if mel_prompt is not None else 0)
    elif mode == 'mel_only':
        assert mel_prompt is not None
        total_T = gen_length
        usable_mel = min(prompt_length, mel_prompt.shape[1])
        usable_chord = 0
    elif mode == 'chord_only':
        assert chord_prompt is not None
        total_T = gen_length
        usable_chord = min(prompt_length, chord_prompt.shape[1])
        usable_mel = 0
    else:
        raise ValueError(mode)

    def silence_frame():
        f = torch.full((B, subseq), model.tokenizer.pad_token,
                       dtype=torch.long, device=device)
        f[:, 0] = model.tokenizer.eos_token
        return f

    h_buffer = torch.zeros(B, 0, H, device=device)
    mel_frames = []
    chord_frames = []

    for t in range(total_T):
        if t % 10 == 0:
            print(f'[mixture-head, mode={mode}] block {t}/{total_T}')

        # Get global prediction states for the next block.
        h_m_pred, h_c_pred = step_global(model, h_buffer)

        # Mixture prior.
        log_pi = model.joint_mixture_head.mixture_log_pi(
            h_m_pred.unsqueeze(1), h_c_pred.unsqueeze(1),
        )[:, 0]  # [B, K]

        if mode == 'co':
            if t < usable_mel and t < usable_chord:
                m_block = mel_prompt[:, t]
                c_block = chord_prompt[:, t]
            else:
                k = torch.multinomial(F.softmax(log_pi, dim=-1), 1).squeeze(-1)
                m_block = sample_block_given_k(
                    model, h_m_pred, k, 'm', temperature, max_subseq_len=subseq,
                )
                c_block = sample_block_given_k(
                    model, h_c_pred, k, 'c', temperature, max_subseq_len=subseq,
                )

        elif mode == 'mel2chord':
            m_block = mel_prompt[:, t]
            if t < usable_chord:
                c_block = chord_prompt[:, t]
            else:
                # Posterior over k given the observed mel block.
                log_p_m = block_logp_per_k(model, h_m_pred, m_block, 'm')
                log_post = log_pi + log_p_m
                log_post = log_post - log_post.logsumexp(dim=-1, keepdim=True)
                k = torch.multinomial(F.softmax(log_post, dim=-1), 1).squeeze(-1)
                c_block = sample_block_given_k(
                    model, h_c_pred, k, 'c', temperature, max_subseq_len=subseq,
                )

        elif mode == 'chord2mel':
            c_block = chord_prompt[:, t]
            if t < usable_mel:
                m_block = mel_prompt[:, t]
            else:
                log_p_c = block_logp_per_k(model, h_c_pred, c_block, 'c')
                log_post = log_pi + log_p_c
                log_post = log_post - log_post.logsumexp(dim=-1, keepdim=True)
                k = torch.multinomial(F.softmax(log_post, dim=-1), 1).squeeze(-1)
                m_block = sample_block_given_k(
                    model, h_m_pred, k, 'm', temperature, max_subseq_len=subseq,
                )

        elif mode == 'mel_only':
            if t < usable_mel:
                m_block = mel_prompt[:, t]
            else:
                k = torch.multinomial(F.softmax(log_pi, dim=-1), 1).squeeze(-1)
                m_block = sample_block_given_k(
                    model, h_m_pred, k, 'm', temperature, max_subseq_len=subseq,
                )
            c_block = silence_frame()

        elif mode == 'chord_only':
            if t < usable_chord:
                c_block = chord_prompt[:, t]
            else:
                k = torch.multinomial(F.softmax(log_pi, dim=-1), 1).squeeze(-1)
                c_block = sample_block_given_k(
                    model, h_c_pred, k, 'c', temperature, max_subseq_len=subseq,
                )
            m_block = silence_frame()

        mel_frames.append(m_block)
        chord_frames.append(c_block)

        # Encode the committed frames and append to the buffer for the next step.
        m_h = model._encode_frame(m_block, 0)  # [B, 1, H]
        c_h = model._encode_frame(c_block, 1)
        h_buffer = torch.cat([h_buffer, m_h, c_h], dim=1)

    return mel_frames, chord_frames


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _infer_global_num_layers(ckpt_path, ck, model_size):
    if isinstance(ck, dict):
        hp = ck.get('hyper_parameters') or {}
        if isinstance(hp, dict) and hp.get('global_num_layers') is not None:
            return int(hp['global_num_layers']), 'hyper_parameters'
    m = re.search(r'_gnl(\d+)_', os.path.basename(ckpt_path))
    if m:
        return int(m.group(1)), 'filename'
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    if isinstance(state, dict):
        idxs = set()
        pat = re.compile(r'fusion_blocks\.(\d+)\.')
        for k in state.keys():
            mm = pat.search(k)
            if mm:
                idxs.add(int(mm.group(1)))
        if idxs:
            return max(idxs) + 1, 'state_dict_inference'
    return (12 if model_size == 'large' else 6), 'size_default'


def _infer_mixture_K(ckpt_path, ck):
    if isinstance(ck, dict):
        hp = ck.get('hyper_parameters') or {}
        if isinstance(hp, dict) and hp.get('mixture_K') is not None:
            return int(hp['mixture_K']), 'hyper_parameters'
    m = re.search(r'_K(\d+)_', os.path.basename(ckpt_path))
    if m:
        return int(m.group(1)), 'filename'
    return 8, 'default'


def load_model(ckpt_path, model_size='small', moe_num_experts=4, moe_topk=2,
               moe_intermediate_size=None, global_num_layers=None,
               mixture_K=None):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if global_num_layers is None:
        global_num_layers, src = _infer_global_num_layers(
            ckpt_path, ck, model_size,
        )
        print(f'[load_model] global_num_layers={global_num_layers} '
              f'(auto from {src})')
    if mixture_K is None:
        mixture_K, src = _infer_mixture_K(ckpt_path, ck)
        print(f'[load_model] mixture_K={mixture_K} (auto from {src})')

    net = M2CMixtureHead(
        large=(model_size == 'large'),
        with_velocity=False,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_intermediate_size=moe_intermediate_size,
        global_num_layers=global_num_layers,
        mixture_K=mixture_K,
    )
    state = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[load_model] missing: {missing[:5]}'
              f'{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[load_model] unexpected: {unexpected[:5]}'
              f'{"..." if len(unexpected) > 5 else ""}')
    return net


# ---------------------------------------------------------------------------
# Per-pair runner + CLI
# ---------------------------------------------------------------------------

def _load_prompt_tokens(model, midi_path, max_polyphony):
    from cp_transformer_m2c_moe_inference import _load_prompt_tokens as f
    return f(model, midi_path, max_polyphony)


def _decode_to_midi(model, mel_frames, chord_frames, save_path,
                    write_mel=True, write_chord=True):
    from cp_transformer_m2c_moe_inference import decode_m2c_frames
    decode_m2c_frames(
        mel_frames, chord_frames,
        save_path=save_path,
        tokenizer=model.tokenizer,
        with_velocity=model.with_velocity,
        write_mel=write_mel,
        write_chord=write_chord,
    )


def run_one(model, mode, mel_path, chord_path, args, out_subdir):
    mel_prompt = (_load_prompt_tokens(model, mel_path, args.max_polyphony)
                  if mel_path else None)
    chord_prompt = (_load_prompt_tokens(model, chord_path, args.max_polyphony)
                    if chord_path else None)

    mel_frames, chord_frames = generate_modes(
        model, mode, mel_prompt, chord_prompt,
        prompt_length=args.prompt_length,
        gen_length=args.gen_length,
        temperature=args.temperature,
    )

    save_name = getattr(model, 'save_name', 'm2c_mixture_head')
    out_dir = os.path.join(f'temp/{save_name}', out_subdir, mode)
    write_mel = mode != 'chord_only'
    write_chord = mode != 'mel_only'
    _decode_to_midi(
        model, mel_frames, chord_frames,
        save_path=os.path.join(
            out_dir,
            f'sample_temp{args.temperature}.mid',
        ),
        write_mel=write_mel,
        write_chord=write_chord,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--melody', help='single melody midi')
    p.add_argument('--chord', help='single chord midi')
    p.add_argument('--mel-folder')
    p.add_argument('--chord-folder')
    p.add_argument('--prompt-length', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-polyphony', type=int, default=4)
    p.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'large'])
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--moe-intermediate-size', type=int, default=None)
    p.add_argument('--global-num-layers', type=int, default=None)
    p.add_argument('--mixture-K', type=int, default=None,
                   help='Auto-detected from ckpt if omitted.')
    args = p.parse_args()

    if args.mel_folder or args.chord_folder:
        if args.melody or args.chord:
            p.error('--melody/--chord cannot be combined with folder args')

    model = load_model(
        args.ckpt,
        model_size=args.model_size,
        moe_num_experts=args.moe_num_experts,
        moe_topk=args.moe_topk,
        moe_intermediate_size=args.moe_intermediate_size,
        global_num_layers=args.global_num_layers,
        mixture_K=args.mixture_K,
    )
    model.save_name = os.path.basename(args.ckpt)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if args.mel_folder or args.chord_folder:
        mel_files = sorted(
            p for p in glob(os.path.join(args.mel_folder or '', '*'))
            if p.lower().endswith(('.mid', '.midi'))
        )
        chord_index = {os.path.basename(q): q for q in sorted(
            p for p in glob(os.path.join(args.chord_folder or '', '*'))
            if p.lower().endswith(('.mid', '.midi'))
        )}
        for mel_path in mel_files:
            chord_path = chord_index.get(os.path.basename(mel_path))
            if chord_path is None:
                continue
            sid = os.path.splitext(os.path.basename(mel_path))[0]
            try:
                run_one(model, args.mode, mel_path, chord_path, args, sid)
            except Exception as e:
                print(f'  failed: {e!r}')
    else:
        sid = os.path.splitext(os.path.basename(
            args.melody if args.melody else args.chord
        ))[0]
        run_one(model, args.mode, args.melody, args.chord, args, sid)


if __name__ == '__main__':
    main()
