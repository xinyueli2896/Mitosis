"""Does the model USE its query slots, or has it learned to ignore them?

The A.7 post-mortem hypothesis: a PARAMETRIC corruption (time
displacement, and pitch shift would be the same) can be undone by a
shortcut -- "detect that this content is displaced, discard it" -- so
the model reaches low query loss by IGNORING the slot rather than by
learning to coordinate. If true, the same-instant channel carries
nothing at decode, which is what coupling below the no-slot control
looks like.

This probe measures that directly, with no training and no decoding.
Over the SAME validation batches it runs the model on several query
frames, changing only what the two query slots contain, and reads the
query logits. Conditions are named by WHICH SLOT is filled,
not by whose point of view it is -- one condition serves both streams,
because filling the melody slot is the melody's OWN channel and the
chord's PARTNER channel at the same time:

  ctx        both slots masked      -> prediction from history alone
  mel_true   melody slot = its true frame
  chd_true   chord  slot = its true frame
  mel_decoy  melody slot = a DISPLACED real melody frame (t + lag)
  chd_decoy  chord  slot = a DISPLACED real chord  frame (t + lag)
  mel_shift  melody slot = a TRANSPOSED real melody frame
  chd_shift  chord  slot = a TRANSPOSED real chord frame

Per stream, read off the melody column under mel_* (own) and under
chd_* (partner), and the chord column the other way round:

  OWN GAIN      = CE(ctx) - CE(own slot filled with its true frame)
      the sanity floor. Must be large -- if it is not, the injection
      path is broken and nothing else here means anything.
  PARTNER GAIN  = CE(ctx) - CE(partner slot filled with its true frame)
      how much filling the partner slot with the truth improves this
      stream's prediction. NOT the coordination channel on its own --
      it conflates two effects, so it is split:
  CONTENT  = CE(partner decoy) - CE(partner true)
      the true partner against a WRONG partner that is equally legal
      and equally present. This is the only part that can carry
      coordination: it is what the model gains from knowing WHICH
      frame the partner played. Near zero => the channel is dead even
      if PARTNER GAIN looks positive.
  presence = PARTNER GAIN - CONTENT
      the remainder: the model preferring any real frame in that slot
      over the mask token. Costs nothing to produce at decode and
      carries no information about the partner.
  DECOY SENSITIVITY = divergence(partner decoy, ctx)
      how far a wrong-but-legal partner moves the prediction. Near zero
      => the model ignores slot content it judges untrustworthy (the
      shortcut story). Large, with CE worse than ctx => it is misled
      instead (the opposite story).

CONTENT is reported both raw and as a fraction of OWN GAIN. The raw
value depends on how predictable the corpus is; the fraction is what
survives comparing two models trained on different corpora, so prefer
it when the checkpoints are not corpus-matched.

Slot contents are injected through the model's own self-conditioning
override (k=0 plus sc_mask/sc_emb), which both the A.3 and A.7 forwards
honour, so nothing about the model is monkey-patched. Models run in
eval mode, where A.7's decoy branch is inactive by construction -- the
probe measures LEARNED BEHAVIOUR under controlled inputs, not the
training-time kernel.

Usage (via probe_slot_use.sbatch):
    python probe_slot_use.py --ckpt A3=ckpt/<run>/ --ckpt A7=ckpt/<run>/
"""

import argparse
import os
import tempfile

import torch
import torch.nn.functional as F

from cp_transformer_m2c_moe import FramedDataset, TRAIN_LENGTH
from cp_transformer_m2c_duet_block_diffusion_inference import load_model
from tasks import get_task


def collect_batches(task, batch_size, n_batches):
    val = FramedDataset(task.mod_b_path, TRAIN_LENGTH, batch_size,
                        split='val', mel_path=task.mod_a_path)
    loader = torch.utils.data.DataLoader(val, batch_size=None, num_workers=0)
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        out.append(b)
    return out


def interleave(net, batch, device):
    """Batch -> the interleaved, preprocessed tensor forward() expects."""
    x_mel, x_acc, pitch_shift = (t.to(device) if torch.is_tensor(t) else t
                                 for t in batch)
    x_mel, x_acc = net.preprocess(x_mel, pitch_shift, y=x_acc)
    B, T, S = x_mel.shape
    x = torch.stack([x_mel, x_acc], dim=2).view(B, T * 2, S)
    return x, T, S


def transpose_frame(net, tokens, semitones):
    """Shift the pitch-dur tokens of a frame, keeping programs/EOS/pad.

    pitch-dur ids are pitch + (dur+1)*128, so adding n semitones is
    +n on the id, clamped inside the pitch-dur range.
    """
    tok = net.tokenizer
    out = tokens.clone()
    S = tokens.shape[1]
    is_pd = (torch.arange(S, device=tokens.device) % 2 == 1)
    real = is_pd.unsqueeze(0) & (tokens != tok.pad_token) & \
        (tokens != tok.eos_token) & (tokens < tok.n_normal_tokens)
    out[real] = (out[real] + semitones).clamp(128, tok.n_normal_tokens - 1)
    return out


def frame_ce(logits, target, pad):
    """Mean CE over non-pad token positions of one frame."""
    V = logits.shape[-1]
    keep = (target != pad)
    if keep.sum() == 0:
        return float('nan')
    ce = F.cross_entropy(logits[keep].view(-1, V).float(),
                         target[keep].view(-1), reduction='mean')
    return float(ce)


def frame_divergence(logits_a, logits_b, target, pad):
    """Jensen-Shannon divergence between two predictive distributions,
    plus argmax disagreement, over non-pad positions."""
    keep = (target != pad)
    if keep.sum() == 0:
        return float('nan'), float('nan')
    pa = F.softmax(logits_a[keep].float(), dim=-1)
    pb = F.softmax(logits_b[keep].float(), dim=-1)
    m = 0.5 * (pa + pb)
    kl = lambda p, q: (p * (p.clamp_min(1e-12).log()
                            - q.clamp_min(1e-12).log())).sum(-1)
    js = (0.5 * kl(pa, m) + 0.5 * kl(pb, m)).mean()
    disagree = (pa.argmax(-1) != pb.argmax(-1)).float().mean()
    return float(js), float(disagree)


def split_slots(q_logits, B):
    """Query logits come back FLATTENED as [B*2, S, V] (local_decode's
    layout: index = b*2 + slot), not [B, 2, S, V]. Normalise, so the
    melody slot is [:, 0] and the chord slot [:, 1]."""
    if q_logits.dim() == 3:
        S, V = q_logits.shape[-2], q_logits.shape[-1]
        q_logits = q_logits.view(B, -1, S, V)
    return q_logits[:, 0], q_logits[:, 1]


@torch.no_grad()
def run_condition(net, x, t, K, mode, lag, semitones, device):
    """One forward with the slots set up for `mode`.

    `mode` is 'ctx', or '<mel|chd>_<true|decoy|shift>' -- the stream
    names WHICH SLOT is filled; the other stays masked at k=K.
    """
    B, L, S = x.shape
    T_full = L // 2
    full = lambda v: torch.full((B,), v, dtype=torch.long, device=device)

    def content_emb(mod, kind):
        if kind == 'shift':
            tokens = transpose_frame(net, x[:, 2 * t + mod], semitones)
        else:                                     # 'decoy': displaced
            tokens = x[:, 2 * ((t + lag) % T_full) + mod]
        return net._encode_frame(tokens, mod)

    k = {0: full(K), 1: full(K)}
    sc = {0: None, 1: None}
    emb = {0: None, 1: None}
    if mode != 'ctx':
        stream, kind = mode.split('_')
        mod = 0 if stream == 'mel' else 1
        k[mod] = full(0)                          # reveal this slot
        if kind != 'true':
            sc[mod] = torch.ones(B, dtype=torch.bool, device=device)
            emb[mod] = content_emb(mod, kind)

    _, q_logits, _ = net(x, T_query=t, k_m=k[0], k_c=k[1],
                         sc_mask_m=sc[0], sc_emb_m=emb[0],
                         sc_mask_c=sc[1], sc_emb_c=emb[1])
    return q_logits


def paired_stats(a, b):
    """mean and standard error of the per-batch paired difference a - b."""
    diffs = [x - y for x, y in zip(a, b) if x == x and y == y]
    n = len(diffs)
    if n == 0:
        return float('nan'), float('nan')
    mu = sum(diffs) / n
    if n == 1:
        return mu, float('nan')
    var = sum((d - mu) ** 2 for d in diffs) / (n - 1)
    return mu, (var / n) ** 0.5


@torch.no_grad()
def block_probe(net, batches, K, pad, n_frames, device):
    """A.8 only: does the block let a slot read the FUTURE?

    In A.8 every slot in the block reads every other slot, and with
    A.3's kernel a neighbour holds its TRUE frame whenever its k < K.
    So the offset-0 slot can predict frame t0 from true frames
    t0+1..t0+B-1 -- bidirectional teacher forcing that history-only
    prediction cannot match. If the model learned that, it shows as
    the offset-0 slot doing much better with true neighbours than with
    masked ones (LEAK), while a LONE slot -- what val_query_loss scores,
    and a regime training never visits -- sits where the masked-
    neighbour case sits or worse (REGIME).

    Forwards per frame, all with the block's other slots as stated:
      alone        T_query = t0 only          (the val_loss regime)
      nbr_masked   block, every slot at k=K   (decode round r=K)
      nbr_true     block, slot 0 at k=K, the rest at k=0
      only_j       block, slot j at k=K, the rest at k=0  (j = 1..B-1)
    Reports, per offset j: CE with everything masked (how much harder
    each further-ahead frame is with history cut at t0), CE with the
    slot itself revealed (copy sanity, must be ~0), and CE with only
    the slot masked (the leak at that offset -- interior offsets have
    true neighbours on BOTH sides).

    Every cell also counts frames that came back NaN, split by cause:
    an all-pad target (nothing to score) vs NaN logits (a forward
    defect, e.g. an attention row with no permitted key). A NaN cell
    is printed as nan, never averaged away.
    """
    Bq = int(net.query_block)
    print(f'\n  --- block regime (query_block={Bq}) ---')
    conds = ['alone', 'nbr_masked', 'nbr_true']
    per = {c: {'m': [], 'c': []} for c in conds}
    per_off = {kind: {j: {'m': [], 'c': []} for j in range(Bq)}
               for kind in ('all_masked', 'self_true', 'only_self_masked')}
    nan_pad = {}      # (label, stream) -> frames with all-pad target
    nan_logit = {}    # (label, stream) -> frames with NaN logits
    n_seen = 0

    def score(label, logits, target, s):
        """frame_ce plus NaN bookkeeping under `label`."""
        key = (label, s)
        if (target != pad).sum() == 0:
            nan_pad[key] = nan_pad.get(key, 0) + 1
            return float('nan')
        if not torch.isfinite(logits).all():
            nan_logit[key] = nan_logit.get(key, 0) + 1
            return float('nan')
        return frame_ce(logits, target, pad)

    def bm(v):
        good = [u for u in v if u == u]
        return sum(good) / len(good) if good else float('nan')

    for batch in batches:
        x, T_full, S = interleave(net, batch, device)
        B = x.shape[0]
        lo, hi = T_full // 4, min((3 * T_full) // 4, T_full - Bq)
        span = max(hi - lo, 1)
        frames = sorted({lo + (i * span) // n_frames for i in range(n_frames)})
        acc = {c: {'m': [], 'c': []} for c in conds}
        acc_off = {kind: {j: {'m': [], 'c': []} for j in range(Bq)}
                   for kind in per_off}
        for t0 in frames:
            n_seen += 1
            tq = tuple(range(t0, t0 + Bq))
            kK = torch.full((B, Bq), K, dtype=torch.long, device=device)
            k0 = torch.zeros_like(kK)

            def block(k):
                q = net(x, T_query=tq, k_m=k, k_c=k)[1]
                return q.view(B, -1, q.shape[-2], q.shape[-1])

            q_alone = net(x, T_query=t0, k_m=kK[:, 0].contiguous(),
                          k_c=kK[:, 0].contiguous())[1]
            q_alone = q_alone.view(B, -1, q_alone.shape[-2], q_alone.shape[-1])
            q_masked = block(kK)
            q_true = block(k0)                       # every slot revealed
            only = {}
            for j in range(Bq):
                k = k0.clone()
                k[:, j] = K
                only[j] = block(k)                   # slot j alone masked
            tm, tc = x[:, 2 * t0], x[:, 2 * t0 + 1]
            acc['alone']['m'].append(score('alone', q_alone[:, 0], tm, 'm'))
            acc['alone']['c'].append(score('alone', q_alone[:, 1], tc, 'c'))
            acc['nbr_masked']['m'].append(score('nbr_masked', q_masked[:, 0], tm, 'm'))
            acc['nbr_masked']['c'].append(score('nbr_masked', q_masked[:, 1], tc, 'c'))
            acc['nbr_true']['m'].append(score('nbr_true', only[0][:, 0], tm, 'm'))
            acc['nbr_true']['c'].append(score('nbr_true', only[0][:, 1], tc, 'c'))
            for j in range(Bq):
                t = t0 + j
                tm, tc = x[:, 2 * t], x[:, 2 * t + 1]
                for kind, q in (('all_masked', q_masked), ('self_true', q_true),
                                ('only_self_masked', only[j])):
                    lab = f'{kind}@{j}'
                    acc_off[kind][j]['m'].append(score(lab, q[:, 2 * j], tm, 'm'))
                    acc_off[kind][j]['c'].append(score(lab, q[:, 2 * j + 1], tc, 'c'))
        for c in conds:
            for s in ('m', 'c'):
                per[c][s].append(bm(acc[c][s]))
        for kind in per_off:
            for j in per_off[kind]:
                for s in ('m', 'c'):
                    per_off[kind][j][s].append(bm(acc_off[kind][j][s]))

    mean = bm
    print(f'  {"offset-0 slot, k=K":<22}{"CE mel":>9}{"CE chd":>9}')
    for c in conds:
        print(f'  {c:<22}{mean(per[c]["m"]):>9.4f}{mean(per[c]["c"]):>9.4f}')
    for s, name in (('m', 'melody'), ('c', 'chord')):
        leak, leak_se = paired_stats(per['nbr_masked'][s], per['nbr_true'][s])
        reg, reg_se = paired_stats(per['alone'][s], per['nbr_masked'][s])
        print(f'  [{name}]  LEAK  CE(nbr_masked) - CE(nbr_true) = '
              f'{leak:+.4f} +- {leak_se:.4f}')
        print(f'            REGIME CE(alone) - CE(nbr_masked) = '
              f'{reg:+.4f} +- {reg_se:.4f}')
    print(f'  {"per offset":<12}{"all masked":>18}{"self revealed":>18}'
          f'{"only self masked":>18}')
    print(f'  {"":<12}{"mel":>9}{"chd":>9}{"mel":>9}{"chd":>9}{"mel":>9}{"chd":>9}')
    for j in range(Bq):
        row = f'  {"t0+" + str(j):<12}'
        for kind in ('all_masked', 'self_true', 'only_self_masked'):
            row += (f'{mean(per_off[kind][j]["m"]):>9.4f}'
                    f'{mean(per_off[kind][j]["c"]):>9.4f}')
        print(row)
    print(f'  frames scored per cell: {n_seen}')
    if nan_pad or nan_logit:
        print('  NaN frames (excluded from every mean above):')
        for key in sorted(set(nan_pad) | set(nan_logit)):
            print(f'    {key[0]:<22} {key[1]}  all-pad target: '
                  f'{nan_pad.get(key, 0):>4}   NaN logits: '
                  f'{nan_logit.get(key, 0):>4}')
    else:
        print('  no NaN frames in any cell')
    print('  LEAK >> 0: the slot learned to read true future frames from '
          'its neighbours,\n  which decode can only ever fill with drafts. '
          'REGIME > 0: the lone-slot\n  eval that picks the checkpoint '
          'scores a regime the model never trained in.\n  "self revealed" '
          'must be ~0 (copy); "only self masked" is the leak per offset.')


def exact_ckpt(path):
    """Let a NAME=.../last.ckpt spec mean the FINAL weights.

    resolve_best_ckpt redirects a file literally named last.ckpt to its
    best-val sibling -- the bias it exists to prevent. For a run whose
    val_loss rises from the start (A.8), best-val is the ~9k
    checkpoint, and probing it says nothing about the trained design.
    An explicitly named file is honoured, so hand it a differently
    named symlink. The run directory's name is kept in the link so the
    gnl<N> filename detection still works.
    """
    if os.path.isfile(path) and os.path.basename(path).lower() == 'last.ckpt':
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        link = os.path.join(tempfile.mkdtemp(prefix='probe_ckpt_'),
                            f'{run}.final.ckpt')
        os.symlink(os.path.abspath(path), link)
        real = os.path.realpath(path)
        st = os.stat(real)
        print(f'[ckpt] probing FINAL weights: {path}\n'
              f'       resolves to {real}\n'
              f'       {st.st_size / 1e6:.1f} MB, mtime {st.st_mtime:.0f}')
        return link
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True,
                   help='NAME=PATH[=TASK], repeatable. The optional third '
                        'field probes that checkpoint on its OWN training '
                        'corpus instead of --task; use it whenever the '
                        'checkpoints are not corpus-matched, and compare '
                        'the "% of own" column rather than raw gains.')
    p.add_argument('--task', default='melchord',
                   help='default corpus for checkpoints with no TASK field')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--n_batches', type=int, default=25)
    p.add_argument('--n_frames', type=int, default=8,
                   help='query frames scored per batch, evenly spaced '
                        'over the middle half. Averaged within the '
                        'batch, so this cuts noise without inflating n.')
    p.add_argument('--lag', type=int, default=16,
                   help='decoy displacement in frames (16 = one bar)')
    p.add_argument('--semitones', type=int, default=1,
                   help='transposition for the pitch-shift condition')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = {}

    def batches_for(task_name):
        if task_name not in cache:
            cache[task_name] = collect_batches(get_task(task_name),
                                               args.batch_size,
                                               args.n_batches)
            print(f'[data] {task_name}: {len(cache[task_name])} val '
                  f'batches x {args.batch_size}')
        return cache[task_name]

    MODES = ['ctx', 'mel_true', 'chd_true', 'mel_decoy', 'chd_decoy',
             'mel_shift', 'chd_shift']
    for spec in args.ckpt:
        name, rest = spec.split('=', 1)
        path, _, task_name = rest.partition('=')
        task_name = task_name or args.task
        batches = batches_for(task_name)
        net = load_model(exact_ckpt(path)).to(device)
        net.eval()
        K = net.diffusion_K
        pad = net.tokenizer.pad_token
        print(f'\n{"=" * 66}\n{name}: {os.path.basename(path.rstrip("/"))}\n'
              f'  corpus: {task_name}\n{"=" * 66}')

        acc = {m: {'ce_m': [], 'ce_c': [], 'js_m': [], 'js_c': [],
                   'dis_m': [], 'dis_c': []} for m in MODES}
        for batch in batches:
            x, T_full, S = interleave(net, batch, device)
            # Several query frames per batch, evenly spaced over the
            # middle half so every one has real history behind it and
            # room for the +lag decoy ahead of it. Scoring a single
            # frame per batch left the paired CONTENT difference with a
            # standard error as large as the effect. Averaging WITHIN
            # the batch first keeps the batch as the independent unit
            # -- frames of the same songs are not independent samples,
            # so they must not inflate n.
            lo, hi = T_full // 4, (3 * T_full) // 4
            span = max(hi - lo, 1)
            frames = sorted({lo + (i * span) // args.n_frames
                             for i in range(args.n_frames)})
            per_batch = {m: {k: [] for k in acc[m]} for m in MODES}
            for t in frames:
                logits = {m: run_condition(net, x, t, K, m, args.lag,
                                           args.semitones, device)
                          for m in MODES}
                tgt_m, tgt_c = x[:, 2 * t], x[:, 2 * t + 1]
                B = x.shape[0]
                cm, cc = split_slots(logits['ctx'], B)
                for m in MODES:
                    lm, lc = split_slots(logits[m], B)
                    per_batch[m]['ce_m'].append(frame_ce(lm, tgt_m, pad))
                    per_batch[m]['ce_c'].append(frame_ce(lc, tgt_c, pad))
                    js, dis = frame_divergence(lm, cm, tgt_m, pad)
                    per_batch[m]['js_m'].append(js)
                    per_batch[m]['dis_m'].append(dis)
                    js, dis = frame_divergence(lc, cc, tgt_c, pad)
                    per_batch[m]['js_c'].append(js)
                    per_batch[m]['dis_c'].append(dis)
            for m in MODES:
                for key, vals in per_batch[m].items():
                    good = [v for v in vals if v == v]
                    acc[m][key].append(sum(good) / len(good) if good
                                       else float('nan'))

        mean = lambda v: sum(v) / max(len(v), 1)
        print(f'{"condition":<16}{"CE mel":>9}{"CE chd":>9}'
              f'{"JS vs ctx (m/c)":>20}{"argmax diff (m/c)":>22}')
        for m in MODES:
            a = acc[m]
            print(f'{m:<16}{mean(a["ce_m"]):>9.4f}{mean(a["ce_c"]):>9.4f}'
                  f'{mean(a["js_m"]):>10.4f}/{mean(a["js_c"]):<9.4f}'
                  f'{mean(a["dis_m"]):>11.3f}/{mean(a["dis_c"]):<10.3f}')

        # Per stream: its OWN slot is the one named after it, its
        # PARTNER slot is the other one. One condition, two readings.
        print()
        content = {}
        for stream, col, own, partner in (('melody', 'm', 'mel', 'chd'),
                                          ('chord', 'c', 'chd', 'mel')):
            base = mean(acc['ctx'][f'ce_{col}'])
            true_ce = mean(acc[f'{partner}_true'][f'ce_{col}'])
            decoy_ce = mean(acc[f'{partner}_decoy'][f'ce_{col}'])
            own_gain = base - mean(acc[f'{own}_true'][f'ce_{col}'])
            gain = base - true_ce
            # PARTNER GAIN conflates two effects: the model may simply
            # prefer ANY legal frame in that slot over the mask token
            # (presence), regardless of which frame it is. Subtracting
            # the decoy -- wrong partner, equally legal, equally
            # present -- isolates the part that is about the partner's
            # actual content, which is the only part that can carry
            # coordination.
            content_gain = decoy_ce - true_ce
            presence = gain - content_gain
            # CONTENT is a PAIRED difference -- same batches, same
            # frame, only the partner's identity changes -- so its
            # standard error comes from the per-batch differences, not
            # from the spread of either condition. Without it a +0.005
            # gain and a +0.02 gain look equally real.
            diffs = [d - t for d, t in zip(acc[f'{partner}_decoy'][f'ce_{col}'],
                                           acc[f'{partner}_true'][f'ce_{col}'])]
            n = len(diffs)
            if n > 1:
                mu = sum(diffs) / n
                var = sum((d - mu) ** 2 for d in diffs) / (n - 1)
                sem = (var / n) ** 0.5
            else:
                sem = float('nan')
            sens = mean(acc[f'{partner}_decoy'][f'js_{col}'])
            shift = mean(acc[f'{partner}_shift'][f'js_{col}'])
            frac = content_gain / own_gain if own_gain > 1e-6 else float('nan')
            tstat = content_gain / sem if sem and sem == sem else float('nan')
            content[stream] = tstat
            print(f'  [{stream}]  OWN GAIN {own_gain:+.4f}   '
                  f'PARTNER GAIN {gain:+.4f}')
            print(f'            = CONTENT {content_gain:+.4f} +- {sem:.4f} '
                  f'(t={tstat:.1f}, {100 * frac:.1f}% of own)'
                  f'  +  presence {presence:+.4f}')
            print(f'            partner DECOY JS {sens:.4f}     '
                  f'partner SHIFT JS {shift:.4f}')
        # Whether the channel exists is a question about zero, so key
        # the verdict on the standard error, not on a share-of-own
        # threshold: a 0.5%-of-own effect can be solidly non-zero while
        # a 1.1% one is noise. How BIG the channel is, is what the
        # "% of own" figure is for, and the two must not be conflated.
        alive = [s for s, t in content.items() if t == t and t > 2.0]
        print(f"  -> same-instant channel carries CONTENT (t>2) for: "
              f'{", ".join(alive) if alive else "NEITHER stream"}')

        if int(getattr(net, 'query_block', 1)) > 1:
            block_probe(net, batches, K, pad, args.n_frames, device)

    print(f'\n{"=" * 66}')
    print('How to read: CONTENT is the information the same-instant edge '
          'actually\ncarries -- PARTNER GAIN minus the part that is just '
          'a real frame beating\nthe mask token. Compare it against its '
          'own standard error first: these\nare small numbers. If A.7 '
          'carries far less CONTENT than A.3 while its\nDECOY sensitivity '
          'is ~0, the parametric corruption taught it to discard\nthe '
          'channel, and pitch-shift (also parametric, also invertible) '
          'would meet\nthe same fate. If A.7 carries as much but is '
          'MISLED, the channel lives and\nthe failure is elsewhere.')


if __name__ == '__main__':
    main()
