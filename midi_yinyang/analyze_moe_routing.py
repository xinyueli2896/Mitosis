"""Is the MoE router collapsed, and does it specialize by modality?

MoERoutingMonitor logs mean expert PROBABILITY during training. That is
not load: a router can show near-uniform mean probs while its top-k
selection always lands on the same experts, because the mean is taken
over a softmax that never has to commit. Collapse lives in the argmax,
so this reads the routing distribution off a trained checkpoint and
reports what the mean hides:

  load_e        fraction of tokens whose TOP-1 is expert e. Balanced is
                1/E; collapse is one expert near 1.0.
  max_load      the largest of those. The single collapse number.
  entropy       mean per-token routing entropy / log(E). 1.0 means the
                router is uniform -- it never decides anything, and the
                MoE is a dense FFN with E x the parameters. 0.0 means
                every token is routed with full confidence.
  dead          experts that appear in NO token's top-k. Dead experts are
                pure wasted parameters and cannot recover: they receive
                no gradient.
  mod L1        L1 distance between the mod_a and mod_b mean routing
                distributions. This is the "do experts specialize by
                stream" question. ~0 means the router ignores modality.

Runs on CPU. One real batch from the task's own dataset, one forward,
then every SimpleMoEFFN's cached `_last_routing_probs` is read.

PROBE MODES (--probe) -- stamp vs content. In the ENCODING nothing
distinguishes the streams (same local encoder, token_type_embeddings
zeroed, programs constant on melchord): at the global-stack input the
only difference is content. But the router reads the hidden AFTER the
per-modality attention projections, and once training diverges o_m from
o_c every position is STAMPED with its slot parity before the first
router sees it. So high modality purity has two candidate mechanisms:
the router reads content (density, register), or it merely reads the
architectural stamp. The probes separate them on a trained ckpt, no
retraining:

  --probe identical   duplicate the MELODY content into the chord slots.
                      Content is then identical across parities, so ANY
                      remaining parity separation IS the stamp.
  --probe swap        exchange the streams' contents. Content and slot
                      now disagree; whichever the experts follow wins.
  --probe within      WITHIN-STREAM content-responsiveness: does routing
                      vary with what is in the frame (note density,
                      register, mean duration), inside each stream, vs a
                      shuffle null? This is the right content question
                      for per-modality-gate (A.2.moe_permod) ckpts --
                      where swap's slot/content dichotomy is undefined --
                      and runs on shared-router ckpts too, so baselines
                      are directly comparable. Interleaved layouts only.

Either probe runs a baseline pass on the real batch first and prints a
per-layer comparison plus a verdict.

Usage (via analyze_moe_routing.sbatch -- do not call this directly):
    python analyze_moe_routing.py --variant c1 --task melchord_nottingham \\
        --ckpt ckpt/<run>/ --batch-size 2
"""

import argparse
import math
import os
import sys

import torch

# Which inference module owns each variant's load_model, and how a
# position in that variant's global-stack sequence maps to a modality.
# Getting this wrong would mislabel the per-modality split, so a variant
# whose layout is not known here reports global metrics only.
VARIANTS = {
    'a1': ('cp_transformer_m2c_intra_cross_attn_inference', 'interleaved'),
    'a2': ('cp_transformer_m2c_duet_block_diffusion_inference', 'interleaved'),
    'b1': ('cp_transformer_m2c_duet_anticipatory_inference', 'interleaved'),
    'c1': ('cp_transformer_m2c_duet_rehearsal_inference', 'prefix_interleaved'),
    'c2': ('cp_transformer_m2c_duet_prefix_inference', 'two_block'),
}


def modality_of(layout, L, T):
    """-> list of 'a' / 'b' / 'prefix' per position, or None if unknown."""
    if layout == 'interleaved':            # [a_0, b_0, a_1, b_1, ...]
        return ['a' if i % 2 == 0 else 'b' for i in range(L)]
    if layout == 'prefix_interleaved':     # [mod_a prefix (T)] + interleaved
        return (['prefix'] * T
                + ['a' if (i - T) % 2 == 0 else 'b' for i in range(T, L)])
    if layout == 'two_block':              # [mod_a block (T)][mod_b block]
        return ['a' if i < T else 'b' for i in range(L)]
    return None


def stats(probs, topk):
    """probs: [N, E] -> collapse metrics for one set of positions."""
    N, E = probs.shape
    if N == 0:
        return None
    top1 = probs.argmax(dim=-1)
    load = torch.bincount(top1, minlength=E).float() / N
    p = probs.clamp_min(1e-9)
    ent = float((-(p * p.log()).sum(-1)).mean() / math.log(E))
    used = torch.zeros(E, dtype=torch.bool)
    used[probs.topk(min(topk, E), dim=-1).indices.reshape(-1).unique()] = True
    return {'load': load, 'max_load': float(load.max()), 'entropy': ent,
            'dead': [e for e in range(E) if not used[e]], 'n': N}


def snapshot(layers, content=False):
    """content=True reads _last_content_probs -- the softmax of the
    UNBIASED router logits, which only exists on a modality-bias
    (A.2.moe_improved) model. That is the pathway the probes must test
    there: the explicit bias separates the modalities BY DESIGN, so
    running the stamp probe on the full probs would just re-measure the
    bias. The success metric is the stamp share of the content pathway."""
    if content:
        return [l.ffn._last_content_probs.clone() for l in layers]
    return [l.ffn._last_routing_probs.clone() for l in layers]


def parity_profile(pr, layout, topk):
    """One layer's routing summarised per slot parity: mean-prob L1
    between parities, and each parity's top-1-preferred expert."""
    B, L, E = pr.shape
    labels = modality_of(layout, L, L // 3 if layout == 'prefix_interleaved'
                         else L // 2)
    ia = [j for j, x in enumerate(labels) if x == 'a']
    ib = [j for j, x in enumerate(labels) if x == 'b']
    fa = pr[:, ia].reshape(-1, E)
    fb = pr[:, ib].reshape(-1, E)
    return {'l1': float((fa.mean(0) - fb.mean(0)).abs().sum()),
            'pref_a': int(stats(fa, topk)['load'].argmax()),
            'pref_b': int(stats(fb, topk)['load'].argmax())}


def run_probe(net, batch, probe, base_prs, layers, layout, topk,
              content=False, gates=False):
    """Second forward with manipulated streams; compare to baseline."""
    x_mel, x_acc, ps = batch[0], batch[1], batch[2]
    if probe == 'identical':
        probed = (x_mel, x_mel.clone(), ps)
        what = ('melody content DUPLICATED into the chord slots -- content '
                'is identical across parities, so any remaining parity '
                'separation is the architectural stamp')
    else:
        probed = (x_acc, x_mel, ps)
        what = ('stream contents SWAPPED -- content and slot parity now '
                'disagree; whichever the experts follow wins')
    with torch.no_grad():
        net.loss(*probed)
    probe_prs = snapshot(layers, content=content)

    print('\n' + '=' * 72)
    print(f'PROBE: {probe}')
    print(f'  {what}')
    if content:
        print('  NOTE: modality-bias router -- this probe runs on the '
              'CONTENT pathway')
        print('  (unbiased logits). The explicit bias separates the '
              'parities by design;')
        print('  the question is whether the input-driven part still '
              'carries the stamp.')
    if gates:
        print('  NOTE: per-modality gates -- the stamp framing does NOT '
              'apply here.')
        print('  identical: any remaining parity separation is gate_m-vs-'
              'gate_c weight')
        print('  divergence (expected once trained), not a stamp read. '
              'swap: if each')
        print('  gate routes by content, expert preferences should FOLLOW '
              'the moved')
        print('  content -- this is the live question for this variant.')
    print('=' * 72)
    base = [parity_profile(p, layout, topk) for p in base_prs]
    prob = [parity_profile(p, layout, topk) for p in probe_prs]

    if probe == 'identical':
        print(f'\n  {"layer":>5} {"L1 real":>9} {"L1 identical":>13}   '
              f'(parity separation with content equalised)')
        for i, (b, q) in enumerate(zip(base, prob)):
            print(f'  {i:>5} {b["l1"]:>9.3f} {q["l1"]:>13.3f}')
        mb = sum(x['l1'] for x in base) / len(base)
        mq = sum(x['l1'] for x in prob) / len(prob)
        ratio = mq / mb if mb > 0 else float('nan')
        print(f'\n  mean L1: real {mb:.3f} -> identical-content {mq:.3f}  '
              f'(stamp share ~ {ratio:.0%})')
        if gates:
            print('\n  VERDICT (per-modality gates):')
            print(f'  GATE DIVERGENCE ~{ratio:.0%}: with identical content, '
                  f'the remaining parity')
            print('  separation is gate_m vs gate_c mapping the SAME input '
                  'to different')
            print('  expert distributions -- expected of two independently '
                  'trained matrices,')
            print('  and NOT a stamp read (the stamp cannot steer '
                  'within-stream routing here).')
            print('  The shared-router STAMP/CONTENT verdict below does '
                  'not apply; it is')
            print('  suppressed for this variant.')
            return
        if content:
            print('\n  A.2.moe_improved success metric: this stamp share '
                  'should be near ZERO --')
            print('  the explicit bias should have absorbed the parity bit, '
                  'leaving the content')
            print('  pathway free of it. The baseline (job 178945, no bias) '
                  'measured ~69%.')
        print('\n  VERDICT:')
        if ratio < 0.15:
            print('  CONTENT-DRIVEN. With identical content the parity '
                  'separation collapses;')
            print('  the router reads what is in the frame (density, '
                  'register), not the slot.')
            print('  The layer-0 density caveat therefore stands as the '
                  'live concern.')
        elif ratio > 0.6:
            print('  STAMP-DRIVEN. The separation survives content '
                  'equalisation almost intact:')
            print('  the diverged per-modality projections imprint slot '
                  'parity on the hidden and')
            print('  the router reads the imprint. The "router discovers '
                  'the streams" claim')
            print('  should be restated: the ATTENTION projections '
                  'specialised; the router piggybacks.')
        else:
            print(f'  MIXED. ~{ratio:.0%} of the separation survives '
                  f'content equalisation (stamp),')
            print('  the rest was content. Report both mechanisms.')
        return

    # swap: does each parity's preferred expert stay with the SLOT or
    # follow the CONTENT to the other parity?
    slot_hits = content_hits = usable = 0
    print(f'\n  {"layer":>5} {"base a/b pref":>14} {"swap a/b pref":>14}   verdict')
    for i, (b, q) in enumerate(zip(base, prob)):
        if b['pref_a'] == b['pref_b']:
            print(f'  {i:>5} {"e%d/e%d" % (b["pref_a"], b["pref_b"]):>14} '
                  f'{"e%d/e%d" % (q["pref_a"], q["pref_b"]):>14}   (ambiguous: '
                  f'base parities share a preference)')
            continue
        usable += 1
        slot = q['pref_a'] == b['pref_a'] and q['pref_b'] == b['pref_b']
        cont = q['pref_a'] == b['pref_b'] and q['pref_b'] == b['pref_a']
        slot_hits += slot
        content_hits += cont
        tag = 'SLOT' if slot else ('CONTENT' if cont else 'neither')
        print(f'  {i:>5} {"e%d/e%d" % (b["pref_a"], b["pref_b"]):>14} '
              f'{"e%d/e%d" % (q["pref_a"], q["pref_b"]):>14}   {tag}')
    print(f'\n  layers following the slot: {slot_hits}/{usable}   '
          f'following the content: {content_hits}/{usable}')
    if gates:
        print('\n  VERDICT (per-modality gates):')
        print('  The SLOT/CONTENT dichotomy assumes ONE router mapping '
              'content -> expert')
        print('  consistently across slots. With per-modality gates that '
              'is structurally')
        print('  impossible: under swap, moved content is scored by the '
              'OTHER stream\'s')
        print('  gate, a different matrix with no trained correspondence '
              '-- "neither" is')
        print('  the expected outcome for a content-responsive gate, and '
              'CONTENT rows')
        print('  would be coincidence. The meaningful split here:')
        print(f'    prefs CHANGED under swap (input-sensitive layers): '
              f'{usable - slot_hits}/{usable}')
        print(f'    prefs UNCHANGED (weight-prior routing, content-'
              f'insensitive): {slot_hits}/{usable}')
        print('  A rigorous within-stream content-responsiveness test '
              'needs a same-stream')
        print('  probe (e.g. routing correlation with frame register/'
              'density), not swap.')
        return
    print('\n  VERDICT:')
    if usable == 0:
        print('  No usable layers (base parities always shared a '
              'preference); rely on --probe identical.')
    elif content_hits > slot_hits:
        print('  CONTENT-DRIVEN: expert preferences travel with the '
              'content when it moves')
        print('  to the other slot. The router recognises the material, '
              'not the position.')
    elif slot_hits > content_hits:
        print('  STAMP-DRIVEN: expert preferences stay with the slot '
              'parity even when the')
        print('  content moves. The router reads the per-modality '
              'projections\' imprint.')
    else:
        print('  SPLIT. Layers disagree; read the per-layer column and '
              'pair with --probe identical.')


def frame_features(frame, eos, pad):
    """(density, mean pitch, mean duration) of one frame's
    (program, pitch-dur) token pairs; NaN pitch/dur for empty frames.
    Decoding mirrors decode_output: pitch=(pd-2)%128, dur=(pd-2)//128."""
    t = frame.reshape(-1).tolist()
    pitches, durs = [], []
    for i in range(0, len(t) - 1, 2):
        prog, pd = t[i], t[i + 1]
        if prog == eos or prog == pad or pd == pad or pd == eos:
            break
        pitches.append((pd - 2) % 128)
        durs.append((pd - 2) // 128)
    if not pitches:
        return 0.0, float('nan'), float('nan')
    return (float(len(pitches)),
            sum(pitches) / len(pitches),
            sum(durs) / len(durs))


def _bucket_l1(vals, probs, n_shuffle=20):
    """Content-responsiveness of routing to one feature: split tokens
    into feature terciles, mean routing distribution per tercile, mean
    pairwise L1 between them -- against a shuffle null (95th pct of the
    same metric with feature values permuted). Returns (l1, null95) or
    None when the feature is (near-)constant."""
    keep = ~torch.isnan(vals)
    vals, probs = vals[keep], probs[keep]
    if vals.numel() < 30:
        return None
    q1, q2 = torch.quantile(vals, 1 / 3), torch.quantile(vals, 2 / 3)
    if q1 == q2:            # degenerate feature (e.g. constant density)
        return None
    groups = [vals <= q1, (vals > q1) & (vals <= q2), vals > q2]
    if min(int(g.sum()) for g in groups) < 10:
        return None

    def metric(p):
        means = [p[g].mean(0) for g in groups]
        return float(sum((means[i] - means[j]).abs().sum()
                         for i in range(3) for j in range(i + 1, 3)) / 3)

    real = metric(probs)
    nulls = []
    g = torch.Generator().manual_seed(0)
    for _ in range(n_shuffle):
        perm = torch.randperm(probs.shape[0], generator=g)
        nulls.append(metric(probs[perm]))
    nulls.sort()
    return real, nulls[int(0.95 * len(nulls)) - 1]


def run_within_probe(net, batch, layers, layout, E):
    """Within-stream content-responsiveness: does routing vary with what
    is IN the frame (density, register, duration), inside each stream?
    This is the question swap cannot answer for per-modality gates (and
    answers it for shared routers too, so baselines are comparable)."""
    if layout != 'interleaved':
        print('\nPROBE=within currently supports the interleaved layout '
              f'only (got {layout}).')
        return
    # Re-run one loss pass capturing the EXACT tokens that get routed
    # (preprocess applies pitch shift; features must describe the
    # shifted tokens, not the raw batch).
    captured = {}
    orig = net.preprocess

    def wrap(*a, **k):
        out = orig(*a, **k)
        captured['xm'], captured['xa'] = out[0].detach(), out[1].detach()
        return out

    net.preprocess = wrap
    try:
        with torch.no_grad():
            net.loss(*batch)
    finally:
        net.preprocess = orig
    prs = snapshot(layers)
    eos, pad = net.tokenizer.eos_token, net.tokenizer.pad_token
    xm, xa = captured['xm'], captured['xa']         # [B, T, S] each
    B, T, S = xm.shape

    feats = {}   # stream -> feature name -> [B*T]
    for key, x in (('a', xm), ('b', xa)):
        rows = [frame_features(x[b, t], eos, pad)
                for b in range(B) for t in range(T)]
        feats[key] = {
            'density': torch.tensor([r[0] for r in rows]),
            'register': torch.tensor([r[1] for r in rows]),
            'mean dur': torch.tensor([r[2] for r in rows]),
        }

    print('\n' + '=' * 72)
    print('PROBE: within -- routing vs frame content, INSIDE each stream')
    print('  For each layer/stream/feature: tokens split into feature '
          'terciles;')
    print('  metric = mean pairwise L1 between the terciles\' mean '
          'routing')
    print('  distributions, vs a shuffle null (95th pct, 20 perms). A '
          'value well')
    print('  above its null means routing follows that aspect of the '
          'content.')
    print('  Applies to shared routers and per-modality gates alike.')
    print('=' * 72)

    hits = {'a': 0, 'b': 0}
    for key, label in (('a', 'mod_a'), ('b', 'mod_b')):
        print(f'\n  {label}:  {"layer":>5} '
              + ''.join(f'{f:>22}' for f in feats[key]))
        layer_hit = [False] * len(prs)
        for i, pr in enumerate(prs):
            L = pr.shape[1]
            idx = [j for j in range(min(L, 2 * T))
                   if (j % 2 == 0) == (key == 'a')]
            p = pr[:, idx].reshape(-1, E)[: B * T]
            cells = []
            for fname, v in feats[key].items():
                r = _bucket_l1(v[: p.shape[0]], p)
                if r is None:
                    cells.append(f'{"flat/skip":>22}')
                    continue
                real, null95 = r
                mark = ' *' if real > null95 else '  '
                layer_hit[i] = layer_hit[i] or real > null95
                cells.append(f'{real:>10.3f} ({null95:.3f}){mark}')
            print(f'  {"":8}{i:>5} ' + ''.join(cells))
        hits[key] = sum(layer_hit)
        print(f'    -> {label}: routing follows content in '
              f'{hits[key]}/{len(prs)} layers (* = beats the null)')

    print('\n  VERDICT:')
    total = 2 * len(prs)
    n = hits['a'] + hits['b']
    if n >= total * 0.6:
        print(f'  CONTENT-DRIVEN WITHIN STREAMS: {n}/{total} '
              f'(stream, layer) pairs route')
        print('  differently by frame content, well beyond shuffle '
              'noise.')
    elif n > 0:
        print(f'  PARTIAL: {n}/{total} (stream, layer) pairs beat the '
              f'null; the rest route')
        print('  mostly from position/weight priors. Read the per-layer '
              'stars above.')
    else:
        print('  NOT CONTENT-DRIVEN: no (stream, layer) pair beats the '
              'shuffle null --')
        print('  within a stream, routing ignores what is in the frame.')
    print('=' * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    p.add_argument('--ckpt', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--model-size', default='large')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    p.add_argument('--probe', default='none',
                   choices=['none', 'identical', 'swap', 'within'],
                   help='stamp-vs-content probe; see module docstring')
    p.add_argument('--seed', type=int, default=0,
                   help='seeds torch RNG before the dataset draw, so two '
                        'runs (e.g. mg vs baseline ckpts) analyze the '
                        'SAME batch -- FramedDataset picks songs and '
                        'window offsets with torch.randperm/rand. '
                        'Without this, cross-ckpt comparisons are '
                        'confounded by the batch draw (different songs, '
                        'and features can degenerate to flat/skip in '
                        'one run but not the other).')
    args = p.parse_args()
    torch.manual_seed(args.seed)

    module_name, layout = VARIANTS[args.variant]
    mod = __import__(module_name)
    if not hasattr(mod, 'load_model'):
        sys.exit(f'{module_name} has no load_model(); cannot analyze '
                 f'variant {args.variant}')

    from cp_transformer_m2c_moe import FramedDataset, TRAIN_LENGTH
    from tasks import get_task

    task = get_task(args.task)
    print('=' * 72)
    print('MoE ROUTER COLLAPSE ANALYSIS')
    print(f'variant={args.variant}  layout={layout}  task={task.name}')
    print(f'ckpt={args.ckpt}')
    print('=' * 72)

    net = mod.load_model(
        args.ckpt, model_size=args.model_size,
        moe_num_experts=args.moe_num_experts, moe_topk=args.moe_topk,
    )
    net.eval()

    ds = FramedDataset(task.mod_b_path, TRAIN_LENGTH, args.batch_size,
                       split='val', mel_path=task.mod_a_path)
    # Seed HERE, not only at startup: model construction above consumes
    # RNG (random init before the state-dict load) and consumes a
    # different amount per variant (mg has extra gate params), so an
    # early seed alone still yields different batches per ckpt.
    torch.manual_seed(args.seed)
    print(f'[seed] batch draw seeded with {args.seed} -- same seed => '
          f'same songs/windows across ckpts')
    batch = list(next(iter(ds)))
    if len(batch) < 3:
        sys.exit(f'expected a (x_mel, x_acc, pitch_shift) batch, got '
                 f'{len(batch)} elements')
    with torch.no_grad():
        net.loss(*batch)          # populates _last_routing_probs per layer

    stack = getattr(net, 'global_layers', None)
    if stack is None:
        sys.exit('model has no .global_layers; this analyzer targets the duet variants')
    layers = [l for l in stack if getattr(l, 'ffn', None) is not None
              and getattr(l.ffn, '_last_routing_probs', None) is not None]
    if not layers:
        sys.exit('no SimpleMoEFFN routing probs found -- was this model '
                 'built with moe_num_experts > 1?')
    E = layers[0].ffn.num_experts
    topk = layers[0].ffn.topk
    has_bias = any(getattr(l.ffn, 'modality_bias', None) is not None
                   for l in layers)
    has_gates = any(getattr(l.ffn, 'modality_gates', False) for l in layers)
    print(f'\n{len(layers)} MoE layer(s), {E} experts, top-{topk}. '
          f'Balanced load = {1/E:.3f} per expert.')
    if has_gates:
        print('PER-MODALITY GATES (A.2.moe_permod): each stream is routed '
              'by its own')
        print('gate matrix over a fully shared, unassigned expert pool. '
              'Parity')
        print('separation (mod L1) is therefore BY CONSTRUCTION -- two '
              'different gates')
        print('-- and is not the finding. The finding is the PURITY table: '
              'an expert')
        print('near 0%/100% is a learned per-stream specialist, one near '
              'the base rate')
        print('that both gates keep using is a learned INTEGRATOR serving '
              'both streams.')
        print('The probes below test content-responsiveness within each '
              'gate, not the')
        print('stamp (which cannot steer within-stream routing here).')
    if has_bias:
        print('MODALITY-BIAS ROUTER (A.2.moe_improved): this ckpt carries '
              'an explicit')
        print('per-modality bias on the router logits. The tables below '
              'describe the')
        print('ACTUAL routing (bias included), so a modality split there is '
              'by design,')
        print('not a discovery. The bias-vs-content section after the '
              'verdict separates')
        print('what the explicit bias contributes from what the '
              'input-driven pathway does.')
    print()

    hdr = (f'{"layer":>5} {"max_load":>9} {"entropy":>8} {"dead":>6}  '
           f'{"per-expert top-1 load":<28} {"mod L1":>7}')
    print(hdr)
    print('-' * len(hdr))
    worst_load, min_ent, any_dead, mod_l1s = 0.0, 1.0, [], []
    per_mod = []
    purity = []
    max_ent = 0.0
    for i, layer in enumerate(layers):
        pr = layer.ffn._last_routing_probs        # [B, L, E]
        B, L, _ = pr.shape
        flat = pr.reshape(-1, E)
        s = stats(flat, topk)
        labels = modality_of(layout, L, L // 3 if layout == 'prefix_interleaved'
                             else L // 2)
        l1 = float('nan')
        if labels is not None:
            ia = [j for j, x in enumerate(labels) if x == 'a']
            ib = [j for j, x in enumerate(labels) if x == 'b']
            if ia and ib:
                ma = pr[:, ia].reshape(-1, E).mean(0)
                mb = pr[:, ib].reshape(-1, E).mean(0)
                l1 = float((ma - mb).abs().sum())
                mod_l1s.append(l1)
                # Which expert each stream prefers by TOP-1 load, not by
                # mean prob -- an expert can lead on average without ever
                # winning the argmax.
                la = stats(pr[:, ia].reshape(-1, E), topk)['load']
                lb = stats(pr[:, ib].reshape(-1, E), topk)['load']
                per_mod.append((i, int(la.argmax()), float(la.max()),
                                int(lb.argmax()), float(lb.max())))
                # EXPERT PURITY -- the direct "has this expert specialised
                # in a modality" question. The L1 above is
                # P(expert | modality); this is P(modality | expert): of
                # the tokens an expert actually WINS, what share are
                # mod_a? 0.5 means the expert is modality-agnostic, 0 or 1
                # means it is a pure specialist.
                ta = pr[:, ia].reshape(-1, E).argmax(-1)
                tb = pr[:, ib].reshape(-1, E).argmax(-1)
                ca = torch.bincount(ta, minlength=E).float()
                cb = torch.bincount(tb, minlength=E).float()
                share_a = ca / (ca + cb).clamp_min(1.0)
                # base rate: mod_a's share of all positions, so an expert
                # is only "specialised" relative to what it would get by
                # chance.
                base = len(ia) / float(len(ia) + len(ib))
                purity.append((i, share_a, (ca + cb), base))
        worst_load = max(worst_load, s['max_load'])
        min_ent = min(min_ent, s['entropy'])
        max_ent = max(max_ent, s['entropy'])
        any_dead += [(i, e) for e in s['dead']]
        loads = ' '.join(f'{v:.3f}' for v in s['load'])
        print(f'{i:>5} {s["max_load"]:>9.3f} {s["entropy"]:>8.3f} '
              f'{len(s["dead"]):>6}  {loads:<28} {l1:>7.3f}')

    print('\n' + '=' * 72)
    print('VERDICT')
    balanced = 1.0 / E
    uniform = min_ent > 0.95
    if uniform:
        # Check this FIRST: a uniform router's argmax is decided by
        # floating-point ties, which land on one index, so max_load reads
        # ~1.0 -- indistinguishable from collapse by load alone. Entropy
        # is what separates "one expert wins everything" from "the router
        # never decides anything".
        print(f'  NOT ROUTING: minimum normalized entropy {min_ent:.3f}. The '
              f'router is essentially uniform over experts.')
        print('  Every token gets the same blend, so the MoE is behaving as a')
        print(f'  dense FFN with {E}x the parameters. NOTE the load numbers')
        print('  above are meaningless in this regime -- a uniform softmax')
        print('  has no real argmax, so max_load reads ~1.0 from tie-breaking')
        print('  and looks like collapse. It is the opposite problem.')
        print('  Lever: the router gets no gradient to differentiate while')
        print('  experts are identical (the warm start replicates the same')
        print('  dense FFN into all of them); aux_loss_weight is what breaks')
        print('  that symmetry. 0.01 may be too small.')
    else:
        if any_dead:
            print(f'  DEAD EXPERTS: {len(any_dead)} (layer, expert) pair(s) '
                  f'never appear in any token\'s top-{topk}, e.g. '
                  f'{any_dead[:5]}.')
            print('  Those parameters receive no gradient and cannot recover.')
            print('  Raise aux_loss_weight (the trainers default to 0.01).')
        if worst_load > 0.8:
            print(f'  COLLAPSED: one expert takes {worst_load:.1%} of top-1 '
                  f'routing in some layer (balanced is {balanced:.1%}), while '
                  f'entropy {min_ent:.3f} says the router IS deciding.')
            print('  The load-balancing aux loss is not holding. Raise '
                  'aux_loss_weight.')
        elif worst_load < balanced * 1.5:
            print(f'  HEALTHY: worst max_load {worst_load:.3f} vs '
                  f'{balanced:.3f} ideal, no dead experts, and entropy '
                  f'{min_ent:.3f} shows real routing decisions.')
        else:
            # The band between "well balanced" and "collapsed" had no
            # message at all, which reads as no comment rather than a
            # result. It is the normal healthy-but-specialised regime.
            print(f'  NO COLLAPSE, MILD IMBALANCE: worst max_load '
                  f'{worst_load:.3f} against {balanced:.3f} ideal, no dead '
                  f'experts, entropy {min_ent:.3f}-{max_ent:.3f}.')
            print('  Imbalance in this band is what SPECIALIZATION looks '
                  'like -- an expert that owns a recognisable slice of the')
            print('  input necessarily takes more than its even share. Read '
                  'it together with the modality numbers below: if the')
            print('  imbalanced layers are also the high-L1 ones, the load '
                  'skew IS the specialization, not a failure of the aux')
            print('  loss.')
        if 0.3 < min_ent < 0.95:
            print(f'  Routing is SOFT (entropy {min_ent:.3f}): experts are '
                  f'being blended more than selected. With top-{topk} of {E} '
                  f'that means the chosen experts get similar weights.')
    if mod_l1s:
        m = sum(mod_l1s) / len(mod_l1s)
        print(f'\n  MODALITY SPECIALIZATION: mean L1 between mod_a and mod_b '
              f'routing = {m:.3f} (max possible 2.0).')
        if m < 0.05:
            print('  The router ignores modality: the same experts serve both')
            print('  streams. Note it has no modality INPUT -- the gate is a')
            print('  linear layer on the hidden state, and')
            print('  token_type_embeddings is zeroed and frozen in the duet')
            print('  variants -- so modality reaches it only indirectly via')
            print('  the per-modality attention output. A hard or biased')
            print('  modality route would be a mask on the router logits, no')
            print('  new module.')
        else:
            print('  The router DOES separate the streams -- with no modality')
            print('  input of its own, purely from the hidden state. Worth')
            print('  reporting: implicit modality routing is a result, not a')
            print('  configuration.')
        if purity:
            base = purity[0][3]
            print(f'\n  EXPERT PURITY -- of the tokens each expert WINS, '
                  f'the % that are mod_a.')
            print(f'  Base rate is {base:.0%} (mod_a\'s share of all '
                  f'positions), so {base:.0%} means modality-agnostic and')
            print(f'  0% or 100% means a pure specialist.\n')
            print('  layer  ' + '  '.join(f'{"e"+str(e):>10}' for e in range(E))
                  + f'  {"spec":>6}')
            worst_spec, best_spec = 1.0, 0.0
            for (i, share_a, counts, b) in purity:
                cells = []
                for e in range(E):
                    if counts[e] < 1:
                        cells.append(f'{"unused":>10}')
                    else:
                        cells.append(f'{float(share_a[e]):>9.1%} ')
                # how far the most one-sided expert is from the base rate,
                # normalised so 1.0 = a perfectly pure specialist
                dev = max(abs(float(share_a[e]) - b) for e in range(E)
                          if counts[e] >= 1) / max(b, 1 - b)
                worst_spec = min(worst_spec, dev)
                best_spec = max(best_spec, dev)
                print(f'  {i:>5}  ' + '  '.join(cells) + f'  {dev:>6.2f}')
            print(f'\n  most modality-specialised layer: {best_spec:.2f}   '
                  f'least: {worst_spec:.2f}   (1.00 = a pure specialist)')
            if best_spec < 0.15:
                print('  VERDICT: experts have NOT specialised by modality. '
                      'Every expert sees')
                print('  both streams in roughly their base-rate proportion.')
            elif best_spec < 0.5:
                print('  VERDICT: PARTIAL modality specialisation. Experts '
                      'lean toward one')
                print('  stream but none is dedicated to it -- the router is '
                      'biasing, not partitioning.')
            else:
                print('  VERDICT: experts HAVE specialised by modality; the '
                      'most one-sided')
                print('  is close to a dedicated per-stream expert.')
        if per_mod:
            print(f'\n  per-layer top-1 preference -- P(expert | modality): '
                  f'which expert each STREAM')
            print(f'  most often lands on. Purity above is the other '
                  f'conditional, P(modality | expert),')
            print(f'  so the two tables can legitimately name different '
                  f'experts (a small pure expert')
            print(f'  vs a large mixed one).')
            print(f'  {"":28s}{"mod_a":>14}  {"mod_b":>14}   same?')
            for (i, ea, va, eb, vb) in per_mod:
                same = 'SAME' if ea == eb else ''
                print(f'    layer {i:>2}                      '
                      f'e{ea} ({va:.3f})     e{eb} ({vb:.3f})   {same}')
    print('=' * 72)

    if has_bias:
        print('\n' + '=' * 72)
        print('BIAS vs CONTENT PATHWAY -- division of labour per layer')
        print('  full L1     parity separation of the actual routing '
              '(bias included)')
        print('  content L1  parity separation of the input-driven pathway '
              'alone')
        print('              (softmax of the unbiased logits)')
        print('  bias delta  max |bias_a[e] - bias_b[e]| -- how strongly '
              'the explicit')
        print('              bias itself separates the modalities')
        print(f'\n  {"layer":>5} {"full L1":>9} {"content L1":>11} '
              f'{"bias delta":>11}')
        for i, layer in enumerate(layers):
            fl = parity_profile(layer.ffn._last_routing_probs, layout,
                                topk)['l1']
            cl = parity_profile(layer.ffn._last_content_probs, layout,
                                topk)['l1']
            bd = float((layer.ffn.modality_bias[0]
                        - layer.ffn.modality_bias[1]).detach().abs().max())
            print(f'  {i:>5} {fl:>9.3f} {cl:>11.3f} {bd:>11.3f}')
        print('\n  Healthy A.2.moe_improved: bias delta grows during '
              'training while')
        print('  content L1 on the REAL batch reflects genuine content '
              'differences only')
        print('  -- confirmed by PROBE=identical driving the content-'
              'pathway stamp share')
        print('  toward zero (baseline without the bias: ~69%).')
        print('=' * 72)

    if args.probe == 'within':
        run_within_probe(net, batch, layers, layout, E)
    elif args.probe != 'none':
        # Baseline probs are still live on the layers (nothing has run a
        # forward since); snapshot them before the probe pass overwrites.
        # On a modality-bias model the probes target the content pathway.
        base_prs = snapshot(layers, content=has_bias)
        run_probe(net, batch, args.probe, base_prs, layers, layout, topk,
                  content=has_bias, gates=has_gates)


if __name__ == '__main__':
    main()
