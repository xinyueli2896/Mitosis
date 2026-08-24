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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    p.add_argument('--ckpt', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--model-size', default='large')
    p.add_argument('--moe-num-experts', type=int, default=4)
    p.add_argument('--moe-topk', type=int, default=2)
    args = p.parse_args()

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
    batch = next(iter(ds))
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
    print(f'\n{len(layers)} MoE layer(s), {E} experts, top-{topk}. '
          f'Balanced load = {1/E:.3f} per expert.\n')

    hdr = (f'{"layer":>5} {"max_load":>9} {"entropy":>8} {"dead":>6}  '
           f'{"per-expert top-1 load":<28} {"mod L1":>7}')
    print(hdr)
    print('-' * len(hdr))
    worst_load, min_ent, any_dead, mod_l1s = 0.0, 1.0, [], []
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
        worst_load = max(worst_load, s['max_load'])
        min_ent = min(min_ent, s['entropy'])
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
        elif worst_load < balanced * 1.5 and not any_dead:
            print(f'  HEALTHY: worst max_load {worst_load:.3f} vs '
                  f'{balanced:.3f} ideal, no dead experts, and entropy '
                  f'{min_ent:.3f} shows real routing decisions.')
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
            print('  The router does separate the streams to some degree.')
    print('=' * 72)


if __name__ == '__main__':
    main()
