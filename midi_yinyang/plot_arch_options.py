"""Option sheet: conventional ways ML papers draw an attention layer and
a mixture-of-experts router, each adapted to Duet, so one can be picked
for the model figure.

Row 1, attention:
  A  flow diagram in the Vaswani et al. style (Linear Q/K/V -> three
     scaled dot-product attention branches with a Mask box each ->
     gates -> sum -> Linear W_O)
  B  admission matrix (queries x keys, cell colour = pass), the
     UniLM / Longformer / BigBird convention, all three passes in one map
  C  arcs over the token row (Transformer-XL / sparse-attention
     convention): which keys one query may reach, coloured by pass
Row 2, router:
  D  Switch-Transformer style: token -> router -> fan-out to experts
     with gate probabilities, top-k solid, the rest dashed
  E  Mixtral style: gating network with a bar of gate weights over the
     expert boxes and the weighted-sum equation
  F  ours: one router per stream feeding one shared pool, experts
     coloured by the pretrained model they were copied from

Usage:
  python plot_arch_options.py --out results/fig_arch_options
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Arc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_e1_box import SURFACE, INK, INK_2, PALETTE  # noqa: E402
from plot_arch_brief import masks, BLUE, RED, GOLD, BLUE_L, RED_L, GOLD_L, GREY_L, BLOCK  # noqa: E402

FS = 6.2
DOT = (0, (1.2, 1.2))
DASH = (0, (3, 1.5))
PASS_C = {'same': '#6f8fc9', 'cross': '#c0596c', 'frame': GOLD}
PASS_L = {'same': '#d6e0f3', 'cross': '#efd3d8', 'frame': GOLD_L}


def box(ax, x, y, w, h, text='', fc='white', ec=INK, lw=0.6, ls='-', fs=FS, z=3, color=INK,
        stack=0):
    for k in range(stack, -1, -1):
        off = 0.06 * k
        ax.add_patch(FancyBboxPatch((x + off, y + off), w, h,
                                    boxstyle='round,pad=0.02,rounding_size=0.05',
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=z - k))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color,
                zorder=z + 1)


def arr(ax, p, q, color=INK_2, lw=0.6, ls='-', ms=5, z=4, style='-|>', rad=0.0):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        zorder=z, shrinkA=1, shrinkB=1, connectionstyle=f'arc3,rad={rad}')
    a.set_linestyle(ls)
    ax.add_patch(a)


def panel(ax, letter, title):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.text(0.0, 10.0, letter, fontsize=FS + 2.5, weight='bold', ha='left', va='top', color=INK)
    ax.text(0.7, 9.98, title, fontsize=FS - 0.2, ha='left', va='top', color=INK, linespacing=1.25)


# ------------------------------------------------------------ attention
def att_flow(ax):
    panel(ax, 'A', 'flow diagram\n(Vaswani et al. style)')
    # inputs and per-stream linear projections
    for x, col, ec, lab, src in ((2.3, BLUE_L, BLUE, '$h^x$', r'$W^x$ from $\mathrm{LM}_x$'),
                                 (7.7, RED_L, RED, '$h^y$', r'$W^y$ from $\mathrm{LM}_y$')):
        ax.text(x, 0.3, lab, fontsize=FS, ha='center', va='center', color=INK)
        for k, nm in enumerate('QKV'):
            xx = x - 1.55 + k * 1.05
            box(ax, xx, 0.95, 0.95, 0.55, f'Lin. {nm}', fc=col, ec=INK, ls=DOT, fs=FS - 1.8,
                stack=2)
            arr(ax, (x, 0.55), (xx + 0.47, 0.9), color=ec)
        ax.text(x, 1.95, src, fontsize=FS - 1.7, ha='center', va='center', color=INK_2)
    # three SDPA branches, mask box under each
    names = [('same', 'same stream, causal', 1.0), ('cross', 'cross, earlier frames', 3.95),
             ('frame', 'same frame', 6.9)]
    for key, lab, x in names:
        box(ax, x, 4.35, 2.3, 1.0, f'SDPA\n{lab}', fc=PASS_L[key], ec=INK, fs=FS - 1.7, stack=2)
        box(ax, x + 0.45, 3.35, 1.4, 0.5, 'Mask', fc='white', ec=INK, fs=FS - 1.2)
        arr(ax, (x + 1.15, 3.9), (x + 1.15, 4.3))
        arr(ax, (2.3, 2.25), (x + 0.7, 3.3), color=BLUE, lw=0.5, style='-')
        arr(ax, (7.7, 2.25), (x + 1.6, 3.3), color=RED, lw=0.5, style='-')
    ax.text(9.55, 5.75, 'multi-head\n(stacked)', fontsize=FS - 2.0, ha='center', va='center', color=INK_2)
    # gates and sum
    for x, g in ((5.1, '$g$'), (8.05, '$f$')):
        box(ax, x - 0.3, 6.0, 0.6, 0.6, g, fc=GOLD_L, ec=GOLD, fs=FS - 0.6)
        arr(ax, (x, 5.45), (x, 5.95))
        arr(ax, (x, 6.6), (5.0, 7.55))
    arr(ax, (2.15, 5.45), (5.0, 7.55))
    ax.add_patch(Circle((5.0, 7.75), 0.25, fc='white', ec=INK, lw=0.6, zorder=4))
    ax.text(5.0, 7.75, '+', fontsize=FS + 1, ha='center', va='center', zorder=5)
    box(ax, 4.0, 8.35, 2.0, 0.6, 'Linear $W_O$', fc=GREY_L, ec=INK, ls=DOT, fs=FS - 1.0)
    arr(ax, (5.0, 8.0), (5.0, 8.3))
    arr(ax, (5.0, 8.95), (5.0, 9.45))
    ax.text(5.35, 9.3, 'out', fontsize=FS - 1.0, ha='left', va='center', color=INK)
    ax.text(6.2, 8.65, 'per stream', fontsize=FS - 1.8, ha='left', va='center', color=INK_2)


def att_matrix(ax):
    panel(ax, 'B', 'admission matrix\n(UniLM / Longformer style)')
    intra, cross, frame, clean_len = masks(3)
    L = clean_len + 2
    labels = [rf'$x_{i // 2 + 1}$' if i % 2 == 0 else rf'$y_{i // 2 + 1}$' for i in range(clean_len)]
    labels += [r'$q_x$', r'$q_y$']
    ox, oy, cs = 1.5, 1.0, 0.8
    for i in range(L):
        for j in range(L):
            key = 'same' if intra[i, j] else 'cross' if cross[i, j] else 'frame' if frame[i, j] else None
            x, y = ox + j * cs, oy + (L - 1 - i) * cs
            if key:
                ax.add_patch(Rectangle((x, y), cs, cs, fc=PASS_L[key], ec=PASS_C[key], lw=0.4))
            else:
                blocked = j > i and i < clean_len
                ax.add_patch(Rectangle((x, y), cs, cs, fc='white', ec='#dddad6', lw=0.25,
                                       hatch='////' if blocked else ''))
        c = GOLD if i >= clean_len else (BLUE if i % 2 == 0 else RED)
        ax.text(ox - 0.15, oy + (L - 1 - i) * cs + cs / 2, labels[i], fontsize=FS - 1.2,
                ha='right', va='center', color=c)
        ax.text(ox + i * cs + cs / 2, oy + L * cs + 0.1, labels[i], fontsize=FS - 1.2,
                ha='center', va='bottom', color=c)
    ax.plot([ox, ox + L * cs], [oy + 2 * cs, oy + 2 * cs], color=INK, lw=0.5)
    ax.plot([ox + clean_len * cs] * 2, [oy, oy + L * cs], color=INK, lw=0.5)
    ax.text(ox + L * cs / 2, 0.4, 'keys', fontsize=FS - 1.0, ha='center', color=INK_2)
    ax.text(0.45, oy + L * cs / 2, 'queries', fontsize=FS - 1.0, ha='center', va='center',
            rotation=90, color=INK_2)
    for k, (key, lab) in enumerate((('same', 'same stream,\ncausal'),
                                    ('cross', 'cross stream,\nearlier frames'),
                                    ('frame', 'same frame'))):
        y = 7.6 - k * 0.95
        ax.add_patch(Rectangle((8.15, y), 0.4, 0.36, fc=PASS_L[key], ec=PASS_C[key], lw=0.4))
        ax.text(8.65, y + 0.18, lab, fontsize=FS - 2.0, ha='left', va='center', color=INK,
                linespacing=1.1)
    ax.add_patch(Rectangle((8.15, 4.75), 0.4, 0.36, fc='white', ec='#cfcbc6', lw=0.4, hatch='////'))
    ax.text(8.65, 4.93, 'blocked', fontsize=FS - 2.0, ha='left', va='center', color=INK)
    ax.text(8.15, 3.9, 'one map;\ncolour = the\npass that\nadmits the pair', fontsize=FS - 1.8,
            ha='left', va='top', color=INK_2, linespacing=1.15)


def att_arcs(ax):
    panel(ax, 'C', 'reach arcs over the token row\n(Transformer-XL style)')
    toks = ['$x_1$', '$y_1$', '$x_2$', '$y_2$', '$x_3$', '$y_3$', '$q_x$', '$q_y$']
    cols = [BLUE, RED] * 3 + [GOLD, GOLD]
    fcs = [BLUE_L, RED_L] * 3 + [GOLD_L, GOLD_L]
    xs = [1.0 + i * 1.12 for i in range(8)]
    y0 = 4.3
    for x, t, c, f in zip(xs, toks, cols, fcs):
        ax.add_patch(Rectangle((x - 0.38, y0), 0.76, 0.76, fc=f, ec=c, lw=0.6, zorder=3))
        ax.text(x, y0 - 0.4, t, fontsize=FS - 0.8, ha='center', va='center', color=INK)

    def arc(i, j, key, above=True, lw=0.7):
        xi, xj = xs[i], xs[j]
        c = (xi + xj) / 2; r = abs(xi - xj) / 2
        h = min(0.35 + r * 0.45, 2.2)
        y = y0 + 0.76 if above else y0 - 0.55
        a = Arc((c, y), 2 * r, 2 * h, theta1=0 if above else 180, theta2=180 if above else 360,
                color=PASS_C[key], lw=lw, zorder=2)
        a.set_linestyle('-' if key != 'cross' else DASH)
        ax.add_patch(a)

    for j in (0, 2):
        arc(4, j, 'same')
    for j in (1, 3):
        arc(4, j, 'cross')
    arc(4, 5, 'frame')
    ax.text(xs[4], y0 + 3.2, 'query $x_3$ (content)', fontsize=FS - 1.4, ha='center', color=INK)
    for j in (0, 2, 4):
        arc(6, j, 'same', above=False)
    for j in (1, 3, 5):
        arc(6, j, 'cross', above=False)
    arc(6, 7, 'frame', above=False)
    ax.text(xs[6] - 1.0, y0 - 3.4, 'query $q_x$ (harmonizer of frame 4)', fontsize=FS - 1.4,
            ha='center', color=INK)
    for k, (key, lab) in enumerate((('same', 'same stream'), ('cross', 'cross stream, earlier'),
                                    ('frame', 'same frame'))):
        ax.plot([0.4, 1.0], [8.7 - k * 0.42] * 2, color=PASS_C[key], lw=0.9,
                ls='-' if key != 'cross' else DASH)
        ax.text(1.15, 8.7 - k * 0.42, lab, fontsize=FS - 1.9, va='center', color=INK)


# --------------------------------------------------------------- router
def expert_row(ax, y, n=4, x0=1.6, w=1.4, gap=0.6, colour_by_source=False, sub=None):
    xs = []
    for i in range(n):
        x = x0 + i * (w + gap)
        fc = (BLUE_L if i < 2 else RED_L) if colour_by_source else 'white'
        lab = rf'$\mathrm{{FFN}}_{i + 1}$' + (f'\n{sub[i]}' if sub else '')
        box(ax, x, y, w, 0.9, lab, fc=fc, ec=INK, ls=DOT, fs=FS - 0.4 if not sub else FS - 1.2)
        xs.append(x + w / 2)
    return xs


def router_switch(ax):
    panel(ax, 'D', 'Switch-Transformer style:\nrouter fans a token out to the top-k experts')
    ps = [0.61, 0.27, 0.08, 0.04]
    xs = expert_row(ax, 5.0, sub=[f'$p={p:.2f}$' for p in ps])
    box(ax, 3.9, 2.2, 2.2, 0.7, 'router', fc=GREY_L, ec=INK, fs=FS - 0.4)
    ax.text(5.0, 0.5, 'token $h$', fontsize=FS - 0.4, ha='center', va='center', color=INK)
    arr(ax, (5.0, 0.85), (5.0, 2.15))
    for x, p in zip(xs, ps):
        top = p >= 0.27
        c = INK if top else '#b8b3ad'
        arr(ax, (5.0, 2.9), (x, 4.95), color=c, lw=0.8 if top else 0.5, ls='-' if top else DASH)
        arr(ax, (x, 5.9), (5.0, 7.3), color=c, lw=0.8 if top else 0.5, ls='-' if top else DASH)
    ax.add_patch(Circle((5.0, 7.55), 0.25, fc='white', ec=INK, lw=0.6, zorder=4))
    ax.text(5.0, 7.55, '+', fontsize=FS + 1, ha='center', va='center', zorder=5)
    ax.text(5.0, 8.4, r'$y=\sum_{i\in\mathrm{top}\!-\!2} p_i\,\mathrm{FFN}_i(h)$',
            fontsize=FS - 0.4, ha='center', va='center', color=INK)
    ax.text(8.4, 2.5, 'top-2 solid,\nothers dashed', fontsize=FS - 1.8, ha='center', va='center',
            color=INK_2)


def router_mixtral(ax):
    panel(ax, 'E', 'Mixtral style: gating network,\ngate weights as a bar over the experts')
    xs = expert_row(ax, 3.9)
    ws = [0.61, 0.27, 0.08, 0.04]
    base = 5.05
    for x, wgt in zip(xs, ws):
        top = wgt >= 0.27
        ax.add_patch(Rectangle((x - 0.45, base), 0.9, 2.2 * wgt, fc=GOLD if top else '#d9d5cf',
                               ec='none', zorder=3))
        ax.text(x, base + 2.2 * wgt + 0.12, f'{wgt:.2f}', fontsize=FS - 1.8, ha='center',
                va='bottom', color=INK if top else '#9a948e')
    ax.plot([1.3, 9.3], [base, base], color=INK_2, lw=0.5)
    ax.text(9.4, base, '$G(h)$', fontsize=FS - 1.0, ha='left', va='center', color=INK)
    box(ax, 3.6, 7.35, 2.8, 0.7, 'gating network $G$', fc=GREY_L, ec=INK, fs=FS - 0.6)
    ax.text(5.0, 8.65, 'token $h$', fontsize=FS - 0.4, ha='center', va='center', color=INK)
    arr(ax, (5.0, 8.4), (5.0, 8.1))
    ax.text(5.0, 7.05, 'softmax over experts, keep top-2', fontsize=FS - 1.8, ha='center',
            va='center', color=INK_2)
    for x, wgt in zip(xs, ws):
        top = wgt >= 0.27
        arr(ax, (x, 3.85), (5.0, 2.55), color=INK if top else '#b8b3ad',
            lw=0.8 if top else 0.5, ls='-' if top else DASH)
    ax.add_patch(Circle((5.0, 2.3), 0.25, fc='white', ec=INK, lw=0.6, zorder=4))
    ax.text(5.0, 2.3, '+', fontsize=FS + 1, ha='center', va='center', zorder=5)
    ax.text(5.0, 1.25, r'$y=\sum_{i\in\mathrm{top}\!-\!2(G(h))} G(h)_i\,\mathrm{FFN}_i(h)$',
            fontsize=FS - 0.4, ha='center', va='center', color=INK)
    ax.text(5.0, 0.45, 'the bar is the usual way to show the gate output', fontsize=FS - 1.8,
            ha='center', va='center', color=INK_2)


def router_duet(ax):
    panel(ax, 'F', 'ours: one router per stream, one shared pool,\nexperts coloured by source')
    xs = expert_row(ax, 5.4, colour_by_source=True)
    ax.text(5.0, 6.85, r'shared pool: $\mathrm{FFN}_{1,2}$ copied from $\mathrm{LM}_x$, '
            r'$\mathrm{FFN}_{3,4}$ from $\mathrm{LM}_y$', fontsize=FS - 1.6, ha='center',
            va='center', color=INK_2)
    ax.text(5.0, 8.0, r'$y^s=\sum_{i\in\mathrm{top}\!-\!2(G^s(h))} G^s(h)_i\,\mathrm{FFN}_i(h),'
            r'\quad s\in\{x,y\}$', fontsize=FS - 0.6, ha='center', va='center', color=INK)
    ax.text(5.0, 8.75, 'the routers are the only stream-specific part of the FFN sublayer',
            fontsize=FS - 1.8, ha='center', va='center', color=INK_2)
    for x_r, col, ec, lab, tok in ((2.6, BLUE_L, BLUE, 'router$_x$', '$h^x$'),
                                   (7.4, RED_L, RED, 'router$_y$', '$h^y$')):
        box(ax, x_r - 1.0, 2.7, 2.0, 0.7, lab, fc=col, ec=ec, fs=FS - 0.4)
        ax.text(x_r, 1.05, tok, fontsize=FS - 0.4, ha='center', va='center', color=INK)
        arr(ax, (x_r, 1.4), (x_r, 2.65), color=ec)
        for x in xs:
            arr(ax, (x_r, 3.4), (x, 5.35), color=ec, lw=0.5, ls=DASH)
    ax.text(5.0, 0.35, 'both streams pick top-2 of the same four experts; aux loss per router',
            fontsize=FS - 2.0, ha='center', va='center', color=INK_2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
    att_flow(axes[0][0]); att_matrix(axes[0][1]); att_arcs(axes[0][2])
    router_switch(axes[1][0]); router_mixtral(axes[1][1]); router_duet(axes[1][2])
    fig.text(0.01, 0.985, 'attention layer', fontsize=FS + 1, weight='bold', va='top', color=INK_2)
    fig.text(0.01, 0.495, 'router / mixture of experts', fontsize=FS + 1, weight='bold',
             va='top', color=INK_2)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.955, bottom=0.02, wspace=0.1, hspace=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')


if __name__ == '__main__':
    main()
