"""Model figure in the brief style of the original sketch: left, one
transformer block of the global stack with the interleaved hidden
states under it and the loss; right, the attention as masked passes
that reuse each token's per-stream Q, K, V, with the masks taken from
the model's own mask builder.

Conventions copied from the sketch: a DOTTED border means "initialised
from the pretrained single-stream model"; stream colour on the hidden
states and on the per-stream blocks; the mask matrices show kept cells
in the query's stream colour and blocked cells hatched. Extended for
Duet: a third, same-frame pass; two harmonizers appended to the
sequence (gold); a router per stream; gates on the two cross-stream
passes.

Usage:
  python plot_arch_brief.py --out results/fig_model_brief [--no-lora]
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_e1_box import SURFACE, INK, INK_2, PALETTE  # noqa: E402

BLUE, RED, GOLD, GREY = PALETTE['slate'], PALETTE['maroon'], PALETTE['gold'], PALETTE['grey']
BLUE_L, RED_L, GOLD_L, GREY_L = '#c9d9f3', '#ecc9cf', '#fbe6b3', '#e3e0dc'
BLOCK = '#ecebe8'
FS = 6.4


def rbox(ax, x, y, w, h, text='', fc=GREY_L, ec=INK, lw=0.6, ls='-', fs=FS,
         color=INK, z=3, pad=0.02, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad},rounding_size=0.04',
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
                color=color, zorder=z + 1, weight=weight)


def arrow(ax, p, q, color=INK_2, lw=0.6, ms=5, z=4, rad=0.0, style='-|>'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, zorder=z, shrinkA=1, shrinkB=1,
                                 connectionstyle=f'arc3,rad={rad}'))


# ---------------------------------------------------------------- left
def draw_block(ax, lora=True):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    dotted = (0, (1.2, 1.2))

    # legend (top-left)
    lx, ly = 0.25, 9.3
    ax.add_patch(Rectangle((lx, ly), 0.4, 0.3, fc='white', ec=INK, lw=0.6, ls=dotted))
    ax.text(lx + 0.55, ly + 0.15, 'initialised from\npretrained model',
            fontsize=FS - 1.2, va='center', color=INK)
    ax.add_patch(Rectangle((lx, ly - 0.6), 0.4, 0.3, fc=BLUE_L, ec=BLUE, lw=0.5))
    ax.text(lx + 0.55, ly - 0.45, 'melody hidden state', fontsize=FS - 1.2, va='center', color=INK)
    ax.add_patch(Rectangle((lx, ly - 1.1), 0.4, 0.3, fc=RED_L, ec=RED, lw=0.5))
    ax.text(lx + 0.55, ly - 0.95, 'chord hidden state', fontsize=FS - 1.2, va='center', color=INK)
    ax.add_patch(Rectangle((lx, ly - 1.6), 0.4, 0.3, fc=GOLD_L, ec=GOLD, lw=0.6))
    ax.text(lx + 0.55, ly - 1.45, 'harmonizer (added)', fontsize=FS - 1.2, va='center', color=INK)

    # block frame
    bx, by, bw, bh = 3.65, 3.2, 6.2, 6.3
    rbox(ax, bx, by, bw, bh, fc='white', ec=INK, lw=0.7, z=2)
    ax.text(bx + bw / 2, by + bh - 0.3, r'global-stack block $\times L$', fontsize=FS + 0.6,
            ha='center', va='center', color=INK)

    # expert pool
    ey = by + bh - 2.0
    rbox(ax, bx + 0.25, ey, bw - 0.5, 1.45, fc=BLOCK, ec='none', z=2)
    ax.text(bx + bw / 2, ey + 1.22, 'expert pool  (top-2 per token)', fontsize=FS, ha='center',
            va='center', color=INK)
    ew = (bw - 0.5 - 0.5 * 3 - 0.4) / 4
    for i in range(4):
        rbox(ax, bx + 0.45 + i * (ew + 0.5), ey + 0.2, ew, 0.75, r'$\mathrm{FFN}_%d$' % (i + 1),
             fc='white', ec=INK, ls=dotted, lw=0.6)
    # routers
    ry = ey - 0.85
    rbox(ax, bx + 1.3, ry, 1.4, 0.5, 'router', fc=BLUE_L, ec=BLUE, lw=0.5, fs=FS - 0.4)
    rbox(ax, bx + bw - 2.7, ry, 1.4, 0.5, 'router', fc=RED_L, ec=RED, lw=0.5, fs=FS - 0.4)
    ax.text(bx + bw / 2, ry - 0.2, 'one router per stream', fontsize=FS - 1.2, ha='center', va='center', color=INK_2)
    for x in (bx + 2.0, bx + bw - 2.0):
        arrow(ax, (x, ry + 0.5), (x, ey - 0.02), color=INK_2)
    # attention bar
    ay = ry - 0.85
    rbox(ax, bx + 0.25, ay, bw - 0.5, 0.55, fc=BLOCK, ec='none', z=2)
    ax.text(bx + bw / 2, ay + 0.28, 'Duet attention: three masked passes', fontsize=FS,
            ha='center', va='center', color=INK)
    # gold additions
    gy = ay - 0.75
    rbox(ax, bx + 0.25, gy, 1.2, 0.5, 'gates $g,f$', fc=GOLD_L, ec=GOLD, lw=0.6, fs=FS - 1.0)
    rbox(ax, bx + 1.6, gy, 2.95, 0.5, 'same-frame pass', fc=GOLD_L, ec=GOLD, lw=0.6, fs=FS - 1.0)
    if lora:
        rbox(ax, bx + 4.7, gy, bw - 4.95, 0.5, r'$\Delta W$ low-rank', fc=GOLD_L, ec=GOLD, lw=0.6, fs=FS - 1.2)
    # qkv per stream
    qy = gy - 0.75
    rbox(ax, bx + 0.25, qy, (bw - 0.7) / 2, 0.55, r'$W_Q^x, W_K^x, W_V^x, W_O^x$', fc=BLUE_L,
         ec=INK, ls=dotted, lw=0.6, fs=FS - 0.6)
    rbox(ax, bx + 0.45 + (bw - 0.7) / 2, qy, (bw - 0.7) / 2, 0.55,
         r'$W_Q^y, W_K^y, W_V^y, W_O^y$', fc=RED_L, ec=INK, ls=dotted, lw=0.6, fs=FS - 0.6)
    ax.text(bx + bw / 2, qy - 0.22, 'one projection set per pretrained model',
            fontsize=FS - 1.1, ha='center', va='center', color=INK_2)

    # hidden states below the block: interleaved x, y, ... then q_x, q_y
    hy = 1.7
    n = 3
    xs = []
    x = bx + 0.1
    for i in range(n):
        for col, ec, lab in ((BLUE_L, BLUE, rf'$x_{i + 1}$'), (RED_L, RED, rf'$y_{i + 1}$')):
            ax.add_patch(Rectangle((x, hy), 0.55, 0.55, fc=col, ec=ec, lw=0.5, zorder=3))
            ax.text(x + 0.275, hy - 0.25, lab, fontsize=FS - 0.8, ha='center', va='center', color=INK)
            xs.append(x + 0.275); x += 0.68
    ax.text(x + 0.02, hy + 0.27, r'$\cdots$', fontsize=FS + 1, va='center', color=INK); x += 0.5
    for col, ec, lab in ((GOLD_L, GOLD, r'$q_x$'), (GOLD_L, GOLD, r'$q_y$')):
        ax.add_patch(Rectangle((x, hy), 0.55, 0.55, fc=col, ec=ec, lw=0.6, zorder=3))
        ax.text(x + 0.275, hy - 0.25, lab, fontsize=FS - 0.8, ha='center', va='center', color=INK)
        xs.append(x + 0.275); x += 0.68
    # wires from hidden states into the matching projection box
    left_c, right_c = bx + 0.25 + (bw - 0.7) / 4, bx + 0.45 + 3 * (bw - 0.7) / 4
    for i, xc in enumerate(xs):
        gold = i >= 2 * n
        tgt = left_c if (i % 2 == 0) else right_c
        col = GOLD if gold else (BLUE if i % 2 == 0 else RED)
        arrow(ax, (xc, hy + 0.57), (tgt + (i - n) * 0.12, qy - 0.02), color=col, lw=0.5,
              ms=4, rad=0.0, style='-')
    ax.text(bx + bw / 2, hy - 0.7, 'interleaved content tokens; harmonizers',
            fontsize=FS - 1.1, ha='center', va='center', color=INK_2)

    # loss
    ax.text(bx + bw / 2, 0.55,
            r'$\mathcal{L}=\mathcal{L}_{\mathrm{AR}}+\lambda\,\mathcal{L}_{\mathrm{harm}}'
            r'+\lambda_{\mathrm{aux}}\,\mathcal{L}_{\mathrm{aux}}$',
            fontsize=FS + 2, ha='center', va='center', color=INK)
    ax.text(bx + bw / 2, 0.12,
            r'$\mathcal{L}_{\mathrm{AR}}$: CE at content tokens; '
            r'$\mathcal{L}_{\mathrm{harm}}$: CE at harmonizers',
            fontsize=FS - 1.5, ha='center', va='center', color=INK_2)


# --------------------------------------------------------------- right
def masks(T):
    from cp_transformer_m2c_duet_block import M2CDuetBlockLayer
    layer = M2CDuetBlockLayer(hidden_size=16, num_heads=2, intermediate_size=32,
                              moe_num_experts=1, moe_topk=1, moe_intermediate_size=32)
    clean_len = 2 * T
    intra, cross, frame = layer._build_masks(clean_len, T, torch.device('cpu'))
    diag = np.eye(clean_len + 2, dtype=bool)
    cross = cross.numpy() & ~diag                  # drop the numerical self guard
    return intra.numpy(), cross, frame.numpy(), clean_len


def cell_colors(i, clean_len):
    if i >= clean_len:
        return GOLD_L, GOLD
    return (BLUE_L, BLUE) if i % 2 == 0 else (RED_L, RED)


def draw_mask(ax, m, clean_len, title, labels):
    L = m.shape[0]
    ax.set_xlim(-0.5, L - 0.5); ax.set_ylim(L - 0.5, -0.5); ax.set_aspect('equal')
    for i in range(L):
        for j in range(L):
            if m[i, j]:
                fc, ec = cell_colors(i, clean_len)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=fc, ec=ec, lw=0.3))
            else:
                blocked_by_causality = j > i and i < clean_len
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc='white', ec='#dddad6',
                                       lw=0.25, hatch='////' if blocked_by_causality else ''))
    ax.axhline(clean_len - 0.5, color=INK, lw=0.5); ax.axvline(clean_len - 0.5, color=INK, lw=0.5)
    ax.set_xticks(range(L)); ax.set_yticks(range(L))
    ax.set_xticklabels(labels, fontsize=FS - 2.4); ax.set_yticklabels(labels, fontsize=FS - 2.4)
    for lab, i in zip(ax.get_xticklabels() + ax.get_yticklabels(), list(range(L)) * 2):
        lab.set_color(cell_colors(i, clean_len)[1])
    ax.tick_params(length=0, pad=0.8)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.5)
    ax.set_title(title, fontsize=FS - 0.6, color=INK, pad=8)


def stack(ax, x, y_top, L, clean_len, w=0.4, h=0.15, step=0.2):
    for i in range(L):
        fc, ec = cell_colors(i, clean_len)
        ax.add_patch(Rectangle((x, y_top - i * step), w, h, fc=fc, ec=ec, lw=0.3))


def draw_attention(fig, rect, lora=True):
    """rect: [x0, y0, w, h] in figure fraction for the whole right panel."""
    x0, y0, w, h = rect
    T = 4
    intra, cross, frame, clean_len = masks(T)
    L = clean_len + 2
    labels = [rf'$x_{i // 2 + 1}$' if i % 2 == 0 else rf'$y_{i // 2 + 1}$' for i in range(clean_len)]
    labels += [r'$q_x$', r'$q_y$']
    ax = fig.add_axes(rect); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.text(5, 9.8, "Duet attention: three masked passes reuse each token's per-stream Q, K, V",
            fontsize=FS + 0.4, ha='center', va='center', color=INK, weight='bold')
    # legend, bottom-left under the Q K V note
    lx = 0.25
    for k, (fc, ec, txt) in enumerate(((BLUE_L, BLUE, 'melody query, kept'),
                                       (RED_L, RED, 'chord query, kept'),
                                       (GOLD_L, GOLD, 'harmonizer query, kept'),
                                       ('white', '#cfcbc6', 'blocked by causality'))):
        yy = 1.55 - k * 0.42
        ax.add_patch(Rectangle((lx, yy, ), 0.3, 0.26, fc=fc, ec=ec, lw=0.4,
                               hatch='////' if txt.startswith('blocked') else ''))
        ax.text(lx + 0.4, yy + 0.13, txt, fontsize=FS - 1.6, va='center', color=INK)

    # Q K V stacks
    for k, name in enumerate(('Q', 'K', 'V')):
        xx = 0.45 + k * 0.6
        ax.text(xx + 0.2, 8.55, name, fontsize=FS, ha='center', va='center', color=INK)
        stack(ax, xx, 8.05, L, clean_len, w=0.42, h=0.3, step=0.42)
    ax.text(1.25, 3.65, 'per-stream projections\n(dotted boxes on the left)\nRoPE on Q, K'
            + ('\n$\\Delta W$ on cross-stream Q, K, V' if lora else ''),
            fontsize=FS - 1.5, ha='center', va='top', color=INK_2)

    # masks: three insets, tops in panel units
    names = [('same stream, causal', intra, r'$u_{\mathrm{same}}$', None),
             ('cross stream, earlier frames', cross, r'$u_{\mathrm{cross}}$', '$g$'),
             ('same frame', frame, r'$u_{\mathrm{frame}}$', '$f$')]
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    mh_in = 0.74                                     # mask edge in inches
    mw = mh_in / fw; mh = mh_in / fh                  # figure fractions
    tops_u = [8.85, 6.0, 3.15]                        # panel units
    ycs = []
    for (title, m, uname, gname), top_u in zip(names, tops_u):
        top = y0 + h * top_u / 10
        axm = fig.add_axes([x0 + w * 0.29, top - mh, mw, mh])
        draw_mask(axm, m, clean_len, title, labels)
        yc = top_u - (mh / h) * 10 / 2
        ycs.append(yc)
        right_u = (0.29 * w + mw) / w * 10
        arrow(ax, (2.3, 6.2), (2.55, yc), color=INK_2)
        ux = right_u + 0.55
        arrow(ax, (right_u + 0.05, yc), (ux - 0.05, yc), color=INK_2)
        stack(ax, ux, yc + 0.85, L, clean_len, w=0.4, h=0.14, step=0.19)
        ax.text(ux + 0.2, yc + 1.25, uname, fontsize=FS - 0.4, ha='center', va='center', color=INK)
        if gname:
            rbox(ax, ux + 0.75, yc - 0.25, 0.5, 0.5, gname, fc=GOLD_L, ec=GOLD, lw=0.6, fs=FS)
            arrow(ax, (ux + 0.42, yc), (ux + 0.72, yc), color=INK_2)
            arrow(ax, (ux + 1.27, yc), (7.75, 5.0), color=INK_2)
        else:
            arrow(ax, (ux + 0.42, yc), (7.75, 5.0), color=INK_2)
    # sum, W_O, output
    ax.add_patch(Circle((7.95, 5.0), 0.2, fc='white', ec=INK, lw=0.6, zorder=4))
    ax.text(7.95, 5.0, '+', fontsize=FS + 1, ha='center', va='center', color=INK, zorder=5)
    rbox(ax, 8.35, 4.72, 0.8, 0.56, r'$W_O$', fc='white', ec=INK, ls=(0, (1.2, 1.2)), lw=0.6, fs=FS)
    ax.text(8.75, 4.45, 'per stream', fontsize=FS - 1.6, ha='center', va='center', color=INK_2)
    arrow(ax, (8.15, 5.0), (8.33, 5.0), color=INK_2)
    arrow(ax, (9.17, 5.0), (9.38, 5.0), color=INK_2)
    stack(ax, 9.42, 5.85, L, clean_len, w=0.4, h=0.14, step=0.19)
    ax.text(9.62, 6.25, r'$o$', fontsize=FS, ha='center', va='center', color=INK)
    ax.text(9.62, 3.75, '+ residual, LN', fontsize=FS - 1.6, ha='center', va='center', color=INK_2)
    # formula
    ax.text(6.6, 0.85, r'$o = W_O\,(u_{\mathrm{same}} + g\,u_{\mathrm{cross}} + f\,u_{\mathrm{frame}})$',
            fontsize=FS + 0.4, ha='center', va='center', color=INK)
    ax.text(6.6, 0.35, r'$g=\sigma(w_g^{\top}h+b_g)$, $f=\sigma(w_f^{\top}h+b_f)$; '
            r'$b\approx-10$, so $g,f\approx0$ at init', fontsize=FS - 1.4, ha='center',
            va='center', color=INK_2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-lora', action='store_true')
    args = ap.parse_args()
    fig = plt.figure(figsize=(7.2, 3.6))
    fig.patch.set_facecolor(SURFACE)
    axl = fig.add_axes([0.0, 0.02, 0.34, 0.96])
    draw_block(axl, lora=not args.no_lora)
    draw_attention(fig, [0.35, 0.03, 0.64, 0.94], lora=not args.no_lora)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')


if __name__ == '__main__':
    main()
