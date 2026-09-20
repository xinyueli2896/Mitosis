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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Arc
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_e1_box import SURFACE, INK, INK_2, PALETTE  # noqa: E402

BLUE, RED, GOLD, GREY = PALETTE['slate'], PALETTE['maroon'], PALETTE['gold'], PALETTE['grey']
BLUE_L, RED_L, GOLD_L, GREY_L = '#c9d9f3', '#ecc9cf', '#fbe6b3', '#e3e0dc'
BLOCK = '#ecebe8'
PASS_C = {'same': '#6f8fc9', 'cross': '#c0596c', 'frame': GOLD}
PASS_L = {'same': '#d6e0f3', 'cross': '#efd3d8', 'frame': GOLD_L}
DASH = (0, (3, 1.5))
FS = 7.0


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
def draw_block(ax, lora=True, W=14.6, style='brief'):
    """Panel units: W wide x 10 tall, square units (the caller sizes the
    axes to match)."""
    ax.set_xlim(0, W); ax.set_ylim(0, 10); ax.axis('off')
    dotted = (0, (1.2, 1.2))

    # block frame
    bx, bw = (3.9, 8.4) if style == 'brief' else (2.85, 9.45)
    by, bh = (3.2, 6.4) if style == 'brief' else (2.35, 7.25)
    rbox(ax, bx, by, bw, bh, fc='white', ec=INK, lw=0.7, z=2)
    ax.text(bx + bw / 2, by + bh - 0.3, r'global-stack block $\times L$', fontsize=FS + 0.8,
            ha='center', va='center', color=INK)

    # the two pretrained models, whose weights fill the dotted parts
    px, pw = (0.3, 3.1) if style == 'brief' else (0.3, 2.3)
    for y_, col, ecol, nm, tag, ffn in () if style != 'brief' else ((7.7, BLUE_L, BLUE, 'x', 'melody', '1, 2'),
                                        (4.85, RED_L, RED, 'y', 'chord', '3, 4')):
        rbox(ax, px, y_, pw, 1.45, fc=col, ec=ecol, lw=0.6)
        ax.text(px + pw / 2, y_ + 1.15, f'pretrained $\\mathrm{{LM}}_{nm}$ ({tag})',
                fontsize=FS - 0.2, ha='center', va='center', color=INK)
        ax.text(px + pw / 2, y_ + 0.55,
                f'attention $W_{{Q,K,V,O}}\\;\\rightarrow\\;W^{nm}$\n'
                f'dense FFN $\\rightarrow$ $\\mathrm{{FFN}}_{{{ffn}}}$',
                fontsize=FS - 1.3, ha='center', va='center', color=INK, linespacing=1.4)
        arrow(ax, (px + pw, y_ + 0.72), (bx - 0.05, y_ + 0.72), color=ecol, style='-|>')
        ax.patches[-1].set_linestyle(dotted)
    if style == 'brief':
        ax.text(px + pw / 2, 7.0, 'weights copied into the dotted\nboxes of the same colour;\n'
                'routers, gates, harmonizers are new', fontsize=FS - 1.4, ha='center',
                va='center', color=INK_2)

    # legend (bottom-left)
    lx, ly = (0.3, 3.55) if style == 'brief' else (0.3, 9.0)
    for k, (fc, ec, ls, txt) in enumerate(((BLUE_L, BLUE, '-', 'stream $x$ (melody)'),
                                           (BLUE_L, INK, dotted, 'weights from $\\mathrm{LM}_x$'),
                                           (RED_L, RED, '-', 'stream $y$ (chord)'),
                                           (RED_L, INK, dotted, 'weights from $\\mathrm{LM}_y$'),
                                           (GOLD_L, GOLD, '-', 'harmonizer (new)'))):
        yy = ly - k * 0.5
        ax.add_patch(Rectangle((lx, yy), 0.4, 0.3, fc=fc, ec=ec, lw=0.6, ls=ls))
        ax.text(lx + 0.55, yy + 0.15, txt, fontsize=FS - 1.0, va='center', color=INK)

    # expert pool
    sp = 1.0                                      # room for the residual spine
    ps = (0.61, 0.27, 0.08, 0.04)                 # example gate values (a melody token)
    if style == 'brief':
        ey = by + bh - 2.05
        rbox(ax, bx + 0.25, ey, bw - 0.5 - sp, 1.45, fc=BLOCK, ec='none', z=2)
        ax.text(bx + (bw - sp) / 2, ey + 1.22,
                'expert pool: two copies of each pretrained FFN, top-2 per token',
                fontsize=FS - 0.5, ha='center', va='center', color=INK)
    else:
        ey = by + bh - 2.35
        rbox(ax, bx + 0.25, ey, bw - 0.5 - sp, 1.75, fc=BLOCK, ec='none', z=2)
    ew = (bw - 0.5 - sp - 0.5 * 3 - 0.6) / 4
    for i in range(4):
        lab = r'$\mathrm{FFN}_%d$' % (i + 1)
        if style != 'brief':
            lab += '\n$p{=}%.2f$' % ps[i]
        rbox(ax, bx + 0.55 + i * (ew + 0.5), ey + 0.2, ew, 0.75, lab,
             fc=BLUE_L if i < 2 else RED_L, ec=INK, ls=dotted, lw=0.6,
             fs=FS if style == 'brief' else FS - 1.2)
    ffn_cx = [bx + 0.55 + i * (ew + 0.5) + ew / 2 for i in range(4)]
    y_ffn_out = ey + 0.72
    if style != 'brief':
        # Switch-style: the token's top-2 experts (solid) are summed with
        # their gate values; the others (dashed) are skipped
        cxp = bx + 0.25 + (bw - 0.5 - sp) / 2
        y_ffn_out = ey + 1.45
        ax.add_patch(Circle((cxp, y_ffn_out), 0.17, fc='white', ec=INK, lw=0.6, zorder=5))
        ax.text(cxp, y_ffn_out, '+', fontsize=FS - 0.4, ha='center', va='center', color=INK,
                zorder=6)
        for i, cx_ in enumerate(ffn_cx):
            on = i in (0, 2)
            arrow(ax, (cx_, ey + 0.97), (cxp + (-0.14 if cx_ < cxp else 0.14), y_ffn_out - 0.1),
                  color=INK if on else '#b8b3ad', lw=0.7 if on else 0.45, ms=4)
            if not on:
                ax.patches[-1].set_linestyle(DASH)
        ax.text(bx + 0.45, y_ffn_out + 0.02, 'expert pool\n(pretrained FFN copies)',
                fontsize=FS - 1.5, ha='left', va='center', color=INK, linespacing=1.15)
        ax.text(bx + 0.25 + (bw - 0.5 - sp) - 0.2, y_ffn_out + 0.02,
                r'$y=\sum_{i\in\mathrm{top}\text{-}2} p_i\,\mathrm{FFN}_i(h)$',
                fontsize=FS - 1.3, ha='right', va='center', color=INK)
    # routers
    ry = ey - 0.85 if style == 'brief' else ey - 1.25
    rbox(ax, bx + 1.5, ry, 1.5, 0.5, 'router', fc=BLUE_L, ec=BLUE, lw=0.5, fs=FS - 0.4)
    rbox(ax, bx + bw - sp - 3.0, ry, 1.5, 0.5, 'router', fc=RED_L, ec=RED, lw=0.5, fs=FS - 0.4)
    if style == 'brief':
        ax.text(bx + (bw - sp) / 2, ry + 0.25, 'one router\nper stream', fontsize=FS - 1.2,
                ha='center', va='center', color=INK_2)
        for x in (bx + 2.25, bx + bw - sp - 2.25):
            arrow(ax, (x, ry + 0.5), (x, ey - 0.02), color=INK_2)
    else:
        # Switch-style fan-out: every router reaches every expert; its
        # top-2 (an example) solid, the rest dashed
        for x_r, col, top in ((bx + 2.25, BLUE, (0, 2)), (bx + bw - sp - 2.25, RED, ())):
            for i, cx in enumerate(ffn_cx):
                on = i in top
                arrow(ax, (x_r, ry + 0.5), (cx + (-0.12 if col == BLUE else 0.12), ey + 0.18),
                      color=col, lw=0.7 if on else 0.45, ms=4)
                if not on:
                    ax.patches[-1].set_linestyle(DASH)
        ax.text(bx + (bw - sp) / 2, ry - 0.28, 'one router per stream; top-2 solid',
                fontsize=FS - 1.6, ha='center', va='center', color=INK_2)
    if style == 'brief':
        # attention bar
        ay = ry - 0.8
        rbox(ax, bx + 0.25, ay, bw - 0.5 - sp, 0.55, fc=BLOCK, ec='none', z=2)
        ax.text(bx + (bw - sp) / 2, ay + 0.28, 'Duet attention: three masked passes',
                fontsize=FS, ha='center', va='center', color=INK)
        # gold additions
        gy = ay - 0.72
        gw = (bw - 0.5 - sp - 0.3 * 2) / 3 if lora else (bw - 0.5 - sp - 0.3) / 2
        labels = ['gates $g$, $f$', 'same-frame pass']
        if lora:
            labels.append(r'$\Delta W$ on cross $Q,K,V$')
        for i, lab in enumerate(labels):
            rbox(ax, bx + 0.25 + i * (gw + 0.3), gy, gw, 0.5, lab, fc=GOLD_L, ec=GOLD, lw=0.6,
                 fs=FS - 1.0)
        qy = gy - 0.78
        y_ln_att, y_add_att = gy + 0.25, ay + 0.28
    else:
        # the attention as its dataflow: per-stream Q/K/V -> three masked
        # passes -> gates on the two cross passes -> sum -> per-stream W_O
        aw = bw - 0.5 - sp
        cx = bx + 0.25 + aw / 2
        y_wo, y_sum, y_gate, y_sd, y_q = ry - 0.85, ry - 1.28, ry - 1.72, ry - 2.6, ry - 3.4
        ax.text(bx + 0.5, y_wo + 0.22, 'Duet attention\n(masks: see b)', fontsize=FS - 1.0,
                ha='left', va='center', color=INK, linespacing=1.2)
        # per-stream output projection, tied to the residual add on the spine
        rbox(ax, bx + 0.25 + aw / 2 - 1.55, y_wo, 3.1, 0.45, r'output projection $W_O^x$ / $W_O^y$',
             fc=GREY_L, ec=INK, ls=dotted, lw=0.6, fs=FS - 1.3)
        ax.plot([cx + 1.55, bx + bw - 0.25 - sp], [y_wo + 0.22] * 2, color=INK_2, lw=0.6, zorder=2)
        # sum
        ax.add_patch(Circle((cx, y_sum), 0.17, fc='white', ec=INK, lw=0.6, zorder=5))
        ax.text(cx, y_sum, '+', fontsize=FS - 0.4, ha='center', va='center', color=INK, zorder=6)
        arrow(ax, (cx, y_sum + 0.19), (cx, y_wo - 0.02), color=INK_2)
        # three passes
        sw = (aw - 0.5) / 3
        passes = (('same', 'same stream\ncausal', None), ('cross', 'cross stream\nearlier frames', '$g$'),
                  ('frame', 'same frame\nboth streams', '$f$'))
        for i, (key, lab, gate) in enumerate(passes):
            x0_ = bx + 0.25 + i * (sw + 0.25)
            for k in (2, 1, 0):                      # stacked = multi-head
                rbox(ax, x0_ + 0.05 * k, y_sd + 0.05 * k, sw, 0.62, lab if k == 0 else '',
                     fc=PASS_L[key], ec=PASS_C[key], lw=0.5, fs=FS - 1.6, z=4 - k)
            xm = x0_ + sw / 2
            if gate:
                ax.add_patch(Circle((xm, y_gate), 0.19, fc=GOLD_L, ec=GOLD, lw=0.6, zorder=5))
                ax.text(xm, y_gate, gate, fontsize=FS - 0.8, ha='center', va='center', color=INK,
                        zorder=6)
                arrow(ax, (xm, y_sd + 0.64), (xm, y_gate - 0.21), color=INK_2)
                arrow(ax, (xm, y_gate + 0.21), (cx + (-0.12 if xm < cx else 0.12), y_sum - 0.14),
                      color=INK_2)
            else:
                arrow(ax, (xm, y_sd + 0.64), (cx - 0.14, y_sum - 0.1), color=INK_2)
        ax.text(bx + 0.25 + aw - 0.05, y_gate, 'gates', fontsize=FS - 1.8, ha='right',
                va='center', color=INK_2)
        qy = y_q
        y_ln_att, y_add_att = qy + 0.27, y_wo + 0.22
    hw = (bw - 0.8 - sp) / 2
    rbox(ax, bx + 0.25, qy, hw, 0.55, r'$W_Q^x, W_K^x, W_V^x, W_O^x$' if style == 'brief' else r'$W_Q^x, W_K^x, W_V^x$', fc=BLUE_L,
         ec=INK, ls=dotted, lw=0.6, fs=FS - 0.4)
    rbox(ax, bx + 0.55 + hw, qy, hw, 0.55, r'$W_Q^y, W_K^y, W_V^y, W_O^y$' if style == 'brief' else r'$W_Q^y, W_K^y, W_V^y$', fc=RED_L,
         ec=INK, ls=dotted, lw=0.6, fs=FS - 0.4)

    # residual spine: pre-LN, sublayer, add (both streams share the layout)
    xs_ = bx + bw - sp / 2
    ax.plot([xs_, xs_], [qy + 0.1, ey + 1.45], color=INK_2, lw=0.7, zorder=2)
    arrow(ax, (xs_, ey + 1.3), (xs_, by + bh - 0.5), color=INK_2, lw=0.7)
    for y_ln, y_add, y_tie in ((y_ln_att, y_add_att, y_add_att),
                               (ry + 0.25, y_ffn_out, y_ffn_out)):
        rbox(ax, xs_ - 0.3, y_ln - 0.17, 0.6, 0.34, 'LN', fc='white', ec=INK_2, lw=0.5,
             fs=FS - 1.6, z=4)
        ax.add_patch(Circle((xs_, y_add), 0.17, fc='white', ec=INK_2, lw=0.6, zorder=4))
        ax.text(xs_, y_add, '+', fontsize=FS - 0.6, ha='center', va='center', color=INK, zorder=5)
        ax.plot([bx + bw - 0.25 - sp, xs_ - 0.18], [y_tie, y_tie], color=INK_2, lw=0.6, zorder=2)
    if style == 'brief':
        ax.text(xs_, qy - 0.1, 'residual', fontsize=FS - 1.8, ha='center', va='top', color=INK_2)
    else:
        ax.text(xs_ - 0.35, by + bh - 0.32, 'residual', fontsize=FS - 1.8, ha='right', va='center', color=INK_2)

    # hidden states below the block: interleaved x, y, ... then q_x, q_y
    hy = 1.75 if style == 'brief' else 1.25
    n = 4
    xs = []
    sq, step = 0.58, 0.74
    x = bx + 0.3
    for i in range(n):
        for col, ec, lab in ((BLUE_L, BLUE, rf'$x_{i + 1}$'), (RED_L, RED, rf'$y_{i + 1}$')):
            ax.add_patch(Rectangle((x, hy), sq, sq, fc=col, ec=ec, lw=0.5, zorder=3))
            ax.text(x + sq / 2, hy - 0.27, lab, fontsize=FS - 0.6, ha='center', va='center', color=INK)
            xs.append(x + sq / 2); x += step
    ax.text(x + 0.05, hy + sq / 2, r'$\cdots$', fontsize=FS + 1, va='center', color=INK); x += 0.6
    for col, ec, lab in ((GOLD_L, GOLD, r'$q_x$'), (GOLD_L, GOLD, r'$q_y$')):
        ax.add_patch(Rectangle((x, hy), sq, sq, fc=col, ec=ec, lw=0.6, zorder=3))
        ax.text(x + sq / 2, hy - 0.27, lab, fontsize=FS - 0.6, ha='center', va='center', color=INK)
        xs.append(x + sq / 2); x += step
    if style == 'brief':
        # wires from hidden states into the matching projection box
        left_c, right_c = bx + 0.25 + hw / 2, bx + 0.55 + 1.5 * hw
        for i, xc in enumerate(xs):
            gold = i >= 2 * n
            tgt = left_c if (i % 2 == 0) else right_c
            col = GOLD if gold else (BLUE if i % 2 == 0 else RED)
            arrow(ax, (xc, hy + sq + 0.02), (tgt + (i - n) * 0.12, qy - 0.02), color=col, lw=0.5,
                  ms=4, rad=0.0, style='-')
    else:
        # one arrow into the block; colour says which projection set a token uses
        xm = (xs[0] + xs[-1]) / 2
        arrow(ax, (xm, hy + sq + 0.05), (xm, by - 0.03), color=INK_2, lw=0.8, ms=6)
        ax.text(xm + 0.25, (hy + sq + by) / 2, 'colour = projection set ($q_x$: $W^x$, $q_y$: $W^y$)',
                fontsize=FS - 1.5, ha='left', va='center', color=INK_2)
    if style == 'brief':
        ax.text(bx + bw / 2, hy - 0.8,
                'shared sequence: the two streams interleaved by frame, then the harmonizers of frame $t$;\n'
                'wires: each token is projected by its own stream\'s $W$',
                fontsize=FS - 1.1, ha='center', va='center', color=INK_2)
    else:
        ax.text(bx + bw / 2, hy - 0.62,
                'interleaved streams, then the harmonizers of frame $t$',
                fontsize=FS - 1.4, ha='center', va='center', color=INK_2)

    if style == 'brief':
        # loss: formula under the sequence, its terms in the left column
        ax.text(bx + bw / 2, 0.42 if style == 'brief' else 0.22,
                r'$\mathcal{L}=\mathcal{L}_{\mathrm{AR}}+\lambda\,\mathcal{L}_{\mathrm{harm}}'
                r'+\lambda_{\mathrm{aux}}\,\mathcal{L}_{\mathrm{aux}}$',
                fontsize=FS + 1.4, ha='center', va='center', color=INK)
        ax.text(px + pw / 2, 0.6,
                r'$\mathcal{L}_{\mathrm{AR}}$: CE at content tokens' + '\n'
                r'$\mathcal{L}_{\mathrm{harm}}$: CE at harmonizers' + '\n'
                r'$\mathcal{L}_{\mathrm{aux}}$: expert load balance',
                fontsize=FS - 1.4, ha='center', va='center', color=INK_2)


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
    """rect: [x0, y0, w, h] in figure fraction for the whole right panel.
    Compact: the three masks stacked, each feeding its term of the gated
    sum on the right."""
    x0, y0, w, h = rect
    T = 4
    intra, cross, frame, clean_len = masks(T)
    L = clean_len + 2
    labels = [rf'$x_{i // 2 + 1}$' if i % 2 == 0 else rf'$y_{i // 2 + 1}$' for i in range(clean_len)]
    labels += [r'$q_x$', r'$q_y$']
    H = 16
    ax = fig.add_axes(rect); ax.set_xlim(0, 10); ax.set_ylim(0, H); ax.axis('off')
    ax.text(5, H - 0.35, 'Duet attention: three masked passes', fontsize=FS + 0.4, ha='center',
            va='center', color=INK, weight='bold')

    names = [('same stream, causal', intra, r'$u_{\mathrm{same}}$', None),
             ('cross stream, earlier frames', cross, r'$u_{\mathrm{cross}}$', '$g$'),
             ('same frame', frame, r'$u_{\mathrm{frame}}$', '$f$')]
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    mh_in = 0.66                                     # mask edge in inches
    mw = mh_in / fw; mh = mh_in / fh                  # figure fractions
    tops_u = [14.2, 9.95, 5.7]                        # panel units
    mleft_u = 1.35
    right_u = mleft_u + mw / w * 10
    sum_x, sum_y = 8.0, 8.9
    for (title, m, uname, gname), top_u in zip(names, tops_u):
        top = y0 + h * top_u / H
        axm = fig.add_axes([x0 + w * mleft_u / 10, top - mh, mw, mh])
        draw_mask(axm, m, clean_len, title, labels)
        yc = top_u - (mh / h) * H / 2
        ax.text(right_u + 0.1, yc + 0.9, uname, fontsize=FS - 0.4, ha='left', va='center',
                color=INK)
        if gname:
            rbox(ax, right_u + 0.45, yc - 0.4, 0.75, 0.8, gname, fc=GOLD_L, ec=GOLD, lw=0.6, fs=FS)
            arrow(ax, (right_u + 0.05, yc), (right_u + 0.42, yc), color=INK_2)
            arrow(ax, (right_u + 1.25, yc), (sum_x - 0.32, sum_y), color=INK_2)
        else:
            arrow(ax, (right_u + 0.05, yc), (sum_x - 0.32, sum_y), color=INK_2)
    # sum, W_O, output
    ax.add_patch(Circle((sum_x, sum_y), 0.3, fc='white', ec=INK, lw=0.6, zorder=4))
    ax.text(sum_x, sum_y, '+', fontsize=FS + 1, ha='center', va='center', color=INK, zorder=5)
    rbox(ax, sum_x - 0.85, sum_y + 1.0, 1.7, 1.15, '$W_O$\n(per stream)', fc='white', ec=INK,
         ls=(0, (1.2, 1.2)), lw=0.6, fs=FS - 1.2)
    arrow(ax, (sum_x, sum_y + 0.32), (sum_x, sum_y + 0.97), color=INK_2)
    arrow(ax, (sum_x, sum_y + 2.17), (sum_x, sum_y + 2.75), color=INK_2)
    ax.text(sum_x, sum_y + 3.05, r'$o$', fontsize=FS, ha='center', va='center', color=INK)
    ax.text(sum_x, sum_y + 3.55, '+ residual, LN', fontsize=FS - 1.6, ha='center', va='center',
            color=INK_2)
    # formula + legend
    ax.text(5, 1.55, r'$o = W_O\,(u_{\mathrm{same}} + g\,u_{\mathrm{cross}} + f\,u_{\mathrm{frame}})$',
            fontsize=FS + 0.2, ha='center', va='center', color=INK)
    ax.text(5, 1.0, r'$g,f=\sigma(\cdot)$, bias $-10$: both $\approx 0$ at init',
            fontsize=FS - 1.4, ha='center', va='center', color=INK_2)
    lx = 0.35
    for k, (fc, ec, txt) in enumerate(((BLUE_L, BLUE, 'melody query'),
                                       (RED_L, RED, 'chord query'),
                                       (GOLD_L, GOLD, 'harmonizer query'),
                                       ('white', '#cfcbc6', 'blocked (causality)'))):
        xx = lx + (k % 2) * 4.9; yy = 0.45 - (k // 2) * 0.4
        ax.add_patch(Rectangle((xx, yy), 0.4, 0.28, fc=fc, ec=ec, lw=0.4,
                               hatch='////' if txt.startswith('blocked') else ''))
        ax.text(xx + 0.5, yy + 0.14, txt, fontsize=FS - 1.6, va='center', color=INK)


def draw_arcs(fig, rect):
    """Panel (b): who attends to whom in one layer, as reach arcs over the
    token row: a content query above the row, a harmonizer query below,
    arcs coloured by pass. Admission rule from the model's own masks."""
    ax = fig.add_axes(rect); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    intra, cross, frame, clean_len = masks(3)
    L = clean_len + 2
    ax.text(0.0, 9.95, 'b', fontsize=FS + 2.5, weight='bold', ha='left', va='top', color=INK)
    ax.text(0.7, 9.93, 'who attends to whom (one layer, frames 1-3 committed,\n'
            'harmonizers drafting frame 4)', fontsize=FS - 0.4, ha='left', va='top',
            color=INK, linespacing=1.25)
    toks = ['$x_1$', '$y_1$', '$x_2$', '$y_2$', '$x_3$', '$y_3$', '$q_x$', '$q_y$']
    cols = [BLUE, RED] * 3 + [GOLD, GOLD]
    fcs = [BLUE_L, RED_L] * 3 + [GOLD_L, GOLD_L]
    xs = [0.85 + i * 1.2 for i in range(L)]
    y0 = 4.55
    for x, t, c, f in zip(xs, toks, cols, fcs):
        ax.add_patch(Rectangle((x - 0.4, y0), 0.8, 0.7, fc=f, ec=c, lw=0.6, zorder=3))
        ax.text(x, y0 - 0.38, t, fontsize=FS - 0.6, ha='center', va='center', color=INK)

    def key_of(i, j):
        return 'same' if intra[i, j] else 'cross' if cross[i, j] else 'frame' if frame[i, j] else None

    def arc(i, j, above):
        key = key_of(i, j)
        if key is None or i == j:
            return
        xi, xj = xs[i], xs[j]
        c = (xi + xj) / 2; r = abs(xi - xj) / 2
        h = min(0.4 + r * 0.5, 2.3)
        y = y0 + 0.7 if above else y0 - 0.55
        a = Arc((c, y), 2 * r, 2 * h, theta1=0 if above else 180, theta2=180 if above else 360,
                color=PASS_C[key], lw=0.8, zorder=2)
        a.set_linestyle('-' if key != 'cross' else DASH)
        ax.add_patch(a)

    qi, hi = 4, 6                                   # x_3 above, q_x below
    for j in range(L):
        arc(qi, j, True)
        arc(hi, j, False)
    ax.add_patch(Rectangle((xs[qi] - 0.46, y0 - 0.06), 0.92, 0.82, fc='none', ec=INK, lw=0.8,
                           zorder=4))
    ax.text(0.45, 8.15, 'above: content query $x_3$', fontsize=FS - 1.1, ha='left',
            va='center', color=INK)
    ax.text(0.45, 1.05, 'below: harmonizer query $q_x$', fontsize=FS - 1.1, ha='left',
            va='center', color=INK)
    ax.add_patch(Rectangle((xs[hi] - 0.46, y0 - 0.06), 0.92, 0.82, fc='none', ec=INK, lw=0.8,
                           zorder=4))
    # legend
    for k, (key, lab, gate) in enumerate((('same', 'same stream, causal', ''),
                                          ('cross', 'cross stream, earlier', r'$\times g$'),
                                          ('frame', 'same frame', r'$\times f$'))):
        yy = 8.75 - k * 0.4
        ax.plot([4.7, 5.3], [yy, yy], color=PASS_C[key], lw=1.0,
                ls='-' if key != 'cross' else DASH)
        ax.text(5.45, yy, lab + (f'  {gate}' if gate else ''), fontsize=FS - 1.6, va='center',
                color=INK)
    ax.text(5.0, 0.35, 'every token is a query of the same three passes; a content token\n'
            'never sees a harmonizer, and the two harmonizers see each other',
            fontsize=FS - 1.6, ha='center', va='center', color=INK_2, linespacing=1.2)


def draw_matrix(fig, rect, T=4):
    """Panel (b) as one admission matrix: queries on rows, keys on
    columns, cell colour = the pass that admits the pair (the three
    masks are disjoint), hatched = blocked by causality."""
    x0, y0, w, h = rect
    ax = fig.add_axes(rect); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.text(0.0, 9.95, 'b', fontsize=FS + 2.5, weight='bold', ha='left', va='top', color=INK)
    ax.text(0.7, 9.93, 'who attends to whom (one layer)',
            fontsize=FS - 0.6, ha='left', va='top', color=INK, linespacing=1.25)
    intra, cross, frame, clean_len = masks(T)
    L = clean_len + 2
    labels = [rf'$x_{i // 2 + 1}$' if i % 2 == 0 else rf'$y_{i // 2 + 1}$' for i in range(clean_len)]
    labels += [r'$q_x$', r'$q_y$']
    # square inset, sized in inches
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    edge_in = min(w * fw * 0.66, h * fh * 0.6)
    mw, mh = edge_in / fw, edge_in / fh
    axm = fig.add_axes([x0 + w * 0.14, y0 + h * 0.27, mw, mh])
    axm.set_xlim(-0.5, L - 0.5); axm.set_ylim(L - 0.5, -0.5); axm.set_aspect('equal')
    for i in range(L):
        for j in range(L):
            key = 'same' if intra[i, j] else 'cross' if cross[i, j] else 'frame' if frame[i, j] else None
            if key:
                axm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=PASS_L[key], ec=PASS_C[key],
                                        lw=0.35))
            else:
                blocked = j > i and i < clean_len
                axm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc='white', ec='#dddad6',
                                        lw=0.25, hatch='////' if blocked else ''))
    axm.axhline(clean_len - 0.5, color=INK, lw=0.6); axm.axvline(clean_len - 0.5, color=INK, lw=0.6)
    axm.set_xticks(range(L)); axm.set_yticks(range(L))
    axm.set_xticklabels(labels, fontsize=FS - 1.4); axm.set_yticklabels(labels, fontsize=FS - 1.4)
    for lab, i in zip(axm.get_xticklabels() + axm.get_yticklabels(), list(range(L)) * 2):
        lab.set_color(GOLD if i >= clean_len else (BLUE if i % 2 == 0 else RED))
    axm.tick_params(length=0, pad=1.5)
    axm.xaxis.tick_top()
    for sp_ in axm.spines.values():
        sp_.set_color(INK); sp_.set_linewidth(0.6)
    axm.xaxis.set_label_position('top')
    axm.set_xlabel('keys', fontsize=FS - 1.0, color=INK_2, labelpad=2)
    axm.set_ylabel('queries', fontsize=FS - 1.0, color=INK_2, labelpad=2)
    # legend, below the matrix
    for k, (key, lab, gate) in enumerate((('same', 'same stream, causal', ''),
                                          ('cross', 'cross stream, earlier', r'$\times g$'),
                                          ('frame', 'same frame', r'$\times f$'))):
        xx, yy = 0.7 + (k % 2) * 4.8, 1.55 - (k // 2) * 0.55
        ax.add_patch(Rectangle((xx, yy), 0.42, 0.32, fc=PASS_L[key], ec=PASS_C[key], lw=0.4))
        ax.text(xx + 0.55, yy + 0.16, lab + (f'  {gate}' if gate else ''), fontsize=FS - 1.6,
                va='center', color=INK)
    ax.add_patch(Rectangle((5.5, 1.0), 0.42, 0.32, fc='white', ec='#cfcbc6', lw=0.4, hatch='////'))
    ax.text(6.05, 1.16, 'blocked by causality', fontsize=FS - 1.6, va='center', color=INK)


def draw_decode(fig, rect):
    """Panel (c): alternating-commit decode of one frame t, as the
    inference schedule ctc_alt runs it: (1) with both harmonizers masked,
    draft the leader stream's frame from the committed prefix; (2) commit
    it into the leader's harmonizer and predict the follower from its
    masked harmonizer, which sees the leader through the same-frame pass;
    (3) commit both as content tokens; the leader alternates with frame
    parity."""
    ax = fig.add_axes(rect); ax.set_xlim(0, 30); ax.set_ylim(0, 8); ax.axis('off')
    ax.text(0.0, 7.85, 'c', fontsize=FS + 2.5, weight='bold', ha='left', va='top', color=INK)
    ax.text(0.9, 7.8, 'alternating-commit decode of frame $t$ ($x$ leads; $y$ leads at $t{+}1$)',
            fontsize=FS - 0.4, ha='left', va='top', color=INK)
    sq, gap = 0.78, 0.16
    y0 = 3.3

    def token(x, y, col, ec, lab='', masked=False, lw=0.6, z=3):
        ax.add_patch(Rectangle((x, y), sq, sq, fc='white' if masked else col, ec=ec, lw=lw,
                               hatch='////' if masked else '', zorder=z))
        if lab:
            ax.text(x + sq / 2, y - 0.42, lab, fontsize=FS - 1.4, ha='center', va='center',
                    color=INK)
        return x + sq / 2

    def prefix(x):
        xs = []
        for i, (col, ec, lab) in enumerate(((BLUE_L, BLUE, '$x_1$'), (RED_L, RED, '$y_1$'))):
            xs.append(token(x + i * (sq + gap), y0, col, ec, lab))
        ax.text(x + 2 * (sq + gap) + 0.2, y0 + sq / 2, r'$\cdots$', fontsize=FS, va='center',
                color=INK)
        x2 = x + 2 * (sq + gap) + 0.75
        for i, (col, ec, lab) in enumerate(((BLUE_L, BLUE, '$x_{t-1}$'), (RED_L, RED, '$y_{t-1}$'))):
            xs.append(token(x2 + i * (sq + gap), y0, col, ec, lab))
        ax.plot([x - 0.15, x2 + 2 * sq + gap + 0.05], [y0 - 0.85] * 2, color=INK_2, lw=0.5)
        ax.text((x - 0.15 + x2 + 2 * sq + gap) / 2, y0 - 1.15, 'committed prefix',
                fontsize=FS - 1.6, ha='center', va='center', color=INK_2)
        return x2 + 2 * (sq + gap) + 0.25, xs

    def step(x, n, title, q_state, out, note):
        ax.text(x, 6.6, f'{n}  {title}', fontsize=FS - 0.6, ha='left', va='center', color=INK,
                weight='bold')
        xh, pxs = prefix(x)
        # harmonizer pair
        cxs = []
        for k, (st, lab) in enumerate(zip(q_state, ('$q_x$', '$q_y$'))):
            xx = xh + k * (sq + gap)
            if st == 'mask':
                cxs.append(token(xx, y0, GOLD_L, GOLD, lab, masked=True))
            elif st == 'hold_x':
                cxs.append(token(xx, y0, BLUE_L, BLUE, lab, lw=0.9))
                ax.text(xx + sq / 2, y0 + sq / 2, '$x_t$', fontsize=FS - 1.2, ha='center',
                        va='center', color=INK, zorder=4)
            else:
                cxs.append(token(xx, y0, GOLD_L, GOLD, lab))
        if out:
            src, lab, col, ec = out
            src_x = {'prefix_last': pxs[-1], 'q_x': cxs[0], 'q_y': cxs[1]}[src]
            arrow(ax, (src_x, y0 + sq + 0.05), (src_x, y0 + sq + 1.0), color=INK_2)
            token(src_x - sq / 2, y0 + sq + 1.05, col, ec)
            ax.text(src_x + sq / 2 + 0.15, y0 + sq + 1.05 + sq / 2, lab, fontsize=FS - 1.2,
                    ha='left', va='center', color=INK)
        ax.text(x, 0.55, note, fontsize=FS - 1.6, ha='left', va='center', color=INK_2,
                linespacing=1.2)
        return xh, cxs

    # step 1: both harmonizers masked, leader drafted from the prefix
    xh, cxs = step(0.4, '1', 'draft the leader', ('mask', 'mask'),
                   ('prefix_last', r'$x_t \sim p(x_t\mid\mathrm{prefix})$', BLUE_L, BLUE),
                   'harmonizers masked')
    # the arrow for step 1 comes from the last prefix token of stream y (the
    # head at position 2t-1 predicts x_t): redraw out with that x
    # step 2: leader committed into its harmonizer, follower predicted
    xh2, cxs2 = step(10.6, '2', 'condition the follower', ('hold_x', 'mask'),
                     ('q_y', r'$y_t \sim p(y_t\mid x_t,\mathrm{prefix})$', RED_L, RED),
                     '$x_t$ written into $q_x$; $q_y$ reads it')
    cxa, cxb = cxs2
    ax.add_patch(Arc(((cxa + cxb) / 2, y0 + sq), cxb - cxa, 0.9, theta1=0, theta2=180,
                     color=GOLD, lw=0.9, zorder=2))
    ax.text(cxa - sq / 2 - 0.1, y0 + sq + 0.45, 'same-frame\npass', fontsize=FS - 1.9, ha='right',
            va='center', color=INK_2, linespacing=1.1)
    # step 3: commit both, roles swap
    ax.text(19.9, 6.6, '3  commit, swap roles', fontsize=FS - 0.6, ha='left', va='center',
            color=INK, weight='bold')
    xh3, _ = prefix(19.9)
    token(xh3, y0, BLUE_L, BLUE, '$x_t$', lw=0.9)
    token(xh3 + sq + gap, y0, RED_L, RED, '$y_t$', lw=0.9)
    ax.text(xh3 + 2 * (sq + gap) + 0.1, y0 + sq / 2, r'$\rightarrow$ at $t{+}1$,' + '\n$y$ leads',
            fontsize=FS - 1.5, ha='left', va='center', color=INK, linespacing=1.15)
    return cxs, cxs2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-lora', action='store_true')
    ap.add_argument('--style', choices=['masks', 'arcs', 'matrix'], default='matrix',
                    help='right panel: three mask matrices (old layout), reach arcs, or one admission matrix')
    args = ap.parse_args()
    if args.style == 'masks':
        fig = plt.figure(figsize=(7.2, 3.6))
        fig.patch.set_facecolor(SURFACE)
        lw_, lh_ = 0.69, 0.96
        W = lw_ * fig.get_figwidth() / (lh_ * fig.get_figheight()) * 10
        axl = fig.add_axes([0.005, 0.02, lw_, lh_])
        draw_block(axl, lora=not args.no_lora, W=W, style='brief')
        draw_attention(fig, [0.70, 0.03, 0.295, 0.94], lora=not args.no_lora)
    else:
        fig = plt.figure(figsize=(7.2, 5.0))
        fig.patch.set_facecolor(SURFACE)
        lw_, lh_ = 0.62, 0.69
        W = lw_ * fig.get_figwidth() / (lh_ * fig.get_figheight()) * 10
        # bottom-up reading order: the block (a) and the map (b) at the
        # bottom, the decode stage (c) on top
        axl = fig.add_axes([0.005, 0.01, lw_, lh_])
        draw_block(axl, lora=not args.no_lora, W=W, style='sublayer')
        axl.text(0.0, 9.95, 'a', fontsize=FS + 2.5, weight='bold', ha='left', va='top', color=INK)
        if args.style == 'arcs':
            draw_arcs(fig, [0.635, 0.015, 0.36, 0.68])
        else:
            draw_matrix(fig, [0.635, 0.015, 0.36, 0.68])
        draw_decode(fig, [0.01, 0.715, 0.98, 0.27])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')


if __name__ == '__main__':
    main()
