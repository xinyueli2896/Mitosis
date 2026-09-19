"""Model figure for the paper: (a) the Duet architecture as a schematic,
(b) the attention admission matrix of one layer, computed from the
model's own mask builder (M2CDuetBlockLayer._build_masks) for six
frames plus the two harmonizers of frame 7, so (b) is the real pattern
and not a drawing of it.

Vocabulary: stream, frame, content token, harmonizer, same-frame
pathway, cross-stream pathway, local encoder, global stack, local
decoder, expert, router, low-rank update, leader, follower.

Palette: melody blue, chord red, everything that binds the streams
gold, pretrained parts grey, ink near-black -- the E1 palette.

Usage:
  python plot_arch.py --out results/fig_model [--no-lora]
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_e1_box import SURFACE, INK, INK_2, GRID, PALETTE, grouped_legend  # noqa: E402

BLUE, RED, GOLD, GREY = PALETTE['slate'], PALETTE['maroon'], PALETTE['gold'], PALETTE['grey']
GREY_FILL = '#e6e3df'
FS = 6.8


def box(ax, x, y, w, h, text='', fc=GREY_FILL, ec=GREY, lw=0.7, fs=FS,
        weight='normal', color=INK, style='round,pad=0.02,rounding_size=0.06',
        z=3, ha='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=lw, zorder=z))
    if text:
        ax.text(x + (w / 2 if ha == 'center' else 0.06), y + h / 2, text,
                ha=ha, va='center', fontsize=fs, color=color, weight=weight,
                zorder=z + 1)


def arrow(ax, p, q, color=INK, lw=0.8, style='-|>', ms=6, z=4, ls='-',
          shrinkA=1, shrinkB=1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, ls=ls, zorder=z,
                                 shrinkA=shrinkA, shrinkB=shrinkB,
                                 connectionstyle=f'arc3,rad={rad}'))


def draw_arch(ax, lora=True):
    ax.set_xlim(-0.45, 10); ax.set_ylim(0, 7.0); ax.axis('off')

    # ---- content tokens (two rows) --------------------------------
    ax.text(0.05, 6.55, 'pretrained\nmodels', fontsize=FS - 0.8, color=INK,
            weight='bold', va='center')
    # two pretrained models as objects: each with its own attention and
    # feed-forward, each fed by its own stream
    for (yy, name, col, fc, stream) in ((5.3, r'$\mathrm{LM}_x$', BLUE, '#dbe6f7', 'melody'),
                                         (4.1, r'$\mathrm{LM}_y$', RED, '#f3dade', 'chord')):
        box(ax, 0.05, yy, 1.0, 0.85, fc=fc, ec=col, lw=0.9, z=2)
        ax.text(0.55, yy + 0.7, name, fontsize=FS, color=col, ha='center', va='center', weight='bold')
        box(ax, 0.12, yy + 0.1, 0.4, 0.42, 'attn', fc=GREY_FILL, ec=GREY, fs=FS - 1.6)
        box(ax, 0.58, yy + 0.1, 0.4, 0.42, 'FFN', fc=GREY_FILL, ec=GREY, fs=FS - 1.6)
        ax.text(1.35, yy + 0.9, f'{stream} stream, the input of ' + name, fontsize=FS - 1.2,
                color=col, va='center')
    ax.text(1.35, 6.55, 'content tokens', fontsize=FS, color=INK, weight='bold', va='center')
    n = 5
    x0, dx = 1.35, 0.5
    for i in range(n):
        box(ax, x0 + i * dx, 5.45, 0.4, 0.55, rf'$x_{i + 1}$', fc='#dbe6f7', ec=BLUE, fs=FS)
        box(ax, x0 + i * dx, 4.25, 0.4, 0.55, rf'$y_{i + 1}$', fc='#f3dade', ec=RED, fs=FS)
    ax.text(x0 + n * dx + 0.05, 5.72, r'$\cdots$', fontsize=FS + 1, va='center', color=INK)
    ax.text(x0 + n * dx + 0.05, 4.52, r'$\cdots$', fontsize=FS + 1, va='center', color=INK)
    # local encoder bracket
    ax.annotate('', xy=(x0 - 0.05, 4.05), xytext=(x0 + n * dx + 0.4, 4.05),
                arrowprops=dict(arrowstyle='-', color=GREY, lw=0.7))
    ax.text(x0 + (n * dx + 0.35) / 2, 3.85, 'local encoder (pretrained), one vector per frame',
            fontsize=FS - 0.6, color=INK_2, ha='center', va='center')

    # ---- harmonizers --------------------------------------------
    hx = x0 + n * dx + 0.6
    box(ax, hx, 5.45, 0.62, 0.55, r'$q_x$', fc='#fdeec7', ec=GOLD, fs=FS, lw=0.9)
    box(ax, hx, 4.25, 0.62, 0.55, r'$q_y$', fc='#fdeec7', ec=GOLD, fs=FS, lw=0.9)
    ax.text(hx + 0.31, 6.55, 'harmonizers', fontsize=FS, color=INK, weight='bold', ha='center', va='center')
    ax.text(hx + 0.68, 5.9, r'$k$', fontsize=FS - 1, color=INK_2, va='center')
    ax.text(hx + 0.68, 4.7, r'$k$', fontsize=FS - 1, color=INK_2, va='center')
    arrow(ax, (hx + 0.31, 5.43), (hx + 0.31, 4.82), color=GOLD, lw=1.2, style='<|-|>', ms=6)
    ax.text(hx + 0.42, 5.12, 'same-frame\npathway', fontsize=FS - 1.2, color=INK, va='center')
    ax.text(hx + 0.75, 4.05, 'hold frame $t$:\na draft, or the mask', fontsize=FS - 1,
            color=INK_2, ha='left', va='center')

    # ---- global stack block ------------------------------------------
    gx, gy, gw, gh = 0.15, 0.95, 6.9, 2.55
    box(ax, gx, gy, gw, gh, fc='none', ec=INK, lw=0.8, style='round,pad=0.02,rounding_size=0.1', z=2)
    ax.text(gx + 0.12, gy + gh - 0.18, r'global stack, $\times L$ layers', fontsize=FS,
            weight='bold', color=INK, va='center')
    # attention row
    ay = gy + 1.55
    box(ax, gx + 0.2, ay, 1.58, 0.7, '', fc='#dbe6f7', ec=BLUE)
    box(ax, gx + 1.82, ay, 1.58, 0.7, '', fc='#f3dade', ec=RED)
    ax.text(gx + 0.99, ay + 0.47, r'attention of $\mathrm{LM}_x$', fontsize=FS, ha='center',
            va='center', color=INK)
    ax.text(gx + 0.99, ay + 0.2, 'weights inherited', fontsize=FS - 1.2, ha='center',
            va='center', color=INK_2)
    ax.text(gx + 2.61, ay + 0.47, r'attention of $\mathrm{LM}_y$', fontsize=FS, ha='center',
            va='center', color=INK)
    ax.text(gx + 2.61, ay + 0.2, 'weights inherited', fontsize=FS - 1.2, ha='center',
            va='center', color=INK_2)
    # inheritance arrows from the model icons (dashed, grey)
    arrow(ax, (0.05, 5.5), (gx + 0.5, ay + 0.72), color=GREY, lw=0.7, ms=5, ls=(0, (2, 1.5)), z=1, rad=0.55)
    arrow(ax, (0.32, 4.1), (gx + 2.2, ay + 0.72), color=GREY, lw=0.7, ms=5, ls=(0, (2, 1.5)), z=1)
    # gold tabs
    tab1 = 'cross-stream pathway' + ('\n(low-rank update)' if lora else '')
    box(ax, gx + 3.55, ay + 0.37, 1.55, 0.36, tab1, fc='#fdeec7', ec=GOLD, fs=FS - 1.2, lw=0.9)
    box(ax, gx + 3.55, ay - 0.03, 1.55, 0.36, 'same-frame pathway', fc='#fdeec7', ec=GOLD,
        fs=FS - 1.2, lw=0.9)
    # gates
    box(ax, gx + 5.3, ay + 0.17, 0.5, 0.36, r'$\sigma$ gates', fc='#fdeec7', ec=GOLD, fs=FS - 1.4, lw=0.9)
    arrow(ax, (gx + 5.1, ay + 0.55), (gx + 5.3, ay + 0.4), color=GOLD, lw=0.7, ms=5)
    arrow(ax, (gx + 5.1, ay + 0.15), (gx + 5.3, ay + 0.3), color=GOLD, lw=0.7, ms=5)
    arrow(ax, (gx + 3.4, ay + 0.35), (gx + 5.3, ay + 0.35), color=GREY, lw=0.7, ms=5)
    # feed-forward row
    fy = gy + 0.25
    box(ax, gx + 0.2, fy + 0.5, 1.1, 0.34, 'router, melody', fc='#fdeec7', ec=GOLD, fs=FS - 1.4, lw=0.9)
    box(ax, gx + 0.2, fy + 0.05, 1.1, 0.34, 'router, chord', fc='#fdeec7', ec=GOLD, fs=FS - 1.4, lw=0.9)
    for i in range(4):
        box(ax, gx + 1.75 + i * 0.75, fy + 0.2, 0.62, 0.5, rf'$\mathrm{{FFN}}_{i + 1}$',
            fc=GREY_FILL, ec=GREY, fs=FS)
    ax.text(gx + 3.25, fy - 0.05, 'shared expert pool: four copies of the pretrained feed-forward network',
            fontsize=FS - 1.2, color=INK_2, ha='center', va='center')

    for yy in (fy + 0.67, fy + 0.22):
        arrow(ax, (gx + 1.3, yy), (gx + 1.75, fy + 0.45), color=GOLD, lw=0.7, ms=5)
    ax.text(gx + 5.0, fy + 0.5, 'top-2 per token', fontsize=FS - 1.2, color=INK_2, va='center')
    # flow arrows: tokens -> stack -> decoder
    arrow(ax, (x0 + 1.0, 4.2), (x0 + 1.0, gy + gh + 0.02), color=INK, lw=0.8, ms=6)
    arrow(ax, (hx + 0.31, 4.2), (hx + 0.31, gy + gh + 0.02), color=GOLD, lw=0.9, ms=6)
    # (the 'content tokens never read harmonizers' note lives in panel b)

    # ---- local decoder / output -------------------------------------
    box(ax, 7.45, 2.05, 1.55, 0.55, 'local decoder\n(pretrained)', fc=GREY_FILL, ec=GREY, fs=FS - 0.6)
    arrow(ax, (gx + gw, 2.32), (7.45, 2.32), color=INK, lw=0.8, ms=6)
    ax.text(9.5, 3.15, 'next frame', fontsize=FS, color=INK, ha='center', va='center', weight='bold')
    box(ax, 9.15, 2.55, 0.7, 0.34, r'$x_{t}$', fc='#dbe6f7', ec=BLUE, fs=FS)
    box(ax, 9.15, 1.85, 0.7, 0.34, r'$y_{t}$', fc='#f3dade', ec=RED, fs=FS)
    arrow(ax, (9.0, 2.45), (9.15, 2.7), color=BLUE, lw=0.7, ms=5)
    arrow(ax, (9.0, 2.2), (9.15, 2.02), color=RED, lw=0.7, ms=5)

    # ---- decoding strip ---------------------------------------------
    sy = 0.08
    box(ax, 0.15, sy, 9.7, 0.72, fc='none', ec=GRID, lw=0.7, style='round,pad=0.02,rounding_size=0.08', z=2)
    ax.text(0.3, sy + 0.5, 'decoding frame $t$', fontsize=FS, weight='bold', color=INK, va='center')
    ax.text(2.35, sy + 0.5, '1  draft: both streams draft $x_t$ and $y_t$ from the content positions.   '
            '2  commit and condition: the leader commits its draft into its harmonizer ($k=0$);',
            fontsize=FS - 0.9, color=INK, va='center')
    ax.text(2.35, sy + 0.22, "the follower's harmonizer, masked, reads it through the same-frame pathway and "
            "predicts its frame.   The leader alternates: melody at odd $t$, chord at even $t$.",
            fontsize=FS - 0.9, color=INK, va='center')

    # ---- legend --------------------------------------------------------
    handles = [Rectangle((0, 0), 1, 1, fc=GREY_FILL, ec=GREY, lw=0.7),
               Rectangle((0, 0), 1, 1, fc='#fdeec7', ec=GOLD, lw=0.9),
               Line2D([0], [0], color=GREY, lw=0.7, ls=(0, (2, 1.5)))]
    ax.legend(handles, ['pretrained, inherited', 'added: binds the streams',
                        'weights copied from the pretrained model'],
              loc='upper right', bbox_to_anchor=(0.96, 0.93), frameon=False,
              fontsize=FS - 0.8, handlelength=1.4, labelcolor=INK)


def real_masks(T=6):
    """The layer's own masks for T frames + one harmonizer pair for frame T."""
    from cp_transformer_m2c_duet_block import M2CDuetBlockLayer
    layer = M2CDuetBlockLayer(hidden_size=16, num_heads=2, intermediate_size=32,
                              moe_num_experts=1, moe_topk=1, moe_intermediate_size=32)
    clean_len = 2 * T
    intra, cross, frame = layer._build_masks(clean_len, T, torch.device('cpu'))
    return intra.numpy(), cross.numpy(), frame.numpy(), clean_len


def draw_masks(ax, T=6):
    intra, cross, frame, clean_len = real_masks(T)
    L = clean_len + 2
    # cross carries a diagonal guard (self) for empty rows; that is
    # numerics, not a pathway: treat the diagonal as same-stream
    diag = np.eye(L, dtype=bool)
    same = intra | (cross & diag)
    crossp = cross & ~diag & ~frame
    img = np.zeros((L, L, 3))
    img[:] = 1.0
    cols = {'same': GREY, 'cross': BLUE, 'frame': GOLD}
    for name, m in (('same', same), ('cross', crossp), ('frame', frame)):
        rgb = matplotlib.colors.to_rgb(cols[name])
        img[m] = rgb
    # gap between content and harmonizer blocks: draw as a separate image via extent
    ax.imshow(img, interpolation='nearest', origin='upper', extent=(-0.5, L - 0.5, L - 0.5, -0.5))
    # white grid lines between cells
    for k in range(L + 1):
        ax.axhline(k - 0.5, color=SURFACE, lw=0.6); ax.axvline(k - 0.5, color=SURFACE, lw=0.6)
    # separator between content and harmonizers
    ax.axhline(clean_len - 0.5, color=INK, lw=0.9); ax.axvline(clean_len - 0.5, color=INK, lw=0.9)
    labels = [rf'$x_{i // 2 + 1}$' if i % 2 == 0 else rf'$y_{i // 2 + 1}$' for i in range(clean_len)]
    labels += [r'$q_x$', r'$q_y$']
    ax.set_xticks(range(L)); ax.set_yticks(range(L))
    ax.set_xticklabels(labels, fontsize=FS - 0.6); ax.set_yticklabels(labels, fontsize=FS - 0.6)
    for lab, i in zip(ax.get_xticklabels() + ax.get_yticklabels(), list(range(L)) * 2):
        lab.set_color(GOLD if i >= clean_len else (BLUE if i % 2 == 0 else RED))
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel('token read (key)', fontsize=FS, color=INK)
    ax.set_ylabel('token reading (query)', fontsize=FS, color=INK)
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(0.7)
    ax.text(clean_len + 0.5, clean_len / 2 - 0.5, 'content tokens\nnever read\nharmonizers',
            fontsize=FS - 1.4, color=INK_2, ha='center', va='center', rotation=90)
    handles = [Rectangle((0, 0), 1, 1, fc=GREY), Rectangle((0, 0), 1, 1, fc=BLUE),
               Rectangle((0, 0), 1, 1, fc=GOLD)]
    ax.legend(handles, [r'same stream, frames $\leq t$', r'cross stream, frames $< t$',
                        'same frame'], title='Pathway', loc='lower left',
              bbox_to_anchor=(0.0, -0.36), frameon=False, fontsize=FS - 0.8,
              title_fontsize=FS - 0.6, handlelength=1.2, ncol=1, labelcolor=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='path WITHOUT extension')
    ap.add_argument('--no-lora', action='store_true',
                    help='omit the low-rank-update label on the cross pathway')
    args = ap.parse_args()
    fig = plt.figure(figsize=(7.0, 3.3))
    fig.patch.set_facecolor(SURFACE)
    axa = fig.add_axes([0.005, 0.02, 0.655, 0.96])
    axb = fig.add_axes([0.74, 0.22, 0.25, 0.66])
    draw_arch(axa, lora=not args.no_lora)
    draw_masks(axb)
    axa.text(-0.42, 7.0, 'a', fontsize=9, weight='bold', color=INK, va='top')
    axb.annotate('b', xy=(-0.30, 1.0), xycoords='axes fraction', xytext=(0, 22),
                 textcoords='offset points', ha='left', va='bottom',
                 fontsize=9, weight='bold', color=INK)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, facecolor=SURFACE)
        print(f'wrote {args.out}.{ext}')


if __name__ == '__main__':
    main()
