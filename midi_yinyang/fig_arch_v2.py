"""Model figure v2, built on the hand sketch of 2026-09-21: (a) the
architecture read bottom-up for one frame -- context, the two per-stream
key/value banks, the three gated readouts of x_t and y_t, W^O and
Add & LN, the per-stream routers with their pi bars, the expert pool,
Add & LN, and the predicted tokens, repeated x L -- and (b) the
alternating-commit decoding of one frame in three stacked steps.

Notation follows paper/method.tex: streams x, y; tokens x_t, y_t;
context x_<t, y_<t; key sets K_intra, K_cross, K_frame; scalar gates
g^c_s, g^f_s; projections W^Q_s, W^K_s, W^V_s, W^O_s; routers G^(x),
G^(y) with softmax pi over E experts, top-k; FFN_i; post-LN
Add & LN; harmonizer tokens q_x, q_y.

Usage:
  python fig_arch_v2.py --out results/fig_model_v2     # .pptx + .png preview
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_arch_pptx import (Spec, write_pptx, write_preview, m, plain,  # noqa: E402
                           BLUE, BLUE_L, RED, RED_L, GOLD, GOLD_L, INK, INK_2, PANEL)
import fig_arch_pptx

fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 14.8, 9.0
GREY = '999999'


def sub(base, s):
    return m(base, (s, 'sub'))


def build():
    S = Spec()
    # =============================================================== (a)
    S.text(0.3, 0.12, 5.0, 0.32, [('a', 'b'), ('  Model: one block, predicting frame ', ''), ('t', 'i')],
           size=13, align='l')
    # geometry
    tk = 0.38                                      # token edge
    y_in, y_out = 7.62, 0.75
    xq, yq = 6.5, 7.6                              # columns of x_t and y_t
    hx, hy = 8.4, 9.0                              # harmonizer columns
    ctx = [0.7, 1.15, None, 1.95, 2.4]             # context columns (None = dots)
    ctx_lab = [sub('x', '1'), sub('y', '1'), None, sub('x', 't−2'), sub('y', 't−2')]

    # block frame and the x L loop
    bx0, bx1, by0, by1 = 0.45, 9.85, 1.45, 7.42
    S.rect(bx0, by0, bx1 - bx0, by1 - by0, fill='FFFFFF', line=INK, lw=1.0, z=0)
    S.text(bx0 + 0.12, by0 + 0.05, 3.0, 0.3, [('Transformer block, ×', ''), ('L', 'i')], size=11,
           align='l')
    S.line(bx1 - 0.02, by0 + 0.6, bx1 + 0.22, by0 + 0.6, lw=0.75)
    S.line(bx1 + 0.22, by0 + 0.6, bx1 + 0.22, by1 - 0.4, lw=0.75)
    S.line(bx1 + 0.22, by1 - 0.4, bx1 + 0.02, by1 - 0.4, lw=0.75, arrow=True)
    S.text(bx1 + 0.05, (by0 + by1) / 2 - 0.15, 0.4, 0.3, [('×', ''), ('L', 'i')], size=10, align='l')

    def token(cx, y, kind, label=None, inner=None, dash=False, size=10):
        fill, line = {'x': (BLUE_L, BLUE), 'y': (RED_L, RED), 'q': (GOLD_L, GOLD),
                      'qm': ('FFFFFF', GOLD), 'qx': (BLUE_L, BLUE), 'qy': (RED_L, RED),
                      'ox': ('FFFFFF', BLUE), 'oy': ('FFFFFF', RED)}[kind]
        S.rect(cx - tk / 2, y, tk, tk, fill=fill, line=line, lw=1.1, dash=dash or kind == 'qm',
               runs=inner, size=8.5)
        if label:
            S.text(cx - 0.4, y + tk + 0.02, 0.8, 0.24, label, size=size)

    # ---- input row
    for cx, lab in zip(ctx, ctx_lab):
        if cx is None:
            S.text(1.35, y_in + 0.05, 0.4, 0.3, plain('…'), size=12); continue
        token(cx, y_in, 'x' if lab[0][0] == 'x' else 'y', lab)
    S.line(0.5, y_in + tk + 0.3, 2.6, y_in + tk + 0.3, color=INK_2, lw=0.6)
    S.text(0.5, y_in + tk + 0.32, 2.1, 0.22, plain('earlier context'), size=9, color=INK_2)
    token(xq, y_in, 'x', sub('x', 't−1'))
    token(yq, y_in, 'y', sub('y', 't−1'))
    token(hx, y_in, 'q', sub('q', 'x'))
    token(hy, y_in, 'q', sub('q', 'y'))
    S.text(xq - 0.5, y_in + tk + 0.32, 2.0, 0.22, [('last committed frame; each', '')], size=8.5, color=INK_2, align='l')
    S.text(xq - 0.5, y_in + tk + 0.5, 2.0, 0.22, [('position predicts frame ', ''), ('t', 'i')], size=8.5, color=INK_2, align='l')
    S.text(hx - 0.55, y_in + tk + 0.32, 1.9, 0.22, [('harmonizers, predict frame ', ''), ('t', 'i')], size=8.5, color=INK_2, align='l')
    S.text(hx - 0.55, y_in + tk + 0.5, 1.9, 0.22, [('input: mask, or enc(', ''), ('x', 'i'), ('t', 'sub i'), (')+', ''), ('q', 'i'), ('x', 'sub i'), (' (b2)', '')], size=8.5, color=INK_2, align='l')

    # ---- key/value banks per stream (context split by stream)
    yb = 6.45
    xb, ybk = 3.65, 5.25                           # bank centres
    S.rect(xb - 0.7, yb, 1.4, 0.48, fill=BLUE_L, line=INK, lw=0.75, dash=True,
           runs=[('stream ', ''), ('x', 'i'), (' context', ''), ('\n', ''), ('W', 'i'), ('K', 'sup i'), ('x', 'sub i'),
                 (', ', ''), ('W', 'i'), ('V', 'sup i'), ('x', 'sub i')], size=8.5)
    S.rect(ybk - 0.7, yb, 1.4, 0.48, fill=RED_L, line=INK, lw=0.75, dash=True,
           runs=[('stream ', ''), ('y', 'i'), (' context', ''), ('\n', ''), ('W', 'i'), ('K', 'sup i'), ('y', 'sub i'),
                 (', ', ''), ('W', 'i'), ('V', 'sup i'), ('y', 'sub i')], size=8.5)
    S.line(1.55, y_in - 0.02, xb - 0.2, yb + 0.5, lw=0.75, arrow=True)
    S.line(2.2, y_in - 0.02, ybk - 0.2, yb + 0.5, lw=0.75, arrow=True)
    # ---- gated readouts
    ys = 4.85                                      # sum row
    r = 0.15

    def gate(cx, cy, s, kind):
        S.rect(cx - r, cy - r, 2 * r, 2 * r, fill=GOLD_L, line=GOLD, lw=0.9, shape='ellipse',
               runs=[('g', 'i'), (kind, 'sup i'), (s, 'sub i')], size=8, z=5)

    gcx, gfx = (xq - 0.65, 5.5), (xq + 0.55, 5.5)
    gfy, gcy = (yq - 0.45, 5.95), (yq + 0.55, 5.95)
    # x_t: intra from x bank; cross from y bank via g^c_x; frame from y_t via g^f_x; own query
    S.line(xb + 0.3, yb - 0.02, xq - 0.1, ys + 0.1, color=BLUE, lw=0.9, arrow=True)
    S.line(ybk + 0.35, yb - 0.02, gcx[0], gcx[1] + r, color=RED, lw=0.9)
    gate(*gcx, 'x', 'c')
    S.line(gcx[0] + 0.1, gcx[1] - r, xq - 0.08, ys + 0.1, color=RED, lw=0.9, arrow=True)
    S.line(yq, 6.25, gfx[0], gfx[1] + r, color=RED, lw=0.9)
    gate(*gfx, 'x', 'f')
    S.line(gfx[0] - 0.1, gfx[1] - r, xq + 0.08, ys + 0.1, color=RED, lw=0.9, arrow=True)
    S.line(xq, y_in - 0.02, xq, ys + 0.12, color=BLUE, lw=0.9, arrow=True)
    # y_t: intra from y bank; cross from x bank via g^c_y (routed along y = 6.75); frame from x_t
    S.line(ybk - 0.2, yb - 0.02, yq - 0.1, ys + 0.1, color=RED, lw=0.9, arrow=True)
    S.line(xb + 0.45, yb - 0.02, xb + 0.45, 6.28, color=BLUE, lw=0.9)
    S.line(xb + 0.45, 6.28, gcy[0], 6.28, color=BLUE, lw=0.9)
    S.line(gcy[0], 6.28, gcy[0], gcy[1] + r, color=BLUE, lw=0.9)
    gate(*gcy, 'y', 'c')
    S.line(gcy[0] - 0.1, gcy[1] - r, yq + 0.08, ys + 0.1, color=BLUE, lw=0.9, arrow=True)
    S.line(xq, 6.55, gfy[0], gfy[1] + r, color=BLUE, lw=0.9)
    gate(*gfy, 'y', 'f')
    S.line(gfy[0] + 0.1, gfy[1] - r, yq - 0.08, ys + 0.1, color=BLUE, lw=0.9, arrow=True)
    S.line(yq, y_in - 0.02, yq, ys + 0.12, color=RED, lw=0.9, arrow=True)
    S.node(xq, ys, r=0.12); S.node(yq, ys, r=0.12)
    # key-set labels, left of the readouts
    S.text(0.6, 5.0, 2.4, 0.22, [('K', 'i'), ('intra', 'sub'), (': own stream, causal', '')], size=8.5, align='l')
    S.text(0.6, 5.24, 2.4, 0.22, [('K', 'i'), ('cross', 'sub'), (': other stream, earlier frames', '')], size=8.5, align='l')
    S.text(0.6, 5.48, 2.4, 0.22, [('K', 'i'), ('frame', 'sub'), (': partner, same frame', '')], size=8.5, align='l')
    S.text(0.6, 5.74, 2.6, 0.22, [('gates ', ''), ('g', 'i'), ('c', 'sup i'), ('s', 'sub i'), (', ', ''),
                                  ('g', 'i'), ('f', 'sup i'), ('s', 'sub i'), (' ∈ (0,1): learned scalars', '')],
           size=8.5, align='l')
    S.text(0.6, 4.55, 2.6, 0.22, [('o', 'i'), (' = ', ''), ('W', 'i'), ('O', 'sup i'), ('s', 'sub i'),
                                  ('(', ''), ('u', 'i'), ('intra', 'sub'), (' + ', ''), ('g', 'i'), ('c', 'sup i'), ('s', 'sub i'),
                                  ('u', 'i'), ('cross', 'sub'), (' + ', ''), ('g', 'i'), ('f', 'sup i'), ('s', 'sub i'),
                                  ('u', 'i'), ('frame', 'sub'), (')', '')], size=8.5, align='l')
    # harmonizers: own-stream context, other-stream context via g^c, partner via g^f
    for cx in (hx, hy):
        S.line(cx, y_in - 0.02, cx, ys + 0.12, color=GOLD, lw=0.9, arrow=True)
        S.node(cx, ys, r=0.12)
    gm = ((hx + hy) / 2, 5.5)
    S.line(hx, 6.3, gm[0] - 0.05, gm[1] + r, color=GOLD, lw=0.9)
    S.line(hy, 6.3, gm[0] + 0.05, gm[1] + r, color=GOLD, lw=0.9)
    gate(*gm, 's', 'f')
    S.line(gm[0] - 0.08, gm[1] - r, hx + 0.08, ys + 0.1, color=GOLD, lw=0.9, arrow=True)
    S.line(gm[0] + 0.08, gm[1] - r, hy - 0.08, ys + 0.1, color=GOLD, lw=0.9, arrow=True)
    # context stubs: q_x reads x_<t (own stream) and y_<t (other stream, gated); q_y mirrors
    S.text(hx - 0.62, 4.8, 0.5, 0.18, [('x', 'i'), ('<t', 'sub i')], size=7.5, color=BLUE, align='l')
    S.line(hx - 0.58, 5.06, hx - 0.1, ys + 0.1, color=BLUE, lw=0.9, arrow=True)
    S.text(hx - 0.62, 5.68, 0.5, 0.18, [('y', 'i'), ('<t', 'sub i')], size=7.5, color=RED, align='l')
    S.line(hx - 0.58, 5.55, hx - 0.5, 5.55, color=RED, lw=0.9)
    gate(hx - 0.35, 5.55, 'x', 'c')
    S.line(hx - 0.25, 5.55 - r + 0.02, hx - 0.05, ys + 0.12, color=RED, lw=0.9, arrow=True)
    S.text(hy + 0.15, 4.8, 0.5, 0.18, [('y', 'i'), ('<t', 'sub i')], size=7.5, color=RED, align='r')
    S.line(hy + 0.58, 5.06, hy + 0.1, ys + 0.1, color=RED, lw=0.9, arrow=True)
    S.text(hy + 0.15, 5.68, 0.5, 0.18, [('x', 'i'), ('<t', 'sub i')], size=7.5, color=BLUE, align='r')
    S.line(hy + 0.58, 5.55, hy + 0.5, 5.55, color=BLUE, lw=0.9)
    gate(hy + 0.35, 5.55, 'y', 'c')
    S.line(hy + 0.25, 5.55 - r + 0.02, hy + 0.05, ys + 0.12, color=BLUE, lw=0.9, arrow=True)
    S.text(gm[0] - 0.4, gm[1] + 0.17, 0.8, 0.18, plain('partner'), size=7, color=INK_2)
    # ---- W^O and Add & LN, per stream
    yo = 4.25
    for cx, s, fill in ((xq, 'x', BLUE_L), (yq, 'y', RED_L), (hx, 'x', GOLD_L), (hy, 'y', GOLD_L)):
        S.line(cx, ys - 0.12, cx, yo + 0.3, lw=0.75, arrow=True)
        S.rect(cx - 0.28, yo, 0.56, 0.28, fill=fill, line=INK, lw=0.75, dash=True,
               runs=[('W', 'i'), ('O', 'sup i'), (s, 'sub i')], size=8.5)
        S.line(cx, yo - 0.02, cx, yo - 0.28, lw=0.75, arrow=True)
        S.rect(cx - 0.3, yo - 0.56, 0.6, 0.28, fill='FFFFFF', line=INK, lw=0.75,
               runs=plain('Add & LN'), size=7.5)
    # ---- routers with pi bars, expert pool
    yr = 3.1
    pis = {'x': (0.61, 0.08, 0.27, 0.04), 'y': (0.04, 0.27, 0.61, 0.08)}
    for cx, s, fill, col in ((xq, 'x', BLUE_L, BLUE), (yq, 'y', RED_L, RED)):
        S.line(cx, yo - 0.58, cx, yr + 0.3, lw=0.75, arrow=True)
        S.rect(cx - 0.42, yr, 0.84, 0.28, fill=fill, line=col, lw=0.75,
               runs=[('router ', ''), ('G', 'i'), ('(' + s + ')', 'sup i')], size=8.5)
        bx = cx - 0.95 if s == 'x' else cx + 0.5
        for i, p in enumerate(pis[s]):
            S.rect(bx + i * 0.11, yr + 0.26 - 0.26 * p, 0.08, 0.26 * p, fill=col, line=None)
        S.text(bx - 0.2 if s == 'x' else bx + 0.46, yr + 0.02, 0.2, 0.24, [('π', 'i')], size=9, align='l')
    for cx in (hx, hy):                            # harmonizers go through the same layer
        S.line(cx, yo - 0.58, cx, y_out + tk + 0.02, color=GOLD, lw=0.9, arrow=True)
    yp = 2.15
    S.rect(2.9, yp - 0.12, 5.6, 0.72, fill=PANEL, line=None, z=0)
    S.text(2.95, yp - 0.1, 3.0, 0.2, [('expert pool: each position uses its top-', ''), ('k', 'i'), (' of ', ''), ('E', 'i')],
           size=8, color=INK_2, align='l')
    ffn_x = [3.1, 4.45, 5.8, 7.15]
    for i, fx in enumerate(ffn_x):
        S.rect(fx, yp + 0.12, 1.15, 0.4, fill='FFFFFF', line=INK, lw=0.75, dash=True,
               runs=[('FFN', ''), (str(i + 1), 'sub')], size=9.5)
    for cx, s, col in ((xq, 'x', BLUE), (yq, 'y', RED)):
        for fx, p in zip(ffn_x, pis[s]):
            on = p >= 0.27
            S.line(cx, yr - 0.02, fx + 0.57 + (0.1 if s == 'y' else -0.1), yp + 0.54,
                   color=col if on else GREY, lw=1.0 if on else 0.5, dash=not on, arrow=True)
    # ---- Add & LN after the experts, then the predicted tokens
    ya = 1.55
    for cx, s, col in ((xq, 'x', BLUE), (yq, 'y', RED)):
        for fx, p in zip(ffn_x, pis[s]):
            if p >= 0.27:
                S.line(fx + 0.57 + (0.1 if s == 'y' else -0.1), yp + 0.1, cx, ya + 0.3,
                       color=col, lw=1.0, arrow=True)
        S.rect(cx - 0.3, ya, 0.6, 0.28, fill='FFFFFF', line=INK, lw=0.75, runs=plain('Add & LN'),
               size=7.5)
        S.line(cx, ya - 0.02, cx, y_out + tk + 0.02, lw=0.75, arrow=True)
    token(xq, y_out, 'x'); S.text(xq - 0.5, y_out - 0.27, 1.0, 0.24, sub('x', 't'), size=10)
    token(yq, y_out, 'y'); S.text(yq - 0.5, y_out - 0.27, 1.0, 0.24, sub('y', 't'), size=10)
    token(hx, y_out, 'ox'); S.text(hx - 0.5, y_out - 0.27, 1.0, 0.24, sub('x', 't'), size=10)
    token(hy, y_out, 'oy'); S.text(hy - 0.5, y_out - 0.27, 1.0, 0.24, sub('y', 't'), size=10)
    S.text(xq - 0.5, 0.22, 1.9, 0.2, plain('next-frame heads: leader (b1)'), size=7.5,
           color=INK_2, align='l')
    S.text(hx - 0.4, 0.22, 1.7, 0.2, plain('harmonizers: follower (b2)'), size=7.5, color=INK_2,
           align='l')
    # ---- legend
    lx, ly = 0.55, 8.68
    for k, (fill, line, dash, lab) in enumerate((
            (BLUE_L, BLUE, False, [('stream ', ''), ('x', 'i'), (' (melody)', '')]),
            (RED_L, RED, False, [('stream ', ''), ('y', 'i'), (' (chord)', '')]),
            (GOLD_L, GOLD, False, plain('harmonizer')),
            ('FFFFFF', GOLD, True, plain('masked harmonizer')),
            ('FFFFFF', INK, True, [('dotted box: initialised from LM', ''), ('s', 'sub i')]))):
        xx = lx + k * 1.75
        S.rect(xx, ly, 0.26, 0.2, fill=fill, line=line, lw=0.8, dash=dash)
        S.text(xx + 0.32, ly - 0.03, 1.7, 0.26, lab, size=8.5, align='l')

    # =============================================================== (b)
    x0 = 10.4
    S.text(x0 - 0.1, 0.12, 4.8, 0.32, [('b', 'b'), ('  Alternating-commit decoding of frame ', ''), ('t', 'i')],
           size=13, align='l')
    ts, tg = 0.3, 0.06

    def row(x, y, kinds, labels, inner=None):
        xs = []
        for kind, lab in zip(kinds, labels):
            if kind is None:
                S.text(x, y + 0.02, ts, ts, plain('…'), size=12); x += ts + tg; continue
            fill, line = {'x': (BLUE_L, BLUE), 'y': (RED_L, RED), 'qm': ('FFFFFF', GOLD),
                          'qx': (BLUE_L, BLUE)}[kind]
            S.rect(x, y, ts, ts, fill=fill, line=line, lw=1.1, dash=(kind == 'qm'),
                   runs=sub('x', 't') if kind == 'qx' else None, size=8)
            S.text(x - 0.2, y + ts + 0.02, ts + 0.4, 0.22, lab, size=9)
            xs.append(x + ts / 2); x += ts + tg
        return xs

    pre_k = ['x', 'y', None, 'x', 'y']
    pre_l = [sub('x', '1'), sub('y', '1'), None, sub('x', 't−1'), sub('y', 't−1')]
    steps = [
        (1.0, '1  draft the leader', ['qm', 'qm'], [sub('q', 'x'), sub('q', 'y')], 'x',
         m('x', ('t', 'sub'), (' ∼ ', 'r'), 'p', ('(', 'r'), 'x', ('t', 'sub'), (' | ', 'r'), 'x', ('<t', 'sub'),
           (', ', 'r'), 'y', ('<t', 'sub'), (')', 'r')),
         [('both harmonizers masked; ', ''), ('x', 'i'), ('t', 'sub i'),
          (' is sampled from the stream-', ''), ('x', 'i'), (' next-frame head', '')]),
        (3.55, '2  condition the follower', ['qx', 'qm'], [sub('q', 'x'), sub('q', 'y')], 'y',
         m('y', ('t', 'sub'), (' ∼ ', 'r'), 'p', ('(', 'r'), 'y', ('t', 'sub'), (' | ', 'r'), 'x', ('t', 'sub'),
           (', ', 'r'), 'x', ('<t', 'sub'), (', ', 'r'), 'y', ('<t', 'sub'), (')', 'r')),
         [('x', 'i'), ('t', 'sub i'), (' is written into ', ''), ('q', 'i'), ('x', 'sub i'), ('; ', ''),
          ('q', 'i'), ('y', 'sub i'), (' reads it through ', ''), ('K', 'i'), ('frame', 'sub'), (' and emits ', ''),
          ('y', 'i'), ('t', 'sub i')]),
        (6.1, '3  commit both, swap roles', ['x', 'y'], [sub('x', 't'), sub('y', 't')], None, None,
         [('x', 'i'), ('t', 'sub i'), (', ', ''), ('y', 'i'), ('t', 'sub i'), (' join the context; at ', ''),
          ('t', 'i'), ('+1 stream ', ''), ('y', 'i'), (' leads', '')]),
    ]
    for y0, title, qk, ql, out, formula, note in steps:
        S.text(x0, y0, 4.5, 0.26, [(title, 'b')], size=10.5, align='l')
        yr_ = y0 + 1.0
        xs = row(x0, yr_, pre_k + qk, pre_l + ql)
        S.line(x0, yr_ + ts + 0.28, xs[3] + ts / 2, yr_ + ts + 0.28, color=INK_2, lw=0.6)
        S.text(x0, yr_ + ts + 0.3, xs[3] + ts / 2 - x0, 0.2, plain('context'), size=8, color=INK_2)
        if formula:
            src = xs[3] if out == 'x' else xs[5]
            fill, line = (BLUE_L, BLUE) if out == 'x' else (RED_L, RED)
            S.line(src, yr_ - 0.03, src, yr_ - 0.3, arrow=True, lw=0.9)
            S.rect(src - ts / 2, yr_ - 0.64, ts, ts, fill=fill, line=line, lw=1.1)
            S.text(src + ts / 2 + 0.08, yr_ - 0.66, 2.5, 0.38, formula, size=9.5, align='l')
            if out == 'y':
                S.line(xs[4] + ts / 2 - 0.04, yr_ - 0.1, xs[5] - ts / 2 + 0.02, yr_ - 0.1, color=GOLD,
                       lw=1.4, arrow=True)
        S.text(x0, y0 + 1.95, 4.3, 0.24, note, size=8, color=INK_2, align='l')
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    S = build()
    write_pptx(S, args.out + '.pptx'); print('wrote', args.out + '.pptx')
    write_preview(S, args.out + '.png'); print('wrote', args.out + '.png')


if __name__ == '__main__':
    main()
