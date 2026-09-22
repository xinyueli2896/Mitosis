"""Model figure v3, on the design of 2026-09-22 (context banks feeding a
block read bottom-up; pretrained model and legend on the left; decoding
on the right), laid out wide.

Concrete frame: three frames committed, frame 4 being predicted.
  context      x_1 y_1 x_2 y_2 (banks) and the last committed x_3, y_3
  harmonizers  q_x, q_y, predicting frame 4
  outputs      x_4, y_4 from the next-frame heads; x_4, y_4 estimates
               from the harmonizers

Notation (paper/method.tex): streams x, y; W^Q_s, W^K_s, W^V_s, W^O_s;
key sets K_intra, K_cross, K_frame; gates g^c_s, g^f_s; routers G^(x),
G^(y), softmax pi, top-k of E experts FFN_i; post-LN Add & LN; the
decode conditionals p(x_t | x_<t, y_<t), p(y_t | x_t, x_<t, y_<t).

Usage:
  python fig_arch_v3.py --out results/fig_model_v3     # .pptx + .png preview
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_arch_pptx import (Spec, write_pptx, write_preview, m, plain,  # noqa: E402
                           BLUE, BLUE_L, RED, RED_L, GOLD, GOLD_L, INK, INK_2, PANEL)
import fig_arch_pptx

fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 15.0, 8.05
GREY = '9A9A9A'
LIGHT = 'F2F1EE'


def sub(base, s):
    return m(base, (s, 'sub'))


def W(role, s):
    return [('W', 'i'), (role, 'sup i'), (s, 'sub i')]


def build():
    S = Spec()
    tk = 0.36                                      # token edge

    def token(cx, y, kind, label=None, inner=None, size=9.5, above=False):
        fill, line = {'x': (BLUE_L, BLUE), 'y': (RED_L, RED), 'q': (GOLD_L, GOLD),
                      'qm': ('FFFFFF', GOLD), 'ox': ('FFFFFF', BLUE), 'oy': ('FFFFFF', RED)}[kind]
        S.rect(cx - tk / 2, y, tk, tk, fill=fill, line=line, lw=1.1, dash=(kind == 'qm'),
               runs=inner, size=8)
        if label:
            S.text(cx - 0.45, y - 0.26 if above else y + tk + 0.02, 0.9, 0.24, label, size=size)

    # ================================================================ block
    # columns: x_3, q_x (stream x) | y_3, q_y (stream y)
    cX, cQX, cY, cQY = 5.1, 6.5, 8.2, 9.6
    bx0, bx1 = 3.95, 10.4
    by0, by1 = 1.15, 5.97
    S.rect(bx0, by0, bx1 - bx0, by1 - by0, fill='FFFFFF', line=INK, lw=1.0, z=0)

    # ---- projections (bottom of the block)
    yW = 5.52
    S.rect(bx0 + 0.15, yW, 2.85, 0.3, fill=BLUE_L, line=INK, lw=0.75, dash=True,
           runs=W('Q', 'x') + [(', ', '')] + W('K', 'x') + [(', ', '')] + W('V', 'x'), size=9)
    S.rect(cY - 0.9, yW, 2.85, 0.3, fill=RED_L, line=INK, lw=0.75, dash=True,
           runs=W('Q', 'y') + [(', ', '')] + W('K', 'y') + [(', ', '')] + W('V', 'y'), size=9)
    # ---- readouts: three boxes per column, each labelled with its key set
    yR = 4.98
    bw, bg = 0.4, 0.05
    keysets = {cX: ([sub('x', '≤3')], [sub('y', '≤2')], [sub('y', '3')]),
               cQX: ([sub('x', '≤3')], [sub('y', '≤3')], [sub('q', 'y')]),
               cY: ([sub('y', '≤3')], [sub('x', '≤2')], [sub('x', '3')]),
               cQY: ([sub('y', '≤3')], [sub('x', '≤3')], [sub('q', 'x')])}
    own = {cX: 'x', cQX: 'x', cY: 'y', cQY: 'y'}
    r = 0.14
    for cx in (cX, cQX, cY, cQY):
        s = own[cx]; o = 'y' if s == 'x' else 'x'
        cols = [(BLUE_L, BLUE) if s == 'x' else (RED_L, RED),
                (RED_L, RED) if s == 'x' else (BLUE_L, BLUE),
                (GOLD_L, GOLD)]
        for k, (lab, (fill, line)) in enumerate(zip(keysets[cx], cols)):
            x = cx + (k - 1) * (bw + bg) - bw / 2
            S.rect(x, yR, bw, 0.3, fill=fill, line=line, lw=0.8, runs=lab[0], size=8)
            S.line(cx + (k - 1) * (bw + bg), yW - 0.02, cx + (k - 1) * (bw + bg), yR + 0.32,
                   lw=0.6, arrow=True, color=INK_2)
        # gates on cross and frame, sum, W^O
        ysum = 4.32
        S.node(cx, ysum, r=0.11)
        S.line(cx - (bw + bg), yR - 0.02, cx - 0.09, ysum + 0.09, lw=0.7, arrow=True)
        for k, kind in ((0, 'c'), (1, 'f')):
            gx = cx + k * (bw + bg)
            S.rect(gx - r, 4.55, 2 * r, 2 * r, fill=GOLD_L, line=GOLD, lw=0.9, shape='ellipse',
                   runs=[('g', 'i'), (kind, 'sup i'), (s, 'sub i')], size=7.5, z=5)
            S.line(gx, yR - 0.02, gx, 4.55 + 2 * r + 0.02, lw=0.7, arrow=True)
            S.line(gx - 0.04, 4.53, cx + 0.04, ysum + 0.1, lw=0.7, arrow=True)
        S.line(cx, ysum - 0.11, cx, 3.99 + 0.02, lw=0.7, arrow=True)
        S.rect(cx - 0.3, 3.72, 0.6, 0.27, fill=cols[0][0], line=INK, lw=0.75, dash=True,
               runs=W('O', s), size=8.5)
    S.text(bx0 + 0.1, yR + 0.02, 0.6, 0.26, plain('keys'), size=7.5, color=INK_2, align='l')
    S.text(bx0 + 0.1, 4.57, 0.6, 0.26, plain('gates'), size=7.5, color=INK_2, align='l')
    # attention Add & LN bar
    yA = 3.32
    S.rect(bx0 + 0.15, yA, bx1 - bx0 - 0.3, 0.26, fill=LIGHT, line=INK, lw=0.6,
           runs=[('Add & LN', ''), ('   ', ''), ('h', 'i'), (' ← LN(', ''), ('h', 'i'), (' + ', ''), ('o', 'i'), (')', '')], size=8.5)
    for cx in (cX, cQX, cY, cQY):
        S.line(cx, 3.72 - 0.02, cx, yA + 0.28, lw=0.7, arrow=True)
    # ---- MoE FFN
    yRt = 2.82
    S.rect(bx0 + 0.15, yRt, 2.85, 0.28, fill=BLUE_L, line=BLUE, lw=0.75,
           runs=[('router ', ''), ('G', 'i'), ('(x)', 'sup i'), ('   π = softmax(', ''), ('G', 'i'), ('(x)', 'sup i'), ('h', 'i'), (')', '')], size=8.5)
    S.rect(cY - 0.9, yRt, 2.85, 0.28, fill=RED_L, line=RED, lw=0.75,
           runs=[('router ', ''), ('G', 'i'), ('(y)', 'sup i'), ('   π = softmax(', ''), ('G', 'i'), ('(y)', 'sup i'), ('h', 'i'), (')', '')], size=8.5)
    for cx in (cX, cQX, cY, cQY):
        S.line(cx, yA - 0.02, cx, yRt + 0.3, lw=0.7, arrow=True)
    pis = {'x': (0.61, 0.08, 0.27, 0.04), 'y': (0.04, 0.27, 0.61, 0.08)}
    # pi bars above the outer ends of the routers
    for s, x0_, col in (('x', bx0 + 0.2, BLUE), ('y', bx1 - 0.75, RED)):
        for i, p in enumerate(pis[s]):
            S.rect(x0_ + i * 0.13, yRt - 0.08 - 0.28 * p, 0.1, 0.28 * p, fill=col, line=None)
    S.text(bx0 + 0.75, yRt - 0.3, 0.9, 0.2, [('π', 'i'), (' (', ''), ('E', 'i'), (' = 4)', '')], size=7,
           color=INK_2, align='l')
    yE = 2.18
    ffn_x = [bx0 + 0.55, bx0 + 1.9, bx0 + 3.25, bx0 + 4.6]
    for i, fx in enumerate(ffn_x):
        S.rect(fx, yE, 1.05, 0.32, fill='FFFFFF', line=INK, lw=0.75, dash=True,
               runs=[('FFN', ''), (str(i + 1), 'sub')], size=9)
    for s, rx, col in (('x', bx0 + 0.15 + 1.425, BLUE), ('y', cY + 0.525, RED)):
        for fx, p in zip(ffn_x, pis[s]):
            on = p >= 0.27
            S.line(rx, yRt - 0.02, fx + 0.52 + (0.12 if s == 'y' else -0.12), yE + 0.34,
                   color=col if on else GREY, lw=1.0 if on else 0.5, dash=not on, arrow=True)
    # per-stream mixtures, then Add & LN
    yM = 1.72
    for s, x0_, col in (('x', bx0 + 0.15, BLUE), ('y', cY - 0.9, RED)):
        S.rect(x0_, yM, 2.85, 0.26, fill=LIGHT, line=col, lw=0.75,
               runs=[('Σ', ''), ('i∈top-k', 'sub'), (' ', ''), ('π', 'i'), ('i', 'sub i'), (' FFN', ''), ('i', 'sub i'), ('(', ''), ('h', 'i'), (')', '')],
               size=8.5)
        for fx, p in zip(ffn_x, pis[s]):
            if p >= 0.27:
                S.line(fx + 0.52 + (0.12 if s == 'y' else -0.12), yE - 0.02,
                       x0_ + 1.425 + (0.25 if s == 'y' else -0.25), yM + 0.28, color=col, lw=1.0,
                       arrow=True)
    # top Add & LN bar (after the experts) sits just above the block top edge line inside
    yT = by0 + 0.02
    S.rect(bx0 + 0.15, by0 + 0.15, bx1 - bx0 - 0.3, 0.26, fill=LIGHT, line=INK, lw=0.6,
           runs=plain('Add & LN'), size=8.5)
    for cx in (cX, cQX, cY, cQY):
        S.line(cx, yM - 0.02, cx, by0 + 0.43, lw=0.7, arrow=True)
        S.line(cx, by0 + 0.13, cx, 0.42 + tk + 0.02, lw=0.7, arrow=True)
    # ---- outputs above the block
    yO = 0.42
    token(cX, yO, 'x', sub('x', '4'), above=True)
    token(cQX, yO, 'ox', sub('x', '4'), above=True)
    token(cY, yO, 'y', sub('y', '4'), above=True)
    token(cQY, yO, 'oy', sub('y', '4'), above=True)
    S.text(cX + 0.25, yO + 0.02, 1.0, 0.3, plain('next-frame head'), size=7, color=INK_2, align='l')
    S.text(cQX + 0.25, yO + 0.02, 1.3, 0.3, plain('harmonizer estimate'), size=7, color=INK_2, align='l')
    S.text(cY + 0.25, yO + 0.02, 1.0, 0.3, plain('next-frame head'), size=7, color=INK_2, align='l')
    S.text(cQY + 0.25, yO + 0.02, 1.3, 0.3, plain('harmonizer estimate'), size=7, color=INK_2, align='l')

    # ---- inputs below the block: banks and the current tokens
    yBk, yC = 6.4, 7.12
    # context row (bottom left)
    ctx = [(0.55, 'x', sub('x', '1')), (1.0, 'y', sub('y', '1')), (1.45, 'x', sub('x', '2')), (1.9, 'y', sub('y', '2'))]
    for cx, k, lab in ctx:
        token(cx, yC, k, lab)
    S.text(0.4, yC + tk + 0.26, 2.0, 0.2, plain('committed context'), size=8, color=INK_2, align='l')
    # banks
    S.rect(0.4, yBk - 0.06, 1.1, tk + 0.12, fill='FFFFFF', line=BLUE, lw=0.8)
    token(0.65, yBk, 'x'); token(1.1, yBk, 'x')
    S.text(0.4, yBk - 0.28, 1.2, 0.2, [('stream ', ''), ('x', 'i'), (' keys/values', '')], size=7.5, color=BLUE, align='l')
    S.rect(1.75, yBk - 0.06, 1.1, tk + 0.12, fill='FFFFFF', line=RED, lw=0.8)
    token(2.0, yBk, 'y'); token(2.45, yBk, 'y')
    S.text(1.75, yBk - 0.28, 1.2, 0.2, [('stream ', ''), ('y', 'i'), (' keys/values', '')], size=7.5, color=RED, align='l')
    S.line(1.0, yC - 0.02, 0.9, yBk + tk + 0.08, lw=0.9, arrow=True)
    S.line(1.45, yC - 0.02, 2.25, yBk + tk + 0.08, lw=0.9, arrow=True)
    # bank -> projection bars
    S.line(1.5, yBk + 0.05, bx0 + 0.35, yBk + 0.05, color=BLUE, lw=0.9)
    S.line(bx0 + 0.35, yBk + 0.05, bx0 + 0.35, yW + 0.32, color=BLUE, lw=0.9, arrow=True)
    S.line(2.85, yBk + 0.25, cY - 0.7, yBk + 0.25, color=RED, lw=0.9)
    S.line(cY - 0.7, yBk + 0.25, cY - 0.7, yW + 0.32, color=RED, lw=0.9, arrow=True)
    # current tokens and harmonizers under their columns
    token(cX, yC, 'x', sub('x', '3'))
    token(cQX, yC, 'q', sub('q', 'x'))
    token(cY, yC, 'y', sub('y', '3'))
    token(cQY, yC, 'q', sub('q', 'y'))
    for cx in (cX, cQX, cY, cQY):
        S.line(cx, yC - 0.02, cx, yW + 0.32, lw=0.9, arrow=True,
               color=GOLD if cx in (cQX, cQY) else (BLUE if cx == cX else RED))
    S.text(cX - 0.45, yC + tk + 0.26, 1.0, 0.2, plain('last committed'), size=7.5, color=INK_2, align='l')
    S.text(cQX - 0.45, yC + tk + 0.26, 1.6, 0.2, plain('harmonizer: mask + '), size=7.5, color=INK_2, align='l')
    S.text(cQX - 0.45, yC + tk + 0.42, 1.6, 0.2, [('q', 'i'), ('x', 'sub i'), (', or enc(', ''), ('x', 'i'), ('4', 'sub i'), (') + ', ''), ('q', 'i'), ('x', 'sub i'), (' (b2)', '')], size=7.5, color=INK_2, align='l')
    S.text(cY - 0.45, yC + tk + 0.26, 1.0, 0.2, plain('last committed'), size=7.5, color=INK_2, align='l')
    S.text(cQY - 0.45, yC + tk + 0.26, 1.6, 0.2, plain('harmonizer: mask + '), size=7.5, color=INK_2, align='l')
    S.text(cQY - 0.45, yC + tk + 0.42, 1.6, 0.2, [('q', 'i'), ('y', 'sub i')], size=7.5, color=INK_2, align='l')

    # ================================================================ left column
    # legend (top)
    lx, ly = 0.4, 0.55
    S.text(lx, ly - 0.02, 2.0, 0.24, [('Legend', 'b')], size=9, align='l')
    entries = [((BLUE_L, BLUE, False), [('stream ', ''), ('x', 'i'), (' (melody) token', '')]),
               ((RED_L, RED, False), [('stream ', ''), ('y', 'i'), (' (chord) token', '')]),
               ((GOLD_L, GOLD, False), plain('harmonizer token')),
               (('FFFFFF', GOLD, True), plain('masked harmonizer')),
               (('FFFFFF', INK, True), [('initialised from LM', ''), ('s', 'sub i')]),
               ((GOLD_L, GOLD, 'g'), [('scalar gate, ', ''), ('g', 'i'), (' ∈ (0,1)', '')]),
               (('FFFFFF', GREY, 'l'), plain('router → expert not in the top-k'))]
    for k, ((fill, line, dash), lab) in enumerate(entries):
        yy = ly + 0.28 + k * 0.24
        if dash == 'g':
            S.rect(lx, yy, 0.22, 0.22, fill=fill, line=line, lw=0.9, shape='ellipse')
        elif dash == 'l':
            S.line(lx, yy + 0.11, lx + 0.26, yy + 0.11, color=GREY, lw=0.6, dash=True)
        else:
            S.rect(lx, yy + 0.02, 0.26, 0.18, fill=fill, line=line, lw=0.8, dash=bool(dash))
        S.text(lx + 0.34, yy - 0.02, 2.6, 0.26, lab, size=8, align='l')
    S.rect(lx, ly + 0.28 + 7 * 0.24 + 0.02, 0.4, 0.18, fill=BLUE_L, line=BLUE, lw=0.8,
           runs=sub('x', '≤3'), size=6.5)
    S.text(lx + 0.46, ly + 0.28 + 7 * 0.24 - 0.02, 2.6, 0.26, plain('readout: the keys a query may attend'), size=8, align='l')

    # pretrained model (brief), aligned with the block's attention and FFN rows
    px0, px1, py0, py1 = 0.4, 3.35, 2.9, 5.7
    S.rect(px0, py0, px1 - px0, py1 - py0, fill='FFFFFF', line=INK, lw=0.8)
    S.text(px0 + 0.1, py0 + 0.05, 2.9, 0.26, [('Pretrained LM', 'b'), ('s', 'sub i b'), ('  (single stream)', '')], size=9, align='l')
    S.text(px0 + 0.1, py0 + 0.3, 2.9, 0.22, [('one per stream; here LM', ''), ('x', 'sub i'), (' = LM', ''), ('y', 'sub i')], size=7.5, color=INK_2, align='l')
    stack = [(py1 - 0.42, 'local encoder: token → vector', LIGHT),
             (py1 - 0.9, 'attention  W', BLUE_L), (py1 - 1.4, 'FFN', BLUE_L),
             (py1 - 1.95, 'local decoder: vector → token', LIGHT)]
    S.rect(px0 + 0.2, py1 - 1.55, 2.55, 0.85, fill='FFFFFF', line=INK_2, lw=0.6, z=0)
    S.text(px0 + 2.3, py1 - 1.57, 0.45, 0.18, [('×', ''), ('L', 'i')], size=7.5, align='r', color=INK_2)
    S.rect(px0 + 0.3, py1 - 0.42, 2.35, 0.26, fill=LIGHT, line=INK_2, lw=0.6, runs=plain('local encoder'), size=8)
    S.rect(px0 + 0.3, py1 - 0.95, 2.35, 0.26, fill='FFFFFF', line=INK, lw=0.7, dash=True,
           runs=[('attention  ', ''), ('W', 'i'), ('Q', 'sup i'), (', ', ''), ('W', 'i'), ('K', 'sup i'), (', ', ''), ('W', 'i'), ('V', 'sup i'), (', ', ''), ('W', 'i'), ('O', 'sup i')], size=8)
    S.rect(px0 + 0.3, py1 - 1.42, 2.35, 0.26, fill='FFFFFF', line=INK, lw=0.7, dash=True, runs=plain('FFN'), size=8)
    S.rect(px0 + 0.3, py1 - 1.95, 2.35, 0.26, fill=LIGHT, line=INK_2, lw=0.6, runs=plain('local decoder'), size=8)
    for ya, yb_ in ((py1 - 0.42, py1 - 0.69), (py1 - 0.95, py1 - 1.16), (py1 - 1.42, py1 - 1.69)):
        S.line(px0 + 1.475, ya - 0.02, px0 + 1.475, yb_ + 0.02, lw=0.6, arrow=True)
    # where the copies go
    S.line(px0 + 2.65, py1 - 0.82, bx0 - 0.02, yW + 0.15, color=INK_2, lw=0.7, dash=True, arrow=True)
    S.text(px1 + 0.02, py1 - 0.72, 0.6, 0.36, [('→ ', ''), ('W', 'i'), ('s', 'sub i')], size=7.5, color=INK_2, align='l')
    S.line(px0 + 2.65, py1 - 1.29, bx0 - 0.02, yE + 0.16, color=INK_2, lw=0.7, dash=True, arrow=True)
    S.text(px1 + 0.02, py1 - 1.62, 0.6, 0.36, [('→ FFN', ''), ('i', 'sub i')], size=7.5, color=INK_2, align='l')
    S.text(px0, py1 + 0.04, 3.2, 0.2, plain('attention copied once per stream; FFN copied E times'), size=7, color=INK_2, align='l')

    # ================================================================ decoding (right)
    x0 = 10.55
    S.text(x0, 0.12, 4.4, 0.3, [('b', 'b'), ('  Alternating-commit decoding of frame 4', '')], size=11, align='l')
    ts, tg = 0.31, 0.07

    def row(x, y, kinds, labels):
        xs = []
        for kind, lab in zip(kinds, labels):
            fill, line = {'x': (BLUE_L, BLUE), 'y': (RED_L, RED), 'qm': ('FFFFFF', GOLD),
                          'qx': (BLUE_L, BLUE)}[kind]
            S.rect(x, y, ts, ts, fill=fill, line=line, lw=1.1, dash=(kind == 'qm'),
                   runs=sub('x', '4') if kind == 'qx' else None, size=7.5)
            S.text(x - 0.2, y + ts + 0.01, ts + 0.4, 0.2, lab, size=8.5)
            xs.append(x + ts / 2); x += ts + tg
        return xs

    pre_k = ['x', 'y', 'x', 'y', 'x', 'y']
    pre_l = [sub('x', '1'), sub('y', '1'), sub('x', '2'), sub('y', '2'), sub('x', '3'), sub('y', '3')]
    steps = [
        (0.55, '1  draft the leader', ['qm', 'qm'], [sub('q', 'x'), sub('q', 'y')], 'x',
         m('x', ('4', 'sub'), (' ∼ ', 'r'), 'p', ('(', 'r'), 'x', ('4', 'sub'), (' | ', 'r'), 'x', ('≤3', 'sub'),
           (', ', 'r'), 'y', ('≤3', 'sub'), (')', 'r')),
         [('both harmonizers masked; ', ''), ('x', 'i'), ('4', 'sub i'), (' from the stream-', ''), ('x', 'i'), (' next-frame head', '')]),
        (2.85, '2  condition the follower', ['qx', 'qm'], [sub('q', 'x'), sub('q', 'y')], 'y',
         m('y', ('4', 'sub'), (' ∼ ', 'r'), 'p', ('(', 'r'), 'y', ('4', 'sub'), (' | ', 'r'), 'x', ('4', 'sub'),
           (', ', 'r'), 'x', ('≤3', 'sub'), (', ', 'r'), 'y', ('≤3', 'sub'), (')', 'r')),
         [('x', 'i'), ('4', 'sub i'), (' written into ', ''), ('q', 'i'), ('x', 'sub i'), ('; ', ''), ('q', 'i'), ('y', 'sub i'),
          (' reads it through ', ''), ('K', 'i'), ('frame', 'sub'), (' and emits ', ''), ('y', 'i'), ('4', 'sub i')]),
        (5.15, '3  commit both, swap roles', ['x', 'y'], [sub('x', '4'), sub('y', '4')], None, None,
         [('x', 'i'), ('4', 'sub i'), (', ', ''), ('y', 'i'), ('4', 'sub i'), (' join the context; at frame 5 stream ', ''), ('y', 'i'), (' leads', '')]),
    ]
    for y0, title, qk, ql, out, formula, note in steps:
        S.text(x0, y0, 4.4, 0.24, [(title, 'b')], size=9.5, align='l')
        yr_ = y0 + 0.85
        xs = row(x0, yr_, pre_k + qk, pre_l + ql)
        S.line(x0, yr_ + ts + 0.24, xs[5] + ts / 2, yr_ + ts + 0.24, color=INK_2, lw=0.6)
        S.text(x0, yr_ + ts + 0.25, xs[5] + ts / 2 - x0, 0.18, plain('context'), size=7.5, color=INK_2)
        if formula:
            src = xs[4] if out == 'x' else xs[7]
            fill, line = (BLUE_L, BLUE) if out == 'x' else (RED_L, RED)
            S.line(src, yr_ - 0.03, src, yr_ - 0.26, arrow=True, lw=0.9)
            S.rect(src - ts / 2, yr_ - 0.58, ts, ts, fill=fill, line=line, lw=1.1)
            if out == 'x':
                S.text(src + ts / 2 + 0.06, yr_ - 0.6, 2.4, 0.36, formula, size=9, align='l')
            else:
                S.text(src - ts / 2 - 2.55, yr_ - 0.6, 2.45, 0.36, formula, size=9, align='r')
                S.line(xs[6] + ts / 2 - 0.03, yr_ - 0.08, xs[7] - ts / 2 + 0.02, yr_ - 0.08, color=GOLD,
                       lw=1.4, arrow=True)
        S.text(x0, y0 + 1.72, 4.4, 0.22, note, size=7.5, color=INK_2, align='l')
    # panel letter for (a)
    S.text(0.4, 0.1, 3.4, 0.28, [('a', 'b'), ('  Model: one block (×', ''), ('L', 'i'), ('), frame 4', '')], size=11, align='l')
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
