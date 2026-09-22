"""Model figure v4: the design of 2026-09-22 (image 3.png) built out.

Layout, as drawn: decoding strip across the top; legend and a brief
pretrained-model box on the left; the transformer block on the right
with attention (W_{q,k,v} per stream) at the bottom, the gated readouts,
routers with pi bars, Exp 1-4, per-stream mixture, Add & Norm, and the
predicted tokens above; the context row at the bottom left, split into
a stream-x bank and a stream-y bank that feed the attention; the last
committed x_3, y_3 and the harmonizers q_x, q_y under their columns.

Completed relative to the drawing: three readouts per query (own
stream, other stream through g^c_s, partner through g^f_s), W^O_s and a
second Add & Norm after attention, harmonizer columns with the other
harmonizer as partner, mixture boxes as the top-k weighted sum, the
pretrained model's contents and where its parts go. Indices start at 1
as in the paper: context x_1 y_1 x_2 y_2, last committed x_3 y_3,
frame 4 predicted.

Usage:
  python fig_arch_v4.py --out results/fig_model_v4     # .pptx + .png preview
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_arch_pptx import Spec, write_pptx_js, write_preview, m, plain, INK, INK_2  # noqa: E402
import fig_arch_pptx

fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 15.5, 7.5
# the drawing's colours
LAV, LAV_D = 'B8BBEC', '8F93D9'          # stream x
TEAL, TEAL_D = 'B7DBD8', '7CBDB7'        # stream y
HARM, HARM_D = 'F6E2A6', 'D9A93A'        # harmonizer
GREY, GREY_D = 'D9D9D9', 'A6A6A6'
LIGHT = 'EFEFEF'
DASH = '9A9A9A'


def sub(base, s):
    return m(base, (s, 'sub'))


def Wqkv(s):
    return [('W', 'i'), ('Q', 'sup i'), (s, 'sub i'), (', ', ''), ('W', 'i'), ('K', 'sup i'), (s, 'sub i'),
            (', ', ''), ('W', 'i'), ('V', 'sup i'), (s, 'sub i')]


def build():
    S = Spec()
    tk = 0.4

    def token(cx, y, kind, lab, dash=False, size=8.5):
        fill, line = {'x': (LAV, LAV_D), 'y': (TEAL, TEAL_D), 'q': (HARM, HARM_D),
                      'qm': ('FFFFFF', HARM_D), 'ox': ('FFFFFF', LAV_D), 'oy': ('FFFFFF', TEAL_D)}[kind]
        S.rect(cx - tk / 2, y, tk, tk, fill=fill, line=line, lw=1.0, dash=dash or kind == 'qm',
               runs=lab, size=size)

    # ================================================================ block
    # two lanes, one per stream; each lane holds that stream's components
    # ONCE (W_s, W^O_s, G^(s), mixture) and carries two tokens through them:
    # the last committed token and the harmonizer of the stream
    cX, cQX, cY, cQY = 4.95, 6.65, 8.9, 10.6        # slots: x lane (x_3, q_x), y lane (y_3, q_y)
    lanes = (('x', cX, cQX), ('y', cY, cQY))
    bx0, bx1, by0, by1 = 3.65, 11.85, 0.72, 5.6
    S.rect(bx0, by0, bx1 - bx0, by1 - by0, fill='FFFFFF', line=INK, lw=1.0, z=0)
    S.text(bx1 - 0.7, by0 + 0.05, 0.6, 0.24, [('×', ''), ('L', 'i')], size=10, align='r')
    colour = {'x': (LAV, LAV_D), 'y': (TEAL, TEAL_D)}

    def plus(cx, cy, r=0.1):
        S.rect(cx - r, cy - r, 2 * r, 2 * r, fill='FFFFFF', line=INK, lw=0.7, shape='ellipse', z=4)
        S.line(cx - 0.05, cy, cx + 0.05, cy, lw=0.7, z=5)
        S.line(cx, cy - 0.05, cx, cy + 0.05, lw=0.7, z=5)

    def lane_bar(s, y, h, runs, size, fill=None, line=None, dash=False):
        _, c1, c2 = [l for l in lanes if l[0] == s][0]
        x0_, x1_ = c1 - 0.75, c2 + 0.75
        S.rect(x0_, y, x1_ - x0_, h, fill=fill or colour[s][0], line=line or INK, lw=0.7, dash=dash,
               runs=runs, size=size)
        return x0_, x1_

    # attention bar with one projection set per stream
    yAt = 4.95
    S.rect(bx0 + 0.2, yAt, bx1 - bx0 - 0.4, 0.5, fill=GREY, line=None, z=0)
    S.text(bx0 + 0.25, yAt + 0.01, 1.2, 0.2, plain('attention'), size=8, align='l', color=INK_2)
    for s, c1, c2 in lanes:
        lane_bar(s, yAt + 0.2, 0.26, Wqkv(s), 8.5, dash=True)
    # readouts per token, labelled with the key set
    yR = 4.3
    bw, bg = 0.38, 0.05
    keys = {cX: (sub('x', '≤3'), sub('y', '≤2'), sub('y', '3')),
            cQX: (sub('x', '≤3'), sub('y', '≤3'), sub('q', 'y')),
            cY: (sub('y', '≤3'), sub('x', '≤2'), sub('x', '3')),
            cQY: (sub('y', '≤3'), sub('x', '≤3'), sub('q', 'x'))}
    r = 0.13
    yG, ySum, yWo = 3.88, 3.55, 3.15
    for s, c1, c2 in lanes:
        own = colour[s]; oth = colour['y' if s == 'x' else 'x']
        for cx in (c1, c2):
            for k, (lab, (fill, line)) in enumerate(zip(keys[cx], (own, oth, (HARM, HARM_D)))):
                x = cx + (k - 1) * (bw + bg)
                S.rect(x - bw / 2, yR, bw, 0.28, fill=fill, line=line, lw=0.7, runs=lab, size=7.5)
                S.line(x, yAt + 0.18, x, yR + 0.3, lw=0.6, arrow=True, color=INK_2)
            plus(cx, ySum)
            S.line(cx - (bw + bg), yR - 0.02, cx - 0.08, ySum + 0.08, lw=0.6, arrow=True)
            for k, kind in ((0, 'c'), (1, 'f')):
                gx = cx + k * (bw + bg)
                S.rect(gx - r, yG - r, 2 * r, 2 * r, fill=own[0], line=INK_2, lw=0.6,
                       shape='ellipse', runs=[('g', 'i'), (kind, 'sup i'), (s, 'sub i')], size=7, z=5)
                S.line(gx, yR - 0.02, gx, yG + r + 0.02, lw=0.6, arrow=True)
                S.line(gx - 0.03, yG - r, cx + 0.04, ySum + 0.08, lw=0.6, arrow=True)
            S.line(cx, ySum - 0.1, cx, yWo + 0.27, lw=0.6, arrow=True)
        # one output projection per stream
        lane_bar(s, yWo, 0.25, [('output projection  ', ''), ('W', 'i'), ('O', 'sup i'), (s, 'sub i')], 8, dash=True)
    # Add & Norm after attention
    yA1 = 2.72
    S.rect(bx0 + 0.2, yA1, bx1 - bx0 - 0.4, 0.28, fill=GREY, line=None, runs=plain('add & norm'), size=8.5)
    for _, c1, c2 in lanes:
        for cx in (c1, c2):
            S.line(cx, yWo - 0.02, cx, yA1 + 0.3, lw=0.6, arrow=True)
    # one router per stream, pi bars at the outer sides
    yRt = 2.22
    pis = {'x': (0.61, 0.08, 0.27, 0.04), 'y': (0.04, 0.27, 0.61, 0.08)}
    for s, c1, c2 in lanes:
        lane_bar(s, yRt, 0.28, [('router ', ''), ('G', 'i'), ('(' + s + ')', 'sup i')], 8.5, line=colour[s][1])
        for cx in (c1, c2):
            S.line(cx, yA1 - 0.02, cx, yRt + 0.3, lw=0.6, arrow=True)
    for s, x0_ in (('x', bx0 + 0.05), ('y', bx1 - 0.44)):
        for i, p in enumerate(pis[s]):
            S.rect(x0_ + i * 0.1, yRt + 0.28 - 0.28 * p, 0.08, 0.28 * p, fill=colour[s][1], line=None)
    S.text(bx0 + 0.02, yRt - 0.2, 0.5, 0.18, [('π', 'i')], size=8, align='l', color=INK_2)
    S.text(bx1 - 0.5, yRt - 0.2, 0.5, 0.18, [('π', 'i')], size=8, align='r', color=INK_2)
    # experts, shared by both streams
    yE = 1.68
    ex = [bx0 + 1.0, bx0 + 2.6, bx0 + 4.2, bx0 + 5.8]
    for i, fx in enumerate(ex):
        S.rect(fx, yE, 1.15, 0.3, fill=GREY, line=INK, lw=0.6, dash=True,
               runs=[('Exp ', ''), (str(i + 1), '')], size=8.5)
    for s, c1, c2 in lanes:
        rx = (c1 + c2) / 2
        for fx, p in zip(ex, pis[s]):
            on = p >= 0.27
            S.line(rx, yRt - 0.02, fx + 0.575 + (0.1 if s == 'y' else -0.1), yE + 0.32,
                   color=colour[s][1] if on else DASH, lw=0.9 if on else 0.5, dash=not on, arrow=True)
    # one mixture per stream ("output" in the drawing), then Add & Norm and the predicted tokens
    yM = 1.3
    for s, c1, c2 in lanes:
        x0_, x1_ = lane_bar(s, yM, 0.26, [('output  Σ', ''), ('top-k', 'sub'), (' ', ''), ('π', 'i'), ('i', 'sub i'), (' Exp', ''), ('i', 'sub i'), ('(', ''), ('h', 'i'), (')', '')], 8, line=colour[s][1])
        mid = (x0_ + x1_) / 2
        for fx, p in zip(ex, pis[s]):
            if p >= 0.27:
                S.line(fx + 0.575 + (0.1 if s == 'y' else -0.1), yE - 0.02, mid + (0.3 if s == 'y' else -0.3),
                       yM + 0.28, color=colour[s][1], lw=0.9, arrow=True)
    yA2 = by0 + 0.08
    S.rect(bx0 + 0.2, yA2, bx1 - bx0 - 0.4, 0.28, fill=GREY, line=None, runs=plain('add & norm'), size=8.5)
    for _, c1, c2 in lanes:
        for cx in (c1, c2):
            S.line(cx, yM - 0.02, cx, yA2 + 0.3, lw=0.6, arrow=True)
            S.line(cx, yA2 - 0.02, cx, 0.2 + tk + 0.02, lw=0.6, arrow=True)
    token(cX, 0.2, 'x', sub('x', '4'))
    token(cQX, 0.2, 'ox', sub('x', '4'))
    token(cY, 0.2, 'y', sub('y', '4'))
    token(cQY, 0.2, 'oy', sub('y', '4'))
    S.text(cX + 0.25, 0.24, 1.1, 0.16, plain('next-frame head'), size=6.5, color=INK_2, align='l')
    S.text(cQX + 0.25, 0.24, 1.4, 0.16, plain('harmonizer estimate'), size=6.5, color=INK_2, align='l')
    S.text(cY + 0.25, 0.24, 1.1, 0.16, plain('next-frame head'), size=6.5, color=INK_2, align='l')
    S.text(cQY + 0.25, 0.24, 1.4, 0.16, plain('harmonizer estimate'), size=6.5, color=INK_2, align='l')
    S.text(bx0 + 0.25, yG - 0.08, 0.6, 0.16, plain('gates'), size=6.5, color=INK_2, align='l')
    S.text(bx0 + 0.25, yR + 0.06, 0.6, 0.16, plain('keys'), size=6.5, color=INK_2, align='l')
    S.text(cX - 0.75, yAt + 0.01, 3.4, 0.18, [('stream ', ''), ('x', 'i'), (' lane: ', ''), ('x', 'i'), ('3', 'sub i'), (' and ', ''), ('q', 'i'), ('x', 'sub i'), (' share every component', '')], size=6.5, color=INK_2, align='r')
    S.text(cY - 0.75, yAt + 0.01, 3.4, 0.18, [('stream ', ''), ('y', 'i'), (' lane: ', ''), ('y', 'i'), ('3', 'sub i'), (' and ', ''), ('q', 'i'), ('y', 'sub i'), (' share every component', '')], size=6.5, color=INK_2, align='r')

    # ================================================================ context, banks, current tokens
    yC = 6.7
    ctx = [(0.55, 'x', '1'), (1.0, 'y', '1'), (1.45, 'x', '2'), (1.9, 'y', '2')]
    for cx, k, i in ctx:
        token(cx, yC, k, sub(k, i))
    S.text(0.35, yC + tk + 0.02, 2.0, 0.18, plain('context'), size=8, color=INK_2, align='l')
    yB = 5.95
    S.rect(0.4, yB - 0.07, 1.15, tk + 0.14, fill='FFFFFF', line=INK, lw=0.7)
    token(0.68, yB, 'x', sub('x', '1')); token(1.13, yB, 'x', sub('x', '2'))
    S.rect(1.8, yB - 0.07, 1.15, tk + 0.14, fill='FFFFFF', line=INK, lw=0.7)
    token(2.08, yB, 'y', sub('y', '1')); token(2.53, yB, 'y', sub('y', '2'))
    S.line(1.2, yC - 0.02, 1.0, yB + tk + 0.1, lw=1.4, arrow=True)
    S.line(1.3, yC - 0.02, 2.3, yB + tk + 0.1, lw=1.4, arrow=True)
    # banks -> attention projections (elbow lines, as drawn)
    S.line(0.97, yB - 0.09, 0.97, yB - 0.35, lw=0.8)
    S.line(0.97, yB - 0.35, cX - 0.6, yB - 0.35, lw=0.8)
    S.line(cX - 0.6, yB - 0.35, cX - 0.6, yAt + 0.48, lw=0.8, arrow=True)
    S.line(2.37, yB - 0.09, 2.37, yB - 0.2, lw=0.8)
    S.line(2.37, yB - 0.2, cY - 0.6, yB - 0.2, lw=0.8)
    S.line(cY - 0.6, yB - 0.2, cY - 0.6, yAt + 0.48, lw=0.8, arrow=True)
    # current tokens and harmonizers, in sequence order; each enters its stream's lane
    bX, bY, bQX, bQY = cX, cQX, cY, cQY              # bottom positions: x_3, y_3, q_x, q_y
    token(bX, yC, 'x', sub('x', '3'))
    token(bY, yC, 'y', sub('y', '3'))
    token(bQX, yC, 'q', sub('q', 'x'))
    token(bQY, yC, 'q', sub('q', 'y'))
    S.line(bX, yC - 0.02, bX, yAt + 0.48, lw=0.8, arrow=True)
    S.line(bY, yC - 0.02, cY, yAt + 0.48, lw=0.8, arrow=True)
    S.line(bQX, yC - 0.02, cQX, yAt + 0.48, lw=0.8, arrow=True)
    S.line(bQY, yC - 0.02, cQY, yAt + 0.48, lw=0.8, arrow=True)
    S.text(bX - 0.6, yC + tk + 0.02, 1.2, 0.18, plain('last committed'), size=7, color=INK_2)
    S.text(bY - 0.6, yC + tk + 0.02, 1.2, 0.18, plain('last committed'), size=7, color=INK_2)
    S.text(bQX - 0.75, yC + tk + 0.02, 1.5, 0.18, plain('harmonizer, predicts frame 4'), size=7, color=INK_2)
    S.text(bQY - 0.75, yC + tk + 0.02, 1.5, 0.18, plain('harmonizer, predicts frame 4'), size=7, color=INK_2)

    # ================================================================ left column
    # legend
    lx0, ly0 = 0.4, 0.3
    S.rect(lx0, ly0, 2.9, 1.52, fill='FFFFFF', line=INK, lw=0.8)
    S.text(lx0 + 0.1, ly0 + 0.04, 1.5, 0.22, plain('legend'), size=9, align='l')
    entries = [((LAV, LAV_D, False), [('stream ', ''), ('x', 'i'), (' (melody) token', '')]),
               ((TEAL, TEAL_D, False), [('stream ', ''), ('y', 'i'), (' (chord) token', '')]),
               ((HARM, HARM_D, False), plain('harmonizer token')),
               (('FFFFFF', HARM_D, True), plain('masked harmonizer')),
               (('FFFFFF', INK, True), [('copied from the pretrained LM', ''), ('s', 'sub i')])]
    for k, ((fill, line, dash), lab) in enumerate(entries):
        yy = ly0 + 0.34 + k * 0.23
        if dash == 'l':
            S.line(lx0 + 0.12, yy + 0.1, lx0 + 0.36, yy + 0.1, color=DASH, lw=0.6, dash=True)
        else:
            S.rect(lx0 + 0.12, yy + 0.02, 0.24, 0.16, fill=fill, line=line, lw=0.7, dash=bool(dash))
        S.text(lx0 + 0.44, yy - 0.03, 2.4, 0.24, lab, size=7.5, align='l')
    # pretrained model, brief
    px0, py0 = 0.4, 2.35
    S.rect(px0, py0, 2.9, 2.0, fill='FFFFFF', line=INK, lw=0.8)
    S.text(px0 + 0.1, py0 + 0.04, 2.7, 0.22, [('pretrained LM', ''), ('s', 'sub i'), ('  (single stream)', '')], size=9, align='l')
    S.text(px0 + 0.1, py0 + 0.26, 2.7, 0.18, [('one per stream; here LM', ''), ('x', 'sub i'), (' = LM', ''), ('y', 'sub i')], size=7, color=INK_2, align='l')
    rows = [(py0 + 0.55, 'local decoder', LIGHT, INK_2, False),
            (py0 + 0.87, 'FFN', 'FFFFFF', INK, True),
            (py0 + 1.19, 'attention  W^Q, W^K, W^V, W^O', 'FFFFFF', INK, True),
            (py0 + 1.51, 'local encoder', LIGHT, INK_2, False)]
    for yy, lab, fill, line, dash in rows:
        runs = plain(lab) if 'W' not in lab else [('attention  ', ''), ('W', 'i'), ('Q', 'sup i'), (', ', ''), ('W', 'i'), ('K', 'sup i'),
                                                  (', ', ''), ('W', 'i'), ('V', 'sup i'), (', ', ''), ('W', 'i'), ('O', 'sup i')]
        S.rect(px0 + 0.5, yy, 2.2, 0.24, fill=fill, line=line, lw=0.7, dash=dash, runs=runs, size=7.5)
    S.rect(px0 + 0.4, py0 + 0.8, 2.4, 0.7, fill=None, line=INK_2, lw=0.5, z=0)
    S.text(px0 + 0.1, py0 + 0.85, 0.3, 0.2, [('×', ''), ('L', 'i')], size=7, color=INK_2, align='l')
    for yy in (py0 + 1.51, py0 + 1.19, py0 + 0.87):
        S.line(px0 + 1.6, yy - 0.02, px0 + 1.6, yy - 0.08, lw=0.5, arrow=True)
    S.text(px0 + 0.1, py0 + 1.78, 2.7, 0.18, plain('attention → W per stream;  FFN → every expert'), size=7, color=INK_2, align='l')
    # arrows to the block: attention -> attention bar, FFN -> experts
    S.line(px0 + 2.72, py0 + 1.31, bx0 - 0.02, yAt + 0.33, color=INK_2, lw=0.7, dash=True, arrow=True)
    S.line(px0 + 2.72, py0 + 0.99, bx0 - 0.02, yE + 0.15, color=INK_2, lw=0.7, dash=True, arrow=True)

    # ================================================================ decoding (right column, stacked)
    dx0, dx1 = 12.15, 15.3
    S.rect(dx0, 0.3, dx1 - dx0, 6.95, fill='FFFFFF', line=INK, lw=0.8)
    S.text(dx0 + 0.1, 0.34, 3.0, 0.22, plain('decoding frame 4 (alternating commit)'), size=9, align='l')
    ts, tg = 0.28, 0.04

    def row(x, y, kinds, labs):
        xs = []
        for kind, lab in zip(kinds, labs):
            fill, line = {'x': (LAV, LAV_D), 'y': (TEAL, TEAL_D), 'qm': ('FFFFFF', HARM_D),
                          'qx': (LAV, LAV_D)}[kind]
            S.rect(x, y, ts, ts, fill=fill, line=line, lw=0.9, dash=(kind == 'qm'), runs=lab, size=6.5)
            xs.append(x + ts / 2); x += ts + tg
        return xs

    pre_k = ['x', 'y', 'x', 'y', 'x', 'y']
    pre_l = [sub('x', '1'), sub('y', '1'), sub('x', '2'), sub('y', '2'), sub('x', '3'), sub('y', '3')]
    steps = [(0.75, '1  draft the leader', ['qm', 'qm'], [sub('q', 'x'), sub('q', 'y')], 'x',
              m('x', ('4', 'sub'), (' ∼ ', 'r'), 'p', ('(', 'r'), 'x', ('4', 'sub'), (' | ', 'r'), 'x', ('≤3', 'sub'), (', ', 'r'), 'y', ('≤3', 'sub'), (')', 'r')),
              [('harmonizers masked; ', ''), ('x', 'i'), ('4', 'sub i'), (' is sampled from the', '')],
              [('stream-', ''), ('x', 'i'), (' next-frame head', '')]),
             (2.95, '2  condition the follower', ['qx', 'qm'], [sub('x', '4'), sub('q', 'y')], 'y',
              m('y', ('4', 'sub'), (' ∼ ', 'r'), 'p', ('(', 'r'), 'y', ('4', 'sub'), (' | ', 'r'), 'x', ('4', 'sub'), (', ', 'r'), 'x', ('≤3', 'sub'), (', ', 'r'), 'y', ('≤3', 'sub'), (')', 'r')),
              [('x', 'i'), ('4', 'sub i'), (' is written into ', ''), ('q', 'i'), ('x', 'sub i'), ('; ', ''), ('q', 'i'), ('y', 'sub i'), (' reads it as', '')],
              [('its partner and emits ', ''), ('y', 'i'), ('4', 'sub i')]),
             (5.15, '3  commit both, swap roles', ['x', 'y'], [sub('x', '4'), sub('y', '4')], None, None,
              [('x', 'i'), ('4', 'sub i'), (', ', ''), ('y', 'i'), ('4', 'sub i'), (' join the context;', '')],
              [('at frame 5 stream ', ''), ('y', 'i'), (' leads', '')])]
    for y0, title, qk, ql, out, formula, note1, note2 in steps:
        S.text(dx0 + 0.1, y0, 3.0, 0.2, [(title, 'b')], size=8, align='l')
        yr_ = y0 + 0.55
        xs = row(dx0 + 0.1, yr_, pre_k + qk, pre_l + ql)
        if formula:
            src = xs[4] if out == 'x' else xs[7]
            fill, line = (LAV, LAV_D) if out == 'x' else (TEAL, TEAL_D)
            ox_ = xs[-1] + 0.3
            S.line(src, yr_ - 0.02, src, yr_ - 0.2, lw=0.7)
            S.line(src, yr_ - 0.2, ox_ + ts / 2, yr_ - 0.2, lw=0.7)
            S.line(ox_ + ts / 2, yr_ - 0.2, ox_ + ts / 2, yr_ - 0.04, lw=0.7, arrow=True)
            S.rect(ox_, yr_ - 0.02, ts, ts, fill=fill, line=line, lw=0.9,
                   runs=sub('x', '4') if out == 'x' else sub('y', '4'), size=6.5)
            S.text(dx0 + 0.1, yr_ + ts + 0.04, 3.0, 0.2, formula, size=7.5, align='l')
            if out == 'y':
                S.line(xs[6] + ts / 2 - 0.02, yr_ - 0.08, xs[7] - ts / 2 + 0.02, yr_ - 0.08, color=HARM_D, lw=1.2, arrow=True)
        S.text(dx0 + 0.1, yr_ + ts + 0.28, 3.0, 0.18, note1, size=7, color=INK_2, align='l')
        S.text(dx0 + 0.1, yr_ + ts + 0.44, 3.0, 0.18, note2, size=7, color=INK_2, align='l')
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    S = build()
    write_pptx_js(S, args.out + '.pptx')
    write_preview(S, args.out + '.png'); print('wrote', args.out + '.png')


if __name__ == '__main__':
    main()
