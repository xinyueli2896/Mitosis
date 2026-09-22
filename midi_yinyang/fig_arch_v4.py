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

fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 16.0, 7.8
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
    tk = 0.46

    def token(cx, y, kind, lab, dash=False, size=8.5):
        fill, line = {'x': (LAV, LAV_D), 'y': (TEAL, TEAL_D), 'q': (HARM, HARM_D),
                      'qm': ('FFFFFF', HARM_D), 'ox': ('FFFFFF', LAV_D), 'oy': ('FFFFFF', TEAL_D)}[kind]
        S.rect(cx - tk / 2, y, tk, tk, fill=fill, line=INK if kind == 'q' else line,
               lw=1.3 if kind == 'q' else 1.0, dash=dash or kind == 'qm', runs=lab, size=size)

    # ================================================================ block
    # one lane per stream for the content token; harmonizers are only
    # pointed into the block and read out above it (their target is the
    # ground-truth token of the frame). As few arrows and words as possible.
    cX, cY = 5.6, 9.5                                 # lane centres (content tokens)
    hX, hY = 7.2, 11.1                                # harmonizer columns
    lanes = (('x', cX), ('y', cY))
    LW = 2.5
    bx0, bx1, by0, by1 = 3.65, 11.85, 0.68, 5.9
    S.rect(bx0, by0, bx1 - bx0, by1 - by0, fill='FFFFFF', line=INK, lw=1.0, z=0)
    S.text(bx1 - 0.7, by0 + 0.05, 0.6, 0.24, [('×', ''), ('L', 'i')], size=10, align='r')
    colour = {'x': (LAV, LAV_D), 'y': (TEAL, TEAL_D)}

    def plus(cx, cy, r=0.11):
        S.rect(cx - r, cy - r, 2 * r, 2 * r, fill='FFFFFF', line=INK, lw=0.7, shape='ellipse', z=4)
        S.line(cx - 0.05, cy, cx + 0.05, cy, lw=0.7, z=5)
        S.line(cx, cy - 0.05, cx, cy + 0.05, lw=0.7, z=5)

    def lane_bar(s, y, h, runs, size, dash=False):
        cx = dict(lanes)[s]
        S.rect(cx - LW / 2, y, LW, h, fill=colour[s][0], line=INK, lw=0.7, dash=dash, runs=runs, size=size)

    # attention: one projection set per stream
    yAt = 5.25
    S.rect(bx0 + 0.2, yAt, bx1 - bx0 - 0.4, 0.5, fill=GREY, line=None, z=0)
    for s, cx in lanes:
        lane_bar(s, yAt + 0.17, 0.3, Wqkv(s), 8.5, dash=True)
    # three readouts, gates, sum, W^O
    yR = 4.55
    bw, bg = 0.6, 0.1
    keys = {'x': (sub('x', '≤3'), sub('y', '≤2'), sub('y', '3')),
            'y': (sub('y', '≤3'), sub('x', '≤2'), sub('x', '3'))}
    r = 0.17
    yG, ySum, yWo = 4.08, 3.68, 3.28
    for s, cx in lanes:
        own = colour[s]; oth = colour['y' if s == 'x' else 'x']
        for k, (lab, (fill, line)) in enumerate(zip(keys[s], (own, oth, (HARM, HARM_D)))):
            x = cx + (k - 1) * (bw + bg)
            S.rect(x - bw / 2, yR, bw, 0.3, fill=fill, line=line, lw=0.7, runs=lab, size=7.5)
            S.line(x, yAt + 0.15, x, yR + 0.32, lw=0.6, arrow=True)
        plus(cx, ySum)
        S.line(cx - (bw + bg), yR - 0.02, cx - 0.08, ySum + 0.08, lw=0.6, arrow=True)
        for k, kind in ((0, 'c'), (1, 'f')):
            gx = cx + k * (bw + bg)
            S.rect(gx - r, yG - r, 2 * r, 2 * r, fill=own[0], line=INK, lw=0.7, shape='ellipse',
                   runs=[('g', 'i'), (kind, 'sup i'), (s, 'sub i')], size=7, z=5)
            S.line(gx, yR - 0.02, gx, yG + r + 0.02, lw=0.6, arrow=True)
            S.line(gx - 0.03, yG - r, cx + 0.04, ySum + 0.08, lw=0.6, arrow=True)
        S.line(cx, ySum - 0.1, cx, yWo + 0.27, lw=0.6, arrow=True)
        lane_bar(s, yWo, 0.3, [('W', 'i'), ('O', 'sup i'), (s, 'sub i')], 8.5, dash=True)
    yA1 = 2.85
    S.rect(bx0 + 0.2, yA1, bx1 - bx0 - 0.4, 0.3, fill=GREY, line=None, runs=plain('add & norm'), size=8.5)
    for s, cx in lanes:
        S.line(cx, yWo - 0.02, cx, yA1 + 0.3, lw=0.6, arrow=True)
    # routers, experts (top-2 only), mixture
    yRt = 2.32
    pis = {'x': (0.61, 0.08, 0.27, 0.04), 'y': (0.04, 0.27, 0.61, 0.08)}
    for s, cx in lanes:
        lane_bar(s, yRt, 0.3, [('router ', ''), ('G', 'i'), ('(' + s + ')', 'sup i')], 8.5)
        S.line(cx, yA1 - 0.02, cx, yRt + 0.3, lw=0.6, arrow=True)
    for s, x0_ in (('x', bx0 + 0.05), ('y', bx1 - 0.44)):
        for i, p in enumerate(pis[s]):
            S.rect(x0_ + i * 0.1, yRt + 0.28 - 0.28 * p, 0.08, 0.28 * p, fill=colour[s][1], line=None)
    yE = 1.75
    ex = [bx0 + 1.0, bx0 + 2.6, bx0 + 4.2, bx0 + 5.8]
    for i, fx in enumerate(ex):
        S.rect(fx, yE, 1.25, 0.32, fill=GREY, line=INK, lw=0.6, dash=True, runs=[('Exp ', ''), (str(i + 1), '')], size=8.5)
    for s, cx in lanes:
        for fx, p in zip(ex, pis[s]):
            if p >= 0.27:
                S.line(cx, yRt - 0.02, fx + 0.625 + (0.1 if s == 'y' else -0.1), yE + 0.34, color=colour[s][1], lw=0.9, arrow=True)
    yM = 1.3
    for s, cx in lanes:
        lane_bar(s, yM, 0.26, plain('output'), 8.5)
        for fx, p in zip(ex, pis[s]):
            if p >= 0.27:
                S.line(fx + 0.625 + (0.1 if s == 'y' else -0.1), yE - 0.02, cx + (0.3 if fx + 0.6 > cx else -0.3), yM + 0.28, color=colour[s][1], lw=0.9, arrow=True)
    yA2 = by0 + 0.08
    S.rect(bx0 + 0.2, yA2, bx1 - bx0 - 0.4, 0.3, fill=GREY, line=None, runs=plain('add & norm'), size=8.5)
    # outputs: content lanes predict the next frame; harmonizers are read out
    # straight above where they entered, their target being the same frame
    for s, cx in lanes:
        S.line(cx, yM - 0.02, cx, yA2 + 0.32, lw=0.6, arrow=True)
        S.line(cx, yA2 - 0.02, cx, 0.12 + tk + 0.02, lw=0.6, arrow=True)
        token(cx, 0.12, s, sub(s, '4'))
    for s, hx in (('x', hX), ('y', hY)):
        S.line(hx, by0 - 0.02, hx, 0.12 + tk + 0.02, lw=0.6, arrow=True)
        token(hx, 0.12, 'o' + s, sub(s, '4'))
    S.text(hX - 0.6, 0.6, 1.2, 0.2, plain('objective'), size=6.5, color=INK_2)
    S.text(hY - 0.6, 0.6, 1.2, 0.2, plain('objective'), size=6.5, color=INK_2)

    # ================================================================ context, banks, current tokens
    yC = 6.85
    ctx = [(0.55, 'x', '1'), (1.05, 'y', '1'), (1.55, 'x', '2'), (2.05, 'y', '2')]
    for cx, k, i in ctx:
        token(cx, yC, k, sub(k, i))
    S.text(0.35, yC + tk + 0.02, 2.0, 0.18, plain('context'), size=8, color=INK_2, align='l')
    yB = 6.15
    S.rect(0.38, yB - 0.07, 1.2, tk + 0.14, fill='FFFFFF', line=INK, lw=0.7)
    token(0.66, yB, 'x', sub('x', '1')); token(1.3, yB, 'x', sub('x', '2'))
    S.rect(1.85, yB - 0.07, 1.2, tk + 0.14, fill='FFFFFF', line=INK, lw=0.7)
    token(2.13, yB, 'y', sub('y', '1')); token(2.77, yB, 'y', sub('y', '2'))
    S.line(1.3, yC - 0.02, 1.0, yB + tk + 0.1, lw=1.4, arrow=True)
    S.line(1.4, yC - 0.02, 2.4, yB + tk + 0.1, lw=1.4, arrow=True)
    S.line(0.97, yB - 0.09, 0.97, yB - 0.35, lw=0.8)
    S.line(0.97, yB - 0.35, cX - 1.05, yB - 0.35, lw=0.8)
    S.line(cX - 1.05, yB - 0.35, cX - 1.05, yAt + 0.48, lw=0.8, arrow=True)
    S.line(2.45, yB - 0.09, 2.45, yB - 0.2, lw=0.8)
    S.line(2.45, yB - 0.2, cY - 1.05, yB - 0.2, lw=0.8)
    S.line(cY - 1.05, yB - 0.2, cY - 1.05, yAt + 0.48, lw=0.8, arrow=True)
    # the frame's four tokens: the content tokens enter their lane; the harmonizers point into the block
    token(cX, yC, 'x', sub('x', '3')); token(cY, yC, 'y', sub('y', '3'))
    token(hX, yC, 'q', sub('q', 'x')); token(hY, yC, 'q', sub('q', 'y'))
    for cx in (cX, cY):
        S.line(cx, yC - 0.02, cx, yAt + 0.48, lw=0.8, arrow=True)
    for hx in (hX, hY):
        S.line(hx, yC - 0.02, hx, by1 + 0.02, lw=0.8, arrow=True)

    # ================================================================ left column
    # legend
    lx0, ly0 = 0.4, 0.3
    S.rect(lx0, ly0, 2.9, 1.55, fill='FFFFFF', line=INK, lw=0.8)
    S.text(lx0 + 0.1, ly0 + 0.04, 1.5, 0.22, plain('legend'), size=9, align='l')
    entries = [((LAV, LAV_D, False), [('stream ', ''), ('x', 'i'), (' (melody) token', '')]),
               ((TEAL, TEAL_D, False), [('stream ', ''), ('y', 'i'), (' (chord) token', '')]),
               ((HARM, HARM_D, False), plain('harmonizer token')),
               (('FFFFFF', HARM_D, True), plain('masked harmonizer')),
               (('FFFFFF', INK, True), [('copied from LM', ''), ('s', 'sub i'), (', fine-tuned', '')])]
    for k, ((fill, line, dash), lab) in enumerate(entries):
        yy = ly0 + 0.34 + k * 0.23
        S.rect(lx0 + 0.12, yy + 0.02, 0.3, 0.16, fill=fill, line=line, lw=0.7, dash=bool(dash))
        S.text(lx0 + 0.48, yy - 0.03, 2.4, 0.24, lab, size=7.5, align='l')
    # pretrained model, brief
    px0, py0 = 0.4, 2.2
    S.rect(px0, py0, 2.9, 2.0, fill='FFFFFF', line=INK, lw=0.8)
    S.text(px0 + 0.1, py0 + 0.04, 2.7, 0.22, [('pretrained LM', ''), ('s', 'sub i'), (' (one stream)', '')], size=8, align='l')
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
    S.text(px0 + 0.1, py0 + 1.78, 2.7, 0.18, [('attention → ', ''), ('W', 'i'), ('s', 'sub i'), (', ', ''), ('W', 'i'), ('O', 'sup i'), ('s', 'sub i'), (';  FFN → every Exp', '')], size=6.5, color=INK_2, align='l')
    S.line(px0 + 2.72, py0 + 1.31, bx0 - 0.02, yAt + 0.33, color=INK_2, lw=0.7, dash=True, arrow=True)
    S.line(px0 + 2.72, py0 + 0.99, bx0 - 0.02, yE + 0.15, color=INK_2, lw=0.7, dash=True, arrow=True)
    # arrows to the block: attention -> attention bar, FFN -> experts

    # ================================================================ decoding (right column, stacked)
    dx0, dx1 = 12.15, 15.8
    S.rect(dx0, 0.3, dx1 - dx0, 6.95, fill='FFFFFF', line=INK, lw=0.8)
    S.text(dx0 + 0.1, 0.34, 3.5, 0.22, plain('decoding frame 4, alternating commit'), size=8.5, align='l')
    ts, tg = 0.32, 0.05

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
              [('harmonizers masked;', '')],
              [('x', 'i'), ('4', 'sub i'), (' from the stream-', ''), ('x', 'i'), (' next-frame head', '')]),
             (2.95, '2  condition the follower', ['qx', 'qm'], [sub('x', '4'), sub('q', 'y')], 'y',
              m('y', ('4', 'sub'), (' ∼ ', 'r'), 'p', ('(', 'r'), 'y', ('4', 'sub'), (' | ', 'r'), 'x', ('4', 'sub'), (', ', 'r'), 'x', ('≤3', 'sub'), (', ', 'r'), 'y', ('≤3', 'sub'), (')', 'r')),
              [('x', 'i'), ('4', 'sub i'), (' written into ', ''), ('q', 'i'), ('x', 'sub i'), (';', '')],
              [('q', 'i'), ('y', 'sub i'), (' reads it as partner, emits ', ''), ('y', 'i'), ('4', 'sub i')]),
             (5.15, '3  commit both, swap roles', ['x', 'y'], [sub('x', '4'), sub('y', '4')], None, None,
              [('x', 'i'), ('4', 'sub i'), (', ', ''), ('y', 'i'), ('4', 'sub i'), (' join the context;', '')],
              [('at frame 5, stream ', ''), ('y', 'i'), (' leads', '')])]
    for y0, title, qk, ql, out, formula, note1, note2 in steps:
        S.text(dx0 + 0.1, y0, 3.5, 0.2, [(title, 'b')], size=8, align='l')
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
            S.text(dx0 + 0.1, yr_ + ts + 0.04, 3.5, 0.2, formula, size=7.5, align='l')
            if out == 'y':
                S.line(xs[6] + ts / 2 - 0.02, yr_ - 0.08, xs[7] - ts / 2 + 0.02, yr_ - 0.08, color=HARM_D, lw=1.2, arrow=True)
        S.text(dx0 + 0.1, yr_ + ts + 0.3, 3.5, 0.18, note1, size=7, color=INK_2, align='l')
        S.text(dx0 + 0.1, yr_ + ts + 0.48, 3.5, 0.18, note2, size=7, color=INK_2, align='l')
    for it in S.items:                          # every label 1.5x: the slide is 15.5 in wide
        if 'size' in it:
            it['size'] = round(it['size'] * 1.4, 1)
        if it['k'] == 'rect' and it['line']:      # one outline everywhere: black, 0.9 pt; only the type differs
            it['lw'] = 0.9
            it['line'] = INK
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
