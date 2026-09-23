"""Panel b of the model figure: the per-stream expert routing sub-layer,
drawn in the vocabulary and colours of the hand-drawn attention panel
(panel a): lavender = stream x, teal = stream y, grey = shared, black
outlines, Cambria.

Bottom up, per stream: h-tilde (the attention sub-layer's Add & Norm
output) -> top-2 router of that stream -> two of the FOUR SHARED experts
(no expert belongs to a stream; the two routers' arrow sets cross) ->
weighted sum with the renormalised top-2 weights pi -> Add & Norm with
a skip from h-tilde -> h^{l+1}, the next block's input. Matches Eq.
permod_router and the prose of method_per_part_moe.tex: E = 4, k = 2,
post-LN.

    python fig_moe_panel.py --out <path-without-extension>
writes <out>.pptx (editable) and <out>.png (preview).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_arch_pptx  # noqa: E402
fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 4.8, 5.55
from fig_arch_pptx import Spec, write_pptx_js, write_preview, m, plain, INK, INK_2  # noqa: E402

LAV, LAV_D = 'B8BBEC', '8F93D9'
TEAL, TEAL_D = 'B7DBD8', '7CBDB7'
GREY, LIGHT = 'DEDEDE', 'F1F1F1'
FILL = {'x': LAV, 'y': TEAL}
HT = 'h̃'                       # h with combining tilde


def build():
    S = Spec()
    lw = 0.9
    # lanes and rows (inches, y down)
    cx, cy = 1.35, 3.45                # lane centres, stream x / stream y
    bw, bh = 1.45, 0.32                # stream boxes
    yTok = 0.4                          # the output tokens, x2 y2 ... xT yT
    yOut, yAN, ySum, yExp, yRt, yIn = 1.48, 2.1, 2.76, 3.4, 4.18, 4.96
    ex = [0.85, 1.85, 2.95, 3.95]      # expert centres
    ew, eh = 0.86, 0.34

    def hbox(c, y, fill, base, sup, s):
        runs = [(base, 'b i'), (sup, 'sup i'), (s + ',1:T−1', 'sub i')]
        S.rect(c - bw / 2, y, bw, bh, fill=fill, line=INK, lw=lw, runs=runs, size=9.5)

    def plus(px, py, r=0.1):
        S.rect(px - r, py - r, 2 * r, 2 * r, fill='FFFFFF', line=INK, lw=lw, shape='ellipse', z=4)
        S.line(px - 0.05, py, px + 0.05, py, lw=lw, z=5)
        S.line(px, py - 0.05, px, py + 0.05, lw=lw, z=5)

    def arrow(x1, y1, x2, y2, **kw):
        S.line(x1, y1, x2, y2, lw=0.8, arrow=True, **kw)

    def mark(x, y, w, kind='pre'):
        """corner sign: snowflake->fire = initialised from the pretrained
        model and fine-tuned; fire = new component, trained from scratch"""
        glyph = '\u2744\u2192\U0001F525' if kind == 'pre' else '\U0001F525'
        S.text(x + w - 0.6, y - 0.12, 0.66, 0.24, plain(glyph), size=9.5, align='r', z=6)   # badge straddling the top-right corner

    S.text(0.12, 0.05, 3.5, 0.28, [('b. Per-stream expert routing', 'b')], size=10.5, align='l')

    # shared expert pool: one grey band, four experts, no stream subscripts
    px0, py0, pw = ex[0] - ew / 2 - 0.12, yExp - 0.1, ex[-1] - ex[0] + ew + 0.24
    S.rect(px0, py0, pw, eh + 0.2, fill=LIGHT, line=None, z=0)
    mark(px0, py0, pw)                    # the expert pool: replicated from the pretrained FFN
    S.text(px0 + pw - 0.75 - 1.5, py0 - 0.17, 1.5, 0.16, plain('shared pool, E = 4'),
           size=7, align='r', color=INK_2)
    for i, c in enumerate(ex):
        S.rect(c - ew / 2, yExp, ew, eh, fill=GREY, line=INK, lw=lw,
               runs=[('MLP', ''), ('xxyy'[i], 'sub i')], size=9.5)

    chosen = {'x': (0, 1), 'y': (1, 3)}      # top-2 per stream; the second MLP_x serves both
    for s, c, fill, dark in (('x', cx, LAV, LAV_D), ('y', cy, TEAL, TEAL_D)):
        # bottom: the attention sub-layer's output
        hbox(c, yIn, fill, HT, 'l', s)
        # router
        S.rect(c - bw / 2, yRt, bw, bh, fill=GREY, line=INK, lw=lw,
               runs=[('Top-2 Router', ''), (s, 'sub i')], size=9.5)
        mark(c - bw / 2, yRt, bw, kind='new')
        arrow(c, yIn - 0.02, c, yRt + bh + 0.02)
        # router -> its two experts (the two streams' arrows cross)
        for i in chosen[s]:
            arrow(c, yRt - 0.02, ex[i], yExp + eh + 0.02)
        # experts -> weighted sum, labelled with the renormalised weights
        plus(c, ySum)
        for k, i in enumerate(chosen[s]):
            arrow(ex[i], yExp - 0.02, c + (-0.07 if k == 0 else 0.07), ySum + 0.12)
        # add & norm, with the skip from h-tilde around the sub-layer
        S.rect(c - bw / 2, yAN, bw, bh, fill=GREY, line=INK, lw=lw,
               runs=plain('Add & Norm'), size=9.5)
        arrow(c, ySum - 0.12, c, yAN + bh + 0.02)
        side = -1 if s == 'x' else 1
        sx = c + side * (bw / 2 + 0.28)
        S.line(c + side * bw / 2, yIn + bh / 2, sx, yIn + bh / 2, lw=0.8)
        S.line(sx, yIn + bh / 2, sx, yAN + bh / 2, lw=0.8)
        arrow(sx, yAN + bh / 2, c + side * (bw / 2 + 0.02), yAN + bh / 2)
        # output: the next block's input
        hbox(c, yOut, fill, 'h', 'l+1', s)
        arrow(c, yAN - 0.02, c, yOut + bh + 0.02)
    # the two banks turn into the output tokens, one frame ahead of the input
    tk = 0.42
    tok = [('x', '2'), ('y', '2'), ('x', None), ('y', None), ('x', 'T'), ('y', 'T')]
    tx = [0.65 + i * 0.7 for i in range(6)]
    for (s, idx), x in zip(tok, tx):
        lab = plain('\u2026') if idx is None else m(s, (idx, 'sub'))
        S.rect(x - tk / 2, yTok, tk, tk, fill=FILL[s], line=INK, lw=lw, runs=lab, size=9.5)
        S.line({'x': cx, 'y': cy}[s], yOut - 0.02, x, yTok + tk + 0.02, lw=0.7)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='path without extension')
    args = ap.parse_args()
    S = build()
    write_pptx_js(S, args.out + '.pptx'); print('wrote', args.out + '.pptx')
    write_preview(S, args.out + '.png'); print('wrote', args.out + '.png')


if __name__ == '__main__':
    main()
