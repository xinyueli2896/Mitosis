"""Panel a of the model figure: the dual-stream attention sub-layer,
reproduced from the hand-made drawing of 2026-09-23 in the style of
fig_moe_panel.py / fig_decode_panel.py (lavender = stream x, teal =
stream y, black outlines, Cambria).

Bottom up, per stream: the interleaved tokens gather into the stream's
bank h^l -> Linear QKV -> Q, K, V -> Self Attn (own Q, K, V) and Cross
(own Q; the OTHER stream's K, V, drawn in that stream's colour) -> the
cross readout through a gate -> (+) with the self readout -> Output
projection -> o^l -> Add & Norm, with the skip from h^l drawn around the
outside. The drawing's content is kept as is; only the right-hand o box
is relabelled o_y (the drawing said o_x twice).

    python fig_attn_panel.py --out <path-without-extension>
writes <out>.pptx (editable) and <out>.png (preview).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_arch_pptx  # noqa: E402
fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 4.8, 6.1
from fig_arch_pptx import Spec, write_pptx_js, write_preview, m, plain, INK  # noqa: E402

LAV, TEAL = 'B8BBEC', 'B7DBD8'
COMP = 'DEDEDE'                          # every model component
LAV_L, TEAL_L = '8F93D9', '6FB5AE'      # the Q/K/V routing lines
FILL = {'x': LAV, 'y': TEAL}
LINE = {'x': LAV_L, 'y': TEAL_L}


def build():
    S = Spec()
    lw = 0.9
    cx = {'x': 1.2, 'y': 3.6}                # lane centres
    bw, bh = 1.36, 0.38                       # wide boxes
    # rows (top of box), y down
    yAN, yO, yP, yG, yA, yQ, yL, yH, yT = 0.46, 0.94, 1.42, 2.02, 2.38, 3.7, 4.28, 4.85, 5.6
    tk, sq = 0.42, 0.3                       # token size, Q/K/V box size
    cw = 0.38                                # Cross box width

    def mark(x, y, w, kind='pre'):
        """corner sign: snowflake->fire = initialised from the pretrained
        model and fine-tuned; fire = new component, trained from scratch"""
        glyph = '\u2744\u2192\U0001F525' if kind == 'pre' else '\U0001F525'
        # a white sticker straddling the top-right corner, sized to the glyph, so
        # the box outline never shows through the arrow
        bw_ = 0.5 if kind == 'pre' else 0.24
        S.rect(x + w - bw_ + 0.06, y - 0.12, bw_, 0.24, fill='FFFFFF', line=None,
               runs=plain(glyph), size=9.5, align='c', z=6)

    def wide(s, y, runs, fill=COMP, size=9.5, pre=False):
        S.rect(cx[s] - bw / 2, y, bw, bh, fill=fill, line=INK, lw=lw, runs=runs, size=size)
        if pre:
            mark(cx[s] - bw / 2, y, bw)

    def arrow(x1, y1, x2, y2, color=INK, lw_=0.8):
        S.line(x1, y1, x2, y2, lw=lw_, arrow=True, color=color)

    def bank(base, sup, s):
        return [(base, 'b i'), (sup, 'sup i'), (s + ',1:T−1', 'sub i')]

    def plus(px, py, r=0.1):
        S.rect(px - r, py - r, 2 * r, 2 * r, fill='FFFFFF', line=INK, lw=lw, shape='ellipse', z=4)
        S.line(px - 0.05, py, px + 0.05, py, lw=lw, z=5)
        S.line(px, py - 0.05, px, py + 0.05, lw=lw, z=5)

    S.text(0.12, 0.05, 3.5, 0.28, [('a. Dual-stream attention', 'b')], size=10.5, align='l')

    # ------------------------------------------------------------ tokens -> banks
    tok = [('x', '1'), ('y', '1'), ('x', None), ('y', None), ('x', 'T\u22121'), ('y', 'T\u22121')]
    tx = [0.65 + i * 0.7 for i in range(6)]
    for (s, idx), x in zip(tok, tx):
        lab = plain('\u2026') if idx is None else m(s, (idx, 'sub'))
        S.rect(x - tk / 2, yT, tk, tk, fill=FILL[s], line=INK, lw=lw, runs=lab, size=9.5)
        S.line(x, yT - 0.02, cx[s], yH + bh + 0.02, lw=0.7)
    # projections: geometry per stream
    qkv = {s: [cx[s] - 0.45, cx[s], cx[s] + 0.45] for s in 'xy'}
    side = {'x': -1, 'y': 1}
    self_box = {s: (cx[s] - bw / 2, cx[s] + bw / 2) for s in 'xy'}
    cross_box = {'x': (cx['x'] + bw / 2 + 0.07, cx['x'] + bw / 2 + 0.07 + cw),
                 'y': (cx['y'] - bw / 2 - 0.07 - cw, cx['y'] - bw / 2 - 0.07)}
    other = {'x': 'y', 'y': 'x'}

    for s in 'xy':
        c = cx[s]
        wide(s, yH, bank('h', 'l', s), fill=FILL[s])
        arrow(c, yH - 0.02, c, yL + bh + 0.02)
        wide(s, yL, plain('Linear QKV'), pre=True)
        for x, lab in zip(qkv[s], 'QKV'):
            arrow(c, yL - 0.02, x, yQ + sq + 0.02)
            S.rect(x - sq / 2, yQ, sq, sq, fill='FFFFFF', line=INK, lw=lw, runs=m(lab), size=9.5)
        # attention boxes
        S.rect(self_box[s][0], yA, bw, bh, fill=COMP, line=INK, lw=lw,
               runs=plain('Self Attn'), size=9.5)
        mark(self_box[s][0], yA, bw)                 # the attention: pretrained, fine-tuned
        S.rect(cross_box[s][0], yA, cw, bh, fill=COMP, line=INK, lw=lw,
               runs=plain('Cross'), size=9.5)
        # gate on the cross readout, summed with the self readout
        ccx = (cross_box[s][0] + cross_box[s][1]) / 2
        gx0 = c + side[s] * -0.28 if False else (c + 0.3 if s == 'x' else c - 0.3 - 0.36)
        S.rect(gx0, yG - 0.02, 0.36, 0.3, fill=COMP, line=INK, lw=lw, runs=plain('gate'),
               size=7.5)
        mark(gx0, yG - 0.02, 0.36, kind='new')
        gy = yG + 0.13
        S.line(ccx, yA - 0.02, ccx, gy, lw=0.8)
        if s == 'x':
            S.line(ccx, gy, gx0 + 0.36 + 0.02, gy, lw=0.8)
            arrow(gx0 - 0.02, gy, c + 0.12, gy)
        else:
            S.line(ccx, gy, gx0 - 0.02, gy, lw=0.8)
            arrow(gx0 + 0.36 + 0.02, gy, c - 0.12, gy)
        plus(c, gy)
        arrow(c, yA - 0.02, c, gy + 0.12)
        # output projection, o, add & norm, skip
        arrow(c, gy - 0.12, c, yP + bh + 0.02)
        wide(s, yP, plain('Output projection'), size=8.5)
        arrow(c, yP - 0.02, c, yO + bh + 0.02)
        wide(s, yO, bank('o', 'l', s), fill=FILL[s])
        arrow(c, yO - 0.02, c, yAN + bh + 0.02)
        wide(s, yAN, plain('Add & Norm'))
        sx = c + side[s] * (bw / 2 + 0.3)
        S.line(c + side[s] * bw / 2, yH + bh / 2, sx, yH + bh / 2, lw=0.8)
        S.line(sx, yH + bh / 2, sx, yAN + bh / 2, lw=0.8)
        arrow(sx, yAN + bh / 2, c + side[s] * (bw / 2 + 0.02), yAN + bh / 2)

    # ------------------------------------------------------------ Q/K/V routing
    # Self Attn: own Q, K, V straight up. Cross: own Q, the other stream's
    # K and V, in that stream's colour. Every routed line has its own
    # elbow height and its own column, so no two share a segment; the
    # remaining crossings are perpendicular.
    ya = yA + bh + 0.02                      # arrow tips at the attention boxes' bottom
    rl = 0.9
    for s in 'xy':
        for x in qkv[s]:
            arrow(x, yQ - 0.02, x, ya, color=LINE[s], lw_=rl)
    # elbow heights below yQ; all must stay under yQ - ya (= 0.94 here)
    lvl = {('x', 'Q'): 0.14, ('y', 'Q'): 0.14,          # own Q: short, lowest
           ('y', 'K'): 0.36, ('y', 'V'): 0.50,          # y's K, V -> Cross_x
           ('x', 'K'): 0.64, ('x', 'V'): 0.78}          # x's K, V -> Cross_y
    assert max(lvl.values()) < yQ - ya - 0.1
    for s in 'xy':                           # cross box of stream s
        o = other[s]
        x0c, x1c = cross_box[s]
        targets = [x0c + cw * 0.2, x0c + cw * 0.5, x0c + cw * 0.8]
        d = 0.07 if s == 'x' else -0.07     # leave the box beside the straight arrow
        # own Q
        qx = qkv[s][0] + d
        yr = yQ - lvl[(s, 'Q')]
        S.line(qx, yQ - 0.02, qx, yr, lw=rl, color=LINE[s])
        S.line(qx, yr, targets[0], yr, lw=rl, color=LINE[s])
        arrow(targets[0], yr, targets[0], ya, color=LINE[s], lw_=rl)
        # the other stream's K and V
        for k, key in enumerate('KV'):
            xs = qkv[o][k + 1] - d
            yr2 = yQ - lvl[(o, key)]
            S.line(xs, yQ - 0.02, xs, yr2, lw=rl, color=LINE[o])
            S.line(xs, yr2, targets[k + 1], yr2, lw=rl, color=LINE[o])
            arrow(targets[k + 1], yr2, targets[k + 1], ya, color=LINE[o], lw_=rl)
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
