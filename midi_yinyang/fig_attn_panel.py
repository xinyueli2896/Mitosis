"""Panel a of the model figure: the dual-stream attention sub-layer, in
the style of the hand sketch of 2026-09-23 (fig_style.py): blue x,
salmon y, grey components, the x cross path green and the y cross path
yellow, Q and K&V as two boxes, heavier outlines.

Content is unchanged from the earlier version. Bottom up, per stream:
the interleaved tokens gather into the stream's bank h^l -> Linear QKV
-> Q and K&V -> Self Attn (own Q, K&V, grey lines) and Cross Attn (own
Q with the OTHER stream's K&V, in the cross path's colour) -> the cross
readout through a gate -> (+) with the self readout -> Output proj ->
Add & Norm (skip from h^l around the outside) -> h-tilde^(l). Badges: snowflake
-> fire on Linear QKV and Output proj (pretrained, fine-tuned), fire on
the gates (new).

    python fig_attn_panel.py --out <path-without-extension>
writes <out>.pptx (editable) and <out>.png (preview).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_arch_pptx  # noqa: E402
fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 4.8, 6.4
from fig_arch_pptx import Spec, write_pptx_js, write_preview, m, plain, badge  # noqa: E402
from fig_style import FILL, LINE, COMP, CROSS, GREY_L, INK, LW, AW, FS, BH  # noqa: E402


def build(stretch=0.0):
    """stretch: extra inches added to every gap between the nine rows,
    from the token row upward (the combiner uses it to make the panels
    the same height)"""
    S = Spec()
    # geometry measured off the sketch (4.8 in panel): banks 1.58 wide with a
    # 0.3 gap between lanes, the attention pair wider than the bank, boxes
    # 0.4 tall, rows about 0.55 apart
    cx = {'x': 1.47, 'y': 3.33}              # lane centres
    bw, bh = 1.58, BH                        # wide boxes
    # rows (top of box), y down
    yAN, yO, yP, yG, yA, yQ, yL, yH, yT = (
        v - k * stretch for k, v in
        zip(range(8, -1, -1), (0.4, 0.98, 1.58, 2.22, 2.72, 3.6, 4.28, 4.92, 5.8)))
    tk = 0.44                                # token size
    sw, cw = 0.82, 0.82                      # Self Attn and Cross Attn widths
    side = {'x': -1, 'y': 1}
    other = {'x': 'y', 'y': 'x'}

    def mark(x, y, w, kind='pre'):
        """corner sign as pictures: snowflake -> fire = initialised from the
        pretrained model and fine-tuned; fire = new, trained from scratch.
        Inside the block, top-right corner; the label stays centred."""
        badge(S, x + w - 0.035, y + 0.02, kind, h=0.11)

    def box(x, y, w, h, runs, fill=COMP, size=FS, dash=False, z=1):
        S.rect(x, y, w, h, fill=fill, line=INK, lw=LW, dash=dash, runs=runs, size=size, z=z)

    def wide(s, y, runs, fill=COMP, size=FS, pre=False):
        box(cx[s] - bw / 2, y, bw, bh, runs, fill=fill, size=size)
        if pre:
            mark(cx[s] - bw / 2, y, bw)

    def arrow(x1, y1, x2, y2, color=INK, lw_=AW):
        S.line(x1, y1, x2, y2, lw=lw_, arrow=True, color=color)

    def bank(base, sup, s):
        return [(base, 'b i'), (sup, 'sup i'), (s + ',1:T−1', 'sub i')]

    def plus(px, py, r=0.1):
        S.rect(px - r, py - r, 2 * r, 2 * r, fill='FFFFFF', line=INK, lw=LW, shape='ellipse', z=4)
        S.line(px - 0.05, py, px + 0.05, py, lw=LW, z=5)
        S.line(px, py - 0.05, px, py + 0.05, lw=LW, z=5)

    S.text(0.12, 0.05, 3.5, 0.28, [('a. Dual-stream attention', 'b')], size=10.5, align='l')

    # ------------------------------------------------------------ tokens -> banks
    # the block's input: many small unlabelled hidden-state boxes, blue and
    # salmon alternating, an ellipsis in the middle standing for the rest
    tkx = 0.3                                   # small box
    pitch = 0.36
    n_left, n_right = 6, 2                      # boxes before and after the ellipsis
    gap_mid = 0.45
    total = (n_left + n_right) * pitch - (pitch - tkx) + gap_mid
    x_start = (cx['x'] + cx['y']) / 2 - total / 2
    xs_all = []
    for k in range(n_left + n_right):
        x = x_start + k * pitch + (gap_mid if k >= n_left else 0)
        st = 'xy'[k % 2]
        yT2 = yT + (tk - tkx) / 2
        box(x, yT2, tkx, tkx, None, fill=FILL[st], z=2)
        xs_all.append((st, x))
    xm = x_start + n_left * pitch - (pitch - tkx) / 2 + gap_mid / 2
    S.text(xm - 0.2, yT + (tk - tkx) / 2 - 0.02, 0.4, tkx + 0.04, plain('\u2026'), size=FS, align='c')
    # translucent wedges (sketch): each stream's boxes gather into its bank
    for st in 'xy':
        xs = [x for t_, x in xs_all if t_ == st]
        S.poly([(cx[st] - bw / 2, yH + bh), (cx[st] + bw / 2, yH + bh),
                (max(xs) + tkx, yT + (tk - tkx) / 2 + 0.02), (min(xs), yT + (tk - tkx) / 2 + 0.02)],
               fill=LINE[st], alpha=0.45, z=0)

    # geometry of the projection boxes and attention boxes, per stream
    # Q on the OUTER side, K & V on the inner side, mirrored between streams
    qbox = {'x': (cx['x'] - 0.79, 0.38), 'y': (cx['y'] + 0.79 - 0.38, 0.38)}     # (x0, w)
    kvbox = {'x': (cx['x'] - 0.25, 0.84), 'y': (cx['y'] + 0.25 - 0.84, 0.84)}
    self_box = {'x': (cx['x'] - 0.95, sw), 'y': (cx['y'] + 0.95 - sw, sw)}      # outer, past the bank edge
    cross_box = {'x': (cx['x'] + 0.07, cw), 'y': (cx['y'] - 0.07 - cw, cw)}      # inner

    def centre(b):
        return b[0] + b[1] / 2

    for s in 'xy':
        c = cx[s]
        cfill, cline = CROSS[s]
        wide(s, yH, bank('h', '(l)', s), fill=FILL[s])
        arrow(c, yH - 0.02, c, yL + bh + 0.02)
        wide(s, yL, plain('Linear QKV'), pre=True)
        # Q and K & V (three arrows out of Linear QKV, as in the sketch)
        qx0, qw = qbox[s]; kx0, kw = kvbox[s]
        box(qx0, yQ, qw, bh, m('Q'), fill=FILL[s])
        box(kx0, yQ, kw, bh, [('K', 'i'), (' & ', ''), ('V', 'i')], fill=FILL[s])
        for x in (centre(qbox[s]), kx0 + kw * 0.3, kx0 + kw * 0.7):   # three straight verticals
            arrow(x, yL - 0.02, x, yQ + bh + 0.02)
        # attention boxes
        sx0, _ = self_box[s]; cx0, _ = cross_box[s]
        box(sx0, yA, sw, bh, plain('Self Attn'), fill=FILL[s])
        box(cx0, yA, cw, bh, plain('Cross Attn'), fill=cfill)
        scx, ccx = centre(self_box[s]), centre(cross_box[s])
        # gate above Cross Attn, in the cross path's colour; (+) at the lane centre
        gy = yG + 0.17
        gw = 0.56
        box(ccx - gw / 2, yG - 0.02, gw, 0.38, plain('Gate'), fill=cfill, size=FS - 0.5)
        mark(ccx - gw / 2, yG - 0.02, gw, kind='new')
        arrow(ccx, yA - 0.02, ccx, yG + 0.38)
        plus(c, gy)
        arrow(ccx - side[s] * gw / 2 * -1 if False else (ccx - gw / 2 - 0.02 if s == 'x' else ccx + gw / 2 + 0.02),
              gy, c + (0.12 if s == 'x' else -0.12), gy)
        # Self Attn: up, then across into (+)
        S.line(scx, yA - 0.02, scx, gy, lw=AW)
        arrow(scx, gy, c + (-0.12 if s == 'x' else 0.12), gy)
        # output projection -> add & norm (with the skip from h) -> h-tilde
        arrow(c, gy - 0.12, c, yP + bh + 0.02)
        wide(s, yP, plain('Output proj'), pre=True)
        arrow(c, yP - 0.02, c, yO + bh + 0.02)
        wide(s, yO, plain('Add & Norm'))
        arrow(c, yO - 0.02, c, yAN + bh + 0.02)
        wide(s, yAN, bank('h\u0303', '(l)', s), fill=FILL[s])
        skx = c + side[s] * (0.95 + 0.2)
        S.line(c + side[s] * bw / 2, yH + bh / 2, skx, yH + bh / 2, lw=AW)
        S.line(skx, yH + bh / 2, skx, yO + bh / 2, lw=AW)
        arrow(skx, yO + bh / 2, c + side[s] * (bw / 2 + 0.02), yO + bh / 2)

    # ------------------------------------------------------------ routing
    # Curved arrows, as in the sketch. Self Attn: own Q and K&V, grey.
    # Cross Attn: own Q and the OTHER stream's K&V, in the cross path's
    # colour. Each arrow leaves its box top vertically and arrives at the
    # attention box bottom vertically.
    ya = yA + bh + 0.02
    for s in 'xy':
        sd = side[s]                       # +1 outward for y, -1 outward for x
        scx, ccx = centre(self_box[s]), centre(cross_box[s])
        # self: own Q and K&V, straight grey arrows into the Self Attn box (sketch)
        S.line(centre(qbox[s]) + sd * 0.12, yQ - 0.02, scx + sd * 0.2, ya,
               color=GREY_L, lw=AW, arrow=True)
        S.line(centre(kvbox[s]) + sd * 0.12, yQ - 0.02, scx - sd * 0.2, ya,
               color=GREY_L, lw=AW, arrow=True)
    for s in 'xy':
        sd, o = side[s], other[s]
        col = CROSS[s][1]
        ccx = centre(cross_box[s])
        # cross: own Q from its inner half; the OTHER stream's K&V from its inner half
        S.curve(centre(qbox[s]) - sd * 0.12, yQ - 0.02, ccx + sd * 0.2, ya,
                color=col, lw=AW, arrow=True, z=3)
        S.curve(centre(kvbox[o]) + sd * 0.12, yQ - 0.02, ccx - sd * 0.2, ya,
                color=col, lw=AW, arrow=True, z=3)
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
