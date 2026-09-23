"""Panel c of the model figure: decoding one frame by ALTERNATING
COMMIT, in the style of fig_moe_panel.py (lavender = stream x, teal =
stream y, black outlines, Cambria). Layout follows the hand sketch of
2026-09-23; content changed to alternating commit the same day:

  row 1  t = 3, draft     x1 y1 x2 y2 -> Duet -> the leader's draft x*3,
                          sampled from the autoregressive head of the
                          content token x2 (a circle above it)
  row 2  t = 3, denoise   x*3 is written into its appended slot (dashed,
                          stream-coloured); the partner's slot stays
                          EMPTY, drawn shaded and dashed with y*3; one
                          forward reads out BOTH slots: x3 from the
                          leader's slot (x* -> x) and y3 from the empty
                          slot given x*3 (A3_CTC_DENOISE_LEADER=1)
  row 3  t = 4, draft     x3, y3 join the context and the roles swap:
                          the leader is now y, its draft y*4 from the
                          head of y3
The two appended slots are deliberately not named in the figure.

    python fig_decode_panel.py --out <path-without-extension>
writes <out>.pptx (editable) and <out>.png (preview).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_arch_pptx  # noqa: E402
fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 4.8, 5.7
from fig_arch_pptx import Spec, write_pptx_js, write_preview, m, plain, INK_2  # noqa: E402
from fig_style import FILL, COMP, INK, LW, AW, FS, BH  # noqa: E402

SHADE = 'EAEAEA'                   # an empty slot


def build():
    S = Spec()
    lw = LW
    tk, pitch = 0.42, 0.5             # token size and column pitch
    x0 = 1.0                          # first token column (left edge)
    bx0, bx1, bh = 0.9, 4.1, BH       # the model box, same width in every row
    rows_y = (0.44, 2.2, 3.96)        # top of each row
    yo, yb, yt = 0.0, 0.58, 1.14      # offsets within a row: outputs, block, tokens

    def col(i):
        return x0 + i * pitch + tk / 2

    def token(cx, y, s, lab, dash=False, shape='rect', fill=None):
        S.rect(cx - tk / 2, y, tk, tk, fill=FILL[s] if fill is None else fill, line=INK,
               lw=lw, dash=dash, runs=lab, size=FS, shape=shape)

    def sub(base, idx, star=False):
        return m(base, ('*', 'sup'), (idx, 'sub')) if star else m(base, (idx, 'sub'))

    def arrow(x1, y1, x2, y2):
        S.line(x1, y1, x2, y2, lw=AW, arrow=True)

    def block(y):
        S.rect(bx0, y, bx1 - bx0, bh, fill=COMP, line=INK, lw=lw,
               runs=[('Duet', ''), (', ', ''), ('L', 'i'), (' blocks', '')], size=FS)

    def row_label(y, frame, step):
        S.text(0.08, y + yb - 0.02, 0.85, 0.2, m(('t', 'i'), (' = ' + frame, 'r')), size=9.5,
               align='l')
        S.text(0.08, y + yb + 0.18, 0.85, 0.2, plain(step), size=8.5, align='l', color=INK_2)

    S.text(0.12, 0.05, 4.5, 0.28, [('c. Alternating-commit decoding', 'b')],
           size=10.5, align='l')

    # ---------------------------------------------------------- row 1: t = 3, draft
    y = rows_y[0]
    row_label(y, '3', 'draft')
    for i, (s, idx) in enumerate((('x', '1'), ('y', '1'), ('x', '2'), ('y', '2'))):
        token(col(i), y + yt, s, sub(s, idx))
    arrow(col(1.5), y + yt - 0.02, col(1.5), y + yb + bh + 0.02)
    block(y + yb)
    # the leader alone, from the autoregressive head of the token predicting x3
    arrow(col(2), y + yb - 0.02, col(2), y + yo + tk + 0.02)
    token(col(2), y + yo, 'x', sub('x', '3', star=True), shape='ellipse')
    S.text(col(2) + tk / 2 + 0.1, y + yo + 0.05, 1.8, 0.26,
           plain('draft of the leader,\nautoregressive head'), size=7.5, align='l', color=INK_2)

    # ---------------------------------------------------------- row 2: t = 3, refine
    y = rows_y[1]
    row_label(y, '3', 'denoise')
    for i, (s, idx) in enumerate((('x', '1'), ('y', '1'), ('x', '2'), ('y', '2'))):
        token(col(i), y + yt, s, sub(s, idx))
    # the two appended slots: the leader written into its own, the partner's empty
    token(col(4), y + yt, 'x', sub('x', '3', star=True), dash=True)
    token(col(5), y + yt, 'y', sub('y', '3', star=True), dash=True, fill=SHADE)
    arrow(col(2.5), y + yt - 0.02, col(2.5), y + yb + bh + 0.02)
    block(y + yb)
    # both slots are read out: x3 denoised from its draft, y3 given x*3
    for i, s_ in ((4, 'x'), (5, 'y')):
        arrow(col(i), y + yb - 0.02, col(i), y + yo + tk + 0.02)
        token(col(i), y + yo, s_, sub(s_, '3'))
    S.text(col(5) + tk / 2 + 0.1, y + yo + 0.05, 1.0, 0.26, plain('both slots,\ngiven x*'),
           size=7.5, align='l', color=INK_2)

    # ---------------------------------------------------------- row 3: t = 4, draft
    y = rows_y[2]
    row_label(y, '4', 'draft')
    for i, (s, idx) in enumerate((('x', '1'), ('y', '1'), ('x', '2'), ('y', '2'),
                                  ('x', '3'), ('y', '3'))):
        token(col(i), y + yt, s, sub(s, idx))
    arrow(col(2.5), y + yt - 0.02, col(2.5), y + yb + bh + 0.02)
    block(y + yb)
    # roles swap: stream y leads frame 4, from the head of y3
    arrow(col(5), y + yb - 0.02, col(5), y + yo + tk + 0.02)
    token(col(5), y + yo, 'y', sub('y', '4', star=True), shape='ellipse')
    S.text(col(5) + tk / 2 + 0.1, y + yo + 0.05, 1.0, 0.26, plain('roles swap\n…'),
           size=7.5, align='l', color=INK_2)
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
