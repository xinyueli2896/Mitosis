"""Panel c of the model figure: decoding one frame, draft then refine,
in the style of fig_moe_panel.py (lavender = stream x, teal = stream y,
black outlines, Cambria). Content follows the hand sketch of 2026-09-23
and Sec. 3.5 of the method (two passes per frame):

  row 1  t = 3, draft    x1 y1 x2 y2 -> Duet -> drafts x~3, y~3 from the
                         autoregressive heads of the two content tokens
                         predicting frame 3 (drawn as circles above them)
  row 2  t = 3, refine   the drafts are written into the two query tokens
                         (dashed boxes, appended after the sequence); one
                         forward emits the committed x3, y3 together
  row 3  t = 4, draft    x3, y3 join the context; the next frame's drafts
                         come from the content tokens x3, y3

    python fig_decode_panel.py --out <path-without-extension>
writes <out>.pptx (editable) and <out>.png (preview).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_arch_pptx  # noqa: E402
fig_arch_pptx.SLIDE_W, fig_arch_pptx.SLIDE_H = 4.8, 5.3
from fig_arch_pptx import Spec, write_pptx_js, write_preview, m, plain, INK, INK_2  # noqa: E402

LAV, TEAL = 'B8BBEC', 'B7DBD8'
FILL = {'x': LAV, 'y': TEAL}


def build():
    S = Spec()
    lw = 0.9
    tk, pitch = 0.36, 0.45            # token size and column pitch
    x0 = 1.08                         # first token column (left edge)
    bx0, bx1, bh = 0.98, 3.86, 0.34   # the model box, same width in every row
    rows_y = (0.44, 2.06, 3.68)        # top of each row
    yo, yb, yt = 0.0, 0.52, 1.02      # offsets within a row: outputs, block, tokens

    def col(i):
        return x0 + i * pitch + tk / 2

    def token(cx, y, s, lab, dash=False, shape='rect', fill=None):
        S.rect(cx - tk / 2, y, tk, tk, fill=FILL[s] if fill is None else fill, line=INK,
               lw=lw, dash=dash, runs=lab, size=9, shape=shape)

    def sub(base, idx, tilde=False):
        b = base + ('̃' if tilde else '')
        return m(b, (idx, 'sub'))

    def arrow(x1, y1, x2, y2):
        S.line(x1, y1, x2, y2, lw=0.8, arrow=True)

    def block(y):
        S.rect(bx0, y, bx1 - bx0, bh, fill='FFFFFF', line=INK, lw=lw,
               runs=[('Duet', ''), (', ', ''), ('L', 'i'), (' blocks', '')], size=9.5)

    def row_label(y, frame, step):
        S.text(0.08, y + yb - 0.02, 0.85, 0.2, m(('t', 'i'), (' = ' + frame, 'r')), size=9.5,
               align='l')
        S.text(0.08, y + yb + 0.18, 0.85, 0.2, plain(step), size=8.5, align='l', color=INK_2)

    S.text(0.12, 0.05, 4.5, 0.28, [('c. Decoding a frame: draft, then refine', 'b')],
           size=10.5, align='l')

    # ---------------------------------------------------------- row 1: t = 3, draft
    y = rows_y[0]
    row_label(y, '3', 'draft')
    for i, (s, idx) in enumerate((('x', '1'), ('y', '1'), ('x', '2'), ('y', '2'))):
        token(col(i), y + yt, s, sub(s, idx))
    arrow(col(1.5), y + yt - 0.02, col(1.5), y + yb + bh + 0.02)
    block(y + yb)
    for i, s in ((2, 'x'), (3, 'y')):          # drafts above the tokens that predict frame 3
        arrow(col(i), y + yb - 0.02, col(i), y + yo + tk + 0.02)
        token(col(i), y + yo, s, sub(s, '3', tilde=True), shape='ellipse')
    S.text(col(3) + tk / 2 + 0.1, y + yo + 0.05, 1.6, 0.26,
           plain('drafts, from the\nautoregressive heads'), size=7.5, align='l', color=INK_2)

    # ---------------------------------------------------------- row 2: t = 3, refine
    y = rows_y[1]
    row_label(y, '3', 'refine')
    for i, (s, idx) in enumerate((('x', '1'), ('y', '1'), ('x', '2'), ('y', '2'))):
        token(col(i), y + yt, s, sub(s, idx))
    for i, s in ((4, 'x'), (5, 'y')):          # query tokens holding the drafts
        token(col(i), y + yt, s, sub(s, '3', tilde=True), dash=True, fill='FFFFFF')
    S.text(col(4) - 0.5, y + yt + tk + 0.03, 1.9, 0.18, plain('query tokens'), size=7.5,
           align='c', color=INK_2)
    arrow(col(2.5), y + yt - 0.02, col(2.5), y + yb + bh + 0.02)
    block(y + yb)
    for i, s in ((4, 'x'), (5, 'y')):          # committed frame, read out together
        arrow(col(i), y + yb - 0.02, col(i), y + yo + tk + 0.02)
        token(col(i), y + yo, s, sub(s, '3'))
    S.text(col(5) + tk / 2 + 0.1, y + yo + 0.05, 1.0, 0.26, plain('committed\ntogether'),
           size=7.5, align='l', color=INK_2)

    # ---------------------------------------------------------- row 3: t = 4, draft
    y = rows_y[2]
    row_label(y, '4', 'draft')
    for i, (s, idx) in enumerate((('x', '1'), ('y', '1'), ('x', '2'), ('y', '2'),
                                  ('x', '3'), ('y', '3'))):
        token(col(i), y + yt, s, sub(s, idx))
    arrow(col(2.5), y + yt - 0.02, col(2.5), y + yb + bh + 0.02)
    block(y + yb)
    for i, s in ((4, 'x'), (5, 'y')):
        arrow(col(i), y + yb - 0.02, col(i), y + yo + tk + 0.02)
        token(col(i), y + yo, s, sub(s, '4', tilde=True), shape='ellipse')
    S.text(col(5) + tk / 2 + 0.1, y + yo + 0.05, 1.0, 0.26, plain('…'), size=11,
           align='l', color=INK_2)
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
