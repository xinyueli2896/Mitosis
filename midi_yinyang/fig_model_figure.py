"""The model figure: panels a, b, c side by side, bottom-aligned, with
the captions under each panel:
  (a) Dual-stream Attention   (b) Dual-stream MoE   (c) Dual-stream Decoding Process
Each panel is built by its own script (fig_attn_panel, fig_moe_panel,
fig_decode_panel); this one drops their top titles, offsets the items
and adds the captions. Edit a panel in its script; rebuild here.

    python fig_model_figure.py --out <path-without-extension>
writes <out>.pptx (editable) and <out>.png (preview).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_arch_pptx  # noqa: E402
import fig_attn_panel, fig_moe_panel, fig_decode_panel  # noqa: E402,E401
from fig_arch_pptx import Spec, write_pptx_js, write_preview, plain, INK, INK_2  # noqa: E402

GAP = 0.3
PW = 4.8
PANELS = [(fig_attn_panel, '(a) Dual-stream Attention'),
          (fig_moe_panel, '(b) Dual-stream MoE'),
          (fig_decode_panel, '(c) Dual-stream Decoding Process')]
CAP_H = 0.32


def bounds(items):
    """(xmin, xmax, ymin, ymax) of the drawn content"""
    xs, ys = [], []
    for it in items:
        if it['k'] == 'poly':
            xs += [p[0] for p in it['points']]; ys += [p[1] for p in it['points']]
        elif it['k'] in ('line', 'curve'):
            xs += [it['x1'], it['x2']]; ys += [it['y1'], it['y2']]
        elif it['k'] == 'text':
            # text boxes are wider than their glyphs: estimate the glyph run
            # from the longest line and anchor it at the aligned edge
            txt = ''.join(t for t, _ in it['runs'])
            est = max(len(ln) for ln in txt.split('\n')) * it['size'] * 0.5 / 72
            a = {'l': it['x'], 'r': it['x'] + it['w'] - est,
                 'c': it['x'] + it['w'] / 2 - est / 2}[it['align']]
            xs += [a, a + est]; ys += [it['y'], it['y'] + it['h']]
        else:
            xs += [it['x'], it['x'] + it['w']]; ys += [it['y'], it['y'] + it['h']]
    return min(xs), max(xs), min(ys), max(ys)


def shifted(items, dx, dy):
    out = []
    for it in items:
        it = dict(it)
        if it['k'] == 'poly':
            it['points'] = [(px + dx, py + dy) for px, py in it['points']]
        elif it['k'] in ('line', 'curve'):
            it['x1'] += dx; it['x2'] += dx; it['y1'] += dy; it['y2'] += dy
        else:
            it['x'] += dx; it['y'] += dy
        out.append(it)
    return out


def build():
    parts = []
    for mod, cap in PANELS:
        items = mod.build().items
        # drop the panel's own top title ("a. ...", "b. ...", "c. ...")
        items = [it for it in items if not (it['k'] == 'text' and it['runs'] and
                                            it['runs'][0][0][:3] in ('a. ', 'b. ', 'c. '))]
        parts.append((items, cap))
    boxes = [bounds(items) for items, _ in parts]
    tops = [b[2] for b in boxes]; bottoms = [b[3] for b in boxes]
    H = max(b - t for b, t in zip(bottoms, tops))      # tallest panel's content height
    top_margin = 0.15
    # panels keep their fixed pitch (PW + GAP); the slide is then cut at
    # panel (a)'s left content edge and panel (c)'s right content edge
    left = 0 * (PW + GAP) + boxes[0][0]
    right = (len(parts) - 1) * (PW + GAP) + boxes[-1][1]
    fig_arch_pptx.SLIDE_W = right - left
    fig_arch_pptx.SLIDE_H = top_margin + H + 0.12 + CAP_H + 0.1

    S = Spec()
    dxs = []
    for i, ((items, cap), (x0, x1, t, b)) in enumerate(zip(parts, boxes)):
        dx = i * (PW + GAP) - left
        dxs.append((dx, x0, x1))
        dy = top_margin + H - b                          # bottom-aligned
        S.items += shifted(items, dx, dy)
        S.text(dx + x0, top_margin + H + 0.12, x1 - x0, CAP_H, plain(cap), size=11, align='c')
    # legend for the corner badges, one row, in the free space above panel (b)
    dx_b, xb0, xb1 = dxs[1]
    lw_, lh = min(4.8, xb1 - xb0 + 0.3), 0.4
    b_top = top_margin + H - bottoms[1] + tops[1]        # panel (b)'s content top
    ly = max(0.05, top_margin + (b_top - top_margin - lh) / 2)   # centred in the free space
    assert ly + lh < b_top - 0.04, 'legend collides with panel (b)'
    x = dx_b + (xb0 + xb1) / 2 - lw_ / 2
    S.rect(x, ly, lw_, lh, fill='FFFFFF', line=INK, lw=1.3)
    S.text(x + 0.1, ly + 0.07, 0.6, 0.26, plain('\u2744\u2192\U0001F525'), size=11, align='l')
    S.text(x + 0.74, ly + 0.07, 2.9, 0.26,
           plain('initialized from pretrained model, finetuned in ours'), size=9.5, align='l')
    S.text(x + 3.62, ly + 0.07, 0.35, 0.26, plain('\U0001F525'), size=11, align='l')
    S.text(x + 3.97, ly + 0.07, 0.85, 0.26, plain('from scratch'), size=9.5, align='l')
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
