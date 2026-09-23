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
    ys = []
    for it in items:
        if it['k'] == 'line':
            ys += [it['y1'], it['y2']]
        else:
            ys += [it['y'], it['y'] + it['h']]
    return min(ys), max(ys)


def shifted(items, dx, dy):
    out = []
    for it in items:
        it = dict(it)
        if it['k'] == 'line':
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
    bottoms = [bounds(items)[1] for items, _ in parts]
    tops = [bounds(items)[0] for items, _ in parts]
    H = max(b - t for b, t in zip(bottoms, tops))      # tallest panel's content height
    top_margin = 0.15
    fig_arch_pptx.SLIDE_W = len(parts) * PW + (len(parts) - 1) * GAP
    fig_arch_pptx.SLIDE_H = top_margin + H + 0.12 + CAP_H + 0.1

    S = Spec()
    for i, ((items, cap), b, t) in enumerate(zip(parts, bottoms, tops)):
        dx = i * (PW + GAP)
        dy = top_margin + H - b                          # bottom-aligned
        S.items += shifted(items, dx, dy)
        S.text(dx, top_margin + H + 0.12, PW, CAP_H, plain(cap), size=11, align='c')
    # legend for the corner badges, in the empty space above panel (b)
    lx = 1 * (PW + GAP)
    ly = top_margin + 0.55
    lw_, lh = 4.0, 0.78
    S.rect(lx + (PW - lw_) / 2, ly, lw_, lh, fill='FFFFFF', line=INK, lw=0.9)
    for k, (glyph, txt) in enumerate((('\u2744\u2192\U0001F525',
                                       'from the pretrained single-stream model, fine-tuned'),
                                      ('\U0001F525', 'new component, trained from scratch'))):
        yy = ly + 0.1 + k * 0.32
        S.text(lx + (PW - lw_) / 2 + 0.1, yy, 0.6, 0.26, plain(glyph), size=10.5, align='l')
        S.text(lx + (PW - lw_) / 2 + 0.72, yy, lw_ - 0.8, 0.26, plain(txt), size=8.5, align='l')
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
