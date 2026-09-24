"""The model figure: panels a, b, c side by side with the captions under
each panel:
  (a) Dual-stream Attention   (b) Dual-stream MoE   (c) Dual-stream Decoding Process
Each panel is built by its own script (fig_attn_panel, fig_moe_panel,
fig_decode_panel); this one drops their top titles, packs them
horizontally at a fixed gap between their CONTENT boxes, stretches
each panel's row spacing so all three span the same height (tops and
bottoms aligned), and adds the captions and the badge legend. The
legend sits at the top of panel (b)'s column, its top on the common
top line, so the column reads as one unit. Edit a panel in its script;
rebuild here.

    python fig_model_figure.py --out <path-without-extension>
writes <out>.pptx (editable) and <out>.png (preview).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_arch_pptx  # noqa: E402
import fig_attn_panel, fig_moe_panel, fig_decode_panel  # noqa: E402,E401
from fig_arch_pptx import Spec, write_pptx_js, write_preview, plain, badge, INK  # noqa: E402

GAP = 0.42                      # between the content boxes of neighbouring panels
MARGIN = 0.1                    # slide edge to content
CAP_H = 0.3                     # caption row
CAP_GAP = 0.1                   # content bottom to caption
LEG_H, LEG_GAP = 0.7, 0.16      # legend box height, and its gap to panel (b)'s content
# (module, caption, number of row gaps the stretch is spread over)
PANELS = [(fig_attn_panel, '(a) Dual-stream Attention', 8),
          (fig_moe_panel, '(b) Dual-stream MoE', 6),
          (fig_decode_panel, '(c) Dual-stream Decoding Process', 2)]


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


def panel_items(mod, stretch):
    items = mod.build(stretch=stretch).items
    # drop the panel's own top title ("a. ...", "b. ...", "c. ...")
    return [it for it in items if not (it['k'] == 'text' and it['runs'] and
                                       it['runs'][0][0][:3] in ('a. ', 'b. ', 'c. '))]


def build():
    # pass 1: natural heights
    nat = [bounds(panel_items(mod, 0.0)) for mod, _c, _n in PANELS]
    heights = [b[3] - b[2] for b in nat]
    # panel (b) also carries the legend above its content
    need = [heights[0], heights[1] + LEG_H + LEG_GAP, heights[2]]
    H = max(need)
    # pass 2: stretch every panel's rows so its column spans H exactly
    parts, boxes = [], []
    for (mod, cap, n_gaps), nd in zip(PANELS, need):
        stretch = (H - nd) / n_gaps
        items = panel_items(mod, stretch)
        parts.append((items, cap)); boxes.append(bounds(items))
    for (x0, x1, t, b), nd, h0 in zip(boxes, need, heights):
        assert abs((b - t) - (H - (nd - h0))) < 0.02, 'stretch did not land on the common height'

    S = Spec()
    top = MARGIN
    cursor = MARGIN
    for i, ((items, cap), (x0, x1, t, b)) in enumerate(zip(parts, boxes)):
        dx = cursor - x0
        dy = top + H - b                                   # bottom on the common line
        S.items += shifted(items, dx, dy)
        S.text(cursor, top + H + CAP_GAP, x1 - x0, CAP_H, plain(cap), size=11, align='c')
        if i == 1:
            # the badge legend, as wide as the column, its top on the common top line
            lx, lw_ = cursor, x1 - x0
            S.rect(lx, top, lw_, LEG_H, fill='FFFFFF', line=INK, lw=1.0)
            for k, (kind, txt) in enumerate((('pre', 'initialized from pretrained model, finetuned in ours'),
                                             ('new', 'from scratch'))):
                yy = top + 0.07 + k * 0.3
                badge(S, lx + 0.86, yy, kind, h=0.22)                  # same size as the corner badges
                S.text(lx + 0.98, yy - 0.01, lw_ - 1.0, 0.24, plain(txt), size=9.5, align='l')
            assert top + LEG_H + LEG_GAP <= top + H - (b - t) + 0.02, 'legend collides with panel (b)'
        cursor += (x1 - x0) + GAP
    fig_arch_pptx.SLIDE_W = cursor - GAP + MARGIN
    fig_arch_pptx.SLIDE_H = top + H + CAP_GAP + CAP_H + MARGIN
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
