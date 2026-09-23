// Build a .pptx from the drawing spec JSON written by fig_arch_pptx.Spec
// (rect / ellipse / text / line items in inches, y down).
//   node spec_to_pptx.js spec.json out.pptx
const fs = require('fs');
const pptxgen = require('pptxgenjs');

const [specPath, outPath] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
const pres = new pptxgen();
pres.defineLayout({ name: 'FIG', width: spec.width, height: spec.height });
pres.layout = 'FIG';
const slide = pres.addSlide();
const FONT = spec.font || 'Cambria';

function runsToText(runs, size, color, bold) {
  const out = [];
  for (const [text, flags] of runs) {
    const f = flags.split(' ').filter(Boolean);
    if (text === '\n') { if (out.length) out[out.length - 1].options.breakLine = true; continue; }
    const o = { fontFace: FONT, fontSize: size, color: color, italic: f.includes('i'),
                bold: bold || f.includes('b') };
    if (f.includes('sup')) o.superscript = true;
    if (f.includes('sub')) o.subscript = true;
    out.push({ text: text, options: o });
  }
  return out;
}

const items = [...spec.items].sort((a, b) => a.z - b.z);
for (const it of items) {
  if (it.k === 'rect') {
    const opts = { x: it.x, y: it.y, w: it.w, h: it.h, margin: 0 };
    opts.fill = it.fill ? { color: it.fill } : { color: 'FFFFFF', transparency: 100 };
    if (it.line) {
      opts.line = { color: it.line, width: it.lw };
      if (it.dash) opts.line.dashType = 'sysDot';
    }                                   // no outline: leave `line` unset
    const shape = it.shape === 'ellipse' ? pres.ShapeType.ellipse : pres.ShapeType.rect;
    if (it.runs && it.runs.length) {
      slide.addText(runsToText(it.runs, it.size, it.color, false),
        Object.assign(opts, { shape: shape, align: { c: 'center', l: 'left', r: 'right' }[it.align],
                              valign: 'middle', fit: 'none', wrap: false }));
    } else {
      slide.addShape(shape, opts);
    }
  } else if (it.k === 'text') {
    slide.addText(runsToText(it.runs, it.size, it.color, it.bold),
      { x: it.x, y: it.y, w: it.w, h: it.h, margin: 0, isTextBox: true,
        align: { c: 'center', l: 'left', r: 'right' }[it.align],
        valign: it.valign === 't' ? 'top' : 'middle', fit: 'none', wrap: false });
  } else if (it.k === 'poly') {
    const xs = it.points.map(p => p[0]), ys = it.points.map(p => p[1]);
    const x = Math.min(...xs), y = Math.min(...ys);
    const w = Math.max(Math.max(...xs) - x, 0.004), h = Math.max(Math.max(...ys) - y, 0.004);
    const pts = it.points.map(p => ({ x: p[0] - x, y: p[1] - y }));
    pts.push({ close: true });
    slide.addShape(pres.ShapeType.custGeom, {
      x: x, y: y, w: w, h: h, points: pts,
      fill: { color: it.fill, transparency: Math.round((1 - it.alpha) * 100) },
    });
  } else if (it.k === 'curve') {
    // custom geometry: one cubic Bezier with vertical tangents at both ends
    const x = Math.min(it.x1, it.x2), y = Math.min(it.y1, it.y2);
    const w = Math.max(Math.abs(it.x2 - it.x1), 0.004), h = Math.max(Math.abs(it.y2 - it.y1), 0.004);
    const dy = (it.y2 - it.y1) * it.bend;
    const pts = [{ x: it.x1 - x, y: it.y1 - y },
                 { x: it.x2 - x, y: it.y2 - y,
                   curve: { type: 'cubic', x1: it.x1 - x, y1: it.y1 - y + dy,
                            x2: it.x2 - x, y2: it.y2 - y - dy } }];
    const opts = { x: x, y: y, w: w, h: h, points: pts,
                   fill: { color: 'FFFFFF', transparency: 100 },
                   line: { color: it.color, width: it.lw } };
    if (it.arrow) opts.line.endArrowType = 'triangle';
    slide.addShape(pres.ShapeType.custGeom, opts);
  } else if (it.k === 'line') {
    const x = Math.min(it.x1, it.x2), y = Math.min(it.y1, it.y2);
    // never a zero-extent shape: PowerPoint rejects lines with w or h = 0
    const w = Math.max(Math.abs(it.x2 - it.x1), 0.004), h = Math.max(Math.abs(it.y2 - it.y1), 0.004);
    const opts = { x: x, y: y, w: w, h: h, line: { color: it.color, width: it.lw } };
    if (it.dash) opts.line.dashType = 'dash';
    if (it.arrow) opts.line.endArrowType = 'triangle';
    // pptxgenjs draws from top-left to bottom-right; flip to match the spec
    if (it.x2 < it.x1) opts.flipH = true;
    if (it.y2 < it.y1) opts.flipV = true;
    slide.addShape(pres.ShapeType.line, opts);
  }
}
pres.writeFile({ fileName: outPath }).then(() => console.log('wrote', outPath));
