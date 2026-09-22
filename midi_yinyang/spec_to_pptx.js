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
    } else {
      opts.line = { color: 'FFFFFF', width: 0, transparency: 100 };
    }
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
  } else if (it.k === 'line') {
    const x = Math.min(it.x1, it.x2), y = Math.min(it.y1, it.y2);
    const w = Math.abs(it.x2 - it.x1), h = Math.abs(it.y2 - it.y1);
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
