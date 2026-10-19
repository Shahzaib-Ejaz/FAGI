const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, SectionType, HeadingLevel, ShadingType, Footer, PageNumber, Header,
} = require("docx");

const FONT = "Times New Roman";
const COL_W = 4896;
const PAGE = { width: 12240, height: 15840 };
const MARGIN = { top: 1080, bottom: 1080, left: 1008, right: 1008 };

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const body = (text, opts = {}) => new Paragraph({
  children: runs(text, { size: 20 }), alignment: AlignmentType.JUSTIFIED,
  indent: opts.noIndent ? undefined : { firstLine: 202 }, spacing: { after: 0, line: 230 },
});

const section = (title) => new Paragraph({
  children: [new TextRun({ text: title.toUpperCase(), font: FONT, size: 20, smallCaps: true })],
  heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
  keepNext: true,
});

const subsection = (title) => new Paragraph({
  children: [new TextRun({ text: title, font: FONT, size: 20, italics: true })],
  heading: HeadingLevel.HEADING_2, spacing: { before: 120, after: 60 }, keepNext: true,
});

const caption = (text, center = true) => new Paragraph({
  children: runs(text, { size: 16 }), alignment: center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
  spacing: { before: 60, after: 160 },
});

const tableTitle = (num, title) => [
  new Paragraph({ children: [new TextRun({ text: `TABLE ${num}`, font: FONT, size: 16 })],
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true }),
  new Paragraph({ children: [new TextRun({ text: title, font: FONT, size: 16, smallCaps: true })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true }),
];

function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const rule = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const cell = (t, i, isHead, isLast, bold) => new TableCell({
    verticalAlign: "center",
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: isHead ? rule : none, bottom: isHead || isLast ? rule : none, left: none, right: none },
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    children: [new Paragraph({ children: runs(String(t), { size: 15, bold: bold && !isHead }), keepNext: !isLast,
      keepLines: true, alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  const trs = [new TableRow({ tableHeader: true, cantSplit: true,
    children: header.map((h, i) => cell(h, i, true, false, false)) })];
  rows.forEach((r, ri) => {
    const bold = r.bold === true;
    const cells = r.cells || r;
    trs.push(new TableRow({ cantSplit: true,
      children: cells.map((c, i) => cell(c, i, false, ri === rows.length - 1, bold)) }));
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: trs,
    alignment: AlignmentType.CENTER });
}

function figure(path, widthIn, heightIn) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path),
      transformation: { width: Math.round(widthIn * 96), height: Math.round(heightIn * 96) },
      altText: { title: "figure", description: path.split("/").pop(), name: "figure" } })],
    alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
  });
}

function algorithm(title, lines) {
  const rule = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const paras = [new Paragraph({ children: runs(title, { size: 16 }), spacing: { after: 40 } })];
  for (const l of lines) {
    const indent = (l.match(/^ */)[0].length) * 120;
    paras.push(new Paragraph({ children: runs(l.trim(), { size: 15 }), indent: { left: indent, hanging: 0 },
      spacing: { after: 0, line: 210 }, keepNext: true, keepLines: true }));
  }
  return new Table({
    width: { size: COL_W, type: WidthType.DXA }, columnWidths: [COL_W],
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({
      width: { size: COL_W, type: WidthType.DXA }, children: paras,
      borders: { top: rule, bottom: rule, left: none, right: none },
      margins: { top: 60, bottom: 60, left: 60, right: 60 } })] })],
  });
}

const spacer = () => new Paragraph({ children: [], spacing: { after: 60 } });

function reference(i, text) {
  return new Paragraph({ children: [new TextRun({ text: `[${i}]\t`, font: FONT, size: 16 }), ...runs(text, { size: 16 })],
    tabStops: [{ type: "left", position: 360 }], indent: { left: 360, hanging: 360 },
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 } });
}

function build(titleBlock, bodyChildren, outPath, runningHead) {
  const doc = new Document({
    creator: "Shahzaib Ejaz",
    title: "FAGI",
    styles: {
      default: { document: { run: { font: FONT, size: 20 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 20 }, paragraph: { outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 20, italics: true }, paragraph: { outlineLevel: 1 } },
      ],
    },
    sections: [
      { properties: { page: { size: PAGE, margin: MARGIN } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: runningHead, font: FONT, size: 14 })] })] }) },
        footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
        children: titleBlock },
      { properties: { type: SectionType.CONTINUOUS, page: { size: PAGE, margin: MARGIN },
          column: { count: 2, space: 432, equalWidth: true } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: runningHead, font: FONT, size: 14 })] })] }) },
        footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
        children: bodyChildren },
    ],
  });
  return Packer.toBuffer(doc).then((b) => fs.writeFileSync(outPath, b));
}

module.exports = { FONT, COL_W, runs, body, section, subsection, caption, tableTitle, table, figure, algorithm,
  spacer, reference, build, Paragraph, TextRun, AlignmentType };
