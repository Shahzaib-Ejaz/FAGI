// LaTeX backend with the same interface as lib.js (the DOCX backend). build_paper.js selects it with
// FAGI_FORMAT=tex; the functions return plain objects and build() assembles one .tex file for the TMLR style.
// FAGI_ANON=1 writes the anonymous version TMLR requires for submission.
const fs = require("fs");
const path = require("path");

const ANON = !!process.env.FAGI_ANON;

const MAP = {
  "ε": "\\ensuremath{\\epsilon}", "σ": "\\ensuremath{\\sigma}", "δ": "\\ensuremath{\\delta}", "τ": "\\ensuremath{\\tau}",
  "Σ": "\\ensuremath{\\Sigma}", "←": "\\ensuremath{\\leftarrow}", "±": "\\ensuremath{\\pm}", "×": "\\ensuremath{\\times}",
  "−": "\\ensuremath{-}", "½": "\\ensuremath{\\frac{1}{2}}", "‖": "\\ensuremath{\\Vert}", "∅": "\\ensuremath{\\emptyset}",
  "⌈": "\\ensuremath{\\lceil}", "⌉": "\\ensuremath{\\rceil}", "∈": "\\ensuremath{\\in}", "∇": "\\ensuremath{\\nabla}",
  "∞": "\\ensuremath{\\infty}", "≤": "\\ensuremath{\\leq}", "≥": "\\ensuremath{\\geq}", "·": "\\textperiodcentered{}",
  "–": "--", "—": "---", "’": "'", "‘": "`", "é": "\\'e", "è": "\\`e", "“": "``", "”": "''",
};
const SPECIAL = {
  "\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
  "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "<": "\\textless{}", ">": "\\textgreater{}",
};

function esc(t) {
  let o = "";
  for (const ch of t) {
    if (MAP[ch]) o += MAP[ch];
    else if (SPECIAL[ch]) o += SPECIAL[ch];
    else if (ch.charCodeAt(0) > 127) throw new Error(`unmapped character U+${ch.codePointAt(0).toString(16)} (${ch}) in: ${t.slice(0, 60)}`);
    else o += ch;
  }
  return o;
}

function anonymize(t) {
  if (!ANON) return t;
  return t
    .replace(/https:\/\/github\.com\/Shahzaib-Ejaz\/FAGI/g, "an anonymized repository (link withheld for review)")
    .replace(/Ejaz catalogued/g, "Prior work catalogued");
}

// Converts the builder's inline markup (**bold**, *italic*, _{sub}, ^{sup}) plus citation and URL tokens.
function inline(raw) {
  raw = anonymize(raw).replace(/"([^"]*)"/g, "“$1”");
  const parts = raw.split(/(\u0001[^\u0002]*\u0002|https?:\/\/[^\s<>]+?(?=[.,;)]*(?:\s|$))|\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g);
  return parts.map((p) => {
    if (!p) return "";
    if (p.startsWith("\u0001")) return `\\citep{${p.slice(1, -1)}}`;
    if (/^https?:\/\//.test(p)) return `\\url{${p}}`;
    if (p.startsWith("**")) return `\\textbf{${inline(p.slice(2, -2))}}`;
    if (p.startsWith("_{")) return `\\textsubscript{${esc(p.slice(2, -1))}}`;
    if (p.startsWith("^{")) return `\\textsuperscript{${esc(p.slice(2, -1))}}`;
    if (p.startsWith("*")) return `\\emph{${inline(p.slice(1, -1))}}`;
    return esc(p);
  }).join("");
}

const runs = (text) => inline(text);
const body = (text) => ({ kind: "body", tex: inline(text) });
const section = (title) => ({ kind: "section", tex: `\\section{${inline(title.replace(/^[IVX]+\.\s*/, ""))}}` });
const subsection = (title) => ({ kind: "section", tex: `\\subsection{${inline(title.replace(/^[A-Z]\.\s*/, ""))}}` });
const unnumbered = (title) => (title === "References" ? { kind: "skip" }
  : { kind: "section", tex: `\\section*{${inline(title === "Acknowledgment" ? "Acknowledgments" : title)}}` });
const caption = (text, center = true) => ({ kind: "caption", text, center });
const tableTitle = (num, title) => [{ kind: "tableTitle", num, title }];
const spacer = () => ({ kind: "skip" });
const abstract = (text) => [{ kind: "abstract", tex: inline(text) }];
const titleBlock = (meta) => meta;

function table(header, rows) {
  const n = header.length;
  const head = header.map((h) => inline(String(h))).join(" & ");
  const lines = rows.map((r) => {
    const bold = r.bold === true;
    const cells = (r.cells || r).map((c) => (bold ? `\\textbf{${inline(String(c))}}` : inline(String(c))));
    return cells.join(" & ") + " \\\\";
  });
  const tex = `\\begin{tabular}{l${"c".repeat(n - 1)}}\n\\toprule\n${head} \\\\\n\\midrule\n${lines.join("\n")}\n\\bottomrule\n\\end{tabular}`;
  return { kind: "table", tex };
}

const figure = (file) => ({ kind: "figure", file });
const algorithm = (title, lines) => ({ kind: "algorithm", title, lines });
const reference = (i, text, key) => ({ kind: "reference", key, text });

function algorithmTex(a) {
  const rows = a.lines.map((l) => {
    const indent = l.match(/^ */)[0].length;
    return `\\hspace*{${(indent * 0.3).toFixed(2)}em}${inline(l.trim())}`;
  }).join("\\\\\n");
  return `\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}\\small\n\\rule{\\linewidth}{0.6pt}\\\\[-0.2ex]\n${inline(a.title)}\\\\[-0.6ex]\n\\rule{\\linewidth}{0.4pt}\\\\\n${rows}\\\\[-0.3ex]\n\\rule{\\linewidth}{0.6pt}\n\\end{minipage}\n`;
}

function build(meta, items, outPath) {
  const outDir = path.dirname(outPath);
  const figDir = path.join(outDir, "figs");
  fs.mkdirSync(figDir, { recursive: true });
  const out = [];
  let open = null; // a table, figure or algorithm waiting for its caption or note
  const note = (c) => `\\par\\vspace{2pt}{\\footnotesize ${inline(c.text)}\\par}`;
  const closeOpen = () => {
    if (!open) return;
    if (open.type === "table") out.push(`\\begin{table}[tbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\caption{${inline(open.title)}}\n${open.tabular}\n\\end{table}\n`);
    else if (open.type === "algorithm") out.push(algorithmTex(open.alg));
    else throw new Error("figure without caption");
    open = null;
  };
  const refs = [];
  for (const it of items.flat()) {
    if (it.kind === "skip") continue;
    if (it.kind === "tableTitle") { closeOpen(); open = { type: "table", title: it.title, tabular: "" }; continue; }
    if (it.kind === "table") { if (!open) throw new Error("table without title"); open.tabular = it.tex; continue; }
    if (it.kind === "algorithm") { closeOpen(); open = { type: "algorithm", alg: it }; continue; }
    if (it.kind === "figure") { closeOpen(); open = { type: "figure", file: it.file }; continue; }
    if (it.kind === "caption") {
      if (open && open.type === "table") {
        open.tabular += `\n${note(it)}`; closeOpen();
      } else if (open && open.type === "algorithm") {
        out.push(algorithmTex(open.alg) + note(it) + "\n"); open = null;
      } else if (open && open.type === "figure") {
        const pdf = open.file.replace(/\.png$/, ".pdf");
        fs.copyFileSync(pdf, path.join(figDir, path.basename(pdf)));
        const cap = it.text.replace(/^Fig\.\s*\d+\.\s*/, "");
        out.push(`\\begin{figure}[tbp]\n\\centering\n\\includegraphics[width=0.62\\linewidth]{figs/${path.basename(pdf)}}\n\\caption{${inline(cap)}}\n\\end{figure}\n`);
        open = null;
      } else {
        throw new Error("caption with nothing to attach to: " + it.text.slice(0, 60));
      }
      continue;
    }
    closeOpen();
    if (it.kind === "reference") { refs.push(it); continue; }
    if (it.kind === "abstract") { out.push(`\\begin{abstract}\n${it.tex}\n\\end{abstract}\n`); continue; }
    out.push(it.tex + (it.kind === "body" ? "\n" : ""));
  }
  closeOpen();

  const [dept, ...rest] = meta.affiliation.split(", ");
  const authorBlock = `\\author{\\name ${esc(meta.author)} \\email ${esc(meta.email)} \\\\\n      \\addr ${esc(dept)}\\\\\n      ${esc(rest.join(", "))}}`;
  const bib = refs.map((r) => {
    const text = ANON && r.key === "ejaz"
      ? 'Anonymous, "Cloud API management and integration patterns in secure cloud computing environments," 2026.'
      : r.text;
    return `\\bibitem{${r.key}} ${inline(text)}`;
  }).join("\n");
  const now = new Date();
  const tex = `\\documentclass[10pt]{article}
\\PassOptionsToPackage{numbers,sort&compress}{natbib}
${ANON ? "\\usepackage{tmlr}" : "\\usepackage[preprint]{tmlr}"}
% Built by paper/build_paper.js with FAGI_FORMAT=tex. Without FAGI_ANON the [preprint] option shows the author;
% with FAGI_ANON=1 the default option gives the anonymous version TMLR requires for submission.
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath,amssymb}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage[hyphens]{url}
\\usepackage[hidelinks]{hyperref}
\\setcitestyle{numbers,square,comma}
\\renewcommand{\\thesection}{\\Roman{section}}
\\renewcommand{\\thesubsection}{\\thesection-\\Alph{subsection}}
\\renewcommand{\\thetable}{\\Roman{table}}

\\title{${inline(meta.title)}}
${authorBlock}

\\def\\month{${process.env.FAGI_MONTH || now.getMonth() + 1}}
\\def\\year{${process.env.FAGI_YEAR || now.getFullYear()}}

\\begin{document}
\\maketitle

${out.join("\n")}

\\begin{thebibliography}{99}
${bib}
\\end{thebibliography}

\\end{document}
`;
  fs.writeFileSync(outPath, tex);
  return Promise.resolve();
}

module.exports = { runs, body, section, subsection, unnumbered, caption, tableTitle, table, figure, algorithm, spacer,
  reference, abstract, titleBlock, build, format: "tex" };
