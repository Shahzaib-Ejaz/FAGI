#!/usr/bin/env bash
# Builds the LaTeX (TMLR style) version of the paper: out/tex/FAGI.tex (named) and out/tex_anon/FAGI.tex (anonymous).
# Needs tmlr.sty, tmlr.bst and fancyhdr.sty from https://github.com/JmlrOrg/tmlr-style-file in TMLR_STYLE_DIR
# (default: paper/tmlr-style). If pdflatex is installed the PDFs are compiled too.
set -euo pipefail
cd "$(dirname "$0")"
STYLE="${TMLR_STYLE_DIR:-tmlr-style}"
for mode in named anon; do
  out="out/tex"; [ "$mode" = anon ] && out="out/tex_anon"
  mkdir -p "$out"
  if [ "$mode" = anon ]; then FAGI_ANON=1 FAGI_FORMAT=tex node build_paper.js "$out/FAGI.tex"; else FAGI_FORMAT=tex node build_paper.js "$out/FAGI.tex"; fi
  for f in tmlr.sty tmlr.bst fancyhdr.sty; do [ -f "$STYLE/$f" ] && cp "$STYLE/$f" "$out/"; done
  if command -v pdflatex >/dev/null && [ -f "$out/tmlr.sty" ]; then
    work="$out"
    if ! kpsewhich lmodern.sty >/dev/null 2>&1; then
      # No Latin Modern fonts on this machine: compile a throwaway copy with the default fonts, for checking only.
      work="$(mktemp -d)"; cp -r "$out"/. "$work"/
      sed -i 's/\\usepackage\[T1\]{fontenc}/\\usepackage[OT1]{fontenc}/; s/\\usepackage{lmodern}//' "$work/tmlr.sty"
    fi
    (cd "$work" && for i in 1 2 3; do pdflatex -interaction=nonstopmode FAGI.tex >/dev/null 2>&1 || true; done)
    [ "$work" != "$out" ] && cp "$work/FAGI.pdf" "$work/FAGI.log" "$out"/ 2>/dev/null || true
    echo "$out: $(grep -c '^!' "$out/FAGI.log" || true) LaTeX errors, $(grep -c 'Overfull' "$out/FAGI.log" || true) overfull boxes"
  fi
done
