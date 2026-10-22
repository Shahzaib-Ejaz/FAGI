#!/usr/bin/env bash
# Propagates the current experiments/results/*.json into everything derived from them:
# paper_results.json, the figures, dashboard.html and the paper (DOCX).
# Run this after rerunning any experiment. It does not rerun experiments (see run_all.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python3 -W ignore"

echo "[1/4] merging results into paper_results.json"
$PY scripts/build_results.py
echo "[2/4] regenerating figures"
$PY scripts/generate_figures.py
echo "[3/4] rebuilding dashboard.html"
$PY scripts/build_dashboard.py
echo "[4/4] rebuilding the paper (checks every directional claim against the results)"
if command -v node >/dev/null 2>&1 && (cd paper && node -e "require('docx')" >/dev/null 2>&1); then
  (cd paper && node build_paper.js)
else
  echo "      skipped: run 'cd paper && npm install' once, then rerun this script"
fi
echo "done. Review with 'git status', then commit and push."
