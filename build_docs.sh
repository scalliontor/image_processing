#!/usr/bin/env bash
# Rebuild report/main.pdf, slides/main.pdf and slides/main.pptx from the
# LaTeX sources (needs tectonic: https://tectonic-typesetting.github.io).
# Tables and numbers are read from results/tables/ (written by make_tables.py).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
(cd report && tectonic -X compile main.tex)
(cd slides && tectonic -X compile main.tex)
$PY code/make_pptx.py
echo "built report/main.pdf, slides/main.pdf, slides/main.pptx"
