#!/usr/bin/env bash
# Build the preprint PDF. Usage: bash build.sh   (or: latexmk -pdf main.tex)
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "BUILD OK -> main.pdf"
