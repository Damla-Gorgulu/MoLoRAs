#!/bin/bash
# Compile the Beamer slides (requires pdflatex + pgfplots + tikz)
#
# OPTION 1 — On the VALAR software node (interactive):
#   ssh software    # from login01 (requires password)
#   cd /home/eyavuz21/repos/MoLoRAs/lora_attention/presentation
#   bash compile.sh
#
# OPTION 2 — SLURM batch:
#   sbatch compile_slurm.sh
#
# OPTION 3 — Overleaf:
#   Upload slides.tex + images to Overleaf, set compiler to pdfLaTeX.
#   NOTE: Overleaf can't reach /scratch paths — copy images to the
#   project dir first (see images/ subfolder).
#
# NOTE: Images are referenced via absolute /scratch paths.
# If compiling off this machine, update \graphicspath in slides.tex.

set -e
cd "$(dirname "$0")"

echo "[compile] Pass 1..."
pdflatex -interaction=nonstopmode slides.tex > compile.log 2>&1

echo "[compile] Pass 2 (aux refs)..."
pdflatex -interaction=nonstopmode slides.tex >> compile.log 2>&1

echo "[compile] Done → slides.pdf"
ls -lh slides.pdf
