#!/usr/bin/env bash
# Package the submission as required by the course guide:
#   LopXX_NhomYY_TenDeTai.zip = slides (.pptx + .pdf), code/ + requirements.txt +
#   README.md + samples/, report (.pdf), demo video.
# Usage: bash make_submission.sh Lop01 Nhom05   (default: LopXX NhomYY)
set -euo pipefail
cd "$(dirname "$0")"
LOP=${1:-LopXX}
NHOM=${2:-NhomYY}
NAME="${LOP}_${NHOM}_GateFuseLA"
OUT=$(mktemp -d)/"$NAME"
mkdir -p "$OUT"
cp -R code samples requirements.txt README.md run_all.sh build_docs.sh "$OUT"/
mkdir -p "$OUT/results"
cp -R results/tables results/figures results/checkpoints "$OUT/results/"
cp report/main.pdf "$OUT/BaoCao_${NAME}.pdf"
cp slides/main.pdf "$OUT/Slide_${NAME}.pdf"
cp slides/main.pptx "$OUT/Slide_${NAME}.pptx"
[ -f results/demo_video.mp4 ] && cp results/demo_video.mp4 "$OUT/Video_demo_${NAME}.mp4"
mkdir -p "$OUT/latex"
cp -R report "$OUT/latex/report"
cp -R slides "$OUT/latex/slides"
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" \( -name "*.log" -o -name ".DS_Store" \) -delete
(cd "$(dirname "$OUT")" && zip -qr "$NAME.zip" "$NAME")
mv "$(dirname "$OUT")/$NAME.zip" ..
echo "wrote ../$NAME.zip"
