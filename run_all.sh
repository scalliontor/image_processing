#!/usr/bin/env bash
# Reproduce every experiment of the report, in order. Resumable: a run whose
# summary.json exists is skipped. Usage:  LA_WORK=/path/to/work bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export LA_WORK=${LA_WORK:-$(pwd)/work}
PY=${PY:-python}
mkdir -p "$LA_WORK/data" "$LA_WORK/weights/mricore" "$LA_WORK/logs"

# 0. data (2018 LA challenge, UA-MT h5 copy) and MRI-CORE weights ----------
if [ ! -d "$LA_WORK/data/uamt_src/data" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/yulequan/UA-MT.git "$LA_WORK/data/uamt_src"
  git -C "$LA_WORK/data/uamt_src" sparse-checkout set data
fi
[ -f "$LA_WORK/weights/mricore/MRI_CORE_vitb.pth" ] || \
  gdown 1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW -O "$LA_WORK/weights/mricore/MRI_CORE_vitb.pth"

# 1-2. slices and frozen features ------------------------------------------
[ -f "$LA_WORK/prep/meta.csv" ] || $PY code/data_prep.py
for enc in dinov3 mricore; do
  [ -f "$LA_WORK/features/$enc.npy" ] || $PY code/extract_features.py --encoder $enc
done

# 3. classical baseline + all learned configurations ---------------------
[ -f "$LA_WORK/runs/classical/pred_test.npz" ] || $PY code/classical.py
for labels in full few; do
  for seed in 0 1 2; do
    for model in gate dinov3 mricore concat unet gatedge; do
      run="${model}_${labels}_s${seed}"
      [ -f "$LA_WORK/runs/$run/summary.json" ] && continue
      $PY code/train.py --model $model --labels $labels --seed $seed 2>&1 | tee "$LA_WORK/logs/$run.log"
    done
  done
done

# 4-6. evaluation, figures, samples ---------------------------------------
$PY code/evaluate.py
$PY code/analysis.py
$PY code/benchmark.py
$PY code/figures.py
$PY code/make_tables.py
$PY code/make_samples.py
mkdir -p results/checkpoints && cp "$LA_WORK/runs/gate_full_s0/best.pt" results/checkpoints/gatefuse_full_s0.pt
$PY code/make_video.py
# report + slides (needs tectonic):  bash build_docs.sh
