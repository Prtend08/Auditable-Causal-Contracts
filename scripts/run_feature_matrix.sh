#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/a2a
PY="$ROOT/envs/a2a/bin/python"
PROJECT="$ROOT/project"
EXTRACT="$PROJECT/src/extract_features.py"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_openai() {
  local dataset="$1"
  local data_root="$2"
  local data_format="$3"
  local output="$4"
  "$PY" "$EXTRACT" \
    --dataset "$dataset" --data-root "$data_root" --data-format "$data_format" \
    --output "$output" --backend openai --model-name ViT-B/16 \
    --clip-source "$ROOT/data/openai_standard" --weights "$ROOT/cache/clip/ViT-B-16.pt" \
    --batch-size 512 --workers 8
}

run_openclip_l14() {
  local dataset="$1"
  local data_root="$2"
  local data_format="$3"
  local output="$4"
  "$PY" "$EXTRACT" \
    --dataset "$dataset" --data-root "$data_root" --data-format "$data_format" \
    --output "$output" --backend open_clip --model-name ViT-L-14 \
    --weights "$ROOT/data/openclip/open_clip_pytorch_model.bin" \
    --batch-size 192 --workers 8
}

run_openai TerraIncognita "$ROOT/data/TerraIncognita" folder "$ROOT/cache/terra_openai_vitb16.npz"
run_openclip_l14 PACS "$ROOT/data/PACS" folder "$ROOT/cache/pacs_openclip_vitl14.npz"
run_openclip_l14 OfficeHome "$ROOT/data/OfficeHome" officehome_parquet "$ROOT/cache/officehome_openclip_vitl14.npz"
run_openclip_l14 TerraIncognita "$ROOT/data/TerraIncognita" folder "$ROOT/cache/terra_openclip_vitl14.npz"

sha256sum "$ROOT"/cache/*vit*.npz "$ROOT"/cache/*vit*.json
