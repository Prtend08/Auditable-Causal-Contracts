#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/a2a
PROJECT="$ROOT/project"
PY="$ROOT/envs/a2a/bin/python"
RUN="$PROJECT/src/run_streams.py"
OUT="$ROOT/results/v2_development_v5"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_block() {
  local cache="$1" backbone="$2" dataset="$3" confidence="$4"
  local output="$OUT/conf_${confidence}/$backbone/$dataset"
  mkdir -p "$output"
  "$PY" "$RUN" --cache "$cache" --output-dir "$output" \
    --streams iid_random abrupt_domain class_correlated --seeds 0 1 2 3 4 \
    --budgets 1.0 --strategies ask_or_adapt_v2 --pseudo-confidence "$confidence" \
    --pseudo-agreement 0.75 --risk-entropy-threshold 0.50 --summary-only \
    >"$output/console.log" 2>&1
}

pids=()
for confidence in 0.80 0.90 0.95; do
  run_block "$ROOT/cache/pacs_openai_vitb16.npz" openai_vitb16 PACS "$confidence" & pids+=("$!")
  run_block "$ROOT/cache/officehome_openai_vitb16.npz" openai_vitb16 OfficeHome "$confidence" & pids+=("$!")
  run_block "$ROOT/cache/terra_openai_vitb16.npz" openai_vitb16 TerraIncognita "$confidence" & pids+=("$!")
  run_block "$ROOT/cache/pacs_openclip_vitl14.npz" openclip_vitl14 PACS "$confidence" & pids+=("$!")
  run_block "$ROOT/cache/officehome_openclip_vitl14.npz" openclip_vitl14 OfficeHome "$confidence" & pids+=("$!")
  run_block "$ROOT/cache/terra_openclip_vitl14.npz" openclip_vitl14 TerraIncognita "$confidence" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
find "$OUT" -name '*.summary.json' -type f | wc -l
exit "$failed"
