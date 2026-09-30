#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/a2a
PROJECT="$ROOT/project"
PY="$ROOT/envs/a2a/bin/python"
RUN="$PROJECT/src/run_streams.py"
OUT="$ROOT/results/external_selection_v4"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_one_matrix() {
  local cache="$1" backbone="$2" dataset="$3"
  local output="$OUT/$backbone/$dataset"
  mkdir -p "$output"
  "$PY" "$RUN" --cache "$cache" --output-dir "$output" \
    --streams iid_random abrupt_domain class_correlated \
    --seeds 5 6 7 8 9 --budgets 1.0 --strategies dynamic_entropy_online \
    >"$output/console.log" 2>&1
}

pids=()
run_one_matrix "$ROOT/cache/pacs_openai_vitb16.npz" openai_vitb16 PACS & pids+=("$!")
run_one_matrix "$ROOT/cache/officehome_openai_vitb16.npz" openai_vitb16 OfficeHome & pids+=("$!")
run_one_matrix "$ROOT/cache/terra_openai_vitb16.npz" openai_vitb16 TerraIncognita & pids+=("$!")
run_one_matrix "$ROOT/cache/pacs_openclip_vitl14.npz" openclip_vitl14 PACS & pids+=("$!")
run_one_matrix "$ROOT/cache/officehome_openclip_vitl14.npz" openclip_vitl14 OfficeHome & pids+=("$!")
run_one_matrix "$ROOT/cache/terra_openclip_vitl14.npz" openclip_vitl14 TerraIncognita & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
