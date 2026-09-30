#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/a2a
PROJECT="$ROOT/project"
PY="$ROOT/envs/a2a/bin/python"
RUN="$PROJECT/src/run_streams.py"
OUT="$ROOT/results/v2_strict_ablations_v6"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_block() {
  local cache="$1" dataset="$2"
  local output="$OUT/$dataset"
  mkdir -p "$output"
  "$PY" "$RUN" --cache "$cache" --output-dir "$output" \
    --streams iid_random abrupt_domain class_correlated --seeds 10 11 12 13 14 \
    --budgets 1.0 \
    --strategies ask_or_adapt_v2 ask_or_adapt v2_no_gate v2_no_entropy v2_no_disagreement v2_no_coverage \
    --pseudo-confidence 0.80 --pseudo-agreement 0.75 --risk-entropy-threshold 0.50 \
    >"$output/console.log" 2>&1
}

pids=()
run_block "$ROOT/cache/pacs_openai_vitb16.npz" PACS & pids+=("$!")
run_block "$ROOT/cache/officehome_openai_vitb16.npz" OfficeHome & pids+=("$!")
run_block "$ROOT/cache/terra_openai_vitb16.npz" TerraIncognita & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
find "$OUT" -name '*.summary.json' -type f | wc -l
exit "$failed"
