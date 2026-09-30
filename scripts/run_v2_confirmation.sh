#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/a2a
PROJECT="$ROOT/project"
PY="$ROOT/envs/a2a/bin/python"
RUN="$PROJECT/src/run_streams.py"
OUT="$ROOT/results/v2_confirmation_v6"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_block() {
  local cache="$1" backbone="$2" dataset="$3"
  local output="$OUT/$backbone/$dataset"
  mkdir -p "$output"
  "$PY" "$RUN" --cache "$cache" --output-dir "$output" \
    --streams iid_random abrupt_domain class_correlated --seeds 10 11 12 13 14 \
    --budgets 1.0 \
    --strategies random periodic entropy margin disagreement dynamic_entropy_online ask_or_adapt_v2 \
    --pseudo-confidence 0.80 --pseudo-agreement 0.75 --risk-entropy-threshold 0.50 \
    >"$output/console.log" 2>&1
}

pids=()
run_block "$ROOT/cache/pacs_openai_vitb16.npz" openai_vitb16 PACS & pids+=("$!")
run_block "$ROOT/cache/officehome_openai_vitb16.npz" openai_vitb16 OfficeHome & pids+=("$!")
run_block "$ROOT/cache/terra_openai_vitb16.npz" openai_vitb16 TerraIncognita & pids+=("$!")
run_block "$ROOT/cache/pacs_openclip_vitl14.npz" openclip_vitl14 PACS & pids+=("$!")
run_block "$ROOT/cache/officehome_openclip_vitl14.npz" openclip_vitl14 OfficeHome & pids+=("$!")
run_block "$ROOT/cache/terra_openclip_vitl14.npz" openclip_vitl14 TerraIncognita & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
find "$OUT" -name '*.summary.json' -type f | sort > "$OUT/summary_files.txt"
find "$OUT" -name '*.events.csv.gz' -type f | sort > "$OUT/event_files.txt"
wc -l "$OUT/summary_files.txt" "$OUT/event_files.txt"
exit "$failed"
