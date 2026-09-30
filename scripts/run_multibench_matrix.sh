#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/a2a
PROJECT="$ROOT/project"
PY="$ROOT/envs/a2a/bin/python"
RUN="$PROJECT/src/run_streams.py"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

main_run() {
  local cache="$1"
  local output="$2"
  mkdir -p "$output"
  "$PY" "$RUN" --cache "$cache" --output-dir "$output" \
    --streams iid_random abrupt_domain class_correlated \
    --seeds 5 6 7 8 9 --budgets 1.0 \
    --strategies random periodic entropy margin disagreement ask_or_adapt \
    >"$output/console.log" 2>&1
}

ablation_run() {
  local cache="$1"
  local output="$2"
  mkdir -p "$output"
  "$PY" "$RUN" --cache "$cache" --output-dir "$output" \
    --streams iid_random abrupt_domain class_correlated \
    --seeds 5 6 7 8 9 --budgets 1.0 \
    --strategies ask_or_adapt a2a_no_gate a2a_no_entropy a2a_no_disagreement a2a_no_coverage \
    >"$output/console.log" 2>&1
}

pids=()
main_run "$ROOT/cache/officehome_openai_vitb16.npz" "$ROOT/results/multibench_v3/openai_vitb16/OfficeHome" & pids+=("$!")
main_run "$ROOT/cache/terra_openai_vitb16.npz" "$ROOT/results/multibench_v3/openai_vitb16/TerraIncognita" & pids+=("$!")
main_run "$ROOT/cache/pacs_openclip_vitl14.npz" "$ROOT/results/multibench_v3/openclip_vitl14/PACS" & pids+=("$!")
main_run "$ROOT/cache/officehome_openclip_vitl14.npz" "$ROOT/results/multibench_v3/openclip_vitl14/OfficeHome" & pids+=("$!")
main_run "$ROOT/cache/terra_openclip_vitl14.npz" "$ROOT/results/multibench_v3/openclip_vitl14/TerraIncognita" & pids+=("$!")
ablation_run "$ROOT/cache/pacs_openai_vitb16.npz" "$ROOT/results/multibench_v3/ablations/PACS" & pids+=("$!")
ablation_run "$ROOT/cache/officehome_openai_vitb16.npz" "$ROOT/results/multibench_v3/ablations/OfficeHome" & pids+=("$!")
ablation_run "$ROOT/cache/terra_openai_vitb16.npz" "$ROOT/results/multibench_v3/ablations/TerraIncognita" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one matrix job failed" >&2
  exit 1
fi

find "$ROOT/results/multibench_v3" -name '*.summary.json' -type f | sort > "$ROOT/results/multibench_v3/summary_files.txt"
find "$ROOT/results/multibench_v3" -name '*.events.csv.gz' -type f | sort > "$ROOT/results/multibench_v3/event_files.txt"
wc -l "$ROOT/results/multibench_v3/summary_files.txt" "$ROOT/results/multibench_v3/event_files.txt"
