#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/a2a
PY="$ROOT/envs/a2a/bin/python"
PROJECT="$ROOT/project"
CACHE="$ROOT/cache/pacs_openai_vitb16.npz"
OUT="$ROOT/results/confirmation_v2"
LOG="$ROOT/logs"
pids=()

rm -rf "$OUT"
mkdir -p "$OUT" "$LOG"

for stream in iid_random abrupt_domain class_correlated; do
  mkdir -p "$OUT/$stream"
  PYTHONPATH="$PROJECT/src" "$PY" "$PROJECT/src/run_streams.py" \
    --cache "$CACHE" \
    --output-dir "$OUT/$stream" \
    --streams "$stream" \
    --seeds 5 6 7 8 9 \
    --budgets 0.1 0.5 1 2 5 \
    --strategies random periodic entropy margin disagreement ask_or_adapt \
    > "$LOG/confirmation_${stream}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

find "$OUT" -mindepth 2 -name run_index.jsonl -print0 \
  | sort -z \
  | xargs -0 cat > "$OUT/run_index.jsonl"

PYTHONPATH="$PROJECT/src" "$PY" "$PROJECT/src/summarize.py" \
  --run-index "$OUT/run_index.jsonl" \
  --output-dir "$OUT/summary"

PYTHONPATH="$PROJECT/src" "$PY" "$PROJECT/src/audit_phase1.py" \
  --root "$OUT" \
  --output "$OUT/summary/audit_report.json"

echo "CONFIRMATION_COMPLETE"
