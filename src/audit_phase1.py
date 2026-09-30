from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = args.root / "run_index.jsonl"
    summaries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    failures: list[str] = []
    action_totals = {"predict": 0, "adapt": 0, "ask": 0}
    seen_run_ids: set[str] = set()

    for summary in summaries:
        run_id = summary["run_id"]
        if run_id in seen_run_ids:
            failures.append(f"duplicate run id: {run_id}")
        seen_run_ids.add(run_id)
        event_files = list(args.root.glob(f"*/{run_id}.events.csv.gz"))
        if len(event_files) != 1:
            failures.append(f"{run_id}: expected one event file, found {len(event_files)}")
            continue
        event_path = event_files[0]
        if sha256(event_path) != summary["event_sha256"]:
            failures.append(f"{run_id}: event SHA-256 mismatch")

        query_count = 0
        sample_count = 0
        with gzip.open(event_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"position", "label", "prediction", "action", "adapted", "queried", "query_revealed_at"}
            if not required.issubset(reader.fieldnames or []):
                failures.append(f"{run_id}: missing required event fields")
                continue
            for row in reader:
                sample_count += 1
                position = int(row["position"])
                action = row["action"]
                adapted = int(row["adapted"])
                queried = int(row["queried"])
                if action not in action_totals:
                    failures.append(f"{run_id}:{position}: invalid action {action}")
                    continue
                action_totals[action] += 1
                query_count += queried
                if (action == "ask") != (queried == 1):
                    failures.append(f"{run_id}:{position}: Ask/query mismatch")
                if (action == "adapt") != (adapted == 1):
                    failures.append(f"{run_id}:{position}: Adapt flag mismatch")
                if queried and adapted:
                    failures.append(f"{run_id}:{position}: sample both queried and pseudo-adapted")
                if queried and int(row["query_revealed_at"]) < position:
                    failures.append(f"{run_id}:{position}: label revealed before prediction")

        if sample_count != summary["samples"]:
            failures.append(f"{run_id}: {sample_count} events != {summary['samples']} samples")
        if query_count != summary["budget_count"] or query_count != summary["query_count"]:
            failures.append(f"{run_id}: query count {query_count} violates budget")
        metric_keys = (
            "online_accuracy", "zero_shot_accuracy", "gain_over_zero_shot",
            "worst_window_accuracy", "negative_adaptation_rate",
            "expected_calibration_error", "label_efficiency_pp_per_100_queries",
            "pseudo_memory_error_rate",
        )
        for key in metric_keys:
            if not math.isfinite(float(summary[key])):
                failures.append(f"{run_id}: non-finite metric {key}")

    expected_conditions = 3 * 5 * 5 * 6
    if len(summaries) != expected_conditions:
        failures.append(f"expected {expected_conditions} summaries, found {len(summaries)}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "runs": len(summaries),
        "expected_runs": expected_conditions,
        "unique_run_ids": len(seen_run_ids),
        "events": sum(action_totals.values()),
        "action_totals": action_totals,
        "failures": failures,
        "run_index_sha256": sha256(index),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
