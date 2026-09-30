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
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_files = sorted(args.root.rglob("*.summary.json"))
    event_files = sorted(args.root.rglob("*.events.csv.gz"))
    event_by_parent_and_stem = {
        (path.parent, path.name.removesuffix(".events.csv.gz")): path for path in event_files
    }
    failures: list[str] = []
    action_totals = {"predict": 0, "adapt": 0, "ask": 0}
    run_ids: set[str] = set()
    total_queries = 0

    if len(summary_files) != args.expected_runs:
        failures.append(f"expected {args.expected_runs} summaries, found {len(summary_files)}")
    if len(event_files) != args.expected_runs:
        failures.append(f"expected {args.expected_runs} event logs, found {len(event_files)}")

    for summary_path in summary_files:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = str(summary["run_id"])
        scoped_id = f"{summary_path.parent.relative_to(args.root)}::{run_id}"
        if scoped_id in run_ids:
            failures.append(f"duplicate scoped run: {scoped_id}")
        run_ids.add(scoped_id)
        event_path = event_by_parent_and_stem.get((summary_path.parent, run_id))
        if event_path is None:
            failures.append(f"{scoped_id}: missing event file")
            continue
        if sha256(event_path) != summary["event_sha256"]:
            failures.append(f"{scoped_id}: SHA-256 mismatch")

        query_count = 0
        sample_count = 0
        previous_position = -1
        with gzip.open(event_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"position", "action", "adapted", "queried", "query_revealed_at"}
            if not required.issubset(reader.fieldnames or []):
                failures.append(f"{scoped_id}: missing event fields")
                continue
            for row in reader:
                sample_count += 1
                position = int(row["position"])
                if position != previous_position + 1:
                    failures.append(f"{scoped_id}:{position}: non-contiguous stream position")
                previous_position = position
                action = row["action"]
                adapted = int(row["adapted"])
                queried = int(row["queried"])
                if action not in action_totals:
                    failures.append(f"{scoped_id}:{position}: invalid action {action}")
                    continue
                action_totals[action] += 1
                query_count += queried
                if (action == "ask") != (queried == 1):
                    failures.append(f"{scoped_id}:{position}: Ask/query mismatch")
                if (action == "adapt") != (adapted == 1):
                    failures.append(f"{scoped_id}:{position}: Adapt/flag mismatch")
                if queried and adapted:
                    failures.append(f"{scoped_id}:{position}: both queried and pseudo-adapted")
                if queried and int(row["query_revealed_at"]) < position:
                    failures.append(f"{scoped_id}:{position}: pre-prediction label reveal")

        total_queries += query_count
        if sample_count != int(summary["samples"]):
            failures.append(f"{scoped_id}: {sample_count} events != {summary['samples']}")
        if query_count != int(summary["budget_count"]) or query_count != int(summary["query_count"]):
            failures.append(f"{scoped_id}: {query_count} queries violate exact budget")
        for key in (
            "online_accuracy", "zero_shot_accuracy", "gain_over_zero_shot",
            "worst_window_accuracy", "negative_adaptation_rate",
            "expected_calibration_error", "label_efficiency_pp_per_100_queries",
            "pseudo_memory_error_rate",
        ):
            if not math.isfinite(float(summary[key])):
                failures.append(f"{scoped_id}: non-finite {key}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "root": str(args.root),
        "expected_runs": args.expected_runs,
        "summary_files": len(summary_files),
        "event_files": len(event_files),
        "unique_scoped_runs": len(run_ids),
        "events": sum(action_totals.values()),
        "queries": total_queries,
        "action_totals": action_totals,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
