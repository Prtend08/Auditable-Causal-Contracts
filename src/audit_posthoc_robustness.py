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
    summaries = sorted(args.root.rglob("*.summary.json"))
    events = sorted(args.root.rglob("*.events.csv.gz"))
    failures, total_events, total_queries, noisy_queries = [], 0, 0, 0
    if len(summaries) != 630:
        failures.append(f"expected 630 summaries, found {len(summaries)}")
    if len(events) != 180:
        failures.append(f"expected 180 event logs, found {len(events)}")
    seen = set()
    for path in summaries:
        row = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(args.root)
        scoped = (rel.parts[0], rel.parts[1], rel.parts[2], row["run_id"])
        if scoped in seen:
            failures.append(f"duplicate scoped run {scoped}")
        seen.add(scoped)
        if row["query_count"] != row["budget_count"]:
            failures.append(f"budget mismatch {path}")
        if not all(math.isfinite(float(row[name])) for name in (
            "online_accuracy", "worst_window_accuracy", "negative_adaptation_rate",
            "expected_calibration_error", "realized_oracle_noise"
        )):
            failures.append(f"non-finite metric {path}")
        if rel.parts[2] == "budget":
            if row["oracle_noise"] != 0.0 or row["noisy_oracle_count"] != 0 or row["event_sha256"] is not None:
                failures.append(f"invalid summary-only clean run {path}")
            continue
        event_path = path.with_name(path.name.replace(".summary.json", ".events.csv.gz"))
        if not event_path.exists():
            failures.append(f"missing event log {event_path}")
            continue
        if sha256(event_path) != row["event_sha256"]:
            failures.append(f"event hash mismatch {event_path}")
        run_events = run_queries = run_noise = 0
        with gzip.open(event_path, "rt", newline="", encoding="utf-8") as handle:
            for event in csv.DictReader(handle):
                run_events += 1
                queried = int(event["queried"])
                adapted = int(event["adapted"])
                if queried and adapted:
                    failures.append(f"ask/adapt overlap {event_path}:{event['position']}")
                if queried:
                    run_queries += 1
                    if event["oracle_label"] == "" or event["query_revealed_at"] == "":
                        failures.append(f"missing delayed oracle fields {event_path}:{event['position']}")
                    if int(event["query_revealed_at"]) < int(event["position"]):
                        failures.append(f"early reveal {event_path}:{event['position']}")
                    run_noise += int(event["oracle_label"] != event["label"])
                elif event["oracle_label"] != "":
                    failures.append(f"oracle label on non-query {event_path}:{event['position']}")
        if run_events != row["samples"] or run_queries != row["query_count"] or run_noise != row["noisy_oracle_count"]:
            failures.append(f"summary/event mismatch {event_path}")
        total_events += run_events; total_queries += run_queries; noisy_queries += run_noise
    report = {
        "status": "PASS" if not failures else "FAIL",
        "summary_files": len(summaries), "event_files": len(events),
        "unique_scoped_runs": len(seen), "noise_events": total_events,
        "noise_queries": total_queries, "noisy_queries": noisy_queries,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
