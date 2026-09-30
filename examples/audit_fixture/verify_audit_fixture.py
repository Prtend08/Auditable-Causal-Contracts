"""Verify the included deterministic event-log audit fixture.

This intentionally uses only the Python standard library so that a reviewer can
run it before installing model or dataset dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVENTS = HERE / "events.csv"
SUMMARY = HERE / "summary.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    digest = sha256(EVENTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    with EVENTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "event log is empty"
    assert [int(r["position"]) for r in rows] == list(range(len(rows)))
    queries = [r for r in rows if r["queried"] == "1"]
    assert len(rows) == int(summary["samples"])
    assert len(queries) == int(summary["query_count"])
    accuracy = sum(int(r["correct"]) for r in rows) / len(rows)
    assert abs(accuracy - float(summary["online_accuracy"])) < 1e-12
    assert digest == summary["event_sha256"], (digest, summary["event_sha256"])
    print(json.dumps({
        "event_sha256": digest,
        "events": len(rows),
        "queries": len(queries),
        "online_accuracy": accuracy,
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
