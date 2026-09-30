"""Recompute worst-window accuracy for several window sizes from audit logs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def read_correct(path: Path) -> np.ndarray:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        values = [int(row["correct"]) for row in csv.DictReader(handle)]
    return np.asarray(values, dtype=np.float64)


def summarize(root: Path, windows: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = "*__ask_or_adapt_v2.events.csv.gz"
    for path in sorted(root.rglob(pattern)):
        parts = path.parts
        try:
            backbone = parts[-3]
            dataset = parts[-2]
        except IndexError as exc:
            raise ValueError(f"unexpected log path: {path}") from exc
        name = path.name.removesuffix(".events.csv.gz")
        fields = name.split("__")
        stream = fields[0]
        seed = int(fields[1].removeprefix("seed"))
        correctness = read_correct(path)
        for window in windows:
            if window > len(correctness):
                raise ValueError(f"window {window} exceeds stream length {len(correctness)}")
            rolling = np.convolve(correctness, np.ones(window) / window, mode="valid")
            rows.append({
                "dataset": dataset,
                "backbone": backbone,
                "stream": stream,
                "seed": seed,
                "window": window,
                "worst_window_accuracy": float(rolling.min()),
                "samples": int(len(correctness)),
                "source": str(path),
            })
    if not rows:
        raise FileNotFoundError(f"no Ask-or-Adapt v2 logs under {root}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", nargs="+", type=int, default=[250, 500, 1000])
    args = parser.parse_args()
    rows = summarize(args.logs_root, args.windows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "window_sensitivity_per_run.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: list[dict[str, object]] = []
    keys = sorted({(r["dataset"], r["backbone"], r["window"]) for r in rows})
    for dataset, backbone, window in keys:
        values = [r["worst_window_accuracy"] for r in rows if (r["dataset"], r["backbone"], r["window"]) == (dataset, backbone, window)]
        grouped.append({
            "dataset": dataset,
            "backbone": backbone,
            "window": window,
            "runs": len(values),
            "mean_accuracy": float(np.mean(values)),
            "std_accuracy": float(np.std(values, ddof=1)),
        })
    summary_path = args.output_dir / "window_sensitivity_summary.json"
    summary_path.write_text(json.dumps({"runs": len(rows), "summary": grouped}, indent=2), encoding="utf-8")
    with (args.output_dir / "window_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    print(json.dumps({"runs": len(rows), "groups": len(grouped), "summary": grouped}, indent=2))


if __name__ == "__main__":
    main()
