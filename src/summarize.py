from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.run_index.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_runs.csv", index=False)
    metrics = [
        "online_accuracy",
        "gain_over_zero_shot",
        "worst_window_accuracy",
        "negative_adaptation_rate",
        "expected_calibration_error",
        "label_efficiency_pp_per_100_queries",
        "pseudo_memory_error_rate",
        "query_count",
    ]
    grouped = frame.groupby(["stream", "budget_percent", "strategy"], as_index=False)[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(filter(None, col)).rstrip("_") for col in grouped.columns]
    grouped.to_csv(args.output_dir / "aggregate_mean_std.csv", index=False)

    tail_rows = []
    for keys, group in frame.groupby(["stream", "budget_percent", "strategy"]):
        values = np.sort(group["online_accuracy"].to_numpy())
        count = max(1, int(np.ceil(0.2 * len(values))))
        tail_rows.append(
            {
                "stream": keys[0],
                "budget_percent": keys[1],
                "strategy": keys[2],
                "expected_tail_accuracy_20pct": float(values[:count].mean()),
                "runs": len(values),
            }
        )
    pd.DataFrame(tail_rows).to_csv(args.output_dir / "expected_tail_accuracy.csv", index=False)
    print(f"Summarized {len(frame)} runs into {args.output_dir}")


if __name__ == "__main__":
    main()
