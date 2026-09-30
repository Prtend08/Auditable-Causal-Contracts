from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.root.rglob("*.summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["pseudo_confidence"] = float(row["config"]["pseudo_confidence"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    table = frame.groupby("pseudo_confidence").agg(
        runs=("run_id", "count"),
        online_accuracy=("online_accuracy", "mean"),
        gain_over_zero_shot=("gain_over_zero_shot", "mean"),
        negative_adaptation_rate=("negative_adaptation_rate", "mean"),
        worst_window_accuracy=("worst_window_accuracy", "mean"),
    ).reset_index()
    table = table.sort_values(
        ["online_accuracy", "negative_adaptation_rate", "worst_window_accuracy"],
        ascending=[False, True, False],
    )
    selected = float(table.iloc[0].pseudo_confidence)
    report = {
        "status": "PASS" if len(frame) == 270 and set(table.runs) == {90} else "FAIL",
        "selection_rule": "accuracy desc, negative adaptation asc, worst-window accuracy desc",
        "selected_pseudo_confidence": selected,
        "candidates": table.to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
