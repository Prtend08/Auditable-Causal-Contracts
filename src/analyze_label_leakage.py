"""Quantify score-time oracle leakage from held-out Ask-or-Adapt event logs.

The counterfactual changes only the scored correctness of queried samples to
one, as if the revealed label were allowed to repair the prediction that
triggered the query. Future predictions and controller state remain unchanged,
so the difference isolates the optimistic bias caused by non-prequential
scoring rather than claiming a deployable method.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, draws: int = 20_000, seed: int = 20260814) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1_000):
        stop = min(draws, start + 1_000)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def parse_scope(path: Path, root: Path) -> dict[str, object]:
    rel = path.relative_to(root)
    backbone, dataset = rel.parts[:2]
    stem = path.name.removesuffix(".events.csv.gz")
    stream, seed_token, budget_token, strategy = stem.split("__", 3)
    return {
        "backbone": backbone,
        "dataset": dataset,
        "stream": stream,
        "seed": int(seed_token.removeprefix("seed")),
        "budget_percent": float(budget_token.removeprefix("budget")),
        "strategy": strategy,
        "event_file": rel.as_posix(),
    }


def analyze_file(path: Path, root: Path) -> dict[str, object]:
    summary_path = path.with_name(path.name.replace(".events.csv.gz", ".summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != summary["event_sha256"]:
        raise AssertionError(f"event hash mismatch: {path}")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        frame = pd.read_csv(handle, usecols=["correct", "queried"])
    causal = frame["correct"].to_numpy(dtype=np.float64)
    queried = frame["queried"].to_numpy(dtype=bool)
    leaked = np.maximum(causal, queried.astype(np.float64))
    window = min(500, len(frame))
    causal_worst = float(pd.Series(causal).rolling(window).mean().min())
    leaked_worst = float(pd.Series(leaked).rolling(window).mean().min())
    if not np.isclose(causal.mean(), summary["online_accuracy"], atol=1e-12):
        raise AssertionError(f"online-accuracy mismatch: {path}")
    if not np.isclose(causal_worst, summary["worst_window_accuracy"], atol=1e-12):
        raise AssertionError(f"worst-window mismatch: {path}")
    query_errors = int(np.sum(queried & (causal == 0)))
    result = parse_scope(path, root)
    result.update(
        {
            "event_sha256": digest,
            "samples": int(len(frame)),
            "queries": int(queried.sum()),
            "queried_errors": query_errors,
            "queried_error_rate": float(query_errors / max(int(queried.sum()), 1)),
            "causal_online_accuracy": float(causal.mean()),
            "leaky_online_accuracy": float(leaked.mean()),
            "online_inflation_pp": float(100 * (leaked.mean() - causal.mean())),
            "causal_worst500_accuracy": causal_worst,
            "leaky_worst500_accuracy": leaked_worst,
            "worst500_inflation_pp": float(100 * (leaked_worst - causal_worst)),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("*__ask_or_adapt_v2.events.csv.gz"))
    if len(paths) != 90:
        raise ValueError(f"expected 90 held-out event logs, found {len(paths)}")
    frame = pd.DataFrame(analyze_file(path, args.root) for path in paths)
    if not (frame["queries"] == np.maximum(1, np.rint(frame["samples"] * frame["budget_percent"] / 100)).astype(int)).all():
        raise AssertionError("query-count audit failed")

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "label_leakage_per_condition.csv", index=False)
    groups = []
    for columns in (["dataset"], ["backbone"], ["stream"], ["dataset", "backbone"]):
        for keys, part in frame.groupby(columns):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {name: value for name, value in zip(columns, keys)}
            row.update(
                {
                    "grouping": "+".join(columns),
                    "conditions": int(len(part)),
                    "online_inflation_pp_mean": float(part["online_inflation_pp"].mean()),
                    "online_inflation_pp_std": float(part["online_inflation_pp"].std(ddof=1)),
                    "worst500_inflation_pp_mean": float(part["worst500_inflation_pp"].mean()),
                    "queried_error_rate_mean": float(part["queried_error_rate"].mean()),
                }
            )
            groups.append(row)
    pd.DataFrame(groups).to_csv(args.output / "label_leakage_subgroups.csv", index=False)

    online = frame["online_inflation_pp"].to_numpy()
    worst = frame["worst500_inflation_pp"].to_numpy()
    online_ci = bootstrap_ci(online)
    worst_ci = bootstrap_ci(worst, seed=20260815)
    headline = {
        "conditions": int(len(frame)),
        "events": int(frame["samples"].sum()),
        "queries": int(frame["queries"].sum()),
        "queried_errors": int(frame["queried_errors"].sum()),
        "queried_error_rate": float(frame["queried_errors"].sum() / frame["queries"].sum()),
        "causal_online_accuracy_percent": float(100 * frame["causal_online_accuracy"].mean()),
        "leaky_online_accuracy_percent": float(100 * frame["leaky_online_accuracy"].mean()),
        "online_inflation_pp_mean": float(online.mean()),
        "online_inflation_pp_ci95": list(online_ci),
        "causal_worst500_accuracy_percent": float(100 * frame["causal_worst500_accuracy"].mean()),
        "leaky_worst500_accuracy_percent": float(100 * frame["leaky_worst500_accuracy"].mean()),
        "worst500_inflation_pp_mean": float(worst.mean()),
        "worst500_inflation_pp_ci95": list(worst_ci),
        "counterfactual": "queried current predictions are scored as correct; future predictions and state are unchanged",
    }
    (args.output / "label_leakage_headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
