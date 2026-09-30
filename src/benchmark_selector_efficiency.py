"""Reproducible cached-feature latency benchmark for selector policies.

The benchmark intentionally excludes feature extraction, cache loading, stream
construction, and disk logging.  It times the common four-expert controller,
the policy decision, and verified/pseudo-memory updates on the same fixed
sample prefix for each policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from a2a.core import RunConfig, StreamController, make_budget_segments
from run_streams import score_for_strategy, stream_order


POLICIES = (
    "periodic",
    "entropy",
    "margin",
    "disagreement",
    "dynamic_entropy_online",
    "ask_or_adapt_v2",
)


def state_bytes(controller: StreamController) -> tuple[int, int]:
    memory = controller.memory
    mutable = sum(
        value.nbytes
        for value in (
            memory.labeled_sum,
            memory.labeled_count,
            memory.pseudo,
            memory.pseudo_count,
            controller.weights,
        )
    )
    including_text = mutable + memory.text.nbytes
    return int(mutable), int(including_text)


def timed_pass(
    features: np.ndarray,
    labels: np.ndarray,
    text_features: np.ndarray,
    policy: str,
    budget_percent: float,
    seed: int,
) -> tuple[int, int, int]:
    config = RunConfig(num_classes=len(text_features))
    controller = StreamController(text_features, config)
    budget = max(1, int(round(len(features) * budget_percent / 100.0)))
    segments = make_budget_segments(len(features), budget)
    segment_end = {int(part[-1]) for part in segments}
    rng = np.random.default_rng(seed + 104729)
    pending: list[dict[str, object]] = []
    entropy_history: list[float] = []
    segment_query_done = False
    query_count = 0

    start = time.perf_counter_ns()
    for position, (feature, label) in enumerate(zip(features, labels)):
        state = controller.predict(feature)
        score = score_for_strategy(policy, state, rng)
        pending.append({"state": state, "label": int(label), "score": score})

        if policy == "dynamic_entropy_online" and not segment_query_done:
            history = np.asarray(entropy_history, dtype=np.float64)
            threshold = 0.9 if len(history) < 30 else float(
                history.mean()
                + (1.96 if query_count / max(position + 1, 1) > budget / len(features) else 1.645)
                * history.std(ddof=0)
            )
            if float(state["entropy"]) > threshold or position in segment_end:
                chosen = pending[-1]
                controller.reveal_label(chosen["state"], int(chosen["label"]))
                for item in pending[:-1]:
                    controller.adapt_unlabeled(item["state"])
                pending.clear()
                query_count += 1
                segment_query_done = True
        entropy_history.append(float(state["entropy"]))

        if position in segment_end:
            if policy == "dynamic_entropy_online":
                for item in pending:
                    controller.adapt_unlabeled(item["state"])
                pending.clear()
                segment_query_done = False
            else:
                if policy == "periodic":
                    chosen = pending[-1]
                elif policy == "ask_or_adapt_v2":
                    predictions = np.asarray([item["state"]["prediction"] for item in pending])
                    switch_rate = float(np.mean(predictions[1:] != predictions[:-1])) if len(predictions) > 1 else 0.0
                    mean_entropy = float(np.mean([item["state"]["entropy"] for item in pending]))
                    if mean_entropy >= config.risk_entropy_threshold:
                        chosen = max(pending, key=lambda item: float(item["state"]["margin"]))
                    elif switch_rate < config.persistence_switch_rate:
                        chosen = pending[-1]
                    else:
                        chosen = max(pending, key=lambda item: float(item["score"]))
                else:
                    chosen = max(pending, key=lambda item: float(item["score"]))
                controller.reveal_label(chosen["state"], int(chosen["label"]))
                for item in pending:
                    if item is not chosen:
                        controller.adapt_unlabeled(item["state"])
                pending.clear()
                query_count += 1

    elapsed = time.perf_counter_ns() - start
    mutable, including_text = state_bytes(controller)
    if query_count != budget:
        raise AssertionError(f"{policy}: expected {budget} queries, got {query_count}")
    return elapsed, mutable, including_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--budget", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for cache_path in sorted(args.cache_dir.glob("*.npz")):
        loaded = np.load(cache_path, allow_pickle=False)
        order = stream_order(loaded["labels"], loaded["domains"], "iid_random", args.seed)
        order = order[: min(args.samples, len(order))]
        features = loaded["features"][order]
        labels = loaded["labels"][order]
        text_features = loaded["text_features"]
        for policy in POLICIES:
            # One unreported warm-up pass stabilizes imports, allocation, and CPU clocks.
            timed_pass(features, labels, text_features, policy, args.budget, args.seed)
            measurements = []
            mutable = including_text = 0
            for _ in range(args.repeats):
                elapsed, mutable, including_text = timed_pass(
                    features, labels, text_features, policy, args.budget, args.seed
                )
                measurements.append(elapsed / len(features) / 1e6)
            rows.append(
                {
                    "cache": cache_path.stem,
                    "policy": policy,
                    "samples": len(features),
                    "repeats": args.repeats,
                    "median_ms_per_sample": statistics.median(measurements),
                    "min_ms_per_sample": min(measurements),
                    "max_ms_per_sample": max(measurements),
                    "mutable_state_mib": mutable / (1024**2),
                    "state_including_text_mib": including_text / (1024**2),
                }
            )

    periodic = {row["cache"]: float(row["median_ms_per_sample"]) for row in rows if row["policy"] == "periodic"}
    for row in rows:
        row["relative_to_periodic"] = float(row["median_ms_per_sample"]) / periodic[row["cache"]]

    csv_path = args.output / "selector_efficiency_per_cache.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headline: dict[str, object] = {
        "scope": {
            "caches": len({row["cache"] for row in rows}),
            "samples_per_cache": args.samples,
            "repeats": args.repeats,
            "stream": "iid_random",
            "seed": args.seed,
            "budget_percent": args.budget,
            "timed_components": "four-expert controller, selector decision, and memory updates",
            "excluded_components": "cache loading, CLIP image encoding, stream permutation, and disk logging",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "policies": {},
    }
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        headline["policies"][policy] = {
            "macro_median_ms_per_sample": statistics.median(float(row["median_ms_per_sample"]) for row in group),
            "macro_mean_relative_to_periodic": statistics.mean(float(row["relative_to_periodic"]) for row in group),
            "range_ms_per_sample": [
                min(float(row["median_ms_per_sample"]) for row in group),
                max(float(row["median_ms_per_sample"]) for row in group),
            ],
        }
    headline["memory"] = {
        "mutable_state_mib_range": [
            min(float(row["mutable_state_mib"]) for row in rows),
            max(float(row["mutable_state_mib"]) for row in rows),
        ],
        "state_including_text_mib_range": [
            min(float(row["state_including_text_mib"]) for row in rows),
            max(float(row["state_including_text_mib"]) for row in rows),
        ],
    }
    (args.output / "selector_efficiency_headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
