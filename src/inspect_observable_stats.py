from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from a2a.core import RunConfig, StreamController, make_budget_segments
from run_streams import stream_order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    loaded = np.load(args.cache, allow_pickle=False)
    features, labels, domains = loaded["features"], loaded["labels"], loaded["domains"]
    rows = []
    for stream in ("iid_random", "abrupt_domain", "class_correlated"):
        for seed in args.seeds:
            order = stream_order(labels, domains, stream, seed)
            budget = int(round(len(order) * .01))
            segments = make_budget_segments(len(order), budget)
            controller = StreamController(loaded["text_features"], RunConfig(num_classes=len(loaded["classes"])))
            for segment in segments:
                states = [controller.predict(features[order[position]]) for position in segment]
                predictions = np.asarray([state["prediction"] for state in states])
                rows.append({
                    "mean_entropy": float(np.mean([state["entropy"] for state in states])),
                    "switch_rate": float(np.mean(predictions[1:] != predictions[:-1])) if len(predictions) > 1 else 0.0,
                })
                for state in states:
                    controller.adapt_unlabeled(state)
    entropies = np.asarray([row["mean_entropy"] for row in rows])
    switches = np.asarray([row["switch_rate"] for row in rows])
    report = {
        "dataset": args.dataset, "backbone": args.backbone, "segments": len(rows),
        "mean_entropy_quantiles": dict(zip(["q10", "q25", "q50", "q75", "q90"], np.quantile(entropies, [.1,.25,.5,.75,.9]).tolist())),
        "switch_rate_quantiles": dict(zip(["q10", "q25", "q50", "q75", "q90"], np.quantile(switches, [.1,.25,.5,.75,.9]).tolist())),
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
