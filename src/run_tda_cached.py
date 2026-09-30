"""Feature-cache replay of the published training-free TDA adapter.

The implementation follows Karmanov et al. (CVPR 2024): a positive cache keeps
the lowest-entropy pseudo-label examples per predicted class, while an optional
negative cache subtracts uncertain-class affinities. It uses no human labels,
gradients, or target labels and is therefore reported as a zero-label published
TTA reference, not as an exact-budget selector.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_streams import stream_order, expected_calibration_error


def entropy(probs):
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum() / np.log(len(probs)))


def cache_update(cache, pred, feature, loss, capacity, prob_map=None):
    item = (feature.copy(), float(loss), None if prob_map is None else prob_map.copy())
    cache.setdefault(int(pred), []).append(item)
    cache[int(pred)] = sorted(cache[int(pred)], key=lambda x: x[1])[:capacity]


def cache_logits(feature, cache, num_classes, alpha, beta, negative=False):
    keys, values = [], []
    for cls in sorted(cache):
        for key, _loss, prob_map in cache[cls]:
            keys.append(key)
            if negative:
                # TDA's negative mask keeps uncertain class probabilities.
                values.append(((prob_map > 0.03) & (prob_map < 1.0)).astype(np.float64))
            else:
                one = np.zeros(num_classes, dtype=np.float64); one[cls] = 1.0
                values.append(one)
    if not keys:
        return np.zeros(num_classes, dtype=np.float64)
    affinity = np.asarray(keys) @ feature
    weights = np.exp(-beta * (1.0 - affinity))
    return alpha * (weights[:, None] * np.asarray(values)).sum(axis=0)


def run_one(cache, stream, seed, pos_capacity=3, pos_alpha=2.0, pos_beta=5.0,
            neg_capacity=2, neg_alpha=0.117, neg_beta=1.0):
    features, labels, domains, text = cache["features"], cache["labels"], cache["domains"], cache["text_features"]
    order = stream_order(labels, domains, stream, seed)
    classes = text.shape[0]
    positive, negative = {}, {}
    correct, zero_correct, conf = [], [], []
    for idx in order:
        feature = np.asarray(features[idx], dtype=np.float64)
        base = 100.0 * feature @ text.T
        probs = np.exp(base - base.max()); probs /= probs.sum()
        pred = int(base.argmax())
        loss = entropy(probs)
        # This is the official TDA order: pseudo-cache update precedes the
        # adapted prediction and uses only the model's current pseudo-label.
        cache_update(positive, pred, feature, loss, pos_capacity)
        if 0.2 < loss < 0.5:
            cache_update(negative, pred, feature, loss, neg_capacity, probs)
        adapted = base + cache_logits(feature, positive, classes, pos_alpha, pos_beta)
        adapted -= cache_logits(feature, negative, classes, neg_alpha, neg_beta, negative=True)
        final_pred = int(adapted.argmax())
        correct.append(float(final_pred == int(labels[idx])))
        zero_correct.append(float(pred == int(labels[idx])))
        conf.append(float(np.max(np.exp(adapted - adapted.max()) / np.exp(adapted - adapted.max()).sum())))
    correct = np.asarray(correct); zero_correct = np.asarray(zero_correct); conf = np.asarray(conf)
    window = min(500, len(correct))
    rolling = np.convolve(correct, np.ones(window) / window, mode="valid")
    return {
        "strategy": "TDA",
        "stream": stream,
        "seed": seed,
        "samples": len(correct),
        "budget_percent": 0.0,
        "query_count": 0,
        "online_accuracy": float(correct.mean()),
        "zero_shot_accuracy": float(zero_correct.mean()),
        "gain_over_zero_shot": float(correct.mean() - zero_correct.mean()),
        "worst_window_accuracy": float(rolling.min()),
        "negative_adaptation_rate": float(((zero_correct == 1) & (correct == 0)).mean()),
        "expected_calibration_error": expected_calibration_error(conf, correct),
        "positive_cache_size": int(sum(len(v) for v in positive.values())),
        "negative_cache_size": int(sum(len(v) for v in negative.values())),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache-root", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--seeds", nargs="+", type=int, default=[10])
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.cache_root.glob("*.npz")):
        loaded = np.load(path, allow_pickle=False); cache = {k: loaded[k] for k in loaded.files}
        for stream in ("iid_random", "abrupt_domain", "class_correlated"):
            for seed in args.seeds:
                row = run_one(cache, stream, seed); row["cache"] = path.stem; rows.append(row)
                print(path.stem, stream, seed, f"{row['online_accuracy']:.6f}", flush=True)
    with (args.output_dir / "tda_cached_runs.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} TDA rows")


if __name__ == "__main__":
    main()
