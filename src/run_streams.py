from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from a2a.core import EPS, RunConfig, StreamController, make_budget_segments


def stream_order(labels: np.ndarray, domains: np.ndarray, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    if kind == "iid_random":
        rng.shuffle(indices)
        return indices
    if kind == "abrupt_domain":
        domain_order = rng.permutation(np.unique(domains))
        chunks = []
        for domain in domain_order:
            part = indices[domains == domain].copy()
            rng.shuffle(part)
            chunks.append(part)
        return np.concatenate(chunks)
    if kind == "class_correlated":
        class_order = rng.permutation(np.unique(labels))
        chunks = []
        for label in class_order:
            part = indices[labels == label].copy()
            rng.shuffle(part)
            chunks.append(part)
        return np.concatenate(chunks)
    raise KeyError(kind)


def score_for_strategy(strategy: str, state: Dict[str, object], rng: np.random.Generator) -> float:
    if strategy in {"zero_shot", "ask_only"}:
        return 0.0
    if strategy == "random":
        return float(rng.random())
    if strategy == "periodic":
        return 0.0
    if strategy == "entropy":
        return float(state["entropy"])
    if strategy == "dynamic_entropy_online":
        return float(state["entropy"])
    if strategy == "margin":
        return float(state["margin"])
    if strategy == "disagreement":
        return float(state["disagreement"])
    if strategy in {"ask_or_adapt", "ask_or_adapt_v2", "a2a_no_gate", "v2_no_gate"}:
        return float(state["query_value"])
    if strategy in {"a2a_no_entropy", "v2_no_entropy"}:
        return float(max(float(state["disagreement"]) * float(state["coverage"]), 0.0) ** 0.5)
    if strategy in {"a2a_no_disagreement", "v2_no_disagreement"}:
        return float(max(float(state["entropy"]) * float(state["coverage"]), 0.0) ** 0.5)
    if strategy in {"a2a_no_coverage", "v2_no_coverage"}:
        return float(max(float(state["entropy"]) * float(state["disagreement"]), 0.0) ** 0.5)
    raise KeyError(strategy)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def serialize(values) -> str:
    return json.dumps(np.asarray(values).tolist(), separators=(",", ":"))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            mask |= confidence == 0.0
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def run_one(
    cache: Dict[str, np.ndarray],
    output_dir: Path,
    stream_kind: str,
    seed: int,
    budget_percent: float,
    strategy: str,
    config: RunConfig,
    write_events: bool = True,
    oracle_noise: float = 0.0,
    oracle_noise_mode: str = "symmetric",
    confusable_temperature: float = 0.10,
) -> Dict[str, object]:
    features = cache["features"]
    labels = cache["labels"]
    domains = cache["domains"]
    paths = cache["paths"]
    text_features = cache["text_features"]
    order = stream_order(labels, domains, stream_kind, seed)
    # Zero-shot is an explicit no-query anchor, regardless of the caller's
    # requested budget.  All other policies retain the exact-budget contract.
    if strategy == "zero_shot":
        budget_percent = 0.0
    budget = int(round(len(order) * budget_percent / 100.0))
    budget = max(1, budget) if budget_percent > 0 else 0
    segments = make_budget_segments(len(order), budget) if budget else []
    segment_end_to_id = {int(part[-1]): i for i, part in enumerate(segments)}

    controller = StreamController(text_features, config)
    rng = np.random.default_rng(seed + 104729)
    oracle_rng = np.random.default_rng(seed + 32452843)
    pending: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []
    pseudo_total = 0
    pseudo_errors = 0
    entropy_history: List[float] = []
    segment_query_done = False
    online_query_count = 0
    noisy_oracle_count = 0

    if not 0.0 <= oracle_noise <= 1.0:
        raise ValueError(f"oracle_noise must be in [0, 1], got {oracle_noise}")
    if oracle_noise_mode not in {"symmetric", "class_confusable"}:
        raise ValueError(f"unknown oracle_noise_mode: {oracle_noise_mode}")

    # A deterministic class-confusable oracle model.  It uses frozen text
    # prototype similarity, so the noise mechanism is fixed before the stream
    # starts and never inspects the current label during prediction.
    confusable = None
    if oracle_noise_mode == "class_confusable":
        similarity = np.asarray(text_features, dtype=np.float64) @ np.asarray(text_features, dtype=np.float64).T
        similarity -= np.eye(similarity.shape[0]) * 1e6
        similarity = (similarity - similarity.max(axis=1, keepdims=True)) / max(confusable_temperature, 1e-6)
        confusable = np.exp(similarity)
        confusable /= np.maximum(confusable.sum(axis=1, keepdims=True), EPS)

    def oracle_response(true_label: int) -> int:
        """Return a delayed expert label under symmetric class-conditional noise."""
        nonlocal noisy_oracle_count
        if oracle_noise == 0.0 or oracle_rng.random() >= oracle_noise:
            return true_label
        if oracle_noise_mode == "class_confusable":
            draw = int(oracle_rng.choice(config.num_classes, p=confusable[true_label]))
            # Numerical safety: the diagonal is zero, but reject it explicitly
            # so a malformed cache cannot return the true label as a flip.
            while draw == true_label:
                draw = int(oracle_rng.choice(config.num_classes, p=confusable[true_label]))
        else:
            draw = int(oracle_rng.integers(config.num_classes - 1))
            draw += int(draw >= true_label)
        noisy_oracle_count += 1
        return draw

    def apply_unlabeled(items: List[Dict[str, object]], excluded=None) -> None:
        nonlocal pseudo_total, pseudo_errors
        for item in items:
            if item is excluded:
                continue
            pseudo_label = controller.adapt_unlabeled(item["state"])
            item_record = item["record"]
            if pseudo_label is None:
                item_record["action"] = "predict"
                continue
            item_record["action"] = "adapt"
            item_record["adapted"] = 1
            pseudo_total += 1
            pseudo_errors += int(pseudo_label != int(item["label"]))

    for position, original_index in enumerate(order):
        state = controller.predict(features[original_index])
        true_label = int(labels[original_index])
        is_correct = int(state["prediction"] == true_label)
        zero_shot_prediction = int(np.argmax(features[original_index] @ text_features.T))
        zero_shot_correct = int(zero_shot_prediction == true_label)
        score = score_for_strategy(strategy, state, rng)
        record = {
            "position": position,
            "original_index": int(original_index),
            "path": str(paths[original_index]),
            "domain": int(domains[original_index]),
            "label": true_label,
            "oracle_label": "",
            "prediction": int(state["prediction"]),
            "correct": is_correct,
            "zero_shot_prediction": zero_shot_prediction,
            "zero_shot_correct": zero_shot_correct,
            "confidence": float(state["confidence"]),
            "entropy": float(state["entropy"]),
            "margin": float(state["margin"]),
            "disagreement": float(state["disagreement"]),
            "coverage": float(state["coverage"]),
            "query_value": float(state["query_value"]),
            "selection_score": score,
            "selection_mode": "",
            "action": "pending",
            "adapted": 0,
            "queried": 0,
            "query_revealed_at": "",
        }
        records.append(record)
        pending.append({"state": state, "label": true_label, "record": record, "score": score})

        if strategy == "dynamic_entropy_online" and not segment_query_done:
            history = np.asarray(entropy_history, dtype=np.float64)
            if len(history) < 30:
                threshold = 0.9
            else:
                # Published TAPS selection rule: an online mean-plus-z-standard-
                # deviation entropy threshold, tightened when query rate is high.
                observed_rate = online_query_count / max(position + 1, 1)
                target_rate = budget / max(len(order), 1)
                z_value = 1.96 if observed_rate > target_rate else 1.645
                threshold = float(history.mean() + z_value * history.std(ddof=0))
            must_close = position in segment_end_to_id
            if float(state["entropy"]) > threshold or must_close:
                chosen = pending[-1]
                revealed_label = oracle_response(int(chosen["label"]))
                controller.reveal_label(chosen["state"], revealed_label)
                chosen_record = chosen["record"]
                chosen_record["action"] = "ask"
                chosen_record["selection_mode"] = "dynamic_entropy_online"
                chosen_record["queried"] = 1
                chosen_record["oracle_label"] = revealed_label
                chosen_record["query_revealed_at"] = position
                online_query_count += 1
                apply_unlabeled(pending, excluded=chosen)
                pending.clear()
                segment_query_done = True
        entropy_history.append(float(state["entropy"]))

        if position in segment_end_to_id:
            if strategy == "dynamic_entropy_online":
                apply_unlabeled(pending)
                pending.clear()
                segment_query_done = False
            else:
                selection_mode = strategy
                if strategy in {"periodic", "ask_only"}:
                    chosen = pending[-1]
                elif strategy in {
                    "ask_or_adapt", "ask_or_adapt_v2", "a2a_no_entropy",
                    "a2a_no_disagreement", "a2a_no_coverage", "v2_no_gate",
                    "v2_no_entropy", "v2_no_disagreement", "v2_no_coverage",
                }:
                    predictions = np.asarray([item["state"]["prediction"] for item in pending])
                    switch_rate = float(np.mean(predictions[1:] != predictions[:-1])) if len(predictions) > 1 else 0.0
                    mean_entropy = float(np.mean([item["state"]["entropy"] for item in pending]))
                    is_v2 = strategy == "ask_or_adapt_v2" or strategy.startswith("v2_")
                    if is_v2 and mean_entropy >= config.risk_entropy_threshold:
                        chosen = max(pending, key=lambda row: float(row["state"]["margin"]))
                        selection_mode = "risk_margin"
                    elif strategy != "v2_no_gate" and switch_rate < config.persistence_switch_rate:
                        chosen = pending[-1]
                        selection_mode = "persistence_periodic"
                    else:
                        chosen = max(pending, key=lambda row: row["score"])
                        selection_mode = "query_value"
                else:
                    chosen = max(pending, key=lambda row: row["score"])
                revealed_label = oracle_response(int(chosen["label"]))
                controller.reveal_label(chosen["state"], revealed_label)
                chosen_record = chosen["record"]
                chosen_record["action"] = "ask"
                chosen_record["selection_mode"] = selection_mode
                chosen_record["queried"] = 1
                chosen_record["oracle_label"] = revealed_label
                chosen_record["query_revealed_at"] = position

                # Delay both supervised and pseudo-label updates until segment close.
                # Thus current-segment interventions can affect future predictions only,
                # and the queried sample is never also used as a pseudo-label.
                if strategy != "ask_only":
                    apply_unlabeled(pending, excluded=chosen)
                pending.clear()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{stream_kind}__seed{seed}__budget{budget_percent:g}__{strategy}"
    event_path = output_dir / f"{run_id}.events.csv.gz"
    if write_events:
        with gzip.open(event_path, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    correctness = np.asarray([row["correct"] for row in records], dtype=np.float64)
    confidence = np.asarray([row["confidence"] for row in records], dtype=np.float64)
    window = min(500, len(correctness))
    rolling = np.convolve(correctness, np.ones(window) / window, mode="valid")
    zs_correct = np.asarray([row["zero_shot_correct"] for row in records], dtype=np.float64)
    negative_flips = (zs_correct == 1.0) & (correctness == 0.0)
    gain = float(correctness.mean() - zs_correct.mean())
    summary = {
        "run_id": run_id,
        "stream": stream_kind,
        "seed": seed,
        "budget_percent": budget_percent,
        "budget_count": budget,
        "query_count": int(sum(int(row["queried"]) for row in records)),
        "oracle_noise": oracle_noise,
        "oracle_noise_mode": oracle_noise_mode,
        "noisy_oracle_count": noisy_oracle_count,
        "realized_oracle_noise": float(noisy_oracle_count / max(budget, 1)),
        "strategy": strategy,
        "samples": len(records),
        "online_accuracy": float(correctness.mean()),
        "zero_shot_accuracy": float(zs_correct.mean()),
        "gain_over_zero_shot": gain,
        "worst_window_accuracy": float(rolling.min()),
        "negative_adaptation_rate": float(negative_flips.mean()),
        "expected_calibration_error": expected_calibration_error(confidence, correctness),
        "label_efficiency_pp_per_100_queries": float(10000.0 * gain / max(budget, 1)),
        "pseudo_memory_error_rate": float(pseudo_errors / max(pseudo_total, 1)),
        "pseudo_memory_size": pseudo_total,
        "event_sha256": hash_file(event_path) if write_events else None,
        "config": asdict(config),
        "final_state": controller.audit_state(),
    }
    (output_dir / f"{run_id}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--streams", nargs="+", default=["iid_random", "abrupt_domain", "class_correlated"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "periodic", "entropy", "margin", "disagreement", "ask_or_adapt"],
    )
    parser.add_argument("--pseudo-confidence", type=float, default=0.80)
    parser.add_argument("--pseudo-agreement", type=float, default=0.75)
    parser.add_argument("--persistence-switch-rate", type=float, default=0.35)
    parser.add_argument("--risk-entropy-threshold", type=float, default=0.50)
    parser.add_argument("--oracle-noise", type=float, default=0.0)
    parser.add_argument("--oracle-noise-mode", choices=["symmetric", "class_confusable"], default="symmetric")
    parser.add_argument("--confusable-temperature", type=float, default=0.10)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    loaded = np.load(args.cache, allow_pickle=False)
    cache = {name: loaded[name] for name in loaded.files}
    config = RunConfig(
        num_classes=len(cache["classes"]),
        pseudo_confidence=args.pseudo_confidence,
        pseudo_agreement=args.pseudo_agreement,
        persistence_switch_rate=args.persistence_switch_rate,
        risk_entropy_threshold=args.risk_entropy_threshold,
    )
    summaries = []
    for stream in args.streams:
        for seed in args.seeds:
            for budget in args.budgets:
                for strategy in args.strategies:
                    summary = run_one(
                        cache, args.output_dir, stream, seed, budget, strategy, config,
                        write_events=not args.summary_only,
                        oracle_noise=args.oracle_noise,
                        oracle_noise_mode=args.oracle_noise_mode,
                        confusable_temperature=args.confusable_temperature,
                    )
                    summaries.append(summary)
                    print(json.dumps({k: summary[k] for k in ["run_id", "online_accuracy", "query_count"]}))
    with (args.output_dir / "run_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
