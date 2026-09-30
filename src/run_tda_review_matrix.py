"""Audited, resumable cached-feature TDA matrix (NumPy only).

The default entropy gate follows the released TDA CODE, not the old local
reimplementation: natural-log entropy is divided by log2(C). All other cache
equations/settings retain the existing local implementation. This is a
single-view feature replay, not an end-to-end published-result reproduction.

Example (from repository root):
  python src/run_tda_review_matrix.py --workers 4

Use --entropy-mode legacy_ln --seeds 10 --output-dir <separate-directory>
to audit the previous 18 rows without overwriting official-code results.
"""
from __future__ import annotations

import os

# Set before importing NumPy, including in Windows spawn workers.
for _thread_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse
import csv
import hashlib
import inspect
import json
import platform
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STREAMS = ("iid_random", "abrupt_domain", "class_correlated")
OFFICIAL_COMMIT = "e697fb0c8078cdeff93daa56bcf8860702542069"
OFFICIAL_BASE = f"https://github.com/kdiAAA/TDA/blob/{OFFICIAL_COMMIT}"
DATASETS = {"officehome": "OfficeHome", "pacs": "PACS", "terra": "TerraIncognita"}
BACKBONES = {"openai_vitb16": "OpenAI ViT-B/16", "openclip_vitl14": "OpenCLIP ViT-L/14"}
PARAMS = {"pos_capacity": 3, "pos_alpha": 2.0, "pos_beta": 5.0,
          "neg_capacity": 2, "neg_alpha": 0.117, "neg_beta": 1.0,
          "entropy_lower": 0.2, "entropy_upper": 0.5,
          "mask_lower": 0.03, "mask_upper": 1.0, "logit_scale": 100.0}


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_order(labels, domains, kind, seed):
    """Exactly the benchmark's ordering, constructed before adaptation."""
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


def entropy(probs, mode):
    natural_entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum()
    denominator = np.log2(len(probs)) if mode == "official_log2" else np.log(len(probs))
    return float(natural_entropy / denominator)


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
                values.append(((prob_map > 0.03) & (prob_map < 1.0)).astype(np.float64))
            else:
                one = np.zeros(num_classes, dtype=np.float64)
                one[cls] = 1.0
                values.append(one)
    if not keys:
        return np.zeros(num_classes, dtype=np.float64)
    affinity = np.asarray(keys) @ feature
    weights = np.exp(-beta * (1.0 - affinity))
    return alpha * (weights[:, None] * np.asarray(values)).sum(axis=0)


def ece(confidence, correct, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            mask |= confidence == 0.0
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def worst500(correct):
    window = min(500, len(correct))
    return float(np.convolve(correct, np.ones(window) / window, mode="valid").min())


def run_replay(cache, stream, seed, mode):
    features, labels = cache["features"], cache["labels"]
    text, domains = cache["text_features"], cache["domains"]
    order = stream_order(labels, domains, stream, seed)
    classes = text.shape[0]
    positive, negative = {}, {}
    predictions, zero_predictions, confidence = [], [], []
    # Labels are NOT read inside the adaptation loop. They are used only above
    # to construct the synthetic stream, and below to evaluate finished scores.
    for idx in order:
        feature = np.asarray(features[idx], dtype=np.float64)
        base = 100.0 * feature @ text.T
        probs = np.exp(base - base.max())
        probs /= probs.sum()
        pred = int(base.argmax())
        loss = entropy(probs, mode)
        cache_update(positive, pred, feature, loss, 3)
        if 0.2 < loss < 0.5:
            cache_update(negative, pred, feature, loss, 2, probs)
        adapted = base + cache_logits(feature, positive, classes, 2.0, 5.0)
        adapted -= cache_logits(feature, negative, classes, 0.117, 1.0, negative=True)
        predictions.append(int(adapted.argmax()))
        # Preserve the benchmark's raw-cache zero-shot operation exactly.
        zero_predictions.append(int(np.argmax(features[idx] @ text.T)))
        final_probs = np.exp(adapted - adapted.max())
        confidence.append(float(final_probs.max() / final_probs.sum()))
    predictions = np.asarray(predictions, dtype=np.int32)
    zero_predictions = np.asarray(zero_predictions, dtype=np.int32)
    confidence = np.asarray(confidence)
    targets = labels[order]
    correct = (predictions == targets).astype(np.float64)
    zero_correct = (zero_predictions == targets).astype(np.float64)
    row = {"strategy": "TDA", "stream": stream, "seed": seed,
           "samples": len(correct), "budget_percent": 0.0, "query_count": 0,
           "online_accuracy": float(correct.mean()),
           "zero_shot_accuracy": float(zero_correct.mean()),
           "gain_over_zero_shot": float(correct.mean() - zero_correct.mean()),
           "worst_window_accuracy": worst500(correct),
           "zero_shot_worst_window_accuracy": worst500(zero_correct),
           "negative_adaptation_rate": float(((zero_correct == 1) & (correct == 0)).mean()),
           "expected_calibration_error": ece(confidence, correct),
           "positive_cache_size": int(sum(len(v) for v in positive.values())),
           "negative_cache_size": int(sum(len(v) for v in negative.values())),
           "entropy_mode": mode,
           "order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
           "prediction_sha256": hashlib.sha256(predictions.astype("<i4").tobytes()).hexdigest()}
    trace = {"original_index": order, "prediction": predictions,
             "zero_shot_prediction": zero_predictions, "correct": correct.astype(np.uint8),
             "zero_shot_correct": zero_correct.astype(np.uint8), "confidence": confidence}
    return row, trace


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_job(job):
    cache_path, output_dir, stream, seed, mode, provenance = job
    cache_path, output_dir = Path(cache_path), Path(output_dir)
    run_id = f"{stream}__seed{seed}__{mode}"
    folder = output_dir / "checkpoints" / cache_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint, trace_path = folder / f"{run_id}.json", folder / f"{run_id}.npz"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["provenance"] != provenance:
            raise RuntimeError(f"Stale checkpoint provenance: {checkpoint}")
        if not trace_path.exists() or hash_file(trace_path) != saved["row"]["trace_sha256"]:
            raise RuntimeError(f"Missing/corrupt checkpoint trace: {trace_path}")
        return saved["row"], True
    started = time.perf_counter()
    with np.load(cache_path, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in ("features", "labels", "domains", "text_features")}
    row, trace = run_replay(cache, stream, seed, mode)
    np.savez_compressed(trace_path, **trace)
    dataset, backbone = cache_path.stem.split("_", 1)
    row.update(cache=cache_path.stem, dataset=DATASETS[dataset], backbone=BACKBONES[backbone],
               elapsed_seconds=time.perf_counter() - started, trace_sha256=hash_file(trace_path))
    atomic_json(checkpoint, {"provenance": provenance, "row": row})
    return row, False


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)


def implementation_audit(cache_root, output_dir):
    """Small deterministic checks, independent of the reported dataset outcomes."""
    uniform = np.ones(7) / 7
    assert np.isclose(entropy(uniform, "legacy_ln"), 1.0)
    assert np.isclose(entropy(uniform, "official_log2"), np.log(2.0))
    rng = np.random.default_rng(83017)
    x = rng.normal(size=(75, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    text = rng.normal(size=(7, 16))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    synthetic = {"features": x, "text_features": text,
                 "labels": np.arange(75) % 7, "domains": np.arange(75) % 4}
    other_labels = {**synthetic, "labels": (synthetic["labels"] + 2) % 7}
    invariance = []
    for mode in ("legacy_ln", "official_log2"):
        for stream in ("iid_random", "abrupt_domain"):
            _, original = run_replay(synthetic, stream, 10, mode)
            _, changed = run_replay(other_labels, stream, 10, mode)
            assert np.array_equal(original["prediction"], changed["prediction"])
            assert np.array_equal(original["confidence"], changed["confidence"])
            invariance.append({"mode": mode, "stream": stream, "predictions_and_confidence_identical": True})
    # Class-correlated order legitimately uses the evaluator's labels. Test its
    # ordering equivalence separately, rather than claiming label permutation
    # should leave that artificially constructed stream unchanged.
    from run_streams import stream_order as benchmark_order
    from run_tda_cached import run_one as old_run
    for stream in STREAMS:
        for seed in range(10, 15):
            assert np.array_equal(stream_order(synthetic["labels"], synthetic["domains"], stream, seed),
                                  benchmark_order(synthetic["labels"], synthetic["domains"], stream, seed))
    new, _ = run_replay(synthetic, "iid_random", 10, "legacy_ln")
    old = old_run(synthetic, "iid_random", 10)
    old_deltas = {key: float(new[key]) - float(value) for key, value in old.items()
                  if isinstance(value, (float, int))}
    assert all(abs(value) < 1e-12 for value in old_deltas.values())
    verified_caches = []
    for path in sorted(cache_root.glob("*.npz")):
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        actual = hash_file(path)
        assert metadata["cache_sha256"] == actual, f"Cache metadata hash mismatch: {path}"
        verified_caches.append({"cache": path.stem, "sha256": actual, "matches_metadata": True})
    official_local = ROOT / ".tmp_tda"
    local_sources = []
    for relative in ("tda_runner.py", "utils.py", "configs/imagenet.yaml"):
        path = official_local / relative
        if path.exists():
            local_sources.append({"source": f"{OFFICIAL_BASE}/{relative}", "sha256": hash_file(path)})
    audit = {"status": "PASS", "uniform_entropy_official": entropy(uniform, "official_log2"),
             "uniform_entropy_legacy": entropy(uniform, "legacy_ln"),
             "label_invariance_checks": invariance, "stream_order_equivalence_cases": 15,
             "synthetic_legacy_metric_deltas": old_deltas, "verified_caches": verified_caches,
             "official_local_sources": local_sources,
             "scope": "Synthetic label-invariance and algebra checks plus cache-hash verification; not an official PyTorch numerical-equivalence claim."}
    atomic_json(output_dir / "implementation_audit.json", audit)
    return audit


def matched_comparison(rows, main_csv, anchors, output_dir):
    with main_csv.open(encoding="utf-8-sig", newline="") as handle:
        main = [r for r in csv.DictReader(handle) if r["strategy"] == "ask_or_adapt_v2"
                and float(r["budget_percent"]) == 1.0]
    main_index = {}
    for row in main:
        key = (row["dataset"], row["backbone"], row["stream"], int(row["seed"]))
        if key in main_index:
            raise ValueError(f"Duplicate main condition: {key}")
        main_index[key] = row
    matched, long_rows = [], []
    metrics = ("online_accuracy", "worst_window_accuracy", "negative_adaptation_rate",
               "query_count", "budget_percent")
    for tda in rows:
        key = (tda["dataset"], tda["backbone"], tda["stream"], tda["seed"])
        a2a = main_index[key]
        if int(a2a["samples"]) != tda["samples"]:
            raise ValueError(f"Sample-count mismatch: {key}")
        if abs(float(a2a["zero_shot_accuracy"]) - tda["zero_shot_accuracy"]) > 1e-12:
            raise ValueError(f"Frozen baseline mismatch: {key}")
        if int(a2a["query_count"]) != max(1, round(tda["samples"] / 100)):
            raise ValueError(f"Main exact-budget mismatch: {key}")
        identity = {k: tda[k] for k in ("cache", "dataset", "backbone", "stream", "seed", "samples")}
        match = dict(identity)
        zero = {"online_accuracy": tda["zero_shot_accuracy"],
                "worst_window_accuracy": tda["zero_shot_worst_window_accuracy"],
                "negative_adaptation_rate": 0.0, "query_count": 0, "budget_percent": 0.0}
        ask_file = anchors / tda["cache"] / f"{tda['stream']}__seed{tda['seed']}__budget1__ask_only.summary.json"
        methods = [("zero_shot", zero), ("TDA", tda), ("ask_or_adapt_v2", a2a)]
        if ask_file.exists():
            ask = json.loads(ask_file.read_text(encoding="utf-8"))
            if int(ask["samples"]) != tda["samples"] or int(ask["query_count"]) != int(a2a["query_count"]):
                raise ValueError(f"Ask-only pairing or budget mismatch: {key}")
            methods.append(("ask_only", ask))
        for method, source in methods:
            clean = {metric: float(source[metric]) for metric in metrics}
            clean["query_count"] = int(clean["query_count"])
            clean.update(identity, strategy=method)
            long_rows.append(clean)
            for metric in metrics:
                match[f"{method}__{metric}"] = clean[metric]
        match["a2a_minus_tda_online_pp"] = 100 * (float(a2a["online_accuracy"]) - tda["online_accuracy"])
        match["a2a_minus_tda_worst500_pp"] = 100 * (float(a2a["worst_window_accuracy"]) - tda["worst_window_accuracy"])
        matched.append(match)
    write_csv(output_dir / "matched_conditions.csv", matched)
    method_counts = {method: sum(r["strategy"] == method for r in long_rows)
                     for method in {r["strategy"] for r in long_rows}}
    if any(count != len(rows) for count in method_counts.values()):
        raise ValueError(f"Incomplete method pairing: {method_counts}")
    write_csv(output_dir / "comparison_long.csv", long_rows)
    summaries = []
    for grouping in ((), ("dataset",), ("dataset", "backbone"), ("stream",)):
        groups = defaultdict(list)
        for row in long_rows:
            groups[tuple(row[k] for k in grouping) + (row["strategy"],)].append(row)
        for group, members in sorted(groups.items()):
            summary = {"scope": "+".join(grouping) or "overall", "strategy": group[-1],
                       "conditions": len(members), **dict(zip(grouping, group[:-1]))}
            for metric in metrics:
                summary[metric] = float(np.mean([row[metric] for row in members]))
            summary["query_count_min"] = min(r["query_count"] for r in members)
            summary["query_count_max"] = max(r["query_count"] for r in members)
            summaries.append(summary)
    write_csv(output_dir / "comparison_summary.csv", summaries)
    used_seeds = sorted({int(r["seed"]) for r in rows})
    headline = {"conditions": len(rows), "seeds": used_seeds, "methods": sorted({r["strategy"] for r in long_rows}),
                "overall": [r for r in summaries if r["scope"] == "overall"],
                "a2a_vs_tda_online_wins": sum(r["a2a_minus_tda_online_pp"] > 0 for r in matched),
                "a2a_vs_tda_worst500_wins": sum(r["a2a_minus_tda_worst500_pp"] > 0 for r in matched),
                "budget_caveat": "TDA and zero-shot use 0 labels. Ask-or-Adapt and Ask-only use 1% labels. This is not an equal-budget comparison.",
                "aggregation": "Equal-weight mean across cache x stream x seed conditions, not pooled samples.",
                "uncertainty": f"{len(used_seeds)} stream-order seed(s) reuse each fixed dataset. Conditions are not independent datasets."}
    atomic_json(output_dir / "headline.json", headline)
    return headline


def aggregate_gpu(args):
    """Evaluate full CUDA traces and independently check available NumPy traces."""
    with (args.gpu_dir / "tda_cuda_runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        gpu_rows = list(csv.DictReader(handle))
    if len(gpu_rows) != 90:
        raise ValueError(f"Expected complete 90-condition GPU matrix; found {len(gpu_rows)}")
    expected_keys = {(p.stem, stream, seed) for p in args.cache_root.glob("*.npz")
                     for stream in STREAMS for seed in range(10, 15)}
    actual_keys = {(row["cache"], row["stream"], int(row["seed"])) for row in gpu_rows}
    if actual_keys != expected_keys or len(actual_keys) != len(gpu_rows):
        raise ValueError("GPU matrix duplicate, missing, or unexpected conditions")
    sources, checks, rows = [], [], []
    for cache_path in sorted(args.cache_root.glob("*.npz")):
        cache_hash = hash_file(cache_path)
        with np.load(cache_path, allow_pickle=False) as loaded:
            features, text = loaded["features"], loaded["text_features"]
            labels, domains = loaded["labels"], loaded["domains"]
        zero_pred = np.asarray([int(np.argmax(feature @ text.T)) for feature in features])
        sources.append({"cache": cache_path.stem, "cache_sha256": cache_hash})
        for source in [r for r in gpu_rows if r["cache"] == cache_path.stem]:
            if source["cache_sha256"] != cache_hash:
                raise ValueError(f"GPU/cache source mismatch: {cache_path.stem}")
            stream, seed = source["stream"], int(source["seed"])
            trace_path = args.gpu_dir / f"{cache_path.stem}__{stream}__seed{seed}.npz"
            with np.load(trace_path, allow_pickle=False) as trace:
                order, prediction, confidence = trace["original_index"], trace["prediction"], trace["confidence"]
            expected_order = stream_order(labels, domains, stream, seed)
            if not np.array_equal(order, expected_order):
                raise ValueError(f"GPU stream-order mismatch: {trace_path}")
            if prediction.shape != labels.shape or confidence.shape != labels.shape:
                raise ValueError(f"GPU trace length mismatch: {trace_path}")
            correct = (prediction == labels[order]).astype(np.float64)
            zero_correct = (zero_pred[order] == labels[order]).astype(np.float64)
            dataset, backbone = cache_path.stem.split("_", 1)
            row = {"cache": cache_path.stem, "dataset": DATASETS[dataset], "backbone": BACKBONES[backbone],
                   "stream": stream, "seed": seed, "samples": len(labels), "strategy": "TDA",
                   "execution_backend": "PyTorch CUDA FP64", "entropy_mode": "official_log2",
                   "budget_percent": 0.0, "query_count": 0, "online_accuracy": float(correct.mean()),
                   "zero_shot_accuracy": float(zero_correct.mean()),
                   "gain_over_zero_shot": float(correct.mean() - zero_correct.mean()),
                   "worst_window_accuracy": worst500(correct), "zero_shot_worst_window_accuracy": worst500(zero_correct),
                   "negative_adaptation_rate": float(((zero_correct == 1) & (correct == 0)).mean()),
                   "expected_calibration_error": ece(confidence, correct), "cache_sha256": cache_hash,
                   "trace_sha256": hash_file(trace_path),
                   "order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
                   "prediction_sha256": hashlib.sha256(prediction.astype("<i4").tobytes()).hexdigest()}
            for metric in ("online_accuracy", "worst_window_accuracy", "expected_calibration_error"):
                if abs(row[metric] - float(source[metric])) > 1e-12:
                    raise ValueError(f"GPU summary/trace mismatch for {trace_path}: {metric}")
            rows.append(row)
            for cpu_root in args.cpu_audit_dirs:
                cpu_path = cpu_root / "checkpoints" / cache_path.stem / f"{stream}__seed{seed}__official_log2.npz"
                json_path = cpu_path.with_suffix(".json")
                if not json_path.exists():
                    continue
                saved = json.loads(json_path.read_text(encoding="utf-8"))
                if saved["provenance"]["cache_sha256"] != cache_hash or hash_file(cpu_path) != saved["row"]["trace_sha256"]:
                    raise ValueError(f"Invalid CPU checkpoint provenance: {cpu_path}")
                with np.load(cpu_path, allow_pickle=False) as cpu:
                    assert np.array_equal(cpu["original_index"], order)
                    differences = int(np.count_nonzero(cpu["prediction"] != prediction))
                    confidence_delta = float(np.max(np.abs(cpu["confidence"] - confidence)))
                checks.append({"cache": cache_path.stem, "stream": stream, "seed": seed,
                               "samples": len(labels), "prediction_disagreements": differences,
                               "confidence_max_abs_error": confidence_delta,
                               "online_accuracy_delta_gpu_minus_cpu": row["online_accuracy"] - saved["row"]["online_accuracy"],
                               "worst500_delta_gpu_minus_cpu": row["worst_window_accuracy"] - saved["row"]["worst_window_accuracy"],
                               "cpu_trace": str(cpu_path), "gpu_trace_sha256": row["trace_sha256"]})
    rows.sort(key=lambda row: (row["cache"], row["stream"], row["seed"]))
    write_csv(args.output_dir / "tda_gpu_matched_runs.csv", rows)
    write_csv(args.output_dir / "cuda_vs_cpu_trace_checks.csv", checks)
    if args.previous_csv.exists():
        with args.previous_csv.open(encoding="utf-8-sig", newline="") as handle:
            previous = {(r["cache"], r["stream"], int(r["seed"])): r for r in csv.DictReader(handle)}
        correction = []
        correction_metrics = ("online_accuracy", "worst_window_accuracy", "negative_adaptation_rate")
        for row in rows:
            key = (row["cache"], row["stream"], row["seed"])
            if key not in previous:
                continue
            old = previous[key]
            item = {"cache": row["cache"], "stream": row["stream"], "seed": row["seed"]}
            for metric in correction_metrics:
                item[f"legacy__{metric}"] = float(old[metric])
                item[f"official__{metric}"] = row[metric]
                item[f"delta_pp__{metric}"] = 100 * (row[metric] - float(old[metric]))
            correction.append(item)
        write_csv(args.output_dir / "entropy_gate_correction_matched_seed10.csv", correction)
        atomic_json(args.output_dir / "entropy_gate_correction_summary.json", {
            "matched_conditions": len(correction), "seed": 10,
            "old_csv_sha256": hash_file(args.previous_csv),
            "means": {key: float(np.mean([r[key] for r in correction])) for key in correction[0]
                      if key.startswith(("legacy__", "official__", "delta_pp__"))},
            "interpretation": "This same-seed subset isolates the entropy-gate correction plus the audited backend. The difference between old18 and new90 also includes four added seeds and is not purely a correction effect."})
    headline = matched_comparison(rows, args.main_csv, args.anchors, args.output_dir)
    headline["tda_execution_backend"] = "PyTorch CUDA FP64"
    headline["cpu_crosscheck_conditions"] = len(checks)
    headline["cpu_crosscheck_prediction_disagreements"] = sum(r["prediction_disagreements"] for r in checks)
    headline["cpu_crosscheck_distinct_caches"] = len({r["cache"] for r in checks})
    atomic_json(args.output_dir / "headline.json", headline)
    atomic_json(args.output_dir / "manifest.json", {
        "status": "COMPLETE", "conditions": len(rows), "source_backend": "PyTorch CUDA FP64",
        "gpu_manifest": json.loads((args.gpu_dir / "manifest.json").read_text(encoding="utf-8")),
        "gpu_csv_sha256": hash_file(args.gpu_dir / "tda_cuda_runs.csv"), "cache_sources": sources,
        "main_csv_sha256": hash_file(args.main_csv), "cpu_crosschecks": len(checks),
        "cpu_crosscheck_distinct_caches": len({r["cache"] for r in checks}),
        "cpu_prediction_disagreements": sum(r["prediction_disagreements"] for r in checks),
        "cpu_confidence_max_abs_error": max((r["confidence_max_abs_error"] for r in checks), default=None),
        "source_directory": str(args.gpu_dir), "official_commit": OFFICIAL_COMMIT,
        "reproduction_caveat": "Single-view cached-feature replay using official-code entropy gate and fixed settings. Not a full image/augmentation/end-to-end official benchmark reproduction.",
        "cpu_scope": "Available completed NumPy conditions are independent crosschecks, not a claim of a completed 90-condition CPU matrix."})
    print(json.dumps(headline, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "remote_artifacts/20260814_multibench_v6/cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/review_revision_round2/tda")
    parser.add_argument("--main-csv", type=Path, default=ROOT / "analysis/v2_confirmation/all_confirmation_runs.csv")
    parser.add_argument("--anchors", type=Path, default=ROOT / "analysis/review_revision/anchors")
    parser.add_argument("--previous-csv", type=Path, default=ROOT / "analysis/review_revision/tda/tda_cached_runs.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12, 13, 14])
    parser.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    parser.add_argument("--entropy-mode", choices=("official_log2", "legacy_ln"), default="official_log2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--self-test", action="store_true", help="Run implementation audit and exit")
    parser.add_argument("--cache-names", nargs="+", help="Explicit cache subset for CPU crosschecks")
    parser.add_argument("--gpu-dir", type=Path, help="Aggregate completed full90 CUDA traces instead of running CPU")
    parser.add_argument("--cpu-audit-dirs", nargs="+", type=Path, default=[ROOT / "analysis/review_revision_round2/tda"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        print(json.dumps(implementation_audit(args.cache_root, args.output_dir), indent=2))
        return
    if args.gpu_dir:
        aggregate_gpu(args)
        return
    caches = sorted(args.cache_root.glob("*.npz"))
    if len(caches) != 6:
        raise ValueError(f"Expected six caches; found {len(caches)}")
    if args.cache_names:
        caches = [p for p in caches if p.stem in args.cache_names]
        if {p.stem for p in caches} != set(args.cache_names):
            raise ValueError("Requested cache subset contains an unknown name")
    algorithm_source = "\n".join(inspect.getsource(f) for f in
                                  (stream_order, entropy, cache_update, cache_logits, ece, worst500, run_replay))
    algorithm_hash = hashlib.sha256(algorithm_source.encode()).hexdigest()
    jobs, sources = [], []
    for path in caches:
        provenance = {"cache_sha256": hash_file(path), "algorithm_sha256": algorithm_hash,
                      "entropy_mode": args.entropy_mode, "numpy": np.__version__,
                      "parameters": PARAMS, "official_commit": OFFICIAL_COMMIT}
        sources.append({"path": str(path.resolve()), **provenance})
        for stream in args.streams:
            for seed in args.seeds:
                jobs.append((str(path), str(args.output_dir), stream, seed, args.entropy_mode, provenance))
    manifest = {"status": "RUNNING", "expected_runs": len(jobs), "completed_runs": 0,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "workers": args.workers,
                "blas_threads": 1, "sources": sources,
                "main_csv_sha256": hash_file(args.main_csv),
                "official_sources": [f"{OFFICIAL_BASE}/tda_runner.py", f"{OFFICIAL_BASE}/utils.py", f"{OFFICIAL_BASE}/configs/imagenet.yaml"],
                "entropy_audit": "Official utils.get_entropy divides natural-log softmax entropy by log2(C). Legacy local code used ln(C). The corrected gate equals legacy entropy times ln(2). No old rows are reused.",
                "reproduction_caveat": "Single-view cached-feature NumPy float64 replay. Same frozen inputs/prompts as main methods; no encoder/image augmentation rerun, no dataset-specific tuning, no claim of exact official numerical or end-to-end reproduction.",
                "evaluation_order": "Construct synthetic order once. Cache updates and adapted score use feature and pseudo-label only. Targets are indexed for correctness only after all predictions are finished. Current unlabeled feature enters cache before its adapted score, as in official TDA."}
    atomic_json(args.output_dir / "manifest.json", manifest)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            row, resumed = future.result()
            rows.append(row)
            manifest["completed_runs"] = len(rows)
            atomic_json(args.output_dir / "manifest.json", manifest)
            print(f"[{len(rows)}/{len(jobs)}] {row['cache']} {row['stream']} seed{row['seed']} acc={row['online_accuracy']:.6f} {'resumed' if resumed else 'fresh'}", flush=True)
    rows.sort(key=lambda row: (row["cache"], row["stream"], row["seed"]))
    if len({(r["cache"], r["stream"], r["seed"]) for r in rows}) != len(jobs):
        raise ValueError("Duplicate or missing conditions")
    write_csv(args.output_dir / "tda_cached_runs.csv", rows)
    if args.entropy_mode == "legacy_ln" and args.previous_csv.exists():
        with args.previous_csv.open(encoding="utf-8-sig", newline="") as handle:
            previous = {(r["cache"], r["stream"], int(r["seed"])): r for r in csv.DictReader(handle)}
        checks = []
        for row in rows:
            old = previous.get((row["cache"], row["stream"], row["seed"]))
            if old is not None:
                for metric in ("online_accuracy", "zero_shot_accuracy", "worst_window_accuracy", "negative_adaptation_rate", "expected_calibration_error", "positive_cache_size", "negative_cache_size"):
                    delta = float(row[metric]) - float(old[metric])
                    checks.append({"cache": row["cache"], "stream": row["stream"], "seed": row["seed"], "metric": metric, "delta": delta})
                    if abs(delta) > 1e-12:
                        raise ValueError(f"Legacy reproduction mismatch: {checks[-1]}")
        write_csv(args.output_dir / "legacy_reproduction_checks.csv", checks)
    headline = matched_comparison(rows, args.main_csv, args.anchors, args.output_dir)
    manifest.update(status="COMPLETE", completed_utc=datetime.now(timezone.utc).isoformat(),
                    output_csv_sha256=hash_file(args.output_dir / "tda_cached_runs.csv"))
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
