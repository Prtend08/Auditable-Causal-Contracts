"""New post-hoc paired 1%/5% replay with full immutable-prediction events.

Uses frozen run_streams.run_one without altering selection or numerical ties.
Never replaces historical confirmation/budget results. Resume only within an
identical cache/core/config/runtime contract and verify every completed trace.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from a2a.core import RunConfig, make_budget_segments
from run_streams import hash_file, run_one, stream_order

STREAMS = ("iid_random", "abrupt_domain", "class_correlated")
SEEDS = tuple(range(20, 25))
BUDGETS = (1.0, 5.0)
DATASETS = {"pacs": "PACS", "officehome": "OfficeHome", "terra": "TerraIncognita"}
BACKBONES = ("openai_vitb16", "openclip_vitl14")
KEY = ["dataset", "backbone", "stream", "seed"]
STRATA = KEY[:-1]
METRICS = ("causal_online_percent", "leaky_online_percent", "online_inflation_pp",
           "causal_worst500_percent", "leaky_worst500_percent", "worst500_inflation_pp",
           "queried_error_percent")


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


@lru_cache(maxsize=1)
def load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    cache["reference_prediction"] = np.asarray(
        [int(np.argmax(feature @ cache["text_features"].T)) for feature in cache["features"]], dtype=np.int64)
    return cache


def worst(correct, width=500):
    width = min(width, len(correct))
    cumulative = np.concatenate(([0], np.cumsum(correct, dtype=np.int64)))
    return float(np.min(cumulative[width:] - cumulative[:-width]) / width)


def audit_run(cache, output_dir, stream, seed, budget, contract_hash):
    run_id = f"{stream}__seed{seed}__budget{budget:g}__ask_or_adapt_v2"
    summary_path = output_dir / (run_id + ".summary.json")
    event_path = output_dir / (run_id + ".events.csv.gz")
    receipt_path = output_dir / (run_id + ".audit.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_event_hash = hash_file(event_path)
    require(actual_event_hash == summary["event_sha256"], f"Event SHA mismatch: {event_path}")
    config = RunConfig(num_classes=len(cache["classes"]))
    require(summary["config"] == asdict(config), "Frozen config mismatch")
    require(summary["run_id"] == run_id and summary["stream"] == stream and summary["seed"] == seed
            and summary["budget_percent"] == budget and summary["strategy"] == "ask_or_adapt_v2", "Run identity mismatch")
    require(summary["oracle_noise"] == summary["noisy_oracle_count"] == 0, "Unexpected oracle noise")
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(previous["contract_sha256"] == contract_hash, "Resume contract mismatch")
        require(previous["event_sha256"] == actual_event_hash and previous["summary_sha256"] == hash_file(summary_path),
                "Completed artifact changed since prior audit")
    numeric_signals = ["confidence", "entropy", "margin", "disagreement", "coverage", "query_value", "selection_score"]
    columns = ["position", "original_index", "path", "domain", "label", "prediction", "correct",
               "zero_shot_prediction", "zero_shot_correct", "queried", "adapted", "action", "oracle_label", "query_revealed_at"] + numeric_signals
    events = pd.read_csv(event_path, usecols=columns, keep_default_na=False)
    require(np.isfinite(events[numeric_signals].to_numpy(dtype=float)).all(), "Non-finite pre-reveal event signal")
    n = len(events)
    order = stream_order(cache["labels"], cache["domains"], stream, seed)
    require(n == len(order) == summary["samples"], "Sample count mismatch")
    require(np.array_equal(events.position, np.arange(n)), "Event position mismatch")
    require(np.array_equal(events.original_index, order), "Seed/stream order mismatch")
    for field in ("label", "domain", "path"):
        source = {"label": "labels", "domain": "domains", "path": "paths"}[field]
        actual, expected = events[field].to_numpy(), cache[source][order]
        if field == "path":
            actual, expected = actual.astype(str), expected.astype(str)
        require(np.array_equal(actual, expected), f"Cache {field} mismatch")
    correct = events.correct.to_numpy(dtype=np.int64)
    queried = events.queried.to_numpy(dtype=np.int64)
    adapted = events.adapted.to_numpy(dtype=np.int64)
    require(np.isin(correct, [0, 1]).all() and np.isin(queried, [0, 1]).all() and np.isin(adapted, [0, 1]).all(),
            "Nonbinary event flag")
    require(np.array_equal(correct, (events.prediction == events.label).astype(int)), "Correctness mismatch")
    require(np.array_equal(events.zero_shot_prediction, cache["reference_prediction"][order]), "Raw zero-shot prediction mismatch")
    require(np.array_equal(events.zero_shot_correct, (events.zero_shot_prediction == events.label).astype(int)), "Zero correctness mismatch")
    q = queried.astype(bool)
    require(not (queried & adapted).any(), "Ask/Adapt overlap")
    require(np.array_equal(events.action.eq("ask"), q), "Ask action mismatch")
    require(np.array_equal(events.action.eq("adapt"), adapted.astype(bool)), "Adapt action mismatch")
    require(events.action.isin(["ask", "adapt", "predict"]).all(), "Unresolved event action")
    require((events.loc[~q, "oracle_label"] == "").all() and (events.loc[~q, "query_revealed_at"] == "").all(),
            "Oracle info on unqueried event")
    require(np.array_equal(events.loc[q, "oracle_label"].astype(int), events.loc[q, "label"]), "Clean oracle mismatch")
    actual_queries = int(queried.sum())
    expected_queries = max(1, round(n * budget / 100))
    require(actual_queries == expected_queries == summary["query_count"] == summary["budget_count"], "Exact budget failure")
    for segment in make_budget_segments(n, actual_queries):
        positions = np.flatnonzero(q[segment[0]:segment[-1] + 1]) + segment[0]
        require(len(positions) == 1, "Not exactly one query per segment")
        selected = int(positions[0])
        require(int(events.loc[selected, "query_revealed_at"]) == int(segment[-1]) >= selected,
                "Early/non-segment-close reveal")
    leaky = np.maximum(correct, queried)
    causal_online, leaky_online = float(correct.mean()), float(leaky.mean())
    causal_worst, leaky_worst = worst(correct), worst(leaky)
    require(abs(causal_online - summary["online_accuracy"]) < 1e-12, "Summary online mismatch")
    require(abs(causal_worst - summary["worst_window_accuracy"]) < 1e-12, "Summary worst500 mismatch")
    errors = int((queried * (1 - correct)).sum())
    require(np.isclose(leaky_online - causal_online, errors / n, rtol=0, atol=1e-15), "Exact leakage identity failure")
    row = {"stream": stream, "seed": seed, "budget_percent": budget, "samples": n,
           "queries": actual_queries, "queried_errors": errors, "causal_correct": int(correct.sum()),
           "leaky_correct": int(leaky.sum()), "actual_budget_percent": 100 * actual_queries / n,
           "causal_online_percent": 100 * causal_online, "leaky_online_percent": 100 * leaky_online,
           "online_inflation_pp": 100 * errors / n,
           "causal_worst500_percent": 100 * causal_worst, "leaky_worst500_percent": 100 * leaky_worst,
           "worst500_inflation_pp": 100 * (leaky_worst - causal_worst),
           "queried_error_percent": 100 * errors / actual_queries,
           "event_path": str(event_path.resolve()), "event_sha256": actual_event_hash,
           "summary_path": str(summary_path.resolve()), "summary_sha256": hash_file(summary_path),
           "order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
           "zero_prediction_sha256": hashlib.sha256(events.zero_shot_prediction.to_numpy(dtype="<i8").tobytes()).hexdigest(),
           "contract_sha256": contract_hash, "audit_pass": True}
    save_json(receipt_path, row)
    return row


def paired_job(job):
    cache_path, output_root, stream, seed, contract_hash = job
    started = time.perf_counter()
    cache = load_cache(cache_path)
    cache_file = Path(cache_path)
    dataset_code, backbone = cache_file.stem.split("_", 1)
    output_dir = Path(output_root) / "runs" / cache_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget in BUDGETS:
        run_id = f"{stream}__seed{seed}__budget{budget:g}__ask_or_adapt_v2"
        summary_path = output_dir / (run_id + ".summary.json")
        event_path = output_dir / (run_id + ".events.csv.gz")
        resumed = summary_path.exists() and event_path.exists()
        if not resumed:
            for incomplete in (summary_path, event_path):
                if incomplete.exists():
                    quarantine = Path(output_root) / "incomplete_preserved" / cache_file.stem
                    quarantine.mkdir(parents=True, exist_ok=True)
                    incomplete.rename(quarantine / (incomplete.name + f".{time.time_ns()}"))
            run_one(cache, output_dir, stream, seed, budget, "ask_or_adapt_v2",
                    RunConfig(num_classes=len(cache["classes"])), write_events=True, oracle_noise=0.0)
        row = audit_run(cache, output_dir, stream, seed, budget, contract_hash)
        row.update(cache=cache_file.stem, dataset=DATASETS[dataset_code], backbone=backbone, resumed=resumed)
        rows.append(row)
    require(rows[0]["order_sha256"] == rows[1]["order_sha256"]
            and rows[0]["zero_prediction_sha256"] == rows[1]["zero_prediction_sha256"], "Budget pairing identity failure")
    return rows, time.perf_counter() - started


def intervals(series):
    groups = [part.to_numpy(dtype=float) for _, part in series.groupby(level=STRATA)]
    require(len(groups) == 18 and all(len(group) == 5 for group in groups), "Expected paired18x5 grid")
    values = np.asarray(groups)
    rng = np.random.default_rng(20260814)
    idx = rng.integers(0, 5, size=(20000, 18, 5))
    fixed = values[np.arange(18)[None, :, None], idx].mean(axis=2).mean(axis=1)
    stratum_means = values.mean(axis=1)
    rng = np.random.default_rng(20260814)
    clusters = stratum_means[rng.integers(0, 18, size=(100000, 18))].mean(axis=1)
    return {"mean": float(values.mean()), "fixed_strata_ci95": np.quantile(fixed, [.025, .975]).tolist(),
            "stratum_resampled_ci95": np.quantile(clusters, [.025, .975]).tolist()}


def analyze(rows, metadata, output_root):
    frame = pd.DataFrame(rows).sort_values(KEY + ["budget_percent"])
    require(len(frame) == 180 and not frame.duplicated(KEY + ["budget_percent"]).any(), "Incomplete180 grid")
    expected_grid = {(dataset, backbone, stream, seed, budget) for dataset in DATASETS.values()
                     for backbone in BACKBONES for stream in STREAMS for seed in SEEDS for budget in BUDGETS}
    require(set(frame[KEY + ["budget_percent"]].itertuples(index=False, name=None)) == expected_grid,
            "Unexpected condition in180 grid")
    reports = {}
    indexed = {}
    for budget, part in frame.groupby("budget_percent"):
        part = part.set_index(KEY).sort_index()
        require(len(part) == 90, "Incomplete budget grid")
        indexed[int(budget)] = part
        total_samples, total_queries, total_errors = int(part.samples.sum()), int(part.queries.sum()), int(part.queried_errors.sum())
        reports[str(int(budget))] = {"conditions": len(part), "samples": total_samples,
            "queries": total_queries, "queried_errors": total_errors,
            "macro": {metric: intervals(part[metric]) for metric in METRICS},
            "micro": {"causal_online_percent": 100 * int(part.causal_correct.sum()) / total_samples,
                      "leaky_online_percent": 100 * int(part.leaky_correct.sum()) / total_samples,
                      "online_inflation_pp": 100 * total_errors / total_samples,
                      "actual_budget_percent": 100 * total_queries / total_samples,
                      "queried_error_percent": 100 * total_errors / total_queries},
            "micro_worst_window": "Not pooled: worst500 is computed separately per stream, then macro-averaged; concatenation would invent boundaries.",
            "by_dataset": {dataset: {metric: float(group[metric].mean()) for metric in METRICS}
                           for dataset, group in part.reset_index().groupby("dataset")}}
    require(indexed[1].index.equals(indexed[5].index), "Budget index mismatch")
    differences = indexed[5][list(METRICS)] - indexed[1][list(METRICS)]
    paired = {metric: intervals(differences[metric]) for metric in METRICS}
    save_csv(output_root / "paired_5_minus_1.csv", differences.reset_index())
    save_csv(output_root / "per_run.csv", frame)
    report = {"status": "PASS", "experiment": metadata["purpose"],
        "finished_at_utc": utc(), "run_count": len(frame), "event_count": int(frame.samples.sum()),
        "query_count": int(frame.queries.sum()), "all_event_audits_pass": bool(frame.audit_pass.all()),
        "contract_sha256": metadata["contract_sha256"], "by_budget_percent": reports,
        "paired_5_minus_1": paired,
        "method": {"estimand": "Equal mean of18 dataset/backbone/stream stratum means, each averaging5 matched seeds20-24.",
                   "fixed_strata_ci": "Paired percentile bootstrap, resample5 seeds within each of18 fixed strata,20000 draws, default_rng20260814; budgets and score-repair states retain pairing.",
                   "heterogeneity_sensitivity_ci": "Resample18 seed-averaged stratum means,100000 draws, default_rng20260814. Shared dataset images induce dependence; not an unseen-domain generalization guarantee.",
                   "multiplicity": "Post-hoc marginal95% intervals, not multiplicity-adjusted or new confirmatory claims.",
                   "counterfactual": "Scored correctness becomes max(original_correct, queried); future predictions/state unchanged.",
                   "identity": "Per-run Delta_online_pp=100*K/T=actual_budget_percent*(K/Q); macro is mean of these products; micro is100*sumK/sumT. No fixed queried-error assumption across budgets.",
                   "causal_scope": "Frozen code and event validation establish ordering/selection records; event files do not contain per-step prototype or weight snapshots.",
                   "historical_scope": "New same-runtime post-hoc experiment; does not recover or replace archived seed20-24 summary-only trajectories or the original confirmation.",
                   "memory_scope": "Single shared frozen feature cache replay; no training, no changed thresholds, no numerical tie-handling modification."},
        "output_hashes": {name: hash_file(output_root / name) for name in ("per_run.csv", "paired_5_minus_1.csv")}}
    save_json(output_root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    require(args.workers >= 1, "Workers must be positive")
    thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    require(all(os.environ.get(key) == "1" for key in thread_vars), "Set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 before launch")
    caches = sorted(args.cache_root.resolve().glob("*.npz"))
    expected = {f"{dataset}_{backbone}" for dataset in DATASETS for backbone in BACKBONES}
    require({path.stem for path in caches} == expected, "Expected exactly six frozen caches")
    output = args.output_dir.resolve()
    require("budget_leakage_replay" in output.parts or "round3_budget_review" in output.parts,
            "Use a distinct budget_leakage_replay or round3_budget_review directory, never historical results")
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    frozen_sources = [source_root / "run_streams.py", source_root / "a2a/core.py", Path(__file__).resolve()]
    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.show_config()
    blas_build = numpy_config.getvalue()
    contract = {"cache_sha256": {p.name: hash_file(p) for p in caches},
                "source_sha256": {p.relative_to(source_root).as_posix(): hash_file(p) for p in frozen_sources},
                "config_template": asdict(RunConfig(num_classes=0)), "streams": list(STREAMS), "seeds": list(SEEDS),
                "budgets_percent": list(BUDGETS), "strategy": "ask_or_adapt_v2", "oracle_noise": 0.0,
                "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                            "platform": platform.platform(), "executable": sys.executable,
                            "numpy_path": np.__file__, "pandas_path": pd.__file__,
                            "numpy_blas_build": blas_build,
                            "numpy_blas_build_sha256": hashlib.sha256(blas_build.encode()).hexdigest(),
                            "blas_threads": {key: os.environ.get(key) for key in thread_vars}}}
    contract_hash = canonical_hash(contract)
    metadata_path = output / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        require(metadata["runner_sha256"] == hash_file(Path(__file__)), "Cannot resume with a changed replay/audit runner")
        require(metadata["contract_sha256"] == contract_hash, "Cannot resume in changed cache/core/config/runtime")
    else:
        existing_runs = output / "runs"
        require(not existing_runs.exists() or not any(existing_runs.rglob("*")),
                "Refusing to adopt existing runs without their generating run_metadata.json")
        metadata = {"purpose": "New post-hoc matched1%/5% event-enabled replay; no model/threshold selection and no historical result replacement.",
                    "started_at_utc": utc(), "contract": contract, "contract_sha256": contract_hash,
                    "runner_path": str(Path(__file__).resolve()), "runner_sha256": hash_file(Path(__file__)),
                    "workers": args.workers, "expected_pairs": 90, "expected_runs": 180,
                    "cache_root": str(args.cache_root.resolve()), "output_dir": str(output), "argv": sys.argv}
        save_json(metadata_path, metadata)
    jobs = [(str(cache), str(output), stream, seed, contract_hash)
            for cache in caches for stream in STREAMS for seed in SEEDS]
    completed, start = [], time.perf_counter()
    print(json.dumps({"status": "started", "pairs": 90, "runs": 180, "workers": args.workers,
                      "output": str(output), "contract_sha256": contract_hash}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(paired_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            pair, elapsed = future.result()
            completed.extend(pair)
            progress = {"completed_pairs": count, "completed_runs": len(completed), "total_runs": 180,
                        "elapsed_seconds": time.perf_counter() - start, "last_cache": pair[0]["cache"],
                        "last_stream": pair[0]["stream"], "last_seed": pair[0]["seed"], "pair_seconds": elapsed,
                        "last_1pct_leakage_pp": pair[0]["online_inflation_pp"], "last_5pct_leakage_pp": pair[1]["online_inflation_pp"]}
            save_json(output / "progress.json", progress)
            save_csv(output / "completed_runs.csv", completed)
            print(json.dumps(progress), flush=True)
    require(all(hash_file(p) == contract["source_sha256"][p.relative_to(source_root).as_posix()] for p in frozen_sources),
            "Frozen controller source changed during execution")
    require(all(hash_file(p) == contract["cache_sha256"][p.name] for p in caches), "Cache changed during execution")
    report = analyze(completed, metadata, output)
    print(json.dumps({"status": report["status"], "runs": 180, "events": report["event_count"],
                      "elapsed_seconds": time.perf_counter() - start, "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
