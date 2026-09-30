"""Audit and recompute paired window/anchor comparisons from archived runs.

This is a post-hoc analysis, not an additional confirmatory experiment.  Its
sign-flip, bootstrap, and primary Holm families reproduce the protocol in
analyze_v2_confirmation.py: average five paired seeds in each fixed stratum,
enumerate all 2**18 two-sided sign flips, and bootstrap seeds *within* strata.
The vectorized routines preserve the original bootstrap random-number order.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from a2a.core import softmax
from run_streams import expected_calibration_error, stream_order


SELECTORS = ["random", "periodic", "entropy", "margin", "disagreement", "dynamic_entropy_online"]
METHOD = "ask_or_adapt_v2"
KEY = ["dataset", "backbone", "stream", "seed"]
STRATA = KEY[:-1]
WINDOWS = (250, 500, 1000)
DATASETS = {"officehome": "OfficeHome", "pacs": "PACS", "terra": "TerraIncognita"}
BACKBONES = {"openai_vitb16": "OpenAI ViT-B/16", "openclip_vitl14": "OpenCLIP ViT-L/14"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted, running = {}, 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * values[name]))
        adjusted[name] = running
    return adjusted


def sign_flip(values: np.ndarray) -> float:
    """Exact two-sided test, including all sign assignments (no plus-one)."""
    observed = abs(values.mean())
    total, extreme = 1 << len(values), 0
    bits = np.arange(len(values), dtype=np.uint32)
    for start in range(0, total, 16384):
        masks = np.arange(start, min(start + 16384, total), dtype=np.uint32)
        signs = np.where((masks[:, None] & (1 << bits)) != 0, 1.0, -1.0)
        extreme += np.count_nonzero(abs((signs * values).mean(axis=1)) >= observed - 1e-15)
    return float(extreme / total)


def bootstrap(delta: pd.Series, draws: int = 20000) -> tuple[float, float]:
    groups = [part.to_numpy() for _, part in delta.groupby(level=STRATA)]
    require(all(len(group) == 5 for group in groups), "Expected exactly five seeds per stratum")
    values = np.asarray(groups)
    rng = np.random.default_rng(20260814)
    # Ordering is draw, stratum, seed, exactly as the original nested rng.choice.
    indices = rng.integers(0, 5, size=(draws, len(groups), 5))
    sampled = values[np.arange(len(groups))[None, :, None], indices].mean(axis=2).mean(axis=1)
    low, high = np.quantile(sampled, [.025, .975])
    return float(low), float(high)


def statistics(delta: pd.Series) -> dict:
    strata = delta.groupby(level=STRATA).mean()
    low, high = bootstrap(delta)
    standard_deviation = strata.std(ddof=1)
    return {
        "paired_runs": len(delta), "strata": len(strata), "seeds_per_stratum": 5,
        "mean_delta_pp": float(100 * delta.mean()),
        "ci95_low_pp": 100 * low, "ci95_high_pp": 100 * high,
        "stratum_win_rate": float((strata > 0).mean()),
        "stratum_wins": int((strata > 0).sum()),
        "stratum_ties": int((strata == 0).sum()),
        "paired_effect_dz": float(strata.mean() / standard_deviation) if standard_deviation else None,
        "p_sign_flip": sign_flip(strata.to_numpy()),
    }


def worst(correct: np.ndarray, window: int) -> float:
    """Equivalent to the archived valid-mode convolution, without roundoff."""
    counts = np.concatenate(([0], np.cumsum(correct, dtype=np.int64)))
    return float(np.min(counts[window:] - counts[:-window]) / window)


def write_frame(out: Path, name: str, rows) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.json").write_text(frame.to_json(orient="records", indent=2, double_precision=15), encoding="utf-8")


def validate_statistics_implementation() -> None:
    """Small deterministic cross-check against the actual source functions."""
    # Execute only the two original function definitions, avoiding unrelated
    # plotting dependencies that this audit does not need.
    source = ast.parse(Path(__file__).with_name("analyze_v2_confirmation.py").read_text(encoding="utf-8"))
    selected = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef)
        and node.name in {"bootstrap", "sign_flip"}], type_ignores=[])
    reference = {"np": np, "pd": pd}
    exec(compile(selected, "analyze_v2_confirmation.py", "exec"), reference)
    reference_bootstrap, reference_sign_flip = reference["bootstrap"], reference["sign_flip"]
    index = pd.MultiIndex.from_product([["D"], ["B1", "B2"], ["S1", "S2", "S3"], range(10, 15)], names=KEY)
    delta = pd.Series(np.sin(np.arange(len(index))) / 100, index=index)
    require(np.allclose(bootstrap(delta, 100), reference_bootstrap(delta, 100), rtol=0, atol=1e-18),
            "Bootstrap differs from original implementation")
    values = delta.groupby(level=STRATA).mean().to_numpy()
    require(sign_flip(values) == reference_sign_flip(values), "Sign-flip differs from original implementation")


def complete_zero_shot_metrics(out: Path, cache_root: Path) -> None:
    """Complete canonical ECE after the event audit, without rerunning a policy.

    Confidence is softmax(100 * raw cached per-vector dot products), preserving
    the prediction reference used by the archived run_streams.py.  Do not use a
    batched matrix multiplication: floating-point near ties can change argmax.
    """
    audit_path = out / "comparison_audit.json"
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    require(report["status"] == "PASS", "Canonical metrics require a passing event audit")
    combined = pd.read_csv(out / "anchor_and_selector_per_run.csv")
    canonical = combined[combined.strategy == "raw_zero_shot"].copy()
    cache_manifest = pd.read_csv(out / "cache_provenance.csv")
    ece_rows = []
    for (dataset, backbone), group in canonical.groupby(["dataset", "backbone"]):
        prefix = next(name for name, display in DATASETS.items() if display == dataset)
        path = cache_root / f"{prefix}_{backbone}.npz"
        known = cache_manifest[(cache_manifest.dataset == dataset) & (cache_manifest.backbone == backbone)].iloc[0]
        require(sha256(path) == known.sha256, f"Cache changed after event audit: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            features, text, labels = loaded["features"], loaded["text_features"], loaded["labels"]
            predictions = np.empty(len(features), dtype=np.int64)
            confidence = np.empty(len(features), dtype=np.float64)
            for index, feature in enumerate(features):
                similarities = feature @ text.T
                predictions[index] = int(similarities.argmax())
                probabilities = softmax(100.0 * similarities)
                confidence[index] = probabilities[predictions[index]]
        # Every original sample occurs exactly once in any one event stream.
        event_path = Path(group.iloc[0].source_path)
        events = pd.read_csv(event_path, usecols=["original_index", "zero_shot_prediction", "zero_shot_correct"])
        order = events.original_index.to_numpy()
        require(np.array_equal(predictions[order], events.zero_shot_prediction),
                f"Local raw predictions differ from canonical event reference: {event_path}")
        correct = (predictions == labels).astype(np.float64)
        require(np.array_equal(correct[order], events.zero_shot_correct), f"Raw correctness differs: {event_path}")
        ece = expected_calibration_error(confidence, correct)
        require(np.allclose(group.online_accuracy, correct.mean(), rtol=0, atol=1e-15), f"Canonical accuracy differs: {path}")
        canonical.loc[group.index, "expected_calibration_error"] = ece
        canonical.loc[group.index, "confidence_logit_scale"] = 100.0
        canonical.loc[group.index, "gain_over_zero_shot"] = 0.0
        canonical.loc[group.index, "pseudo_memory_size"] = 0
        canonical.loc[group.index, "pseudo_memory_error_rate"] = 0.0
        ece_rows.append({"dataset": dataset, "backbone": backbone, "samples": len(features),
                         "expected_calibration_error": ece, "all_raw_predictions_match_original_events": True})
    require(len(canonical) == 90 and canonical.expected_calibration_error.notna().all(), "Incomplete canonical zero-shot ECE")
    write_frame(out, "canonical_zero_shot_per_run", canonical)
    metrics = ["online_accuracy", "worst_window_accuracy", "negative_adaptation_rate", "expected_calibration_error"]
    group_rows = []
    for scope, part in [("All", canonical)] + list(canonical.groupby("dataset")):
        group_rows.append({"dataset_scope": scope, "runs": len(part),
            **{metric + "_percent": float(100 * part[metric].mean()) for metric in metrics}, "mean_queries": 0.0})
    write_frame(out, "canonical_zero_shot_dataset_means", group_rows)
    write_frame(out, "canonical_zero_shot_dataset_backbone_means", canonical.groupby(["dataset", "backbone"])[metrics].mean().reset_index())
    report["canonical_zero_shot"] = {
        "runs": 90, "confidence_convention": "softmax(100 * raw cached feature @ text_features.T), per-vector operations in original cache dtype",
        "ece": "15 equal-width confidence bins, same expected_calibration_error function as run_streams.py",
        "all_predictions_verified_against_original_events": True, "cache_results": ece_rows,
        "aggregation": "equal mean over runs, matching main selector table; not sample-weighted across datasets",
        "reference_scope": "uses the exact original raw zero-shot reference defining reported gain and harmful-adaptation metrics"}
    audit_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def write_manifest(out: Path, reference_csv: Path) -> None:
    manifest = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in sorted(out.iterdir()) if path.is_file() and path.name != "manifest.json"]
    manifest.extend({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in [Path(__file__), Path("src/analyze_v2_confirmation.py"), Path("src/run_review_supplement.py"), Path("src/run_streams.py"), reference_csv])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-root", type=Path, default=Path("remote_artifacts/20260814_multibench_v6/results/v2_confirmation_v6"))
    parser.add_argument("--cache-root", type=Path, default=Path("remote_artifacts/20260814_multibench_v6/cache"))
    parser.add_argument("--anchor-root", type=Path, default=Path("analysis/review_revision/anchors"))
    parser.add_argument("--reference-csv", type=Path, default=Path("analysis/v2_confirmation/all_confirmation_runs.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/review_revision_round2/comparisons"))
    args = parser.parse_args()
    validate_statistics_implementation()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference_csv).set_index(KEY + ["strategy"])
    cache_archive_audit = json.loads((args.confirmation_root.parent / "multibench_cache_audit.json").read_text(encoding="utf-8"))
    expected_cache_hashes = {Path(row["path"]).name: row["sha256"] for row in cache_archive_audit["caches"]}
    caches, cache_rows = {}, []
    for path in sorted(args.cache_root.glob("*.npz")):
        prefix, backbone = path.stem.split("_", 1)
        dataset = DATASETS[prefix]
        actual_hash = sha256(path)
        require(actual_hash == expected_cache_hashes[path.name], f"Cache hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            caches[dataset, backbone] = {name: loaded[name] for name in ("labels", "domains", "paths")}
        cache_rows.append({"dataset": dataset, "backbone": backbone, "path": str(path),
                           "sha256": actual_hash, "archived_sha256_match": True})
    require(len(caches) == 6, "Expected six feature caches")

    summary_files = sorted(args.confirmation_root.rglob("*.summary.json"))
    event_files = sorted(args.confirmation_root.rglob("*.events.csv.gz"))
    require(len(summary_files) == len(event_files) == 630, "Expected 630 summary/event pairs")
    rows, audit_rows, true_zero_rows = [], [], []
    identities, configurations, zs_predictions = {}, {}, {}
    sample_total, queries_total = 0, 0
    event_columns = ["position", "original_index", "path", "domain", "label", "prediction", "correct",
                     "zero_shot_prediction", "zero_shot_correct", "action", "adapted", "queried", "query_revealed_at"]
    for number, path in enumerate(summary_files, 1):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(args.confirmation_root)
        backbone, dataset = rel.parts[:2]
        require(backbone in BACKBONES and dataset in DATASETS.values(), f"Unknown scope: {path}")
        require(summary["strategy"] in SELECTORS + [METHOD], f"Unknown selector: {path}")
        require(summary["seed"] in range(10, 15), f"Unexpected seed: {path}")
        require(summary["budget_percent"] == 1, f"Unexpected budget: {path}")
        event_path = path.with_name(summary["run_id"] + ".events.csv.gz")
        event_hash = sha256(event_path)
        require(event_hash == summary["event_sha256"], f"Event SHA-256 mismatch: {event_path}")
        events = pd.read_csv(event_path, usecols=event_columns, keep_default_na=False)
        cache = caches[dataset, backbone]
        order = stream_order(cache["labels"], cache["domains"], summary["stream"], summary["seed"])
        n = len(events)
        require(n == summary["samples"] == len(order), f"Sample count mismatch: {path}")
        require(np.array_equal(events.position, np.arange(n)), f"Positions not contiguous: {path}")
        require(np.array_equal(events.original_index, order), f"Stream/seed order mismatch: {path}")
        require(np.array_equal(events.label, cache["labels"][order]), f"Cache label mismatch: {path}")
        require(np.array_equal(events.domain, cache["domains"][order]), f"Cache domain mismatch: {path}")
        require(np.array_equal(events.path.astype(str), cache["paths"][order].astype(str)), f"Cache path mismatch: {path}")
        require(np.array_equal(events.correct, (events.prediction == events.label).astype(int)), f"Wrong correctness: {path}")
        require(np.array_equal(events.zero_shot_correct, (events.zero_shot_prediction == events.label).astype(int)), f"Wrong zero-shot correctness: {path}")
        require(events.correct.isin([0, 1]).all(), f"Nonbinary correctness: {path}")
        require(events.action.isin(["predict", "adapt", "ask"]).all(), f"Unknown event action: {path}")
        require(np.array_equal(events.action == "ask", events.queried == 1), f"Ask flag mismatch: {path}")
        require(np.array_equal(events.action == "adapt", events.adapted == 1), f"Adapt flag mismatch: {path}")
        require(not ((events.adapted == 1) & (events.queried == 1)).any(), f"Double intervention: {path}")
        queried = events.queried == 1
        require((events.loc[queried, "query_revealed_at"].astype(int) >= events.loc[queried, "position"]).all(), f"Early label reveal: {path}")
        require(int(events.queried.sum()) == summary["query_count"] == summary["budget_count"], f"Budget mismatch: {path}")
        scope = (dataset, backbone, summary["stream"], summary["seed"])
        identity_hash = hashlib.sha256(events[["position", "original_index", "path", "domain", "label"]].to_csv(index=False).encode()).hexdigest()
        if scope in identities:
            require(identity_hash == identities[scope], f"Unpaired sample identity: {path}")
            require(summary["config"] == configurations[scope], f"Unpaired configuration: {path}")
            require(np.array_equal(events.zero_shot_prediction, zs_predictions[scope]), f"Unpaired zero-shot reference: {path}")
        else:
            identities[scope] = identity_hash
            configurations[scope] = summary["config"]
            zs_predictions[scope] = events.zero_shot_prediction.to_numpy()
            true_zero_rows.append(dict(zip(KEY, scope)) | {
                "strategy": "raw_zero_shot", "samples": n, "budget_count": 0, "query_count": 0,
                "online_accuracy": float(events.zero_shot_correct.mean()), "zero_shot_accuracy": float(events.zero_shot_correct.mean()),
                "negative_adaptation_rate": 0.0, "worst_window_accuracy": worst(events.zero_shot_correct.to_numpy(), 500),
                "source_path": str(event_path), "source_kind": "SHA-256-verified confirmation zero_shot_correct events",
            })
        correct = events.correct.to_numpy()
        computed = {"online_accuracy": float(correct.mean()), "zero_shot_accuracy": float(events.zero_shot_correct.mean()),
                    "negative_adaptation_rate": float(((events.zero_shot_correct == 1) & (events.correct == 0)).mean()),
                    "worst_window_accuracy": worst(correct, 500)}
        reference_key = (dataset, BACKBONES[backbone], summary["stream"], summary["seed"], summary["strategy"])
        ref = reference.loc[reference_key]
        require(ref.event_sha256 == event_hash, f"Existing analysis references another event file: {path}")
        for metric, value in computed.items():
            require(abs(value - summary[metric]) < 1e-12, f"Metric mismatch {metric}: {path}")
            require(abs(value - ref[metric]) < 1e-12, f"Existing analysis metric mismatch {metric}: {path}")
        base = dict(zip(KEY, scope)) | {"strategy": summary["strategy"], "samples": n,
            "online_accuracy": computed["online_accuracy"], "negative_adaptation_rate": computed["negative_adaptation_rate"],
            "budget_count": summary["budget_count"], "query_count": summary["query_count"],
            "zero_shot_accuracy": computed["zero_shot_accuracy"], "source_path": str(path)}
        for window in WINDOWS:
            rows.append(base | {"window": window, "worst_window_accuracy": worst(correct, window)})
        audit_rows.append(dict(zip(KEY, scope)) | {"strategy": summary["strategy"], "summary_path": str(path),
            "summary_sha256": sha256(path), "event_path": str(event_path), "event_sha256": event_hash,
            "identity_sha256": identity_hash, "samples": n, "queries": int(events.queried.sum()), "status": "PASS"})
        sample_total += n
        queries_total += int(events.queried.sum())
        if number % 70 == 0:
            print(f"Audited {number}/630 logs ({sample_total:,} events)", flush=True)
    require(len(identities) == 90, "Expected 90 paired conditions")
    frame = pd.DataFrame(rows)
    write_frame(args.output, "confirmation_event_audit", audit_rows)
    write_frame(args.output, "window_per_run", frame)
    write_frame(args.output, "cache_provenance", cache_rows)

    paired_rows, stratum_rows, stat_rows = [], [], []
    for window in WINDOWS:
        matrix = frame[frame.window == window].pivot(index=KEY, columns="strategy", values="worst_window_accuracy")
        require(matrix.shape == (90, 7) and not matrix.isna().any().any(), "Incomplete paired window matrix")
        window_stats = []
        for baseline in SELECTORS:
            delta = matrix[METHOD] - matrix[baseline]
            row = {"window": window, "metric": "worst_window_accuracy", "baseline": baseline, **statistics(delta)}
            window_stats.append(row)
            for key, value in delta.items():
                paired_rows.append(dict(zip(KEY, key)) | {"window": window, "baseline": baseline,
                    "method_worst_accuracy": float(matrix.loc[key, METHOD]),
                    "baseline_worst_accuracy": float(matrix.loc[key, baseline]), "delta_pp": float(value * 100)})
            for key, value in delta.groupby(level=STRATA).mean().items():
                stratum_rows.append(dict(zip(STRATA, key)) | {"window": window, "baseline": baseline, "mean_delta_pp": float(value * 100)})
        adjusted = holm({row["baseline"]: row["p_sign_flip"] for row in window_stats})
        for row in window_stats:
            row["p_holm"] = adjusted[row["baseline"]]
            row["holm_family"] = "six comparator contrasts within window"
        stat_rows.extend(window_stats)
    all_adjusted = holm({f"{row['window']}:{row['baseline']}": row["p_sign_flip"] for row in stat_rows})
    for row in stat_rows:
        row["p_holm_all_18"] = all_adjusted[f"{row['window']}:{row['baseline']}"]
    archived_stats = pd.read_csv(args.reference_csv.with_name("paired_statistics.csv"))
    archived_500 = archived_stats[archived_stats.metric == "worst_window_accuracy"].set_index("baseline")
    new_500 = pd.DataFrame(stat_rows).query("window == 500").set_index("baseline").loc[archived_500.index]
    check_columns = ["mean_delta_pp", "ci95_low_pp", "ci95_high_pp", "p_sign_flip", "p_holm"]
    require(np.allclose(new_500[check_columns], archived_500[check_columns], rtol=0, atol=1e-12),
            "Recomputed 500-window statistics differ from archived analysis")
    write_frame(args.output, "window_paired_statistics", stat_rows)
    write_frame(args.output, "window_paired_differences", paired_rows)
    write_frame(args.output, "window_stratum_differences", stratum_rows)
    window_means = frame.groupby(["window", "strategy"])["worst_window_accuracy"].agg(["mean", "std", "count"]).reset_index()
    write_frame(args.output, "window_strategy_means", window_means)

    anchor_rows, anchor_audit = [], []
    for path in sorted(args.anchor_root.rglob("*.summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        prefix, backbone = path.parent.name.split("_", 1)
        dataset = DATASETS[prefix]
        scope = (dataset, backbone, summary["stream"], summary["seed"])
        require(scope in identities, f"Anchor condition not in confirmation: {path}")
        require(summary["config"] == configurations[scope], f"Anchor config differs: {path}")
        reference_row = frame[(frame.dataset == dataset) & (frame.backbone == backbone)
            & (frame.stream == summary["stream"]) & (frame.seed == summary["seed"])].iloc[0]
        require(summary["samples"] == reference_row.samples, f"Anchor sample count differs: {path}")
        require(abs(summary["zero_shot_accuracy"] - reference_row.zero_shot_accuracy) < 1e-12, f"Anchor raw reference differs: {path}")
        require(summary["strategy"] in ["zero_shot", "ask_only"], f"Unexpected anchor strategy: {path}")
        require(summary["event_sha256"] is None, f"Unexpected hashed anchor; inspect provenance: {path}")
        require(summary["pseudo_memory_size"] == 0, f"Pseudo updates in anchor: {path}")
        require(sum(summary["final_state"]["pseudo_count"]) == 0, f"Pseudo memory in anchor: {path}")
        expected_queries = 0 if summary["strategy"] == "zero_shot" else reference_row.budget_count
        require(summary["query_count"] == summary["budget_count"] == expected_queries, f"Anchor budget differs: {path}")
        anchor_rows.append(dict(zip(KEY, scope)) | {key: summary[key] for key in ["strategy", "samples", "budget_count", "query_count",
            "online_accuracy", "zero_shot_accuracy", "worst_window_accuracy", "negative_adaptation_rate"]}
            | {"source_path": str(path), "source_kind": "summary-only anchor; no recorded event hash"})
        anchor_audit.append(dict(zip(KEY, scope)) | {"strategy": summary["strategy"], "summary_path": str(path),
            "summary_sha256_now": sha256(path), "event_sha256_recorded": None,
            "event_file_exists": path.with_name(summary["run_id"] + ".events.csv.gz").exists(),
            "config_samples_budget_zero_shot_reference_match": True})
    require(len(anchor_rows) == 180, "Expected 180 anchors")
    require(not any(row["event_file_exists"] for row in anchor_audit), "Unexpected anchor events; inspect provenance")
    anchor_frame = pd.DataFrame(anchor_rows)
    selector_frame = frame[frame.window == 500].drop(columns="window").copy()
    selector_frame["source_kind"] = "SHA-256-verified confirmation events"
    combined = pd.concat([anchor_frame, selector_frame, pd.DataFrame(true_zero_rows)], ignore_index=True)
    write_frame(args.output, "anchor_provenance", anchor_audit)
    write_frame(args.output, "anchor_and_selector_per_run", combined)
    anchor_means = []
    for scope_name, data in [("All", combined)] + list(combined.groupby("dataset")):
        for strategy, part in data.groupby("strategy"):
            anchor_means.append({"dataset_scope": scope_name, "strategy": strategy, "runs": len(part),
                **{metric + "_percent": float(100 * part[metric].mean()) for metric in
                   ("online_accuracy", "worst_window_accuracy", "negative_adaptation_rate")},
                "mean_queries": float(part.query_count.mean())})
    write_frame(args.output, "anchor_dataset_means", anchor_means)
    group_means = combined.groupby(["dataset", "backbone", "strategy"])[
        ["online_accuracy", "worst_window_accuracy", "negative_adaptation_rate", "query_count"]].mean().reset_index()
    write_frame(args.output, "anchor_dataset_backbone_means", group_means)

    anchor_stats, anchor_deltas = [], []
    contrasts = [(METHOD, "raw_zero_shot"), ("ask_only", "raw_zero_shot"), (METHOD, "ask_only"), ("ask_only", "periodic")]
    for scope_name, data in [("All", combined)] + list(combined.groupby("dataset")):
        for metric in ("online_accuracy", "worst_window_accuracy", "negative_adaptation_rate"):
            matrix = data.pivot(index=KEY, columns="strategy", values=metric)
            family = []
            for method, baseline in contrasts:
                delta = matrix[method] - matrix[baseline]
                if metric == "negative_adaptation_rate":
                    delta = -delta
                contrast = f"{method}_minus_{baseline}"
                family.append({"dataset_scope": scope_name, "metric": metric, "method": method,
                    "baseline": baseline, "contrast": contrast, **statistics(delta)})
                for key, value in delta.items():
                    anchor_deltas.append(dict(zip(KEY, key)) | {"dataset_scope": scope_name, "metric": metric,
                        "method": method, "baseline": baseline, "delta_pp": float(value * 100)})
            adjusted = holm({row["contrast"]: row["p_sign_flip"] for row in family})
            for row in family:
                row["p_holm"] = adjusted[row["contrast"]]
                row["holm_family"] = "four anchor contrasts within dataset scope and metric"
            anchor_stats.extend(family)
    write_frame(args.output, "anchor_paired_statistics", anchor_stats)
    write_frame(args.output, "anchor_paired_differences", anchor_deltas)

    report = {
        "status": "PASS", "analysis_type": "post-hoc paired sensitivity and supplementary anchor analysis",
        "confirmation": {"runs": 630, "events": sample_total, "queries": queries_total,
            "paired_conditions": len(identities), "strata": 18, "seeds": list(range(10, 15)),
            "event_hashes_verified": 630, "cache_hashes_match_archived_audit": 6,
            "all_event_orders_paths_labels_domains_match_cache_stream_seed": True,
            "all_selector_zero_shot_predictions_match_within_condition": True,
            "summary_metrics_match_events_and_existing_confirmation_csv": True},
        "statistics": {"direction": "positive favors method; harmful adaptation differences are sign-reversed",
            "paired_unit": "one dataset x backbone x stream x seed run",
            "sign_flip_unit": "mean paired difference over five seeds in each dataset x backbone x stream stratum",
            "sign_flip": "exact two-sided enumeration, observed absolute mean, tolerance 1e-15, no plus-one correction",
            "bootstrap": "20000 percentile resamples of five seeds within every fixed stratum; RNG 20260814; no stratum resampling",
            "bootstrap_scope": "seed uncertainty conditional on these datasets, backbones, and streams; not unseen-domain uncertainty",
            "holm": "six selectors within each window, matching original per-metric family; all-18 adjustment also reported",
            "implementation_crosscheck_against_original_source": "PASS",
            "recomputed_500_statistics_match_archived_paired_statistics_at_1e_12": True,
            "windows": list(WINDOWS), "window_metric": "minimum over all valid contiguous fixed-width windows of each run",
            "dataset_anchor_tests": "six strata per dataset; 18 strata pooled; four contrasts per metric/scope; post-hoc"},
        "anchors": {"runs": 180, "strategies": ["zero_shot", "ask_only"], "event_files": 0,
            "non_null_recorded_event_hashes": 0, "event_audited": False,
            "provenance": "run_review_supplement.py invokes run_one(write_events=False). Summary config, samples, seed/stream, budget, and raw zero-shot reference agree with confirmation. Anchor summaries do not record cache hashes or source-code hashes; current summary hashes do not retrospectively establish execution provenance.",
            "ask_only_policy": "exact 1% budget; periodic segment-end selection; supervised state/weight updates; no pseudo-label updates",
            "causal_caveat": "AoA versus Ask-only changes both selector and pseudo-update policy, so it is not a pure pseudo-update ablation. Ask-only versus Periodic keeps segment-end selection and isolates disabling pseudo updates within the shared learner (subject to summary-only provenance).",
            "zero_shot_caveat": "The stored zero_shot online_accuracy is the unchanged four-expert controller. It can differ numerically from the raw float32 dot-product zero_shot_prediction reference; nonzero negative_adaptation_rate in that stored anchor does not represent adaptation. Use raw_zero_shot for performance/gain comparisons and retain the stored controller only as an implementation reference.",
            "raw_zero_shot_source": "recomputed from SHA-256-verified confirmation zero_shot_correct events, identical across seven selectors for each condition; zero queries and harmful-adaptation rate zero by definition"},
        "window_statistics": stat_rows,
    }
    (args.output / "comparison_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    complete_zero_shot_metrics(args.output, args.cache_root)
    write_manifest(args.output, args.reference_csv)
    print(pd.DataFrame(stat_rows).to_string(index=False))
    print(pd.DataFrame(anchor_means).query("strategy in ['ask_only', 'ask_or_adapt_v2', 'raw_zero_shot', 'zero_shot', 'periodic']").to_string(index=False))
    print(json.dumps(report["confirmation"], indent=2))


if __name__ == "__main__":
    main()
