"""Post-hoc, one-at-a-time threshold diagnostics; never a model-selection run.

Evaluate seven fixed configurations on all six frozen caches, three streams,
seed 10, exact 1% annotation budget, and an error-free delayed oracle. No
configuration is selected and the frozen confirmatory defaults remain intact.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from a2a.core import RunConfig
from run_streams import hash_file, run_one


STREAMS = ("iid_random", "abrupt_domain", "class_correlated")
CONFIGURATIONS = (
    ("default", {}),
    ("gamma_0.70", {"pseudo_confidence": 0.70}),
    ("gamma_0.90", {"pseudo_confidence": 0.90}),
    ("switch_0.25", {"persistence_switch_rate": 0.25}),
    ("switch_0.45", {"persistence_switch_rate": 0.45}),
    ("entropy_0.40", {"risk_entropy_threshold": 0.40}),
    ("entropy_0.60", {"risk_entropy_threshold": 0.60}),
)
DATASETS = {"pacs": "PACS", "officehome": "OfficeHome", "terra": "TerraIncognita"}
BACKBONES = {"openai_vitb16": "OpenAI ViT-B/16", "openclip_vitl14": "OpenCLIP ViT-L/14"}
EXPECTED_CACHES = {f"{dataset}_{backbone}" for dataset in DATASETS for backbone in BACKBONES}
METRICS_TO_VERIFY = (
    "samples", "budget_count", "query_count", "online_accuracy",
    "zero_shot_accuracy", "gain_over_zero_shot", "worst_window_accuracy",
    "negative_adaptation_rate", "expected_calibration_error",
    "label_efficiency_pp_per_100_queries", "pseudo_memory_error_rate",
    "pseudo_memory_size",
)


@lru_cache(maxsize=1)
def load_cache(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def one(job):
    cache_path, output_root, stream, config_id, overrides = job
    path = Path(cache_path)
    cache = load_cache(cache_path)
    dataset_key, backbone_key = path.stem.split("_", 1)
    config = RunConfig(num_classes=len(cache["classes"]), **overrides)
    output_dir = Path(output_root) / config_id / path.stem
    summary = run_one(
        cache, output_dir, stream, 10, 1.0, "ask_or_adapt_v2", config,
        write_events=False, oracle_noise=0.0,
    )
    if summary["query_count"] != summary["budget_count"]:
        raise AssertionError(f"exact-budget violation: {path.stem} {stream} {config_id}")
    summary_path = output_dir / f"{summary['run_id']}.summary.json"
    return {
        "configuration": config_id,
        "cache": path.stem,
        "dataset": DATASETS[dataset_key],
        "backbone": BACKBONES[backbone_key],
        "pseudo_confidence": config.pseudo_confidence,
        "persistence_switch_rate": config.persistence_switch_rate,
        "risk_entropy_threshold": config.risk_entropy_threshold,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": hash_file(summary_path),
        **{key: value for key, value in summary.items() if key not in {"config", "final_state"}},
        "config_json": json.dumps(asdict(config), sort_keys=True),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    deltas = np.asarray([row["delta_accuracy_pp"] for row in rows])
    return {
        "conditions": len(rows),
        "mean_online_accuracy_percent": 100.0 * float(np.mean([row["online_accuracy"] for row in rows])),
        "mean_paired_delta_pp": float(deltas.mean()),
        "min_paired_delta_pp": float(deltas.min()),
        "max_paired_delta_pp": float(deltas.max()),
        "mean_absolute_paired_delta_pp": float(np.abs(deltas).mean()),
        "lower_than_default": int(np.sum(deltas < 0)),
        "equal_to_default": int(np.sum(deltas == 0)),
        "higher_than_default": int(np.sum(deltas > 0)),
    }


def verify_defaults(rows: list[dict], reference: Path) -> dict:
    with reference.open(newline="", encoding="utf-8") as handle:
        existing = [r for r in csv.DictReader(handle)
                    if r["strategy"] == "ask_or_adapt_v2" and int(r["seed"]) == 10
                    and float(r["budget_percent"]) == 1.0]
    reference_rows = {(r["dataset"], r["backbone"], r["stream"]): r for r in existing}
    if len(reference_rows) != 18 or len(existing) != 18:
        raise AssertionError(f"expected 18 unique original default rows, got {len(existing)}")
    errors = []
    max_errors = {metric: 0.0 for metric in METRICS_TO_VERIFY}
    for row in (row for row in rows if row["configuration"] == "default"):
        key = (row["dataset"], row["backbone"], row["stream"])
        original = reference_rows[key]
        if json.loads(row["config_json"]) != ast.literal_eval(original["config"]):
            errors.append({"condition": key, "metric": "config"})
        for metric in METRICS_TO_VERIFY:
            error = abs(float(row[metric]) - float(original[metric]))
            max_errors[metric] = max(max_errors[metric], error)
            if error > 1e-12:
                errors.append({"condition": key, "metric": metric, "absolute_error": error})
    return {
        "reference_path": str(reference.resolve()),
        "reference_sha256": hash_file(reference),
        "compared_default_conditions": 18,
        "absolute_tolerance": 1e-12,
        "all_pass": not errors,
        "max_absolute_errors": max_errors,
        "mismatches": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference-csv", type=Path, default=Path("analysis/v2_confirmation/all_confirmation_runs.csv"))
    args = parser.parse_args()
    caches = sorted(args.cache_root.glob("*.npz"))
    if {path.stem for path in caches} != EXPECTED_CACHES:
        raise ValueError("Expected exactly the six PACS/OfficeHome/Terra × two-backbone caches")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve(), source_root / "a2a/core.py", source_root / "run_streams.py"]
    metadata = {
        "purpose": "Post-hoc one-at-a-time sensitivity diagnostic, not confirmatory evidence or hyperparameter selection.",
        "selection_policy": "No threshold is selected; default remains gamma=0.80, switch_rate=0.35, risk_entropy=0.50.",
        "scope": "All six frozen feature caches × three streams × seed 10 × seven fixed configurations; clean delayed oracle; exact 1% budget.",
        "limits": "Single seed; frozen-feature stream replay; one-at-a-time grid does not identify threshold interactions or seed variability.",
        "seed": 10,
        "budget_percent": 1.0,
        "oracle_noise": 0.0,
        "strategy": "ask_or_adapt_v2",
        "streams": STREAMS,
        "configurations": [{"id": name, "config_template": asdict(RunConfig(num_classes=0, **overrides)),
                             "note": "num_classes is filled from each frozen cache"}
                            for name, overrides in CONFIGURATIONS],
        "expected_run_count": 126,
        "all_defaults_rerun": True,
        "cache_sha256": {str(path.resolve()): hash_file(path) for path in caches},
        "source_sha256": {str(path): hash_file(path) for path in sources},
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                    "executable": sys.executable, "workers": args.workers,
                    "blas_thread_environment": {key: os.environ.get(key) for key in
                                                ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}},
        "argv": sys.argv,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    jobs = [(str(cache.resolve()), str(args.output_dir.resolve()), stream, name, overrides)
            for cache in caches for stream in STREAMS for name, overrides in CONFIGURATIONS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"{count}/{len(jobs)} {row['cache']} {row['stream']} {row['configuration']} accuracy={row['online_accuracy']:.9f}", flush=True)
    default = {(r["cache"], r["stream"]): r for r in rows if r["configuration"] == "default"}
    if len(default) != 18 or len(rows) != 126:
        raise AssertionError("incomplete run grid")
    for row in rows:
        reference = default[(row["cache"], row["stream"])]
        row["default_online_accuracy"] = reference["online_accuracy"]
        row["delta_accuracy_pp"] = 100.0 * (row["online_accuracy"] - reference["online_accuracy"])
    rows.sort(key=lambda r: (r["configuration"], r["cache"], r["stream"]))
    write_csv(args.output_dir / "per_run.csv", rows)
    verification = verify_defaults(rows, args.reference_csv)
    summary = {
        "scope": metadata["scope"], "purpose": metadata["purpose"], "limits": metadata["limits"],
        "selection_policy": metadata["selection_policy"],
        "aggregation": "Unweighted arithmetic mean over paired dataset × backbone × stream conditions; deltas are variant minus rerun default in percentage points.",
        "default_verification": verification,
        "by_configuration": {name: aggregate([r for r in rows if r["configuration"] == name])
                             for name, _ in CONFIGURATIONS},
    }
    for group in ("dataset", "backbone", "stream", "cache"):
        summary[f"by_{group}"] = {
            value: {name: aggregate([r for r in rows if r[group] == value and r["configuration"] == name])
                    for name, _ in CONFIGURATIONS}
            for value in sorted({r[group] for r in rows})
        }
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["per_run_sha256"] = hash_file(args.output_dir / "per_run.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    aggregate_rows = [{"configuration": name, **values} for name, values in summary["by_configuration"].items()]
    write_csv(args.output_dir / "summary.csv", aggregate_rows)
    dataset_rows = [{"dataset": dataset, "configuration": name, **values}
                    for dataset, configurations in summary["by_dataset"].items()
                    for name, values in configurations.items()]
    write_csv(args.output_dir / "summary_by_dataset.csv", dataset_rows)
    print(json.dumps(summary["by_configuration"], indent=2), flush=True)
    if not verification["all_pass"]:
        raise SystemExit("Default verification mismatches recorded in summary.json; inspect before interpretation")
    print("All 126 runs finished; all 18 defaults match confirmation metrics to 1e-12.", flush=True)


if __name__ == "__main__":
    main()
