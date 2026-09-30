"""Post-hoc paired anchor CIs and evidence audit for five-percent leakage.

No policy is rerun. No event-level data are synthesized. Original source files
are read only; outputs contain the matched differences and explicit provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_revision_comparisons import bootstrap as original_bootstrap
from analyze_revision_comparisons import validate_statistics_implementation

KEY = ["dataset", "backbone", "stream", "seed"]
STRATA = KEY[:-1]
BACKBONES = {"OpenAI ViT-B/16": "openai_vitb16", "OpenCLIP ViT-L/14": "openclip_vitl14"}
STRATEGY = "ask_or_adapt_v2"


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def cluster_ci(values, draws=100000, seed=20260814):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (draws, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/review_revision_round3/paired_anchor_ci")
    parser.add_argument("--cluster-draws", type=int, default=100000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def path(relative):
        value = ROOT / relative
        inputs[str(value.relative_to(ROOT))] = {"sha256": sha256(value), "bytes": value.stat().st_size}
        return value

    def read_csv(relative):
        result = pd.read_csv(path(relative))
        if "backbone" in result:
            result["backbone"] = result.backbone.replace(BACKBONES)
        return result

    validate_statistics_implementation()
    path("src/analyze_revision_comparisons.py")
    path("src/analyze_v2_confirmation.py")
    path(Path(__file__).resolve().relative_to(ROOT))
    confirmation = read_csv("analysis/v2_confirmation/all_confirmation_runs.csv")
    aoa = confirmation[confirmation.strategy == STRATEGY].copy()
    combined = read_csv("analysis/review_revision_round2/comparisons/anchor_and_selector_per_run.csv")
    ask = combined[combined.strategy == "ask_only"].copy()
    zero = read_csv("analysis/review_revision_round2/comparisons/canonical_zero_shot_per_run.csv")
    tda = read_csv("analysis/review_revision_round2/tda/gpu_comparison/tda_gpu_matched_runs.csv")
    expected = {(d, b, s, seed) for d in ["PACS", "OfficeHome", "TerraIncognita"]
                for b in ["openai_vitb16", "openclip_vitl14"]
                for s in ["iid_random", "abrupt_domain", "class_correlated"] for seed in range(10, 15)}
    for label, frame in [("AoA", aoa), ("Ask-only", ask), ("zero-shot", zero), ("TDA", tda)]:
        require(len(frame) == 90 and not frame.duplicated(KEY).any(), f"Invalid {label} run grid")
        require(set(frame[KEY].itertuples(index=False, name=None)) == expected, f"Incomplete {label} grid")
        require(np.isfinite(frame[["online_accuracy", "worst_window_accuracy"]]).all().all(), f"Missing {label} metrics")
    # Anchor metrics have summary provenance only. Verify the actual 90 files,
    # without incorrectly describing those checksums as historical event hashes.
    for row in ask.itertuples(index=False):
        summary_path = ROOT / row.source_path
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary["event_sha256"] is None, "Unexpected Ask-only provenance")
        require(summary["seed"] == row.seed and summary["stream"] == row.stream, "Anchor identity mismatch")
        require(summary["samples"] == row.samples and summary["query_count"] == row.query_count, "Anchor size mismatch")
        require(np.isclose(summary["online_accuracy"], row.online_accuracy, rtol=0, atol=1e-15), "Anchor accuracy mismatch")
        path(row.source_path)
    comparisons, paired_rows, stratum_rows = [], [], []
    specifications = [("Ask-only minus Ask-or-Adapt", ask, "online_accuracy"),
                      ("TDA minus Ask-or-Adapt", tda, "worst_window_accuracy"),
                      ("Zero-shot minus Ask-or-Adapt", zero, "worst_window_accuracy")]
    for label, other, metric in specifications:
        left = other.set_index(KEY).sort_index()
        right = aoa.set_index(KEY).sort_index()
        require(left.index.equals(right.index), f"Pairing failure: {label}")
        require(np.array_equal(left.samples, right.samples), f"Sample-count mismatch: {label}")
        require(np.allclose(left.zero_shot_accuracy, right.zero_shot_accuracy, rtol=0, atol=1e-12),
                f"Frozen-reference mismatch: {label}")
        delta = left[metric] - right[metric]
        strata = delta.groupby(level=STRATA).mean()
        require(len(strata) == 18 and (delta.groupby(level=STRATA).size() == 5).all(), "Expected 18x5 matched grid")
        primary = cluster_ci(100 * strata.to_numpy(), args.cluster_draws)
        fixed = [100 * value for value in original_bootstrap(delta, 20000)]
        cache_means = delta.groupby(level=["dataset", "backbone"]).mean()
        dataset_means = delta.groupby(level="dataset").mean()
        result = {"comparison": label, "metric": metric, "paired_runs": 90, "strata": 18,
                  "seeds_per_stratum": 5, "mean_delta_pp": float(100 * strata.mean()),
                  "cluster_18_ci95_pp": primary,
                  "fixed_18_strata_seed_bootstrap_ci95_pp": fixed,
                  "sensitivity_cluster_6_cache_ci95_pp": cluster_ci(100 * cache_means.to_numpy(), args.cluster_draws),
                  "sensitivity_cluster_3_dataset_ci95_pp": cluster_ci(100 * dataset_means.to_numpy(), args.cluster_draws),
                  "dataset_mean_delta_pp": {key: float(100 * value) for key, value in dataset_means.items()}}
        comparisons.append(result)
        for keys, value in delta.items():
            paired_rows.append({"comparison": label, "metric": metric, **dict(zip(KEY, keys)),
                                "baseline_value": float(left.loc[keys, metric]),
                                "aoa_value": float(right.loc[keys, metric]), "delta_pp": float(100 * value)})
        for keys, value in strata.items():
            stratum_rows.append({"comparison": label, "metric": metric, **dict(zip(STRATA, keys)),
                                 "mean_delta_pp": float(100 * value), "paired_seeds": 5})
    pd.DataFrame(paired_rows).to_csv(args.output / "paired_differences.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(args.output / "stratum_differences.csv", index=False)

    # Check the complete archived five-percent grid, its launch command, and the
    # archive itself. Absence of queried correctness is not replaced by a model.
    archive_base = "remote_artifacts/20260814_posthoc_v7/complete_archive"
    result_root = ROOT / archive_base / "extracted/results/posthoc_robustness_v7"
    budget_files = sorted(result_root.rglob("*__seed*__budget5__ask_or_adapt_v2.summary.json"))
    budget_rows = []
    for file in budget_files:
        summary = json.loads(file.read_text(encoding="utf-8"))
        backbone, dataset, subdir = file.relative_to(result_root).parts[:3]
        require(subdir == "budget" and summary["budget_percent"] == 5.0, "Unexpected budget scope")
        require(summary["event_sha256"] is None and summary["oracle_noise"] == 0, "Unexpected budget provenance")
        require(summary["query_count"] == summary["budget_count"] == round(.05 * summary["samples"]), "Budget mismatch")
        budget_rows.append({"dataset": dataset, "backbone": backbone, "stream": summary["stream"],
                            "seed": summary["seed"], "samples": summary["samples"], "queries": summary["query_count"],
                            "actual_budget_percent": 100 * summary["query_count"] / summary["samples"],
                            "event_sha256": summary["event_sha256"], "summary_path": str(file.relative_to(ROOT)),
                            "summary_sha256": sha256(file)})
    budget_frame = pd.DataFrame(budget_rows)
    budget_expected = {(d, b, s, seed + 10) for d, b, s, seed in expected}
    require(len(budget_frame) == 90 and set(budget_frame[KEY].itertuples(index=False, name=None)) == budget_expected,
            "Five-percent archived grid incomplete")
    budget_frame.to_csv(args.output / "budget5_summary_inventory.csv", index=False)
    launcher_path = path(archive_base + "/extracted/project/scripts/run_posthoc_robustness_v7.sh")
    launcher = launcher_path.read_text(encoding="utf-8")
    budget_command = launcher.split('--output-dir "$base/budget"', 1)[1].split('"$PY" "$RUN"', 1)[0]
    require("--summary-only" in budget_command and "--seeds 20 21 22 23 24" in budget_command,
            "Archived launcher does not establish summary-only seed20-24 provenance")
    posthoc_audit = json.loads(path(archive_base + "/extracted/results/posthoc_robustness_v7/audit.json").read_text())
    archive_path = path(archive_base + "/posthoc_v7_complete.tar.gz")
    declared_sha = path(archive_base + "/posthoc_v7_complete.tar.gz.sha256").read_text().split()[0]
    require(sha256(archive_path) == declared_sha, "Complete archive checksum mismatch")
    with tarfile.open(archive_path) as archive:
        members = archive.getnames()
    archive_budget = [name for name in members if "budget5__ask_or_adapt_v2" in name]
    event_members = [name for name in archive_budget if ".events." in name]
    local_budget_events = [str(p.relative_to(ROOT)) for p in (ROOT / "remote_artifacts").rglob("*budget5__ask_or_adapt_v2.events*")]
    require(not event_members and not local_budget_events, "Five-percent logs found; must analyze rather than report unavailable")
    require(sum(name.endswith(".summary.json") for name in archive_budget) == 90, "Archive budget summary count mismatch")

    # Validate the exact identity against the existing one-percent event-derived
    # count table. This does not supply any missing five-percent observation.
    leak = read_csv("analysis/label_leakage/label_leakage_per_condition.csv")
    require(len(leak) == 90, "Expected 90 one-percent leakage records")
    per_run_pp = 100 * leak.queried_errors / leak.samples
    require(np.allclose(per_run_pp, leak.online_inflation_pp, rtol=0, atol=1e-11), "Leakage identity failed")
    require(np.allclose(per_run_pp, 100 * leak.queries / leak.samples * leak.queried_error_rate, rtol=0, atol=1e-11),
            "Budget-times-queried-error identity failed")
    pooled_budget_percent = float(100 * leak.queries.sum() / leak.samples.sum())
    pooled_error_rate = float(leak.queried_errors.sum() / leak.queries.sum())
    original_example = {"budget_nominal_percent": 1, "runs": 90, "samples": int(leak.samples.sum()),
                        "queries": int(leak.queries.sum()), "queried_errors": int(leak.queried_errors.sum()),
                        "macro_online_inflation_pp": float(per_run_pp.mean()),
                        "micro_online_inflation_pp": float(100 * leak.queried_errors.sum() / leak.samples.sum()),
                        "pooled_actual_budget_percent": pooled_budget_percent,
                        "pooled_queried_error_rate": pooled_error_rate,
                        "micro_product_pp": pooled_budget_percent * pooled_error_rate,
                        "macro_queried_error_rate": float(leak.queried_error_rate.mean()),
                        "identity_max_absolute_error_pp": float(abs(per_run_pp - leak.online_inflation_pp).max())}
    report = {
        "status": "COMPLETE_WITH_UNAVAILABLE_5_PERCENT_LEAKAGE",
        "comparisons": comparisons,
        "method": {
            "estimand": "Equal mean of 18 dataset/backbone/stream stratum means, each averaging five matched seed differences; equal to mean over the balanced90 grid.",
            "pairing": KEY, "difference_sign": "Named comparator minus Ask-or-Adapt; positive means comparator higher.",
            "primary_interval": "Retain all18 strata, independently resample5 paired seeds with replacement within each, average seed means then strata;20,000 draws and rng20260814 exactly match analyze_v2_confirmation.py. Conditions on these fixed benchmark strata.",
            "cluster_18_sensitivity": f"Percentile paired cluster bootstrap: sample18 seed-averaged stratum means with replacement, average them, {args.cluster_draws} draws, numpy default_rng(20260814), quantiles .025/.975.",
            "coarser_cluster_sensitivities": "Average all matched differences within each of6 dataset/backbone caches or3 datasets, then resample those whole units with replacement with the same draw count/seed. Exploratory; three datasets cannot establish broad population generalization.",
            "dependence_limit": "Streams and backbones reuse dataset images;18 strata are not18 independent datasets. Cluster18 intervals are descriptive benchmark-heterogeneity intervals, not population/domain generalization guarantees.",
            "multiplicity": "Three post-hoc marginal95% percentile intervals, not simultaneous/multiplicity-adjusted intervals. No new significance family is claimed.",
            "provenance_limit": "Ask-only is summary-only, not event audited; paired metrics do not remove that limitation. Its comparison also changes query selection and pseudo-updates. TDA uses zero labels/current-feature updates, unlike delayed1% supervision.",
        },
        "score_repair_identity": {
            "definitions": "For run r, T_r samples, Q_r actual integer queries, K_r queried predictions originally incorrect; c_rt in{0,1}, q_rt in{0,1}, beta_r=Q_r/T_r, e_r=K_r/Q_r. In manuscript Section3, b denotes the requested percentage, so Q_r=round(T_r*b/100), subject to its positive-budget minimum.",
            "per_sample": "c'_rt=c_rt+q_rt*(1-c_rt); future predictions and states unchanged.",
            "run_fraction": "Delta_r=K_r/T_r=beta_r*e_r.",
            "run_percentage_points": "Delta_r_pp=100*K_r/T_r=(100*Q_r/T_r)*e_r.",
            "macro_percentage_points": "Delta_macro_pp=(100/R)*sum_r(K_r/T_r)=(1/R)*sum_r(actual_budget_percent_r*e_r). Do not multiply nominal budget by a pooled error rate.",
            "micro_percentage_points": "Delta_micro_pp=100*sum_r(K_r)/sum_r(T_r)=[100*sum_r(Q_r)/sum_r(T_r)]*[sum_r(K_r)/sum_r(Q_r)].",
            "rounding": "Use realized Q_r/T_r. If every actual budget fraction equals beta, macro Delta=beta*mean_r(e_r), but the pooled queried-error rate remains a different weighting. Linearity across budgets is only conditional on a fixed queried-error rate; adaptation/selection need not preserve it.",
            "worst_window": "No analogous global b*e identity: Delta_worst=min_w[a_w+K_queried_error,w/W]-min_w[a_w], because the minimizing window may change.",
            "one_percent_observed_identity_check": original_example,
        },
        "five_percent_leakage": {
            "available": False, "online_inflation_pp": None, "worst500_inflation_pp": None,
            "reason": "All90 original five-percent seed20-24 runs were launched with --summary-only. Their event hashes are null; neither extracted files nor the checksum-verified complete archive contain five-percent events. Queried-sample correctness and positions cannot be recovered from aggregate summaries/final state.",
            "summary_runs": 90, "seeds": [20, 21, 22, 23, 24], "strata": 18,
            "summary_samples": int(budget_frame.samples.sum()), "summary_queries": int(budget_frame.queries.sum()),
            "actual_budget_percent_range": [float(budget_frame.actual_budget_percent.min()), float(budget_frame.actual_budget_percent.max())],
            "archive_sha256_verified": True, "archive_members": len(members), "archive_budget_events": event_members,
            "archive_sha256": sha256(archive_path), "archive_declared_sha256": declared_sha,
            "archive_budget_summary_members": sum(name.endswith(".summary.json") for name in archive_budget),
            "archive_all_event_members": sum(".events." in name for name in members),
            "extracted_budget_summary_files": len(budget_files),
            "local_budget_events": local_budget_events, "archived_audit": posthoc_audit,
            "no_extrapolation": "The1% queried-error rate cannot be assumed constant at5%; budget changes selection, state trajectories, and errors. A new event-enabled replay would be a new experiment, not recovery of these historical trajectories.",
        },
        "inputs": inputs,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "five_percent_available": False,
                      "one_percent_identity": original_example, "report": str(args.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
