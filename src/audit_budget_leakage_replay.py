"""Independent read-only verification of downloaded paired budget replays.

Writes only independent_audit.json. Does not regenerate or modify experiment
events, summaries, receipts, source, or reported results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_streams import stream_order

KEY = ["dataset", "backbone", "stream", "seed"]
STRATA = KEY[:-1]
METRICS = ["causal_online_percent", "leaky_online_percent", "online_inflation_pp",
           "causal_worst500_percent", "leaky_worst500_percent", "worst500_inflation_pp", "queried_error_percent"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def match(actual, expected, description, tolerance=1e-10):
    require(np.allclose(actual, expected, rtol=0, atol=tolerance), description)


def check_ci(frame, report):
    grouped = frame.groupby(STRATA, sort=True)
    require(len(grouped) == 18 and set(grouped.size()) == {5}, "Incomplete18x5 stratum grid")
    values = np.array([group.sort_values("seed")[METRICS].to_numpy() for _, group in grouped])
    # Draw/stratum/seed ordering preserves the documented fixed-seed random
    # protocol. One shared index tensor applies to all seven metrics.
    selected = np.random.default_rng(20260814).integers(5, size=(20000, 18, 5))
    sampled = np.take_along_axis(values[None, :, :, :], selected[:, :, :, None], axis=2).mean(axis=(1, 2))
    fixed = np.quantile(sampled, [.025, .975], axis=0)
    means = values.mean(axis=1)
    indices = np.random.default_rng(20260814).integers(18, size=(100000, 18))
    sensitivity = np.quantile(means[indices].mean(axis=1), [.025, .975], axis=0)
    for index, metric in enumerate(METRICS):
        match(values[:, :, index].mean(), report[metric]["mean"], f"Mean mismatch {metric}")
        match(fixed[:, index], report[metric]["fixed_strata_ci95"], f"Fixed CI mismatch {metric}")
        match(sensitivity[:, index], report[metric]["stratum_resampled_ci95"], f"Cluster CI mismatch {metric}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "analysis/review_revision_round3/budget_leakage_replay")
    parser.add_argument("--skip-events", action="store_true", help="Mark event verification pending, never claim a full pass")
    parser.add_argument("--archive", type=Path, help="Optional original transfer archive; read and compare without extraction")
    parser.add_argument("--archive-sha256", help="SHA-256 obtained independently from the generating server")
    args = parser.parse_args()
    require(bool(args.archive) == bool(args.archive_sha256), "Supply both --archive and --archive-sha256")
    folder = args.root.resolve()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
    checks = []
    require(hashlib.sha256(json.dumps(metadata["contract"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == report["contract_sha256"] == metadata["contract_sha256"], "Contract hash mismatch")
    for name, expected in report["output_hashes"].items():
        require(sha(folder / name) == expected, f"Downloaded report-file hash mismatch: {name}")
    require(sha(ROOT / "src/run_budget_leakage_review.py") == metadata["runner_sha256"], "Canonical runner hash mismatch")
    for name, expected in metadata["contract"]["source_sha256"].items():
        require(sha(ROOT / "src" / name) == expected, f"Frozen source mismatch: {name}")
    runtime = metadata["contract"]["runtime"]
    require(hashlib.sha256(runtime["numpy_blas_build"].encode()).hexdigest() == runtime["numpy_blas_build_sha256"], "BLAS build hash mismatch")
    checks.append("Downloaded result files, contract, canonical frozen source, and recorded BLAS-build hashes verified")
    frame = pd.read_csv(folder / "per_run.csv")
    expected = {(d, b, s, seed, budget) for d in ["PACS", "OfficeHome", "TerraIncognita"]
                for b in ["openai_vitb16", "openclip_vitl14"] for s in ["iid_random", "abrupt_domain", "class_correlated"]
                for seed in range(20, 25) for budget in [1.0, 5.0]}
    require(len(frame) == 180 and set(frame[KEY + ["budget_percent"]].itertuples(index=False, name=None)) == expected,
            "Incomplete/duplicate run grid")
    require(set(frame.contract_sha256) == {metadata["contract_sha256"]} and frame.audit_pass.all(), "Per-run contract/audit mismatch")
    require(np.array_equal(frame.queries, np.rint(frame.samples * frame.budget_percent / 100).astype(int)), "Integer budgets differ")
    match(frame.online_inflation_pp, 100 * frame.queried_errors / frame.samples, "K/T identity failure")
    match(frame.leaky_online_percent - frame.causal_online_percent, frame.online_inflation_pp, "Score difference failure")
    match(frame.queried_error_percent, 100 * frame.queried_errors / frame.queries, "Queried-error fraction failure")
    grouped = {}
    for budget, part in frame.groupby("budget_percent"):
        grouped[budget] = part.set_index(KEY).sort_index()
        block = report["by_budget_percent"][str(int(budget))]
        check_ci(part, block["macro"])
        total_n, total_q, total_k = int(part.samples.sum()), int(part.queries.sum()), int(part.queried_errors.sum())
        require([total_n, total_q, total_k] == [block["samples"], block["queries"], block["queried_errors"]], "Count totals differ")
        micro = {"causal_online_percent": 100 * part.causal_correct.sum() / total_n,
                 "leaky_online_percent": 100 * part.leaky_correct.sum() / total_n,
                 "online_inflation_pp": 100 * total_k / total_n,
                 "actual_budget_percent": 100 * total_q / total_n,
                 "queried_error_percent": 100 * total_k / total_q}
        for metric, value in micro.items():
            match(value, block["micro"][metric], "Micro aggregate mismatch " + metric)
        for dataset, group in part.groupby("dataset"):
            for metric in METRICS:
                match(group[metric].mean(), block["by_dataset"][dataset][metric], "Dataset aggregate mismatch")
    require(grouped[1].index.equals(grouped[5].index), "Budget pairing mismatch")
    for field in ["samples", "order_sha256", "zero_prediction_sha256"]:
        require(np.array_equal(grouped[1][field], grouped[5][field]), f"Paired identity mismatch {field}")
    delta = grouped[5][METRICS] - grouped[1][METRICS]
    check_ci(delta.reset_index(), report["paired_5_minus_1"])
    stored_delta = pd.read_csv(folder / "paired_5_minus_1.csv").set_index(KEY).sort_index()
    match(delta.to_numpy(), stored_delta[METRICS].to_numpy(), "Paired-difference CSV mismatch")
    require(int(frame.samples.sum()) == report["event_count"] == 3022020 and int(frame.queries.sum()) == report["query_count"] == 90660,
            "Full experiment totals differ")
    checks.append("All180 paired rows, counts, macro/micro means, dataset means,21 fixed-stratum CIs and21 heterogeneity CIs independently reproduced")
    archive_audit = {"status": "NOT_REQUESTED"}
    if args.archive:
        archive_path = args.archive.resolve()
        digest = sha(archive_path)
        require(digest == args.archive_sha256.lower(), "Transfer archive differs from generating-server SHA-256")
        inventory = []
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                require(not name.is_absolute() and ".." not in name.parts, "Unsafe archive member")
                require(member.isdir() or member.isfile(), "Unsupported archive member type")
                if not member.isfile():
                    continue
                local_path = folder.joinpath(*name.parts)
                digest_member = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1048576), b""):
                        digest_member.update(block)
                require(local_path.is_file() and sha(local_path) == digest_member.hexdigest(),
                        f"Extracted file differs from archive: {name}")
                inventory.append(str(name))
        expected_inventory = {"run_metadata.json", "progress.json", "completed_runs.csv", "paired_5_minus_1.csv",
                              "per_run.csv", "report.json"}
        for row in frame.itertuples(index=False):
            event_name = "runs/" + row.cache + "/" + Path(row.event_path).name
            expected_inventory.update([event_name, event_name.replace(".events.csv.gz", ".summary.json"),
                                       event_name.replace(".events.csv.gz", ".audit.json")])
        require(len(inventory) == len(set(inventory)) and set(inventory) == expected_inventory,
                "Archive inventory differs from complete expected experiment")
        archive_audit = {"status": "PASS", "path": str(archive_path), "sha256": digest,
                         "sha256_reference": "Generating-server sha256sum supplied independently by coordinator",
                         "bytes": archive_path.stat().st_size, "regular_files": len(inventory),
                         "event_files": 180, "summary_files": 180, "audit_receipts": 180, "aggregate_files": 6,
                         "all_member_bytes_equal_local_files": True}
        checks.append("Transfer archive matches generating-server SHA-256; all546 file contents and exact inventory match local experiment")
    event_audit = {"status": "PENDING", "runs_verified": 0, "events_verified": 0}
    if not args.skip_events:
        cache_root = ROOT / "remote_artifacts/20260814_multibench_v6/cache"
        for cache_name, part in frame.groupby("cache", sort=True):
            cache_path = cache_root / (cache_name + ".npz")
            require(sha(cache_path) == metadata["contract"]["cache_sha256"][cache_path.name], "Local cache hash mismatch")
            with np.load(cache_path, allow_pickle=False) as archive:
                labels, domains, paths = archive["labels"], archive["domains"], archive["paths"]
            for row in part.itertuples(index=False):
                event = folder / "runs" / cache_name / Path(row.event_path).name
                summary_path = event.with_name(event.name.replace(".events.csv.gz", ".summary.json"))
                require(sha(event) == row.event_sha256, "Downloaded event hash mismatch")
                require(sha(summary_path) == row.summary_sha256, "Downloaded summary hash mismatch")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                receipt_path = event.with_name(event.name.replace(".events.csv.gz", ".audit.json"))
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                for key, value in receipt.items():
                    expected_value = getattr(row, key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        match(value, expected_value, "Audit receipt/CSV numerical mismatch: " + key)
                    else:
                        require(value == expected_value, "Audit receipt/CSV provenance mismatch: " + key)
                require(summary["event_sha256"] == row.event_sha256, "Summary/event SHA mismatch")
                data = pd.read_csv(event, keep_default_na=False)
                order = stream_order(labels, domains, row.stream, row.seed)
                require(len(data) == row.samples and np.array_equal(data.position, np.arange(row.samples)), "Event order invalid")
                require(np.array_equal(data.original_index, order), "Seed/stream index mismatch")
                require(np.array_equal(data.label, labels[order]) and np.array_equal(data.domain, domains[order])
                        and np.array_equal(data.path.astype(str), paths[order].astype(str)), "Cache identity mismatch")
                require(np.array_equal(data.correct, (data.prediction == data.label).astype(int)), "Event correctness mismatch")
                require(np.array_equal(data.zero_shot_correct, (data.zero_shot_prediction == data.label).astype(int)), "Raw-zero correctness mismatch")
                require(hashlib.sha256(order.astype("<i8").tobytes()).hexdigest() == row.order_sha256, "Order trace hash mismatch")
                require(hashlib.sha256(data.zero_shot_prediction.to_numpy(dtype="<i8").tobytes()).hexdigest() == row.zero_prediction_sha256,
                        "Raw-zero trace hash mismatch")
                signals = ["confidence", "entropy", "margin", "disagreement", "coverage", "query_value", "selection_score"]
                require(np.isfinite(data[signals].to_numpy(dtype=float)).all(), "Nonfinite event signals")
                query = data.queried.astype(bool).to_numpy()
                correct = data.correct.to_numpy(dtype=np.int64)
                require(not (query & data.adapted.astype(bool)).any(), "Query/pseudo overlap")
                require(np.array_equal(data.action.eq("ask"), query), "Query action mismatch")
                require(np.array_equal(data.loc[query, "oracle_label"].astype(int), data.loc[query, "label"]), "Oracle correctness mismatch")
                require((data.loc[~query, "oracle_label"] == "").all() and (data.loc[~query, "query_revealed_at"] == "").all(),
                        "Premature/nonquery oracle fields")
                for segment in np.array_split(np.arange(row.samples), row.queries):
                    queried = segment[query[segment]]
                    require(len(queried) == 1 and int(data.loc[int(queried[0]), "query_revealed_at"]) == int(segment[-1]),
                            "Not one delayed query per segment")
                k = int(np.sum(query & (correct == 0)))
                require(int(query.sum()) == row.queries and k == row.queried_errors and int(correct.sum()) == row.causal_correct,
                        "Event/CSV count mismatch")
                leaky = np.maximum(correct, query.astype(np.int64))
                require(int(leaky.sum()) == row.leaky_correct, "Counterfactual event count mismatch")
                # A separate rolling-mean implementation checks the runner's
                # integer-cumulative-sum window calculation.
                a = float(pd.Series(correct).rolling(min(500, row.samples)).mean().min())
                b = float(pd.Series(leaky).rolling(min(500, row.samples)).mean().min())
                match([100 * a, 100 * b], [row.causal_worst500_percent, row.leaky_worst500_percent], "Worst-window mismatch")
                match([correct.mean(), a], [summary["online_accuracy"], summary["worst_window_accuracy"]], "Summary metric mismatch")
                event_audit["runs_verified"] += 1
                event_audit["events_verified"] += row.samples
            print(json.dumps({"verified_cache": cache_name, **event_audit}), flush=True)
        require(event_audit["runs_verified"] == 180 and event_audit["events_verified"] == 3022020, "Incomplete event audit")
        event_audit["status"] = "PASS"
        checks.append("All180 downloaded event/summary hashes,3,022,020 sample identities, delayed query ordering, and independently recomputed worst500 values verified")
    result = {"status": "PARTIAL_EVENT_DOWNLOAD_PENDING" if args.skip_events else "PASS",
              "checks": checks, "event_audit": event_audit, "archive_audit": archive_audit,
              "source_files": {name: sha(folder / name) for name in ["report.json", "per_run.csv", "paired_5_minus_1.csv", "run_metadata.json"]},
              "auditor_sha256": sha(Path(__file__)),
              "limits": "No new model replay; verifies immutable artifacts and statistics. Does not recover historical trajectories or certify per-step prototype snapshots."}
    (folder / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
