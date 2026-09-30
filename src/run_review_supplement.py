"""Run reproducible review-revision anchors on the six cached feature sets.

This runner deliberately keeps the anchor policies separate from the confirmatory
selector table: zero-shot is the no-label reference and ask-only spends the exact
budget but performs no pseudo-label updates.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from a2a.core import RunConfig
from run_streams import run_one


def one(job):
    cache_path, out_dir, stream, seed, strategy, noise, noise_mode = job
    cache_path = Path(cache_path)
    out_dir = Path(out_dir)
    loaded = np.load(cache_path, allow_pickle=False)
    cache = {name: loaded[name] for name in loaded.files}
    cfg = RunConfig(num_classes=len(cache["classes"]))
    return run_one(
        cache, out_dir, stream, seed, 1.0, strategy, cfg,
        write_events=False,
        oracle_noise=noise, oracle_noise_mode=noise_mode,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--strategies", nargs="+", default=["zero_shot", "ask_only"])
    ap.add_argument("--oracle-noise", type=float, default=0.0)
    ap.add_argument("--oracle-noise-mode", choices=["symmetric", "class_confusable"], default="symmetric")
    args = ap.parse_args()
    caches = sorted(args.cache_root.glob("*.npz"))
    if not caches:
        raise SystemExit(f"no caches under {args.cache_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(c), str(args.output_dir / c.stem), stream, seed, strategy, args.oracle_noise, args.oracle_noise_mode)
        for c in caches
        for stream in ("iid_random", "abrupt_domain", "class_correlated")
        for seed in range(10, 15)
        for strategy in args.strategies
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            print(f"{i}/{len(jobs)} {row['strategy']} {row['run_id']} {row['online_accuracy']:.6f}", flush=True)
    rows.sort(key=lambda r: (r["strategy"], r["run_id"]))
    (args.output_dir / "all_review_anchor_runs.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    import csv
    with (args.output_dir / "all_review_anchor_runs.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
