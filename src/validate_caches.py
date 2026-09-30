from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> dict[str, object]:
    loaded = np.load(path, allow_pickle=False)
    required = {"features", "labels", "domains", "paths", "text_features", "classes", "domain_names"}
    missing = required - set(loaded.files)
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    n = len(loaded["labels"])
    if any(len(loaded[name]) != n for name in ("features", "domains", "paths")):
        raise ValueError(f"{path}: sample-level arrays have inconsistent lengths")
    if len(np.unique(loaded["labels"])) != len(loaded["classes"]):
        raise ValueError(f"{path}: class cardinality mismatch")
    if len(np.unique(loaded["domains"])) != len(loaded["domain_names"]):
        raise ValueError(f"{path}: domain cardinality mismatch")
    norms = np.linalg.norm(loaded["features"], axis=1)
    predictions = (loaded["features"] @ loaded["text_features"].T).argmax(axis=1)
    by_domain = {}
    for domain_id, name in enumerate(loaded["domain_names"]):
        mask = loaded["domains"] == domain_id
        by_domain[str(name)] = float(np.mean(predictions[mask] == loaded["labels"][mask]))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "samples": n,
        "classes": len(loaded["classes"]),
        "domains": len(loaded["domain_names"]),
        "feature_dim": int(loaded["features"].shape[1]),
        "finite": bool(np.isfinite(loaded["features"]).all()),
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "zero_shot_accuracy": float(np.mean(predictions == loaded["labels"])),
        "zero_shot_by_domain": by_domain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "PASS", "caches": [validate(path) for path in args.caches]}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
