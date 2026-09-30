from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata


DISPLAY = {
    "random": "Random", "periodic": "Periodic", "entropy": "Entropy",
    "margin": "Margin", "disagreement": "Disagreement",
    "dynamic_entropy_online": "Dynamic entropy (selection-only)",
    "ask_or_adapt_v2": "Ask-or-Adapt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(root: Path) -> pd.DataFrame:
    rows = []
    mapping = {
        "openai_vitb16": "OpenAI ViT-B/16",
        "openclip_vitl14": "OpenCLIP ViT-L/14",
    }
    for path in root.rglob("*.summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root)
        row["backbone"] = mapping[rel.parts[0]]
        row["dataset"] = rel.parts[1]
        row["source_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted, running, m = {}, 0.0, len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * values[name]))
        adjusted[name] = running
    return adjusted


def sign_flip(values: np.ndarray) -> float:
    observed = abs(values.mean())
    extreme, total = 0, 1 << len(values)
    for mask in range(total):
        signs = np.fromiter((1 if mask & (1 << i) else -1 for i in range(len(values))), float)
        extreme += abs(np.mean(values * signs)) >= observed - 1e-15
    return extreme / total


def bootstrap(delta: pd.Series, draws: int = 20000) -> tuple[float, float]:
    frame = delta.rename("delta").reset_index()
    groups = [part.delta.to_numpy() for _, part in frame.groupby(["dataset", "backbone", "stream"])]
    rng, sampled = np.random.default_rng(20260814), np.empty(draws)
    for index in range(draws):
        sampled[index] = np.mean([rng.choice(group, len(group), replace=True).mean() for group in groups])
    return tuple(np.quantile(sampled, [.025, .975]))


def auc(values: np.ndarray, errors: np.ndarray) -> float:
    positive = errors == 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    ranks = rankdata(values, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.root)
    if len(df) != 630:
        raise RuntimeError(f"Expected 630 summaries, found {len(df)}")
    df.to_csv(out / "all_confirmation_runs.csv", index=False)

    key = ["dataset", "backbone", "stream", "seed"]
    matrix = df.pivot(index=key, columns="strategy", values="online_accuracy")
    if matrix.shape != (90, 7) or matrix.isna().any().any():
        raise RuntimeError(f"Incomplete paired matrix {matrix.shape}")

    aggregate = df.groupby(["dataset", "backbone", "strategy"])[
        ["online_accuracy", "zero_shot_accuracy", "gain_over_zero_shot", "worst_window_accuracy",
         "negative_adaptation_rate", "expected_calibration_error", "pseudo_memory_error_rate"]
    ].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(value).strip("_") for value in aggregate.columns]
    aggregate.to_csv(out / "table_dataset_backbone.csv", index=False)

    rows = []
    for metric in ("online_accuracy", "worst_window_accuracy", "negative_adaptation_rate"):
        metric_matrix = df.pivot(index=key, columns="strategy", values=metric)
        p_values, metric_rows = {}, []
        for baseline in DISPLAY:
            if baseline == "ask_or_adapt_v2":
                continue
            delta = metric_matrix.ask_or_adapt_v2 - metric_matrix[baseline]
            if metric == "negative_adaptation_rate":
                delta = -delta  # Positive means lower harmful adaptation for Ask-or-Adapt.
            strata = delta.groupby(level=["dataset", "backbone", "stream"]).mean()
            low, high = bootstrap(delta)
            p = sign_flip(strata.to_numpy())
            p_values[baseline] = p
            metric_rows.append({
                "metric": metric, "baseline": baseline, "paired_runs": len(delta), "strata": len(strata),
                "mean_delta_pp": 100 * delta.mean(), "ci95_low_pp": 100 * low,
                "ci95_high_pp": 100 * high, "stratum_win_rate": (strata > 0).mean(),
                "paired_effect_dz": strata.mean() / strata.std(ddof=1), "p_sign_flip": p,
            })
        adjusted = holm(p_values)
        for row in metric_rows:
            row["p_holm"] = adjusted[row["baseline"]]
        rows.extend(metric_rows)
    statistics = pd.DataFrame(rows)
    statistics.to_csv(out / "paired_statistics.csv", index=False)

    means = df.groupby(["dataset", "backbone", "strategy"]).online_accuracy.mean().unstack("strategy")
    heat = pd.DataFrame({DISPLAY[name]: 100 * (means.ask_or_adapt_v2 - means[name]) for name in DISPLAY if name != "ask_or_adapt_v2"})
    heat.index = [f"{dataset}\n{backbone.split()[-1]}" for dataset, backbone in heat.index]
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    sns.heatmap(heat, annot=True, fmt="+.2f", center=0, cmap="RdBu_r", linewidths=.5,
                cbar_kws={"label": "Ask-or-Adapt advantage (pp)"}, ax=ax)
    ax.set_xlabel("Exact-budget comparator"); ax.set_ylabel("")
    ax.set_title("Cross-benchmark accuracy differences at a 1% label budget")
    fig.tight_layout(); fig.savefig(out / "policy_delta_heatmap.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "policy_delta_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

    method = df[df.strategy == "ask_or_adapt_v2"].copy()
    method["headroom"] = 1 - method.zero_shot_accuracy
    colors = {"PACS": "#2563eb", "OfficeHome": "#f59e0b", "TerraIncognita": "#dc2626"}
    markers = {"OpenAI ViT-B/16": "o", "OpenCLIP ViT-L/14": "s"}
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for (dataset, backbone), part in method.groupby(["dataset", "backbone"]):
        ax.scatter(100 * part.headroom, 100 * part.gain_over_zero_shot, s=48, alpha=.72,
                   color=colors[dataset], marker=markers[backbone], edgecolors="white", linewidths=.4,
                   label=f"{dataset}, {backbone.split()[-1]}")
    x, y = 100 * method.headroom.to_numpy(), 100 * method.gain_over_zero_shot.to_numpy()
    slope, intercept = np.polyfit(x, y, 1); grid = np.linspace(x.min(), x.max(), 100)
    correlation = np.corrcoef(x, y)[0, 1]
    ax.plot(grid, slope * grid + intercept, "--", color="#111827", linewidth=1.2, label=f"Linear fit, r={correlation:.2f}")
    ax.axhline(0, color="#6b7280", linewidth=.8); ax.grid(alpha=.2)
    ax.set_xlabel("Zero-shot error / available headroom (pp)"); ax.set_ylabel("Gain over zero-shot (pp)")
    ax.set_title("Label utility increases with deployment headroom"); ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(out / "headroom_gain_scatter.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "headroom_gain_scatter.pdf", bbox_inches="tight"); plt.close(fig)

    risk_modes = []
    signals = []
    for dataset in ("PACS", "OfficeHome", "TerraIncognita"):
        values = {name: [] for name in ("entropy", "margin", "disagreement", "coverage", "query_value")}
        errors, modes = [], []
        for path in (args.root / "openai_vitb16" / dataset).glob("*ask_or_adapt_v2.events.csv.gz"):
            with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    errors.append(1 - int(row["correct"])); modes.append(row["selection_mode"])
                    for name in values: values[name].append(float(row[name]))
        error_array = np.asarray(errors)
        for name, value in values.items():
            signals.append({"dataset": dataset, "signal": name, "events": len(value), "error_detection_auc": auc(np.asarray(value), error_array)})
        selected = [mode for mode in modes if mode]
        for mode in sorted(set(selected)):
            risk_modes.append({"dataset": dataset, "selection_mode": mode, "queries": selected.count(mode), "share": selected.count(mode) / len(selected)})
    signal_df = pd.DataFrame(signals); signal_df.to_csv(out / "signal_auc.csv", index=False)
    mode_df = pd.DataFrame(risk_modes); mode_df.to_csv(out / "selection_mode_usage.csv", index=False)
    signal_heat = signal_df.pivot(index="dataset", columns="signal", values="error_detection_auc")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    sns.heatmap(signal_heat, annot=True, fmt=".3f", vmin=.45, vmax=1, cmap="YlOrRd", linewidths=.5,
                cbar_kws={"label": "AUC for detecting prediction error"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Observable query signals target current errors")
    fig.tight_layout(); fig.savefig(out / "signal_auc_heatmap.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "signal_auc_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

    headline = {
        "status": "PASS", "runs": len(df), "paired_conditions_per_strategy": len(matrix),
        "a2a_mean_accuracy": method.online_accuracy.mean(),
        "a2a_gain_over_zero_shot_pp": 100 * method.gain_over_zero_shot.mean(),
        "a2a_worst_window_accuracy": method.worst_window_accuracy.mean(),
        "a2a_negative_adaptation_rate": method.negative_adaptation_rate.mean(),
        "headroom_gain_correlation": correlation,
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    manifest = [{"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file()]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2)); print(statistics.to_string(index=False)); print(mode_df.to_string(index=False))


if __name__ == "__main__":
    main()
