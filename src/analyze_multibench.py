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
    "random": "Random",
    "periodic": "Periodic",
    "entropy": "Entropy",
    "margin": "Margin",
    "disagreement": "Disagreement",
    "dynamic_entropy_online": "Dynamic entropy (selection-only)",
    "ask_or_adapt": "Ask-or-Adapt",
}


def load_summaries(root: Path, dataset: str, backbone: str, budget: float = 1.0) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*.summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if float(row["budget_percent"]) != budget:
            continue
        row["dataset"] = dataset
        row["backbone"] = backbone
        row["source_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(values.mean())
    n = len(values)
    extreme = 0
    total = 1 << n
    for mask in range(total):
        signs = np.fromiter((1.0 if mask & (1 << i) else -1.0 for i in range(n)), float, n)
        extreme += abs(np.mean(values * signs)) >= observed - 1e-15
    return float(extreme / total)


def stratified_seed_bootstrap(delta: pd.Series, seed: int = 20260814, draws: int = 20000) -> tuple[float, float]:
    frame = delta.rename("delta").reset_index()
    groups = [part["delta"].to_numpy() for _, part in frame.groupby(["dataset", "backbone", "stream"])]
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        means = [rng.choice(group, size=len(group), replace=True).mean() for group in groups]
        values[draw] = np.mean(means)
    return tuple(np.quantile(values, [0.025, 0.975]))


def component_auc(values: np.ndarray, errors: np.ndarray) -> float:
    pos = errors == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(values, method="average")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.results_root, args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    frames = [
        load_summaries(root / "confirmation_v2", "PACS", "OpenAI ViT-B/16"),
        load_summaries(root / "multibench_v3/openai_vitb16/OfficeHome", "OfficeHome", "OpenAI ViT-B/16"),
        load_summaries(root / "multibench_v3/openai_vitb16/TerraIncognita", "TerraIncognita", "OpenAI ViT-B/16"),
        load_summaries(root / "multibench_v3/openclip_vitl14/PACS", "PACS", "OpenCLIP ViT-L/14"),
        load_summaries(root / "multibench_v3/openclip_vitl14/OfficeHome", "OfficeHome", "OpenCLIP ViT-L/14"),
        load_summaries(root / "multibench_v3/openclip_vitl14/TerraIncognita", "TerraIncognita", "OpenCLIP ViT-L/14"),
    ]
    main_df = pd.concat(frames, ignore_index=True)
    ext_frames = []
    for backbone_dir, backbone in [("openai_vitb16", "OpenAI ViT-B/16"), ("openclip_vitl14", "OpenCLIP ViT-L/14")]:
        for dataset in ("PACS", "OfficeHome", "TerraIncognita"):
            ext_frames.append(load_summaries(root / f"external_selection_v4/{backbone_dir}/{dataset}", dataset, backbone))
    full = pd.concat([main_df, *ext_frames], ignore_index=True)
    full.to_csv(out / "all_primary_runs.csv", index=False)

    key = ["dataset", "backbone", "stream", "seed"]
    accuracy = full.pivot(index=key, columns="strategy", values="online_accuracy")
    if accuracy.isna().any().any() or len(accuracy) != 90:
        raise RuntimeError(f"Incomplete primary paired matrix: shape={accuracy.shape}, missing={int(accuracy.isna().sum().sum())}")

    aggregate = full.groupby(["dataset", "backbone", "strategy"])[
        ["online_accuracy", "zero_shot_accuracy", "gain_over_zero_shot", "worst_window_accuracy", "negative_adaptation_rate", "expected_calibration_error"]
    ].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(item).strip("_") for item in aggregate.columns]
    aggregate.to_csv(out / "table_dataset_backbone.csv", index=False)

    comparisons = []
    raw_p = {}
    for baseline in [name for name in DISPLAY if name != "ask_or_adapt"]:
        delta = accuracy["ask_or_adapt"] - accuracy[baseline]
        strata = delta.groupby(level=["dataset", "backbone", "stream"]).mean()
        lower, upper = stratified_seed_bootstrap(delta)
        p_value = exact_sign_flip_p(strata.to_numpy())
        raw_p[baseline] = p_value
        comparisons.append({
            "baseline": baseline,
            "paired_runs": len(delta),
            "strata": len(strata),
            "mean_delta_pp": 100 * delta.mean(),
            "ci95_low_pp": 100 * lower,
            "ci95_high_pp": 100 * upper,
            "stratum_win_rate": float((strata > 0).mean()),
            "paired_effect_dz": float(strata.mean() / strata.std(ddof=1)),
            "p_exact_sign_flip": p_value,
        })
    adjusted = holm_adjust(raw_p)
    for row in comparisons:
        row["p_holm"] = adjusted[row["baseline"]]
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(out / "paired_statistics.csv", index=False)

    # Heatmap: policy deltas for every dataset/backbone row.
    means = full.groupby(["dataset", "backbone", "strategy"]).online_accuracy.mean().unstack("strategy")
    heat = pd.DataFrame({DISPLAY[b]: 100 * (means["ask_or_adapt"] - means[b]) for b in DISPLAY if b != "ask_or_adapt"})
    heat.index = [f"{dataset}\n{backbone.replace('OpenAI ', '').replace('OpenCLIP ', '')}" for dataset, backbone in heat.index]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    sns.heatmap(heat, annot=True, fmt="+.2f", center=0, cmap="RdBu_r", linewidths=.5,
                cbar_kws={"label": "Ask-or-Adapt accuracy advantage (pp)"}, ax=ax)
    ax.set_xlabel("Exact-budget query policy")
    ax.set_ylabel("")
    ax.set_title("Paired policy advantage at the 1% label budget")
    fig.tight_layout()
    fig.savefig(out / "policy_delta_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "policy_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Scatter: headroom vs gain, one point per dataset/backbone/stream/seed condition.
    a2a = full[full.strategy == "ask_or_adapt"].copy()
    a2a["headroom"] = 1 - a2a.zero_shot_accuracy
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    palette = {"PACS": "#2563eb", "OfficeHome": "#f59e0b", "TerraIncognita": "#dc2626"}
    markers = {"OpenAI ViT-B/16": "o", "OpenCLIP ViT-L/14": "s"}
    for (dataset, backbone), part in a2a.groupby(["dataset", "backbone"]):
        ax.scatter(100 * part.headroom, 100 * part.gain_over_zero_shot, alpha=.72, s=48,
                   c=palette[dataset], marker=markers[backbone], edgecolors="white", linewidths=.4,
                   label=f"{dataset}, {backbone.replace('OpenAI ', '').replace('OpenCLIP ', '')}")
    x = 100 * a2a.headroom.to_numpy()
    y = 100 * a2a.gain_over_zero_shot.to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, slope * grid + intercept, color="#111827", linestyle="--", linewidth=1.2,
            label=f"Linear fit, r={np.corrcoef(x, y)[0,1]:.2f}")
    ax.axhline(0, color="#6b7280", linewidth=.8)
    ax.set_xlabel("Zero-shot error / available headroom (pp)")
    ax.set_ylabel("Ask-or-Adapt gain over zero-shot (pp)")
    ax.set_title("Adaptation gain grows with task headroom")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "headroom_gain_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "headroom_gain_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    # Component ablations: one frozen backbone, paired across streams/seeds.
    ablation_frames = [
        load_summaries(root / f"multibench_v3/ablations/{dataset}", dataset, "OpenAI ViT-B/16")
        for dataset in ("PACS", "OfficeHome", "TerraIncognita")
    ]
    ablation = pd.concat(ablation_frames, ignore_index=True)
    ablation.to_csv(out / "all_ablation_runs.csv", index=False)
    abl_means = ablation.groupby(["dataset", "strategy"]).online_accuracy.mean().unstack("strategy")
    variants = ["a2a_no_gate", "a2a_no_entropy", "a2a_no_disagreement", "a2a_no_coverage"]
    labels = ["No persistence gate", "No entropy", "No disagreement", "No coverage"]
    abl_heat = pd.DataFrame({label: 100 * (abl_means["ask_or_adapt"] - abl_means[var]) for label, var in zip(labels, variants)})
    abl_heat.to_csv(out / "component_ablation_deltas.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    sns.heatmap(abl_heat, annot=True, fmt="+.2f", center=0, cmap="RdBu_r", linewidths=.5,
                cbar_kws={"label": "Full method advantage (pp)"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Component necessity under frozen hyperparameters")
    fig.tight_layout()
    fig.savefig(out / "component_ablation_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "component_ablation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Error-detection validity of query signals on Ask-or-Adapt event logs.
    signal_rows = []
    roots = {
        "PACS": root / "confirmation_v2",
        "OfficeHome": root / "multibench_v3/openai_vitb16/OfficeHome",
        "TerraIncognita": root / "multibench_v3/openai_vitb16/TerraIncognita",
    }
    for dataset, event_root in roots.items():
        values = {name: [] for name in ("entropy", "disagreement", "coverage", "query_value")}
        errors = []
        for path in event_root.rglob("*budget1__ask_or_adapt.events.csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    errors.append(1 - int(row["correct"]))
                    for name in values:
                        values[name].append(float(row[name]))
        error_array = np.asarray(errors, dtype=np.int8)
        for name, raw in values.items():
            array = np.asarray(raw, dtype=np.float64)
            signal_rows.append({
                "dataset": dataset, "signal": name, "events": len(array),
                "error_rate": error_array.mean(),
                "error_detection_auc": component_auc(array, error_array),
            })
    signal_df = pd.DataFrame(signal_rows)
    signal_df.to_csv(out / "signal_error_detection.csv", index=False)
    signal_heat = signal_df.pivot(index="dataset", columns="signal", values="error_detection_auc")
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    sns.heatmap(signal_heat, annot=True, fmt=".3f", vmin=.5, vmax=1, cmap="YlOrRd", linewidths=.5,
                cbar_kws={"label": "AUC for detecting an incorrect prediction"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Query signals target prediction errors")
    fig.tight_layout()
    fig.savefig(out / "signal_error_auc_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "signal_error_auc_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Deterministic duplicate check for full method in the ablation folders.
    primary_b16 = full[(full.backbone == "OpenAI ViT-B/16") & (full.strategy == "ask_or_adapt")]
    duplicate = ablation[ablation.strategy == "ask_or_adapt"]
    duplicate_check = primary_b16.merge(duplicate, on=["dataset", "stream", "seed", "budget_percent"], suffixes=("_main", "_ablation"))
    max_duplicate_delta = float(np.max(np.abs(duplicate_check.online_accuracy_main - duplicate_check.online_accuracy_ablation)))
    if max_duplicate_delta != 0.0:
        raise RuntimeError(f"Full-method duplicate mismatch: {max_duplicate_delta}")

    headline = {
        "status": "PASS",
        "primary_runs": len(full),
        "paired_conditions_per_strategy": len(accuracy),
        "datasets": sorted(full.dataset.unique()),
        "backbones": sorted(full.backbone.unique()),
        "a2a_mean_accuracy": float(full[full.strategy == "ask_or_adapt"].online_accuracy.mean()),
        "a2a_mean_gain_over_zero_shot": float(full[full.strategy == "ask_or_adapt"].gain_over_zero_shot.mean()),
        "headroom_gain_correlation": float(np.corrcoef(x, y)[0, 1]),
        "full_method_duplicate_max_accuracy_delta": max_duplicate_delta,
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    manifest = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            manifest.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2))
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
