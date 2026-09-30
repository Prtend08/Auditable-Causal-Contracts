from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BACKBONES = {
    "openai_vitb16": "ViT-B/16",
    "openclip_vitl14": "ViT-L/14",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*.summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root)
        row["backbone"] = BACKBONES[rel.parts[0]]
        row["dataset"] = rel.parts[1]
        row["analysis"] = rel.parts[2]
        row["source_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, draws: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(20260814)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def summarize_budget(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in df[df.analysis == "budget"].groupby(["dataset", "backbone", "budget_percent"]):
        low, high = bootstrap_ci(part.online_accuracy.to_numpy())
        rows.append({
            "dataset": keys[0], "backbone": keys[1], "budget_percent": keys[2],
            "runs": len(part), "online_accuracy_mean": part.online_accuracy.mean(),
            "online_accuracy_ci_low": low, "online_accuracy_ci_high": high,
            "worst_window_mean": part.worst_window_accuracy.mean(),
            "gain_over_zero_shot_mean": part.gain_over_zero_shot.mean(),
            "label_efficiency_mean": part.label_efficiency_pp_per_100_queries.mean(),
        })
    return pd.DataFrame(rows)


def summarize_noise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    noisy = df[df.analysis.isin(["noise10", "noise20"])].copy()
    for keys, part in noisy.groupby(["dataset", "backbone", "oracle_noise"]):
        low, high = bootstrap_ci(part.online_accuracy.to_numpy())
        rows.append({
            "dataset": keys[0], "backbone": keys[1], "oracle_noise": keys[2],
            "runs": len(part), "online_accuracy_mean": part.online_accuracy.mean(),
            "online_accuracy_ci_low": low, "online_accuracy_ci_high": high,
            "worst_window_mean": part.worst_window_accuracy.mean(),
            "gain_over_zero_shot_mean": part.gain_over_zero_shot.mean(),
            "negative_adaptation_mean": part.negative_adaptation_rate.mean(),
            "realized_oracle_noise_mean": part.realized_oracle_noise.mean(),
            "queries": int(part.query_count.sum()),
            "noisy_queries": int(part.noisy_oracle_count.sum()),
        })
    return pd.DataFrame(rows)


def paired_noise_effects(df: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "backbone", "stream", "seed"]
    clean = df[(df.analysis == "budget") & (df.budget_percent == 1.0)].set_index(key)
    rows = []
    for noise_rate, analysis in [(0.1, "noise10"), (0.2, "noise20")]:
        noisy = df[df.analysis == analysis].set_index(key)
        for dataset in sorted(df.dataset.unique()):
            for backbone in sorted(df.backbone.unique()):
                mask = (noisy.index.get_level_values("dataset") == dataset) & (
                    noisy.index.get_level_values("backbone") == backbone)
                delta = noisy.loc[mask, "online_accuracy"] - clean.loc[mask, "online_accuracy"]
                tail = noisy.loc[mask, "worst_window_accuracy"] - clean.loc[mask, "worst_window_accuracy"]
                low, high = bootstrap_ci(delta.to_numpy())
                rows.append({
                    "dataset": dataset, "backbone": backbone, "oracle_noise": noise_rate,
                    "paired_runs": len(delta), "online_delta_pp": 100 * delta.mean(),
                    "online_delta_ci_low_pp": 100 * low, "online_delta_ci_high_pp": 100 * high,
                    "worst_window_delta_pp": 100 * tail.mean(),
                })
    return pd.DataFrame(rows)


def plot_budget(summary: pd.DataFrame, output: Path) -> None:
    palette = {"PACS": "#2563eb", "OfficeHome": "#f59e0b", "TerraIncognita": "#dc2626"}
    markers = {"ViT-B/16": "o", "ViT-L/14": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    for (dataset, backbone), part in summary.groupby(["dataset", "backbone"]):
        part = part.sort_values("budget_percent")
        x = part.budget_percent.to_numpy()
        for ax, value, ylabel in [
            (axes[0], "online_accuracy_mean", "Online accuracy (%)"),
            (axes[1], "worst_window_mean", "Worst-500 accuracy (%)"),
        ]:
            ax.plot(x, 100 * part[value], marker=markers[backbone], color=palette[dataset],
                    linestyle="-" if backbone == "ViT-B/16" else "--", linewidth=1.4,
                    markersize=4.8, label=f"{dataset}, {backbone}")
            ax.set_xscale("log"); ax.set_xlabel("Label budget (%)"); ax.set_ylabel(ylabel)
            ax.grid(alpha=.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Post-hoc exact-budget robustness across datasets and backbones")
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.savefig(output / "budget_robustness.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "budget_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_noise(noise: pd.DataFrame, clean: pd.DataFrame, output: Path) -> None:
    clean1 = clean[clean.budget_percent == 1.0][["dataset", "backbone", "online_accuracy_mean", "worst_window_mean"]]
    clean1 = clean1.assign(oracle_noise=0.0)
    frame = pd.concat([clean1, noise[["dataset", "backbone", "oracle_noise", "online_accuracy_mean", "worst_window_mean"]]])
    for metric, filename, title in [
        ("online_accuracy_mean", "oracle_noise_accuracy_heatmap", "Accuracy under simulated oracle-label noise"),
        ("worst_window_mean", "oracle_noise_tail_heatmap", "Worst-window accuracy under oracle-label noise"),
    ]:
        pivot = frame.pivot(index=["dataset", "backbone"], columns="oracle_noise", values=metric)
        pivot.columns = [f"{100 * value:.0f}% noise" for value in pivot.columns]
        pivot.index = [f"{dataset}\n{backbone}" for dataset, backbone in pivot.index]
        fig, ax = plt.subplots(figsize=(5.7, 4.3))
        sns.heatmap(100 * pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5,
                    cbar_kws={"label": "Accuracy (%)"}, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(title)
        fig.tight_layout(); fig.savefig(output / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / f"{filename}.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = load(args.root)
    if len(df) != 630:
        raise RuntimeError(f"Expected 630 summaries, found {len(df)}")
    if len(df[df.analysis == "budget"]) != 450 or len(df[df.analysis != "budget"]) != 180:
        raise RuntimeError("Incomplete budget/noise partition")
    df.to_csv(args.output / "all_posthoc_runs.csv", index=False)
    budget = summarize_budget(df); budget.to_csv(args.output / "budget_summary.csv", index=False)
    noise = summarize_noise(df); noise.to_csv(args.output / "oracle_noise_summary.csv", index=False)
    effects = paired_noise_effects(df); effects.to_csv(args.output / "oracle_noise_paired_effects.csv", index=False)
    plot_budget(budget, args.output); plot_noise(noise, budget, args.output)
    headline = {
        "status": "PASS", "runs": len(df), "budget_runs": 450, "noise_runs": 180,
        "events_expected_in_noise_logs": int(df[df.analysis != "budget"].samples.sum()),
        "queries_in_noise_runs": int(df[df.analysis != "budget"].query_count.sum()),
        "noisy_queries": int(df[df.analysis != "budget"].noisy_oracle_count.sum()),
    }
    (args.output / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    manifest = [{"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2)); print(budget.to_string(index=False)); print(noise.to_string(index=False)); print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
