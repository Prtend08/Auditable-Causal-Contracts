from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY = {
    "random": "Random",
    "periodic": "Periodic",
    "entropy": "Entropy",
    "margin": "Margin",
    "disagreement": "Disagreement",
    "ask_or_adapt": "Ask-or-Adapt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.all_runs)

    metrics = [
        "online_accuracy", "worst_window_accuracy", "gain_over_zero_shot",
        "negative_adaptation_rate", "expected_calibration_error",
        "label_efficiency_pp_per_100_queries", "pseudo_memory_error_rate",
    ]
    aggregate = df.groupby(["stream", "budget_percent", "strategy"])[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.reset_index().to_csv(out / "table_all_conditions.csv", index=False)

    stream_table = df.groupby(["stream", "strategy"])[metrics].agg(["mean", "std"])
    stream_table.columns = [f"{metric}_{stat}" for metric, stat in stream_table.columns]
    stream_table.reset_index().to_csv(out / "table_by_stream.csv", index=False)

    overall = df.groupby("strategy")[metrics].agg(["mean", "std"])
    overall.columns = [f"{metric}_{stat}" for metric, stat in overall.columns]
    overall.reset_index().to_csv(out / "table_overall.csv", index=False)

    key = ["stream", "budget_percent", "seed"]
    accuracy = df.pivot(index=key, columns="strategy", values="online_accuracy")
    paired_rows = []
    for baseline in [name for name in DISPLAY if name != "ask_or_adapt"]:
        delta = accuracy["ask_or_adapt"] - accuracy[baseline]
        seed_means = delta.groupby("seed").mean()
        paired_rows.append({
            "baseline": baseline,
            "conditions": int(delta.count()),
            "mean_delta_pp": float(100 * delta.mean()),
            "median_delta_pp": float(100 * delta.median()),
            "condition_win_rate": float((delta > 0).mean()),
            "seed_mean_delta_pp": float(100 * seed_means.mean()),
            "seed_std_delta_pp": float(100 * seed_means.std(ddof=1)),
            "positive_seed_count": int((seed_means > 0).sum()),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_differences.csv", index=False)

    best_baseline = accuracy.drop(columns="ask_or_adapt").max(axis=1)
    delta_best = accuracy["ask_or_adapt"] - best_baseline
    summary = {
        "runs": int(len(df)),
        "seeds": sorted(int(x) for x in df.seed.unique()),
        "conditions_per_strategy": int(len(accuracy)),
        "a2a_mean_accuracy": float(df[df.strategy == "ask_or_adapt"].online_accuracy.mean()),
        "a2a_mean_gain_over_zero_shot": float(df[df.strategy == "ask_or_adapt"].gain_over_zero_shot.mean()),
        "a2a_condition_wins_vs_oracle_best_baseline": int((delta_best > 0).sum()),
        "a2a_mean_delta_vs_oracle_best_baseline": float(delta_best.mean()),
        "note": "The per-condition best baseline is an oracle diagnostic, not a deployable method.",
    }
    (out / "headline_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = {
        "random": "#9ca3af", "periodic": "#6b7280", "entropy": "#60a5fa",
        "margin": "#34d399", "disagreement": "#f59e0b", "ask_or_adapt": "#dc2626",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    for axis, stream in zip(axes, ["iid_random", "abrupt_domain", "class_correlated"]):
        sub = df[df.stream == stream]
        for strategy in DISPLAY:
            stats = sub[sub.strategy == strategy].groupby("budget_percent").online_accuracy.agg(["mean", "std"])
            axis.errorbar(stats.index, 100 * stats["mean"], yerr=100 * stats["std"],
                          marker="o", linewidth=2 if strategy == "ask_or_adapt" else 1,
                          markersize=4, capsize=2, label=DISPLAY[strategy], color=colors[strategy])
        axis.set_xscale("log")
        axis.set_xticks([0.1, 0.5, 1, 2, 5], labels=["0.1", "0.5", "1", "2", "5"])
        axis.set_xlabel("Label budget (%)")
        axis.set_title(stream.replace("_", " ").title())
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Online accuracy (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out / "accuracy_budget_curves.pdf", bbox_inches="tight")
    fig.savefig(out / "accuracy_budget_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
