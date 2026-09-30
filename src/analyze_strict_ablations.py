from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DISPLAY = {
    "ask_or_adapt": "No risk switch",
    "v2_no_gate": "No persistence gate",
    "v2_no_entropy": "No entropy",
    "v2_no_disagreement": "No disagreement",
    "v2_no_coverage": "No coverage",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.root.rglob("*.summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["dataset"] = path.relative_to(args.root).parts[0]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "all_strict_ablation_runs.csv", index=False)
    means = frame.groupby(["dataset", "strategy"]).online_accuracy.mean().unstack("strategy")
    heat = pd.DataFrame({label: 100 * (means.ask_or_adapt_v2 - means[name]) for name, label in DISPLAY.items()})
    heat.to_csv(args.output / "strict_ablation_deltas.csv")
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    sns.heatmap(heat, annot=True, fmt="+.2f", center=0, cmap="RdBu_r", linewidths=.5,
                cbar_kws={"label": "Full method advantage (pp)"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Strict one-component ablations on OpenAI ViT-B/16")
    fig.tight_layout(); fig.savefig(args.output / "strict_ablation_heatmap.png", dpi=300, bbox_inches="tight"); fig.savefig(args.output / "strict_ablation_heatmap.pdf", bbox_inches="tight"); plt.close(fig)
    print(heat.to_string())


if __name__ == "__main__":
    main()
