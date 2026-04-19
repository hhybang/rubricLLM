#!/usr/bin/env python3
"""Main paper figure: individual (persona, task) trajectories across A → C → D,
with condition means overlaid. One panel per metric."""

from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "analysis" / "condition_comparison.csv"
OUT = ROOT / "results" / "analysis"

EXCLUDE = {"Tom Brennan", "Miriam Osei"}
COND_ORDER = ["A", "C", "D"]
COND_LABELS = {
    "A": "A\n(Single-shot)",
    "C": "C\n(Iterative)",
    "D": "D\n(Iterative +\nCorrection)",
}
COND_X = {"A": 0, "C": 1, "D": 2}

COLORS = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
    "#8b5cf6", "#ec4899", "#06b6d4", "#f97316",
]


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[~df["persona"].isin(EXCLUDE)]
    df = df[df["condition"].isin(COND_ORDER)]

    personas = sorted(df["persona"].unique())
    color_map = {p: COLORS[i % len(COLORS)] for i, p in enumerate(personas)}

    metrics = [
        ("final_ground_truth_alignment", "Ground Truth Alignment (1–10)", (0.5, 10.8)),
        ("hidden_pref_satisfaction_rate", "Hidden-Preference Satisfaction Rate", (-0.05, 1.12)),
        ("final_overall_rubric_score", "Rubric Score (weighted, 0–1)", (-0.05, 1.12)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, (col, title, ylim) in zip(axes, metrics):
        # Individual trajectories (one line per persona–task pair)
        for persona in personas:
            psub = df[df["persona"] == persona]
            tasks = sorted(psub["task_index"].unique())
            for ti in tasks:
                tsub = psub[psub["task_index"] == ti].sort_values(
                    "condition", key=lambda s: s.map(COND_X)
                )
                xs = [COND_X[c] for c in tsub["condition"]]
                ys = tsub[col].values
                label = persona if ti == tasks[0] else None
                ax.plot(
                    xs, ys,
                    color=color_map[persona], alpha=0.35, linewidth=1.0,
                    marker="o", markersize=4, zorder=2, label=label,
                )

        # Condition means (thick black line with white edge markers)
        means = []
        for c in COND_ORDER:
            csub = df[df["condition"] == c][col].dropna()
            means.append(csub.mean())
        ax.plot(
            [0, 1, 2], means,
            color="black", linewidth=3, marker="D", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, zorder=4,
            label="Condition mean",
        )
        for xi, m in zip([0, 1, 2], means):
            fmt = f"{m:.1f}" if ylim[1] > 2 else f"{m:.2f}"
            ax.annotate(
                fmt, (xi, m), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
            )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.25)

    # Single shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5, fontsize=8,
        bbox_to_anchor=(0.5, -0.08), frameon=True,
    )

    fig.suptitle(
        "Effect of Iterative Grading and Rubric Correction on Draft Quality",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_results.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_main_results.pdf", bbox_inches="tight")
    print(f"Saved {OUT / 'fig_main_results.png'}")
    print(f"Saved {OUT / 'fig_main_results.pdf'}")


if __name__ == "__main__":
    main()
