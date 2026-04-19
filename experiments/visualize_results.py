#!/usr/bin/env python3
"""Per-persona panel: individual task scores across conditions A → C → D."""

from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "analysis" / "condition_comparison.csv"
OUT = ROOT / "results" / "analysis"

EXCLUDE = {"Tom Brennan", "Miriam Osei"}
COND_ORDER = ["A", "C", "D"]
COND_X = {"A": 0, "C": 1, "D": 2}
TASK_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}
TASK_COLORS = {0: "#2563eb", 1: "#dc2626"}


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[~df["persona"].isin(EXCLUDE)]
    df = df[df["condition"].isin(COND_ORDER)]

    personas = sorted(df["persona"].unique())
    n = len(personas)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    # --- Figure 1: Ground Truth Alignment (1–10) ---
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    fig1.suptitle("Ground Truth Alignment by Persona (1–10 scale, Opus judge)",
                  fontsize=14, fontweight="bold", y=1.01)

    for idx, persona in enumerate(personas):
        ax = axes1[idx // ncols][idx % ncols]
        sub = df[df["persona"] == persona]
        tasks = sorted(sub["task_index"].unique())
        for ti in tasks:
            tsub = sub[sub["task_index"] == ti].sort_values("condition", key=lambda s: s.map(COND_X))
            xs = [COND_X[c] for c in tsub["condition"]]
            ys = tsub["final_ground_truth_alignment"].values
            ax.plot(xs, ys, marker=TASK_MARKERS.get(ti, "o"), color=TASK_COLORS.get(ti, "#6b7280"),
                    linewidth=1.4, markersize=7, label=f"Task {ti}", alpha=0.85)
        ax.set_title(persona, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(COND_ORDER)
        ax.set_ylim(0.5, 10.8)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.grid(axis="y", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("GT alignment (1–10)")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for idx in range(len(personas), nrows * ncols):
        axes1[idx // ncols][idx % ncols].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(OUT / "fig_gt_alignment_by_persona.png", dpi=180, bbox_inches="tight")
    fig1.savefig(OUT / "fig_gt_alignment_by_persona.pdf", bbox_inches="tight")
    print(f"Saved {OUT / 'fig_gt_alignment_by_persona.png'}")

    # --- Figure 2: Hidden-Pref Satisfaction Rate (0–1) ---
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    fig2.suptitle("Hidden-Preference Satisfaction Rate by Persona (0–1)",
                  fontsize=14, fontweight="bold", y=1.01)

    for idx, persona in enumerate(personas):
        ax = axes2[idx // ncols][idx % ncols]
        sub = df[df["persona"] == persona]
        tasks = sorted(sub["task_index"].unique())
        for ti in tasks:
            tsub = sub[sub["task_index"] == ti].sort_values("condition", key=lambda s: s.map(COND_X))
            xs = [COND_X[c] for c in tsub["condition"]]
            ys = tsub["hidden_pref_satisfaction_rate"].values
            ax.plot(xs, ys, marker=TASK_MARKERS.get(ti, "o"), color=TASK_COLORS.get(ti, "#6b7280"),
                    linewidth=1.4, markersize=7, label=f"Task {ti}", alpha=0.85)
        ax.set_title(persona, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(COND_ORDER)
        ax.set_ylim(-0.05, 1.12)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.grid(axis="y", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("Satisfaction rate")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for idx in range(len(personas), nrows * ncols):
        axes2[idx // ncols][idx % ncols].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_satisfaction_by_persona.png", dpi=180, bbox_inches="tight")
    fig2.savefig(OUT / "fig_satisfaction_by_persona.pdf", bbox_inches="tight")
    print(f"Saved {OUT / 'fig_satisfaction_by_persona.png'}")

    # --- Figure 3: Rubric Score (0–1) ---
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    fig3.suptitle("Rubric Score by Persona (0–1 weighted criterion aggregate)",
                  fontsize=14, fontweight="bold", y=1.01)

    for idx, persona in enumerate(personas):
        ax = axes3[idx // ncols][idx % ncols]
        sub = df[df["persona"] == persona]
        tasks = sorted(sub["task_index"].unique())
        for ti in tasks:
            tsub = sub[sub["task_index"] == ti].sort_values("condition", key=lambda s: s.map(COND_X))
            xs = [COND_X[c] for c in tsub["condition"]]
            ys = tsub["final_overall_rubric_score"].values
            ax.plot(xs, ys, marker=TASK_MARKERS.get(ti, "o"), color=TASK_COLORS.get(ti, "#6b7280"),
                    linewidth=1.4, markersize=7, label=f"Task {ti}", alpha=0.85)
        ax.set_title(persona, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(COND_ORDER)
        ax.set_ylim(-0.05, 1.12)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.grid(axis="y", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("Rubric score (0–1)")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for idx in range(len(personas), nrows * ncols):
        axes3[idx // ncols][idx % ncols].set_visible(False)
    fig3.tight_layout()
    fig3.savefig(OUT / "fig_rubric_score_by_persona.png", dpi=180, bbox_inches="tight")
    fig3.savefig(OUT / "fig_rubric_score_by_persona.pdf", bbox_inches="tight")
    print(f"Saved {OUT / 'fig_rubric_score_by_persona.png'}")


if __name__ == "__main__":
    main()
