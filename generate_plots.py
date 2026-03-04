#!/usr/bin/env python3
"""
Generate evaluation plots from simulation results.

Adapts the evaluate.ipynb plotting code to work with local simulation output
(JSON files) instead of Supabase. Produces the same RQ1/RQ2/RQ3 figures.

Usage:
    python generate_plots.py eval_output/sim_20260302_185324
    python generate_plots.py eval_output/sim_*            # multiple runs
    python generate_plots.py eval_output/sim_* --out figures/
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D


# ═════════════════════════════════════════════════════════════════════════════
# Visualization defaults (same as notebook)
# ═════════════════════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
PALETTE = sns.color_palette("colorblind")


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading — From local simulation JSON files
# ═════════════════════════════════════════════════════════════════════════════

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_simulation_data(run_dirs):
    """Load simulation data from local JSON files and build DataFrames.

    Mirrors the Supabase data loading + Cell 2/3 parsing from evaluate.ipynb.
    Returns dict of DataFrames keyed by name.
    """
    # Collect raw records per data type, per persona
    all_classification_records = []
    all_alignment_records = []
    all_rubric_records = []
    all_conversation_records = []
    user_counter = 0
    user_map = {}  # (run_dir, slug) -> anon label

    for run_dir in run_dirs:
        summary = load_json(os.path.join(run_dir, "summary.json"))
        if not summary:
            continue

        for entry in sorted(os.listdir(run_dir)):
            persona_dir = os.path.join(run_dir, entry)
            if not os.path.isdir(persona_dir) or entry.startswith("."):
                continue

            user_counter += 1
            user_label = f"P{user_counter}"
            user_map[(run_dir, entry)] = user_label

            pd_dir = os.path.join(persona_dir, "project_data")

            # --- Classification data (RQ1) ---
            clf_data = load_json(os.path.join(pd_dir, "criteria_classification_feedback.json"))
            if clf_data:
                items = clf_data if isinstance(clf_data, list) else [clf_data]
                for item in items:
                    if isinstance(item, dict):
                        item["_user"] = user_label
                        all_classification_records.append(item)

            # --- Alignment diagnostic data (RQ2 + RQ3) ---
            alignment_data = load_json(os.path.join(pd_dir, "alignment_diagnostic.json"))
            if alignment_data:
                items = alignment_data if isinstance(alignment_data, list) else [alignment_data]
                for item in items:
                    if isinstance(item, dict):
                        item["_user"] = user_label
                        all_alignment_records.append(item)

            # --- Rubric history (RQ3) ---
            rubric_hist = load_json(os.path.join(persona_dir, "rubric_history.json"))
            if rubric_hist:
                items = rubric_hist if isinstance(rubric_hist, list) else [rubric_hist]
                for item in items:
                    if isinstance(item, dict):
                        criteria = item.get("rubric", [])
                        all_rubric_records.append({
                            "_user": user_label,
                            "version": item.get("version", 0),
                            "n_criteria": len(criteria),
                            "criteria_names": [c.get("name", "") for c in criteria],
                            "source": item.get("source", ""),
                        })

            # --- Conversations ---
            conv = load_json(os.path.join(persona_dir, "conversations.json"))
            if conv:
                messages = conv.get("messages", []) or []
                n_drafts = sum(
                    1 for m in messages
                    if re.search(r'<draft>.*?</draft>', m.get('content', ''), re.DOTALL)
                )
                all_conversation_records.append({
                    "_user": user_label,
                    "n_messages": len(messages),
                    "n_drafts": n_drafts,
                    "n_user_msgs": sum(1 for m in messages if m.get("role") == "user"),
                    "n_assistant_msgs": sum(1 for m in messages if m.get("role") == "assistant"),
                })

    df_classifications = pd.DataFrame(all_classification_records)
    df_alignment = pd.DataFrame(all_alignment_records)
    df_rubric_versions = pd.DataFrame(all_rubric_records)
    df_conversations = pd.DataFrame(all_conversation_records)

    users = sorted(set(
        [r["_user"] for r in all_classification_records] +
        [r["_user"] for r in all_alignment_records]
    )) or [f"P{i+1}" for i in range(user_counter)]

    return {
        "users": users,
        "df_classifications": df_classifications,
        "df_alignment": df_alignment,
        "df_rubric_versions": df_rubric_versions,
        "df_conversations": df_conversations,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Aggregate DataFrames (mirrors Cell 3 of notebook)
# ═════════════════════════════════════════════════════════════════════════════

def build_rq1_data(users, df_classifications, df_alignment):
    """Build RQ1 per-user summary and criteria importance DataFrames."""
    rq1_rows = []
    rq1_criteria_rows = []

    for user in users:
        row = {"user": user}
        uc = df_classifications[df_classifications["_user"] == user] if not df_classifications.empty else pd.DataFrame()

        if not uc.empty:
            r = uc.iloc[-1]
            row["stated"] = r.get("n_stated", 0)
            row["real"] = r.get("n_real", 0)
            row["hallucinated"] = r.get("n_hallucinated", 0)
            row["precision"] = r.get("precision", np.nan)

            classifications = r.get("classifications", {})
            importance_ranking = r.get("importance_ranking", [])
            if isinstance(classifications, str):
                classifications = json.loads(classifications)
            if isinstance(importance_ranking, str):
                importance_ranking = json.loads(importance_ranking)

            total_criteria = len(classifications) if classifications else 1
            for crit_name, origin in (classifications or {}).items():
                rank = np.nan
                if importance_ranking:
                    try:
                        rank = importance_ranking.index(crit_name) + 1
                    except ValueError:
                        rank = np.nan
                rq1_criteria_rows.append({
                    "user": user, "criterion": crit_name, "origin": origin,
                    "importance_rank": rank, "n_criteria": total_criteria,
                    "normalized_rank": rank / total_criteria if pd.notna(rank) and total_criteria > 0 else np.nan,
                })
        else:
            row.update({"stated": 0, "real": 0, "hallucinated": 0, "precision": np.nan})

        total = row.get("stated", 0) + row.get("real", 0) + row.get("hallucinated", 0)
        row["total_criteria"] = total
        row["stated_rate"] = row.get("stated", 0) / total if total > 0 else np.nan
        row["real_rate"] = row.get("real", 0) / total if total > 0 else np.nan
        row["hallucinated_rate"] = row.get("hallucinated", 0) / total if total > 0 else np.nan

        # Rubric rank from alignment diagnostic
        ua = df_alignment[df_alignment["_user"] == user] if not df_alignment.empty else pd.DataFrame()
        if not ua.empty:
            ranking = ua.iloc[-1].get("user_ranking", [])
            if isinstance(ranking, str):
                ranking = json.loads(ranking)
            row["rubric_rank"] = (ranking.index("rubric") + 1) if "rubric" in ranking else np.nan
        else:
            row["rubric_rank"] = np.nan

        rq1_rows.append(row)

    return pd.DataFrame(rq1_rows), pd.DataFrame(rq1_criteria_rows)


def build_rq2_data(users, df_alignment):
    """Build RQ2 ranking alignment and draft preference DataFrames."""
    CONDITIONS_DRAFT = ["generic", "pref_desc", "rubric"]
    source_to_cond = {"rubric": "rubric", "generic": "generic", "preference": "pref_desc"}

    rq2_rank_pair_rows = []
    rq2_draft_pref_rows = []

    for user in users:
        ua = df_alignment[df_alignment["_user"] == user] if not df_alignment.empty else pd.DataFrame()
        if ua.empty:
            continue

        arow = ua.iloc[-1]
        user_ranking = arow.get("user_ranking", [])
        shuffle_order = arow.get("shuffle_order", [])
        rubric_judge = arow.get("rubric_judge_result", {})
        is_3draft = arow.get("is_3draft", False)

        if isinstance(user_ranking, str):
            user_ranking = json.loads(user_ranking)
        if isinstance(shuffle_order, str):
            shuffle_order = json.loads(shuffle_order)
        if isinstance(rubric_judge, str):
            rubric_judge = json.loads(rubric_judge)

        label_to_source = {}
        for pair in (shuffle_order or []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                label_to_source[pair[0]] = pair[1]

        user_rank_map = {src: i + 1 for i, src in enumerate(user_ranking)}

        rj_ranking = (rubric_judge or {}).get("overall_ranking", [])
        rj_rank_map = {}
        if rj_ranking and is_3draft:
            for rank_idx, draft_label in enumerate(rj_ranking):
                letter = draft_label.replace("DRAFT_", "").replace("draft_", "")
                source = label_to_source.get(letter, label_to_source.get(letter.upper()))
                if source:
                    rj_rank_map[source] = rank_idx + 1

        for src in user_rank_map:
            cond = source_to_cond.get(src, src)
            rq2_rank_pair_rows.append({
                "user": user, "condition": cond,
                "user_rank": user_rank_map[src],
                "judge_rank": rj_rank_map.get(src, np.nan),
            })

        for src_i, src in enumerate(user_ranking):
            cond = source_to_cond.get(src, src)
            rq2_draft_pref_rows.append({
                "user": user, "condition": cond, "rank": src_i + 1,
            })

    return pd.DataFrame(rq2_rank_pair_rows), pd.DataFrame(rq2_draft_pref_rows)


def build_rq3_data(users, df_alignment):
    """Build RQ3 criteria trajectory and version summary DataFrames."""
    CLASS_ORDER = ["DIFFERENTIATING", "REDUNDANT", "PREFERENCE_GAP", "UNDERPERFORMING"]
    rq3_crit_rows = []
    rq3_version_rows = []

    for user in users:
        ua = df_alignment[df_alignment["_user"] == user] if not df_alignment.empty else pd.DataFrame()
        if ua.empty:
            continue

        ua_sorted = ua.sort_values("rubric_version") if "rubric_version" in ua.columns else ua

        for _, arow in ua_sorted.iterrows():
            version = arow.get("rubric_version", 0)
            criteria_analysis = arow.get("criteria_analysis", [])
            if isinstance(criteria_analysis, str):
                criteria_analysis = json.loads(criteria_analysis)
            if not criteria_analysis:
                continue

            version_counts = Counter()
            version_gaps = []

            for crit in criteria_analysis:
                cname = crit.get("name", "?")
                classification = crit.get("classification", "UNKNOWN")
                gap = crit.get("gap", 0)
                rubric_score = crit.get("rubric_score", np.nan)
                generic_score = crit.get("generic_score", np.nan)

                rq3_crit_rows.append({
                    "user": user, "rubric_version": version,
                    "criterion": cname, "classification": classification,
                    "gap": gap, "rubric_score": rubric_score,
                    "generic_score": generic_score,
                })
                version_counts[classification] += 1
                if gap is not None:
                    version_gaps.append(gap)

            n_crit = len(criteria_analysis)
            rq3_version_rows.append({
                "user": user, "rubric_version": version, "n_criteria": n_crit,
                "n_differentiating": version_counts.get("DIFFERENTIATING", 0),
                "n_redundant": version_counts.get("REDUNDANT", 0),
                "n_preference_gap": version_counts.get("PREFERENCE_GAP", 0),
                "n_underperforming": version_counts.get("UNDERPERFORMING", 0),
                "pct_differentiating": version_counts.get("DIFFERENTIATING", 0) / n_crit if n_crit else 0,
                "mean_gap": float(np.mean(version_gaps)) if version_gaps else 0,
            })

    return pd.DataFrame(rq3_crit_rows), pd.DataFrame(rq3_version_rows)


# ═════════════════════════════════════════════════════════════════════════════
# Plotting (mirrors Cells 5-7 of notebook)
# ═════════════════════════════════════════════════════════════════════════════

def plot_rq1(df_rq1, df_rq1_criteria, out_dir):
    """RQ1: Does iterative elicitation surface preferences that direct elicitation misses?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.4)
    d = df_rq1[df_rq1["precision"].notna()]
    N = len(d)

    # (a) Precision & Real Rate
    ax = axes[0]
    if not d.empty:
        total_all = d["total_criteria"].sum()
        pooled_stated_rate = d["stated"].sum() / total_all if total_all else 0
        pooled_real_rate = d["real"].sum() / total_all if total_all else 0

        metrics = ["Stated\n(explicit)", "Inferred\n(real)"]
        pooled_vals = [pooled_stated_rate, pooled_real_rate]
        bar_colors = [PALETTE[2], PALETTE[0]]

        bars = ax.bar(metrics, pooled_vals, color=bar_colors, width=0.5,
                      edgecolor="white", linewidth=1.5, alpha=0.35, zorder=2)

        rate_cols = ["stated_rate", "real_rate"]
        jitter_rng = np.random.default_rng(42)
        for i, (col, color) in enumerate(zip(rate_cols, bar_colors)):
            vals = d[col].dropna()
            if not vals.empty:
                jitter = jitter_rng.uniform(-0.1, 0.1, size=len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=90,
                           color=color, edgecolors="white", linewidth=1.2,
                           zorder=4, alpha=0.9)
                ax.plot([i - 0.18, i + 0.18], [vals.mean(), vals.mean()],
                        color=color, linewidth=3, zorder=5)

        for bar, v in zip(bars, pooled_vals):
            ax.text(bar.get_x() + bar.get_width()/2, max(v, 0.02) + 0.04,
                    f"{v:.0%}", ha="center", fontsize=12, fontweight="bold", color="black")

        ax.set_ylabel("Proportion of Rubric Criteria")
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color="lightgray", linestyle="--", linewidth=0.8, zorder=1)
        ax.text(0.97, 0.97, f"N = {N} users",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, "No classification data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("(a) Criteria Origin Rates", fontweight="bold", fontsize=11)

    # (b) Importance Ranking by Origin
    ax = axes[1]
    crit = df_rq1_criteria.copy() if not df_rq1_criteria.empty else pd.DataFrame()
    crit = crit[crit["origin"].isin(["stated", "real"])] if not crit.empty else crit
    crit_with_rank = crit[crit["importance_rank"].notna()] if not crit.empty else pd.DataFrame()

    if not crit_with_rank.empty:
        origin_order = ["stated", "real"]
        origin_labels = ["Stated\n(explicit)", "Inferred\n(real)"]
        origin_colors = [PALETTE[2], PALETTE[0]]

        max_rank = crit_with_rank["importance_rank"].max()
        jitter_rng = np.random.default_rng(44)

        for i, (origin, label, color) in enumerate(zip(origin_order, origin_labels, origin_colors)):
            subset = crit_with_rank[crit_with_rank["origin"] == origin]
            if not subset.empty:
                jitter = jitter_rng.uniform(-0.12, 0.12, size=len(subset))
                ax.scatter(np.full(len(subset), i) + jitter, subset["importance_rank"],
                           s=110, color=color, edgecolors="white", linewidth=1.2,
                           zorder=3, alpha=0.85)
                mean_rank = subset["importance_rank"].mean()
                ax.plot([i - 0.22, i + 0.22], [mean_rank, mean_rank],
                        color=color, linewidth=3, zorder=4)
                ax.text(i + 0.28, mean_rank, f"M={mean_rank:.1f}",
                        va="center", fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(len(origin_labels)))
        ax.set_xticklabels(origin_labels)
        ax.set_ylabel("User Importance Rank")
        ax.set_ylim(0.3, max_rank + 0.7)
        ax.invert_yaxis()

        ax.annotate("More\nimportant",
                    xy=(-0.18, 0.97), xycoords="axes fraction",
                    xytext=(-0.18, 0.72), textcoords="axes fraction",
                    fontsize=7, color="gray", ha="center", va="top",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

        mid = max_rank / 2
        ax.axhline(mid, color="lightgray", linestyle="--", linewidth=0.8, zorder=1)

        ax.text(0.97, 0.03, f"N = {N} users",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.55, "Importance ranking data\nnot yet available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_title("(b) Importance Ranking by Origin", fontweight="bold", fontsize=11)

    fig.suptitle("RQ1: Does iterative elicitation surface preferences that direct elicitation misses?",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig01_rq1.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig01_rq1.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig01_rq1: N={N} users with classification data")


def plot_rq2(df_rq2_rank_pairs, df_rq2_draft_pref, out_dir):
    """RQ2: Does rubric-grounding improve LLM-as-judge alignment?"""
    DRAFT_LABELS = {"generic": "Generic", "pref_desc": "Pref-desc", "rubric": "Rubric"}
    DRAFT_COLORS = {"generic": PALETTE[7], "pref_desc": PALETTE[1], "rubric": PALETTE[0]}
    COND_LABELS = {"generic": "Generic", "rubric": "Rubric", "pref_desc": "Pref-desc"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.4)
    conds = ["rubric", "generic", "pref_desc"]

    # (a) Ranking Alignment
    ax = axes[0]
    if not df_rq2_rank_pairs.empty:
        users_with_data = df_rq2_rank_pairs["user"].unique()
        n_users = len(users_with_data)

        x = np.arange(len(conds))
        width = 0.3
        u_means = [df_rq2_rank_pairs[df_rq2_rank_pairs["condition"] == d]["user_rank"].mean() for d in conds]
        j_means = [df_rq2_rank_pairs[df_rq2_rank_pairs["condition"] == d]["judge_rank"].mean() for d in conds]

        n_drafts = 3
        u_scores = [n_drafts + 1 - m for m in u_means]
        j_scores = [n_drafts + 1 - m for m in j_means]
        colors = [DRAFT_COLORS[d] for d in conds]

        bars1 = ax.bar(x - width/2, u_scores, width, color=colors, alpha=0.5,
                       edgecolor="white", linewidth=1.5, label="User")
        bars2 = ax.bar(x + width/2, j_scores, width, color=colors, alpha=1.0,
                       edgecolor="white", linewidth=1.5, hatch="//", label="Rubric Judge")

        ax.set_xticks(x)
        ax.set_xticklabels([DRAFT_LABELS[d] for d in conds])
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["#3 (worst)", "#2", "#1 (best)"])
        ax.set_ylim(0, 3.6)
        ax.legend(fontsize=9, loc="upper right")
        ax.text(0.03, 0.97, f"N = {n_users} {'user' if n_users == 1 else 'users'}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, "Awaiting ranking data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_title("(a) Ranking Alignment", fontweight="bold", fontsize=11)

    # (b) Draft Preference Distribution
    ax = axes[1]
    if not df_rq2_draft_pref.empty:
        n_users = df_rq2_draft_pref["user"].nunique()
        for cond in conds:
            cond_data = df_rq2_draft_pref[df_rq2_draft_pref["condition"] == cond]
            if not cond_data.empty:
                counts = cond_data["rank"].value_counts().sort_index()
                for rank_val in [1, 2, 3]:
                    if rank_val not in counts.index:
                        counts[rank_val] = 0
                counts = counts.sort_index()
                pcts = counts / counts.sum()

        # Simple bar chart: how often each condition is ranked #1
        rank1_counts = {}
        for cond in conds:
            cond_data = df_rq2_draft_pref[df_rq2_draft_pref["condition"] == cond]
            rank1_counts[cond] = len(cond_data[cond_data["rank"] == 1]) if not cond_data.empty else 0

        total = sum(rank1_counts.values()) or 1
        x_pos = np.arange(len(conds))
        colors = [DRAFT_COLORS[d] for d in conds]
        vals = [rank1_counts[d] / total for d in conds]

        bars = ax.bar(x_pos, vals, color=colors, width=0.5,
                      edgecolor="white", linewidth=1.5, alpha=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, max(v, 0.02) + 0.03,
                    f"{v:.0%}", ha="center", fontsize=12, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([DRAFT_LABELS[d] for d in conds])
        ax.set_ylabel("Proportion Ranked #1")
        ax.set_ylim(0, 1.15)
        ax.text(0.97, 0.97, f"N = {n_users} users",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, "Awaiting draft preference data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_title("(b) Draft Preference", fontweight="bold", fontsize=11)

    fig.suptitle("RQ2: Does rubric-grounding improve LLM-as-judge alignment with user preferences?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig02_rq2.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig02_rq2.png", bbox_inches="tight")
    plt.close(fig)

    n = df_rq2_rank_pairs["user"].nunique() if not df_rq2_rank_pairs.empty else 0
    print(f"  fig02_rq2: N={n} users with RQ2 data")


def plot_rq3(df_rq3_criteria, df_rq3_versions, out_dir):
    """RQ3: Does iterative rubric refinement improve alignment over time?"""
    CLASS_ORDER = ["DIFFERENTIATING", "REDUNDANT", "PREFERENCE_GAP", "UNDERPERFORMING"]
    CLASS_COLORS = {
        "DIFFERENTIATING": PALETTE[2],
        "REDUNDANT": PALETTE[7],
        "PREFERENCE_GAP": PALETTE[1],
        "UNDERPERFORMING": PALETTE[3],
    }
    CLASS_LABELS = {
        "DIFFERENTIATING": "Differentiating",
        "REDUNDANT": "Redundant",
        "PREFERENCE_GAP": "Pref. Gap",
        "UNDERPERFORMING": "Underperforming",
    }

    N3 = df_rq3_versions["user"].nunique() if not df_rq3_versions.empty else 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.4)

    # (a) Classification over Iterations
    ax = axes[0]
    if not df_rq3_criteria.empty:
        all_versions = sorted(df_rq3_criteria["rubric_version"].unique())
        users = sorted(df_rq3_criteria["user"].unique())

        means = {cls: [] for cls in CLASS_ORDER}
        valid_versions = []
        for v in all_versions:
            v_data = df_rq3_criteria[df_rq3_criteria["rubric_version"] == v]
            v_users = v_data["user"].unique()
            if len(v_users) == 0:
                continue
            valid_versions.append(v)
            for cls in CLASS_ORDER:
                per_user_pcts = []
                for u in v_users:
                    u_data = v_data[v_data["user"] == u]
                    n = len(u_data)
                    per_user_pcts.append(
                        len(u_data[u_data["classification"] == cls]) / n if n > 0 else 0
                    )
                means[cls].append(np.mean(per_user_pcts))

        x_pos = np.array(valid_versions, dtype=float)
        bottom = np.zeros(len(valid_versions))
        for cls in CLASS_ORDER:
            vals = np.array(means[cls])
            ax.bar(x_pos, vals, bottom=bottom, width=0.6,
                   color=CLASS_COLORS[cls], alpha=0.7, edgecolor="white", linewidth=1,
                   label=CLASS_LABELS[cls])
            bottom += vals

        ax.set_xlabel("Rubric Version")
        ax.set_ylabel("Proportion of Criteria")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(valid_versions)
        ax.set_xticklabels([f"v{int(v)}" for v in valid_versions])
        ax.legend(fontsize=7, loc="upper right",
                  bbox_to_anchor=(1.0, -0.08), ncol=4, frameon=False)
        ax.text(0.03, 0.97, f"N = {N3} {'user' if N3 == 1 else 'users'}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, "Awaiting diagnostic data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_title("(a) Classification over Iterations", fontweight="bold", fontsize=11)

    # (b) Score Gap over Iterations
    ax = axes[1]
    if not df_rq3_criteria.empty:
        all_versions = sorted(df_rq3_criteria["rubric_version"].unique())
        users = sorted(df_rq3_criteria["user"].unique())

        for u in users:
            u_data = df_rq3_criteria[df_rq3_criteria["user"] == u]
            u_versions = sorted(u_data["rubric_version"].unique())
            u_gaps = [u_data[u_data["rubric_version"] == v]["gap"].mean() for v in u_versions]
            ax.plot(u_versions, u_gaps, "o-", color=PALETTE[0], alpha=0.25,
                    linewidth=1.2, markersize=5)

        pooled_vs, pooled_gaps = [], []
        for v in all_versions:
            v_data = df_rq3_criteria[df_rq3_criteria["rubric_version"] == v]
            v_users = v_data["user"].unique()
            if len(v_users) == 0:
                continue
            user_means = [v_data[v_data["user"] == u]["gap"].mean() for u in v_users]
            pooled_vs.append(v)
            pooled_gaps.append(np.mean(user_means))

        ax.plot(pooled_vs, pooled_gaps, "o-", color=PALETTE[0], linewidth=3, markersize=10,
                markeredgecolor="white", markeredgewidth=2, zorder=5)

        for v, g in zip(pooled_vs, pooled_gaps):
            color = PALETTE[2] if g >= 0 else PALETTE[3]
            ax.text(v + 0.08, g + 0.03, f"{g:+.2f}",
                    fontsize=10, fontweight="bold", color=color)

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Rubric Version")
        ax.set_ylabel("Mean Rubric - Generic Gap")
        ax.set_xticks(all_versions)
        ax.set_xticklabels([f"v{int(v)}" for v in all_versions])

        ax.text(0.97, 0.97, "Rubric better ↑", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=PALETTE[2], alpha=0.6)
        ax.text(0.97, 0.03, "↓ Generic better", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=PALETTE[3], alpha=0.6)
        ax.text(0.03, 0.97, f"N = {N3} {'user' if N3 == 1 else 'users'}\n(thin = per user, bold = mean)",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, "Awaiting diagnostic data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_title("(b) Score Gap over Iterations", fontweight="bold", fontsize=11)

    fig.suptitle("RQ3: Does iterative rubric refinement improve alignment over time?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig03_rq3.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig03_rq3.png", bbox_inches="tight")
    plt.close(fig)

    if df_rq3_criteria.empty:
        print("  fig03_rq3: Awaiting alignment diagnostic data")
    else:
        n_versions = df_rq3_criteria["rubric_version"].nunique()
        print(f"  fig03_rq3: N={N3} users, {n_versions} rubric version(s)")


# ═════════════════════════════════════════════════════════════════════════════
# Termination-specific plots (new for simulation pipeline)
# ═════════════════════════════════════════════════════════════════════════════

def plot_termination(run_dirs, out_dir):
    """Plot termination metrics: satisfaction/quality scores over drafts, goal-reached rates."""
    all_turn_metrics = []
    all_iteration_summaries = []

    for run_dir in run_dirs:
        for entry in sorted(os.listdir(run_dir)):
            persona_dir = os.path.join(run_dir, entry)
            if not os.path.isdir(persona_dir) or entry.startswith("."):
                continue

            tlog = load_json(os.path.join(persona_dir, "project_data", "termination_log.json"))
            if not tlog:
                continue

            log_entry = tlog[-1] if isinstance(tlog, list) else tlog
            for it in log_entry.get("iterations", []):
                reason = it.get("termination_reason", "unknown")
                drafts = it.get("drafts_produced", it.get("turns_used", 0))
                all_iteration_summaries.append({
                    "persona": entry,
                    "iteration": it.get("iteration", 0),
                    "reason": reason,
                    "drafts_produced": drafts,
                })

                for m in it.get("turn_metrics", []):
                    if m.get("has_draft") and "satisfaction" in m:
                        all_turn_metrics.append({
                            "persona": entry,
                            "iteration": it.get("iteration", 0),
                            "draft_number": m.get("draft_number", m.get("turn", 0)),
                            "satisfaction": m["satisfaction"].get("score", 0),
                            "quality": m.get("quality", {}).get("score", 0),
                        })

    if not all_turn_metrics and not all_iteration_summaries:
        print("  fig04_termination: No termination data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.4)

    # (a) Satisfaction & Quality scores over draft number
    ax = axes[0]
    if all_turn_metrics:
        df_tm = pd.DataFrame(all_turn_metrics)
        draft_nums = sorted(df_tm["draft_number"].unique())

        sat_means = [df_tm[df_tm["draft_number"] == d]["satisfaction"].mean() for d in draft_nums]
        qual_means = [df_tm[df_tm["draft_number"] == d]["quality"].mean() for d in draft_nums]

        ax.plot(draft_nums, sat_means, "o-", color=PALETTE[0], linewidth=2.5,
                markersize=8, label="Satisfaction", markeredgecolor="white", markeredgewidth=1.5)
        ax.plot(draft_nums, qual_means, "s--", color=PALETTE[1], linewidth=2.5,
                markersize=8, label="Quality", markeredgecolor="white", markeredgewidth=1.5)

        # Per-persona thin lines
        for persona in df_tm["persona"].unique():
            p_data = df_tm[df_tm["persona"] == persona]
            p_drafts = sorted(p_data["draft_number"].unique())
            p_sat = [p_data[p_data["draft_number"] == d]["satisfaction"].mean() for d in p_drafts]
            p_qual = [p_data[p_data["draft_number"] == d]["quality"].mean() for d in p_drafts]
            ax.plot(p_drafts, p_sat, "o-", color=PALETTE[0], alpha=0.15, linewidth=0.8, markersize=3)
            ax.plot(p_drafts, p_qual, "s--", color=PALETTE[1], alpha=0.15, linewidth=0.8, markersize=3)

        ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(max(draft_nums) + 0.1, 0.8, "threshold", fontsize=7, color="gray", va="center")
        ax.set_xlabel("Draft Number")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        n_personas = df_tm["persona"].nunique()
        ax.text(0.03, 0.03, f"N = {n_personas} personas\n(thin = per persona, bold = mean)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, "No draft-level data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("(a) Scores over Drafts", fontweight="bold", fontsize=11)

    # (b) Termination reasons
    ax = axes[1]
    if all_iteration_summaries:
        df_is = pd.DataFrame(all_iteration_summaries)
        reason_counts = df_is["reason"].value_counts()

        colors = {"goal_reached": PALETTE[2], "user_accepted": PALETTE[2], "max_drafts": PALETTE[3], "max_turns": PALETTE[3]}
        bar_colors = [colors.get(r, PALETTE[7]) for r in reason_counts.index]

        bars = ax.bar(range(len(reason_counts)), reason_counts.values,
                      color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(reason_counts)))
        ax.set_xticklabels(reason_counts.index, fontsize=10)
        ax.set_ylabel("Number of Iterations")

        for bar, v in zip(bars, reason_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                    str(v), ha="center", fontsize=12, fontweight="bold")

        total = len(df_is)
        goal_pct = (reason_counts.get("goal_reached", 0) + reason_counts.get("user_accepted", 0)) / total * 100
        ax.text(0.97, 0.97, f"User accepted: {goal_pct:.0f}%\n({total} total iterations)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, "No iteration data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("(b) Termination Reasons", fontweight="bold", fontsize=11)

    fig.suptitle("Goal-Based Termination Metrics",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig04_termination.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig04_termination.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig04_termination: {len(all_iteration_summaries)} iterations, {len(all_turn_metrics)} draft metrics")


def plot_log_changes(run_dirs, out_dir):
    """Plot rubric edit types from Log Changes across personas."""
    all_edits = []

    for run_dir in run_dirs:
        for entry in sorted(os.listdir(run_dir)):
            persona_dir = os.path.join(run_dir, entry)
            if not os.path.isdir(persona_dir) or entry.startswith("."):
                continue

            lc = load_json(os.path.join(persona_dir, "project_data", "log_changes.json"))
            if not lc:
                continue

            lc_entry = lc[-1] if isinstance(lc, list) else lc
            edits = lc_entry.get("edit_classification", {})

            for edit_type in ("added", "removed", "reweighted", "reworded", "dimensions_changed"):
                count = len(edits.get(edit_type, []))
                if count > 0:
                    all_edits.append({
                        "persona": entry,
                        "edit_type": edit_type,
                        "count": count,
                    })

    if not all_edits:
        print("  fig05_log_changes: No log changes data found")
        return

    df = pd.DataFrame(all_edits)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate by edit type
    type_totals = df.groupby("edit_type")["count"].sum().sort_values(ascending=False)
    type_order = type_totals.index.tolist()
    type_labels = {
        "added": "Added",
        "removed": "Removed",
        "reweighted": "Reweighted",
        "reworded": "Reworded",
        "dimensions_changed": "Dims Changed",
    }

    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(type_order))]
    bars = ax.bar(
        range(len(type_order)),
        [type_totals[t] for t in type_order],
        color=bar_colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels([type_labels.get(t, t) for t in type_order], fontsize=10)
    ax.set_ylabel("Number of Edits")
    ax.set_title("Rubric Edits by Type (Log Changes)", fontweight="bold", fontsize=12)

    for bar, v in zip(bars, [type_totals[t] for t in type_order]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                str(int(v)), ha="center", fontsize=12, fontweight="bold")

    n_personas = df["persona"].nunique()
    total_edits = int(df["count"].sum())
    ax.text(0.97, 0.97, f"{total_edits} total edits\nacross {n_personas} personas",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))

    fig.savefig(out_dir / "fig05_log_changes.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig05_log_changes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig05_log_changes: {total_edits} edits across {n_personas} personas")


def plot_surveys(run_dirs, out_dir):
    """Plot survey results: understanding/effort scores, comparisons, and accuracy."""
    all_task_a = []
    all_task_b = []
    all_final = []

    for run_dir in run_dirs:
        for entry in sorted(os.listdir(run_dir)):
            persona_dir = os.path.join(run_dir, entry)
            if not os.path.isdir(persona_dir) or entry.startswith("."):
                continue

            pd_dir = os.path.join(persona_dir, "project_data")

            # Task A
            sa = load_json(os.path.join(pd_dir, "survey_task_a.json"))
            if sa:
                items = sa if isinstance(sa, list) else [sa]
                for item in items:
                    if isinstance(item, dict) and item.get("q1") is not None:
                        all_task_a.append({
                            "persona": entry, "survey": "Task A (no rubric)",
                            "understanding": item["q1"], "effort": item["q2"],
                        })

            # Task B
            sb = load_json(os.path.join(pd_dir, "survey_task_b.json"))
            if sb:
                items = sb if isinstance(sb, list) else [sb]
                for item in items:
                    if isinstance(item, dict) and item.get("q1") is not None:
                        all_task_b.append({
                            "persona": entry,
                            "iteration": item.get("iteration", 1),
                            "understanding": item["q1"], "effort": item["q2"],
                            "comparison": item.get("q4", "About the same"),
                        })

            # Final
            sf = load_json(os.path.join(pd_dir, "survey_final_review.json"))
            if sf:
                item = sf[-1] if isinstance(sf, list) else sf
                ratings = item.get("criteria_ratings", {})
                for crit_name, rating in ratings.items():
                    if isinstance(rating, dict):
                        all_final.append({
                            "persona": entry,
                            "criterion": crit_name,
                            "accuracy": rating.get("accuracy", "Partially right"),
                        })

    has_ab = bool(all_task_a or all_task_b)
    has_final = bool(all_final)
    if not has_ab and not has_final:
        print("  fig06_surveys: No survey data found")
        return

    n_panels = (1 if has_ab else 0) + (1 if all_task_b else 0) + (1 if has_final else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # Panel (a): Understanding & Effort — Task A vs Task B
    if has_ab:
        ax = axes[ax_idx]; ax_idx += 1

        scores = []
        for item in all_task_a:
            scores.append({"type": "Task A", "metric": "Understanding", "score": item["understanding"]})
            scores.append({"type": "Task A", "metric": "Effort", "score": item["effort"]})
        for item in all_task_b:
            scores.append({"type": f"Task B (iter {item['iteration']})", "metric": "Understanding", "score": item["understanding"]})
            scores.append({"type": f"Task B (iter {item['iteration']})", "metric": "Effort", "score": item["effort"]})

        df_scores = pd.DataFrame(scores)

        # Group by type+metric
        grouped = df_scores.groupby(["type", "metric"])["score"].mean().reset_index()

        # Simple grouped bar: Understanding and Effort side by side per survey type
        types = sorted(grouped["type"].unique(), key=lambda x: (0 if "Task A" in x else 1, x))
        x = np.arange(len(types))
        width = 0.35

        for i, metric in enumerate(["Understanding", "Effort"]):
            vals = []
            for t in types:
                subset = grouped[(grouped["type"] == t) & (grouped["metric"] == metric)]
                vals.append(subset["score"].values[0] if len(subset) > 0 else 0)
            offset = -width / 2 + i * width
            bars = ax.bar(x + offset, vals, width, label=metric,
                          color=PALETTE[i], edgecolor="white", linewidth=1)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                        f"{v:.1f}", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(types, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Score (1-5)")
        ax.set_ylim(0, 5.8)
        ax.set_title("(a) Understanding & Effort", fontweight="bold", fontsize=11)
        ax.legend(fontsize=9)

    # Panel (b): Comparison distribution (Task B Q4)
    if all_task_b:
        ax = axes[ax_idx]; ax_idx += 1

        comp_order = ["Much better", "Somewhat better", "About the same", "Somewhat worse", "Much worse"]
        comp_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4] if len(PALETTE) > 4 else "#999"]
        comp_counts = {c: 0 for c in comp_order}
        for item in all_task_b:
            comp = item.get("comparison", "About the same")
            if comp in comp_counts:
                comp_counts[comp] += 1

        vals = [comp_counts[c] for c in comp_order]
        bars = ax.barh(range(len(comp_order)), vals,
                       color=comp_colors, edgecolor="white", linewidth=1)
        ax.set_yticks(range(len(comp_order)))
        ax.set_yticklabels(comp_order, fontsize=9)
        ax.set_xlabel("Count")
        ax.set_title("(b) Task B: vs Previous Task", fontweight="bold", fontsize=11)
        ax.invert_yaxis()

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(v), va="center", fontsize=10, fontweight="bold")

    # Panel (c): Final Review accuracy
    if has_final:
        ax = axes[ax_idx]; ax_idx += 1

        df_final = pd.DataFrame(all_final)
        acc_counts = df_final["accuracy"].value_counts()
        acc_order = ["Accurate", "Partially right", "Inaccurate"]
        acc_colors = [PALETTE[0], PALETTE[2], PALETTE[3] if len(PALETTE) > 3 else "#e74c3c"]

        vals = [acc_counts.get(a, 0) for a in acc_order]
        total = sum(vals)
        bars = ax.bar(range(len(acc_order)), vals,
                      color=acc_colors, edgecolor="white", linewidth=1.5, width=0.5)
        ax.set_xticks(range(len(acc_order)))
        ax.set_xticklabels(acc_order, fontsize=10)
        ax.set_ylabel("Number of Criteria")
        ax.set_title("(c) Final Review: Rubric Accuracy", fontweight="bold", fontsize=11)

        for bar, v in zip(bars, vals):
            pct = f"{v / total * 100:.0f}%" if total > 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                    f"{v} ({pct})", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "fig06_surveys.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig06_surveys.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig06_surveys: {len(all_task_a)} Task A, {len(all_task_b)} Task B, {len(all_final)} Final Review ratings")


def plot_coverage_trajectory(run_dirs, out_dir):
    """Plot preference coverage score trajectory across iterations per persona."""
    all_records = []

    for run_dir in run_dirs:
        for entry in sorted(os.listdir(run_dir)):
            persona_dir = os.path.join(run_dir, entry)
            if not os.path.isdir(persona_dir) or entry.startswith("."):
                continue

            cov = load_json(os.path.join(persona_dir, "project_data", "preference_coverage.json"))
            if not cov:
                continue

            items = cov if isinstance(cov, list) else [cov]
            for item in items:
                if isinstance(item, dict) and "coverage_score" in item:
                    all_records.append({
                        "persona": entry,
                        "iteration": item.get("iteration", 1),
                        "coverage_score": item["coverage_score"],
                        "covered": item.get("covered_count", 0),
                        "partial": item.get("partially_covered_count", 0),
                        "not_covered": item.get("not_covered_count", 0),
                        "total": item.get("total_preferences", 0),
                    })

    if not all_records:
        print("  fig07_coverage: No preference coverage data found")
        return

    df = pd.DataFrame(all_records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Coverage score trajectory per persona
    ax = axes[0]
    personas = sorted(df["persona"].unique())
    for i, persona in enumerate(personas):
        pdata = df[df["persona"] == persona].sort_values("iteration")
        ax.plot(pdata["iteration"], pdata["coverage_score"],
                marker="o", color=PALETTE[i % len(PALETTE)],
                linewidth=2, markersize=8, label=persona.replace("_", " ").title())

    # Mean line
    if len(personas) > 1:
        mean_by_iter = df.groupby("iteration")["coverage_score"].mean()
        ax.plot(mean_by_iter.index, mean_by_iter.values,
                marker="s", color="black", linewidth=2.5, markersize=8,
                linestyle="--", label="Mean", zorder=10)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Preference Coverage Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Preference Coverage Over Iterations", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    # Force integer x ticks
    max_iter = int(df["iteration"].max())
    ax.set_xticks(range(1, max_iter + 1))

    # Panel (b): Stacked bar of covered/partial/not_covered per iteration (averaged)
    ax2 = axes[1]
    iter_groups = df.groupby("iteration").agg({
        "covered": "mean", "partial": "mean", "not_covered": "mean"
    }).reset_index()

    x = np.arange(len(iter_groups))
    width = 0.5

    ax2.bar(x, iter_groups["covered"], width, label="Covered",
            color=PALETTE[0], edgecolor="white", linewidth=1)
    ax2.bar(x, iter_groups["partial"], width, bottom=iter_groups["covered"],
            label="Partially Covered", color=PALETTE[2], edgecolor="white", linewidth=1)
    ax2.bar(x, iter_groups["not_covered"], width,
            bottom=iter_groups["covered"] + iter_groups["partial"],
            label="Not Covered", color=PALETTE[3] if len(PALETTE) > 3 else "#e74c3c",
            edgecolor="white", linewidth=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Iter {int(i)}" for i in iter_groups["iteration"]])
    ax2.set_ylabel("Avg Number of Preferences")
    ax2.set_title("(b) Coverage Breakdown by Iteration", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig07_coverage.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig07_coverage.png", bbox_inches="tight")
    plt.close(fig)

    # Summary stats
    first_iter = df[df["iteration"] == df["iteration"].min()]["coverage_score"].mean()
    last_iter = df[df["iteration"] == df["iteration"].max()]["coverage_score"].mean()
    print(f"  fig07_coverage: {len(personas)} personas, "
          f"coverage {first_iter:.0%} → {last_iter:.0%}")


def plot_importance_vs_ground_truth(run_dirs, out_dir):
    """Plot importance ranking vs ground truth: do stated prefs rank higher than hidden, and hallucinated lowest?"""
    all_records = []

    for run_dir in run_dirs:
        for entry in sorted(os.listdir(run_dir)):
            persona_dir = os.path.join(run_dir, entry)
            if not os.path.isdir(persona_dir) or entry.startswith("."):
                continue

            clf = load_json(os.path.join(persona_dir, "project_data",
                                         "criteria_classification_feedback.json"))
            if not clf:
                continue

            items = clf if isinstance(clf, list) else [clf]
            # Take the latest iteration per persona
            latest = None
            for item in items:
                if isinstance(item, dict) and "classifications" in item:
                    latest = item
            if not latest:
                continue

            classifications = latest.get("classifications", {})
            importance_ranking = latest.get("importance_ranking", [])
            if isinstance(classifications, str):
                classifications = json.loads(classifications)
            if isinstance(importance_ranking, str):
                importance_ranking = json.loads(importance_ranking)

            if not classifications or not importance_ranking:
                continue

            n_criteria = len(importance_ranking)
            for crit_name, origin in classifications.items():
                try:
                    rank = importance_ranking.index(crit_name) + 1
                except ValueError:
                    continue
                all_records.append({
                    "persona": entry,
                    "criterion": crit_name,
                    "origin": origin,
                    "importance_rank": rank,
                    "n_criteria": n_criteria,
                    "normalized_rank": rank / n_criteria,
                })

    if not all_records:
        print("  fig09_importance_gt: No importance ranking + classification data found")
        return

    df = pd.DataFrame(all_records)
    origin_order = ["stated", "real", "hallucinated"]
    origin_labels = ["Stated\n(core prefs)", "Inferred\n(hidden prefs)", "Hallucinated"]
    origin_colors = [PALETTE[2], PALETTE[0], PALETTE[3] if len(PALETTE) > 3 else "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.4)
    N = df["persona"].nunique()

    # Panel (a): Normalized rank distribution by origin (all three categories)
    ax = axes[0]
    jitter_rng = np.random.default_rng(45)

    present_origins = [o for o in origin_order if o in df["origin"].values]
    present_labels = [origin_labels[origin_order.index(o)] for o in present_origins]
    present_colors = [origin_colors[origin_order.index(o)] for o in present_origins]

    for i, (origin, label, color) in enumerate(zip(present_origins, present_labels, present_colors)):
        subset = df[df["origin"] == origin]
        if not subset.empty:
            jitter = jitter_rng.uniform(-0.15, 0.15, size=len(subset))
            ax.scatter(np.full(len(subset), i) + jitter, subset["normalized_rank"],
                       s=90, color=color, edgecolors="white", linewidth=1.2,
                       zorder=3, alpha=0.8)
            mean_val = subset["normalized_rank"].mean()
            ax.plot([i - 0.22, i + 0.22], [mean_val, mean_val],
                    color=color, linewidth=3, zorder=4)
            ax.text(i + 0.28, mean_val, f"M={mean_val:.2f}",
                    va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(present_labels)))
    ax.set_xticklabels(present_labels)
    ax.set_ylabel("Normalized Importance Rank\n(0 = most important, 1 = least)")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.5, color="lightgray", linestyle="--", linewidth=0.8, zorder=1)

    ax.annotate("More\nimportant",
                xy=(-0.18, 0.03), xycoords="axes fraction",
                xytext=(-0.18, 0.28), textcoords="axes fraction",
                fontsize=7, color="gray", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    ax.text(0.97, 0.97, f"N = {N} users",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
    ax.set_title("(a) Importance Rank by Origin", fontweight="bold", fontsize=11)

    # Panel (b): Mean normalized rank bar chart with error bars
    ax2 = axes[1]
    means = []
    stds = []
    counts = []
    for origin in present_origins:
        subset = df[df["origin"] == origin]["normalized_rank"]
        means.append(subset.mean() if len(subset) > 0 else 0)
        stds.append(subset.std() if len(subset) > 1 else 0)
        counts.append(len(subset))

    x_pos = np.arange(len(present_origins))
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5,
                   color=present_colors, width=0.5,
                   edgecolor="white", linewidth=1.5, alpha=0.7)
    for i, (bar, m, n) in enumerate(zip(bars, means, counts)):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + stds[i] + 0.04,
                 f"{m:.2f}\n(n={n})", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(present_labels)
    ax2.set_ylabel("Mean Normalized Rank (lower = more important)")
    ax2.set_ylim(0, 1.3)
    ax2.axhline(0.5, color="lightgray", linestyle="--", linewidth=0.8, zorder=1)
    ax2.set_title("(b) Mean Rank by Origin (with SD)", fontweight="bold", fontsize=11)

    fig.suptitle("Importance Ranking vs Ground Truth Preferences",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig09_importance_gt.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig09_importance_gt.png", bbox_inches="tight")
    plt.close(fig)

    # Print summary
    for origin in present_origins:
        subset = df[df["origin"] == origin]
        m = subset["normalized_rank"].mean()
        print(f"  fig09_importance_gt: {origin} mean normalized rank = {m:.2f} (n={len(subset)})")


def plot_precision_trajectory(run_dirs, out_dir):
    """Plot rubric precision (% non-hallucinated) trajectory across iterations."""
    all_records = []

    for run_dir in run_dirs:
        for entry in sorted(os.listdir(run_dir)):
            persona_dir = os.path.join(run_dir, entry)
            if not os.path.isdir(persona_dir) or entry.startswith("."):
                continue

            clf = load_json(os.path.join(persona_dir, "project_data",
                                         "criteria_classification_feedback.json"))
            if not clf:
                continue

            items = clf if isinstance(clf, list) else [clf]
            for item in items:
                if isinstance(item, dict) and "precision" in item:
                    all_records.append({
                        "persona": entry,
                        "iteration": item.get("iteration", 1),
                        "precision": item["precision"],
                        "n_stated": item.get("n_stated", 0),
                        "n_real": item.get("n_real", 0),
                        "n_hallucinated": item.get("n_hallucinated", 0),
                        "n_criteria": item.get("n_criteria", 0),
                    })

    if not all_records:
        print("  fig08_precision: No classification trajectory data found")
        return

    df = pd.DataFrame(all_records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Precision trajectory per persona
    ax = axes[0]
    personas = sorted(df["persona"].unique())
    for i, persona in enumerate(personas):
        pdata = df[df["persona"] == persona].sort_values("iteration")
        ax.plot(pdata["iteration"], pdata["precision"],
                marker="o", color=PALETTE[i % len(PALETTE)],
                linewidth=2, markersize=8, label=persona.replace("_", " ").title())

    # Mean line
    if len(personas) > 1:
        mean_by_iter = df.groupby("iteration")["precision"].mean()
        ax.plot(mean_by_iter.index, mean_by_iter.values,
                marker="s", color="black", linewidth=2.5, markersize=8,
                linestyle="--", label="Mean", zorder=10)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Precision (% non-hallucinated)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Rubric Precision Over Iterations", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    max_iter = int(df["iteration"].max())
    ax.set_xticks(range(1, max_iter + 1))

    # Panel (b): Stacked bar of stated/real/hallucinated per iteration (averaged)
    ax2 = axes[1]
    iter_groups = df.groupby("iteration").agg({
        "n_stated": "mean", "n_real": "mean", "n_hallucinated": "mean"
    }).reset_index()

    x = np.arange(len(iter_groups))
    width = 0.5

    ax2.bar(x, iter_groups["n_stated"], width, label="Stated",
            color=PALETTE[0], edgecolor="white", linewidth=1)
    ax2.bar(x, iter_groups["n_real"], width, bottom=iter_groups["n_stated"],
            label="Real (unstated)", color=PALETTE[1], edgecolor="white", linewidth=1)
    ax2.bar(x, iter_groups["n_hallucinated"], width,
            bottom=iter_groups["n_stated"] + iter_groups["n_real"],
            label="Hallucinated", color=PALETTE[3] if len(PALETTE) > 3 else "#e74c3c",
            edgecolor="white", linewidth=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Iter {int(i)}" for i in iter_groups["iteration"]])
    ax2.set_ylabel("Avg Number of Criteria")
    ax2.set_title("(b) Criteria Classification by Iteration", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig08_precision.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig08_precision.png", bbox_inches="tight")
    plt.close(fig)

    first_iter = df[df["iteration"] == df["iteration"].min()]["precision"].mean()
    last_iter = df[df["iteration"] == df["iteration"].max()]["precision"].mean()
    print(f"  fig08_precision: {len(personas)} personas, "
          f"precision {first_iter:.0%} → {last_iter:.0%}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from simulation results")
    parser.add_argument("paths", nargs="+",
                        help="Simulation output directories (or glob patterns)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for figures (default: eval_output/figures/)")
    parser.add_argument("--tables", action="store_true",
                        help="Also save CSV tables alongside figures")
    args = parser.parse_args()

    # Resolve glob patterns
    run_dirs = []
    for p in args.paths:
        expanded = sorted(glob.glob(p))
        if expanded:
            run_dirs.extend(d for d in expanded if os.path.isdir(d))
        elif os.path.isdir(p):
            run_dirs.append(p)

    if not run_dirs:
        print("No simulation directories found.", file=sys.stderr)
        sys.exit(1)

    # Output directory
    out_dir = Path(args.out) if args.out else Path("eval_output") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tables:
        tables_dir = out_dir.parent / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {len(run_dirs)} run(s)...")
    data = load_simulation_data(run_dirs)

    users = data["users"]
    df_clf = data["df_classifications"]
    df_align = data["df_alignment"]

    print(f"  Users: {len(users)}")
    print(f"  Classification records: {len(df_clf)}")
    print(f"  Alignment records: {len(df_align)}")

    # Build aggregate DataFrames
    df_rq1, df_rq1_criteria = build_rq1_data(users, df_clf, df_align)
    df_rq2_rank_pairs, df_rq2_draft_pref = build_rq2_data(users, df_align)
    df_rq3_criteria, df_rq3_versions = build_rq3_data(users, df_align)

    # Save tables if requested
    if args.tables:
        if not df_rq1.empty:
            df_rq1.to_csv(tables_dir / "rq1_per_user.csv", index=False)
        if not df_rq1_criteria.empty:
            df_rq1_criteria.to_csv(tables_dir / "rq1_criteria_importance.csv", index=False)
        if not df_rq2_rank_pairs.empty:
            df_rq2_rank_pairs.to_csv(tables_dir / "rq2_rank_pairs.csv", index=False)
        if not df_rq3_criteria.empty:
            df_rq3_criteria.to_csv(tables_dir / "rq3_criteria_trajectory.csv", index=False)
        if not df_rq3_versions.empty:
            df_rq3_versions.to_csv(tables_dir / "rq3_version_summary.csv", index=False)
        print(f"\nTables saved to {tables_dir}/")

    # Generate plots
    print(f"\nGenerating figures in {out_dir}/...")
    plot_rq1(df_rq1, df_rq1_criteria, out_dir)
    plot_rq2(df_rq2_rank_pairs, df_rq2_draft_pref, out_dir)
    plot_rq3(df_rq3_criteria, df_rq3_versions, out_dir)
    plot_termination(run_dirs, out_dir)
    plot_log_changes(run_dirs, out_dir)
    plot_surveys(run_dirs, out_dir)
    plot_coverage_trajectory(run_dirs, out_dir)
    plot_precision_trajectory(run_dirs, out_dir)
    plot_importance_vs_ground_truth(run_dirs, out_dir)

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
