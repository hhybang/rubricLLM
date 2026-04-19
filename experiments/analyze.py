#!/usr/bin/env python3
"""Aggregate results/raw into analysis CSVs and summary_stats.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from scipy import stats

from rubric_utils import canonical_criterion_name

RAW = _ROOT / "results" / "raw"
AN = _ROOT / "results" / "analysis"


def _strip_meta_key(name: str) -> str:
    if " (priority" in name:
        return name.split(" (priority", 1)[0].strip()
    return name.strip()


def _final_rubric_score(rec: dict) -> float | None:
    """Last-round rubric score on a 0–1 scale."""
    cond = rec.get("condition")
    if cond == "E":
        cands = rec.get("candidates") or []
        idx = int(rec.get("selected_index") or 0)
        if 0 <= idx < len(cands):
            g = cands[idx].get("grading") or {}
            return g.get("mean_overall")
        return None
    rounds = rec.get("rounds") or []
    if not rounds:
        return None
    g = rounds[-1].get("grading") or {}
    mo = g.get("mean_overall")
    if mo is not None:
        return float(mo)
    m110 = g.get("mean_score_1_10")
    if m110 is not None:
        try:
            return float(m110) / 10.0
        except (TypeError, ValueError):
            return None
    return None


def _gt_alignment(rec: dict) -> float | None:
    gt = rec.get("ground_truth_eval") or {}
    v = gt.get("overall_alignment")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _dealbreaker_count(rec: dict) -> int:
    gt = rec.get("ground_truth_eval") or {}
    dv = gt.get("dealbreaker_violations") or []
    return len(dv) if isinstance(dv, list) else 0


def _satisfaction_rate(rec: dict) -> float | None:
    gt = rec.get("ground_truth_eval") or {}
    hps = gt.get("hidden_pref_satisfaction") or []
    if not isinstance(hps, list) or not hps:
        return None
    yes = sum(1 for x in hps if (x or {}).get("satisfied") == "yes")
    partial = sum(1 for x in hps if (x or {}).get("satisfied") == "partially")
    return (yes + 0.5 * partial) / len(hps)


def load_all_records() -> list[dict]:
    out = []
    for path in sorted(RAW.glob("**/condition_*.json")):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def build_improvement_curves(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for rec in rows:
        persona = rec.get("persona")
        task_index = rec.get("task_index")
        cond = rec.get("condition")
        for rnd in rec.get("rounds") or []:
            ri = rnd.get("round")
            g = rnd.get("grading") or {}
            mean_scores = g.get("mean_scores") or {}
            std_scores = g.get("std_scores") or {}
            mo = g.get("mean_overall")
            for cname, ms in mean_scores.items():
                cn = _strip_meta_key(str(cname))
                recs.append(
                    {
                        "persona": persona,
                        "task_index": task_index,
                        "condition": cond,
                        "round": ri,
                        "criterion_name": cn,
                        "mean_score": ms,
                        "std_score": std_scores.get(cname) or std_scores.get(cn),
                        "overall_weighted_mean": mo,
                    }
                )
    return pd.DataFrame(recs)


def build_condition_comparison(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for rec in rows:
        recs.append(
            {
                "persona": rec.get("persona"),
                "task_index": rec.get("task_index"),
                "condition": rec.get("condition"),
                "final_overall_rubric_score": _final_rubric_score(rec),
                "final_ground_truth_alignment": _gt_alignment(rec),
                "num_dealbreaker_violations": _dealbreaker_count(rec),
                "hidden_pref_satisfaction_rate": _satisfaction_rate(rec),
            }
        )
    return pd.DataFrame(recs)


def build_criterion_level(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for rec in rows:
        cond = rec.get("condition")
        if cond == "E":
            continue
        rounds = rec.get("rounds") or []
        if len(rounds) < 2:
            continue
        g1 = (rounds[0].get("grading") or {}).get("mean_scores") or {}
        g5 = (rounds[-1].get("grading") or {}).get("mean_scores") or {}
        rub = (rec.get("rubric_used") or {}).get("rubric") or []
        pmap = {c.get("name"): c for c in rub}
        for name in set(g1) | set(g5):
            skey = name
            s1 = float(g1.get(skey, g1.get(_strip_meta_key(str(skey)), 0)) or 0)
            s5 = float(g5.get(skey, g5.get(_strip_meta_key(str(skey)), 0)) or 0)
            pr = (pmap.get(_strip_meta_key(str(name))) or pmap.get(name) or {}).get("priority")
            cat = (pmap.get(_strip_meta_key(str(name))) or pmap.get(name) or {}).get("category")
            recs.append(
                {
                    "persona": rec.get("persona"),
                    "task_index": rec.get("task_index"),
                    "condition": cond,
                    "criterion_name": _strip_meta_key(str(name)),
                    "round_1_score": s1,
                    "round_5_score": s5,
                    "improvement": s5 - s1,
                    "criterion_priority": pr,
                    "criterion_category": cat,
                }
            )
    return pd.DataFrame(recs)


def build_correction_analysis(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for rec in rows:
        if rec.get("condition") != "D":
            continue
        corr = rec.get("corrections_applied") or {}
        names = {c.get("criterion_name") for c in (corr.get("corrections") or [])}
        rounds = rec.get("rounds") or []
        pre = next((x for x in rounds if x.get("round") == 2), None)
        post = next((x for x in rounds if x.get("round") == 5), None)
        g_pre = (pre or {}).get("grading") or {}
        g_post = (post or {}).get("grading") or {}
        m_pre = g_pre.get("mean_overall")
        m_post = g_post.get("mean_overall")
        for cn in names:
            recs.append(
                {
                    "persona": rec.get("persona"),
                    "task_index": rec.get("task_index"),
                    "criterion_name": cn,
                    "was_corrected": True,
                    "pre_correction_score_mean": m_pre,
                    "post_correction_score_mean": m_post,
                    "pre_correction_ground_truth": None,
                    "post_correction_ground_truth": None,
                }
            )
    return pd.DataFrame(recs)


def build_grader_variance_by_criterion(rows: list[dict]) -> pd.DataFrame:
    """Per-criterion stats on std_scores across all graded rounds (finds noisy criteria)."""
    stds: dict[str, list[float]] = defaultdict(list)
    for rec in rows:
        rub = rec.get("rubric_used") or {}
        for rnd in rec.get("rounds") or []:
            g = rnd.get("grading") or {}
            for raw_name, sd in (g.get("std_scores") or {}).items():
                try:
                    f = float(sd)
                except (TypeError, ValueError):
                    continue
                cn = canonical_criterion_name(str(raw_name), rub) or _strip_meta_key(str(raw_name))
                stds[str(cn)].append(f)
    recs = []
    for name, vals in stds.items():
        if not vals:
            continue
        recs.append(
            {
                "criterion_name": name,
                "max_std": max(vals),
                "mean_std": sum(vals) / len(vals),
                "n_grading_events": len(vals),
            }
        )
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values("max_std", ascending=False)
    return df


def pairwise_wilcoxon(df: pd.DataFrame, col: str, a: str, b: str) -> dict:
    sub = df.pivot_table(index=["persona", "task_index"], columns="condition", values=col, aggfunc="first")
    if a not in sub.columns or b not in sub.columns:
        return {"error": "missing condition column", "a": a, "b": b}
    paired = sub[[a, b]].dropna()
    x = paired[a].values
    y = paired[b].values
    if len(x) < 3:
        return {"n": len(x), "note": "too few pairs", "diff_mean": float((x - y).mean()) if len(x) else None}
    try:
        stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
    except Exception as e:
        return {"error": str(e), "n": len(x)}
    return {
        "n": len(x),
        "median_diff": float(pd.Series(x - y).median()),
        "mean_diff": float((x - y).mean()),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p),
        "significant_0.05": bool(p < 0.05),
    }


def grading_consistency(rows: list[dict]) -> dict:
    stds = []
    for rec in rows:
        for rnd in rec.get("rounds") or []:
            g = rnd.get("grading") or {}
            for _, v in (g.get("std_scores") or {}).items():
                stds.append(float(v))
    if not stds:
        return {"mean_std_across_runs": None, "max_std_criterion": None}
    return {
        "mean_std_across_runs": float(sum(stds) / len(stds)),
        "max_std_criterion": float(max(stds)),
    }


def main() -> None:
    AN.mkdir(parents=True, exist_ok=True)
    rows = load_all_records()
    if not rows:
        print("No JSON files in results/raw — nothing to analyze.", file=sys.stderr)
        sys.exit(0)

    cc = build_condition_comparison(rows)
    cc.to_csv(AN / "condition_comparison.csv", index=False)

    ic = build_improvement_curves(rows)
    ic.to_csv(AN / "improvement_curves.csv", index=False)

    cl = build_criterion_level(rows)
    cl.to_csv(AN / "criterion_level_analysis.csv", index=False)

    ca = build_correction_analysis(rows)
    ca.to_csv(AN / "correction_analysis.csv", index=False)

    gv = build_grader_variance_by_criterion(rows)
    gv.to_csv(AN / "grader_variance_by_criterion.csv", index=False)

    cond_means = {}
    for c in ["A", "B", "C", "D", "E"]:
        sub = cc[cc["condition"] == c]
        if sub.empty:
            continue
        cond_means[c] = {
            "rubric_score_mean": float(sub["final_overall_rubric_score"].mean()),
            "ground_truth_mean": float(sub["final_ground_truth_alignment"].mean()),
            "satisfaction_rate_mean": float(sub["hidden_pref_satisfaction_rate"].mean()),
        }

    pairwise = {
        "C_vs_A_ground_truth": pairwise_wilcoxon(cc, "final_ground_truth_alignment", "C", "A"),
        "D_vs_C_ground_truth": pairwise_wilcoxon(cc, "final_ground_truth_alignment", "D", "C"),
        "D_vs_A_ground_truth": pairwise_wilcoxon(cc, "final_ground_truth_alignment", "D", "A"),
        "C_vs_A_rubric": pairwise_wilcoxon(cc, "final_overall_rubric_score", "C", "A"),
        "C_vs_B_rubric": pairwise_wilcoxon(cc, "final_overall_rubric_score", "C", "B"),
        "E_vs_C_rubric": pairwise_wilcoxon(cc, "final_overall_rubric_score", "E", "C"),
    }

    top_volatile = []
    if not gv.empty:
        top_volatile = gv.head(15).to_dict(orient="records")

    summary = {
        "condition_means": cond_means,
        "pairwise_tests": pairwise,
        "grading_consistency": grading_consistency(rows),
        "top_volatile_criteria_by_max_std": top_volatile,
    }
    with open(AN / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {AN / 'condition_comparison.csv'}", file=sys.stderr)
    print(f"Wrote {AN / 'grader_variance_by_criterion.csv'}", file=sys.stderr)
    print(f"Wrote {AN / 'summary_stats.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
