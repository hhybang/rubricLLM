#!/usr/bin/env python3
"""
Analyze simulation results across personas and runs.

Usage:
    python analyze_results.py eval_output/sim_20260302_185324
    python analyze_results.py eval_output/sim_*              # compare multiple runs
    python analyze_results.py eval_output/sim_* --format md  # markdown output
"""

import argparse
import glob
import json
import os
import sys


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_json(path):
    """Load a JSON file, return None if missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_run(run_dir):
    """Load all data for a single simulation run."""
    summary = load_json(os.path.join(run_dir, "summary.json"))
    if not summary:
        return None

    run = {
        "dir": run_dir,
        "name": os.path.basename(run_dir),
        "config": summary.get("config", {}),
        "completed_at": summary.get("completed_at"),
        "personas": {},
    }

    # Find persona directories
    for entry in sorted(os.listdir(run_dir)):
        persona_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(persona_dir) or entry.startswith("."):
            continue

        pd = os.path.join(persona_dir, "project_data")
        persona_data = {
            "name": entry,
            "project": load_json(os.path.join(persona_dir, "project.json")),
            "coldstart": load_json(os.path.join(persona_dir, "coldstart.json")),
            "conversations": load_json(os.path.join(persona_dir, "conversations.json")),
            "rubric_history": load_json(os.path.join(persona_dir, "rubric_history.json")),
            "gold_draft": load_json(os.path.join(pd, "gold_draft.json")),
            "termination_log": load_json(os.path.join(pd, "termination_log.json")),
            "classification": load_json(os.path.join(pd, "criteria_classification_feedback.json")),
            "decision_points": load_json(os.path.join(pd, "decision_point_feedback.json")),
            "alignment_diagnostic": load_json(os.path.join(pd, "alignment_diagnostic.json")),
            "log_changes": load_json(os.path.join(pd, "log_changes.json")),
            "survey_task_a": load_json(os.path.join(pd, "survey_task_a.json")),
            "survey_task_b": load_json(os.path.join(pd, "survey_task_b.json")),
            "survey_final_review": load_json(os.path.join(pd, "survey_final_review.json")),
            "preference_coverage": load_json(os.path.join(pd, "preference_coverage.json")),
        }
        run["personas"][entry] = persona_data

    # Attach per-persona results from summary
    for r in summary.get("results", []):
        slug = r.get("persona", "").replace(" ", "_").lower()
        if slug in run["personas"]:
            run["personas"][slug]["result"] = r

    return run


# ═════════════════════════════════════════════════════════════════════════════
# Per-Persona Analysis
# ═════════════════════════════════════════════════════════════════════════════

def analyze_termination(persona_data):
    """Extract termination metrics for a persona."""
    tlog = persona_data.get("termination_log")
    if not tlog:
        return None

    # termination_log is a list (appended per run); take latest entry
    entry = tlog[-1] if isinstance(tlog, list) else tlog
    iterations = entry.get("iterations", [])

    results = []
    for it in iterations:
        metrics = it.get("turn_metrics", [])
        draft_turns = [m for m in metrics if m.get("has_draft")]

        sat_scores = [m["satisfaction"]["score"] for m in draft_turns if "satisfaction" in m]
        qual_scores = [m["quality"]["score"] for m in draft_turns if "quality" in m]

        results.append({
            "iteration": it["iteration"],
            "drafts_produced": it.get("drafts_produced", len(draft_turns)),
            "max_drafts": it.get("max_drafts", it.get("max_turns", "?")),
            "total_turns": it.get("total_turns", it.get("turns_used", "?")),
            "reason": it["termination_reason"],
            "satisfaction_scores": sat_scores,
            "quality_scores": qual_scores,
            "final_satisfaction": sat_scores[-1] if sat_scores else None,
            "final_quality": qual_scores[-1] if qual_scores else None,
            "satisfaction_trend": _trend(sat_scores),
            "quality_trend": _trend(qual_scores),
        })
    return results


def analyze_rubric_quality(persona_data):
    """Extract rubric quality metrics."""
    clf = persona_data.get("classification")
    if not clf:
        return None

    # classification is a list; take latest
    entry = clf[-1] if isinstance(clf, list) else clf
    return {
        "n_criteria": entry.get("n_criteria", 0),
        "n_stated": entry.get("n_stated", 0),
        "n_real": entry.get("n_real", 0),
        "n_hallucinated": entry.get("n_hallucinated", 0),
        "precision": entry.get("precision", 0),
        "classifications": entry.get("classifications", {}),
        "importance_ranking": entry.get("importance_ranking", []),
    }


def analyze_rubric_trajectory(persona_data):
    """Extract rubric quality metrics across ALL iterations (not just latest)."""
    clf = persona_data.get("classification")
    if not clf:
        return None

    items = clf if isinstance(clf, list) else [clf]
    trajectory = []
    for entry in items:
        if isinstance(entry, dict):
            trajectory.append({
                "iteration": entry.get("iteration"),
                "n_criteria": entry.get("n_criteria", 0),
                "n_stated": entry.get("n_stated", 0),
                "n_real": entry.get("n_real", 0),
                "n_hallucinated": entry.get("n_hallucinated", 0),
                "precision": entry.get("precision", 0),
            })

    return trajectory if trajectory else None


def analyze_alignment(persona_data):
    """Extract alignment diagnostic results."""
    diag = persona_data.get("alignment_diagnostic")
    if not diag:
        return None

    # alignment_diagnostic is a list; take latest
    entry = diag[-1] if isinstance(diag, list) else diag

    user_ranking = entry.get("user_ranking", [])
    rubric_won = user_ranking[0] == "rubric" if user_ranking else False

    criteria_analysis = entry.get("criteria_analysis", [])
    classifications = {}
    for c in criteria_analysis:
        cls = c.get("classification", "unknown")
        classifications[cls] = classifications.get(cls, 0) + 1

    avg_rubric = _avg([c.get("rubric_score", 0) for c in criteria_analysis])
    avg_generic = _avg([c.get("generic_score", 0) for c in criteria_analysis])
    avg_preference = _avg([c.get("preference_score", 0) for c in criteria_analysis if "preference_score" in c])
    avg_gap = _avg([c.get("gap", 0) for c in criteria_analysis])

    return {
        "user_ranking": user_ranking,
        "rubric_won": rubric_won,
        "rubric_rank": user_ranking.index("rubric") + 1 if "rubric" in user_ranking else None,
        "avg_rubric_score": avg_rubric,
        "avg_generic_score": avg_generic,
        "avg_preference_score": avg_preference,
        "avg_gap": avg_gap,
        "criteria_classifications": classifications,
        "user_reason": entry.get("user_reason", "")[:200],
    }


def analyze_log_changes(persona_data):
    """Extract rubric edit (Log Changes) metrics."""
    lc = persona_data.get("log_changes")
    if not lc:
        return None

    entry = lc[-1] if isinstance(lc, list) else lc
    edits = entry.get("edit_classification", {})

    return {
        "n_added": len(edits.get("added", [])),
        "n_removed": len(edits.get("removed", [])),
        "n_reweighted": len(edits.get("reweighted", [])),
        "n_reworded": len(edits.get("reworded", [])),
        "n_dimensions_changed": len(edits.get("dimensions_changed", [])),
        "total_edits": sum(len(v) for v in edits.values() if isinstance(v, list)),
        "draft_regenerated": entry.get("draft_regenerated", False),
        "reasoning": entry.get("reasoning", "")[:200],
    }


def analyze_surveys(persona_data):
    """Extract survey response metrics."""
    task_a = persona_data.get("survey_task_a")
    task_b = persona_data.get("survey_task_b")
    final = persona_data.get("survey_final_review")

    result = {"task_a": None, "task_b": [], "final": None}

    # Task A
    if task_a:
        entry = task_a[-1] if isinstance(task_a, list) else task_a
        result["task_a"] = {
            "understanding": entry.get("q1"),
            "effort": entry.get("q2"),
            "issues": entry.get("q3", "")[:200],
        }

    # Task B (multiple iterations)
    if task_b:
        items = task_b if isinstance(task_b, list) else [task_b]
        for entry in items:
            if isinstance(entry, dict):
                result["task_b"].append({
                    "iteration": entry.get("iteration"),
                    "understanding": entry.get("q1"),
                    "effort": entry.get("q2"),
                    "comparison": entry.get("q4"),
                })

    # Final review
    if final:
        entry = final[-1] if isinstance(final, list) else final
        ratings = entry.get("criteria_ratings", {})
        n_accurate = sum(1 for r in ratings.values() if isinstance(r, dict) and r.get("accuracy") == "Accurate")
        n_partial = sum(1 for r in ratings.values() if isinstance(r, dict) and r.get("accuracy") == "Partially right")
        n_inaccurate = sum(1 for r in ratings.values() if isinstance(r, dict) and r.get("accuracy") == "Inaccurate")
        result["final"] = {
            "n_criteria_rated": len(ratings),
            "n_accurate": n_accurate,
            "n_partial": n_partial,
            "n_inaccurate": n_inaccurate,
            "accuracy_rate": n_accurate / len(ratings) if ratings else 0,
            "unexpected_insights": entry.get("q2", "")[:200],
        }

    return result if any([result["task_a"], result["task_b"], result["final"]]) else None


def analyze_coverage(persona_data):
    """Extract preference coverage trajectory across iterations."""
    cov = persona_data.get("preference_coverage")
    if not cov:
        return None

    items = cov if isinstance(cov, list) else [cov]
    trajectory = []
    for entry in items:
        if isinstance(entry, dict) and "coverage_score" in entry:
            trajectory.append({
                "iteration": entry.get("iteration"),
                "coverage_score": entry.get("coverage_score"),
                "total_preferences": entry.get("total_preferences"),
                "covered": entry.get("covered_count", 0),
                "partial": entry.get("partially_covered_count", 0),
                "not_covered": entry.get("not_covered_count", 0),
            })

    return trajectory if trajectory else None


# ═════════════════════════════════════════════════════════════════════════════
# Formatting
# ═════════════════════════════════════════════════════════════════════════════

def format_persona_report(name, persona_data, fmt="text"):
    """Generate a report for a single persona."""
    lines = []
    project = persona_data.get("project", {})
    persona_info = project.get("persona", {})
    display_name = persona_info.get("name", name)
    role = persona_info.get("role", "")
    result = persona_data.get("result", {})

    if fmt == "md":
        lines.append(f"### {display_name}")
        lines.append(f"**{role}** | {result.get('num_messages', '?')} messages | "
                      f"{result.get('num_rubric_versions', '?')} rubric versions")
    else:
        lines.append(f"  {display_name} ({role})")
        lines.append(f"    Messages: {result.get('num_messages', '?')} | "
                      f"Rubric versions: {result.get('num_rubric_versions', '?')}")

    # Termination
    term = analyze_termination(persona_data)
    if term:
        for it in term:
            reason_icon = "+" if it["reason"] in ("goal_reached", "user_accepted") else "x"
            sat = f"{it['final_satisfaction']:.2f}" if it['final_satisfaction'] is not None else "n/a"
            qual = f"{it['final_quality']:.2f}" if it['final_quality'] is not None else "n/a"

            if fmt == "md":
                lines.append(f"  - **Iteration {it['iteration']}**: "
                             f"`{it['reason']}` — {it['drafts_produced']}/{it['max_drafts']} drafts "
                             f"({it['total_turns']} turns) | "
                             f"satisfaction={sat} quality={qual} | "
                             f"trend: sat {it['satisfaction_trend']} qual {it['quality_trend']}")
            else:
                lines.append(f"    Iter {it['iteration']}: [{reason_icon}] {it['reason']} "
                             f"({it['drafts_produced']}/{it['max_drafts']} drafts, "
                             f"{it['total_turns']} turns) "
                             f"sat={sat} qual={qual} "
                             f"trend: sat {it['satisfaction_trend']} qual {it['quality_trend']}")

    # Rubric quality
    rubric_q = analyze_rubric_quality(persona_data)
    if rubric_q:
        p = rubric_q["precision"]
        if fmt == "md":
            lines.append(f"  - **Rubric precision**: {p:.0%} "
                         f"({rubric_q['n_stated']}S + {rubric_q['n_real']}R + "
                         f"{rubric_q['n_hallucinated']}H / {rubric_q['n_criteria']})")
        else:
            lines.append(f"    Rubric: precision={p:.0%} "
                         f"({rubric_q['n_stated']} stated, {rubric_q['n_real']} real, "
                         f"{rubric_q['n_hallucinated']} hallucinated)")

    # Rubric precision trajectory
    rubric_traj = analyze_rubric_trajectory(persona_data)
    if rubric_traj and len(rubric_traj) > 1:
        prec_strs = [f"{t['precision']:.0%}" for t in rubric_traj]
        trajectory_str = " → ".join(prec_strs)
        if fmt == "md":
            lines.append(f"  - **Precision trajectory**: {trajectory_str}")
        else:
            lines.append(f"    Precision trajectory: {trajectory_str}")

    # Alignment diagnostic
    alignment = analyze_alignment(persona_data)
    if alignment:
        rank = alignment["rubric_rank"]
        rank_str = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th") if rank else "?"
        if fmt == "md":
            lines.append(f"  - **Alignment**: rubric ranked **{rank_str}** by user | "
                         f"avg scores: rubric={alignment['avg_rubric_score']:.1f} "
                         f"generic={alignment['avg_generic_score']:.1f} "
                         f"preference={alignment['avg_preference_score']:.1f} | "
                         f"gap={alignment['avg_gap']:.1f}")
            cls = alignment["criteria_classifications"]
            lines.append(f"    Criteria: {cls}")
        else:
            lines.append(f"    Alignment: rubric={rank_str} | "
                         f"scores: R={alignment['avg_rubric_score']:.1f} "
                         f"G={alignment['avg_generic_score']:.1f} "
                         f"P={alignment['avg_preference_score']:.1f} | "
                         f"gap={alignment['avg_gap']:.1f}")
            lines.append(f"    Criteria types: {alignment['criteria_classifications']}")

    # Log Changes
    lc = analyze_log_changes(persona_data)
    if lc:
        edit_parts = []
        if lc["n_added"]: edit_parts.append(f"{lc['n_added']} added")
        if lc["n_removed"]: edit_parts.append(f"{lc['n_removed']} removed")
        if lc["n_reweighted"]: edit_parts.append(f"{lc['n_reweighted']} reweighted")
        if lc["n_reworded"]: edit_parts.append(f"{lc['n_reworded']} reworded")
        if lc["n_dimensions_changed"]: edit_parts.append(f"{lc['n_dimensions_changed']} dims changed")
        edit_str = ", ".join(edit_parts) if edit_parts else "no edits"
        regen = "yes" if lc["draft_regenerated"] else "no"

        if fmt == "md":
            lines.append(f"  - **Log Changes**: {lc['total_edits']} edits ({edit_str}) | draft regenerated: {regen}")
        else:
            lines.append(f"    Log Changes: {lc['total_edits']} edits ({edit_str}) | draft regen: {regen}")

    # Surveys
    surveys = analyze_surveys(persona_data)
    if surveys:
        if surveys["task_a"]:
            a = surveys["task_a"]
            if fmt == "md":
                lines.append(f"  - **Task A Survey**: understanding={a['understanding']}/5, effort={a['effort']}/5")
            else:
                lines.append(f"    Task A Survey: understanding={a['understanding']}/5, effort={a['effort']}/5")

        for b in surveys["task_b"]:
            if fmt == "md":
                lines.append(f"  - **Task B Survey (iter {b['iteration']})**: "
                             f"understanding={b['understanding']}/5, effort={b['effort']}/5, "
                             f"vs previous={b['comparison']}")
            else:
                lines.append(f"    Task B (iter {b['iteration']}): "
                             f"understanding={b['understanding']}/5, effort={b['effort']}/5, "
                             f"vs prev={b['comparison']}")

        if surveys["final"]:
            f = surveys["final"]
            if fmt == "md":
                lines.append(f"  - **Final Review**: {f['n_accurate']}A + {f['n_partial']}P + "
                             f"{f['n_inaccurate']}I = {f['accuracy_rate']:.0%} accuracy")
            else:
                lines.append(f"    Final Review: {f['n_accurate']} accurate, {f['n_partial']} partial, "
                             f"{f['n_inaccurate']} inaccurate (accuracy={f['accuracy_rate']:.0%})")

    # Preference Coverage
    coverage = analyze_coverage(persona_data)
    if coverage:
        scores = [f"{c['coverage_score']:.0%}" for c in coverage]
        trajectory_str = " → ".join(scores)
        if fmt == "md":
            lines.append(f"  - **Preference Coverage**: {trajectory_str}")
        else:
            lines.append(f"    Preference Coverage: {trajectory_str}")

    return "\n".join(lines)


def format_run_report(run, fmt="text"):
    """Generate a full report for a simulation run."""
    lines = []
    config = run["config"]
    n_personas = len(run["personas"])

    if fmt == "md":
        lines.append(f"## Run: `{run['name']}`")
        lines.append(f"Completed: {run['completed_at']}")
        lines.append(f"")
        lines.append(f"| Setting | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Personas | {n_personas} |")
        lines.append(f"| Iterations | {config.get('num_iterations', '?')} |")
        lines.append(f"| Max drafts | {config.get('max_chat_turns', config.get('chat_turns_per_iteration', '?'))} |")
        lines.append(f"| Threshold | {config.get('satisfaction_threshold', 'n/a')} |")
        lines.append(f"| System | {config.get('model_primary', '?')} / {config.get('model_light', '?')} |")
        lines.append(f"| User model | {config.get('user_provider', '?')}/{config.get('user_model', '?')} |")
        lines.append(f"")
    else:
        lines.append(f"{'='*70}")
        lines.append(f"Run: {run['name']}  ({run['completed_at']})")
        lines.append(f"{'='*70}")
        lines.append(f"  Personas: {n_personas} | "
                      f"Iterations: {config.get('num_iterations', '?')} | "
                      f"Max drafts: {config.get('max_chat_turns', config.get('chat_turns_per_iteration', '?'))} | "
                      f"Threshold: {config.get('satisfaction_threshold', 'n/a')}")
        lines.append(f"  Models: system={config.get('model_primary', '?')}/{config.get('model_light', '?')} "
                      f"user={config.get('user_provider', '?')}/{config.get('user_model', '?')}")
        lines.append(f"")

    # Per-persona reports
    for slug, pdata in sorted(run["personas"].items()):
        lines.append(format_persona_report(slug, pdata, fmt))
        lines.append("")

    # Aggregate stats
    all_term = []
    all_precision = []
    all_rubric_rank = []
    all_gaps = []

    for pdata in run["personas"].values():
        term = analyze_termination(pdata)
        if term:
            all_term.extend(term)

        rq = analyze_rubric_quality(pdata)
        if rq:
            all_precision.append(rq["precision"])

        al = analyze_alignment(pdata)
        if al:
            if al["rubric_rank"]:
                all_rubric_rank.append(al["rubric_rank"])
            all_gaps.append(al["avg_gap"])

    if all_term or all_precision:
        if fmt == "md":
            lines.append(f"### Aggregates")
        else:
            lines.append(f"  {'─'*50}")
            lines.append(f"  AGGREGATES")

        if all_term:
            goal_reached = sum(1 for t in all_term if t["reason"] in ("goal_reached", "user_accepted"))
            max_hit = sum(1 for t in all_term if t["reason"] == "max_drafts")
            avg_drafts = _avg([t["drafts_produced"] for t in all_term])
            avg_sat = _avg([t["final_satisfaction"] for t in all_term if t["final_satisfaction"] is not None])
            avg_qual = _avg([t["final_quality"] for t in all_term if t["final_quality"] is not None])

            if fmt == "md":
                lines.append(f"- **Termination**: {goal_reached} user_accepted / {max_hit} max_drafts "
                             f"({len(all_term)} total iterations)")
                lines.append(f"- **Avg drafts**: {avg_drafts:.1f} | "
                             f"**Avg final satisfaction**: {avg_sat:.2f} | "
                             f"**Avg final quality**: {avg_qual:.2f}")
            else:
                lines.append(f"    Termination: {goal_reached} user_accepted, {max_hit} max_drafts "
                             f"({len(all_term)} iterations)")
                lines.append(f"    Avg drafts: {avg_drafts:.1f} | "
                             f"Avg satisfaction: {avg_sat:.2f} | "
                             f"Avg quality: {avg_qual:.2f}")

        if all_precision:
            if fmt == "md":
                lines.append(f"- **Avg rubric precision**: {_avg(all_precision):.0%}")
            else:
                lines.append(f"    Avg rubric precision: {_avg(all_precision):.0%}")

        if all_rubric_rank:
            if fmt == "md":
                lines.append(f"- **Rubric ranked 1st**: {sum(1 for r in all_rubric_rank if r == 1)}/{len(all_rubric_rank)} | "
                             f"Avg rubric-generic gap: {_avg(all_gaps):.1f}")
            else:
                lines.append(f"    Rubric ranked 1st: {sum(1 for r in all_rubric_rank if r == 1)}/{len(all_rubric_rank)} | "
                             f"Avg gap: {_avg(all_gaps):.1f}")

    return "\n".join(lines)


def format_cross_run_comparison(runs, fmt="text"):
    """Compare metrics across multiple runs."""
    lines = []

    if fmt == "md":
        lines.append(f"## Cross-Run Comparison")
        lines.append(f"")
        lines.append(f"| Run | Personas | Goal% | Avg Drafts | Avg Sat | Avg Qual | Precision | Rubric 1st |")
        lines.append(f"|---|---|---|---|---|---|---|---|")
    else:
        lines.append(f"{'='*90}")
        lines.append(f"CROSS-RUN COMPARISON")
        lines.append(f"{'='*90}")
        header = f"  {'Run':<28} {'#P':>3} {'Goal%':>6} {'Drafts':>6} {'Sat':>6} {'Qual':>6} {'Prec':>6} {'R#1':>5}"
        lines.append(header)
        lines.append(f"  {'─'*80}")

    for run in runs:
        all_term = []
        all_precision = []
        all_rubric_rank = []

        for pdata in run["personas"].values():
            term = analyze_termination(pdata)
            if term:
                all_term.extend(term)
            rq = analyze_rubric_quality(pdata)
            if rq:
                all_precision.append(rq["precision"])
            al = analyze_alignment(pdata)
            if al and al["rubric_rank"]:
                all_rubric_rank.append(al["rubric_rank"])

        n_personas = len(run["personas"])
        goal_pct = f"{sum(1 for t in all_term if t['reason'] in ('goal_reached', 'user_accepted')) / len(all_term) * 100:.0f}%" if all_term else "n/a"
        avg_drafts = f"{_avg([t['drafts_produced'] for t in all_term]):.1f}" if all_term else "n/a"
        avg_sat = f"{_avg([t['final_satisfaction'] for t in all_term if t['final_satisfaction'] is not None]):.2f}" if all_term else "n/a"
        avg_qual = f"{_avg([t['final_quality'] for t in all_term if t['final_quality'] is not None]):.2f}" if all_term else "n/a"
        precision = f"{_avg(all_precision):.0%}" if all_precision else "n/a"
        rubric_1st = f"{sum(1 for r in all_rubric_rank if r == 1)}/{len(all_rubric_rank)}" if all_rubric_rank else "n/a"

        if fmt == "md":
            lines.append(f"| `{run['name']}` | {n_personas} | {goal_pct} | {avg_drafts} | {avg_sat} | {avg_qual} | {precision} | {rubric_1st} |")
        else:
            lines.append(f"  {run['name']:<28} {n_personas:>3} {goal_pct:>6} {avg_drafts:>6} {avg_sat:>6} {avg_qual:>6} {precision:>6} {rubric_1st:>5}")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _avg(values):
    """Average of a list, handling empty."""
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else 0.0


def _trend(scores):
    """Describe the trend of a score list."""
    if len(scores) < 2:
        return "—"
    diff = scores[-1] - scores[0]
    if abs(diff) < 0.05:
        return "flat"
    return f"{'↑' if diff > 0 else '↓'}{abs(diff):.2f}"


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze RubricLLM simulation results")
    parser.add_argument("paths", nargs="+", help="Simulation output directories (or glob patterns)")
    parser.add_argument("--format", choices=["text", "md"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw analysis as JSON")
    args = parser.parse_args()

    # Resolve glob patterns
    run_dirs = []
    for p in args.paths:
        expanded = sorted(glob.glob(p))
        if expanded:
            run_dirs.extend(expanded)
        elif os.path.isdir(p):
            run_dirs.append(p)

    if not run_dirs:
        print("No simulation directories found.", file=sys.stderr)
        sys.exit(1)

    # Load all runs
    runs = []
    for d in run_dirs:
        run = load_run(d)
        if run:
            runs.append(run)
        else:
            print(f"  Skipping {d} (no summary.json)", file=sys.stderr)

    if not runs:
        print("No valid runs found.", file=sys.stderr)
        sys.exit(1)

    # JSON output mode
    if args.json:
        output = []
        for run in runs:
            run_analysis = {
                "name": run["name"],
                "config": run["config"],
                "personas": {},
            }
            for slug, pdata in run["personas"].items():
                run_analysis["personas"][slug] = {
                    "termination": analyze_termination(pdata),
                    "rubric_quality": analyze_rubric_quality(pdata),
                    "rubric_trajectory": analyze_rubric_trajectory(pdata),
                    "alignment": analyze_alignment(pdata),
                    "log_changes": analyze_log_changes(pdata),
                    "surveys": analyze_surveys(pdata),
                    "preference_coverage": analyze_coverage(pdata),
                }
            output.append(run_analysis)
        print(json.dumps(output, indent=2, default=str))
        return

    # Text/Markdown output
    fmt = args.format

    # Cross-run comparison (if multiple runs)
    if len(runs) > 1:
        print(format_cross_run_comparison(runs, fmt))
        print()

    # Per-run details
    for run in runs:
        print(format_run_report(run, fmt))
        print()


if __name__ == "__main__":
    main()
