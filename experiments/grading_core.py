"""Shared grading loops (criterion × N runs, holistic × N)."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    GRADING_MODEL,
    GRADING_TEMPERATURE,
    MAX_TOKENS_GRADING,
    NUM_GRADING_RUNS,
)
from models import complete
from prompts import grading_criterion as gc
from prompts import grading_holistic as gh
from rubric_utils import (
    bottom_k_criteria,
    canonical_criterion_name,
    extract_json_object,
    format_rubric,
    weighted_overall_score,
)


def _normalize_criteria_list(criteria: list[Any] | None, rubric_root: dict[str, Any]) -> None:
    if not isinstance(criteria, list):
        return
    for c in criteria:
        if not isinstance(c, dict):
            continue
        raw = c.get("criterion_name")
        fixed = canonical_criterion_name(str(raw) if raw is not None else None, rubric_root)
        if fixed is not None:
            c["criterion_name"] = fixed


def run_holistic_grading(
    rubric_root: dict[str, Any], draft: str, label: str
) -> dict[str, Any]:
    system = gh.HOLISTIC_SYSTEM
    user = gh.holistic_user_prompt(format_rubric(rubric_root), draft)
    text, meta = complete(
        system=system,
        messages=[{"role": "user", "content": user}],
        model=GRADING_MODEL,
        temperature=GRADING_TEMPERATURE,
        max_tokens=MAX_TOKENS_GRADING,
        label=f"{label}|holistic",
    )
    data = extract_json_object(text) or {}
    if not data.get("score"):
        data["parse_error"] = text[:2000]
    data["_tokens"] = meta
    return data


def run_criterion_grading(
    rubric_root: dict[str, Any], draft: str, label: str
) -> dict[str, Any]:
    system = gc.CRITERION_SYSTEM
    user = gc.criterion_user_prompt(format_rubric(rubric_root), draft)
    text, meta = complete(
        system=system,
        messages=[{"role": "user", "content": user}],
        model=GRADING_MODEL,
        temperature=GRADING_TEMPERATURE,
        max_tokens=MAX_TOKENS_GRADING,
        label=f"{label}|criterion",
    )
    data = extract_json_object(text) or {}
    crits = data.get("criteria")
    if not isinstance(crits, list):
        data["parse_error"] = text[:2000]
        data["criteria"] = []
    else:
        _normalize_criteria_list(crits, rubric_root)
    data["_tokens"] = meta
    return data


def aggregate_criterion_runs(
    runs: list[dict[str, Any]], rubric_root: dict[str, Any]
) -> dict[str, Any]:
    """Build run_1..n, mean_scores, std_scores, mean_overall."""
    by_name: dict[str, list[float]] = {}
    out_runs: dict[str, Any] = {}
    for i, r in enumerate(runs, start=1):
        key = f"run_{i}"
        criteria = r.get("criteria") if isinstance(r.get("criteria"), list) else []
        _normalize_criteria_list(criteria, rubric_root)
        scores_map: dict[str, float] = {}
        for c in criteria:
            name = c.get("criterion_name")
            sc = c.get("score")
            if name is None or sc is None:
                continue
            try:
                scores_map[str(name)] = float(sc)
            except (TypeError, ValueError):
                continue
        w = weighted_overall_score(scores_map, rubric_root)
        out_runs[key] = {
            "criteria": criteria,
            "overall_weighted_score": w,
            "parse_error": r.get("parse_error"),
        }
        for k, v in scores_map.items():
            by_name.setdefault(k, []).append(v)

    mean_scores: dict[str, float] = {}
    std_scores: dict[str, float] = {}
    for name, vals in by_name.items():
        if not vals:
            continue
        mean_scores[name] = float(statistics.mean(vals))
        std_scores[name] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0

    mean_overall = weighted_overall_score(mean_scores, rubric_root) if mean_scores else 0.0

    return {
        **out_runs,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "mean_overall": mean_overall,
    }


def run_criterion_grading_multi(
    rubric_root: dict[str, Any], draft: str, label: str
) -> dict[str, Any]:
    runs = [
        run_criterion_grading(rubric_root, draft, f"{label}|g{i}")
        for i in range(1, NUM_GRADING_RUNS + 1)
    ]
    agg = aggregate_criterion_runs(runs, rubric_root)
    return {"runs_raw": runs, **agg}


def feedback_block_criterion(agg: dict[str, Any], rubric_root: dict[str, Any]) -> str:
    """Build per-criterion feedback string (justifications from run_1; means from mean_scores)."""
    r1 = agg.get("run_1") or {}
    criteria = r1.get("criteria") or []
    lines = ["SCORES:"]
    scores_for_weak: dict[str, float] = agg.get("mean_scores") or {}
    for c in criteria:
        name = c.get("criterion_name") or "?"
        sc = scores_for_weak.get(name)
        if sc is None:
            try:
                sc = float(c.get("score", 0))
            except (TypeError, ValueError):
                sc = 0.0
        just = (c.get("justification") or "").strip()
        lines.append(f"- {name}: {sc:.1f}/10")
        lines.append(f"  Justification: {just}")
    weak = bottom_k_criteria(scores_for_weak, 2)
    lines.append("")
    if weak:
        lines.append(
            f"Your weakest criteria are: {', '.join(weak)}. "
            "Focus your revision on improving these."
        )
    else:
        lines.append(
            "All criteria scored similarly on average — keep the email as a complete "
            "work product: do not switch to meta-commentary about scores or the feedback process. "
            "Polish wording, fill any remaining placeholders, and ensure the draft reads as a "
            "single coherent message to the intended audience."
        )
    return "\n".join(lines)
