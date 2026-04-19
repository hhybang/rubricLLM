"""Rubric formatting, weighted scores, correction merge."""

from __future__ import annotations

import copy
import json
import re
from typing import Any


def priority_to_weight(priority: int) -> int:
    """Priority 1 (most important) -> 4, 2->3, 3->2, 4+ -> 1."""
    m = {1: 4, 2: 3, 3: 2, 4: 1}
    return m.get(int(priority), 1)


def format_rubric(rubric_root: dict[str, Any]) -> str:
    """
    `rubric_root` is the inner object with keys writing_type, user_goals_summary, rubric (list), coaching_notes optional,
    plus optional per-criterion negative_examples, clarification_override, etc.
    """
    lines: list[str] = []
    wt = rubric_root.get("writing_type") or ""
    if wt:
        lines.append(f"Writing type: {wt}")
    ugs = rubric_root.get("user_goals_summary") or ""
    if ugs:
        lines.append(f"User goals: {ugs}")
    lines.append("")
    criteria = rubric_root.get("rubric") or []
    for c in sorted(criteria, key=lambda x: int(x.get("priority", 99))):
        name = c.get("name") or "?"
        cat = c.get("category") or ""
        pr = c.get("priority", "?")
        conf = c.get("confidence", "")
        # Title line = criterion name only so graders don't echo metadata as criterion_name.
        lines.append(f"### {name}")
        lines.append(f"Priority: {pr} · Category: {cat} · Confidence: {conf}")
        desc = (c.get("description") or "").strip()
        if desc:
            lines.append(desc)
        dims = c.get("dimensions") or []
        lines.append("Dimensions:")
        for d in dims:
            did = d.get("id") or ""
            lab = d.get("label") or ""
            lines.append(f"  - [{did}] {lab}")
        ne = c.get("negative_examples")
        if ne:
            lines.append("Negative examples (what this criterion does NOT mean):")
            for ex in ne if isinstance(ne, list) else [ne]:
                lines.append(f"  - {ex}")
        cl = c.get("clarification") or c.get("clarification_addendum")
        if cl:
            lines.append(f"Clarification: {cl}")
        lines.append("")
    cn = rubric_root.get("coaching_notes")
    if cn:
        lines.append(f"Coaching notes: {cn}")
    return "\n".join(lines).strip()


def rubric_dict_for_prompt(persona: dict[str, Any], corrected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the rubric object used in prompts (from persona['rubric'], optionally merged with corrections)."""
    base = copy.deepcopy(persona.get("rubric") or {})
    if not corrected:
        return base
    return merge_corrections(base, corrected)


def merge_corrections(rubric_root: dict[str, Any], corrections_payload: dict[str, Any]) -> dict[str, Any]:
    """Apply `corrections` list from simulated human correction JSON onto criteria by name."""
    out = copy.deepcopy(rubric_root)
    by_name = {c.get("name"): c for c in (out.get("rubric") or [])}
    for corr in corrections_payload.get("corrections") or []:
        name = corr.get("criterion_name")
        if not name or name not in by_name:
            continue
        c = by_name[name]
        ne = corr.get("negative_examples")
        if ne:
            c["negative_examples"] = ne if isinstance(ne, list) else [ne]
        cl = corr.get("clarification")
        if cl:
            c["clarification"] = cl
        pa = corr.get("priority_adjustment")
        if pa is not None:
            try:
                c["priority"] = int(pa)
            except (TypeError, ValueError):
                pass
    return out


def weighted_overall_score(
    criterion_scores: dict[str, float],
    rubric_root: dict[str, Any],
) -> float:
    """
    criterion_scores: criterion_name -> score in [1, 10].
    overall = sum(score * weight) / sum(weight * 10)  -> [0, 1]
    """
    criteria = rubric_root.get("rubric") or []
    num = 0.0
    den = 0.0
    for c in criteria:
        name = c.get("name")
        if not name or name not in criterion_scores:
            continue
        w = priority_to_weight(int(c.get("priority", 4)))
        s = float(criterion_scores[name])
        num += s * w
        den += w * 10.0
    if den <= 0:
        return 0.0
    return num / den


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def canonical_criterion_name(raw: str | None, rubric_root: dict[str, Any]) -> str | None:
    """Map grader output to rubric criterion `name` (handles echoed priority/category suffixes)."""
    if not raw:
        return None
    raw_s = str(raw).strip()
    names = [str(c.get("name")) for c in (rubric_root.get("rubric") or []) if c.get("name")]
    if raw_s in names:
        return raw_s
    for n in names:
        if raw_s.startswith(n):
            return n
    if " (priority" in raw_s:
        base = raw_s.split(" (priority", 1)[0].strip()
        if base in names:
            return base
    return raw_s


def bottom_k_criteria(criterion_scores: dict[str, float], k: int = 2) -> list[str]:
    """Lowest-scoring criteria; empty if all scores tie (avoid fake 'weakest' when all are equal)."""
    if not criterion_scores:
        return []
    vals = list(criterion_scores.values())
    if len(vals) > 1 and max(vals) - min(vals) < 1e-9:
        return []
    items = sorted(criterion_scores.items(), key=lambda x: (x[1], x[0]))
    return [n for n, _ in items[:k]]
