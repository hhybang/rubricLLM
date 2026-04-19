"""Condition E: best-of-N sampling."""

from __future__ import annotations

import sys
from typing import Any

from config import (
    BEST_OF_N,
    BEST_OF_N_TEMPERATURE,
    GENERATION_MODEL,
    MAX_TOKENS_GENERATION,
)
from grading_core import run_criterion_grading_multi
from models import complete
from prompts.generation import generation_system_prompt, generation_user_prompt
from rubric_utils import format_rubric, rubric_dict_for_prompt


def run(
    persona: dict[str, Any],
    task_index: int,
    task_description: str,
    log_prefix: str,
) -> dict[str, Any]:
    rubric_root = rubric_dict_for_prompt(persona)
    sys_prompt = generation_system_prompt(
        format_rubric(rubric_root), persona.get("core_preferences") or ""
    )
    user_msg = generation_user_prompt(task_description)

    candidates: list[dict[str, Any]] = []
    best_idx = 0
    best_score = -1.0

    for i in range(BEST_OF_N):
        print(f"[{log_prefix}] Best-of-N candidate {i+1}/{BEST_OF_N}", file=sys.stderr)
        draft, meta = complete(
            system=sys_prompt,
            messages=[{"role": "user", "content": user_msg}],
            model=GENERATION_MODEL,
            temperature=BEST_OF_N_TEMPERATURE,
            max_tokens=MAX_TOKENS_GENERATION,
            label=f"{log_prefix}|E|n{i}",
        )
        g = run_criterion_grading_multi(rubric_root, draft, f"{log_prefix}|E|n{i}|grade")
        runs_raw = g.pop("runs_raw", [])
        mo = float(g.get("mean_overall") or 0.0)
        candidates.append(
            {
                "draft": draft,
                "grading": {**g, "runs_raw": runs_raw},
                "generation_tokens": meta,
            }
        )
        if mo > best_score:
            best_score = mo
            best_idx = i

    selected = candidates[best_idx]
    return {
        "candidates": candidates,
        "selected_index": best_idx,
        "selected_draft": selected["draft"],
        "rubric_used": rubric_root,
    }
