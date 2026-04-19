"""Condition B: iterative with holistic feedback."""

from __future__ import annotations

import statistics
import sys
from typing import Any

from config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    MAX_TOKENS_GENERATION,
    NUM_GRADING_RUNS,
    NUM_ROUNDS,
)
from grading_core import run_holistic_grading
from models import complete
from prompts.generation import (
    generation_system_prompt,
    generation_user_prompt,
    revision_user_prompt,
)
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
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]

    draft, _meta = complete(
        system=sys_prompt,
        messages=messages,
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MAX_TOKENS_GENERATION,
        label=f"{log_prefix}|B|init",
    )
    messages.append({"role": "assistant", "content": draft})

    rounds_out: list[dict[str, Any]] = []

    for r in range(1, NUM_ROUNDS + 1):
        print(f"[{log_prefix}] Condition B round {r}/{NUM_ROUNDS}", file=sys.stderr)
        hruns = []
        for gi in range(1, NUM_GRADING_RUNS + 1):
            print(
                f"[{log_prefix}] Holistic grading {gi}/{NUM_GRADING_RUNS}...",
                file=sys.stderr,
            )
            hruns.append(
                run_holistic_grading(
                    rubric_root, draft, f"{log_prefix}|B|r{r}|h{gi}"
                )
            )
        scores = []
        for h in hruns:
            try:
                scores.append(int(h.get("score", 0)))
            except (TypeError, ValueError):
                pass
        mean_h = float(statistics.mean(scores)) if scores else 0.0
        fb = (hruns[0].get("feedback") or "") if hruns else ""
        grading_block = {
            "holistic_runs": hruns,
            "mean_score_1_10": mean_h,
            "feedback_used_for_revision": fb,
        }
        feedback_str = (
            f"Overall score (mean across grading runs): {mean_h:.1f}/10\n\nFeedback: {fb}"
        )
        rounds_out.append(
            {
                "round": r,
                "draft": draft,
                "grading": grading_block,
                "feedback_sent_to_model": feedback_str if r < NUM_ROUNDS else None,
            }
        )

        if r >= NUM_ROUNDS:
            break

        rev = revision_user_prompt(feedback_str)
        messages.append({"role": "user", "content": rev})
        draft, _ = complete(
            system=sys_prompt,
            messages=messages,
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_TOKENS_GENERATION,
            label=f"{log_prefix}|B|rev{r}",
        )
        messages.append({"role": "assistant", "content": draft})

    return {"rounds": rounds_out, "rubric_used": rubric_root}
