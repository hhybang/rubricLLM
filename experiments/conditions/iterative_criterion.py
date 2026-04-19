"""Condition C: per-criterion iterative, no human correction."""

from __future__ import annotations

import sys
from typing import Any

from config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    MAX_TOKENS_GENERATION,
    NUM_ROUNDS,
)
from grading_core import feedback_block_criterion, run_criterion_grading_multi
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

    draft, _ = complete(
        system=sys_prompt,
        messages=messages,
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MAX_TOKENS_GENERATION,
        label=f"{log_prefix}|C|init",
    )
    messages.append({"role": "assistant", "content": draft})

    rounds_out: list[dict[str, Any]] = []

    for r in range(1, NUM_ROUNDS + 1):
        print(f"[{log_prefix}] Condition C round {r}/{NUM_ROUNDS}", file=sys.stderr)
        g = run_criterion_grading_multi(rubric_root, draft, f"{log_prefix}|C|r{r}")
        runs_raw = g.pop("runs_raw", [])
        fb = feedback_block_criterion(g, rubric_root)

        rounds_out.append(
            {
                "round": r,
                "draft": draft,
                "grading": {**g, "runs_raw": runs_raw},
                "feedback_sent_to_model": fb if r < NUM_ROUNDS else None,
            }
        )

        if r >= NUM_ROUNDS:
            break

        messages.append({"role": "user", "content": revision_user_prompt(fb)})
        draft, _ = complete(
            system=sys_prompt,
            messages=messages,
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_TOKENS_GENERATION,
            label=f"{log_prefix}|C|rev{r}",
        )
        messages.append({"role": "assistant", "content": draft})

    return {"rounds": rounds_out, "rubric_used": rubric_root}
