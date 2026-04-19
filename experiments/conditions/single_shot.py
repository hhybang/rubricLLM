"""Condition A: single-shot baseline."""

from __future__ import annotations

import sys
from typing import Any

from config import GENERATION_MODEL, GENERATION_TEMPERATURE, MAX_TOKENS_GENERATION
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
    draft, meta = complete(
        system=sys_prompt,
        messages=[{"role": "user", "content": user_msg}],
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MAX_TOKENS_GENERATION,
        label=f"{log_prefix}|gen",
    )
    print(
        f"[{log_prefix}] Generated draft ({meta.get('input_tokens',0)}+{meta.get('output_tokens',0)} tok)",
        file=sys.stderr,
    )

    g = run_criterion_grading_multi(rubric_root, draft, f"{log_prefix}|A")
    runs_raw = g.pop("runs_raw", [])

    return {
        "rounds": [
            {
                "round": 1,
                "draft": draft,
                "grading": {**g, "runs_raw": runs_raw},
                "feedback_sent_to_model": None,
                "generation_tokens": meta,
            }
        ],
        "rubric_used": rubric_root,
    }
