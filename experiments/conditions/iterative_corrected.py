"""Condition D: per-criterion + simulated human rubric correction after round 2."""

from __future__ import annotations

import json
import sys
from typing import Any

from config import (
    CORRECTION_MODEL,
    CORRECTION_TEMPERATURE,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    MAX_TOKENS_GENERATION,
    MAX_TOKENS_GRADING,
    NUM_ROUNDS,
)
from grading_core import feedback_block_criterion, run_criterion_grading_multi
from models import complete
from prompts.correction import build_correction_system
from prompts.generation import (
    generation_system_prompt,
    generation_user_prompt,
    revision_user_prompt,
)
from rubric_utils import (
    extract_json_object,
    format_rubric,
    merge_corrections,
    rubric_dict_for_prompt,
)


def _round_scores_blob(g: dict[str, Any]) -> str:
    return json.dumps(
        {
            "mean_scores": g.get("mean_scores"),
            "mean_overall": g.get("mean_overall"),
            "run_1": g.get("run_1"),
        },
        indent=2,
        ensure_ascii=False,
    )[:12000]


def run(
    persona: dict[str, Any],
    task_index: int,
    task_description: str,
    log_prefix: str,
) -> dict[str, Any]:
    base_rubric = rubric_dict_for_prompt(persona)
    current_rubric = json.loads(json.dumps(base_rubric))
    sys_prompt = generation_system_prompt(
        format_rubric(current_rubric), persona.get("core_preferences") or ""
    )
    user_msg = generation_user_prompt(task_description)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]

    draft, _ = complete(
        system=sys_prompt,
        messages=messages,
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MAX_TOKENS_GENERATION,
        label=f"{log_prefix}|D|init",
    )
    messages.append({"role": "assistant", "content": draft})

    rounds_out: list[dict[str, Any]] = []
    corrections_applied: dict[str, Any] | None = None

    for r in range(1, NUM_ROUNDS + 1):
        print(f"[{log_prefix}] Condition D round {r}/{NUM_ROUNDS}", file=sys.stderr)
        g = run_criterion_grading_multi(current_rubric, draft, f"{log_prefix}|D|r{r}")
        runs_raw = g.pop("runs_raw", [])
        fb = feedback_block_criterion(g, current_rubric)

        rounds_out.append(
            {
                "round": r,
                "draft": draft,
                "grading": {**g, "runs_raw": runs_raw},
                "feedback_sent_to_model": fb if r < NUM_ROUNDS else None,
                "rubric_snapshot": json.loads(json.dumps(current_rubric)),
            }
        )

        if r == 2:
            # Simulated human correction before revising toward round 3
            r1 = rounds_out[0]["grading"]
            r2g = {**g, "runs_raw": runs_raw}
            corr_sys = build_correction_system(
                persona.get("name") or "User",
                persona.get("role") or "",
                persona.get("hidden_preferences") or "",
                persona.get("dealbreakers") or "",
                format_rubric(base_rubric),
                f"DRAFT:\n{rounds_out[0]['draft']}\n\nSCORES:\n{_round_scores_blob(r1)}",
                f"DRAFT:\n{draft}\n\nSCORES:\n{_round_scores_blob(r2g)}",
            )
            print(f"[{log_prefix}] Running rubric correction model...", file=sys.stderr)
            ctext, _ = complete(
                system=corr_sys,
                messages=[{"role": "user", "content": "Respond with JSON only."}],
                model=CORRECTION_MODEL,
                temperature=CORRECTION_TEMPERATURE,
                max_tokens=MAX_TOKENS_GRADING,
                label=f"{log_prefix}|D|correct",
            )
            corrections_applied = extract_json_object(ctext) or {"raw": ctext[:4000]}
            if isinstance(corrections_applied, dict) and corrections_applied.get("corrections"):
                current_rubric = merge_corrections(
                    json.loads(json.dumps(base_rubric)), corrections_applied
                )
                sys_prompt = generation_system_prompt(
                    format_rubric(current_rubric), persona.get("core_preferences") or ""
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
            label=f"{log_prefix}|D|rev{r}",
        )
        messages.append({"role": "assistant", "content": draft})

    return {
        "rounds": rounds_out,
        "rubric_used": base_rubric,
        "corrections_applied": corrections_applied,
        "rubric_final": current_rubric,
    }
