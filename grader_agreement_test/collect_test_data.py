#!/usr/bin/env python3
"""
Generate synthetic test cases: collaborative conversation → inferred rubric → four graded drafts per scenario.
Uses Claude Opus 4.6. Logs progress to stderr.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

from anthropic import Anthropic

from _anthropic_util import call_anthropic, extract_json_object

DIR = Path(__file__).resolve().parent
_ri_path = DIR / "prompts" / "rubric_inference.py"
_spec = importlib.util.spec_from_file_location("ga_rubric_inference", _ri_path)
_rubric_inf = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rubric_inf)
RUBRIC_INFER_ONLY_SYSTEM_PROMPT = _rubric_inf.RUBRIC_INFER_ONLY_SYSTEM_PROMPT
RUBRIC_infer_only_user_prompt = _rubric_inf.RUBRIC_infer_only_user_prompt

MODEL_OPUS = "claude-opus-4-6"

SCENARIOS = [
    {
        "scenario_id": "cold_email_01",
        "writing_type": "cold outreach email",
        "seed": (
            "The user is a startup founder emailing a busy investor they have never met. "
            "They want a meeting but hate sounding salesy. The collaboration should surface "
            "specific preferences about tone, length, proof, and personalization."
        ),
    },
    {
        "scenario_id": "blog_post_01",
        "writing_type": "technical blog post",
        "seed": (
            "The user is an engineer writing a public blog explaining a debugging war story for peers. "
            "They care about narrative, code snippets, honesty about failures, and pacing."
        ),
    },
    {
        "scenario_id": "manager_update_01",
        "writing_type": "manager weekly update",
        "seed": (
            "The user is a team lead sending a concise status update to their director. "
            "They negotiate how much detail, risk framing, and whether to use bullets vs prose."
        ),
    },
    {
        "scenario_id": "cover_letter_01",
        "writing_type": "job cover letter",
        "seed": (
            "The user is applying to a mid-size product company. They want to sound credible without "
            "generic buzzwords, and they debate how much personality to show."
        ),
    },
    {
        "scenario_id": "technical_doc_01",
        "writing_type": "internal technical design doc",
        "seed": (
            "The user documents a new API for other engineers at their company. They care about "
            "precision, examples, security notes, and keeping the doc skimmable."
        ),
    },
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def format_conversation_text(messages: list[dict]) -> str:
    parts = []
    for i, m in enumerate(messages, start=1):
        role = m.get("role", "unknown")
        content = (m.get("content") or "").strip()
        parts.append(f"[Message #{i}] ({role.upper()}):\n{content}\n")
    return "\n".join(parts)


def strip_analysis_tags(text: str) -> str:
    return re.sub(r"<analysis>[\s\S]*?</analysis>\s*", "", text, flags=re.IGNORECASE).strip()


def normalize_rubric_priorities(rubric_data: dict) -> dict:
    out = copy.deepcopy(rubric_data)
    crit = out.get("rubric") or []
    if not crit:
        return out
    indexed = list(enumerate(crit))
    indexed.sort(key=lambda x: (x[1].get("priority", 999), x[0]))
    for rank, (_, c) in enumerate(indexed, 1):
        c["priority"] = rank
    return out


def parse_rubric_json(response_text: str) -> dict | None:
    cleaned = strip_analysis_tags(response_text)
    m = re.search(r"\{[\s\S]*\"rubric\"[\s\S]*\}", cleaned, re.DOTALL)
    if not m:
        return extract_json_object(cleaned)
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return extract_json_object(cleaned)


GENERATE_CONVERSATION_SYSTEM = """You are simulating a realistic collaborative writing session between a user and an AI writing assistant.
Output ONLY valid JSON (no markdown fences, no commentary). The JSON must match this shape:
{
  "task_prompt": "<one clear sentence: what the user is trying to write>",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
Rules:
- Exactly 4 messages alternating user/assistant/user/assistant (4 turns).
- The user should give concrete feedback, constraints, or pushback so preferences are visible.
- The assistant should offer draft snippets or revisions that reflect the discussion.
- Stay in character for the writing scenario given in the user message.
"""


def generate_collaboration(client: Anthropic, scenario: dict) -> tuple[str, list[dict]]:
    user = (
        f"Writing scenario type: {scenario['writing_type']}\n\n"
        f"Creative direction:\n{scenario['seed']}\n\n"
        "Produce the JSON conversation."
    )
    raw, _, _ = call_anthropic(
        client,
        model=MODEL_OPUS,
        system=GENERATE_CONVERSATION_SYSTEM,
        user=user,
        max_tokens=8192,
        temperature=0.0,
    )
    data = extract_json_object(raw)
    if not data or "messages" not in data:
        raise ValueError(f"Bad conversation JSON: {raw[:500]}...")
    messages = data["messages"]
    task = data.get("task_prompt") or scenario["writing_type"]
    if len(messages) != 4:
        log(f"  warning: expected 4 messages, got {len(messages)}; continuing")
    return task, messages


def infer_rubric(client: Anthropic, conversation_text: str) -> dict:
    user_prompt = RUBRIC_infer_only_user_prompt(conversation_text, previous_rubric_json="")
    raw, _, _ = call_anthropic(
        client,
        model=MODEL_OPUS,
        system=RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=32000,
        temperature=1.0,  # required when thinking is enabled
        thinking={"type": "adaptive"},
    )
    rub = parse_rubric_json(raw)
    if not rub or "rubric" not in rub:
        raise ValueError(f"Rubric parse failed: {raw[:800]}...")
    rub.setdefault("version", 1)
    rub.setdefault("source", "inferred_synthetic")
    rub.setdefault("writing_type", rub.get("writing_type") or "")
    return normalize_rubric_priorities(rub)


DRAFT_SPECS = [
    ("good", "high", "Write ONE complete draft that strongly satisfies every rubric criterion and dimension."),
    (
        "targeted_fail",
        "mixed",
        "Write ONE complete draft that reads as generally competent but deliberately fails 1–2 specific rubric dimensions "
        "(name which dimensions you sabotage only in a final HTML comment at the very end, e.g. <!-- fail: dim_id_1 -->). "
        "The body of the draft must not mention the rubric or this instruction.",
    ),
    (
        "mediocre",
        "low",
        "Write ONE mediocre draft: vague, uneven, only partially meets the task; roughly half of rubric dimensions should be weak or missing.",
    ),
    (
        "misaligned",
        "mixed",
        "Write ONE polished draft that would score well on generic 'good writing' but conflicts with the rubric's stated user preferences "
        "(e.g. wrong tone, structure, or emphasis versus what the rubric rewards). Do not explain the mismatch inside the draft.",
    ),
]


def generate_draft(
    client: Anthropic,
    *,
    writing_type: str,
    task_prompt: str,
    rubric: dict,
    draft_key: str,
    instruction: str,
) -> str:
    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2)
    user = f"""Writing type: {writing_type}
Task: {task_prompt}

Rubric (JSON):
{rubric_json}

Instruction for this draft ({draft_key}):
{instruction}

Output ONLY the draft text (no preamble). If you added an HTML comment for targeted failures, keep it at the very end after the draft."""
    raw, _, _ = call_anthropic(
        client,
        model=MODEL_OPUS,
        system="You write drafts exactly as instructed. Output only the draft body unless an HTML comment is required.",
        user=user,
        max_tokens=8192,
        temperature=0.0,
    )
    return raw.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DIR / "test_cases.json",
        help="Output path for test cases",
    )
    args = ap.parse_args()

    client = Anthropic()
    cases: list[dict] = []

    for sc in SCENARIOS:
        log(f"=== Scenario {sc['scenario_id']} ===")
        log("  generating collaboration...")
        task_prompt, messages = generate_collaboration(client, sc)
        conv_text = format_conversation_text(messages)
        log("  inferring rubric (Opus + thinking)...")
        rubric = infer_rubric(client, conv_text)
        rubric["writing_type"] = rubric.get("writing_type") or sc["writing_type"]

        drafts = []
        for draft_key, expected_quality, instr in DRAFT_SPECS:
            log(f"  draft '{draft_key}'...")
            text = generate_draft(
                client,
                writing_type=sc["writing_type"],
                task_prompt=task_prompt,
                rubric=rubric,
                draft_key=draft_key,
                instruction=instr,
            )
            drafts.append(
                {
                    "draft_id": f"{sc['scenario_id']}_{draft_key}",
                    "text": text,
                    "expected_quality": expected_quality,
                }
            )

        cases.append(
            {
                "scenario_id": sc["scenario_id"],
                "writing_type": sc["writing_type"],
                "task_prompt": task_prompt,
                "collaboration_messages": messages,
                "rubric": rubric,
                "drafts": drafts,
            }
        )

    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {len(cases)} scenarios ({len(cases) * 4} drafts) → {args.output}")


if __name__ == "__main__":
    main()
