#!/usr/bin/env python3
"""
Two-task synthetic run to check whether the grader produces real signal
(not 100% MET everywhere) on fresh drafts.

Flow:
  1. Task 1: simulated user describes what they want + a quick opinion on a
     cold-start draft. (short conversation, enough to infer a rubric)
  2. Infer a rubric from Task 1's conversation.
  3. Task 2: same domain, different prompt. Generate a draft using the
     inferred rubric.
  4. Grade both Task 1's final draft AND Task 2's draft against the rubric.
  5. Print: rubric, per-criterion scores for both drafts, and dim-level
     detail (grade + confidence + evidence).

Run:
  python scripts/two_task_synthetic_run.py

Optional:
  --domain [email|blog|summary]
  --no-task1-draft  (skip generating a task-1 draft, use a stub user message instead)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Allow running as a script from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.engine import (
    headless_chat_response,
    headless_infer_rubric,
    extract_draft_from_response,
)
from rubric_writer.draft_grading import grade_draft_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("twotask")

# ─── Domains ────────────────────────────────────────────────────────────────
# Each domain has two related tasks. Task 1 is used to gather preferences and
# infer a rubric; Task 2 is a fresh application of that rubric. The "persona"
# text primes a consistent voice/preferences across both tasks.

DOMAINS = {
    "email": {
        "persona": (
            "I'm writing a professional email to my team about a delayed product "
            "launch. I want to be direct and honest about what's delayed, but not "
            "alarmist. Keep it concise. I want a clear next step at the end. "
            "I dislike corporate fluff -- no 'excited to share' or 'wanted to "
            "circle back'. Opening should get straight to the point."
        ),
        "task1": (
            "Write a short email (under 150 words) to the team announcing that "
            "the v2 launch is slipping from next Friday to two weeks later "
            "because we found a regression in the checkout flow during QA."
        ),
        "task2": (
            "Write a short email (under 150 words) to the same team letting "
            "them know the v2 launch is back on track and will ship this Friday "
            "as originally planned, after the checkout regression was resolved."
        ),
    },
    "blog": {
        "persona": (
            "I'm writing a short technical blog post for engineers. I want a "
            "specific, concrete opener (no 'in today's fast-paced world'). "
            "Use a real code example, not pseudocode. Keep paragraphs short. "
            "Acknowledge tradeoffs honestly -- don't oversell the approach. "
            "End with a one-line takeaway, not a generic 'hope this helps'."
        ),
        "task1": (
            "Write a ~300-word technical blog post about using connection "
            "pooling in a Postgres-backed web service."
        ),
        "task2": (
            "Write a ~300-word technical blog post about setting up "
            "structured logging in a Python web service."
        ),
    },
    "summary": {
        "persona": (
            "I summarize academic papers for a general audience. I want a "
            "one-sentence hook that names a concrete finding (not 'researchers "
            "found...'). Avoid jargon; if I must use it, define it in-line. "
            "Structure: finding → why it matters → one caveat. No bullets."
        ),
        "task1": (
            "Summarize a paper that found large language models show "
            "systematic biases in sentiment analysis of non-English text."
        ),
        "task2": (
            "Summarize a paper that found language models are overconfident "
            "when they make factual errors about obscure historical events."
        ),
    },
}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _strip_draft_tags(text: str) -> str:
    """Pull <draft> content out of a chat response, falling back to the full
    text if no tag is present."""
    draft = extract_draft_from_response(text)
    return draft if draft else text


def _score_line(criterion: dict) -> str:
    name = criterion.get("criterion_name", "?")
    score = criterion.get("score", "?")
    return f"{name}: {score}"


def _detail_block(grades: dict) -> str:
    """Human-readable dim-by-dim breakdown."""
    lines = []
    for c in grades.get("grades") or []:
        name = c.get("criterion_name", "?")
        score = c.get("score", "?")
        lines.append(f"\n  {name}  ({score})")
        for d in c.get("dimension_grades") or []:
            g = (d.get("grade") or "").upper()
            mark = "✓" if g == "MET" else "✗"
            conf = d.get("confidence", "?")
            dim_id = d.get("dimension_id", "?")
            ev = (d.get("evidence") or "").strip()
            note = (d.get("ambiguity_note") or "").strip()
            badge = ""
            if conf == "low":
                badge = " [LOW CONF]"
                if note:
                    badge += f" ({note})"
            elif conf == "medium":
                badge = " [med]"
            lines.append(f"    {mark} {dim_id}{badge}")
            if ev:
                lines.append(f"       evidence: {ev[:140]}")
    return "\n".join(lines)


def _summarize_confidence(grades: dict) -> dict:
    counts = {"high": 0, "medium": 0, "low": 0}
    met_counts = {"MET": 0, "NOT_MET": 0}
    for c in grades.get("grades") or []:
        for d in c.get("dimension_grades") or []:
            conf = (d.get("confidence") or "high").lower()
            if conf in counts:
                counts[conf] += 1
            g = (d.get("grade") or "").upper()
            if g in met_counts:
                met_counts[g] += 1
    return {"confidence": counts, "grades": met_counts}


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run(domain_key: str) -> dict:
    if domain_key not in DOMAINS:
        raise SystemExit(f"unknown domain {domain_key!r}; pick from {list(DOMAINS)}")
    d = DOMAINS[domain_key]

    log.info("=== DOMAIN: %s ===", domain_key)
    log.info("Task 1: %s", d["task1"])

    # --- Task 1: one user turn (persona + task) → assistant draft ---
    task1_user = f"{d['persona']}\n\n{d['task1']}"
    task1_messages = [{"role": "user", "content": task1_user}]
    # No rubric yet; generator is unconstrained.
    task1_response = headless_chat_response(task1_messages, rubric_data=None)
    task1_draft = _strip_draft_tags(task1_response)
    log.info("Task 1 draft (%d chars)", len(task1_draft))

    # Simulate a short user follow-up that gives the rubric-inferrer enough
    # signal to extract dimensions. We use the persona itself as the
    # "user's reaction" because it already contains concrete preferences;
    # this is not cheating for our purposes (we want to see whether the
    # grader produces real variance, not whether inference is hard).
    followup = (
        "Here's my reaction: I like that it's short. But it still reads a "
        "little corporate. The opener should be more direct. Also, I want a "
        "concrete next step the reader can take. Please remember these "
        "preferences going forward."
    )
    task1_messages.append({"role": "assistant", "content": task1_response})
    task1_messages.append({"role": "user", "content": followup})

    # --- Infer a rubric from the Task 1 conversation ---
    log.info("Inferring rubric from Task 1 conversation...")
    rubric_data = headless_infer_rubric(task1_messages)
    if not rubric_data or not rubric_data.get("rubric"):
        raise RuntimeError("Rubric inference returned nothing.")
    n_crits = len(rubric_data["rubric"])
    n_dims = sum(len(c.get("dimensions") or []) for c in rubric_data["rubric"])
    log.info("Inferred rubric: %d criteria, %d dimensions", n_crits, n_dims)

    # --- Task 2: fresh task, same persona, rubric-driven generation ---
    log.info("Task 2: %s", d["task2"])
    task2_messages = [{"role": "user", "content": f"{d['persona']}\n\n{d['task2']}"}]
    task2_response = headless_chat_response(task2_messages, rubric_data=rubric_data)
    task2_draft = _strip_draft_tags(task2_response)
    log.info("Task 2 draft (%d chars)", len(task2_draft))

    # --- Grade BOTH drafts against the rubric ---
    log.info("Grading Task 1 draft against inferred rubric...")
    t1_grades, t1_ms, t1_err = grade_draft_sync(
        rubric_dict=rubric_data, draft_text=task1_draft,
    )
    if t1_err or not t1_grades:
        raise RuntimeError(f"Grading Task 1 failed: {t1_err}")

    log.info("Grading Task 2 draft against inferred rubric...")
    t2_grades, t2_ms, t2_err = grade_draft_sync(
        rubric_dict=rubric_data, draft_text=task2_draft,
    )
    if t2_err or not t2_grades:
        raise RuntimeError(f"Grading Task 2 failed: {t2_err}")

    return {
        "domain": domain_key,
        "task1_prompt": d["task1"],
        "task2_prompt": d["task2"],
        "rubric": rubric_data,
        "task1_draft": task1_draft,
        "task2_draft": task2_draft,
        "task1_grades": t1_grades,
        "task2_grades": t2_grades,
    }


def report(result: dict) -> None:
    print("\n" + "=" * 78)
    print(f"DOMAIN: {result['domain']}")
    print("=" * 78)

    # Rubric summary
    rubric = result["rubric"]["rubric"]
    print(f"\nRUBRIC ({len(rubric)} criteria)")
    for c in rubric:
        name = c.get("name", "?")
        desc = (c.get("description") or "").strip()
        dims = c.get("dimensions") or []
        print(f"  • {name}: {desc[:100]}  ({len(dims)} dims)")

    # Task 1 draft + grades
    print("\n" + "-" * 78)
    print("TASK 1 DRAFT")
    print("-" * 78)
    print(result["task1_draft"])
    print("\nTASK 1 SCORES")
    for c in result["task1_grades"].get("grades") or []:
        print(f"  " + _score_line(c))
    print(_detail_block(result["task1_grades"]))
    t1_summary = _summarize_confidence(result["task1_grades"])
    print(f"\n  Task 1 summary: grades={t1_summary['grades']}  confidence={t1_summary['confidence']}")

    # Task 2 draft + grades
    print("\n" + "-" * 78)
    print("TASK 2 DRAFT")
    print("-" * 78)
    print(result["task2_draft"])
    print("\nTASK 2 SCORES")
    for c in result["task2_grades"].get("grades") or []:
        print(f"  " + _score_line(c))
    print(_detail_block(result["task2_grades"]))
    t2_summary = _summarize_confidence(result["task2_grades"])
    print(f"\n  Task 2 summary: grades={t2_summary['grades']}  confidence={t2_summary['confidence']}")

    # Headline numbers
    t1_total = sum(t1_summary["grades"].values())
    t2_total = sum(t2_summary["grades"].values())
    t1_met = t1_summary["grades"]["MET"]
    t2_met = t2_summary["grades"]["MET"]
    print("\n" + "=" * 78)
    print(f"HEADLINE: Task1 {t1_met}/{t1_total} MET  |  Task2 {t2_met}/{t2_total} MET")
    print(f"  Task1 low-conf dims: {t1_summary['confidence']['low']}")
    print(f"  Task2 low-conf dims: {t2_summary['confidence']['low']}")
    print("=" * 78)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="email", choices=list(DOMAINS.keys()))
    p.add_argument("--json-out", help="optional path to dump full result as JSON")
    args = p.parse_args()

    result = run(args.domain)
    report(result)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[full result written to {args.json_out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
