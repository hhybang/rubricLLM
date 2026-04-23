"""Chat system prompt eval: does the generated draft actually reflect the rubric?

For each fixture:
  1. Build the chat system prompt via `CHAT_build_system_prompt(rubric)`.
  2. Send the user task to the primary model under that system prompt.
  3. Extract the <draft>...</draft> block from the response.
  4. For each rubric dim's yes/no check, ask an LLM judge whether the draft
     satisfies it. Returns YES / NO / UNCLEAR.
  5. Separately ask a naturalness judge whether the draft reads like natural
     writing or feels robotic/over-fitted to the rubric.
  6. Report per-fixture and aggregate pass rates, stratified by priority and
     confidence so we can see if the chat prompt weights dims appropriately.

Usage:
    python -m chat_eval
    python -m chat_eval --fixture cold_email_founder
    python -m chat_eval --report chat_eval/report.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Make top-level modules importable (prompts.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic

from prompts import CHAT_build_system_prompt
from chat_eval.fixtures import (
    FIXTURES, CONFLICT_FIXTURES, LOW_CONF_FIXTURES, SPARSE_FIXTURES,
)


DEFAULT_MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Draft generation
# ---------------------------------------------------------------------------

def generate_draft(rubric: dict, task: str, model: str,
                   client: anthropic.Anthropic) -> tuple[str, str]:
    """Build the chat system prompt around `rubric` and send `task` as the user
    message. Returns (extracted_draft, full_response_text). If no <draft> tag
    is found, returns the full text as the draft (best-effort)."""
    system_prompt = CHAT_build_system_prompt(rubric)
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": task}],
    )
    full = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    m = re.search(r"<draft>([\s\S]*?)</draft>", full)
    draft = m.group(1).strip() if m else full.strip()
    return draft, full


# ---------------------------------------------------------------------------
# Per-dim check judge (YES / NO / UNCLEAR)
# ---------------------------------------------------------------------------

CHECK_JUDGE_SYSTEM = """You are evaluating whether a writing draft satisfies a specific yes/no rubric dimension.

You will be given:
- The WRITING TASK the draft was produced for
- The DRAFT itself
- A single QUESTION phrased so YES = the draft satisfies the rubric dim, NO = it violates it

Rules:
- Answer "YES" only if the draft clearly satisfies the question.
- Answer "NO" only if the draft clearly violates the question.
- Answer "UNCLEAR" if you're genuinely uncertain (the draft is ambiguous, edge case, or the question is hard to evaluate on this draft).
- Be strict on "NO" — if the draft has even one clear violation of the question, answer NO.
- Be strict on "YES" — if the draft only partially satisfies the question, answer UNCLEAR.

Return exactly one word on the first line: YES, NO, or UNCLEAR. Then a one-sentence justification on the second line.

Example:
QUESTION: Does the draft AVOID opening with 'I hope this finds you well' or similar boilerplate?
DRAFT: "Hi Priya, I hope you're doing well. I'm reaching out..."
Answer: NO
The draft opens with 'I hope you're doing well', which is the exact boilerplate the question asks to avoid.
"""


def check_one(task: str, draft: str, question: str, model: str,
              client: anthropic.Anthropic) -> tuple[str, str]:
    user_prompt = (
        f"WRITING TASK:\n{task}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"QUESTION: {question}\n\nAnswer:"
    )
    resp = client.messages.create(
        model=model, max_tokens=400,
        system=CHECK_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    first_line = (text.splitlines()[0] if text else "").strip().upper()
    m = re.search(r"\b(YES|NO|UNCLEAR)\b", first_line)
    verdict = m.group(1) if m else "UNCLEAR"
    return verdict, text


# ---------------------------------------------------------------------------
# Naturalness judge
# ---------------------------------------------------------------------------

NATURALNESS_JUDGE_SYSTEM = """You are evaluating whether a writing draft reads like something a skilled human writer would produce, or whether it feels robotic / awkwardly over-fitted to a rubric.

Consider:
- Does the draft flow naturally, or does it feel like each sentence was forced to satisfy a checklist?
- Are there signs of rubric-fitting artifacts — e.g. the draft crams in every constraint at the expense of readability?
- Does it feel like a real person wrote it, or like a model trying to please a grader?

Three verdicts:
- "NATURAL": The draft reads well. A skilled writer could have produced it. The rubric (if applied) is invisible in the output — it shaped the draft but didn't make the draft feel mechanical.
- "MILDLY_FORCED": The draft mostly works but has one or two signs of rubric-fitting (a sentence that feels shoehorned in, slightly awkward phrasing, etc.).
- "OVER_FITTED": The draft reads as robotic, checkbox-y, or forced. Multiple sentences feel shoehorned in to satisfy constraints rather than to serve the writing.

Return exactly one word on the first line: NATURAL, MILDLY_FORCED, or OVER_FITTED. Then a one-sentence justification.
"""


def check_naturalness(task: str, draft: str, model: str,
                      client: anthropic.Anthropic) -> tuple[str, str]:
    user_prompt = f"WRITING TASK:\n{task}\n\nDRAFT:\n{draft}\n\nVerdict:"
    resp = client.messages.create(
        model=model, max_tokens=400,
        system=NATURALNESS_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    first_line = (text.splitlines()[0] if text else "").strip().upper()
    m = re.search(r"\b(NATURAL|MILDLY_FORCED|OVER_FITTED)\b", first_line)
    verdict = m.group(1) if m else "UNCLEAR"
    return verdict, text


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_fixture(fixture: dict, model: str,
                     client: anthropic.Anthropic,
                     category: str = "baseline") -> dict:
    name = fixture["name"]
    task = fixture["task"]
    rubric = fixture["rubric"]
    checks = fixture["checks"]

    print(f"\n=== [{category}] {name} ===", flush=True)
    print(f"  Task: {task[:80]}...", flush=True)
    print("  Generating draft...", flush=True)
    try:
        draft, full_response = generate_draft(rubric, task, model=model, client=client)
    except Exception as e:
        print(f"  [ERROR] draft generation: {e}", flush=True)
        return {"name": name, "error": str(e), "checks": [], "total": 0, "yes": 0}

    print(f"  Draft ({len(draft)} chars). Running {len(checks)} checks...", flush=True)

    check_results: list[dict] = []
    for c in checks:
        verdict, reasoning = check_one(
            task=task, draft=draft, question=c["question"],
            model=model, client=client,
        )
        check_results.append({
            "criterion": c["criterion"],
            "dim_label": c["dim_label"],
            "priority": c["priority"],
            "confidence": c["confidence"],
            "question": c["question"],
            "verdict": verdict,
            "judge_reasoning": reasoning,
        })
        emoji = {"YES": "✓", "NO": "✗", "UNCLEAR": "?"}.get(verdict, "?")
        print(f"    {emoji} [{verdict}] P{c['priority']}/{c['confidence'][:3]}: "
              f"{c['dim_label'][:60]}", flush=True)

    print("  Checking naturalness...", flush=True)
    nat_verdict, nat_reasoning = check_naturalness(
        task=task, draft=draft, model=model, client=client,
    )
    nat_emoji = {"NATURAL": "✓", "MILDLY_FORCED": "~", "OVER_FITTED": "✗"}.get(nat_verdict, "?")
    print(f"    {nat_emoji} [{nat_verdict}] naturalness", flush=True)

    total = len(check_results)
    yes = sum(1 for r in check_results if r["verdict"] == "YES")
    no = sum(1 for r in check_results if r["verdict"] == "NO")
    unclear = sum(1 for r in check_results if r["verdict"] == "UNCLEAR")
    print(f"  → {yes}/{total} dims satisfied ({no} violated, {unclear} unclear)", flush=True)

    return {
        "name": name,
        "category": category,
        "task": task,
        "draft": draft,
        "full_response": full_response,
        "checks": check_results,
        "total": total,
        "yes": yes,
        "no": no,
        "unclear": unclear,
        "naturalness": {"verdict": nat_verdict, "reasoning": nat_reasoning},
    }


def _stratified(results: list[dict], key_fn) -> dict:
    """Aggregate per-check results by some grouping key (e.g. priority)."""
    buckets: dict = {}
    for r in results:
        for c in r.get("checks", []):
            k = key_fn(c)
            b = buckets.setdefault(k, {"yes": 0, "no": 0, "unclear": 0, "total": 0})
            v = c["verdict"]
            if v == "YES": b["yes"] += 1
            elif v == "NO": b["no"] += 1
            else: b["unclear"] += 1
            b["total"] += 1
    for b in buckets.values():
        b["pass_rate"] = (b["yes"] / b["total"]) if b["total"] else None
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=None,
                        help="Run only this fixture by name (default: all)")
    parser.add_argument("--category", default=None,
                        choices=["baseline", "conflict", "low_conf", "sparse"],
                        help="Run only one category of fixtures")
    parser.add_argument("--report", default="chat_eval/report.json")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()

    categorized: list[tuple[dict, str]] = []
    if args.category in (None, "baseline"):
        categorized += [(f, "baseline") for f in FIXTURES]
    if args.category in (None, "conflict"):
        categorized += [(f, "conflict") for f in CONFLICT_FIXTURES]
    if args.category in (None, "low_conf"):
        categorized += [(f, "low_conf") for f in LOW_CONF_FIXTURES]
    if args.category in (None, "sparse"):
        categorized += [(f, "sparse") for f in SPARSE_FIXTURES]

    if args.fixture:
        categorized = [(f, c) for (f, c) in categorized if f["name"] == args.fixture]
        if not categorized:
            print(f"ERROR: no fixture named {args.fixture!r}", file=sys.stderr)
            return 1

    results = [
        evaluate_fixture(f, model=args.model, client=client, category=c)
        for (f, c) in categorized
    ]

    # --- Aggregates ---
    total = sum(r.get("total", 0) for r in results)
    yes = sum(r.get("yes", 0) for r in results)
    no = sum(r.get("no", 0) for r in results)
    unclear = sum(r.get("unclear", 0) for r in results)

    by_priority = _stratified(results, lambda c: f"P{c['priority']}")
    by_confidence = _stratified(results, lambda c: c["confidence"])

    nat_counts: dict = {}
    for r in results:
        v = (r.get("naturalness") or {}).get("verdict", "UNKNOWN")
        nat_counts[v] = nat_counts.get(v, 0) + 1

    pct = (100.0 * yes / total) if total else 0
    print("\n" + "=" * 60)
    print(f"AGGREGATE: {yes}/{total} dims satisfied ({no} violated, {unclear} unclear, {pct:.1f}%)")
    print(f"Across {len(results)} fixture(s)")
    print()
    print("By priority:")
    for k in sorted(by_priority.keys()):
        b = by_priority[k]
        p = (100.0 * b["yes"] / b["total"]) if b["total"] else 0
        print(f"  {k}: {b['yes']}/{b['total']} ({p:.1f}%)  — {b['no']} violated, {b['unclear']} unclear")
    print()
    print("By confidence:")
    for k in ("high", "medium", "low"):
        if k in by_confidence:
            b = by_confidence[k]
            p = (100.0 * b["yes"] / b["total"]) if b["total"] else 0
            print(f"  {k}: {b['yes']}/{b['total']} ({p:.1f}%)")
    print()
    print("Naturalness:")
    for k in ("NATURAL", "MILDLY_FORCED", "OVER_FITTED"):
        if k in nat_counts:
            print(f"  {k}: {nat_counts[k]}/{len(results)}")

    # By-category breakdown.
    print()
    print("By category:")
    category_groups: dict = {}
    for r in results:
        category_groups.setdefault(r.get("category", "baseline"), []).append(r)
    by_category_stats: dict = {}
    for cat, rs in category_groups.items():
        t = sum(r.get("total", 0) for r in rs)
        y = sum(r.get("yes", 0) for r in rs)
        n = sum(r.get("no", 0) for r in rs)
        u = sum(r.get("unclear", 0) for r in rs)
        p = (100.0 * y / t) if t else 0
        print(f"  {cat}: {y}/{t} ({p:.1f}%)  across {len(rs)} fixture(s)  — {n} violated, {u} unclear")
        by_category_stats[cat] = {"total": t, "yes": y, "no": n, "unclear": u,
                                   "pass_rate": (y / t) if t else None,
                                   "fixtures": len(rs)}
    print("=" * 60)

    report = {
        "model": args.model,
        "aggregate": {
            "total": total, "yes": yes, "no": no, "unclear": unclear,
            "pass_rate": (yes / total) if total else None,
        },
        "by_priority": by_priority,
        "by_confidence": by_confidence,
        "by_category": by_category_stats,
        "naturalness": nat_counts,
        "fixtures": results,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
