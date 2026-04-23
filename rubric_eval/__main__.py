"""Rubric inference sniff-test harness.

For each fixture conversation:
  1. Call the rubric inference prompt directly (no Streamlit, no session state).
  2. Parse the JSON rubric.
  3. For each criterion, ask an LLM judge: "Would a competent generic LLM
     writing this genre without any rubric already satisfy this criterion
     in most drafts?" YES = generic (fails sniff test), NO = user-specific.
  4. Report per-fixture and aggregate pass rates.

Usage:
    python -m rubric_eval
    python -m rubric_eval --model claude-opus-4-7 --report rubric_eval/report.json
    python -m rubric_eval --fixture cold_email_founder   # run one
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Make top-level modules importable (prompts.py, auth_supabase.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic

from prompts import (
    RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
    RUBRIC_infer_only_user_prompt,
)
from rubric_eval.fixtures import FIXTURES, WEAK_FIXTURES


DEFAULT_MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Conversation formatting (tiny reimpl of _build_conversation_text w/o st)
# ---------------------------------------------------------------------------

def _format_conversation(messages: list[dict]) -> str:
    lines: list[str] = []
    for i, m in enumerate(messages, start=1):
        role = m.get("role", "unknown").upper()
        content = m.get("content", "")
        lines.append(f"[Message #{i}] ({role}): {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Rubric inference call
# ---------------------------------------------------------------------------

def infer_rubric(messages: list[dict], model: str, client: anthropic.Anthropic) -> tuple[dict | None, str]:
    conv_text = _format_conversation(messages)
    user_prompt = RUBRIC_infer_only_user_prompt(conv_text, previous_rubric_json="")

    with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        thinking={"type": "adaptive"},
    ) as stream:
        text = ""
        for event in stream:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                text += event.delta.text

    # Extract the JSON rubric. The prompt may wrap it in <analysis>...</analysis>
    # followed by raw JSON, or return JSON only.
    # Strategy: find the FIRST { that begins a balanced object that parses.
    start = text.find("{")
    if start == -1:
        return None, text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate), text
                except json.JSONDecodeError:
                    continue
    return None, text


# ---------------------------------------------------------------------------
# Sniff test (LLM judge)
# ---------------------------------------------------------------------------

SNIFF_JUDGE_SYSTEM = """You are evaluating whether a rubric criterion captures a USER-SPECIFIC writing preference, or merely restates what any competent LLM would already do for the given writing genre.

Rules:
- Answer "GENERIC" if a competent LLM (like Claude or GPT-4), writing a draft in the stated genre with NO rubric and NO user preferences, would already produce drafts that satisfy this criterion in most cases. These are criteria that merely restate genre defaults.
- Answer "SPECIFIC" if this criterion describes a distinctive habit, boundary, or preference that would meaningfully change what the LLM produces — something most baseline LLM drafts would NOT already do.
- When in doubt about a borderline criterion, answer "GENERIC". The bar is high: criteria should clearly distinguish this user from a competent baseline writer in the same genre.

Return exactly one word: GENERIC or SPECIFIC, followed by a one-sentence justification on a new line.

Example 1:
GENRE: professional email to manager
CRITERION name: "Professional tone"
CRITERION description: "Uses respectful, businesslike language."
Answer: GENERIC
Any baseline LLM drafts a professional email with respectful businesslike language by default. No user-specific signal here.

Example 2:
GENRE: professional email to manager
CRITERION name: "Short-sentence punch"
CRITERION description: "Opens with a single declarative sentence under 12 words; no hedging or preamble."
Answer: SPECIFIC
Baseline LLMs tend to add polite preamble and hedging in emails to managers. An explicit under-12-word no-hedging opener is a distinctive choice.
"""


OVERFIT_JUDGE_SYSTEM = """You are evaluating whether a rubric DIMENSION LABEL is over-fixated on the specific conversation it was inferred from, or whether it captures a transferable rule that applies to any piece of the same writing type by the same user.

A dimension label should be a REUSABLE checkable item. If the same user writes a different instance of the same writing type next week -- different people, different topics, different specifics -- the dimension should still apply WITHOUT rewording. If you'd need to change the label to fit a different instance, it's over-fixated.

Three possible verdicts:

- "TRANSFERABLE": The label describes a pattern/rule that applies across different instances of this writing type. It contains no proper nouns (names of people, companies, products), no topic-domain words (e.g. "machine learning," "Q3 revenue"), and no content that only makes sense because you read this specific conversation. A template user could read this label and know what to do.

- "OVERFIT": The label describes something specific to THIS conversation. Signs: proper nouns in the label, specific numbers that only matter for this piece, topic domain words that don't apply broadly, paraphrases of what the user literally said in this thread. A different instance of the same writing type would require rewording the label.

- "GENERIC": The label is so vague or generic that any competent baseline LLM would already satisfy it without a rubric (e.g. "clear and concise," "professional tone"). These fail a different bar -- they're too broad, not too narrow.

Return exactly one word on the first line: TRANSFERABLE, OVERFIT, or GENERIC. Then give a one-sentence justification on a new line.

Example 1:
GENRE: cold outreach emails
DIMENSION label: "Opens with a specific reference to the recipient"
Verdict: TRANSFERABLE
Applies to any cold email regardless of who the recipient is.

Example 2:
GENRE: cold outreach emails
DIMENSION label: "Opens with a reference to Sarah's recent podcast episode"
Verdict: OVERFIT
"Sarah" and "podcast episode" are specific to this one cold email. Different recipients and different openers would require rewording.

Example 3:
GENRE: cold outreach emails
DIMENSION label: "Uses professional tone"
Verdict: GENERIC
Any baseline LLM uses a professional tone in a cold email by default. Not distinctive.

Example 4:
GENRE: physics paper abstract
DIMENSION label: "Names the specific dataset and baseline at the start of the methods section"
Verdict: TRANSFERABLE
Applies to any physics paper; "dataset" and "baseline" are standard concepts for the genre, not specific to one paper.

Example 5:
GENRE: physics paper abstract
DIMENSION label: "Mentions the ImageNet dataset and the ResNet-50 baseline specifically"
Verdict: OVERFIT
Names specific datasets/models from one paper; a different paper on a different topic couldn't satisfy this without rewording.
"""


def overfit_test_dimension(
    genre: str,
    dim_label: str,
    model: str,
    client: anthropic.Anthropic,
) -> tuple[str, str]:
    """Judge a single dimension label as TRANSFERABLE, OVERFIT, or GENERIC."""
    user_prompt = (
        f"GENRE: {genre}\n"
        f"DIMENSION label: \"{dim_label}\"\n\n"
        "Verdict:"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=OVERFIT_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    first_line = (text.splitlines()[0] if text else "").strip().upper()
    m = re.search(r"\b(TRANSFERABLE|OVERFIT|GENERIC)\b", first_line)
    verdict = m.group(1) if m else "UNKNOWN"
    return verdict, text


def sniff_test_criterion(
    genre: str,
    criterion: dict,
    model: str,
    client: anthropic.Anthropic,
) -> tuple[str, str]:
    name = criterion.get("name", "")
    desc = criterion.get("description", "")
    dims = criterion.get("dimensions", []) or []
    dim_lines = "\n".join(
        f"  - {d.get('label') or d.get('description') or d.get('id', '')}"
        for d in dims
    )
    user_prompt = (
        f"GENRE: {genre}\n"
        f"CRITERION name: \"{name}\"\n"
        f"CRITERION description: \"{desc}\"\n"
        f"DIMENSIONS:\n{dim_lines or '  (none)'}\n\n"
        "Answer:"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=SNIFF_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    first_line = (text.splitlines()[0] if text else "").strip().upper()
    m = re.search(r"\b(GENERIC|SPECIFIC)\b", first_line)
    verdict = m.group(1) if m else "UNKNOWN"
    return verdict, text


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_fixture(
    fixture: dict,
    model: str,
    client: anthropic.Anthropic,
    signal: str = "strong",
) -> dict:
    name = fixture["name"]
    genre = fixture["genre"]
    print(f"\n=== [{signal}] {name} ({genre}) ===", flush=True)
    print("  Inferring rubric...", flush=True)
    rubric, raw = infer_rubric(fixture["messages"], model=model, client=client)
    if rubric is None:
        print("  [ERROR] Couldn't parse rubric JSON.", flush=True)
        return {
            "name": name, "genre": genre,
            "error": "parse_failed", "raw_response": raw[:2000],
            "criteria": [], "total": 0, "specific": 0, "generic": 0,
        }

    criteria = rubric.get("rubric") or []
    print(f"  {len(criteria)} criteria inferred. Judging each...", flush=True)

    judged = []
    dim_judgments: list[dict] = []  # per-dimension overfit-test results
    for c in criteria:
        verdict, judge_text = sniff_test_criterion(genre, c, model=model, client=client)
        judged.append({
            "name": c.get("name", ""),
            "description": c.get("description", ""),
            "verdict": verdict,
            "judge_reasoning": judge_text,
        })
        emoji = {"SPECIFIC": "✓", "GENERIC": "✗"}.get(verdict, "?")
        print(f"    {emoji} [{verdict}] {c.get('name', '(unnamed)')}", flush=True)
        # Now judge each dimension label for overfitting to this conversation.
        for d in c.get("dimensions") or []:
            label = (d.get("label") or d.get("description") or d.get("id") or "").strip()
            if not label:
                continue
            dv, dtext = overfit_test_dimension(genre, label, model=model, client=client)
            dim_judgments.append({
                "criterion": c.get("name", ""),
                "dimension_id": d.get("id", ""),
                "label": label,
                "evidence": d.get("evidence", ""),
                "verdict": dv,
                "judge_reasoning": dtext,
            })
            dim_emoji = {"TRANSFERABLE": "✓", "OVERFIT": "✗", "GENERIC": "○"}.get(dv, "?")
            print(f"       {dim_emoji} [{dv}] {label[:70]}", flush=True)

    total = len(judged)
    specific = sum(1 for j in judged if j["verdict"] == "SPECIFIC")
    generic = sum(1 for j in judged if j["verdict"] == "GENERIC")
    print(f"  → criterion-level: {specific}/{total} pass sniff test ({generic} generic)", flush=True)

    dim_total = len(dim_judgments)
    dim_transferable = sum(1 for j in dim_judgments if j["verdict"] == "TRANSFERABLE")
    dim_overfit = sum(1 for j in dim_judgments if j["verdict"] == "OVERFIT")
    dim_generic = sum(1 for j in dim_judgments if j["verdict"] == "GENERIC")
    print(f"  → dim-level:       {dim_transferable}/{dim_total} transferable, "
          f"{dim_overfit} overfit, {dim_generic} generic", flush=True)

    return {
        "name": name,
        "genre": genre,
        "signal": signal,
        "criteria": judged,
        "total": total,
        "specific": specific,
        "generic": generic,
        "dimensions": dim_judgments,
        "dim_total": dim_total,
        "dim_transferable": dim_transferable,
        "dim_overfit": dim_overfit,
        "dim_generic": dim_generic,
        "rubric_raw": rubric,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=None,
                        help="Run only this fixture by name (default: all)")
    parser.add_argument("--set", choices=["all", "strong", "weak"], default="all",
                        help="Which fixture set to run (default: all)")
    parser.add_argument("--report", default="rubric_eval/report.json")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()

    # Build the run list with signal-strength tags.
    run_list: list[tuple[dict, str]] = []
    if args.set in ("all", "strong"):
        for f in FIXTURES:
            run_list.append((f, "strong"))
    if args.set in ("all", "weak"):
        for f in WEAK_FIXTURES:
            run_list.append((f, "weak"))

    if args.fixture:
        run_list = [(f, s) for (f, s) in run_list if f["name"] == args.fixture]
        if not run_list:
            print(f"ERROR: no fixture named {args.fixture!r}", file=sys.stderr)
            return 1

    results = []
    for f, sig in run_list:
        results.append(evaluate_fixture(f, model=args.model, client=client, signal=sig))

    def _agg(subset):
        t = sum(r["total"] for r in subset)
        s = sum(r["specific"] for r in subset)
        g = sum(r["generic"] for r in subset)
        return t, s, g

    strong_res = [r for r in results if r["signal"] == "strong"]
    weak_res = [r for r in results if r["signal"] == "weak"]

    print("\n" + "=" * 60)
    if strong_res:
        t, s, g = _agg(strong_res)
        pct = (100.0 * s / t) if t else 0
        print(f"STRONG-signal: {s}/{t} pass ({g} generic, {pct:.1f}%) across {len(strong_res)} fixtures")
    if weak_res:
        t, s, g = _agg(weak_res)
        pct = (100.0 * s / t) if t else 0
        print(f"WEAK-signal:   {s}/{t} pass ({g} generic, {pct:.1f}%) across {len(weak_res)} fixtures")
    if strong_res and weak_res:
        st_t, st_s, _ = _agg(strong_res)
        wk_t, wk_s, _ = _agg(weak_res)
        st_pct = (100.0 * st_s / st_t) if st_t else 0
        wk_pct = (100.0 * wk_s / wk_t) if wk_t else 0
        delta = st_pct - wk_pct
        print(f"DELTA:         strong - weak = {delta:+.1f}pp (larger gap = prompt pads more when signal is thin)")

    total = sum(r["total"] for r in results)
    specific = sum(r["specific"] for r in results)
    generic = sum(r["generic"] for r in results)
    print(f"AGGREGATE:     {specific}/{total} pass ({generic} generic) across {len(results)} fixtures")

    # --- Dimension-level (overfit) aggregate ---
    def _dim_agg(subset):
        t = sum(r.get("dim_total", 0) for r in subset)
        tr = sum(r.get("dim_transferable", 0) for r in subset)
        of = sum(r.get("dim_overfit", 0) for r in subset)
        g = sum(r.get("dim_generic", 0) for r in subset)
        return t, tr, of, g

    print()
    print("Dimension-level transferability:")
    if strong_res:
        t, tr, of, g = _dim_agg(strong_res)
        pct = (100.0 * tr / t) if t else 0
        print(f"  STRONG: {tr}/{t} transferable ({of} overfit, {g} generic, {pct:.1f}%)")
    if weak_res:
        t, tr, of, g = _dim_agg(weak_res)
        pct = (100.0 * tr / t) if t else 0
        print(f"  WEAK:   {tr}/{t} transferable ({of} overfit, {g} generic, {pct:.1f}%)")
    dim_total_all = sum(r.get("dim_total", 0) for r in results)
    dim_tr_all = sum(r.get("dim_transferable", 0) for r in results)
    dim_of_all = sum(r.get("dim_overfit", 0) for r in results)
    dim_g_all = sum(r.get("dim_generic", 0) for r in results)
    pct_all = (100.0 * dim_tr_all / dim_total_all) if dim_total_all else 0
    print(f"  TOTAL:  {dim_tr_all}/{dim_total_all} transferable "
          f"({dim_of_all} overfit, {dim_g_all} generic, {pct_all:.1f}%)")
    print("=" * 60)

    def _agg_dict(subset):
        t, s, g = _agg(subset)
        return {"total": t, "specific": s, "generic": g,
                "pass_rate": (s / t if t else None)}

    def _dim_agg_dict(subset):
        t, tr, of, g = _dim_agg(subset)
        return {
            "total": t, "transferable": tr, "overfit": of, "generic": g,
            "transferable_rate": (tr / t if t else None),
            "overfit_rate": (of / t if t else None),
        }

    report = {
        "model": args.model,
        "aggregate": _agg_dict(results),
        "by_signal": {
            "strong": _agg_dict(strong_res) if strong_res else None,
            "weak": _agg_dict(weak_res) if weak_res else None,
        },
        "dim_aggregate": _dim_agg_dict(results),
        "dim_by_signal": {
            "strong": _dim_agg_dict(strong_res) if strong_res else None,
            "weak": _dim_agg_dict(weak_res) if weak_res else None,
        },
        "fixtures": results,
    }
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
