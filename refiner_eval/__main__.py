"""Refiner prompt evaluation harness.

Run: python -m refiner_eval [--test-cases path/to/test_cases.json]
                            [--model claude-opus-4-7]
                            [--report path/to/report.json]

For each test case, calls the refiner with the real prompt and validates:
  - scope_change matches expected label
  - reasoning references the user verdict (heuristic: contains second-person 'you')
  - example_annotation references grader evidence (heuristic: word overlap >= 2)

Pass thresholds:
  - scope_change: >= 13/15 (86.7%)
  - reasoning references user: 15/15
  - example_annotation references grader_evidence: 15/15
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


# Make package imports work when this is run as `python -m refiner_eval`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _normalize(text: str) -> list[str]:
    return [w.lower().strip(".,!?;:'\"()[]") for w in (text or "").split() if w.strip()]


def _reasoning_references_user(reasoning: str) -> bool:
    """Heuristic: second-person references (you / your / you're / you've)."""
    rl = (reasoning or "").lower()
    return bool(re.search(r"\b(you|your|you're|you've|you'd)\b", rl))


def _annotation_references_evidence(annotation: str, evidence: str) -> bool:
    """Heuristic: annotation references the grader evidence.

    Passes if:
      (a) annotation contains a quoted fragment (suggesting it's citing the draft), OR
      (b) at least 2 content-word stem overlaps with grader evidence.
    """
    # (a) Quoted text in annotation is a strong signal of citation
    if re.search(r"['\"'\u2018\u2019\u201c\u201d]\w[^'\"'\u2018\u2019\u201c\u201d]{3,}['\"'\u2018\u2019\u201c\u201d]", annotation or ""):
        return True

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "that", "this", "it", "its", "in", "on", "to", "of", "for", "with",
        "and", "or", "but", "as", "by", "at", "from", "has", "have", "had",
        "will", "would", "should", "could", "may", "might", "can", "not",
        "no", "yes", "so", "if", "when", "then", "than", "also", "new",
        "wording", "criterion", "draft", "passage", "example",
    }

    def _stems(words: list[str]) -> set[str]:
        out: set[str] = set()
        for w in words:
            if w in STOPWORDS or len(w) <= 3:
                continue
            # Strip plural -s/-es and hyphenated prefixes ("5-sentence" -> "sentence")
            base = w.split("-")[-1]
            if base.endswith("es"):
                base = base[:-2]
            elif base.endswith("s"):
                base = base[:-1]
            if len(base) > 3:
                out.add(base)
        return out

    ev = _stems(_normalize(evidence))
    an = _stems(_normalize(annotation))
    return len(ev & an) >= 2


def _hedged(text: str) -> bool:
    """Detect hedging language that the prompt explicitly forbids."""
    tl = (text or "").lower()
    hedge_patterns = [
        r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bwould(?! have| not)\b",
        r"\bpotentially\b", r"\bpossibly\b", r"\bperhaps\b",
    ]
    return any(re.search(p, tl) for p in hedge_patterns)


def _call_refiner(case_inputs: dict, model: str) -> tuple[dict | None, str, str]:
    """Call the refiner LLM and return (parsed_suggestion, raw_text, parse_status).

    Mirrors the production pipeline: if the first attempt is over the length
    budget, retry once with an explicit shrink instruction."""
    from rubric_writer.draft_grading_ui import (
        REFINER_SYSTEM_PROMPT,
        _build_refiner_user_prompt,
        _parse_refiner_response,
    )
    from rubric_writer.api_client import _api_call_with_retry

    user_prompt = _build_refiner_user_prompt(
        criterion_wording=case_inputs["criterion_wording"],
        grader_verdict=case_inputs["grader_verdict"],
        grader_confidence=case_inputs["grader_confidence"],
        grader_evidence=case_inputs["grader_evidence"],
        grader_ambiguity_note=case_inputs.get("grader_ambiguity_note", ""),
        user_verdict=case_inputs["user_verdict"],
        draft_excerpt=case_inputs["draft_excerpt"],
    )
    resp = _api_call_with_retry(
        model=model, max_tokens=1500,
        system=REFINER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    parsed, status = _parse_refiner_response(raw)

    # Production parity: if over budget, retry once with shrink instruction.
    if status == "over_budget" and parsed is not None:
        before = (parsed.get("before_wording") or "").strip()
        after = (parsed.get("after_wording") or "").strip()
        target = int(len(before) * 1.5)
        shrink_context = (
            "NOTE: Your previous edit violated the length budget.\n"
            f'  BEFORE ({len(before)} chars): "{before}"\n'
            f'  AFTER ({len(after)} chars): "{after}"\n'
            f"The after_wording MUST be at most {target} characters "
            f"(1.5x the original). Rewrite the edit to fit within that budget. "
            "If you cannot, move the detail into example_annotation -- the "
            "dimension wording itself stays general."
        )
        retry_prompt = shrink_context + "\n\n" + user_prompt
        resp2 = _api_call_with_retry(
            model=model, max_tokens=1500,
            system=REFINER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": retry_prompt}],
        )
        raw2 = "".join(b.text for b in resp2.content if getattr(b, "type", None) == "text")
        parsed2, status2 = _parse_refiner_response(raw2)
        if parsed2 is not None:
            parsed, raw, status = parsed2, raw2, status2

    return parsed, raw, status


def evaluate_case(case: dict, model: str) -> dict:
    expected = case["expected_scope_change"]
    case_id = case["id"]
    try:
        parsed, raw, status = _call_refiner(case["inputs"], model)
    except Exception as e:
        return {
            "case_id": case_id, "error": str(e),
            "passed_scope": False, "passed_reasoning": False, "passed_annotation": False,
        }

    if status == "no_change_needed" or parsed is None:
        return {
            "case_id": case_id,
            "expected_scope": expected,
            "actual_scope": status,
            "passed_scope": False,
            "passed_reasoning": False,
            "passed_annotation": False,
            "raw_response": raw,
            "parse_status": status,
        }

    actual_scope = parsed.get("scope_change", "")
    reasoning = parsed.get("reasoning", "")
    annotation = parsed.get("example_annotation", "")
    evidence = case["inputs"].get("grader_evidence", "")

    return {
        "case_id": case_id,
        "description": case.get("description", ""),
        "expected_scope": expected,
        "actual_scope": actual_scope,
        "before_wording": parsed.get("before_wording", ""),
        "after_wording": parsed.get("after_wording", ""),
        "reasoning": reasoning,
        "example_annotation": annotation,
        "passed_scope": actual_scope == expected,
        "passed_reasoning": _reasoning_references_user(reasoning),
        "passed_annotation": _annotation_references_evidence(annotation, evidence),
        "reasoning_is_hedged": _hedged(reasoning),
        "parse_status": status,
    }


def run_harness(test_cases_path: Path, model: str, report_path: Path | None) -> int:
    with open(test_cases_path) as f:
        cases = json.load(f)

    results = []
    print(f"Running {len(cases)} test cases against {model}...")
    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {case['id']} (expected: {case['expected_scope_change']})", flush=True)
        result = evaluate_case(case, model)
        if "error" in result:
            print(f"    ERROR: {result['error']}", flush=True)
        else:
            s_mark = "✓" if result["passed_scope"] else "✗"
            r_mark = "✓" if result["passed_reasoning"] else "✗"
            a_mark = "✓" if result["passed_annotation"] else "✗"
            print(f"    scope={s_mark} ({result['actual_scope']})  reasoning={r_mark}  annotation={a_mark}", flush=True)
        results.append(result)
        time.sleep(0.5)  # avoid rate limits

    # Aggregate
    total = len(results)
    scope_correct = sum(1 for r in results if r.get("passed_scope"))
    reasoning_correct = sum(1 for r in results if r.get("passed_reasoning"))
    annotation_correct = sum(1 for r in results if r.get("passed_annotation"))

    scope_pass = scope_correct >= 13
    reasoning_pass = reasoning_correct == total
    annotation_pass = annotation_correct == total

    print()
    print("=" * 60)
    print(f"RESULTS ({total} cases)")
    print(f"  scope_change correct:  {scope_correct}/{total}  ({'PASS' if scope_pass else 'FAIL'}; threshold: 13/{total})")
    print(f"  reasoning refs user:   {reasoning_correct}/{total}  ({'PASS' if reasoning_pass else 'FAIL'}; threshold: {total}/{total})")
    print(f"  annotation refs evidence: {annotation_correct}/{total}  ({'PASS' if annotation_pass else 'FAIL'}; threshold: {total}/{total})")
    print()
    overall_pass = scope_pass and reasoning_pass and annotation_pass
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)

    # Per-scope breakdown
    from collections import defaultdict
    by_scope_expected = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if "expected_scope" in r:
            by_scope_expected[r["expected_scope"]]["total"] += 1
            if r.get("passed_scope"):
                by_scope_expected[r["expected_scope"]]["correct"] += 1
    print("\nPer-scope accuracy:")
    for scope, s in sorted(by_scope_expected.items()):
        print(f"  {scope}: {s['correct']}/{s['total']}")

    report = {
        "model": model,
        "total": total,
        "scope_correct": scope_correct,
        "reasoning_correct": reasoning_correct,
        "annotation_correct": annotation_correct,
        "scope_pass": scope_pass,
        "reasoning_pass": reasoning_pass,
        "annotation_pass": annotation_pass,
        "overall_pass": overall_pass,
        "per_scope": {k: dict(v) for k, v in by_scope_expected.items()},
        "results": results,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {report_path}")

    return 0 if overall_pass else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", type=Path,
                        default=Path(__file__).parent / "test_cases.json")
    parser.add_argument("--model", type=str, default=os.environ.get("REFINER_MODEL", "claude-opus-4-7"))
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).parent / "report.json")
    args = parser.parse_args()
    return run_harness(args.test_cases, args.model, args.report)


if __name__ == "__main__":
    sys.exit(main())
