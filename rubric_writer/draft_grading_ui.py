"""Streamlit UI: rubric score dots, drift panels, scorecard / timeline."""
from __future__ import annotations

import html as html_lib
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

import streamlit as st

from auth_supabase import save_project_data
from rubric_writer.draft_grading import extract_primary_draft_text, load_grading_config, normalize_grade_payload, parse_score_pct
from rubric_writer.metrics import log_confirmation, log_fire_rate

_log = logging.getLogger(__name__)


REFINER_SYSTEM_PROMPT = """You are a rubric refinement assistant. Your job is to propose a conservative wording edit to a rubric criterion when a user has disagreed with the grader's evaluation of that criterion on a specific draft.

You will receive:
- CRITERION: the current wording of the rubric dimension
- GRADER_VERDICT: MET or NOT_MET, with confidence
- GRADER_EVIDENCE: the specific passage from the draft that the grader cited
- GRADER_AMBIGUITY_NOTE: (optional) the grader's note about what made the criterion hard to apply
- USER_VERDICT: the user's judgment (agrees or disagrees with the grader)
- DRAFT_EXCERPT: the relevant portion of the draft being evaluated

Your task is to propose a minimal edit to the criterion wording that resolves the disagreement, and produce structured metadata explaining the edit.

## Output format (JSON)

{
  "criterion_name": "<name of the criterion being edited>",
  "dimension_id": "<specific dimension id being edited, or null if criterion-level>",
  "before_wording": "<exact current criterion wording>",
  "after_wording": "<proposed new wording>",
  "reasoning": "<1-2 sentence explanation, second person, tying the edit to the user's feedback>",
  "scope_change": "<one of: expands | narrows | reframes | clarifies>",
  "example_annotation": "<1-2 sentence explanation of how the new wording applies to GRADER_EVIDENCE specifically>",
  "no_change_needed": <true only if no wording change is warranted at all>
}

## Guidance for each field

### after_wording

**HARD CONSTRAINTS -- READ BEFORE WRITING:**

1. **Length budget: after_wording MUST be at most 1.5x the length of before_wording.** If the original dimension is 30 characters, your edit is at most ~45 characters. If the original is 100 characters, your edit is at most ~150. This is a hard ceiling, not a suggestion. If you cannot express the fix within this budget, the fix belongs in example_annotation, not in the dimension wording itself.

2. **No draft-specific content.** Never include phrases, names, topics, or scenarios that refer to the current draft. The dimension must work for a fresh draft of the same type that has none of these specifics. Words to avoid: "when the author...", "such as...", "for example...", "like the sentence 'X'".

3. **No lists of examples inside the wording.** If you want to add a boundary, add ONE short clause (e.g., "regardless of register", "except for technical explanations"). Never stack three examples inside parentheses.

4. **Preserve the voice.** If the original is imperative and short, keep it imperative and short. Do not turn a short dimension into an explanatory paragraph.

Apply the minimum change that resolves the disagreement. Do not rewrite the entire criterion if a clause addition will suffice. Do not change weights.

**Examples:**

BAD (overfits to current draft, way over length budget):
  BEFORE (30 chars): "Uses warm but professional language."
  AFTER (180 chars): "Uses warm but professional language, which can include using 'hey' as a greeting when writing to close colleagues, particularly in casual project updates where informal openers are expected."

GOOD (generalizable, within budget):
  BEFORE (30 chars): "Uses warm but professional language."
  AFTER (80 chars):  "Uses warm but professional language; casual register is acceptable when appropriate to the audience."

BAD (baked-in example, exceeds budget):
  BEFORE: "Claims are supported by evidence proportional to their scope."
  AFTER: "Claims are supported by evidence proportional to their scope, where a single [CITE] placeholder is sufficient for claims framed as the author's observation rather than a general empirical finding."

GOOD (generalizable boundary, within budget):
  BEFORE: "Claims are supported by evidence proportional to their scope."
  AFTER: "Claims are supported by evidence proportional to their scope; author-framed observations need not cite external sources."

### reasoning

Write in second person. Reference the user's feedback explicitly. The reader should understand why the edit was proposed without needing to read the diff.

BAD: "This edit improves the criterion by handling exceptions better."
GOOD: "You indicated that your current draft meets this criterion despite using passive voice in technical sections. This edit adds an exception for technical explanations."

Do not hedge. Do not say "this edit may" or "this edit could"; say what the edit does.

### scope_change

Choose exactly one value based on the behavioral effect of the edit:

- **expands**: More drafts will now count as meeting the criterion than before. Typical signals: adding exceptions, loosening thresholds, allowing alternative forms.
- **narrows**: Fewer drafts will now count as meeting the criterion. Typical signals: adding required conditions, removing exceptions, tightening thresholds.
- **reframes**: The criterion's focus has shifted to a different aspect of the writing. Neither strictly more nor less permissive. Typical signals: replacing one dimension of evaluation with another, shifting the unit of analysis.
- **clarifies**: The wording is sharper but the behavioral scope is unchanged -- the same drafts would meet the criterion under both versions. Typical signals: replacing ambiguous terms with specific ones, adding definitions.

If multiple labels could apply, use these tie-breakers:
1. If the criterion's WHAT-is-being-evaluated has changed (not just how strict), choose **reframes** even if the edit also tightens or loosens.
   Example: going from "has a subject, greeting, body, sign-off" (evaluates structure) to "each section carries meaningful content" (evaluates substance) is **reframes** even though it is also strictly tighter.
2. If the edit is mostly about disambiguating a vague term and the same drafts would pass under either wording, choose **clarifies**.
3. Otherwise, between expands and narrows, pick whichever direction describes the primary effect.

### example_annotation

Reference GRADER_EVIDENCE by quoting or paraphrasing the specific passage. Explain how the new wording applies to that passage. This annotation is the user's concrete check that the edit works for the case they just flagged.

The example_annotation is where draft-specific detail belongs -- NOT in after_wording. Think of after_wording as the reusable rule, and example_annotation as showing that the rule handles this instance correctly.

BAD: "The new wording handles this case better."
GOOD: "The sentence 'The user can be authenticated by the system' was flagged as passive under the original wording. Under the new wording, this sentence is covered by the technical-explanation exception."

## Constraints

- Conservative edits only. Do not rewrite criteria beyond what the disagreement requires.
- Do not change weights.
- Do not propose edits to other criteria in the rubric.
- If the user's feedback does not actually warrant a substantive wording change (e.g., the disagreement is about a one-off stylistic preference, not a systematic interpretation issue), **still produce an edit**: use scope_change "clarifies" with a minimal wording tweak that nods at the user's intent, and explain in reasoning why the edit is small.
- **Do NOT set no_change_needed to true except in the extremely rare case where the criterion is already perfectly unambiguous AND the user's feedback is purely a one-off stylistic preference with no bearing on rubric wording.** When in doubt, produce a minimal clarifies edit.

Produce the JSON output only. No preamble, no explanation, no markdown fences."""


REFINER_MULTI_SYSTEM_PROMPT = """You are a rubric refinement assistant. The user has just disagreed with the grader on MULTIPLE rubric dimensions for the same draft. Your job is to propose coordinated wording edits that address all of the user's feedback at once -- without producing edits that fight each other.

You will receive:
- DRAFT_EXCERPT: the relevant portion of the draft being evaluated (shared across all dimensions)
- DIMENSIONS: a numbered list, each with its own CRITERION wording, GRADER_VERDICT, GRADER_EVIDENCE, GRADER_AMBIGUITY_NOTE, USER_VERDICT

Your task: produce ONE coordinated set of edits, returned as a JSON array. Each array element is an edit for one dimension, in the same order as DIMENSIONS in the input.

## Why coordinated, not independent

Multiple feedback signals on the same draft are often related. For example, if the user says "warmth" was wrongly marked NOT_MET and "professionalism" was wrongly marked MET, the two dimensions may be in tension -- an edit that makes "warmth" easier to satisfy could make "professionalism" harder. Your edits must be self-consistent. If two dimensions push in opposite directions, you must either (a) propose edits that resolve the tension, or (b) explain in reasoning why no edit is appropriate for one of them.

## Output format (JSON array)

[
  {
    "criterion_name": "<name>",
    "dimension_id": "<id>",
    "before_wording": "<exact current wording>",
    "after_wording": "<proposed new wording>",
    "reasoning": "<1-2 sentences, second person, tying the edit to the user's feedback AND noting any cross-dimension considerations>",
    "scope_change": "<expands | narrows | reframes | clarifies>",
    "example_annotation": "<1-2 sentences explaining how the new wording applies to GRADER_EVIDENCE>",
    "no_change_needed": false
  },
  ...
]

The array MUST have the same length as DIMENSIONS, in the same order. If a particular dimension genuinely needs no change (rare), set its no_change_needed to true and leave the wording fields equal.

## Field guidance

ALL the same hard constraints from the single-dim refiner apply per element:
- after_wording at most 1.5x before_wording length
- no draft-specific phrases ("when the author...", quoted fragments, etc.)
- no lists of examples inside the wording
- preserve the voice of the original

Cross-dimension reasoning: in each element's reasoning, briefly note when an edit is shaped by another dimension's feedback ("kept this narrow because the warmth edit already loosens informality"). This is what makes the multi-call worth doing.

scope_change: same 4-label taxonomy. Apply per element.

example_annotation: per element, references that dimension's GRADER_EVIDENCE specifically.

## Constraints

- Conservative edits only.
- Do not change weights.
- Do not propose edits to dimensions not in DIMENSIONS.
- Do NOT set no_change_needed to true except in extremely rare cases.

Produce the JSON array only. No preamble, no explanation, no markdown fences."""


def _build_multi_refiner_user_prompt(
    dim_inputs: list[dict[str, str]],
    draft_excerpt: str,
) -> str:
    """Compose a single user prompt covering N dimensions sharing one draft."""
    lines = [f"DRAFT_EXCERPT: {draft_excerpt}\n", "DIMENSIONS:"]
    for i, di in enumerate(dim_inputs, start=1):
        lines.append(
            f"\n[{i}] criterion_name: {di['criterion_name']}\n"
            f"    dimension_id: {di['dimension_id']}\n"
            f"    CRITERION: {di['criterion_wording']}\n"
            f"    GRADER_VERDICT: {di['grader_verdict']} (confidence: {di['grader_confidence']})\n"
            f"    GRADER_EVIDENCE: {di['grader_evidence']}\n"
            f"    GRADER_AMBIGUITY_NOTE: {di['grader_ambiguity_note'] or '(none)'}\n"
            f"    USER_VERDICT: {di['user_verdict']}"
        )
    lines.append("\n\nProduce the JSON array now, one element per dimension in order.")
    return "\n".join(lines)


def _parse_multi_refiner_response(raw_text: str, expected_count: int) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Parse the multi-refiner response into a list of validated suggestion dicts.

    Returns (suggestions, statuses) where suggestions is None if the array
    couldn't be parsed at all, and statuses is a list with one status per
    element using the same vocabulary as `_parse_refiner_response`."""
    import re as _re
    arr_match = _re.search(r'\[[\s\S]*\]', raw_text or "")
    if not arr_match:
        return None, []
    try:
        parsed = json.loads(arr_match.group())
    except json.JSONDecodeError:
        return None, []
    if not isinstance(parsed, list) or not parsed:
        return None, []

    suggestions: list[dict[str, Any]] = []
    statuses: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            statuses.append("malformed_json")
            suggestions.append({})
            continue
        if item.get("no_change_needed") is True:
            suggestions.append(item)
            statuses.append("no_change_needed")
            continue
        missing = _REFINER_REQUIRED_FIELDS - set(item.keys())
        if missing:
            if "before_wording" in item and "after_wording" in item:
                suggestions.append(item)
                statuses.append("missing_fields")
            else:
                suggestions.append(item)
                statuses.append("missing_fields")
            continue
        scope = (item.get("scope_change") or "").strip().lower()
        if scope not in _REFINER_VALID_SCOPE:
            item["scope_change"] = "clarifies"
            suggestions.append(item)
            statuses.append("invalid_scope")
            continue
        item["scope_change"] = scope
        before = (item.get("before_wording") or "").strip()
        after = (item.get("after_wording") or "").strip()
        if before and after:
            ratio = len(after) / max(len(before), 1)
            item["_length_ratio"] = ratio
            if ratio > 1.5:
                suggestions.append(item)
                statuses.append("over_budget")
                continue
        suggestions.append(item)
        statuses.append("ok")

    return suggestions, statuses



def _build_refiner_user_prompt(
    criterion_wording: str,
    grader_verdict: str,
    grader_confidence: str,
    grader_evidence: str,
    grader_ambiguity_note: str,
    user_verdict: str,
    draft_excerpt: str,
) -> str:
    return (
        f"CRITERION: {criterion_wording}\n"
        f"GRADER_VERDICT: {grader_verdict} (confidence: {grader_confidence})\n"
        f"GRADER_EVIDENCE: {grader_evidence}\n"
        f"GRADER_AMBIGUITY_NOTE: {grader_ambiguity_note or '(none)'}\n"
        f"USER_VERDICT: {user_verdict}\n"
        f"DRAFT_EXCERPT: {draft_excerpt}\n\n"
        "Produce the JSON output now."
    )


# Required fields for a valid suggestion. If any are missing, fall back to diff-only view.
_REFINER_REQUIRED_FIELDS = {"before_wording", "after_wording", "reasoning", "scope_change", "example_annotation"}
_REFINER_VALID_SCOPE = {"expands", "narrows", "reframes", "clarifies"}


def _parse_refiner_response(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse the refiner LLM response into a validated suggestion dict.

    Returns (suggestion, status) where status is one of:
      "ok", "no_change_needed", "malformed_json", "missing_fields", "invalid_scope"
    When status is not "ok" or "no_change_needed", suggestion may be a partial
    dict usable for a diff-only fallback view, or None if nothing usable.
    """
    import re as _re
    js_match = _re.search(r'\{[\s\S]*\}', raw_text or "")
    if not js_match:
        return None, "malformed_json"
    try:
        parsed = json.loads(js_match.group())
    except json.JSONDecodeError:
        return None, "malformed_json"
    if not isinstance(parsed, dict):
        return None, "malformed_json"

    if parsed.get("no_change_needed") is True:
        return parsed, "no_change_needed"

    missing = _REFINER_REQUIRED_FIELDS - set(parsed.keys())
    if missing:
        # Diff-only fallback: return what we have if before/after are present
        if "before_wording" in parsed and "after_wording" in parsed:
            return parsed, "missing_fields"
        # Legacy-compat: accept old_text/new_text if they're there
        if "old_text" in parsed and "new_text" in parsed:
            parsed["before_wording"] = parsed.get("old_text", "")
            parsed["after_wording"] = parsed.get("new_text", "")
            return parsed, "missing_fields"
        return None, "missing_fields"

    scope = (parsed.get("scope_change") or "").strip().lower()
    if scope not in _REFINER_VALID_SCOPE:
        parsed["scope_change"] = "clarifies"  # safest default
        return parsed, "invalid_scope"

    parsed["scope_change"] = scope

    before = (parsed.get("before_wording") or "").strip()
    after = (parsed.get("after_wording") or "").strip()
    if before and after:
        ratio = len(after) / max(len(before), 1)
        parsed["_length_ratio"] = ratio
        if ratio > 1.5:
            return parsed, "over_budget"

    return parsed, "ok"


def _resolve_refiner_inputs(
    feedback_text: str,
    draft_grade: dict[str, Any] | None,
    rubric_dict: dict[str, Any],
) -> dict[str, str] | None:
    """Infer the refiner's structured inputs from the user feedback and the
    current draft_grade. Returns None if we can't identify a specific dimension."""
    # Extract dimension_id from feedback text (our own feedback strings include "Dimension 'xxx'")
    import re as _re
    dim_match = _re.search(r"Dimension '([^']+)'", feedback_text or "")
    dim_hint = dim_match.group(1) if dim_match else ""

    if not draft_grade:
        return None

    # Find the dimension in the grade payload
    target_dim: dict[str, Any] | None = None
    target_crit_name = ""
    target_crit_wording = ""
    for c in draft_grade.get("grades") or []:
        cname = (c.get("criterion_name") or "").strip()
        for d in c.get("dimension_grades") or []:
            did = (d.get("dimension_id") or "").strip()
            did_label = (d.get("label") or "").strip()
            if dim_hint and (did == dim_hint or did_label == dim_hint):
                target_dim = d
                target_crit_name = cname
                break
        if target_dim:
            break
    # If we couldn't match by id/label, fall back to first NOT_MET dim
    if not target_dim:
        for c in draft_grade.get("grades") or []:
            for d in c.get("dimension_grades") or []:
                if (d.get("grade") or "").upper() == "NOT_MET":
                    target_dim = d
                    target_crit_name = (c.get("criterion_name") or "").strip()
                    break
            if target_dim:
                break
    if not target_dim:
        return None

    # Look up the criterion/dimension wording from the rubric
    dim_id = (target_dim.get("dimension_id") or "").strip()
    dim_wording = ""
    for crit in rubric_dict.get("rubric") or []:
        if (crit.get("name") or "").strip() != target_crit_name:
            continue
        target_crit_wording = crit.get("description") or crit.get("name") or ""
        for dim in crit.get("dimensions") or []:
            if (dim.get("id") or "").strip() == dim_id:
                dim_wording = dim.get("label") or dim.get("description") or ""
                break
        break

    # Parse USER_VERDICT from feedback
    fl = (feedback_text or "").lower()
    if "wording_subjective" in fl or "wording is too subjective" in fl:
        # Oscillation: user says the dim flip-flops because the wording is
        # interpretation-dependent. Refiner should operationalize -- replace
        # subjective language with an observable boundary.
        user_verdict = (
            "the dimension wording is too subjective and interpretation-dependent; "
            "operationalize it by replacing subjective language with an observable "
            "boundary or test"
        )
    elif "disagree" in fl or "says it's met" in fl or "says it's mets" in fl:
        user_verdict = "disagrees with grader"
    elif "agree" in fl or "confirms" in fl or "working_on_it" in fl:
        user_verdict = "agrees with grader"
    elif "says met" in fl or "user says met" in fl:
        user_verdict = "disagrees with grader (says MET)"
    elif "says not_met" in fl or "user says not_met" in fl:
        user_verdict = "disagrees with grader (says NOT_MET)"
    else:
        user_verdict = feedback_text

    # Surface the user's free-text reason, if present, by appending it to the
    # canonical verdict label. Some drift panels (spot_check, etc.) let the
    # user type *why* they disagree with the grader; that reasoning is the
    # strongest signal for what the rubric edit should actually do, so we
    # want the refiner to see it verbatim -- not just the MET/NOT_MET flip.
    reason_match = _re.search(r"User reason:\s*(.+?)\s*$",
                              feedback_text or "", flags=_re.DOTALL)
    if reason_match:
        reason_text = reason_match.group(1).strip()
        if reason_text:
            user_verdict = f"{user_verdict}. User's reasoning: {reason_text}"

    return {
        "criterion_name": target_crit_name,
        "dimension_id": dim_id,
        "criterion_wording": dim_wording or target_crit_wording,
        "grader_verdict": (target_dim.get("grade") or "").upper(),
        "grader_confidence": target_dim.get("confidence", "high"),
        "grader_evidence": target_dim.get("evidence", ""),
        "grader_ambiguity_note": target_dim.get("ambiguity_note", ""),
        "user_verdict": user_verdict,
    }


def _call_refiner(
    *,
    inputs: dict[str, str],
    draft_excerpt: str,
    retry_context: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """Call the refiner LLM and return (parsed_suggestion, status).
    retry_context is an optional prefix to include when retrying after a failed verification."""
    from rubric_writer.config import MODEL_PRIMARY
    from rubric_writer.api_client import _api_call_with_retry

    user_prompt = _build_refiner_user_prompt(
        criterion_wording=inputs["criterion_wording"],
        grader_verdict=inputs["grader_verdict"],
        grader_confidence=inputs["grader_confidence"],
        grader_evidence=inputs["grader_evidence"],
        grader_ambiguity_note=inputs["grader_ambiguity_note"],
        user_verdict=inputs["user_verdict"],
        draft_excerpt=draft_excerpt[:4000] if draft_excerpt else "(not available)",
    )
    if retry_context:
        user_prompt = retry_context + "\n\n" + user_prompt

    resp = _api_call_with_retry(
        model=MODEL_PRIMARY, max_tokens=1500,
        system=REFINER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_refiner_response(text)


def _call_multi_refiner(
    *,
    dim_inputs: list[dict[str, str]],
    draft_excerpt: str,
    retry_context: str = "",
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Call the multi-dim refiner. Returns (suggestions, statuses) where each
    suggestion's status is one of the same vocabulary as the single-dim parser."""
    from rubric_writer.config import MODEL_PRIMARY
    from rubric_writer.api_client import _api_call_with_retry

    user_prompt = _build_multi_refiner_user_prompt(
        dim_inputs=dim_inputs,
        draft_excerpt=(draft_excerpt[:4000] if draft_excerpt else "(not available)"),
    )
    if retry_context:
        user_prompt = retry_context + "\n\n" + user_prompt

    resp = _api_call_with_retry(
        model=MODEL_PRIMARY,
        max_tokens=4000,  # larger -- N edits worth of output
        system=REFINER_MULTI_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_multi_refiner_response(text, expected_count=len(dim_inputs))


def _apply_edit_to_rubric_copy(
    rubric_copy: dict[str, Any],
    criterion_name: str,
    dimension_id: str,
    before_text: str,
    after_text: str,
) -> bool:
    """Apply an edit to a rubric deep-copy in memory (no persistence). Returns True if applied."""
    import re as _re
    for crit in rubric_copy.get("rubric") or []:
        if (crit.get("name") or "").strip() != criterion_name.strip():
            continue
        if dimension_id:
            for dim in crit.get("dimensions") or []:
                if (dim.get("id") or "").strip() != dimension_id.strip():
                    continue
                field = "label" if dim.get("label") else "description"
                text = dim.get(field) or ""
                if before_text and before_text in text:
                    dim[field] = text.replace(before_text, after_text, 1)
                    return True
                if before_text and before_text.lower() in text.lower():
                    dim[field] = _re.sub(
                        _re.escape(before_text), after_text, text, count=1, flags=_re.IGNORECASE,
                    )
                    return True
                # No before_text match -- just set the new wording
                dim[field] = after_text
                return True
        else:
            desc = crit.get("description") or ""
            if before_text in desc:
                crit["description"] = desc.replace(before_text, after_text, 1)
                return True
            crit["description"] = after_text
            return True
    return False


def _verify_edit_against_draft(
    suggestion: dict[str, Any],
    current_rubric: dict[str, Any],
    draft_text: str,
) -> tuple[str, dict[str, Any]]:
    """Apply the suggestion to a rubric COPY, re-grade the draft, and check
    that the edit actually resolves the drift that caused the suggestion.

    Each drift kind has its own success criterion, because each drift is caused
    by a different problem:

      - low_confidence: wording was ambiguous -> after edit, grader must (a) give
        the expected grade AND (b) no longer be low-confidence
      - oscillation: wording was unstable -> after edit, grader must (a) give the
        expected grade AND (b) confidence must be high or medium (a proxy for
        stability: a stably-applied dimension is confidently graded)
      - persistent_failure: dimension couldn't be met, user said too strict/vague
        or disagreed -> after edit, grade must match user expectation
      - spot_check: grader was confident but wrong per user -> after edit, grade
        must match user expectation
      - tradeoff: criteria competing -> priority/scope change; no grade to verify.
        Returns "no_expectation" so it ships without strict verification.

    Returns (status, details) where status is one of:
      "aligned"         -> all drift-specific criteria passed
      "mismatch"        -> grader still disagrees with user expectation
      "still_uncertain" -> grade matches but grader is still low-confidence
                           (low_confidence drift)
      "still_unstable"  -> grade matches but grader confidence is still low
                           (oscillation drift -- proxy for instability)
      "no_expectation"  -> couldn't infer user expectation; skip verification
      "error"           -> grading failed
      "dim_not_found"   -> couldn't find the dimension in the re-graded output

    details is a dict with keys like "new_grade", "new_confidence" for display.
    """
    from rubric_writer.draft_grading import grade_draft_sync
    import copy as _copy

    drift_kind = suggestion.get("drift_kind", "")
    user_expected = _infer_user_expected_grade(suggestion)
    if user_expected is None:
        return "no_expectation", {}

    rubric_copy = _copy.deepcopy(current_rubric)
    _apply_edit_to_rubric_copy(
        rubric_copy=rubric_copy,
        criterion_name=suggestion.get("criterion_name", ""),
        dimension_id=suggestion.get("dimension_id", ""),
        before_text=suggestion.get("before_wording") or suggestion.get("old_text", ""),
        after_text=suggestion.get("after_wording") or suggestion.get("new_text", ""),
    )

    grades, _latency, err = grade_draft_sync(rubric_dict=rubric_copy, draft_text=draft_text)
    if err or not grades:
        return "error", {"error": err}

    new_grade_val = None
    new_confidence = None
    crit_name = suggestion.get("criterion_name", "")
    dim_id = suggestion.get("dimension_id", "")
    # Lookup by dim_id (case-insensitive) -- dim_id is the stable identifier.
    # Strict criterion-name matching used to silently fail when the grader's
    # criterion_name differed from the suggestion's by case/whitespace.
    target_dim_id = (dim_id or "").strip().lower()
    target_crit = (crit_name or "").strip().lower()
    matches: list[tuple[dict, dict]] = []
    for c in grades.get("grades") or []:
        for d in c.get("dimension_grades") or []:
            if target_dim_id and (d.get("dimension_id") or "").strip().lower() == target_dim_id:
                matches.append((c, d))
    chosen_d = None
    if len(matches) == 1:
        chosen_d = matches[0][1]
    elif len(matches) > 1 and target_crit:
        for c, d in matches:
            if (c.get("criterion_name") or "").strip().lower() == target_crit:
                chosen_d = d
                break
        if chosen_d is None:
            chosen_d = matches[0][1]

    # Fallback: if no exact dim_id match, try matching by criterion alone +
    # the dim's label/description (the after_wording). The grader sometimes
    # drops or rewrites the dim_id string (e.g. strips underscores, lowercases,
    # or substitutes a descriptive phrase), which made verification fail
    # spuriously even when the grader DID grade the dim -- just under a
    # different key. This fallback recovers those cases.
    if chosen_d is None and target_crit:
        after_text = (suggestion.get("after_wording") or suggestion.get("new_text") or "").strip().lower()
        for c in grades.get("grades") or []:
            if (c.get("criterion_name") or "").strip().lower() != target_crit:
                continue
            for d in c.get("dimension_grades") or []:
                label = (d.get("label") or d.get("description") or "").strip().lower()
                if after_text and label and (label == after_text or after_text in label or label in after_text):
                    chosen_d = d
                    break
            if chosen_d is None and len(c.get("dimension_grades") or []) == 1:
                # Single-dim criterion -- unambiguous match even if ids differ.
                chosen_d = c["dimension_grades"][0]
            if chosen_d is not None:
                break

    # Diagnostic: if still not found, log what IS in the grader output so we
    # can see what the grader emitted vs. what we expected. The "couldn't
    # verify" warning in the UI is opaque on its own.
    if chosen_d is None:
        emitted_ids = []
        for c in grades.get("grades") or []:
            cname = (c.get("criterion_name") or "").strip()
            for d in c.get("dimension_grades") or []:
                emitted_ids.append(f"{cname}::{d.get('dimension_id') or '(no id)'}")
        _log.warning(
            "[verify] dim_not_found: expected dim_id=%r under crit=%r; "
            "grader emitted: %s",
            dim_id, crit_name, emitted_ids[:20],
        )

    if chosen_d is not None:
        new_grade_val = (chosen_d.get("grade") or "").upper() or None
        new_confidence = (chosen_d.get("confidence") or "high").lower()

    details = {"new_grade": new_grade_val, "new_confidence": new_confidence,
               "user_expected": user_expected}

    if new_grade_val is None:
        return "dim_not_found", details
    if new_grade_val != user_expected.upper():
        return "mismatch", details

    # Grade matches. Apply drift-specific follow-up checks.
    if drift_kind == "low_confidence" and new_confidence == "low":
        # The whole point of the edit was to remove the grader's ambiguity --
        # if the re-grade is still low-confidence, the edit didn't clarify enough.
        return "still_uncertain", details

    if drift_kind == "oscillation" and new_confidence == "low":
        # Oscillation was caused by unstable interpretation. Low confidence on
        # the re-grade suggests the edit didn't stabilize the wording either.
        return "still_unstable", details

    return "aligned", details


def _queue_verified_suggestion(
    suggestion: dict[str, Any],
    inputs: dict[str, str],
    drift_kind: str,
    feedback_text: str,
    parse_status: str,
    pre_verification: dict[str, Any],
) -> None:
    """Attach metadata and append to session state + Supabase."""
    import uuid as _uuid
    sb = st.session_state.get("supabase")
    pid = st.session_state.get("current_project_id")
    conv_id = st.session_state.get("selected_conversation", "")
    suggestions_ref = st.session_state.setdefault("rubric_edit_suggestions", [])

    # Drop suggestions whose dim/criterion didn't resolve. This happens when
    # the refiner proposes an edit for a dim that no longer exists in the
    # current rubric (e.g. user renamed/removed it between proposal and
    # verification). Without this, the suggestion lands in the sidebar
    # with an empty dim_id and Apply silently fails. Better to drop it
    # upstream than ship a broken suggestion to the user.
    _resolved_dim = (inputs.get("dimension_id") or "").strip()
    _resolved_crit = (inputs.get("criterion_name") or "").strip()
    if not _resolved_dim or not _resolved_crit:
        _log.warning(
            "[queue suggestion] DROPPED: unresolved inputs (dim=%r crit=%r). "
            "Suggestion will not appear in sidebar.",
            _resolved_dim, _resolved_crit,
        )
        # Still log to project_data for research, but with disposition="dropped_unresolved"
        # so we can count how often this happens.
        if sb and pid:
            try:
                save_project_data(sb, pid, "refiner_proposal", {
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": conv_id,
                    "edit_id": str(_uuid.uuid4()),
                    "drift_kind": drift_kind,
                    "target_criterion": _resolved_crit,
                    "target_dim_id": _resolved_dim,
                    "disposition": "dropped_unresolved",
                    "feedback_text": feedback_text,
                })
            except Exception:
                pass
        return

    # FORCE the criterion_name and dimension_id to our canonical inputs, not
    # whatever the refiner echoed back. The refiner occasionally hallucinates
    # a plausible-looking dim_id that doesn't exist in the rubric (e.g.
    # `claim_before_system_name` when the real id was `claim_first_framing`),
    # which leads to silent Apply failures downstream.
    suggestion["criterion_name"] = inputs["criterion_name"]
    suggestion["dimension_id"] = inputs["dimension_id"]
    suggestion["feedback_text"] = feedback_text
    suggestion["drift_kind"] = drift_kind
    suggestion["timestamp"] = datetime.now().isoformat()
    suggestion["conversation_id"] = conv_id
    suggestion["status"] = "pending"
    suggestion["edit_id"] = str(_uuid.uuid4())
    suggestion["parse_status"] = parse_status
    suggestion["grader_evidence"] = inputs["grader_evidence"]
    suggestion["grader_verdict"] = inputs["grader_verdict"]
    suggestion["pre_verification"] = pre_verification
    suggestions_ref.append(suggestion)
    _log.warning(
        "[queue suggestion] appended suggestion id=%s dim=%s status=%s (list size=%d)",
        suggestion.get("edit_id"), suggestion.get("dimension_id"),
        suggestion.get("status"), len(suggestions_ref),
    )

    # P0.1: log the proposal at creation time with disposition="proposed".
    # When the user acts on it in the sidebar, a second row is written with
    # disposition in {"applied","dismissed"} and the same edit_id so the
    # two can be joined. Without this, a session where the user dismisses
    # every refiner output looks identical to one where the refiner never
    # fired -- both show rubric_edit_applied=0.
    if sb and pid:
        try:
            save_project_data(sb, pid, "refiner_proposal", {
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conv_id,
                "edit_id": suggestion["edit_id"],
                "drift_kind": drift_kind,
                "target_criterion": inputs["criterion_name"],
                "target_dim_id": inputs["dimension_id"],
                "proposed_edit": {
                    "before": suggestion.get("before_wording") or suggestion.get("old_text", ""),
                    "after": suggestion.get("after_wording") or suggestion.get("new_text", ""),
                    "scope_change": suggestion.get("scope_change", ""),
                    "reasoning": suggestion.get("reasoning", ""),
                },
                "disposition": "proposed",
                "parse_status": parse_status,
                "pre_verification": pre_verification,
                "is_retry": suggestion.get("is_retry", False),
                "is_best_attempt": suggestion.get("is_best_attempt", False),
                "feedback_text": feedback_text,
            })
        except Exception as _e:
            _log.warning("refiner_proposal log (proposed) failed: %s", _e)

    try:
        from rubric_writer.metrics import log_edit_shown
        log_edit_shown(
            edit_id=suggestion["edit_id"],
            criterion_id=suggestion.get("criterion_name", ""),
            dimension_id=suggestion.get("dimension_id", ""),
            scope_change=suggestion.get("scope_change", ""),
        )
    except Exception:
        pass

    if sb and pid:
        try:
            save_project_data(sb, pid, "rubric_edit_suggestion", suggestion)
        except Exception:
            pass


def _queue_pending_feedback(
    panel_id: str,
    feedback_text: str,
    drift_kind: str,
    draft_grade: dict[str, Any] | None,
    draft_excerpt: str,
) -> None:
    """Queue a single dimension's feedback for later batched processing.
    Called by each dimension's button; the actual refinement fires only when
    `_flush_pending_feedback_if_complete` is called with all dims resolved."""
    pending = st.session_state.setdefault("_pending_rubric_feedback", {})
    pending.setdefault(panel_id, []).append({
        "feedback_text": feedback_text,
        "drift_kind": drift_kind,
        "draft_grade": draft_grade,
        "draft_excerpt": draft_excerpt,
    })


def _defer_flush(panel_id: str) -> None:
    """Mark a panel as ready to flush its queued feedback on the next render
    PASS, but only AFTER the drift expander has rendered. Running
    `_flush_pending_feedback` directly from inside a button handler puts the
    refiner's spinners (and any `st.caption` progress messages) INSIDE the
    drift expander, which visually clobbers the panel while the user is
    still looking at their resolved dims. Deferring until after the expander
    closes keeps the panel visible and renders the refiner output below it."""
    deferred = st.session_state.setdefault("_deferred_flush_panels", [])
    if panel_id not in deferred:
        deferred.append(panel_id)


def _defer_schedule_refinement(
    feedback_text: str,
    drift_kind: str,
    draft_grade: dict[str, Any] | None,
    draft_excerpt: str,
) -> None:
    """Queue a direct (non-batched) refiner call to run AFTER the drift
    expander renders. Used by tradeoff buttons, which fire one refinement
    per click rather than batching a panel's dims. Same rationale as
    `_defer_flush`: keeps refiner spinners out of the drift expander."""
    queue = st.session_state.setdefault("_deferred_refinements", [])
    queue.append({
        "feedback_text": feedback_text,
        "drift_kind": drift_kind,
        "draft_grade": draft_grade,
        "draft_excerpt": draft_excerpt,
    })


def run_deferred_refiner_work() -> None:
    """Drain both deferred queues. Call from the main chat render loop AFTER
    all messages (and therefore all drift expanders) have rendered, so any
    spinners/captions the refiner emits appear in a stable location below
    the conversation instead of replacing the contents of a drift expander
    while the user is still looking at it."""
    deferred_flushes = st.session_state.pop("_deferred_flush_panels", None) or []
    deferred_refs = st.session_state.pop("_deferred_refinements", None) or []
    if not deferred_flushes and not deferred_refs:
        return
    for panel_id in deferred_flushes:
        try:
            _flush_pending_feedback(panel_id)
        except Exception as e:
            _log.warning("deferred flush failed for %s: %s", panel_id, e)

    for item in deferred_refs:
        try:
            _schedule_feedback_rubric_refinement(
                feedback_text=item["feedback_text"],
                drift_kind=item["drift_kind"],
                draft_grade=item["draft_grade"],
                draft_excerpt=item["draft_excerpt"],
            )
        except Exception as e:
            _log.warning("deferred refinement failed: %s", e)


def _is_remove_feedback(feedback_text: str) -> bool:
    fl = (feedback_text or "").lower()
    return "user wants to remove" in fl or "user says remove" in fl


def _is_no_edit_feedback(feedback_text: str) -> bool:
    """Feedback that explicitly says 'no rubric change needed' -- we should
    not send these to the refiner even in the 3+-dim batch path.
    Covers oscillation's `drafts_varying`, `rubric_fine` (replacement for
    legacy `just_right`), and the legacy `just_right` slug for backward
    compatibility with sessions 1+2."""
    fl = (feedback_text or "").lower()
    return (
        "says drafts_varying" in fl
        or "drafts_varying" in fl
        or "says rubric_fine" in fl
        or "rubric_fine" in fl
        or "says just_right" in fl
    )


def _flush_pending_feedback(panel_id: str) -> None:
    """Process queued feedback for the panel.

    Remove-dimension actions are handled first as a single batch (one new
    rubric version with all removals applied), because they're structural
    changes that don't go through the refiner. Feedback marked as "no edit
    needed" (drafts_varying, just_right) is dropped silently -- the user
    explicitly said they don't want a rubric change. Any remaining items
    then follow the normal refiner path:

      For 1-2 dims: per-dim refiner calls (each independent).
      For 3+ dims: single combined refiner call for cross-dim nuance."""
    pending = st.session_state.get("_pending_rubric_feedback") or {}
    items = pending.pop(panel_id, [])
    st.session_state["_pending_rubric_feedback"] = pending
    if not items:
        return

    # Split into remove-actions, no-edit-actions, and everything else.
    remove_items = [i for i in items if _is_remove_feedback(i.get("feedback_text", ""))]
    no_edit_items = [i for i in items
                     if not _is_remove_feedback(i.get("feedback_text", ""))
                     and _is_no_edit_feedback(i.get("feedback_text", ""))]
    other_items = [i for i in items
                   if not _is_remove_feedback(i.get("feedback_text", ""))
                   and not _is_no_edit_feedback(i.get("feedback_text", ""))]
    if no_edit_items:
        _log.info("[flush] dropping %d no-edit-needed items from panel=%s",
                  len(no_edit_items), panel_id)

    if remove_items:
        # Resolve each remove item to (criterion_name, dimension_id) via the
        # same input resolver the refiner uses, so we can delete by id.
        from rubric_writer.persistence import get_active_rubric
        rubric_dict, _, _ = get_active_rubric()
        removals: list[dict[str, str]] = []
        for item in remove_items:
            ri = _resolve_refiner_inputs(
                item["feedback_text"], item["draft_grade"], rubric_dict or {},
            )
            if not ri:
                continue
            removals.append({
                "criterion_name": ri["criterion_name"],
                "dimension_id": ri["dimension_id"],
                "feedback_text": item["feedback_text"],
            })
        if removals:
            if len(removals) > 1:
                st.caption(f"Removing {len(removals)} dimensions in a single version...")
            _remove_dimensions_and_save(removals=removals)

    if not other_items:
        return

    total = len(other_items)
    if total >= 3:
        st.caption(f"Processing {total} dimensions together for coordinated edits...")
        _schedule_multi_feedback_rubric_refinement(items=other_items)
        return
    for idx, item in enumerate(other_items, start=1):
        if total > 1:
            st.caption(f"Processing feedback {idx}/{total}...")
        _schedule_feedback_rubric_refinement(
            feedback_text=item["feedback_text"],
            drift_kind=item["drift_kind"],
            draft_grade=item["draft_grade"],
            draft_excerpt=item["draft_excerpt"],
        )


def _schedule_feedback_rubric_refinement(
    feedback_text: str,
    drift_kind: str,
    draft_grade: dict[str, Any] | None,
    draft_excerpt: str = "",
) -> None:
    """Full pipeline: refiner -> verify -> retry once if misaligned -> queue.

    Only suggestions that verify successfully (or the best-effort attempt
    after the retry) are shown to the user. Runs synchronously with visible
    spinners so the user sees progress."""
    from rubric_writer.persistence import get_active_rubric

    _log.warning(
        "[refiner] entering for drift_kind=%s feedback=%r",
        drift_kind, (feedback_text or "")[:120],
    )

    rubric_dict, _, _ = get_active_rubric()
    if not rubric_dict or not rubric_dict.get("rubric"):
        _log.warning("[refiner] bail: no active rubric")
        return

    # Tradeoff feedback is a CRITERION-level preference, not a dimension-level
    # edit. The refiner generates dimension-wording edits, so funneling tradeoff
    # feedback through it is meaningless -- `_resolve_refiner_inputs` can't
    # find a "Dimension 'xxx'" tag in the tradeoff feedback text and falls
    # through to "first NOT_MET" dim, which would produce an edit to a random
    # unrelated dimension.
    #
    # Instead we record the preference as a system message in the conversation.
    # System messages get picked up by `_build_conversation_text` which is what
    # `infer_final_rubric` / `regenerate_rubric` feed to the model. That way
    # when the user re-infers the rubric, the model sees their criterion-level
    # priority alongside the rest of the conversation and can weight the new
    # rubric accordingly.
    if drift_kind == "tradeoff":
        _log.info("[refiner] tradeoff preference -> system message: %r",
                  (feedback_text or "")[:200])
        sb = st.session_state.get("supabase")
        pid = st.session_state.get("current_project_id")
        if sb and pid:
            try:
                save_project_data(sb, pid, "tradeoff_preference", {
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": st.session_state.get("selected_conversation"),
                    "feedback_text": feedback_text,
                })
            except Exception:
                pass
        # Append as a system message so rubric inference sees it.
        conv_id = st.session_state.get("selected_conversation")
        sys_content = f"User tradeoff preference: {feedback_text}"
        already_logged = any(
            m.get("role") == "system"
            and m.get("is_tradeoff_preference")
            and m.get("content") == sys_content
            and m.get("conversation_id") == conv_id
            for m in st.session_state.get("messages", []) or []
        )
        if not already_logged:
            st.session_state.setdefault("messages", []).append({
                "role": "system",
                "content": sys_content,
                "conversation_id": conv_id,
                "is_tradeoff_preference": True,
            })
            try:
                from rubric_writer.persistence import _auto_save_conversation
                _auto_save_conversation()
            except Exception as e:
                _log.warning("tradeoff preference auto-save failed: %s", e)
        st.info(
            "Got it — noted your preference. It'll be factored into the "
            "conversation context when you next re-infer the rubric."
        )
        return

    # Short-circuits for feedback that explicitly says "no rubric edit needed."
    # These are checked BEFORE the refiner LLM is called so we don't waste a
    # call producing an unwanted suggestion.
    #
    # - drafts_varying (oscillation): user says the flip reflects real draft
    #   variation, not a rubric flaw. No edit, just acknowledgement.
    # - rubric_fine (oscillation): user endorses current wording (replaces
    #   the legacy "just_right" slug, kept for back-compat with sessions 1+2).
    fl_pre = (feedback_text or "").lower()
    if "says drafts_varying" in fl_pre or "drafts_varying" in fl_pre:
        _log.info("[refiner] drafts_varying short-circuit: %r",
                  (feedback_text or "")[:200])
        st.info(
            "Got it — we'll treat this as natural draft variation and leave "
            "the rubric as-is."
        )
        return
    if ("says rubric_fine" in fl_pre or "rubric_fine" in fl_pre
            or "says just_right" in fl_pre):
        _log.info("[refiner] rubric-fine short-circuit: %r",
                  (feedback_text or "")[:200])
        st.info("Got it — leaving this dimension's wording as-is.")
        return

    inputs = _resolve_refiner_inputs(feedback_text, draft_grade, rubric_dict)
    if not inputs:
        _log.warning(
            "[refiner] bail: could not resolve inputs from feedback=%r for drift_kind=%s",
            (feedback_text or "")[:200], drift_kind,
        )
        return

    # Short-circuit: "remove" is a structural change, not a wording change.
    # The refiner generates `after_wording` edits, which doesn't apply here.
    # Delete the dim directly and save a new rubric version.
    fl = (feedback_text or "").lower()
    if "user wants to remove" in fl or "user says remove" in fl:
        _remove_dimension_and_save(
            criterion_name=inputs["criterion_name"],
            dimension_id=inputs["dimension_id"],
            feedback_text=feedback_text,
        )
        return

    # Find the latest draft text for verification
    draft_text_for_verify = draft_excerpt
    if not draft_text_for_verify:
        for m in reversed(st.session_state.get("messages", []) or []):
            if m.get("role") != "assistant":
                continue
            dt = extract_primary_draft_text(m.get("content") or "")
            if dt:
                draft_text_for_verify = dt
                break

    # --- Attempt 1 ---
    with st.spinner("Analyzing your feedback and proposing a rubric edit..."):
        try:
            suggestion, parse_status = _call_refiner(inputs=inputs, draft_excerpt=draft_excerpt)
        except Exception as e:
            _log.warning("refiner call failed: %s", e)
            st.error(f"Couldn't generate a suggestion: {e}")
            return

    if parse_status == "no_change_needed" or suggestion is None:
        _log.warning(
            "[refiner] bail: no usable suggestion (parse_status=%s, suggestion=%s)",
            parse_status, "None" if suggestion is None else "present",
        )
        st.info("No rubric change needed based on your feedback.")
        return

    # If the first attempt is over the length budget, immediately retry with an
    # explicit shrink instruction. We do NOT verify over-budget edits because
    # verification can return "aligned" and lock in a bloated dimension.
    if parse_status == "over_budget":
        before_text_ob = (suggestion.get("before_wording") or "").strip()
        after_text_ob = (suggestion.get("after_wording") or "").strip()
        ratio_ob = suggestion.get("_length_ratio", 0)
        target_chars = int(len(before_text_ob) * 1.5)
        with st.spinner("Tightening the suggested edit to keep it generalizable..."):
            shrink_context = (
                "NOTE: Your previous edit violated the length budget.\n"
                f'  BEFORE ({len(before_text_ob)} chars): "{before_text_ob}"\n'
                f'  AFTER ({len(after_text_ob)} chars, {ratio_ob:.2f}x): "{after_text_ob}"\n'
                f"The after_wording MUST be at most {target_chars} characters "
                f"(1.5x the original). Rewrite the edit to fit within that budget. "
                "If you cannot, move the detail into example_annotation -- the "
                "dimension wording itself stays general."
            )
            try:
                suggestion_ob, parse_status_ob = _call_refiner(
                    inputs=inputs, draft_excerpt=draft_excerpt, retry_context=shrink_context,
                )
            except Exception as e:
                _log.warning("refiner shrink retry failed: %s", e)
                suggestion_ob, parse_status_ob = None, "error"
        # Use the shrunk version if it parsed; fall back to original otherwise.
        if suggestion_ob is not None and parse_status_ob in ("ok", "over_budget", "missing_fields", "invalid_scope"):
            suggestion = suggestion_ob
            parse_status = parse_status_ob

    # Attach minimal metadata needed for verification
    # FORCE the criterion_name and dimension_id to our canonical inputs, not
    # whatever the refiner echoed back. The refiner occasionally hallucinates
    # a plausible-looking dim_id that doesn't exist in the rubric (e.g.
    # `claim_before_system_name` when the real id was `claim_first_framing`),
    # which leads to silent Apply failures downstream.
    suggestion["criterion_name"] = inputs["criterion_name"]
    suggestion["dimension_id"] = inputs["dimension_id"]
    suggestion["feedback_text"] = feedback_text
    suggestion["grader_verdict"] = inputs["grader_verdict"]
    suggestion["drift_kind"] = drift_kind  # needed for verification's low-confidence check

    # --- Verify attempt 1 ---
    with st.spinner("Verifying the edit by re-grading your draft..."):
        try:
            status_1, details_1 = _verify_edit_against_draft(
                suggestion=suggestion, current_rubric=rubric_dict, draft_text=draft_text_for_verify,
            )
        except Exception as e:
            _log.warning("pre-verification failed: %s", e)
            status_1, details_1 = "error", {}

    new_grade_1 = details_1.get("new_grade")
    new_conf_1 = details_1.get("new_confidence")
    pre_verification = {
        "attempt_1_status": status_1,
        "attempt_1_new_grade": new_grade_1,
        "attempt_1_new_confidence": new_conf_1,
        "user_expected": _infer_user_expected_grade(suggestion),
    }

    if status_1 == "aligned" or status_1 == "no_expectation":
        # Ship it -- either verified or not applicable to verify
        _queue_verified_suggestion(
            suggestion=suggestion, inputs=inputs, drift_kind=drift_kind,
            feedback_text=feedback_text, parse_status=parse_status,
            pre_verification=pre_verification,
        )
        return

    # --- Mismatch, still_uncertain, or error: try once more with retry context ---
    with st.spinner("First edit didn't verify -- trying a different approach..."):
        user_expected = pre_verification["user_expected"] or "MET"
        before_text = suggestion.get("before_wording") or suggestion.get("old_text", "")
        after_text = suggestion.get("after_wording") or suggestion.get("new_text", "")
        budget_chars = int(len(before_text) * 1.5)
        budget_reminder = (
            f"\n\nLENGTH BUDGET REMINDER: after_wording MUST stay <= {budget_chars} "
            f"characters (1.5x the original {len(before_text)}). Do NOT pad the "
            f"dimension with examples, quoted phrases, or draft-specific scenarios. "
            f"If a concrete example is needed, put it in example_annotation, not in "
            f"the dimension wording. The dimension must remain generally applicable "
            f"to fresh drafts of this type."
        )
        if status_1 == "still_uncertain":
            retry_context = (
                "NOTE: You previously suggested this edit, which was applied to the rubric:\n"
                f'  BEFORE: "{before_text}"\n'
                f'  AFTER: "{after_text}"\n'
                f"When the grader re-graded the draft with your edit, it arrived at "
                f"{new_grade_1} (matching the user's expectation of {user_expected}), "
                f"but it was STILL LOW CONFIDENCE. The rubric wording is still ambiguous. "
                f"Propose a DIFFERENT edit that operationalizes the dimension with ONE "
                f"short disambiguating clause -- a single observable boundary or test."
                + budget_reminder
            )
        elif status_1 == "still_unstable":
            retry_context = (
                "NOTE: You previously suggested this edit, which was applied to the rubric:\n"
                f'  BEFORE: "{before_text}"\n'
                f'  AFTER: "{after_text}"\n'
                f"When the grader re-graded the draft with your edit, it arrived at "
                f"{new_grade_1} (matching the user's expectation of {user_expected}), "
                f"but the grader's confidence was still LOW. This dimension was already "
                f"oscillating between MET and NOT_MET across drafts. Propose a DIFFERENT "
                f"edit that replaces ONE subjective word with an observable equivalent."
                + budget_reminder
            )
        else:
            retry_context = (
                "NOTE: You previously suggested this edit, which was applied to the rubric:\n"
                f'  BEFORE: "{before_text}"\n'
                f'  AFTER: "{after_text}"\n'
                f"When the grader re-graded the same draft with your edit applied, "
                f"it still gave {new_grade_1}, but the user expected {user_expected}.\n"
                "Your previous edit was not sufficient. Propose a DIFFERENT edit that "
                "would move the grader's verdict in the expected direction."
                + budget_reminder
            )
        try:
            suggestion_2, parse_status_2 = _call_refiner(
                inputs=inputs, draft_excerpt=draft_excerpt, retry_context=retry_context,
            )
        except Exception as e:
            _log.warning("refiner retry call failed: %s", e)
            suggestion_2, parse_status_2 = None, "error"

    if suggestion_2 is None or parse_status_2 == "no_change_needed":
        # Retry also failed to produce a suggestion. Ship the original attempt
        # with a warning so the user at least has something to react to.
        suggestion["is_best_attempt"] = True
        pre_verification["retry_status"] = parse_status_2 or "no_suggestion"
        _queue_verified_suggestion(
            suggestion=suggestion, inputs=inputs, drift_kind=drift_kind,
            feedback_text=feedback_text, parse_status=parse_status,
            pre_verification=pre_verification,
        )
        st.warning(
            "We tried two edits but couldn't find wording that aligns with your feedback. "
            "Showing the closest attempt -- review carefully."
        )
        return

    # Attach minimal metadata for verification
    suggestion_2.setdefault("criterion_name", inputs["criterion_name"])
    suggestion_2.setdefault("dimension_id", inputs["dimension_id"])
    suggestion_2["feedback_text"] = feedback_text
    suggestion_2["grader_verdict"] = inputs["grader_verdict"]
    suggestion_2["drift_kind"] = drift_kind
    suggestion_2["is_retry"] = True
    suggestion_2["previous_attempt"] = {
        "before_wording": before_text,
        "after_wording": after_text,
        "grader_grade": new_grade_1,
        "user_expected": user_expected,
    }

    # --- Verify attempt 2 ---
    with st.spinner("Verifying the new edit..."):
        try:
            status_2, details_2 = _verify_edit_against_draft(
                suggestion=suggestion_2, current_rubric=rubric_dict, draft_text=draft_text_for_verify,
            )
        except Exception as e:
            _log.warning("retry verification failed: %s", e)
            status_2, details_2 = "error", {}

    new_grade_2 = details_2.get("new_grade")
    new_conf_2 = details_2.get("new_confidence")
    pre_verification["attempt_2_status"] = status_2
    pre_verification["attempt_2_new_grade"] = new_grade_2
    pre_verification["attempt_2_new_confidence"] = new_conf_2

    if status_2 == "aligned" or status_2 == "no_expectation":
        _queue_verified_suggestion(
            suggestion=suggestion_2, inputs=inputs, drift_kind=drift_kind,
            feedback_text=feedback_text, parse_status=parse_status_2,
            pre_verification=pre_verification,
        )
        return

    # Both attempts failed verification. Two regimes:
    #
    #   (a) The verifier couldn't judge (still_uncertain / still_unstable /
    #       dim_not_found / error): we DON'T know whether the edit is bad,
    #       only that we couldn't measure it. Ship as best-attempt so the
    #       user can decide, with a warning.
    #
    #   (b) The verifier produced a concrete grade that contradicts the
    #       user's stated expectation on BOTH attempts: the edit is
    #       measurably wrong. Drop it rather than shipping a broken edit
    #       as "best attempt." Silently dropping is a cleaner signal than
    #       a sidebar suggestion the user can't trust.
    _blocking_statuses = {"mismatch"}
    if status_1 in _blocking_statuses and status_2 in _blocking_statuses:
        _log.warning(
            "[refiner] DROPPED both verification attempts returned %r; "
            "not shipping suggestion. attempt_1_grade=%s attempt_2_grade=%s "
            "user_expected=%s",
            status_2, new_grade_1, new_grade_2,
            pre_verification.get("user_expected"),
        )
        # Telemetry: log as a refiner_proposal with disposition="dropped_verification"
        # so we can count how often this happens without shipping to the sidebar.
        try:
            sb_t = st.session_state.get("supabase")
            pid_t = st.session_state.get("current_project_id")
            if sb_t and pid_t:
                import uuid as _uuid
                save_project_data(sb_t, pid_t, "refiner_proposal", {
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": st.session_state.get("selected_conversation"),
                    "edit_id": str(_uuid.uuid4()),
                    "drift_kind": drift_kind,
                    "target_criterion": inputs.get("criterion_name"),
                    "target_dim_id": inputs.get("dimension_id"),
                    "disposition": "dropped_verification",
                    "pre_verification": pre_verification,
                    "feedback_text": feedback_text,
                })
        except Exception:
            pass
        st.info(
            "We couldn't find an edit that aligns with your feedback. "
            "The dimension may need a manual rewrite — try editing it "
            "directly in the Rubric Configuration tab."
        )
        return

    # Otherwise: we have at least one verifier signal we can't fully trust
    # (low-confidence, grader dropped the dim, error). Fall through to the
    # best-attempt ship path.
    def _len_ratio(sug):
        b = (sug.get("before_wording") or "").strip()
        a = (sug.get("after_wording") or "").strip()
        return len(a) / max(len(b), 1)

    final_suggestion = suggestion_2
    final_parse_status = parse_status_2
    if _len_ratio(suggestion_2) > 1.5 and _len_ratio(suggestion) <= _len_ratio(suggestion_2):
        # Retry blew past budget and isn't actually shorter -- ship attempt 1 instead.
        final_suggestion = suggestion
        final_parse_status = parse_status
        final_suggestion["chosen_over_retry_for_length"] = True
    final_suggestion["is_best_attempt"] = True
    _queue_verified_suggestion(
        suggestion=final_suggestion, inputs=inputs, drift_kind=drift_kind,
        feedback_text=feedback_text, parse_status=final_parse_status,
        pre_verification=pre_verification,
    )
    # Drift-specific failure message
    if status_2 == "still_uncertain":
        st.warning(
            f"We tried two edits but the grader is still low-confidence on this dimension. "
            f"The rubric wording remains ambiguous even after both attempts. "
            f"You may need to edit the rubric manually to make the dimension more specific."
        )
    elif status_2 == "still_unstable":
        st.warning(
            f"We tried two edits but the grader is still low-confidence, meaning the "
            f"dimension will likely continue oscillating. You may need to rewrite the "
            f"dimension to replace subjective terms with observable criteria."
        )
    elif status_2 == "dim_not_found":
        # The grader's output didn't include this dim_id -- usually because the
        # grader dropped it or emitted a slightly different id string. We can't
        # verify the edit's effect, but the suggestion is still queued so the
        # user can review it directly.
        st.warning(
            f"We couldn't verify the edit: the grader didn't return a grade for "
            f"this dimension on the re-graded draft. The suggestion is in the "
            f"sidebar -- review the wording and Apply if it looks right."
        )
    elif status_2 == "error":
        err_msg = details_2.get("error", "unknown error")
        st.warning(
            f"We tried two edits but the re-grading step failed ({err_msg}). "
            f"The suggestion is in the sidebar -- review and Apply manually."
        )
    else:
        st.warning(
            f"We tried two edits but the grader still gave **{new_grade_2 or '?'}** "
            f"when we expected **{user_expected}**. Review the suggestion carefully -- "
            f"you may need to edit the rubric manually."
        )


def _schedule_multi_feedback_rubric_refinement(items: list[dict[str, Any]]) -> None:
    """Multi-dim refiner pipeline for 3+ panel feedbacks.

    Single combined refiner call -> per-edit verification -> queue verified
    suggestions individually so the user can apply/dismiss each one. Tradeoff
    vs. the per-dim path: one bad LLM output blocks all suggestions, but the
    refiner can reason about cross-dim nuance (related signals from the same
    draft, dimensions in tension, etc.)."""
    import copy as _copy
    from rubric_writer.persistence import get_active_rubric

    rubric_dict, _, _ = get_active_rubric()
    if not rubric_dict or not rubric_dict.get("rubric"):
        return

    # Resolve refiner inputs for each item; drop ones we can't resolve.
    resolved: list[tuple[dict[str, str], dict[str, Any]]] = []
    for item in items:
        ri = _resolve_refiner_inputs(item["feedback_text"], item["draft_grade"], rubric_dict)
        if ri:
            resolved.append((ri, item))
    if not resolved:
        _log.info("Multi-refinement: no items resolved to refiner inputs")
        return
    if len(resolved) < 2:
        # Fell back to a single dim after resolution -- just use the per-dim path.
        ri, item = resolved[0]
        _schedule_feedback_rubric_refinement(
            feedback_text=item["feedback_text"], drift_kind=item["drift_kind"],
            draft_grade=item["draft_grade"], draft_excerpt=item["draft_excerpt"],
        )
        return

    dim_inputs = [r[0] for r in resolved]
    # All items in a panel share the same draft, so any draft_excerpt works.
    draft_excerpt = ""
    for _, item in resolved:
        if item.get("draft_excerpt"):
            draft_excerpt = item["draft_excerpt"]
            break
    if not draft_excerpt:
        for m in reversed(st.session_state.get("messages", []) or []):
            if m.get("role") != "assistant":
                continue
            dt = extract_primary_draft_text(m.get("content") or "")
            if dt:
                draft_excerpt = dt
                break

    # --- Single combined refiner call ---
    with st.spinner(f"Analyzing your feedback on {len(dim_inputs)} dimensions for coordinated edits..."):
        try:
            suggestions, statuses = _call_multi_refiner(
                dim_inputs=dim_inputs, draft_excerpt=draft_excerpt,
            )
        except Exception as e:
            _log.warning("multi-refiner call failed: %s", e)
            st.error(f"Couldn't generate coordinated suggestions: {e}")
            # Fall back to per-dim
            for ri, item in resolved:
                _schedule_feedback_rubric_refinement(
                    feedback_text=item["feedback_text"], drift_kind=item["drift_kind"],
                    draft_grade=item["draft_grade"], draft_excerpt=item["draft_excerpt"],
                )
            return

    if suggestions is None or len(suggestions) != len(dim_inputs):
        # Parse failed or array length mismatch -- fall back to per-dim
        _log.info("multi-refiner returned malformed array; falling back to per-dim")
        for ri, item in resolved:
            _schedule_feedback_rubric_refinement(
                feedback_text=item["feedback_text"], drift_kind=item["drift_kind"],
                draft_grade=item["draft_grade"], draft_excerpt=item["draft_excerpt"],
            )
        return

    # --- Per-edit verification against a rubric copy that has ALL edits applied ---
    # We verify each edit in the context of all the others being applied, since
    # that's what the user will actually experience after Apply.
    with st.spinner("Verifying each coordinated edit by re-grading your draft..."):
        # Build a rubric copy with every non-skipped edit applied.
        rubric_with_all = _copy.deepcopy(rubric_dict)
        for sug, status in zip(suggestions, statuses):
            if status == "no_change_needed":
                continue
            before = sug.get("before_wording") or sug.get("old_text", "")
            after = sug.get("after_wording") or sug.get("new_text", "")
            if not before or not after:
                continue
            _apply_edit_to_rubric_copy(
                rubric_copy=rubric_with_all,
                criterion_name=sug.get("criterion_name") or "",
                dimension_id=sug.get("dimension_id") or "",
                before_text=before,
                after_text=after,
            )

        # Re-grade once with the fully-edited rubric, then check each dim against
        # the user's expectation.
        from rubric_writer.draft_grading import grade_draft_sync
        try:
            re_grade, _lat, _err = grade_draft_sync(
                draft_text=draft_excerpt or "(empty)",
                rubric_dict=rubric_with_all,
            )
        except Exception as e:
            _log.warning("multi-refiner verification re-grade failed: %s", e)
            re_grade = None

    # Look up each dim's new grade in the re-grade output.
    def _find_dim_grade(grade_payload, crit_name, dim_id):
        # Lookup by dim_id (case-insensitive). Criterion name only used to
        # disambiguate when the same dim_id appears under multiple criteria.
        if not grade_payload:
            return None, None
        target_dim_id = (dim_id or "").strip().lower()
        target_crit = (crit_name or "").strip().lower()
        if not target_dim_id:
            return None, None
        matches = []
        for c in grade_payload.get("grades") or []:
            for d in c.get("dimension_grades") or []:
                if (d.get("dimension_id") or "").strip().lower() == target_dim_id:
                    matches.append((c, d))
        if not matches:
            return None, None
        chosen = matches[0]
        if len(matches) > 1 and target_crit:
            for c, d in matches:
                if (c.get("criterion_name") or "").strip().lower() == target_crit:
                    chosen = (c, d)
                    break
        _, d = chosen
        return (d.get("grade") or "").upper() or None, d.get("confidence", "")

    # --- Queue verified suggestions individually ---
    queued = 0
    for (ri, item), sug, status in zip(resolved, suggestions, statuses):
        if status == "no_change_needed":
            continue
        if not (sug.get("before_wording") or sug.get("old_text")):
            continue
        new_grade, new_conf = _find_dim_grade(re_grade, ri["criterion_name"], ri["dimension_id"])
        sug["criterion_name"] = ri["criterion_name"]
        sug["dimension_id"] = ri["dimension_id"]
        sug["drift_kind"] = item["drift_kind"]
        sug["from_multi_refiner"] = True
        sug["multi_panel_size"] = len(resolved)
        user_expected = _infer_user_expected_grade(sug)
        aligned = (
            user_expected is not None
            and new_grade is not None
            and new_grade == user_expected.upper()
        )
        if not aligned and new_grade is not None:
            sug["is_best_attempt"] = True
        pre_verification = {
            "attempt_1_status": "aligned" if aligned else ("mismatch" if new_grade else "error"),
            "attempt_1_new_grade": new_grade,
            "attempt_1_new_confidence": new_conf,
            "user_expected": user_expected,
            "verified_with_all_edits": True,
        }
        _queue_verified_suggestion(
            suggestion=sug, inputs=ri, drift_kind=item["drift_kind"],
            feedback_text=item["feedback_text"], parse_status=status,
            pre_verification=pre_verification,
        )
        queued += 1

    if queued == 0:
        st.info("The model didn't propose any rubric changes for this batch of feedback.")


def _aggregate_tradeoff_pairs(messages: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    """Map (improved_criterion, worse_criterion) -> draft_index list for session callouts."""
    out: dict[tuple[str, str], list[int]] = defaultdict(list)
    for m in messages:
        drift = m.get("draft_drift") or {}
        if drift.get("kind") != "tradeoff":
            continue
        di = (m.get("draft_grade_meta") or {}).get("draft_index")
        if di is None:
            continue
        try:
            di_int = int(di)
        except (TypeError, ValueError):
            continue
        ti = drift.get("tradeoff_improvements") or []
        td = drift.get("tradeoff_drops") or []
        if ti and td:
            for timp in ti:
                for tdr in td:
                    imp = str((timp or {}).get("name") or "").strip()
                    w = str((tdr or {}).get("name") or "").strip()
                    if imp and w:
                        out[(imp, w)].append(di_int)
        else:
            for t in drift.get("tradeoffs") or []:
                imp = (t.get("improved") or {}).get("name") or ""
                w = (t.get("worse") or {}).get("name") or ""
                imp, w = str(imp).strip(), str(w).strip()
                if imp and w:
                    out[(imp, w)].append(di_int)
    for k in out:
        out[k] = sorted(set(out[k]))
    return dict(out)


def _dot_row_for_grade(grade_payload: dict[str, Any] | None, drift: dict[str, Any] | None) -> str:
    if not grade_payload or not grade_payload.get("grades"):
        return ""
    parts = []
    d_obj = drift or {}
    # Tradeoff pulse fires ONLY when the drift panel is actually showing a
    # tradeoff. Previously we pulsed whenever tradeoff_improvements AND
    # tradeoff_drops were non-empty, but those arrays get populated on every
    # draft that has any criterion improve+drop pair -- even when the drift
    # panel decides to show a higher-priority panel (low_confidence,
    # oscillation, persistent_failure) OR suppresses drift entirely (early
    # drafts, user edits, small rubrics). Result: dots pulsed with no panel
    # to explain what the pulse meant. Now the dots only pulse when the
    # tradeoff is the kind that actually renders.
    trade = (d_obj.get("kind") == "tradeoff")
    # Only the criteria that actually appear in the tradeoff (improvements or
    # drops) should be painted orange. Previously every criterion with pct<1
    # got orange when a tradeoff panel was open, which made a zero-score
    # criterion (red) render the same as a genuinely-traded criterion.
    tradeoff_crits: set[str] = set()
    if trade:
        for t in (d_obj.get("tradeoff_improvements") or []):
            n = (t.get("name") or "").strip()
            if n:
                tradeoff_crits.add(n)
        for t in (d_obj.get("tradeoff_drops") or []):
            n = (t.get("name") or "").strip()
            if n:
                tradeoff_crits.add(n)
    # Normalize dim_ids to lowercase-stripped so they match the per-criterion
    # NOT_MET lookup below (which also strips). Without this, a persistent dim
    # with trailing whitespace or differing case silently fails the set
    # intersection and its criterion doesn't show the `!` badge.
    oscillating_dims = {
        (o.get("dimension_id") or "").strip().lower()
        for o in d_obj.get("oscillations") or []
    }
    persistent_dims = {
        (p.get("dimension_id") or "").strip().lower()
        for p in d_obj.get("persistent_failure") or []
    }
    has_low_conf = {
        lc.get("criterion")
        for lc in d_obj.get("low_confidence_dims") or []
    }

    for c in sorted(
        grade_payload["grades"],
        key=lambda x: (x.get("criterion_priority", 99), x.get("criterion_name") or ""),
    ):
        cname = c.get("criterion_name") or ""
        pct = parse_score_pct(c.get("score"))

        # --- Determine dot color ---
        # pct is the fraction of dimensions MET for this criterion.
        #   None       → grey (unscored / no parseable score)
        #   >= 0.999   → green (all met)
        #   in tradeoff→ orange (this criterion is part of the tradeoff)
        #   == 0       → red (zero dims met — full fail, NOT "partial")
        #   0 < p < 1  → yellow (partial)
        in_tradeoff = trade and cname in tradeoff_crits
        if pct is None:
            color = "#9e9e9e"
        elif pct >= 0.999:
            color = "#2e7d32"
        elif in_tradeoff:
            color = "#e65100"
        elif pct <= 0.001:
            color = "#c62828"
        else:
            color = "#f9a825"

        pulse = " animation:pulse 1.2s ease-in-out infinite;" if in_tradeoff else ""

        # --- Determine delta arrow ---
        # Only flag dimensions that are currently NOT_MET. Case-insensitive
        # strip to match oscillating_dims / persistent_dims normalization.
        crit_not_met_ids = {
            (d.get("dimension_id") or "").strip().lower()
            for d in c.get("dimension_grades") or []
            if (d.get("grade") or "").upper() == "NOT_MET"
        }
        has_oscillating = bool(crit_not_met_ids & oscillating_dims)
        has_persistent = bool(crit_not_met_ids & persistent_dims)

        if has_oscillating:
            arrow = "~"
            arrow_color = "#e65100"
            arrow_title = "oscillating"
        elif has_persistent:
            arrow = "!"
            arrow_color = "#c62828"
            arrow_title = "persistent failure"
        else:
            arrow = ""
            arrow_color = ""
            arrow_title = ""

        # Low-confidence ring: dashed border
        border = "border:1.5px dashed #ff9800;" if cname in has_low_conf else ""

        tooltip = html_lib.escape(cname)
        if arrow_title:
            tooltip += f" ({arrow_title})"

        dot_html = (
            f'<span title="{tooltip}" '
            f'style="display:inline-block;position:relative;width:10px;height:10px;'
            f'border-radius:50%;background:{color};margin-right:4px;{pulse}{border}">'
        )
        if arrow:
            dot_html += (
                f'<span style="position:absolute;top:-9px;left:50%;transform:translateX(-50%);'
                f'font-size:8px;line-height:1;color:{arrow_color};font-weight:bold;"'
                f' title="{html_lib.escape(arrow_title)}">{arrow}</span>'
            )
        dot_html += "</span>"
        parts.append(dot_html)

    return (
        '<div style="display:flex;align-items:center;gap:2px;margin:6px 0 2px 0;font-size:11px;padding-top:6px;">'
        '<span style="color:#666;margin-right:6px;">Rubric</span>'
        + "".join(parts)
        + "</div>"
        + '<style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.45}}</style>'
    )


# Hover-to-reveal legend using pure CSS. Streamlit's unsafe_allow_html strips
# <script> tags so JS positioning isn't available -- but :hover + absolutely
# positioned popup inside the icon's wrapper works fine. The popup is
# `position:absolute` so it's pinned to the icon's coordinates (relative to
# the wrapper's offsetParent), not the viewport, which means no JS needed.
# `z-index:9999` keeps it above other dots.
_DOT_LEGEND_HTML = (
    '<style>'
    '.dot-legend-wrap{position:relative;display:inline-block;margin-left:6px;'
    'vertical-align:middle;}'
    '.dot-legend-icon{font-size:11px;color:#999;cursor:help;}'
    '.dot-legend-popup{display:none;position:absolute;left:0;top:100%;margin-top:4px;'
    'background:#fff;border:1px solid #ddd;border-radius:6px;'
    'box-shadow:0 4px 12px rgba(0,0,0,0.2);padding:8px 12px;font-size:10px;'
    'color:#555;line-height:1.8;white-space:nowrap;z-index:9999;}'
    '.dot-legend-wrap:hover .dot-legend-popup{display:block;}'
    '</style>'
    '<span class="dot-legend-wrap">'
    '<span class="dot-legend-icon">ⓘ</span>'
    '<span class="dot-legend-popup">'
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#2e7d32;vertical-align:middle;"></span> all met &nbsp;&nbsp;'
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f9a825;vertical-align:middle;"></span> partial &nbsp;&nbsp;'
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#c62828;vertical-align:middle;"></span> none met &nbsp;&nbsp;'
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#e65100;vertical-align:middle;"></span> tradeoff &nbsp;&nbsp;'
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#9e9e9e;vertical-align:middle;"></span> n/a<br>'
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;border:1.5px dashed #ff9800;vertical-align:middle;"></span> low conf &nbsp;&nbsp;'
    '<b>~</b> oscillating &nbsp;<b>!</b> persistent failure'
    '</span>'
    '</span>'
)


def render_draft_grading_chrome(message: dict[str, Any]) -> None:
    cfg = load_grading_config()
    if not cfg.get("show_score_indicator"):
        return
    if message.get("_pre_rubric"):
        return
    if not extract_primary_draft_text(message.get("content") or ""):
        return
    dg = message.get("draft_grade")
    drift = message.get("draft_drift") or {}
    if dg:
        dot_html = _dot_row_for_grade(dg, drift)
        dot_html = dot_html.replace("</div>", _DOT_LEGEND_HTML + "</div>", 1)
        st.markdown(dot_html, unsafe_allow_html=True)
    elif cfg.get("enabled"):
        # Only show "scoring in progress" if grading is enabled and there's an active rubric
        try:
            from rubric_writer.persistence import get_active_rubric
            _rubric, _, _ = get_active_rubric()
            if _rubric and _rubric.get("rubric"):
                st.caption("🔄 _Rubric scoring in progress..._")
        except Exception:
            pass


def _render_dimension_calibration_buttons(
    dims: list[dict[str, Any]], safe_msg_id: str, message: dict[str, Any],
    drift_kind: str, prefix: str,
) -> None:
    """Render 'Too strict / Just right / Remove' buttons for a list of dimensions."""
    for j, dim_info in enumerate(dims):
        did = dim_info.get("dimension_id", "")
        crit = _resolve_crit_for_dim(did, dim_info.get("criterion", ""), message)
        dim_desc = _lookup_dimension_description(did, crit)
        dim_label = dim_desc if dim_desc else did

        resolved_key = f"{prefix}_resolved_{safe_msg_id}_{j}"
        if st.session_state.get(resolved_key):
            st.caption(f"~~{html_lib.escape(dim_label)}~~ _({st.session_state[resolved_key]})_")
            continue

        st.markdown(f"_{html_lib.escape(crit)}_: **{html_lib.escape(dim_label)}**")
        if dim_info.get("streak"):
            st.caption(f"NOT MET for {dim_info['streak']} consecutive drafts")

        cal_meta = message.get("draft_grade_meta") or {}
        col1, col2, col3, col4 = st.columns(4)
        btn_base = f"{prefix}_{safe_msg_id}_{j}"
        for col, label, action in [
            (col1, "📏 Too strict", "too_strict"),
            (col2, "🔍 Too vague", "too_vague"),
            (col3, "✅ Just right", "just_right"),
            (col4, "🗑 Remove", "remove"),
        ]:
            with col:
                if st.button(label, key=f"{btn_base}_{action}"):
                    pre_grade = json.loads(json.dumps(message.get("draft_grade"))) if message.get("draft_grade") else None
                    _store_user_verdict(message, crit, did, action)
                    # Always use `did` (real dim_id) in the feedback text, never the
                    # human description. `_resolve_refiner_inputs` extracts the hint
                    # via a `Dimension '([^']+)'` regex and matches it against the
                    # grade payload's `dimension_id` field; grader payloads don't
                    # reliably carry `label`, so putting the description here made
                    # the resolver fall through to "first NOT_MET" and act on the
                    # wrong dim -- or silently no-op for remove actions.
                    feedback = f"Dimension '{did}' ({crit}): user says {action}."
                    # "just_right" = user confirms grader, "too_strict"/"remove" = user disagrees
                    log_confirmation(
                        source="drift_detected", draft_index=cal_meta.get("draft_index", 0),
                        drift_type=drift_kind, dimension_id=did,
                        dimension_text=dim_label, grader_verdict="NOT_MET",
                        grader_confidence="high", user_response_raw=action,
                        user_confirms_grader=(action == "just_right"),
                    )
                    sb = st.session_state.get("supabase")
                    pid = st.session_state.get("current_project_id")
                    if sb and pid:
                        try:
                            save_project_data(sb, pid, f"{drift_kind}_feedback", {
                                "timestamp": datetime.now().isoformat(),
                                "conversation_id": st.session_state.get("selected_conversation"),
                                "message_id": message.get("message_id"),
                                "dimension_id": did,
                                "criterion": crit,
                                "action": action,
                            })
                        except Exception:
                            pass
                    _defer_schedule_refinement(
                        feedback_text=feedback,
                        drift_kind=drift_kind,
                        draft_grade=pre_grade,
                        draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                    )
                    st.session_state[resolved_key] = action.replace("_", " ")
                    st.rerun()


def render_drift_panel(message: dict[str, Any], safe_msg_id: str) -> None:
    """Render exactly one drift expander per draft. Four heuristics:
    1. low_confidence  2. oscillation  3. persistent_failure  4. tradeoff

    Drift panels are *anchored* to their draft: once we render one for a
    given message_id, we stash it in session_state under `_drift_anchor_{mid}`
    and keep rendering from that anchor even if `message["draft_drift"]`
    gets cleared or overwritten on a later rerun (which can happen when the
    grade-poll fragment races with a button click, or when the conversation
    is reloaded from the DB and the row's drift_json comes back differently).

    Past drafts (those that aren't the latest graded draft) never show a
    panel -- the panel is only actionable for the current state, and a
    stale oscillation/persistent_failure panel on an old draft is confusing
    once the dim has stabilized. The colored dots on each past draft already
    capture the historical record; only the latest draft owns the panel.
    """
    cfg = load_grading_config()
    if not cfg.get("show_drift_panels"):
        return

    # Gate: only the latest graded draft may show a drift panel. If this
    # message isn't the latest, bail -- regardless of what the anchor or
    # live drift contain. We also wipe any stale anchor so it doesn't leak
    # into a future render where this message becomes latest again (it
    # won't, but be defensive).
    messages = st.session_state.get("messages") or []
    latest_draft_mid: str | None = None
    for _m in reversed(messages):
        if _m.get("role") != "assistant":
            continue
        if extract_primary_draft_text(_m.get("content") or ""):
            latest_draft_mid = str(_m.get("message_id") or "")
            break
    this_mid = str(message.get("message_id") or "")
    if latest_draft_mid is not None and this_mid != latest_draft_mid:
        st.session_state.pop(f"_drift_anchor_{safe_msg_id}", None)
        return

    anchor_key = f"_drift_anchor_{safe_msg_id}"
    live_drift = message.get("draft_drift") or {}
    live_kind = live_drift.get("kind") or "none"

    # Anchor: prefer the session-state-stored drift over whatever's currently
    # on the message. If the live drift has a real (non-none) kind, refresh
    # the anchor with it so the latest computation wins. Otherwise fall back
    # to whatever we anchored earlier.
    if live_kind != "none":
        st.session_state[anchor_key] = live_drift
        drift = live_drift
        kind = live_kind
    else:
        anchored = st.session_state.get(anchor_key)
        if isinstance(anchored, dict) and (anchored.get("kind") or "none") != "none":
            drift = anchored
            kind = anchored.get("kind") or "none"
            # Re-attach to the message so downstream code (button handlers,
            # _flush_pending_feedback lookups) sees a consistent drift bundle.
            message["draft_drift"] = drift
        else:
            return

    ek = f"drift_exp_{safe_msg_id}"
    if ek not in st.session_state:
        st.session_state[ek] = True

    # Each drift kind has a title template. Counts are computed against the
    # dimension list AND the distinct criteria those dimensions span, because
    # the `!` dots in the scoring row are one-per-criterion (if a criterion
    # has 2 NOT_MET dims, it still shows a single `!`). Without showing the
    # criterion count, users see "3 dimensions" in the panel title and 2
    # dots and think the UI is miscounting.
    _dim_entries_by_kind = {
        "low_confidence": drift.get("low_confidence_dims") or [],
        "spot_check": drift.get("spot_check_dims") or [],
        "oscillation": drift.get("oscillations") or [],
        "persistent_failure": drift.get("persistent_failure") or [],
    }
    _entries = _dim_entries_by_kind.get(kind, [])

    # Pre-filter: drop entries whose dim_id no longer exists in the active
    # rubric (the rubric may have been edited after the drift was detected
    # and persisted). Without this, the panel title counts stale entries
    # that the body then filters out -- producing a title like "2 dimensions
    # oscillating" with an empty expander.
    try:
        from rubric_writer.persistence import get_active_rubric
        _rb_for_filter, _, _ = get_active_rubric()
        _live_dim_ids_pre = {
            (d.get("id") or "").strip().lower()
            for c in (_rb_for_filter or {}).get("rubric", []) or []
            for d in c.get("dimensions") or []
            if (d.get("id") or "").strip()
        }
    except Exception:
        _live_dim_ids_pre = None
    if _live_dim_ids_pre is not None and _entries:
        _filtered_entries = [
            e for e in _entries
            if (e.get("dimension_id") or "").strip().lower() in _live_dim_ids_pre
        ]
        if len(_filtered_entries) != len(_entries):
            _log.info(
                "[drift panel pre-filter] %s: dropped %d stale dim(s)",
                kind, len(_entries) - len(_filtered_entries),
            )
        _entries = _filtered_entries
        # Also update the underlying drift record so downstream code in this
        # function (button handlers, flush, etc.) sees the filtered list.
        if kind in ("low_confidence", "spot_check", "oscillation", "persistent_failure"):
            _drift_key = {
                "low_confidence": "low_confidence_dims",
                "spot_check": "spot_check_dims",
                "oscillation": "oscillations",
                "persistent_failure": "persistent_failure",
            }[kind]
            drift[_drift_key] = _entries
        # If every entry was stale, suppress the panel entirely -- there's
        # nothing actionable to show.
        if not _entries:
            return

    _n_dims = len(_entries)
    # Oscillations don't carry a `criterion` field directly -- they're indexed
    # by dimension_id and the criterion is looked up from the draft_grade.
    # For all other kinds, the detector output includes `criterion`.
    _crit_set: set[str] = set()
    if kind == "oscillation":
        dg = message.get("draft_grade") or {}
        dim_to_crit: dict[str, str] = {}
        for c in dg.get("grades") or []:
            cname = (c.get("criterion_name") or "").strip()
            for d in c.get("dimension_grades") or []:
                did = (d.get("dimension_id") or "").strip()
                if did:
                    dim_to_crit[did] = cname
        for e in _entries:
            _crit_set.add(dim_to_crit.get(e.get("dimension_id", ""), ""))
    else:
        for e in _entries:
            _crit_set.add((e.get("criterion") or "").strip())
    _crit_set.discard("")
    _n_crits = len(_crit_set)

    def _dim_crit_label(n_dims: int, n_crits: int) -> str:
        """Render count as 'N dimension(s)' or 'N dimensions across C criteria'."""
        dim_word = "dimension" if n_dims == 1 else "dimensions"
        crit_word = "criterion" if n_crits == 1 else "criteria"
        if n_crits and n_crits != n_dims:
            return f"{n_dims} {dim_word} across {n_crits} {crit_word}"
        return f"{n_dims} {dim_word}"

    _count_label = _dim_crit_label(_n_dims, _n_crits)
    title = {
        "low_confidence": f"🔍 System uncertain on {_count_label}",
        "oscillation": f"〰️ {_count_label} oscillating",
        "persistent_failure": f"🔄 {_count_label} consistently not met",
        "tradeoff": "⚖️ Rubric tradeoff detected",
        "spot_check": f"✅ Quick check: {_count_label}",
    }.get(kind, "Rubric drift")

    with st.expander(title, expanded=st.session_state.get(ek, True)):

        # --- 1. Low confidence ---
        if kind == "low_confidence":
            low_conf = drift.get("low_confidence_dims") or []
            total_lc = len(low_conf)
            resolved_lc = sum(1 for k in range(total_lc)
                              if st.session_state.get(f"lc_resolved_{safe_msg_id}_{k}"))
            st.caption(
                f"Does your draft meet this? **Please respond to all {total_lc} "
                f"dimension(s) to receive a rubric suggestion.** "
                f"_({resolved_lc}/{total_lc} reviewed)_"
            )
            _render_low_confidence_clarifications(low_conf, safe_msg_id, message)

        # --- 2. Oscillation ---
        elif kind == "oscillation":
            oscs = drift.get("oscillations") or []
            dg = message.get("draft_grade") or {}
            dim_to_crit: dict[str, str] = {}
            for c in dg.get("grades") or []:
                cname = (c.get("criterion_name") or "").strip()
                for d in c.get("dimension_grades") or []:
                    did = (d.get("dimension_id") or "").strip()
                    if did:
                        dim_to_crit[did] = cname

            # NOTE: Stale dim_ids (dims no longer in the active rubric) are
            # already filtered out by the pre-filter step above, before the
            # title/count was computed. No per-kind filtering needed here.
            total_osc = len(oscs)
            resolved_osc = sum(1 for j in range(total_osc)
                               if st.session_state.get(f"osc_resolved_{safe_msg_id}_{j}"))
            panel_id_osc = f"osc_{safe_msg_id}"
            st.caption(
                f"The following dimensions have been going back and forth across your "
                f"recent drafts. What's going on? **Please respond to all {total_osc} "
                f"dimension(s) to receive a rubric suggestion.** "
                f"_({resolved_osc}/{total_osc} reviewed)_"
            )
            # One-time legend for the three action buttons below.
            st.markdown(
                "<div style='font-size:11px;color:#666;line-height:1.5;"
                "margin:4px 0 8px 0;padding:6px 10px;background:#f6f6f6;"
                "border-left:3px solid #bbb;border-radius:3px;'>"
                "<b>🔀 Subjective wording</b> — rubric language is interpretation-dependent; let's operationalize it.<br>"
                "<b>✅ No rubric change</b> — the rubric is fine; tell us why and we'll skip the edit.<br>"
                "<b>🗑 Remove</b> — delete this dimension from the rubric."
                "</div>",
                unsafe_allow_html=True,
            )
            for j, o in enumerate(oscs):
                did = o.get("dimension_id", "")
                # Resolve criterion: prefer current draft_grade's mapping,
                # fall back to active-rubric lookup. Handles the cases
                # where the user manually edited the rubric mid-session
                # (added/removed dim, renamed criterion).
                crit = _resolve_crit_for_dim(did, dim_to_crit.get(did, ""), message)
                dim_desc = _lookup_dimension_description(did, crit)
                dim_label = dim_desc if dim_desc else did
                hist_str = " -> ".join(o.get("history") or [])

                resolved_key = f"osc_resolved_{safe_msg_id}_{j}"
                if st.session_state.get(resolved_key):
                    st.caption(f"~~{html_lib.escape(dim_label)}~~ _({st.session_state[resolved_key]})_")
                    continue

                st.markdown(f"_{html_lib.escape(crit)}_: **{html_lib.escape(dim_label)}**")
                st.caption(f"Grade history: {hist_str}")

                # Oscillation buttons.
                # Three buttons; middle one stages a sub-radio so the user
                # can say "no rubric change needed" with a reason. Two
                # reasons are tracked separately for §4.3 paper analysis:
                #   - rubric_fine       (the wording is fine as-is)
                #   - drafts_varying    (the flip reflects real draft
                #                        variation, not a rubric flaw)
                #
                # All three buttons short-circuit the refiner except the
                # first (Subjective wording), which queues an edit.
                #
                # Downstream action map (action slug → behavior):
                #   - wording_subjective → refiner edit
                #   - rubric_fine        → short-circuit, no rubric change
                #   - drafts_varying     → short-circuit, no rubric change
                #   - remove             → short-circuit, delete the dim
                btn_base = f"osc_{safe_msg_id}_{j}"
                pending_noedit_key = f"osc_pending_noedit_{safe_msg_id}_{j}"
                pending_noedit = st.session_state.get(pending_noedit_key, False)

                # Helper to finalize an action: log telemetry, queue
                # feedback for the refiner short-circuit / refinement
                # path, mark resolved, flush if all dims are done.
                def _osc_finalize_action(action: str, semantic: str) -> None:
                    pre_grade = json.loads(json.dumps(message.get("draft_grade"))) if message.get("draft_grade") else None
                    _store_user_verdict(message, crit, did, action)
                    # Use `did` (real dim_id), not `dim_label` -- the
                    # refiner-resolver regex matches against dim_id.
                    feedback = f"Dimension '{did}' ({crit}) oscillates ({o.get('flips', 0)} flips): user says {action}. {semantic}"
                    log_confirmation(
                        source="drift_detected",
                        draft_index=(message.get("draft_grade_meta") or {}).get("draft_index", 0),
                        drift_type="oscillation", dimension_id=did,
                        dimension_text=dim_label, grader_verdict="oscillating",
                        grader_confidence="high", user_response_raw=action,
                        # rubric_fine is the closest analog of the old
                        # "just_right" → "user agrees the rubric is OK."
                        user_confirms_grader=(action == "rubric_fine"),
                    )
                    _queue_pending_feedback(
                        panel_id=panel_id_osc,
                        feedback_text=feedback,
                        drift_kind="oscillation",
                        draft_grade=pre_grade,
                        draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                    )
                    sb = st.session_state.get("supabase")
                    pid = st.session_state.get("current_project_id")
                    if sb and pid:
                        try:
                            save_project_data(sb, pid, "oscillation_feedback", {
                                "timestamp": datetime.now().isoformat(),
                                "conversation_id": st.session_state.get("selected_conversation"),
                                "message_id": message.get("message_id"),
                                "dimension_id": did, "criterion": crit,
                                "action": action,
                            })
                        except Exception:
                            pass
                    st.session_state[resolved_key] = action.replace("_", " ")
                    new_resolved = sum(1 for k in range(total_osc)
                                       if st.session_state.get(f"osc_resolved_{safe_msg_id}_{k}"))
                    if new_resolved == total_osc:
                        _defer_flush(panel_id_osc)

                if not pending_noedit:
                    # Three top-level buttons.
                    cols = st.columns(3)
                    with cols[0]:
                        if st.button("🔀 Subjective wording",
                                     key=f"{btn_base}_wording_subjective",
                                     use_container_width=True):
                            _osc_finalize_action(
                                "wording_subjective",
                                "Rubric language leaves room for interpretation — let's operationalize it.",
                            )
                            st.rerun()
                    with cols[1]:
                        if st.button("✅ No rubric change",
                                     key=f"{btn_base}_no_change",
                                     use_container_width=True):
                            st.session_state[pending_noedit_key] = True
                            st.rerun()
                    with cols[2]:
                        if st.button("🗑 Remove",
                                     key=f"{btn_base}_remove",
                                     use_container_width=True):
                            _osc_finalize_action(
                                "remove",
                                "Delete this dimension from the rubric.",
                            )
                            st.rerun()
                else:
                    # Pending "no rubric change" state: radio + Submit/Cancel.
                    # Radio answers WHY no change is needed -- this is the
                    # paper's §4.3 reason-code distribution at point of
                    # capture, not derived post-hoc.
                    reason_radio_key = f"osc_noedit_reason_{safe_msg_id}_{j}"
                    _NOEDIT_OPTIONS = [
                        ("rubric_fine",
                         "The rubric wording is fine — leave it as-is.",
                         "Current wording is fine; dismiss."),
                        ("drafts_varying",
                         "My drafts are varying — the flip reflects real draft variation, not a rubric flaw.",
                         "The flip reflects real variation in my drafts, not a rubric flaw."),
                    ]
                    _values = [v for v, _, _ in _NOEDIT_OPTIONS]
                    _labels = [lbl for _, lbl, _ in _NOEDIT_OPTIONS]
                    _semantics = {v: s for v, _, s in _NOEDIT_OPTIONS}
                    _picked_label = st.radio(
                        "Which best describes why no rubric change is needed?",
                        options=_labels,
                        index=None,
                        key=reason_radio_key,
                    )
                    _picked_value = (_values[_labels.index(_picked_label)]
                                     if _picked_label in _labels else None)
                    col_sub, col_cancel = st.columns(2)
                    with col_sub:
                        if st.button("Submit",
                                     key=f"{btn_base}_no_change_submit",
                                     type="primary",
                                     disabled=(_picked_value is None),
                                     use_container_width=True):
                            _osc_finalize_action(
                                _picked_value,
                                _semantics.get(_picked_value, ""),
                            )
                            st.session_state.pop(pending_noedit_key, None)
                            st.rerun()
                    with col_cancel:
                        if st.button("Cancel",
                                     key=f"{btn_base}_no_change_cancel",
                                     use_container_width=True):
                            st.session_state.pop(pending_noedit_key, None)
                            st.rerun()

        # --- 3. Persistent failure ---
        elif kind == "persistent_failure":
            dims = drift.get("persistent_failure") or []
            total_pf = len(dims)
            resolved_pf = sum(1 for k in range(total_pf)
                              if st.session_state.get(f"pf_resolved_{safe_msg_id}_{k}"))
            panel_id_pf = f"pf_{safe_msg_id}"
            st.caption(
                f"The following dimensions have been marked as not met across your last few "
                f"drafts. What do you think? **Please respond to all {total_pf} dimension(s) "
                f"to receive a rubric suggestion.** _({resolved_pf}/{total_pf} reviewed)_"
            )
            for j, dim_info in enumerate(dims):
                did = dim_info.get("dimension_id", "")
                crit = _resolve_crit_for_dim(did, dim_info.get("criterion", ""), message)
                streak = dim_info.get("streak", 0)
                dim_desc = _lookup_dimension_description(did, crit)
                dim_label = dim_desc if dim_desc else did

                resolved_key = f"pf_resolved_{safe_msg_id}_{j}"
                if st.session_state.get(resolved_key):
                    st.caption(f"~~{html_lib.escape(dim_label)}~~ _({st.session_state[resolved_key]})_")
                    continue

                st.markdown(f"_{html_lib.escape(crit)}_: **{html_lib.escape(dim_label)}**")

                col1, col2, col3 = st.columns(3)
                btn_base = f"pf_{safe_msg_id}_{j}"

                def _pf_finalize(resolved_value: str):
                    st.session_state[resolved_key] = resolved_value
                    new_resolved = sum(1 for k in range(total_pf)
                                       if st.session_state.get(f"pf_resolved_{safe_msg_id}_{k}"))
                    if new_resolved == total_pf:
                        _defer_flush(panel_id_pf)

                with col1:
                    if st.button("👍 Agree, I'm working on it", key=f"{btn_base}_agree"):
                        _store_user_verdict(message, crit, did, "working_on_it")
                        log_confirmation(
                            source="drift_detected",
                            draft_index=(message.get("draft_grade_meta") or {}).get("draft_index", 0),
                            drift_type="persistent_failure", dimension_id=did,
                            dimension_text=dim_label, grader_verdict="NOT_MET",
                            grader_confidence="high", user_response_raw="working_on_it",
                            user_confirms_grader=True,
                        )
                        _pf_finalize("working on it")
                        st.rerun()
                with col2:
                    if st.button("❌ Disagree, I think this is met", key=f"{btn_base}_disagree"):
                        pre_grade = json.loads(json.dumps(message.get("draft_grade"))) if message.get("draft_grade") else None
                        _store_user_verdict(message, crit, did, "MET")
                        # Use `did` (real dim_id), not `dim_label`. `_resolve_refiner_inputs`
                        # extracts the hint via a `Dimension '([^']+)'` regex and matches
                        # against the grade payload's `dimension_id` field -- grader payloads
                        # don't reliably carry `label`, so using the human label here causes
                        # the resolver to fall through to "first NOT_MET" and act on the
                        # wrong dim (or fail silently).
                        feedback = f"Dimension '{did}' ({crit}): grader says NOT_MET for {streak} drafts, user disagrees and says it's MET."
                        _queue_pending_feedback(
                            panel_id=panel_id_pf,
                            feedback_text=feedback,
                            drift_kind="persistent_failure",
                            draft_grade=pre_grade,
                            draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                        )
                        log_confirmation(
                            source="drift_detected",
                            draft_index=(message.get("draft_grade_meta") or {}).get("draft_index", 0),
                            drift_type="persistent_failure", dimension_id=did,
                            dimension_text=dim_label, grader_verdict="NOT_MET",
                            grader_confidence="high", user_response_raw="disagree_met",
                            user_confirms_grader=False,
                        )
                        _pf_finalize("disagree, is met")
                        st.rerun()
                with col3:
                    if st.button("🗑 Remove", key=f"{btn_base}_remove"):
                        pre_grade = json.loads(json.dumps(message.get("draft_grade"))) if message.get("draft_grade") else None
                        _store_user_verdict(message, crit, did, "remove")
                        # Use `did` (real dim_id). `_flush_pending_feedback` routes
                        # removes through `_resolve_refiner_inputs` to get
                        # (criterion_name, dimension_id), and the regex only finds
                        # the id if we put it in the feedback text here. With
                        # `dim_label` the resolver fell through to a different
                        # NOT_MET dim and removed the wrong one -- or couldn't
                        # match at all and silently no-opped the remove.
                        feedback = f"Dimension '{did}' ({crit}): user wants to remove this dimension."
                        _queue_pending_feedback(
                            panel_id=panel_id_pf,
                            feedback_text=feedback,
                            drift_kind="persistent_failure",
                            draft_grade=pre_grade,
                            draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                        )
                        log_confirmation(
                            source="drift_detected",
                            draft_index=(message.get("draft_grade_meta") or {}).get("draft_index", 0),
                            drift_type="persistent_failure", dimension_id=did,
                            dimension_text=dim_label, grader_verdict="NOT_MET",
                            grader_confidence="high", user_response_raw="remove",
                            user_confirms_grader=False,
                        )
                        _pf_finalize("removed")
                        st.rerun()

        # --- 4. Tradeoff ---
        elif kind == "tradeoff":
            ti = drift.get("tradeoff_improvements") or []
            td = drift.get("tradeoff_drops") or []
            st.caption("Some criteria improved while others slipped. Which matters more to you?")

            # Build dimension detail from current draft_grade
            dg = message.get("draft_grade") or {}
            crit_dims: dict[str, list[dict[str, Any]]] = {}
            for c in dg.get("grades") or []:
                cname = (c.get("criterion_name") or "").strip()
                for d in c.get("dimension_grades") or []:
                    did = (d.get("dimension_id") or "").strip()
                    dim_desc = _lookup_dimension_description(did, cname)
                    dim_label = dim_desc if dim_desc else did
                    grade = (d.get("grade") or "").upper()
                    crit_dims.setdefault(cname, []).append({
                        "label": dim_label, "grade": grade,
                    })

            if ti:
                st.markdown("**Improved**")
                for x in ti:
                    cname = str(x.get("name", ""))
                    st.markdown(f"⬆ _{html_lib.escape(cname)}_ ({x.get('score_str_prev', '')} -> {x.get('score_str_curr', '')})")
                    for d in crit_dims.get(cname, []):
                        icon = "✅" if d["grade"] == "MET" else "❌"
                        st.caption(f"  {icon} {html_lib.escape(d['label'])}")
            if td:
                st.markdown("**Slipped**")
                for x in td:
                    cname = str(x.get("name", ""))
                    st.markdown(f"⬇ _{html_lib.escape(cname)}_ ({x.get('score_str_prev', '')} -> {x.get('score_str_curr', '')})")
                    for d in crit_dims.get(cname, []):
                        icon = "✅" if d["grade"] == "MET" else "❌"
                        st.caption(f"  {icon} {html_lib.escape(d['label'])}")
            # Pairwise priority buttons
            if ti and td:
                for j, (imp, drp) in enumerate([(i, d) for i in ti for d in td]):
                    imp_name = str(imp.get("name", ""))
                    drp_name = str(drp.get("name", ""))
                    resolved_key = f"tradeoff_resolved_{safe_msg_id}_{j}"
                    if st.session_state.get(resolved_key):
                        st.caption(f"_({st.session_state[resolved_key]})_")
                        continue
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"⬆ Prioritize {imp_name}", key=f"trade_imp_{safe_msg_id}_{j}"):
                            _store_user_verdict(message, drp_name, "", "deprioritize")
                            _defer_schedule_refinement(
                                feedback_text=f"Tradeoff: user prioritizes '{imp_name}' over '{drp_name}'.",
                                drift_kind="tradeoff",
                                draft_grade=message.get("draft_grade"),
                                draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                            )
                            st.session_state[resolved_key] = f"prioritized {imp_name}"
                            st.rerun()
                    with col2:
                        if st.button(f"⬆ Prioritize {drp_name}", key=f"trade_drp_{safe_msg_id}_{j}"):
                            _store_user_verdict(message, imp_name, "", "deprioritize")
                            _defer_schedule_refinement(
                                feedback_text=f"Tradeoff: user prioritizes '{drp_name}' over '{imp_name}'.",
                                drift_kind="tradeoff",
                                draft_grade=message.get("draft_grade"),
                                draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                            )
                            st.session_state[resolved_key] = f"prioritized {drp_name}"
                            st.rerun()
                    with col3:
                        if st.button("⚖ Both matter", key=f"trade_both_{safe_msg_id}_{j}"):
                            _defer_schedule_refinement(
                                feedback_text=f"Tradeoff between '{imp_name}' and '{drp_name}': user says both matter equally.",
                                drift_kind="tradeoff",
                                draft_grade=message.get("draft_grade"),
                                draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                            )
                            st.session_state[resolved_key] = "both matter"
                            st.rerun()

        # --- 5. Spot check ---
        elif kind == "spot_check":
            spot_dims = drift.get("spot_check_dims") or []
            total_sc = len(spot_dims)
            resolved_sc = sum(1 for k in range(total_sc)
                              if st.session_state.get(f"spot_resolved_{safe_msg_id}_{k}"))
            panel_id_sc = f"spot_{safe_msg_id}"
            st.caption(
                f"Does your draft meet these? **Please respond to all {total_sc} "
                f"dimension(s) to receive a rubric suggestion if needed.** "
                f"_({resolved_sc}/{total_sc} reviewed)_"
            )
            for j, sc in enumerate(spot_dims):
                did = sc.get("dimension_id", "")
                crit = _resolve_crit_for_dim(did, sc.get("criterion", ""), message)
                grade = sc.get("grade", "")
                evidence = sc.get("evidence", "")
                dim_desc = _lookup_dimension_description(did, crit)
                dim_label = dim_desc if dim_desc else did

                resolved_key = f"spot_resolved_{safe_msg_id}_{j}"
                if st.session_state.get(resolved_key):
                    st.caption(f"~~{html_lib.escape(dim_label)}~~ _({st.session_state[resolved_key]})_")
                    continue

                st.markdown(f"_{html_lib.escape(crit)}_: **{html_lib.escape(dim_label)}**")

                btn_base = f"spot_{safe_msg_id}_{j}"
                grade_label = "MET" if grade == "MET" else "NOT MET"
                st.caption(f"Grader says: {grade_label}")
                if evidence:
                    st.caption(f"Reasoning: _{html_lib.escape(evidence)}_")

                spot_meta = message.get("draft_grade_meta") or {}

                def _sc_finalize():
                    new_resolved = sum(1 for k in range(total_sc)
                                       if st.session_state.get(f"spot_resolved_{safe_msg_id}_{k}"))
                    if new_resolved == total_sc:
                        _defer_flush(panel_id_sc)

                # "No" click flow: first click stages a pending-NO state that
                # reveals a reason text input + Submit button. Second click
                # (Submit) actually resolves the dim and queues the feedback.
                # Yes click resolves immediately.
                pending_no_key = f"spot_pending_no_{safe_msg_id}_{j}"
                reason_key = f"spot_reason_{safe_msg_id}_{j}"

                pending_no = st.session_state.get(pending_no_key, False)

                if not pending_no:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes, it meets this", key=f"{btn_base}_yes"):
                            _store_user_verdict(message, crit, did, "MET")
                            log_confirmation(
                                source="spot_check", draft_index=spot_meta.get("draft_index", 0),
                                drift_type="low_confidence", dimension_id=did,
                                dimension_text=dim_label, grader_verdict=grade,
                                grader_confidence=sc.get("confidence", "high"),
                                user_response_raw="yes",
                                user_confirms_grader=(grade == "MET"),
                            )
                            if grade != "MET":
                                # `_resolve_refiner_inputs` extracts the dim
                                # identifier via a `Dimension '([^']+)'` regex
                                # and matches against dim `id` (not label)
                                # because grader payloads don't carry labels.
                                # So we MUST use `did` here, not `dim_label`,
                                # or the refiner silently bails.
                                _queue_pending_feedback(
                                    panel_id=panel_id_sc,
                                    feedback_text=f"Dimension '{did}' ({crit}): spot check — grader said {grade}, user says MET. Silent misalignment.",
                                    drift_kind="spot_check",
                                    draft_grade=message.get("draft_grade"),
                                    draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                                )
                            sb = st.session_state.get("supabase")
                            pid = st.session_state.get("current_project_id")
                            if sb and pid:
                                try:
                                    save_project_data(sb, pid, "spot_check_feedback", {
                                        "timestamp": datetime.now().isoformat(),
                                        "conversation_id": st.session_state.get("selected_conversation"),
                                        "message_id": message.get("message_id"),
                                        "dimension_id": did, "criterion": crit,
                                        "grader_grade": grade, "user_grade": "MET",
                                        "misaligned": grade != "MET",
                                    })
                                except Exception:
                                    pass
                            st.session_state[resolved_key] = "MET" + (" (misaligned)" if grade != "MET" else "")
                            _sc_finalize()
                            st.rerun()
                    with col2:
                        if st.button("❌ No, it doesn't", key=f"{btn_base}_no"):
                            # Don't resolve yet -- reveal the reason input.
                            st.session_state[pending_no_key] = True
                            st.rerun()
                else:
                    # Pending-NO state: explicit reason code radio + free-text
                    # box + Submit / Cancel. Reason codes per paper §3.2 -
                    # captures categorical "why" so §4.3 can report a
                    # distribution. Spot_check NO is the case where post-hoc
                    # button-implied reasons are weakest (the spot_check
                    # panel doesn't have multiple buttons to discriminate),
                    # so this is where explicit codes add the most value.
                    reason_code_key = f"spot_reason_code_{safe_msg_id}_{j}"
                    _SPOT_NO_REASON_OPTIONS = [
                        ("draft_doesnt_do_this",
                         "The draft doesn't actually do this (grader missed something present)"),
                        ("dim_wording_confusing",
                         "The dimension wording is confusing or ambiguous"),
                        ("dim_not_important",
                         "I don't really care about this dimension"),
                        ("other",
                         "Other (use the text box)"),
                    ]
                    _option_values = [v for v, _ in _SPOT_NO_REASON_OPTIONS]
                    _option_labels = [lbl for _, lbl in _SPOT_NO_REASON_OPTIONS]
                    _picked_label = st.radio(
                        "Which best describes why? (helps qualitative analysis)",
                        options=_option_labels,
                        index=None,
                        key=reason_code_key,
                    )
                    _picked_value = (_option_values[_option_labels.index(_picked_label)]
                                     if _picked_label in _option_labels else None)
                    # Natural-language label for the refiner (without the
                    # parenthetical hint, which is meant for the user, not
                    # the model). Stays as the raw code in the saved
                    # spot_check_feedback row for clean audit-script
                    # filtering -- only the refiner-bound feedback_text uses
                    # the prose form.
                    _NL_LABEL_MAP = {
                        "draft_doesnt_do_this": "the draft doesn't actually do this; the grader missed that the dimension isn't met by the draft",
                        "dim_wording_confusing": "the dimension wording is confusing or ambiguous",
                        "dim_not_important": "the user doesn't care about this dimension as much",
                        "other": "the user has a different reason; see free-text below",
                    }
                    _picked_natural = _NL_LABEL_MAP.get(_picked_value or "")
                    st.text_input(
                        "Anything else? (optional — your reasoning helps the refiner suggest a better rubric edit)",
                        key=reason_key,
                        placeholder="e.g. the draft doesn't actually explain what 'clearly' means here",
                    )
                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        if st.button("Submit feedback", key=f"{btn_base}_no_submit", type="primary"):
                            _store_user_verdict(message, crit, did, "NOT_MET")
                            log_confirmation(
                                source="spot_check", draft_index=spot_meta.get("draft_index", 0),
                                drift_type="low_confidence", dimension_id=did,
                                dimension_text=dim_label, grader_verdict=grade,
                                grader_confidence=sc.get("confidence", "high"),
                                user_response_raw="no",
                                user_confirms_grader=(grade == "NOT_MET"),
                            )
                            user_reason = (st.session_state.get(reason_key, "") or "").strip()
                            if grade != "NOT_MET":
                                # Use `did` (real dim_id), not `dim_label`.
                                # See note on the YES branch above.
                                # Build a NATURAL-LANGUAGE reason for the
                                # refiner. The raw `reason_code` slug stays
                                # in the saved spot_check_feedback row for
                                # audit, but the refiner sees prose.
                                reason_parts = []
                                if _picked_natural:
                                    reason_parts.append(_picked_natural)
                                if user_reason:
                                    reason_parts.append(f"in their words: {user_reason}")
                                reason_suffix = (
                                    f" User reason: {'; '.join(reason_parts)}"
                                    if reason_parts else ""
                                )
                                _queue_pending_feedback(
                                    panel_id=panel_id_sc,
                                    feedback_text=f"Dimension '{did}' ({crit}): spot check — grader said {grade}, user says NOT_MET. Silent misalignment.{reason_suffix}",
                                    drift_kind="spot_check",
                                    draft_grade=message.get("draft_grade"),
                                    draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
                                )
                            sb = st.session_state.get("supabase")
                            pid = st.session_state.get("current_project_id")
                            if sb and pid:
                                try:
                                    save_project_data(sb, pid, "spot_check_feedback", {
                                        "timestamp": datetime.now().isoformat(),
                                        "conversation_id": st.session_state.get("selected_conversation"),
                                        "message_id": message.get("message_id"),
                                        "dimension_id": did, "criterion": crit,
                                        "grader_grade": grade, "user_grade": "NOT_MET",
                                        "misaligned": grade != "NOT_MET",
                                        "user_reason": user_reason,
                                        "reason_code": _picked_value,
                                    })
                                except Exception:
                                    pass
                            st.session_state[resolved_key] = "NOT_MET" + (" (misaligned)" if grade != "NOT_MET" else "")
                            st.session_state.pop(pending_no_key, None)
                            _sc_finalize()
                            st.rerun()
                    with col_cancel:
                        if st.button("Cancel", key=f"{btn_base}_no_cancel"):
                            st.session_state.pop(pending_no_key, None)
                            st.rerun()

    # Deferred refiner work is NOT run here. Running it inside any single
    # panel's expander would put spinners inside that panel, which is what
    # we're trying to avoid. Instead, tab_chat calls `_run_deferred_refiner_work`
    # once AFTER the whole chat message loop, so refiner output appears in a
    # stable location below all messages.


def _heatmap_color(pct: float | None) -> str:
    """Map a 0–1 score to a background color for the heatmap grid."""
    if pct is None:
        return "#e0e0e0"
    if pct >= 0.999:
        return "#2e7d32"
    if pct >= 0.75:
        return "#66bb6a"
    if pct >= 0.5:
        return "#ffa726"
    if pct >= 0.25:
        return "#ef5350"
    return "#c62828"


def render_heatmap_grid(messages: list[dict[str, Any]]) -> None:
    """Render a dimension-level heatmap: rows = criteria, columns = draft iterations."""
    graded = [m for m in messages if m.get("role") == "assistant" and m.get("draft_grade")]
    if len(graded) < 2:
        return

    # Collect criterion names in priority order from the latest draft
    crit_names: list[str] = []
    for c in sorted(
        (graded[-1].get("draft_grade") or {}).get("grades") or [],
        key=lambda x: (x.get("criterion_priority", 99), x.get("criterion_name") or ""),
    ):
        n = (c.get("criterion_name") or "").strip()
        if n and n not in crit_names:
            crit_names.append(n)

    if not crit_names:
        return

    # Build score matrix + detect patterns per criterion
    scores: dict[str, list[float | None]] = {cn: [] for cn in crit_names}
    for m in graded:
        dg = m.get("draft_grade") or {}
        by_name = {
            (c.get("criterion_name") or "").strip(): c for c in dg.get("grades") or []
        }
        for cn in crit_names:
            c = by_name.get(cn)
            scores[cn].append(parse_score_pct(c.get("score")) if c else None)

    num_drafts = len(graded)
    # Limit columns to last 10 drafts for readability
    max_cols = 10
    start = max(0, num_drafts - max_cols)

    # Detect patterns for row annotations
    def _pattern_label(vals: list[float | None]) -> str:
        clean = [v for v in vals if v is not None]
        if len(clean) < 3:
            return ""
        last3 = clean[-3:]
        # Oscillation: alternating up/down
        if len(clean) >= 3:
            dirs = []
            for i in range(1, len(clean)):
                if clean[i] > clean[i - 1] + 0.01:
                    dirs.append("u")
                elif clean[i] < clean[i - 1] - 0.01:
                    dirs.append("d")
                else:
                    dirs.append("=")
            flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1] and dirs[i] != "=" and dirs[i - 1] != "=")
            if flips >= 2:
                return " ~"
        # Plateau
        if len(set(round(v, 3) for v in last3)) == 1 and last3[0] < 0.999:
            return " ="
        # Monotonic decline
        if len(clean) >= 4 and all(clean[i] > clean[i + 1] for i in range(len(clean) - 4, len(clean) - 1)):
            return " ↘"
        return ""

    # Build HTML table
    html_parts = [
        '<div style="margin:8px 0;overflow-x:auto;">',
        '<table style="border-collapse:collapse;font-size:11px;width:100%;">',
        '<tr><th style="text-align:left;padding:2px 8px 2px 0;color:#666;font-weight:normal;"></th>',
    ]
    for i in range(start, num_drafts):
        html_parts.append(
            f'<th style="padding:2px 4px;color:#888;font-weight:normal;min-width:24px;text-align:center;">'
            f'{i + 1}</th>'
        )
    html_parts.append("</tr>")

    for cn in crit_names:
        row_scores = scores[cn][start:]
        pattern = _pattern_label(scores[cn])
        html_parts.append(
            f'<tr><td style="padding:2px 8px 2px 0;white-space:nowrap;color:#444;max-width:140px;'
            f'overflow:hidden;text-overflow:ellipsis;" title="{html_lib.escape(cn)}">'
            f'{html_lib.escape(cn[:20])}{html_lib.escape(pattern)}</td>'
        )
        for pct in row_scores:
            bg = _heatmap_color(pct)
            label = f"{pct:.0%}" if pct is not None else "–"
            text_color = "#fff" if pct is not None and pct < 0.75 else "#fff" if pct is not None else "#999"
            html_parts.append(
                f'<td style="padding:2px 4px;text-align:center;background:{bg};color:{text_color};'
                f'border-radius:3px;border:1px solid rgba(255,255,255,0.3);min-width:24px;"'
                f' title="{html_lib.escape(cn)}: {label}">{label}</td>'
            )
        html_parts.append("</tr>")

    html_parts.append("</table></div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    st.caption(
        "Pattern indicators: **~** oscillating &nbsp; **!** persistent failure"
    )


def _store_user_verdict(
    message: dict[str, Any],
    criterion_name: str,
    dim_id: str,
    user_grade: str,
) -> None:
    """Store the user's verdict on a dimension without mutating the grader's
    draft_grade. The grader's original assessment stays untouched as ground
    truth. User verdicts are stored separately for research and display."""
    verdicts = message.get("user_verdicts") or {}
    verdicts[f"{criterion_name}::{dim_id}"] = {
        "criterion": criterion_name,
        "dimension_id": dim_id,
        "user_grade": user_grade,
    }
    message["user_verdicts"] = verdicts


def _save_low_confidence_feedback(
    message: dict[str, Any],
    dim_id: str,
    criterion: str,
    action: str,
    grade: str,
    ambiguity_note: str,
    panel_id: str | None = None,
) -> None:
    """Persist a user's low-confidence clarification response, apply the grade
    override in memory, and QUEUE rubric refinement (to be flushed when the
    panel is fully resolved). If panel_id is None, fires refinement immediately
    (legacy behavior)."""
    # Snapshot grades BEFORE override so refinement sees the disagreement
    pre_override_grade = json.loads(json.dumps(message.get("draft_grade"))) if message.get("draft_grade") else None

    # Apply the override to the message's draft_grade
    if action == "grade_correct":
        _store_user_verdict(message, criterion, dim_id, grade)
        correction_text = (
            f"Dimension '{dim_id}' ({criterion}): grader was uncertain but the grade "
            f"{grade} is correct. Ambiguity note was: {ambiguity_note}"
        )
    elif action.startswith("grade_flipped_to_"):
        new_grade = action.replace("grade_flipped_to_", "")
        _store_user_verdict(message, criterion, dim_id, new_grade)
        correction_text = (
            f"Dimension '{dim_id}' ({criterion}): grader said {grade} but user says "
            f"it's actually {new_grade}. Ambiguity note was: {ambiguity_note}"
        )
    else:
        correction_text = ""

    # Persist to database
    sb = st.session_state.get("supabase")
    pid = st.session_state.get("current_project_id")
    if not sb or not pid:
        st.error("Not signed in or no project -- feedback was not saved.")
        return
    meta = message.get("draft_grade_meta") or {}
    payload = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": st.session_state.get("selected_conversation"),
        "message_id": message.get("message_id"),
        "draft_index": meta.get("draft_index"),
        "dimension_id": dim_id,
        "criterion": criterion,
        "original_grade": grade,
        "ambiguity_note": ambiguity_note,
        "action": action,
    }
    try:
        save_project_data(sb, pid, "low_confidence_feedback", payload)
    except Exception:
        st.error("Could not save feedback.")
        return

    # Queue or fire refinement
    if correction_text:
        if panel_id:
            _queue_pending_feedback(
                panel_id=panel_id,
                feedback_text=correction_text,
                drift_kind="low_confidence",
                draft_grade=pre_override_grade,
                draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
            )
        else:
            _defer_schedule_refinement(
                feedback_text=correction_text,
                drift_kind="low_confidence",
                draft_grade=pre_override_grade,
                draft_excerpt=extract_primary_draft_text(message.get("content") or "") or "",
            )


def _resolve_crit_for_dim(
    dim_id: str,
    fallback_crit: str = "",
    message: dict | None = None,
) -> str:
    """Resolve a dimension's criterion name when the persisted drift bundle's
    `criterion` field is empty or stale (e.g. user manually edited the rubric
    mid-session: removed/added a dim, renamed a criterion). Returns the live
    criterion name from the current active rubric, or falls back to the
    given `fallback_crit`, or to the dim's criterion in the supplied
    message's draft_grade payload, or `""` if nothing resolves.

    The drift panel render uses this so a stale entry no longer renders as
    `__: <raw_dim_id>` when the criterion can be re-derived.
    """
    target = (dim_id or "").strip().lower()
    if not target:
        return (fallback_crit or "").strip()
    # 1. Active rubric (authoritative for current state).
    try:
        from rubric_writer.persistence import get_active_rubric
        _rb, _, _ = get_active_rubric()
        for _c in (_rb or {}).get("rubric", []) or []:
            for _d in _c.get("dimensions") or []:
                if (_d.get("id") or "").strip().lower() == target:
                    return (_c.get("name") or "").strip()
    except Exception:
        pass
    # 2. Message's draft_grade payload (was graded against the rubric at the
    #    time of grading; criterion_name fields are usually still accurate).
    if message:
        try:
            dg = message.get("draft_grade") or {}
            for _c in dg.get("grades") or []:
                for _d in _c.get("dimension_grades") or []:
                    if (_d.get("dimension_id") or "").strip().lower() == target:
                        return (_c.get("criterion_name") or "").strip()
        except Exception:
            pass
    # 3. Provided fallback (the original `criterion` field on the drift
    #    entry, if it was non-empty).
    return (fallback_crit or "").strip()


def _lookup_dimension_description(dim_id: str, criterion_name: str) -> str:
    """Look up a dimension's human-readable label from the active rubric.

    Resolution order (each step is case- and whitespace-insensitive):
      1. Match on (criterion_name, dim_id) -- the clean case.
      2. Match on dim_id alone across every criterion -- catches the case
         where the user renamed a criterion in the Rubric Configuration
         after a drift panel's oscillation record was created; the record's
         cached criterion name is now stale but the dim_id is still live.

    Without (2) the panel renders the raw dim_id (e.g. `claim_first_framing`)
    after any manual criterion rename, which looks like a regression to the
    user even though the rubric is fine."""
    try:
        from rubric_writer.persistence import get_active_rubric
        rubric_dict, _, _ = get_active_rubric()
        if not rubric_dict:
            return ""
        target_dim = (dim_id or "").strip().lower()
        target_crit = (criterion_name or "").strip().lower()

        # Step 1: exact (crit, dim) pair.
        for crit in rubric_dict.get("rubric") or []:
            if (crit.get("name") or "").strip().lower() != target_crit:
                continue
            for dim in crit.get("dimensions") or []:
                if (dim.get("id") or "").strip().lower() == target_dim:
                    return dim.get("label") or dim.get("description") or ""

        # Step 2: dim_id-only match. Users frequently rename criteria in the
        # rubric config but dim_ids are the stable identifier.
        if target_dim:
            for crit in rubric_dict.get("rubric") or []:
                for dim in crit.get("dimensions") or []:
                    if (dim.get("id") or "").strip().lower() == target_dim:
                        return dim.get("label") or dim.get("description") or ""
    except Exception:
        pass
    return ""


def _lookup_dim_field_and_text(
    rubric_dict: dict[str, Any], criterion_name: str, dim_id: str,
) -> tuple[dict[str, Any] | None, str | None, str]:
    """Return (dim_dict, field_name, current_text) for a dimension in the rubric.

    Lookup is by dimension_id only (case-insensitive trim). Criterion name is
    used as a fallback disambiguator if dim_id is ambiguous, but dim_id alone
    is treated as authoritative because the LLM frequently rephrases criterion
    names while preserving stable dim_ids.

    field_name is the actual key holding the wording in this dim ("label" or
    "description") -- whichever has content. Returns (None, None, "") if not
    found."""
    if not rubric_dict:
        return None, None, ""
    target_dim_id = (dim_id or "").strip().lower()
    target_crit = (criterion_name or "").strip().lower()
    if not target_dim_id:
        return None, None, ""

    matches: list[tuple[dict, dict]] = []  # (crit, dim)
    for crit in rubric_dict.get("rubric") or []:
        for dim in crit.get("dimensions") or []:
            if (dim.get("id") or "").strip().lower() == target_dim_id:
                matches.append((crit, dim))

    if not matches:
        return None, None, ""

    # If multiple dim_id matches across criteria, prefer the one whose crit
    # name matches; otherwise take the first.
    chosen = matches[0]
    if len(matches) > 1 and target_crit:
        for c, d in matches:
            if (c.get("name") or "").strip().lower() == target_crit:
                chosen = (c, d)
                break

    _, dim = chosen
    if dim.get("label"):
        return dim, "label", dim["label"]
    if dim.get("description"):
        return dim, "description", dim["description"]
    return dim, "label", ""


def _render_low_confidence_clarifications(
    low_conf_dims: list[dict[str, Any]], safe_msg_id: str, message: dict[str, Any],
) -> None:
    """Render compact clarification prompts for low-confidence dimensions.

    Design: one line question + two buttons. Details are hidden behind
    an expander so the user can answer without reading a paragraph.
    """
    if not low_conf_dims:
        return
    total = len(low_conf_dims)
    panel_id_lc = f"lc_{safe_msg_id}"

    def _lc_finalize():
        new_resolved = sum(1 for k in range(total)
                           if st.session_state.get(f"lc_resolved_{safe_msg_id}_{k}"))
        if new_resolved == total:
            _defer_flush(panel_id_lc)

    for i, lc in enumerate(low_conf_dims):
        dim_id = lc.get("dimension_id", "")
        crit = _resolve_crit_for_dim(dim_id, lc.get("criterion", ""), message)
        note = lc.get("ambiguity_note", "")
        grade = lc.get("grade", "")
        evidence = lc.get("evidence", "")
        dim_desc = _lookup_dimension_description(dim_id, crit)
        dim_label = dim_desc if dim_desc else dim_id

        resolved_key = f"lc_resolved_{safe_msg_id}_{i}"
        if st.session_state.get(resolved_key):
            resolved_state = st.session_state[resolved_key]
            # resolved_state is one of:
            #   "confirmed MET" / "confirmed NOT_MET" -> user agreed with grader
            #   "flipped to MET" / "flipped to NOT_MET" -> user disagreed, flipped the grade
            final_grade = "MET" if (" MET" in f" {resolved_state}") and "NOT_MET" not in resolved_state else "NOT_MET"
            grade_emoji = "&#9989;" if final_grade == "MET" else "&#10060;"
            st.markdown(f"~~{html_lib.escape(dim_label)}~~ &nbsp; {grade_emoji} **{final_grade}**")
            continue

        st.markdown(f"_{html_lib.escape(crit)}_: **{html_lib.escape(dim_label)}**")

        btn_key_base = f"lc_{safe_msg_id}_{i}"

        if evidence:
            st.caption(f"Reasoning: _{html_lib.escape(evidence)}_")
        if note:
            st.caption(f"Uncertain because: _{html_lib.escape(note)}_")

        # Yes / No buttons
        meta = message.get("draft_grade_meta") or {}
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, it meets this", key=f"{btn_key_base}_met"):
                action = "grade_correct" if grade == "MET" else "grade_flipped_to_MET"
                _save_low_confidence_feedback(message, dim_id, crit, action, grade, note, panel_id=panel_id_lc)
                log_confirmation(
                    source="drift_detected", draft_index=meta.get("draft_index", 0),
                    drift_type="low_confidence", dimension_id=dim_id,
                    dimension_text=dim_label, grader_verdict=grade,
                    grader_confidence="low", user_response_raw="yes",
                    user_confirms_grader=(grade == "MET"),
                )
                st.session_state[resolved_key] = "confirmed MET" if grade == "MET" else "flipped to MET"
                _lc_finalize()
                st.rerun()
        with col2:
            if st.button("❌ No, it doesn't", key=f"{btn_key_base}_notmet"):
                action = "grade_correct" if grade == "NOT_MET" else "grade_flipped_to_NOT_MET"
                _save_low_confidence_feedback(message, dim_id, crit, action, grade, note, panel_id=panel_id_lc)
                log_confirmation(
                    source="drift_detected", draft_index=meta.get("draft_index", 0),
                    drift_type="low_confidence", dimension_id=dim_id,
                    dimension_text=dim_label, grader_verdict=grade,
                    grader_confidence="low", user_response_raw="no",
                    user_confirms_grader=(grade == "NOT_MET"),
                )
                st.session_state[resolved_key] = "confirmed NOT_MET" if grade == "NOT_MET" else "flipped to NOT_MET"
                _lc_finalize()
                st.rerun()




def render_rubric_scores_panel(messages: list[dict[str, Any]]) -> None:
    """Line graph of each criterion's score (% dimensions met) across drafts.

    X-axis: draft index (1-based, chronological).
    Y-axis: 0-100 % dimensions met.
    One line per criterion; hover to see the name + score."""
    cfg = load_grading_config()
    if not cfg["enabled"]:
        st.caption("Background rubric grading is disabled.")
        return

    graded = [m for m in messages if m.get("role") == "assistant" and m.get("draft_grade")]
    if not graded:
        st.info("No rubric scores yet. Scores appear after drafts are graded in the background.")
        return

    # Build a long-format table: one row per (draft_number, criterion, score%).
    # Draft numbering is 1-based over graded assistant drafts -- matches the
    # "Draft N" label shown on each draft in the conversation tab.
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(graded, start=1):
        dg = m.get("draft_grade") or {}
        for c in dg.get("grades") or []:
            name = (c.get("criterion_name") or "").strip()
            if not name:
                continue
            pct = parse_score_pct(c.get("score"))
            if pct is None:
                continue
            rows.append({
                "Draft": i,
                "Criterion": name,
                "Score (%)": round(pct * 100, 1),
            })

    if not rows:
        st.info("No parseable scores to plot yet.")
        return

    n_drafts = len(graded)

    def _render_latest_scorecard() -> None:
        """Dimension-level scorecard for the latest draft. Rendered after
        whichever chart (bar/plotly/altair/line) we end up showing, so the
        user always sees the detailed breakdown."""
        latest = graded[-1]
        dg = latest.get("draft_grade") or {}
        with st.expander(f"Latest draft scorecard (draft {n_drafts})", expanded=False):
            for c in sorted(
                dg.get("grades") or [],
                key=lambda x: (x.get("criterion_priority", 99), x.get("criterion_name") or ""),
            ):
                name = c.get("criterion_name") or ""
                sc = c.get("score") or ""
                st.markdown(f"**{html_lib.escape(name)}** (priority {c.get('criterion_priority', '?')}) — `{sc}`")
                for d in c.get("dimension_grades") or []:
                    g = (d.get("grade") or "").upper()
                    conf = d.get("confidence", "high")
                    mark = "✓" if g == "MET" else "✗"
                    conf_badge = ""
                    if conf == "low":
                        conf_badge = " ⚠️ _low confidence_"
                        note = d.get("ambiguity_note", "")
                        if note:
                            conf_badge += f" — {note}"
                    elif conf == "medium":
                        conf_badge = " 🔸 _medium confidence_"
                    st.caption(f"{mark} `{d.get('dimension_id')}` — {d.get('evidence', '')}{conf_badge}")

    # One draft isn't enough for a line — fall back to a horizontal bar view.
    if n_drafts == 1:
        import pandas as _pd
        df = _pd.DataFrame(rows).set_index("Criterion")["Score (%)"]
        st.bar_chart(df)
        _render_latest_scorecard()
        return

    # Plotly line graph with its built-in legend DISABLED. Streamlit's sidebar
    # is narrow enough that Plotly's legend (horizontal or vertical) clips the
    # criterion names no matter how we position it. Instead we render a
    # custom color-key below the chart as markdown, which wraps cleanly.
    import pandas as _pd
    df = _pd.DataFrame(rows)
    wide = df.pivot(index="Draft", columns="Criterion", values="Score (%)")

    # Try plotly first (gives us full control over the legend). If it fails,
    # use altair as a second option (also supports disabled legends and
    # per-series color). st.line_chart is the last resort and DOES show its
    # own legend in-chart which will clip in the sidebar -- but at least the
    # graph renders.
    _PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    # Stable column order (deterministic across runs) so legend colors match
    # the chart lines.
    columns = list(wide.columns)
    crit_colors: dict[str, str] = {
        cn: _PALETTE[i % len(_PALETTE)] for i, cn in enumerate(columns)
    }

    rendered = False
    # --- Attempt 1: Plotly -------------------------------------------------
    # Fast-path: if plotly isn't importable, skip directly to altair without
    # a stacktrace-noisy warning. Altair is installed in every env we ship to
    # so this is the common silent fallback on machines without plotly.
    _plotly_available = False
    try:
        import plotly.graph_objects as go  # noqa: F401
        _plotly_available = True
    except ImportError:
        _log.debug("plotly not available; using altair")

    if _plotly_available:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for cn in columns:
                fig.add_trace(go.Scatter(
                    x=wide.index.tolist(),
                    y=wide[cn].tolist(),
                    mode="lines+markers",
                    name=cn,
                    line=dict(color=crit_colors[cn]),
                    marker=dict(color=crit_colors[cn]),
                    hovertemplate="<b>%{fullData.name}</b><br>Draft %{x}: %{y:.1f}%<extra></extra>",
                ))
            fig.update_layout(
                xaxis_title="Draft #",
                yaxis_title="% dimensions met",
                yaxis=dict(range=[0, 105]),
                xaxis=dict(tickmode="linear", dtick=1),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=40),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)
            rendered = True
        except Exception as _e:
            # Plotly was importable but rendering failed -- this IS worth a
            # warning since it means a real rendering bug.
            _log.warning("Plotly rendering failed, trying altair: %s", _e)

    # --- Attempt 2: Altair -------------------------------------------------
    if not rendered:
        try:
            import altair as alt

            long_df = wide.reset_index().melt(
                id_vars="Draft", var_name="Criterion", value_name="Score",
            )
            chart = (
                alt.Chart(long_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Draft:Q",
                            axis=alt.Axis(tickMinStep=1, title="Draft #")),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 105]),
                            title="% dimensions met"),
                    color=alt.Color(
                        "Criterion:N",
                        scale=alt.Scale(
                            domain=columns,
                            range=[crit_colors[c] for c in columns],
                        ),
                        legend=None,  # no in-chart legend
                    ),
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
            rendered = True
        except ImportError:
            _log.debug("altair not available; using st.line_chart")
        except Exception as _e:
            _log.warning("Altair rendering failed, falling back to st.line_chart: %s", _e)

    # --- Attempt 3: st.line_chart (last resort, will show its own legend) --
    if not rendered:
        st.line_chart(wide, y_label="% dimensions met", x_label="Draft #")
        # With st.line_chart, Streamlit draws its OWN legend using its own
        # palette. Skip our custom legend to avoid a color mismatch -- the
        # built-in legend is authoritative in this fallback. Still render the
        # scorecard so the user sees dimension-level detail.
        _render_latest_scorecard()
        return

    # --- Custom color key below the chart (plotly/altair path) -------------
    legend_md_parts = []
    for cn in columns:
        color = crit_colors[cn]
        legend_md_parts.append(
            f'<span style="display:inline-flex;align-items:center;margin:2px 8px 2px 0;font-size:0.85rem;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};margin-right:6px;"></span>{html_lib.escape(cn)}</span>'
        )
    st.markdown(
        '<div style="line-height:1.6;">' + "".join(legend_md_parts) + '</div>',
        unsafe_allow_html=True,
    )

    # Dimension-level scorecard for the latest draft (plotly/altair path).
    _render_latest_scorecard()


def invalidate_stale_suggestions() -> int:
    """Mark any pending rubric-edit suggestion whose target dimension no
    longer exists in the active rubric as stale, so the Apply button can't
    fire a silent failure. Returns the number of suggestions invalidated.

    Call this after any save that mutates the rubric's dimension set --
    dim-recognition Confirm (which may remove dims), refiner Apply (which
    modifies wording), etc. -- so pending suggestions from before the save
    can't be applied against a rubric that no longer matches their target."""
    try:
        from rubric_writer.persistence import get_active_rubric
        rubric_dict, _, _ = get_active_rubric()
    except Exception:
        return 0
    if not rubric_dict:
        return 0

    # Build the set of (criterion_name_lower, dim_id_lower) pairs currently in
    # the rubric, plus a set of dim_ids that exist at all (for fallback check).
    live_pairs: set[tuple[str, str]] = set()
    live_dim_ids: set[str] = set()
    for crit in rubric_dict.get("rubric") or []:
        cname = (crit.get("name") or "").strip().lower()
        for dim in crit.get("dimensions") or []:
            did = (dim.get("id") or "").strip().lower()
            if did:
                live_pairs.add((cname, did))
                live_dim_ids.add(did)

    suggestions = st.session_state.get("rubric_edit_suggestions") or []
    invalidated = 0
    for s in suggestions:
        if s.get("status") != "pending":
            continue
        sug_crit = (s.get("criterion_name") or "").strip().lower()
        sug_dim = (s.get("dimension_id") or "").strip().lower()
        if not sug_dim:
            continue
        # Match by dim_id (the stable identifier); criterion name is only a
        # fallback disambiguator since the LLM sometimes rephrases it.
        if sug_dim in live_dim_ids:
            continue
        s["status"] = "stale"
        s["stale_reason"] = (
            f"Dimension `{s.get('dimension_id')}` under **{s.get('criterion_name')}** "
            "is no longer in the current rubric -- it may have been removed or renamed. "
            "This suggestion can no longer be applied."
        )
        invalidated += 1
    return invalidated


def clear_rubric_edit_session_state() -> None:
    """Clear all rubric-edit-related session state. Call when switching
    conversations so stale suggestions from a prior conversation don't leak
    into the new one's sidebar."""
    keys_to_clear = (
        "rubric_edit_suggestions",
        "rubric_edit_verifications",
        "_pending_rubric_feedback",
        "_rubric_apply_warnings",
        "_rubric_apply_successes",
        "rubric_update_result",
        "_deferred_flush_panels",
        "_deferred_refinements",
    )
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    # Drift anchors are message-id-scoped; purge them on conversation switch
    # so the previous conversation's panels don't ghost into the new one's
    # render if message_ids happen to collide.
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("_drift_anchor_"):
            del st.session_state[k]


def render_rubric_edit_suggestions() -> None:
    """Sidebar panel showing pending rubric edit suggestions and verification results."""
    # Invalidate any pending suggestions whose target dim is no longer in the
    # active rubric. This handles the case where the user (a) rejects a dim in
    # the recognition flow, or (b) applies a prior edit that renames a dim,
    # and THEN tries to apply a stale suggestion that still references the
    # old dim. Without this, Apply would silently fail with "Couldn't find
    # dimension ..." after the user already saw it in the sidebar.
    invalidate_stale_suggestions()

    suggestions = st.session_state.get("rubric_edit_suggestions") or []
    verifications = st.session_state.get("rubric_edit_verifications") or []
    pending = [s for s in suggestions if s.get("status") == "pending"]
    stale = [s for s in suggestions if s.get("status") == "stale"]
    # Diagnostic: surface suggestion counts so we can tell whether the sidebar
    # sees nothing (refiner produced nothing) vs. everything got marked stale.
    if suggestions:
        _status_counts: dict = {}
        for _s in suggestions:
            _status_counts[_s.get("status", "?")] = _status_counts.get(_s.get("status", "?"), 0) + 1
        _log.info(
            "[rubric edit sidebar] %d suggestion(s) in session: %s",
            len(suggestions), _status_counts,
        )

    # Drain any apply-result messages that survived the rerun.
    for msg in st.session_state.pop("_rubric_apply_successes", []):
        st.success(msg)
    for msg in st.session_state.pop("_rubric_apply_warnings", []):
        st.warning(msg)

    # Show any freshly-stale suggestions ONCE as a warning, then stop showing.
    for s in stale:
        if not s.get("_stale_shown"):
            st.warning(s.get("stale_reason") or "A suggested rubric edit is no longer applicable.")
            s["_stale_shown"] = True

    # Show verification results for recently applied edits
    if verifications:
        unshown = [v for v in verifications if not v.get("_shown_in_sidebar")]
        for v in unshown:
            cname = v.get("criterion_name", "")
            did = v.get("dimension_id", "")
            label = f"`{did}`" if did else cname
            status = v.get("status")
            if status == "verified":
                if v.get("aligned"):
                    st.success(
                        f"Verified: re-grading **{label}** now gives "
                        f"**{v.get('new_grade')}**, matching your feedback."
                    )
                else:
                    expected = v.get("user_expected", "?")
                    got = v.get("new_grade", "?")
                    st.warning(
                        f"Mismatch: re-grading **{label}** gives **{got}** "
                        f"but your feedback expected **{expected}**."
                    )
            elif status == "skipped":
                st.info(
                    f"Edit applied to **{label}**. Verification skipped ({v.get('reason', 'n/a')})."
                )
            elif status == "error":
                st.error(f"Verification error for **{label}**: {v.get('error', '')}")
            v["_shown_in_sidebar"] = True

    if not pending:
        return

    with st.expander(f"✏️ {len(pending)} suggested rubric edit(s)", expanded=True):
        for i, s in enumerate(pending):
            _render_single_edit_suggestion(s, i)
            st.divider()


# Scope change badge styling: (label, background, text color)
_SCOPE_BADGE: dict[str, tuple[str, str, str]] = {
    "expands":   ("Expands criterion",  "#d4edda", "#155724"),  # green
    "narrows":   ("Narrows criterion",  "#fff3cd", "#856404"),  # amber
    "clarifies": ("Clarifies criterion","#d1ecf1", "#0c5460"),  # blue
    "reframes":  ("Reframes criterion", "#e2d9f3", "#4a2c7a"),  # purple
}


def _render_scope_badge(scope: str) -> None:
    label, bg, fg = _SCOPE_BADGE.get(scope, ("Rubric edit", "#eeeeee", "#333333"))
    st.markdown(
        f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:12px;'
        f'font-size:11px;font-weight:600;">{html_lib.escape(label)}</span>',
        unsafe_allow_html=True,
    )


def _compute_word_diff_html(before: str, after: str) -> tuple[str, str]:
    """Return (before_html, after_html) with changed words visually distinguished."""
    import difflib as _difflib
    before_tokens = (before or "").split()
    after_tokens = (after or "").split()
    sm = _difflib.SequenceMatcher(a=before_tokens, b=after_tokens)
    before_parts: list[str] = []
    after_parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            before_parts.extend(html_lib.escape(t) for t in before_tokens[i1:i2])
            after_parts.extend(html_lib.escape(t) for t in after_tokens[j1:j2])
        elif tag == "delete":
            before_parts.extend(
                f'<span style="background:#ffe0e0;text-decoration:line-through;color:#a02020;">{html_lib.escape(t)}</span>'
                for t in before_tokens[i1:i2]
            )
        elif tag == "insert":
            after_parts.extend(
                f'<span style="background:#e0ffe0;color:#1e7a1e;font-weight:600;">{html_lib.escape(t)}</span>'
                for t in after_tokens[j1:j2]
            )
        elif tag == "replace":
            before_parts.extend(
                f'<span style="background:#ffe0e0;text-decoration:line-through;color:#a02020;">{html_lib.escape(t)}</span>'
                for t in before_tokens[i1:i2]
            )
            after_parts.extend(
                f'<span style="background:#e0ffe0;color:#1e7a1e;font-weight:600;">{html_lib.escape(t)}</span>'
                for t in after_tokens[j1:j2]
            )
    return " ".join(before_parts), " ".join(after_parts)


def _render_single_edit_suggestion(s: dict[str, Any], i: int) -> None:
    """Render one proposed edit: header w/ scope badge, diff, reasoning, expandable example, actions."""
    crit_name = s.get("criterion_name", "")
    dim_id = s.get("dimension_id", "")
    dim_desc = _lookup_dimension_description(dim_id, crit_name) if dim_id else ""
    dim_label = dim_desc if dim_desc else (dim_id or crit_name)

    # Show the actual stored wording if we have it (set by a successful Apply
    # earlier on this suggestion, or by a pre-render lookup). Otherwise fall
    # back to the LLM's echoed before_wording. This keeps the diff honest --
    # what the user sees as "Before" matches what's in the rubric config UI.
    crit_dim_text = ""
    try:
        from rubric_writer.persistence import get_active_rubric
        _r, _, _ = get_active_rubric()
        _, _, crit_dim_text = _lookup_dim_field_and_text(_r, crit_name, dim_id)
    except Exception:
        crit_dim_text = ""
    before = (
        s.get("actual_before_wording")
        or crit_dim_text
        or s.get("before_wording")
        or s.get("old_text", "")
    )
    after = s.get("after_wording") or s.get("new_text", "")
    reasoning = s.get("reasoning", "")
    scope = (s.get("scope_change") or "").strip().lower()
    example_annotation = s.get("example_annotation", "")
    grader_evidence = s.get("grader_evidence", "")
    parse_status = s.get("parse_status", "ok")
    edit_id = s.get("edit_id", f"local_{i}")

    # --- Header: title + dimension + scope badge ---
    col_title, col_badge = st.columns([3, 2])
    with col_title:
        st.markdown(f"**Proposed rubric edit** &nbsp; _{html_lib.escape(dim_label)}_",
                    unsafe_allow_html=True)
    with col_badge:
        if scope in _SCOPE_BADGE:
            _render_scope_badge(scope)

    # Pre-verification indicator
    pre_verify = s.get("pre_verification") or {}
    drift_kind_s = s.get("drift_kind", "")
    if s.get("is_best_attempt"):
        # We tried but couldn't produce a verified edit
        final_status = pre_verify.get("attempt_2_status") or pre_verify.get("attempt_1_status", "")
        new_grade = pre_verify.get("attempt_2_new_grade") or pre_verify.get("attempt_1_new_grade") or "?"
        expected = pre_verify.get("user_expected", "?")
        if final_status == "still_uncertain":
            st.warning(
                f"⚠ Best attempt -- this edit didn't fully verify. "
                f"Re-grading matched your feedback (**{new_grade}**), but the grader was "
                f"still low-confidence. The rubric wording remains ambiguous. "
                f"Review carefully before applying."
            )
        elif final_status == "still_unstable":
            st.warning(
                f"⚠ Best attempt -- this edit didn't fully stabilize the dimension. "
                f"Re-grading matched your feedback (**{new_grade}**), but confidence was "
                f"still low -- the dimension may continue oscillating. "
                f"Review carefully before applying."
            )
        elif final_status in ("dim_not_found", "error") or new_grade in (None, "?", "None"):
            # Verification couldn't locate the dimension after re-grading (e.g.,
            # criterion/dim id mismatch between rubric and grader output). The
            # edit itself may still be fine -- the user just doesn't have a
            # re-grade to compare against.
            st.warning(
                "⚠ Couldn't verify this edit by re-grading -- the grader didn't "
                "return a result for this dimension. The edit itself may still "
                "be reasonable; review the wording change carefully before applying."
            )
        else:
            st.warning(
                f"⚠ Best attempt -- this edit didn't fully verify. "
                f"Re-grading gives **{new_grade}** but your feedback expected **{expected}**. "
                f"Review carefully before applying."
            )
    else:
        # Edit was pre-verified
        final_status = pre_verify.get("attempt_2_status") or pre_verify.get("attempt_1_status", "")
        if final_status == "aligned":
            new_grade = pre_verify.get("attempt_2_new_grade") or pre_verify.get("attempt_1_new_grade", "?")
            attempts = "on retry" if s.get("is_retry") else "on first try"
            # Drift-specific verification messages
            if drift_kind_s == "low_confidence":
                st.caption(
                    f"✓ Verified {attempts}: grader is no longer uncertain and gives "
                    f"**{new_grade}**, matching your feedback."
                )
            elif drift_kind_s == "oscillation":
                st.caption(
                    f"✓ Verified {attempts}: grader is now confident (no more oscillation) "
                    f"and gives **{new_grade}**."
                )
            elif drift_kind_s == "persistent_failure":
                st.caption(
                    f"✓ Verified {attempts}: re-grading with the adjusted bar gives "
                    f"**{new_grade}**, matching your feedback."
                )
            elif drift_kind_s == "spot_check":
                st.caption(
                    f"✓ Verified {attempts}: re-grading now aligns with your judgment "
                    f"(**{new_grade}**)."
                )
            else:
                st.caption(
                    f"✓ Verified {attempts}: re-grading gives **{new_grade}**, matching your feedback."
                )
        elif final_status == "no_expectation":
            # Drift-specific for tradeoff (priority changes, not grade changes)
            if drift_kind_s == "tradeoff":
                st.caption("_Priority adjustment -- no grade change to verify against._")
            else:
                st.caption("_This feedback type doesn't have a specific grade expectation to verify against._")
        elif final_status in ("error", "dim_not_found"):
            st.caption(f"_⚠ Verification skipped ({final_status}). Review the edit carefully._")
        elif not pre_verify:
            st.caption("_This suggestion was created before verification was enabled._")
        else:
            st.caption(f"_Verification status: {final_status or 'unknown'}_")

    # --- Diff section ---
    if before and after:
        before_html, after_html = _compute_word_diff_html(before, after)
        st.caption("Before")
        st.markdown(
            f'<div style="padding:6px 10px;border-left:3px solid #e0a0a0;background:#fafafa;'
            f'font-size:13px;">{before_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption("After")
        st.markdown(
            f'<div style="padding:6px 10px;border-left:3px solid #80c080;background:#fafafa;'
            f'font-size:13px;">{after_html}</div>',
            unsafe_allow_html=True,
        )
    elif before or after:
        # Fallback if we only have one side
        st.caption("Proposed wording")
        st.markdown(f"> {html_lib.escape(after or before)}")

    # --- Reasoning section ---
    if reasoning:
        st.markdown("**Why this change**")
        st.markdown(html_lib.escape(reasoning))
    elif parse_status != "ok":
        # Diff-only fallback when parse failed
        st.caption(f"_(reasoning unavailable: {parse_status})_")

    # --- Expandable example section ---
    if grader_evidence or example_annotation:
        exp_key = f"rubric_sug_example_expanded_{edit_id}_{i}"
        was_expanded = st.session_state.get(exp_key, False)
        with st.expander("Example from your draft", expanded=was_expanded):
            # Fire example_expanded telemetry on first open
            if not was_expanded:
                st.session_state[exp_key] = True
                try:
                    from rubric_writer.metrics import log_example_expanded
                    log_example_expanded(edit_id=edit_id)
                except Exception:
                    pass
            if grader_evidence:
                st.markdown(
                    f'<blockquote style="border-left:3px solid #bbbbbb;padding:4px 10px;'
                    f'color:#555555;font-style:italic;margin:6px 0;">{html_lib.escape(grader_evidence)}</blockquote>',
                    unsafe_allow_html=True,
                )
            if example_annotation:
                st.markdown(html_lib.escape(example_annotation))

    # --- Actions ---
    col_apply, col_dismiss = st.columns(2)
    with col_apply:
        if st.button("✅ Apply", key=f"rubric_sug_apply_{i}", type="primary"):
            ok = _apply_rubric_suggestion(s)
            if ok:
                s["status"] = "applied"
                _append_rubric_edit_system_message(s, decision="applied")
                try:
                    from rubric_writer.metrics import log_edit_decision
                    log_edit_decision(edit_id=edit_id, decision="applied",
                                      scope_change=scope, criterion_id=crit_name,
                                      dimension_id=dim_id)
                except Exception:
                    pass
                # P0.1: terminal disposition for this proposal.
                _log_proposal_disposition(s, "applied")
            else:
                # Mark the suggestion as failed so the UI can surface a retry/edit option.
                s["status"] = "apply_failed"
                _log_proposal_disposition(s, "apply_failed")
            st.rerun()
    with col_dismiss:
        if st.button("✗ Dismiss", key=f"rubric_sug_dismiss_{i}"):
            s["status"] = "dismissed"
            _append_rubric_edit_system_message(s, decision="dismissed")
            try:
                from rubric_writer.metrics import log_edit_decision
                log_edit_decision(edit_id=edit_id, decision="dismissed",
                                  scope_change=scope, criterion_id=crit_name,
                                  dimension_id=dim_id)
            except Exception:
                pass
            _log_proposal_disposition(s, "dismissed")
            st.rerun()


def _log_proposal_disposition(suggestion: dict[str, Any], disposition: str) -> None:
    """P0.1: write a terminal-disposition row for a refiner_proposal so the
    proposed→applied/dismissed/etc. lifecycle is queryable post-hoc."""
    try:
        sb = st.session_state.get("supabase")
        pid = st.session_state.get("current_project_id")
        if not sb or not pid:
            return
        save_project_data(sb, pid, "refiner_proposal", {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": st.session_state.get("selected_conversation"),
            "edit_id": suggestion.get("edit_id"),
            "drift_kind": suggestion.get("drift_kind"),
            "target_criterion": suggestion.get("criterion_name"),
            "target_dim_id": suggestion.get("dimension_id"),
            "disposition": disposition,
        })
    except Exception as _e:
        _log.warning("refiner_proposal log (%s) failed: %s", disposition, _e)


def _append_rubric_edit_system_message(suggestion: dict[str, Any], decision: str) -> None:
    """Append a system message to the conversation describing the rubric edit
    and whether it was applied or dismissed."""
    import time as _time
    crit_name = suggestion.get("criterion_name", "")
    dim_id = suggestion.get("dimension_id", "")
    dim_desc = _lookup_dimension_description(dim_id, crit_name) if dim_id else ""
    dim_label = dim_desc if dim_desc else (dim_id or crit_name)
    # Prefer the actual stored wording (set by _apply_rubric_suggestion) over
    # the LLM's possibly-paraphrased before_wording.
    before = (
        suggestion.get("actual_before_wording")
        or suggestion.get("before_wording")
        or suggestion.get("old_text", "")
    )
    after = suggestion.get("after_wording") or suggestion.get("new_text", "")
    reasoning = suggestion.get("reasoning", "")
    scope = suggestion.get("scope_change", "")

    icon = "✅" if decision == "applied" else "✗"
    verb = "Applied" if decision == "applied" else "Dismissed"

    lines = [f"{icon} **Rubric edit {verb.lower()}**"]
    if crit_name and dim_label:
        lines.append(f"_{crit_name}_: **{dim_label}**")
    if scope:
        lines.append(f"_Scope: {scope}_")
    if decision == "applied":
        if before:
            lines.append(f"\n**Before:** {before}")
        if after:
            lines.append(f"**After:** {after}")
        if reasoning:
            lines.append(f"\n**Why:** {reasoning}")
    else:
        if after:
            lines.append(f"\n_Proposed wording (not applied):_ {after}")

    st.session_state.setdefault("messages", []).append({
        "role": "system",
        "content": "\n".join(lines),
        "message_id": f"rubric_edit_{decision}_{int(_time.time() * 1000000)}",
        "is_system_generated": True,
    })

    # Persist immediately so the message is bound to THIS conversation. Without
    # this, the message stays in-memory until the next auto-save fires, which
    # could write it into a different conversation if the user switches first.
    try:
        from rubric_writer.persistence import _auto_save_conversation
        _auto_save_conversation()
    except Exception as e:
        _log.warning("auto-save after rubric edit system message failed: %s", e)


def _remove_dimensions_and_save(
    *,
    removals: list[dict[str, str]],
) -> bool:
    """Remove MULTIPLE dimensions in a single new rubric version.

    `removals` is a list of dicts with keys `criterion_name`, `dimension_id`,
    and optional `feedback_text`. All removals are applied to ONE deep-copy
    of the active rubric and saved as ONE new version, so a user removing
    N dimensions in one panel gets a single version bump rather than N.
    Returns True on success."""
    import copy as _copy
    from rubric_writer.persistence import (
        load_rubric_history, save_rubric_history, invalidate_rubric_cache,
    )

    if not removals:
        return False

    rubric_history = load_rubric_history(force_reload=True)
    if not rubric_history:
        st.warning("No active rubric — can't remove dimensions.")
        return False
    rubric_dict = rubric_history[-1]
    if not rubric_dict.get("rubric"):
        st.warning("Active rubric has no criteria — can't remove dimensions.")
        return False

    # Normalize the target dim_ids once.
    target_ids = {
        (r.get("dimension_id") or "").strip().lower()
        for r in removals
        if (r.get("dimension_id") or "").strip()
    }
    if not target_ids:
        st.warning("Missing dimension_ids — can't remove.")
        return False

    new_version = _copy.deepcopy(rubric_dict)
    new_version.pop("id", None)
    new_version.pop("version", None)
    new_version.pop("created_at", None)

    # Track what we actually removed for the summary message.
    removed_display: list[tuple[str, str]] = []  # (criterion_name, dim_label)
    for crit in new_version.get("rubric") or []:
        cname = crit.get("name", "")
        dims = crit.get("dimensions") or []
        kept = []
        for d in dims:
            if (d.get("id") or "").strip().lower() in target_ids:
                dim_label = d.get("label") or d.get("description") or d.get("id", "")
                removed_display.append((cname, dim_label))
                continue
            kept.append(d)
        crit["dimensions"] = kept
    new_version["rubric"] = [c for c in new_version["rubric"] if c.get("dimensions")]

    if not removed_display:
        st.warning(
            "Couldn't find any of the requested dimensions in the current rubric — "
            "they may have already been removed."
        )
        return False

    new_version["source"] = "user_removed_persistent_failure"
    rubric_history.append(new_version)

    try:
        saved_version = save_rubric_history(rubric_history)
    except Exception as e:
        _log.warning("save_rubric_history (batch remove) failed: %s", e)
        if rubric_history and rubric_history[-1] is new_version:
            rubric_history.pop()
        st.error(f"Couldn't save rubric after removals: {e}")
        return False

    if saved_version is None:
        if rubric_history and rubric_history[-1] is new_version:
            rubric_history.pop()
        st.error("Couldn't save rubric after removals (no version returned).")
        return False

    # P0.4: structured rubric_edit_event log. Attribute this version bump to
    # the drift-panel Remove button, include which dims were removed, so
    # post-hoc we can say "version v3 came from a user Remove action on
    # persistent_failure panel, removing dims X, Y, Z" without diffing.
    try:
        _prev_version = rubric_dict.get("version", saved_version - 1 if saved_version else None)
        _sb = st.session_state.get("supabase")
        _pid = st.session_state.get("current_project_id")
        if _sb and _pid:
            save_project_data(_sb, _pid, "rubric_edit_event", {
                "timestamp": datetime.now().isoformat(),
                "conversation_id": st.session_state.get("selected_conversation"),
                "from_version": _prev_version,
                "to_version": saved_version,
                "trigger": "drift_panel_remove",
                "edit_summary": {
                    "added_dims": [],
                    "removed_dims": [
                        {"criterion": c, "dim_label": lbl}
                        for c, lbl in removed_display
                    ],
                    "modified_dims": [],
                },
                "feedback_texts": [r.get("feedback_text", "") for r in removals],
            })
    except Exception as _e:
        _log.warning("rubric_edit_event log (remove) failed: %s", _e)

    invalidate_rubric_cache()

    # Refresh shadow state.
    reloaded = load_rubric_history(force_reload=True)
    if reloaded:
        new_active = reloaded[-1]
        new_criteria = new_active.get("rubric", [])
        st.session_state.rubric = new_criteria
        try:
            st.session_state.editing_criteria = _copy.deepcopy(new_criteria)
            st.session_state["editing_criteria_ui_version"] = (
                st.session_state.get("editing_criteria_ui_version", 0) + 1
            )
        except Exception:
            pass
        st.session_state.active_rubric_idx = len(reloaded) - 1
        try:
            from rubric_writer.widget_keys import project_scoped_key
            _rvk = project_scoped_key("rubric_version_selector")
            st.session_state.pop(_rvk, None)
        except Exception:
            pass

    # Log one event per removed dim so analysis can count accurately.
    sb = st.session_state.get("supabase")
    pid = st.session_state.get("current_project_id")
    if sb and pid:
        for r, (cname_rm, dlabel_rm) in zip(removals, removed_display):
            try:
                save_project_data(sb, pid, "rubric_dimension_removed", {
                    "timestamp": datetime.now().isoformat(),
                    "criterion_name": cname_rm,
                    "dimension_id": r.get("dimension_id", ""),
                    "dimension_label": dlabel_rm,
                    "source": "persistent_failure_remove",
                    "feedback_text": r.get("feedback_text", ""),
                    "new_version": saved_version,
                    "batch_size": len(removed_display),
                })
            except Exception:
                pass

    # One system message summarizing the batch removal.
    import time as _time
    if len(removed_display) == 1:
        cname_rm, dlabel_rm = removed_display[0]
        content = (
            f"🗑 **Dimension removed.** _{cname_rm}_: **{dlabel_rm}** was "
            f"removed after being consistently not met — new rubric "
            f"**v{saved_version}** saved."
        )
    else:
        lines = [f"🗑 **{len(removed_display)} dimensions removed** — new rubric **v{saved_version}** saved."]
        for cname_rm, dlabel_rm in removed_display:
            lines.append(f"• _{cname_rm}_: **{dlabel_rm}**")
        content = "\n".join(lines)

    st.session_state.setdefault("messages", []).append({
        "role": "system",
        "content": content,
        "message_id": f"dim_removed_{int(_time.time() * 1000000)}",
        "is_system_generated": True,
    })
    try:
        from rubric_writer.persistence import _auto_save_conversation
        _auto_save_conversation()
    except Exception as e:
        _log.warning("auto-save after dim removal failed: %s", e)

    if len(removed_display) == 1:
        cname_rm, dlabel_rm = removed_display[0]
        st.session_state.setdefault("_rubric_apply_successes", []).append(
            f"Removed **{dlabel_rm}** from _{cname_rm}_ — new rubric v{saved_version} saved."
        )
    else:
        st.session_state.setdefault("_rubric_apply_successes", []).append(
            f"Removed {len(removed_display)} dimensions — new rubric v{saved_version} saved."
        )
    return True


def _remove_dimension_and_save(
    *,
    criterion_name: str,
    dimension_id: str,
    feedback_text: str = "",
) -> bool:
    """Single-dim wrapper around _remove_dimensions_and_save. Kept as a
    convenience for the 1-item case; batched removals from the panel go
    through the plural form directly to produce a single new rubric version."""
    return _remove_dimensions_and_save(removals=[{
        "criterion_name": criterion_name,
        "dimension_id": dimension_id,
        "feedback_text": feedback_text,
    }])


def _apply_rubric_suggestion(suggestion: dict[str, Any]) -> bool:
    """Apply a suggested rubric edit by deep-copying the active rubric, mutating
    the copy, and appending it as a new rubric version. Returns True on success.

    Note: we do NOT use the LLM's `before_wording` to substring-match against
    the rubric. The LLM frequently paraphrases the stored wording when echoing
    it back, which used to cause silent apply failures. Instead we look up the
    actual stored text by (criterion_name, dimension_id) and overwrite the
    whole field with `after_wording`."""
    import copy as _copy
    from rubric_writer.persistence import (
        get_active_rubric, load_rubric_history, save_rubric_history, invalidate_rubric_cache,
    )

    # Always reload from the DB so we're editing the truly latest version, not
    # a stale cached copy.
    rubric_history = load_rubric_history(force_reload=True)
    if not rubric_history:
        st.error("No active rubric to edit.")
        return False
    rubric_dict = rubric_history[-1]
    if not rubric_dict.get("rubric"):
        st.error("Active rubric has no criteria.")
        return False

    crit_name = (suggestion.get("criterion_name") or "").strip()
    dim_id = (suggestion.get("dimension_id") or "").strip()
    new_text = suggestion.get("after_wording") or suggestion.get("new_text", "")

    if not new_text:
        st.session_state.setdefault("_rubric_apply_warnings", []).append(
            f"Suggestion for **{crit_name}** is missing the new wording. Edit not applied."
        )
        return False

    # Deep-copy so the mutation doesn't touch the existing version. The result
    # becomes the new version we append to history.
    new_version = _copy.deepcopy(rubric_dict)
    new_version.pop("id", None)
    new_version.pop("version", None)
    new_version.pop("created_at", None)

    # Look up the actual stored wording in the COPY, then overwrite that field.
    dim, field, actual_before = _lookup_dim_field_and_text(new_version, crit_name, dim_id)
    if dim is None or field is None:
        st.session_state.setdefault("_rubric_apply_warnings", []).append(
            f"Couldn't find dimension `{dim_id}` under **{crit_name}** in the "
            "current rubric. The rubric may have changed since this suggestion "
            "was generated. Edit not applied."
        )
        return False

    dim[field] = new_text
    # Stamp the actual stored wording onto the suggestion so downstream UI
    # (system message, history) shows what really got replaced -- not the LLM's
    # paraphrased echo of the before_wording.
    suggestion["actual_before_wording"] = actual_before

    # Sanity-check the in-memory mutation BEFORE saving.
    _check_dim, _check_field, _check_text = _lookup_dim_field_and_text(
        new_version, crit_name, dim_id,
    )
    if _check_text != new_text:
        _log.error(
            "_apply_rubric_suggestion: in-memory mutation didn't take. "
            "Expected %r, got %r in field %r", new_text, _check_text, _check_field,
        )
        st.session_state.setdefault("_rubric_apply_warnings", []).append(
            f"Internal error: edit was not applied to the in-memory rubric "
            f"copy for **{crit_name}**. Edit not saved."
        )
        return False

    rubric_history.append(new_version)
    try:
        saved_version = save_rubric_history(rubric_history)
        if saved_version is None:
            # DB save failed (no exception, but no version returned). Roll back.
            if rubric_history and rubric_history[-1] is new_version:
                rubric_history.pop()
            st.session_state.setdefault("_rubric_apply_warnings", []).append(
                f"Save returned no version for **{crit_name}** -- the new "
                "rubric version was NOT saved to the database. Edit not applied."
            )
            return False
        invalidate_rubric_cache()

        # Verify the saved version round-trips with the change. If the DB came
        # back with the OLD text, something is silently dropping the mutation.
        _verify_history = load_rubric_history(force_reload=True)
        _v_text = ""
        if _verify_history:
            _, _, _v_text = _lookup_dim_field_and_text(
                _verify_history[-1], crit_name, dim_id,
            )
        if _v_text != new_text:
            _log.error(
                "_apply_rubric_suggestion: saved version doesn't reflect edit. "
                "Expected %r, got %r from DB.", new_text, _v_text,
            )
            st.session_state.setdefault("_rubric_apply_warnings", []).append(
                f"The rubric was saved as v{saved_version} but the dimension "
                f"text in the DB doesn't match the edit. This is a bug -- "
                f"please report it. Expected: {new_text[:80]}... Got: {_v_text[:80] if _v_text else '(empty)'}..."
            )
            return False

        # CRITICAL: refresh the session-state shadow copies that the rubric
        # configuration UI reads from. Without this, the UI keeps showing the
        # OLD version's wording even though the DB has the new version.
        # `editing_criteria` is a deepcopy of the active rubric's criteria list
        # used by the config tab; `rubric` is a flatter shadow used elsewhere.
        new_active = _verify_history[-1]
        new_criteria = new_active.get("rubric", [])
        st.session_state.rubric = new_criteria
        try:
            import copy as _copy_state
            st.session_state.editing_criteria = _copy_state.deepcopy(new_criteria)
            # Bump the editing UI version key so any cached widget state for
            # the old criteria is discarded on rerun.
            st.session_state["editing_criteria_ui_version"] = (
                st.session_state.get("editing_criteria_ui_version", 0) + 1
            )
        except Exception as e:
            _log.warning("failed to refresh editing_criteria: %s", e)
        # Make sure the active index points at the new version.
        st.session_state.active_rubric_idx = len(_verify_history) - 1
        # CLEAR the version-selector widget key so the selectbox re-initializes
        # from `index=active_idx` on the next render. Setting it to a specific
        # value is fragile -- Streamlit can raise StreamlitAPIException if the
        # widget has already been instantiated in this session.
        try:
            from rubric_writer.widget_keys import project_scoped_key
            _rvk = project_scoped_key("rubric_version_selector")
            st.session_state.pop(_rvk, None)
        except Exception:
            pass

        st.session_state.setdefault("_rubric_apply_successes", []).append(
            f"Updated **{crit_name}** -- new rubric version v{saved_version} saved."
        )
        sb = st.session_state.get("supabase")
        pid = st.session_state.get("current_project_id")
        if sb and pid:
            save_project_data(sb, pid, "rubric_edit_applied", {
                "timestamp": datetime.now().isoformat(),
                "criterion_name": crit_name,
                "dimension_id": dim_id,
                "before_wording_llm": suggestion.get("before_wording") or suggestion.get("old_text", ""),
                "before_wording_actual": actual_before,
                "after_wording": new_text,
                "scope_change": suggestion.get("scope_change", ""),
                "reasoning": suggestion.get("reasoning", ""),
                "example_annotation": suggestion.get("example_annotation", ""),
                "edit_id": suggestion.get("edit_id", ""),
                "pre_verification": suggestion.get("pre_verification", {}),
                "is_retry": suggestion.get("is_retry", False),
                "is_best_attempt": suggestion.get("is_best_attempt", False),
                "source": "scoring_feedback",
            })
            # P0.4: parallel rubric_edit_event log with the trigger field so
            # all version transitions can be attributed consistently (remove,
            # refiner apply, manual config edit all write here).
            try:
                save_project_data(sb, pid, "rubric_edit_event", {
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": st.session_state.get("selected_conversation"),
                    "from_version": rubric_dict.get("version"),
                    "to_version": saved_version,
                    "trigger": "refiner_proposal_applied",
                    "proposal_id": suggestion.get("edit_id"),
                    "edit_summary": {
                        "added_dims": [],
                        "removed_dims": [],
                        "modified_dims": [{
                            "criterion": crit_name,
                            "dimension_id": dim_id,
                            "before": actual_before,
                            "after": new_text,
                        }],
                    },
                })
            except Exception as _e:
                _log.warning("rubric_edit_event log (apply) failed: %s", _e)
        return True
    except Exception as e:
        _log.warning("save_rubric_history failed: %s", e)
        st.session_state.setdefault("_rubric_apply_warnings", []).append(
            f"Couldn't save the new rubric version for **{crit_name}**: {e}"
        )
        # Roll back the in-memory append so we don't desync from the DB.
        if rubric_history and rubric_history[-1] is new_version:
            rubric_history.pop()
        return False


def _infer_user_expected_grade(suggestion: dict[str, Any]) -> str | None:
    """Determine what grade the user expects after the rubric edit.

    Uses the grader_verdict that the refiner saw plus the feedback text to
    infer direction. Returns 'MET', 'NOT_MET', or None if we can't tell.
    """
    import re as _re
    feedback_text = (suggestion.get("feedback_text") or "").lower()
    grader_verdict = (suggestion.get("grader_verdict") or "").upper()

    # "user says [it's/actually/...] MET|NOT_MET" -- regex match so we don't
    # care about intervening words/punctuation.
    m = _re.search(r"user\s+says\b[^.]*?\b(not[_ ]met|met)\b", feedback_text)
    if m:
        grade = m.group(1).replace(" ", "_").upper()
        return "NOT_MET" if grade == "NOT_MET" else "MET"

    # Early-return patterns that indicate we can't verify against a grade
    if "user wants to remove" in feedback_text or "user says remove" in feedback_text:
        return None
    # Both the new `rubric_fine` slug and the legacy `just_right` slug mean
    # "user endorses current wording" -- nothing to verify against.
    if ("rubric_fine" in feedback_text
            or "just_right" in feedback_text
            or "just right" in feedback_text):
        return None
    if "prioritize" in feedback_text or "both matter" in feedback_text:
        return None
    if "wording_subjective" in feedback_text or "drafts_varying" in feedback_text:
        # Oscillation: user complaint is about stability, not direction.
        # Refiner's operationalization edit can't be verified against a
        # specific expected grade -- "aligned" just means the edit ran.
        return None

    # Calibration: "too strict" means user thinks it should be MET, not NOT_MET
    if "too_strict" in feedback_text or "too strict" in feedback_text:
        return "MET"
    if "too_vague" in feedback_text or "too vague" in feedback_text:
        # Vague goes either way; default to the opposite of grader since user
        # flagged it as an issue.
        if grader_verdict == "NOT_MET":
            return "MET"
        if grader_verdict == "MET":
            return "NOT_MET"

    # Disagree patterns: user wants the OPPOSITE of what the grader said
    if "disagree" in feedback_text or "wrong" in feedback_text:
        if grader_verdict == "NOT_MET":
            return "MET"
        if grader_verdict == "MET":
            return "NOT_MET"

    # Agree patterns: user wants the SAME as what the grader said. Includes
    # the low-confidence panel's `grade_correct` action (fired when the user
    # confirms the grader's verdict despite the low confidence).
    if ("agrees" in feedback_text or "confirms" in feedback_text
            or "working_on_it" in feedback_text or "i'm working on it" in feedback_text
            or "is correct" in feedback_text
            or "grade_correct" in feedback_text):
        return grader_verdict or None

    return None


def _verify_rubric_edit(
    suggestion: dict[str, Any], updated_rubric: dict[str, Any],
) -> None:
    """Re-grade the latest draft with the updated rubric (synchronously, with
    a visible spinner) and show the result. If mismatched on the first attempt,
    auto-retry the refiner with context about what failed."""
    messages = st.session_state.get("messages") or []
    draft_text = None
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        dt = extract_primary_draft_text(m.get("content") or "")
        if dt:
            draft_text = dt
            break
    if not draft_text:
        return

    feedback_text = suggestion.get("feedback_text", "")
    crit_name = suggestion.get("criterion_name", "")
    dim_id = suggestion.get("dimension_id", "")
    old_text = suggestion.get("before_wording") or suggestion.get("old_text", "")
    new_text = suggestion.get("after_wording") or suggestion.get("new_text", "")
    rubric_copy = json.loads(json.dumps(updated_rubric))
    verification_ref = st.session_state.setdefault("rubric_edit_verifications", [])
    suggestions_ref = st.session_state.setdefault("rubric_edit_suggestions", [])
    is_retry = suggestion.get("is_retry", False)

    sb_ref = st.session_state.get("supabase")
    pid_ref = st.session_state.get("current_project_id")

    try:
        from rubric_writer.draft_grading import grade_draft_sync

        with st.spinner("Verifying the edit by re-grading your latest draft..."):
            grades, latency_ms, err = grade_draft_sync(
                rubric_dict=rubric_copy,
                draft_text=draft_text,
            )

        if err or not grades:
            verification_ref.append({
                "criterion_name": crit_name,
                "dimension_id": dim_id,
                "status": "error",
                "error": err,
                "timestamp": datetime.now().isoformat(),
            })
            st.error(f"Verification failed: {err}")
            return

        # Find the specific dimension grade
        new_grade_val = None
        for c in grades.get("grades") or []:
            if (c.get("criterion_name") or "").strip() != crit_name.strip():
                continue
            for d in c.get("dimension_grades") or []:
                if dim_id and (d.get("dimension_id") or "").strip() == dim_id.strip():
                    new_grade_val = (d.get("grade") or "").upper()
                    break
            break

        user_expected = _infer_user_expected_grade(suggestion)

        if user_expected is None or new_grade_val is None:
            # Can't verify (e.g. "just right" / "remove" / priority changes)
            verification_ref.append({
                "criterion_name": crit_name,
                "dimension_id": dim_id,
                "status": "skipped",
                "reason": "no clear grade expectation from feedback",
                "new_grade": new_grade_val,
                "timestamp": datetime.now().isoformat(),
            })
            st.info("Edit applied. (Verification skipped: no specific grade expectation for this feedback type.)")
            return

        aligned = new_grade_val == user_expected

        result = {
            "criterion_name": crit_name,
            "dimension_id": dim_id,
            "new_grade": new_grade_val,
            "user_expected": user_expected,
            "aligned": aligned,
            "status": "verified",
            "feedback_text": feedback_text,
            "is_retry": is_retry,
            "timestamp": datetime.now().isoformat(),
        }
        verification_ref.append(result)

        if sb_ref and pid_ref:
            try:
                save_project_data(sb_ref, pid_ref, "rubric_edit_verification", result)
            except Exception:
                pass

        # Inline feedback right where the user clicked Apply
        if aligned:
            st.success(
                f"Verified: re-grading gives **{new_grade_val}**, matching your feedback."
            )
        else:
            st.warning(
                f"Mismatch: re-grading gives **{new_grade_val}** but your feedback expected "
                f"**{user_expected}**."
                + (" Trying a different edit..." if not is_retry else "")
            )

        # Auto-retry on mismatch (first attempt only)
        if not aligned and not is_retry:
            with st.spinner("Trying a different edit..."):
                _run_retry_refinement(
                    suggestion=suggestion, rubric_copy=rubric_copy,
                    feedback_text=feedback_text, old_text=old_text,
                    new_text=new_text, new_grade_val=new_grade_val,
                    user_expected=user_expected,
                    suggestions_ref=suggestions_ref,
                    sb_ref=sb_ref, pid_ref=pid_ref,
                )

    except Exception as e:
        _log.warning("rubric edit verification failed: %s", e)
        st.error(f"Verification error: {e}")


def _run_retry_refinement(
    *,
    suggestion: dict[str, Any],
    rubric_copy: dict[str, Any],
    feedback_text: str,
    old_text: str,
    new_text: str,
    new_grade_val: str,
    user_expected: str,
    suggestions_ref: list,
    sb_ref: Any,
    pid_ref: Any,
) -> None:
    """Call the refiner again with context about why the previous edit failed.
    Queues a new suggestion with is_retry=True."""
    from rubric_writer.config import MODEL_PRIMARY
    from rubric_writer.api_client import _api_call_with_retry
    import uuid as _uuid

    rubric_json = json.dumps(rubric_copy.get("rubric", []), ensure_ascii=False, indent=2)
    retry_note = (
        "NOTE: You previously suggested this edit, which was applied to the rubric:\n"
        f'  BEFORE: "{old_text}"\n'
        f'  AFTER: "{new_text}"\n'
        f"When the grader re-graded the same draft with your edit applied, "
        f"it still gave {new_grade_val}, but the user expected {user_expected}.\n"
        "Your previous edit was not sufficient. Propose a DIFFERENT edit that "
        "would move the grader's verdict in the expected direction.\n\n"
        f"CURRENT RUBRIC (with your previous edit already applied):\n{rubric_json}\n\n"
    )
    # Reuse the main refiner inputs from the original suggestion
    user_prompt = retry_note + _build_refiner_user_prompt(
        criterion_wording=new_text,  # current wording after first edit
        grader_verdict=new_grade_val,
        grader_confidence="high",
        grader_evidence=suggestion.get("grader_evidence", ""),
        grader_ambiguity_note="",
        user_verdict=f"disagrees with grader -- expects {user_expected}",
        draft_excerpt="(see previous attempt)",
    )

    try:
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=1500,
            system=REFINER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        retry_suggestion, status = _parse_refiner_response(text)
        if status == "no_change_needed" or retry_suggestion is None:
            st.warning("The refiner couldn't find a better edit. You may need to edit the rubric manually.")
            return

        retry_suggestion.setdefault("criterion_name", suggestion.get("criterion_name", ""))
        retry_suggestion.setdefault("dimension_id", suggestion.get("dimension_id", ""))
        retry_suggestion["feedback_text"] = feedback_text
        retry_suggestion["drift_kind"] = suggestion.get("drift_kind", "")
        retry_suggestion["timestamp"] = datetime.now().isoformat()
        retry_suggestion["status"] = "pending"
        retry_suggestion["is_retry"] = True
        retry_suggestion["edit_id"] = str(_uuid.uuid4())
        retry_suggestion["parse_status"] = status
        retry_suggestion["grader_evidence"] = suggestion.get("grader_evidence", "")
        retry_suggestion["grader_verdict"] = new_grade_val
        retry_suggestion["previous_attempt"] = {
            "before_wording": old_text,
            "after_wording": new_text,
            "grader_grade": new_grade_val,
            "user_expected": user_expected,
        }
        suggestions_ref.append(retry_suggestion)

        try:
            from rubric_writer.metrics import log_edit_shown
            log_edit_shown(
                edit_id=retry_suggestion["edit_id"],
                criterion_id=retry_suggestion.get("criterion_name", ""),
                dimension_id=retry_suggestion.get("dimension_id", ""),
                scope_change=retry_suggestion.get("scope_change", ""),
            )
        except Exception:
            pass

        if sb_ref and pid_ref:
            try:
                save_project_data(sb_ref, pid_ref, "rubric_edit_suggestion", retry_suggestion)
            except Exception:
                pass
    except Exception as e:
        _log.warning("rubric edit retry failed: %s", e)
