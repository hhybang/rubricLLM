"""Headless pipeline functions for the synthetic user simulation.

Adapted from app.py — removes all Streamlit dependencies (st.session_state,
st.empty, st.info, st.error, streaming) and replaces with logging + direct
API calls.
"""

import json
import logging
import random
import re
import time
import copy
from datetime import datetime
from typing import Optional

import anthropic
from scipy.stats import kendalltau

from prompts import (
    CHAT_build_system_prompt,
    RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
    RUBRIC_infer_only_user_prompt,
    RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT,
    RUBRIC_extract_dps_user_prompt,
    RUBRIC_FINAL_INFER_SYSTEM_PROMPT,
    RUBRIC_final_infer_user_prompt,
    RUBRIC_compare_to_coldstart_prompt,
    RUBRIC_apply_suggestion_prompt,
    RUBRIC_suggest_changes_from_feedback_prompt,
    GRADING_generate_writing_task_prompt,
    GRADING_generate_draft_from_rubric_prompt,
    GRADING_generate_draft_from_coldstart_prompt,
    GRADING_generate_draft_generic_prompt,
    GRADING_rubric_judge_prompt,
    GRADING_rubric_judge_3draft_prompt,
    GRADING_generic_judge_prompt,
    ALIGNMENT_diagnostic_suggest_and_apply_prompt,
    ALIGNMENT_generate_annotated_draft_prompt,
    ALIGNMENT_verify_suggested_rubric_prompt,
    GOLD_DRAFT_GENERATION_PROMPT,
    DRAFT_REGENERATE_SYSTEM_PROMPT,
    DRAFT_regenerate_prompt,
    PREFERENCE_COVERAGE_PROMPT,
    DECOMPOSE_PREFERENCES_PROMPT,
    COVERAGE_CHECK_PROMPT,
)

log = logging.getLogger("sim")

# ── Anthropic client (system-side) ───────────────────────────────────────────
_client = anthropic.Anthropic()


# ═════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ═════════════════════════════════════════════════════════════════════════════

def api_call_with_retry(max_retries=3, base_delay=2, **kwargs):
    """Wrapper around client.messages.create with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return _client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"API {e.status_code}, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                continue
            raise
        except anthropic.APIConnectionError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"Connection error, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                continue
            raise


def api_call_streaming(max_retries=3, base_delay=2, **kwargs) -> str:
    """Streaming API call — required for extended thinking which can exceed 10min.

    Returns the text content (ignoring thinking blocks).
    Uses get_final_message() for robust stream accumulation.
    """
    for attempt in range(max_retries + 1):
        try:
            with _client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
            return "".join(
                block.text for block in message.content
                if block.type == "text"
            ).strip()
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"API {e.status_code}, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                continue
            raise
        except anthropic.APIConnectionError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"Connection error, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                continue
            raise


def _extract_text(resp) -> str:
    """Extract text content from an Anthropic API response."""
    return "".join(b.text for b in resp.content if b.type == "text")


def _parse_json(text: str, key: Optional[str] = None):
    """Extract JSON from LLM response text."""
    text = text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if key is None or key in result:
                return result
        except json.JSONDecodeError:
            pass

    # Try finding the last valid JSON object (models often put analysis before JSON)
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            for j in range(i, -1, -1):
                if text[j] == '{':
                    try:
                        result = json.loads(text[j:i + 1])
                        if key is None or key in result:
                            return result
                    except json.JSONDecodeError:
                        continue
            break

    # Fallback: greedy match
    if key:
        pattern = r'\{.*"' + re.escape(key) + r'".*\}'
        match = re.search(pattern, text, re.DOTALL)
    else:
        match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _parse_json_array(text: str):
    """Extract a JSON array from LLM response text."""
    match = re.search(r'\[[\s\S]*\]', text.strip())
    if match:
        return json.loads(match.group())
    return json.loads(text.strip())


def build_conversation_text(messages: list) -> str:
    """Build numbered conversation text from messages for rubric/DP prompts.

    Simplified from app.py _build_conversation_text — handles basic user/assistant
    messages plus special simulation message types.
    """
    conversation_text = ""
    msg_num = 1
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if msg.get("_synthetic_changelog"):
            conversation_text += f"\n\n[Rubric Version Change]\n{content}"
            continue

        enriched = content

        # Alignment diagnostic messages
        if msg.get("is_alignment_diagnostic"):
            parts = [content]
            rs = msg.get("rubric_suggestion", {})
            if rs and rs.get("suggestion_reasons"):
                parts.append("\n[Rubric Suggestion Reasons]")
                for name, reason in rs["suggestion_reasons"].items():
                    parts.append(f"  - {name}: {reason}")
            enriched = "\n".join(parts)

        # Criteria classification log
        elif msg.get("is_criteria_classification_log"):
            cd = msg.get("classification_data", {})
            parts = [content]
            if cd.get("classifications"):
                parts.append("\n[Criteria Classifications]")
                for cn, cat in cd["classifications"].items():
                    parts.append(f"  - {cn}: {cat}")
            enriched = "\n".join(parts)

        conversation_text += f"\n\n[Message #{msg_num} — {role}]\n{enriched}"
        msg_num += 1

    return conversation_text.strip()


def _rubric_to_json_serializable(obj):
    """Deep copy with sets → lists for JSON serialization."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _rubric_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rubric_to_json_serializable(v) for v in obj]
    return obj


def _criterion_changed(old: dict, new: dict) -> bool:
    """Check if a criterion has changed between old and new versions."""
    if old.get("description") != new.get("description"):
        return True
    if old.get("priority") != new.get("priority"):
        return True
    if old.get("category") != new.get("category"):
        return True
    old_dims = [d.get("label", "") for d in old.get("dimensions", [])]
    new_dims = [d.get("label", "") for d in new.get("dimensions", [])]
    return old_dims != new_dims


# ═════════════════════════════════════════════════════════════════════════════
# Goal-Based Termination Helpers
# ═════════════════════════════════════════════════════════════════════════════

def extract_draft_from_response(assistant_response: str) -> Optional[str]:
    """Extract draft content from <draft>...</draft> tags in an assistant response.

    Returns the draft text, or None if no draft tags are found.
    """
    match = re.search(r'<draft>(.*?)</draft>', assistant_response, re.DOTALL)
    if match:
        draft = match.group(1).strip()
        return draft if draft else None
    return None


def generate_gold_draft(persona, task: str, model: str = "claude-sonnet-4-6") -> str:
    """Auto-generate an ideal/gold draft using ALL persona preferences.

    Called when no hand-authored gold_draft exists in the persona definition.
    Uses model_light for cost efficiency since all preferences are provided explicitly.
    """
    prompt = GOLD_DRAFT_GENERATION_PROMPT(persona, task)
    try:
        resp = api_call_with_retry(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(resp).strip()
    except Exception as e:
        log.error(f"Gold draft generation failed: {e}")
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# Rubric Inference Pipeline (headless)
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_rubric_priorities(rubric_data: dict) -> dict:
    """Ensure rubric criteria have unique sequential priorities (1..N).

    LLMs sometimes output duplicate or non-sequential priorities.
    This preserves the relative ordering but assigns clean 1..N ranks.
    """
    criteria = rubric_data.get("rubric", [])
    if not criteria:
        return rubric_data
    # Sort by existing priority (ties broken by original order)
    indexed = [(i, c) for i, c in enumerate(criteria)]
    indexed.sort(key=lambda x: (x[1].get("priority", 999), x[0]))
    for rank, (_, c) in enumerate(indexed, 1):
        c["priority"] = rank
    return rubric_data


def headless_infer_rubric(messages: list, previous_rubric: Optional[dict] = None,
                          model: str = "claude-opus-4-6", version: int = 1) -> Optional[dict]:
    """Infer a rubric from conversation (Step 1 of 5-step flow).

    Returns rubric_data dict or None on failure.
    """
    conversation_text = build_conversation_text(messages)
    previous_rubric_json = ""
    if previous_rubric and previous_rubric.get("rubric"):
        previous_rubric_json = json.dumps(previous_rubric["rubric"], ensure_ascii=False, indent=2)

    user_prompt = RUBRIC_infer_only_user_prompt(conversation_text, previous_rubric_json)

    for attempt in range(3):
        try:
            text = api_call_streaming(
                model=model,
                max_tokens=32000,
                system=RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "enabled", "budget_tokens": 10000},
            )
            rubric_data = _parse_json(text, key="rubric")
            _normalize_rubric_priorities(rubric_data)
            rubric_data["version"] = version
            rubric_data["source"] = "inferred"
            log.info(f"Inferred rubric v{version} with {len(rubric_data.get('rubric', []))} criteria")
            return rubric_data
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            log.error(f"Rubric inference failed: {e}")
            return None
    return None


def headless_classify_criteria(rubric_data: dict, coldstart_text: str,
                               messages: list, model: str = "claude-opus-4-6") -> dict:
    """LLM classifies rubric criteria as stated vs absent relative to cold-start text.

    Returns dict with per-criterion classification.
    """
    rubric_json = json.dumps(rubric_data.get("rubric", []), ensure_ascii=False, indent=2)
    conversation_text = build_conversation_text(messages)
    prompt = RUBRIC_compare_to_coldstart_prompt(rubric_json, coldstart_text, conversation_text)

    try:
        resp = api_call_with_retry(
            model=model, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(resp)
        return _parse_json(text)
    except Exception as e:
        log.error(f"Criteria classification failed: {e}")
        return {}


def headless_extract_dps(messages: list, rubric_json: str,
                         classification_feedback: dict,
                         model: str = "claude-opus-4-6") -> Optional[dict]:
    """Extract decision points from conversation (Step 3 of 5-step flow)."""
    conversation_text = build_conversation_text(messages)
    classification_feedback_json = json.dumps(classification_feedback, ensure_ascii=False, indent=2)
    user_prompt = RUBRIC_extract_dps_user_prompt(conversation_text, rubric_json, classification_feedback_json)

    for attempt in range(3):
        try:
            text = api_call_streaming(
                model=model,
                max_tokens=16000,
                system=RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "enabled", "budget_tokens": 10000},
            )
            parsed = _parse_json(text)
            if "decision_points" in parsed:
                return {"parsed_data": parsed}
            elif "parsed_data" in parsed:
                return parsed
            return {"parsed_data": {"decision_points": []}}
        except Exception as e:
            if ("overloaded" in str(e).lower() or "rate" in str(e).lower()) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            log.error(f"DP extraction failed: {e}")
            return None
    return None


def headless_infer_final_rubric(messages: list, rubric_json: str,
                                 classification_feedback_json: str,
                                 corrected_dps_json: str,
                                 coldstart_text: str = "",
                                 model: str = "claude-opus-4-6",
                                 version: int = 2) -> Optional[dict]:
    """Final rubric incorporating all feedback (Step 5 of 5-step flow)."""
    conversation_text = build_conversation_text(messages)
    user_prompt = RUBRIC_final_infer_user_prompt(
        conversation_text, rubric_json, classification_feedback_json,
        corrected_dps_json, coldstart_text,
    )

    for attempt in range(3):
        try:
            text = api_call_streaming(
                model=model,
                max_tokens=32000,
                system=RUBRIC_FINAL_INFER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "enabled", "budget_tokens": 10000},
            )
            rubric_data = _parse_json(text, key="rubric")
            change_explanation = rubric_data.pop("change_explanation", "")
            refinement_summary = rubric_data.get("refinement_summary", "")
            rubric_data["version"] = version
            rubric_data["source"] = "inferred_final"
            rubric_data["_change_explanation"] = change_explanation
            rubric_data["_refinement_summary"] = refinement_summary
            log.info(f"Final rubric v{version} inferred")
            return rubric_data
        except Exception as e:
            if ("overloaded" in str(e).lower()) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            log.error(f"Final rubric inference failed: {e}")
            return None
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Chat (headless, non-streaming)
# ═════════════════════════════════════════════════════════════════════════════

def headless_chat_response(messages: list, rubric_data: Optional[dict],
                           model: str = "claude-sonnet-4-6",
                           max_tokens: int = 4000) -> str:
    """Generate a chat response given conversation history and rubric.

    Returns the assistant response text (may contain <draft>...</draft> tags).
    """
    rubric_list = rubric_data.get("rubric", []) if rubric_data else []
    system_prompt = CHAT_build_system_prompt(rubric_list)

    api_messages = []
    for msg in messages:
        if msg.get("role") in ("user", "assistant"):
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        resp = api_call_with_retry(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=api_messages,
        )
        text = _extract_text(resp)
        # Strip probe signal tags if present
        text = re.sub(r'\s*<probe_signal>.*?</probe_signal>\s*', '', text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        log.error(f"Chat response failed: {e}")
        return "[Error generating response]"


# ═════════════════════════════════════════════════════════════════════════════
# Alignment Diagnostic (headless)
# ═════════════════════════════════════════════════════════════════════════════

def headless_alignment_diagnostic(
    drafts: dict,
    writing_task: str,
    user_ranking: list,
    user_reason: str,
    rubric_data: dict,
    messages: list,
    coldstart_text: str,
    shuffle_order: list,
    model_primary: str = "claude-opus-4-6",
    model_light: str = "claude-sonnet-4-6",
) -> dict:
    """Run full alignment diagnostic (judge → classify → suggest → verify).

    Args:
        drafts: {"rubric": str, "generic": str, "preference": str}
        writing_task: the writing task description
        user_ranking: ranked source keys, e.g. ["rubric", "preference", "generic"]
        user_reason: free-text reason
        rubric_data: full rubric dict
        messages: conversation messages
        coldstart_text: user's cold-start text
        shuffle_order: list of (label, source) tuples
    Returns:
        result dict compatible with evaluate.ipynb expectations.
    """
    rubric_draft = drafts.get("rubric", "")
    generic_draft = drafts.get("generic", "")
    preference_draft = drafts.get("preference", "")
    is_3draft = bool(preference_draft)

    rubric_json = json.dumps(_rubric_to_json_serializable(rubric_data), indent=2)
    rubric_list = rubric_data.get("rubric", [])

    conv_text = build_conversation_text(messages)
    conv_context = conv_text[-3000:] if len(conv_text) > 3000 else conv_text

    pipeline_messages = []

    # ── Rubric judge ─────────────────────────────────────────────────────
    log.info("Running rubric judge...")
    rubric_judge_result = None
    try:
        if is_3draft:
            # Pass drafts in shuffled order so judge is blind
            source_to_draft = {"rubric": rubric_draft, "generic": generic_draft, "preference": preference_draft}
            shuffled_drafts = [source_to_draft[src] for _, src in shuffle_order]
            prompt = GRADING_rubric_judge_3draft_prompt(
                shuffled_drafts[0], shuffled_drafts[1], shuffled_drafts[2], rubric_json, conv_context
            )
        else:
            prompt = GRADING_rubric_judge_prompt(rubric_draft, generic_draft, rubric_json, conv_context)
        pipeline_messages.append({"role": "user", "content": prompt})
        resp = api_call_with_retry(model=model_primary, max_tokens=2000, messages=pipeline_messages)
        judge_text = _extract_text(resp)
        pipeline_messages.append({"role": "assistant", "content": judge_text})
        rubric_judge_result = _parse_json(judge_text)
    except Exception as e:
        log.error(f"Rubric judge failed: {e}")

    # ── Generic judge ────────────────────────────────────────────────────
    log.info("Running generic judge...")
    generic_judge_result = None
    try:
        prompt = GRADING_generic_judge_prompt(rubric_draft, generic_draft)
        resp = api_call_with_retry(
            model=model_primary, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(resp)
        generic_judge_result = _parse_json(text)
    except Exception as e:
        log.error(f"Generic judge failed: {e}")

    # ── Classify criteria ────────────────────────────────────────────────
    criteria_analysis = []
    if rubric_judge_result and rubric_judge_result.get("per_criterion"):
        for crit in rubric_judge_result["per_criterion"]:
            rubric_score = crit.get("draft_a_score", 3)
            generic_score = crit.get("draft_b_score", 3)
            preference_score = crit.get("draft_c_score") if is_3draft else None
            gap = rubric_score - generic_score

            if is_3draft and preference_score is not None:
                if rubric_score >= generic_score and rubric_score >= preference_score and gap > 0:
                    classification = "DIFFERENTIATING"
                elif preference_score > rubric_score and preference_score >= generic_score:
                    classification = "PREFERENCE_GAP"
                elif generic_score > rubric_score and generic_score >= (preference_score or 0):
                    classification = "UNDERPERFORMING"
                else:
                    classification = "REDUNDANT"
            else:
                if gap > 0:
                    classification = "DIFFERENTIATING"
                elif gap < 0:
                    classification = "UNDERPERFORMING"
                else:
                    classification = "REDUNDANT"

            entry = {
                "name": crit.get("criterion_name", "Unknown"),
                "rubric_score": rubric_score,
                "generic_score": generic_score,
                "gap": gap,
                "classification": classification,
                "reasoning": crit.get("reasoning", ""),
            }
            if preference_score is not None:
                entry["preference_score"] = preference_score
            criteria_analysis.append(entry)

    # Assign priorities from rubric
    priority_map = {c.get("name", "").lower().strip(): c.get("priority", 99) for c in rubric_list}
    for ca in criteria_analysis:
        ca["priority"] = priority_map.get(ca["name"].lower().strip(), 99)
    criteria_analysis.sort(key=lambda x: x["priority"])

    # ── Suggest rubric improvements ──────────────────────────────────────
    suggestion_text = ""
    suggested_rubric = None
    suggestion_reasons = {}

    # Skip suggestions if rubric draft was already ranked first — it's working
    if user_ranking and user_ranking[0] == "rubric":
        log.info("Rubric draft ranked first — skipping rubric suggestions")
    else:
        log.info("Generating rubric suggestions...")
        source_names = {
            "rubric": "rubric-guided",
            "generic": "generic (no rubric)",
            "preference": "preference-based (from original stated preferences)",
        }
        ranking_desc = "The user ranked the drafts: " + " > ".join(
            f"{i+1}. {source_names.get(s, s)}" for i, s in enumerate(user_ranking)
        )

        try:
            rubric_judge_json = json.dumps(rubric_judge_result, indent=2) if rubric_judge_result else "{}"
            _preferred_draft_sim = drafts.get(user_ranking[0], "") if user_ranking else ""
            suggest_prompt = ALIGNMENT_diagnostic_suggest_and_apply_prompt(
                rubric_json, rubric_judge_json, ranking_desc, user_reason,
                preferred_draft=_preferred_draft_sim,
                rubric_guided_draft=rubric_draft,
                writing_task=writing_task,
            )
            pipeline_messages.append({"role": "user", "content": suggest_prompt})
            resp = api_call_with_retry(model=model_light, max_tokens=4000, messages=pipeline_messages)
            sa_raw = _extract_text(resp).strip()
            pipeline_messages.append({"role": "assistant", "content": sa_raw})

            sa_parsed = _parse_json(sa_raw)
            all_reasons = sa_parsed.get("reasons", {}) or sa_parsed.get("criteria", {})
            suggestion_reasons = {k: v for k, v in all_reasons.items() if v}
            rubric_arr = sa_parsed.get("rubric", [])
            if isinstance(rubric_arr, list) and rubric_arr:
                suggested_rubric = rubric_arr

            # Filter out "keep unchanged" reasons
            keep_phrases = ["keep unchanged", "no change", "working well", "kept as-is",
                            "keep as-is", "no changes needed", "performing well"]
            suggestion_reasons = {
                k: v for k, v in suggestion_reasons.items()
                if not any(p in v.lower() for p in keep_phrases)
            }

            # Check if suggested_rubric actually differs
            if suggested_rubric and rubric_list:
                orig_map = {c.get("name", "").lower().strip(): c for c in rubric_list}
                new_map = {c.get("name", "").lower().strip(): c for c in suggested_rubric}
                has_change = False
                for nk in new_map:
                    if nk not in orig_map:
                        has_change = True
                        break
                if not has_change:
                    for ok in orig_map:
                        if ok not in new_map:
                            has_change = True
                            break
                if not has_change:
                    for ck in orig_map:
                        if ck in new_map and _criterion_changed(orig_map[ck], new_map[ck]):
                            has_change = True
                            break
                if not has_change:
                    suggested_rubric = None

            if suggestion_reasons:
                suggestion_text = "\n".join(f"- **{k}**: {v}" for k, v in suggestion_reasons.items())
        except Exception as e:
            log.error(f"Suggestion generation failed: {e}")

    # ── Build result ─────────────────────────────────────────────────────
    result = {
        "timestamp": datetime.now().isoformat(),
        "writing_task": writing_task,
        "drafts": drafts,
        "shuffle_order": shuffle_order,
        "user_preference": user_ranking[0] if user_ranking else "tie",
        "user_ranking": user_ranking,
        "user_reason": user_reason,
        "rubric_version": rubric_data.get("version"),
        "criteria_analysis": criteria_analysis,
        "rubric_judge_result": rubric_judge_result,
        "generic_judge_result": generic_judge_result,
        "suggestion_text": suggestion_text,
        "suggestion_reasons": suggestion_reasons,
        "suggested_rubric": suggested_rubric,
        "is_3draft": is_3draft,
        "pipeline_messages": pipeline_messages,
    }

    log.info(f"Diagnostic complete: {len(criteria_analysis)} criteria analyzed")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Test-Retest Reliability
# ═════════════════════════════════════════════════════════════════════════════

def headless_run_retest(diagnostic_result: dict, rubric_json: str,
                        model_primary: str = "claude-opus-4-6") -> dict:
    """Re-run rubric and generic judges for test-retest reliability."""
    drafts = diagnostic_result.get("drafts", {})
    draft_a = drafts.get("rubric", "")
    draft_b = drafts.get("generic", "")
    original_rubric = diagnostic_result.get("rubric_judge_result")
    original_generic = diagnostic_result.get("generic_judge_result")

    if not draft_a or not draft_b:
        return {}

    conv_context = ""  # simplified — not critical for retest

    retest_rubric = None
    retest_generic = None

    # Re-run rubric judge
    try:
        prompt = GRADING_rubric_judge_prompt(draft_a, draft_b, rubric_json, conv_context)
        resp = api_call_with_retry(model=model_primary, max_tokens=2000,
                                   messages=[{"role": "user", "content": prompt}])
        retest_rubric = _parse_json(_extract_text(resp))
    except Exception as e:
        log.error(f"Retest rubric judge failed: {e}")

    # Re-run generic judge
    try:
        prompt = GRADING_generic_judge_prompt(draft_a, draft_b)
        resp = api_call_with_retry(model=model_primary, max_tokens=2000,
                                   messages=[{"role": "user", "content": prompt}])
        retest_generic = _parse_json(_extract_text(resp))
    except Exception as e:
        log.error(f"Retest generic judge failed: {e}")

    # Compute metrics
    retest_data = {
        "timestamp": diagnostic_result.get("timestamp", ""),
        "retest_rubric_result": retest_rubric,
        "retest_generic_result": retest_generic,
        "original_rubric_result": original_rubric,
        "original_generic_result": original_generic,
        "metrics": {},
    }

    for label, orig, retest in [
        ("rubric", original_rubric, retest_rubric),
        ("generic", original_generic, retest_generic),
    ]:
        if not orig or not retest:
            continue
        orig_scores = {
            c.get("criterion_name", ""): (c.get("draft_a_score", 3), c.get("draft_b_score", 3))
            for c in orig.get("per_criterion", [])
        }
        retest_scores = {
            c.get("criterion_name", ""): (c.get("draft_a_score", 3), c.get("draft_b_score", 3))
            for c in retest.get("per_criterion", [])
        }
        run1, run2 = [], []
        for crit in orig_scores:
            if crit in retest_scores:
                run1.extend(orig_scores[crit])
                run2.extend(retest_scores[crit])
        retest_data["metrics"][f"{label}_run1_scores"] = run1
        retest_data["metrics"][f"{label}_run2_scores"] = run2
        if len(run1) >= 4:
            tau, p = kendalltau(run1, run2)
            retest_data["metrics"][f"{label}_tau"] = tau
            retest_data["metrics"][f"{label}_tau_p"] = p

    # Legacy key
    retest_data["metrics"]["retest_tau"] = retest_data["metrics"].get("rubric_tau")

    return retest_data


# ═════════════════════════════════════════════════════════════════════════════
# Draft Generation Helpers
# ═════════════════════════════════════════════════════════════════════════════

def generate_writing_task(messages: list, model: str = "claude-sonnet-4-6") -> str:
    """Generate a writing task based on conversation context."""
    conv_text = build_conversation_text(messages)
    conv_text = conv_text[-3000:] if len(conv_text) > 3000 else conv_text
    prompt = GRADING_generate_writing_task_prompt(conv_text)
    try:
        resp = api_call_with_retry(
            model=model, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(resp).strip()
    except Exception as e:
        log.error(f"Writing task generation failed: {e}")
        return "Write a short piece in the same domain as the conversation."


def generate_draft(prompt: str, model: str = "claude-sonnet-4-6", max_tokens: int = 2000) -> str:
    """Generate a single draft from a prompt."""
    try:
        resp = api_call_with_retry(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(resp).strip()
    except Exception as e:
        log.error(f"Draft generation failed: {e}")
        return "[Draft generation failed]"


def generate_three_drafts(writing_task: str, rubric_data: dict, coldstart_text: str,
                          model: str = "claude-sonnet-4-6") -> dict:
    """Generate rubric-guided, generic, and preference-based drafts."""
    rubric_json = json.dumps(rubric_data.get("rubric", []), indent=2)
    return {
        "rubric": generate_draft(GRADING_generate_draft_from_rubric_prompt(writing_task, rubric_json), model),
        "generic": generate_draft(GRADING_generate_draft_generic_prompt(writing_task), model),
        "preference": generate_draft(GRADING_generate_draft_from_coldstart_prompt(writing_task, coldstart_text), model),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Log Changes — rubric editing utilities
# ═════════════════════════════════════════════════════════════════════════════

def classify_rubric_edits(old_rubric_list: list, new_rubric_list: list) -> dict:
    """Diff two rubric criteria lists and classify every change by type.

    Adapted from app.py — pure function, no Streamlit dependencies.
    """
    old_map = {}
    for c in (old_rubric_list or []):
        key = c.get('name', '').lower().strip()
        old_map[key] = c

    new_map = {}
    for c in (new_rubric_list or []):
        key = c.get('name', '').lower().strip()
        new_map[key] = c

    edits = {
        "added": [],
        "removed": [],
        "reweighted": [],
        "reworded": [],
        "dimensions_changed": [],
    }

    for key, new_c in new_map.items():
        if key not in old_map:
            edits["added"].append({
                "name": new_c.get("name", ""),
                "description": new_c.get("description", ""),
                "weight": new_c.get("weight", new_c.get("priority", 0)),
            })
        else:
            old_c = old_map[key]

            old_w = old_c.get('weight', old_c.get('priority', 0))
            new_w = new_c.get('weight', new_c.get('priority', 0))
            if old_w != new_w:
                edits["reweighted"].append({
                    "name": new_c.get("name", ""),
                    "old_weight": old_w,
                    "new_weight": new_w,
                })

            old_desc = old_c.get('description', '')
            new_desc = new_c.get('description', '')
            if old_desc != new_desc:
                edits["reworded"].append({
                    "name": new_c.get("name", ""),
                    "field": "description",
                    "old": old_desc,
                    "new": new_desc,
                })

            old_dims = set(
                d.get('label', '').strip()
                for d in old_c.get('dimensions', [])
                if d.get('label', '').strip()
            )
            new_dims = set(
                d.get('label', '').strip()
                for d in new_c.get('dimensions', [])
                if d.get('label', '').strip()
            )
            if old_dims != new_dims:
                edits["dimensions_changed"].append({
                    "name": new_c.get("name", ""),
                    "added_dims": list(new_dims - old_dims),
                    "removed_dims": list(old_dims - new_dims),
                })

    for key, old_c in old_map.items():
        if key not in new_map:
            edits["removed"].append({
                "name": old_c.get("name", ""),
                "description": old_c.get("description", ""),
            })

    return edits


def format_edit_log_message(edit_classification: dict, old_version, new_version, source: str) -> str:
    """Format a rubric edit classification into a conversation log message.

    Adapted from app.py — pure function, no Streamlit dependencies.
    """
    lines = ["**Rubric Changes have been made:**"]

    edits = edit_classification

    if edits["added"]:
        for a in edits["added"]:
            lines.append(f"- **Added:** \"{a['name']}\"")

    if edits["removed"]:
        for r in edits["removed"]:
            lines.append(f"- **Removed:** \"{r['name']}\"")

    if edits["reweighted"]:
        for rw in edits["reweighted"]:
            lines.append(f"- **Reweighted:** \"{rw['name']}\" ({rw['old_weight']} → {rw['new_weight']})")

    if edits["reworded"]:
        for rw in edits["reworded"]:
            lines.append(f"- **Reworded:** \"{rw['name']}\" {rw['field']} changed")

    if edits["dimensions_changed"]:
        for dc in edits["dimensions_changed"]:
            parts = []
            if dc["added_dims"]:
                parts.append(f"+{', '.join(dc['added_dims'])}")
            if dc["removed_dims"]:
                parts.append(f"-{', '.join(dc['removed_dims'])}")
            lines.append(f"- **Dimensions:** \"{dc['name']}\" ({'; '.join(parts)})")

    if not any(edits[k] for k in edits):
        lines.append("- No substantive changes detected")

    log_data = json.dumps({
        "version_from": old_version,
        "version_to": new_version,
        "edits": edit_classification,
        "source": source,
    })
    lines.append(f"<!--RUBRIC_EDIT_LOG:{log_data}-->")

    return "\n".join(lines)


def headless_regenerate_draft_from_rubric_changes(
    original_rubric: list,
    updated_rubric: list,
    current_draft: str,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """Regenerate a draft based on rubric changes.

    Headless version of app.py's regenerate_draft_from_rubric_changes().
    Returns dict with revised_draft, change_summary, annotated_changes, etc.
    or {"error": "..."} on failure.
    """
    user_prompt = DRAFT_regenerate_prompt(original_rubric, updated_rubric, current_draft)

    try:
        resp = api_call_with_retry(
            model=model,
            max_tokens=16000,
            system=DRAFT_REGENERATE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        response_text = _extract_text(resp)
        json_match = re.search(r'\{', response_text)
        if json_match:
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(response_text, json_match.start())
            return result
        return {"error": "Could not parse regenerated draft from model response."}
    except json.JSONDecodeError as e:
        return {"error": f"Error parsing response: {e}"}
    except Exception as e:
        log.error(f"Draft regeneration failed: {e}")
        return {"error": str(e)}


# ── Preference Coverage Evaluation ──────────────────────────────────────────

def decompose_preferences(persona, model: str = "claude-sonnet-4-6") -> list:
    """Decompose a persona's preferences into a fixed list of atomic items.

    Called ONCE per persona before the iteration loop so that coverage
    evaluations across iterations use the same set of items.

    Returns list of dicts with 'preference' and 'source' keys,
    or empty list on failure.
    """
    prompt = DECOMPOSE_PREFERENCES_PROMPT(persona)
    try:
        resp = api_call_with_retry(
            model=model,
            max_tokens=2000,
            system="You are an objective evaluation judge. Be thorough and fair.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(resp)
        json_match = re.search(r'\{', text)
        if not json_match:
            log.error("Could not parse preference decomposition")
            return []
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(text, json_match.start())
        items = result.get("preference_items", [])
        log.info(f"Decomposed {persona.name}'s preferences into {len(items)} atomic items")
        return items
    except Exception as e:
        log.error(f"Preference decomposition failed: {e}")
        return []


def evaluate_preference_coverage(persona, rubric_criteria: list,
                                  model: str = "claude-sonnet-4-6",
                                  preference_items: list = None,
                                  dealbreaker_violated: bool = False) -> dict:
    """Evaluate what fraction of a persona's ground-truth preferences are
    covered by the current rubric.

    If preference_items is provided, uses that fixed list instead of
    re-decomposing (recommended for consistent cross-iteration comparison).

    Dealbreaker handling: dealbreaker items are excluded from the LLM coverage
    check. If no dealbreaker was violated during conversation, they get free
    points (counted as "covered"). If a dealbreaker WAS violated, they count
    as "not_covered" — the violation proves the rubric didn't guard against it.

    Returns dict with preference_items list and coverage metrics,
    or {"error": "..."} on failure.
    """
    empty = {"coverage_score": 0.0, "total_preferences": 0,
             "covered_count": 0, "partially_covered_count": 0,
             "not_covered_count": 0, "preference_items": []}

    if not rubric_criteria:
        return {**empty, "error": "No rubric criteria to evaluate"}

    # Separate dealbreaker items from core/hidden items
    non_dealbreaker_items = None
    dealbreaker_items = []
    if preference_items:
        non_dealbreaker_items = [p for p in preference_items if p.get("source") != "dealbreaker"]
        dealbreaker_items = [p for p in preference_items if p.get("source") == "dealbreaker"]

    # Only send non-dealbreaker items to LLM for coverage check
    if non_dealbreaker_items is not None:
        prompt = COVERAGE_CHECK_PROMPT(non_dealbreaker_items, rubric_criteria)
    elif preference_items is None:
        # No pre-decomposed items — use original prompt (includes all sources)
        # TODO: separate dealbreakers in the single-prompt path too
        prompt = PREFERENCE_COVERAGE_PROMPT(persona, rubric_criteria)
    else:
        prompt = COVERAGE_CHECK_PROMPT(non_dealbreaker_items, rubric_criteria)

    try:
        resp = api_call_with_retry(
            model=model,
            max_tokens=3000,
            system="You are an objective evaluation judge. Be thorough and fair.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(resp)
        json_match = re.search(r'\{', text)
        if not json_match:
            return {**empty, "error": "Could not parse coverage evaluation"}

        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(text, json_match.start())

        # Coverage score is computed over core + hidden items only
        items = result.get("preference_items", [])

        total = len(items)
        covered = sum(1 for i in items if i.get("coverage") == "covered")
        partial = sum(1 for i in items if i.get("coverage") == "partially_covered")
        not_covered = sum(1 for i in items if i.get("coverage") == "not_covered")

        score = (covered + 0.5 * partial) / total if total > 0 else 0.0

        result["preference_items"] = items
        result["total_preferences"] = total
        result["covered_count"] = covered
        result["partially_covered_count"] = partial
        result["not_covered_count"] = not_covered
        result["coverage_score"] = round(score, 3)

        # Dealbreakers are a separate binary metric — pass if none violated
        result["dealbreaker_pass"] = not dealbreaker_violated
        result["dealbreaker_count"] = len(dealbreaker_items)

        return result

    except json.JSONDecodeError as e:
        return {**empty, "error": f"JSON parse error: {e}"}
    except Exception as e:
        log.error(f"Preference coverage evaluation failed: {e}")
        return {**empty, "error": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# Suggest Rubric from Edit Feedback (headless)
# ═════════════════════════════════════════════════════════════════════════════

def headless_suggest_rubric_from_feedback(
    active_rubric: list,
    edited_rubric: list,
    edits_with_feedback: list,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Generate bullet-point suggestions for rubric changes based on user
    feedback on annotated draft edits.

    Args:
        active_rubric: the rubric before the user's manual edits
        edited_rubric: the user's edited rubric (used for draft regen)
        edits_with_feedback: list of dicts with reason, original_text,
            new_text, user_feedback
        model: LLM model to use

    Returns:
        suggestion text (bullet points), or empty string on failure.
    """
    active_json = json.dumps(active_rubric, ensure_ascii=False, indent=2)
    edited_json = json.dumps(edited_rubric, ensure_ascii=False, indent=2)
    prompt = RUBRIC_suggest_changes_from_feedback_prompt(
        active_json, edited_json, edits_with_feedback,
    )
    try:
        resp = api_call_with_retry(
            model=model, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(resp).strip()
    except Exception as e:
        log.error(f"Suggest rubric from feedback failed: {e}")
        return ""


def headless_apply_rubric_suggestion(
    active_rubric: list,
    edited_rubric: list,
    suggestion_text: str,
    model: str = "claude-sonnet-4-6",
) -> Optional[list]:
    """Apply bullet-point suggestions to produce a modified rubric JSON array.

    Args:
        active_rubric: the rubric before the user's manual edits
        edited_rubric: the user's edited rubric
        suggestion_text: bullet-point suggestions from suggest step
        model: LLM model to use

    Returns:
        Modified rubric criteria list, or None on failure.
    """
    active_json = json.dumps(active_rubric, ensure_ascii=False, indent=2)
    edited_json = json.dumps(edited_rubric, ensure_ascii=False, indent=2)
    prompt = RUBRIC_apply_suggestion_prompt(active_json, edited_json, suggestion_text)
    try:
        resp = api_call_with_retry(
            model=model, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(resp).strip()
        return _parse_json_array(text)
    except Exception as e:
        log.error(f"Apply rubric suggestion failed: {e}")
        return None


