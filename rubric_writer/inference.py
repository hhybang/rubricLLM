"""Rubric inference pipeline (infer-only, DPs, final rubric)."""
from rubric_writer.imports import *
from rubric_writer.config import MODEL_PRIMARY, client
from rubric_writer.persistence import (
    load_rubric_history,
    save_rubric_history,
    invalidate_rubric_cache,
    next_version_number,
    get_active_rubric,
    _build_conversation_text,
)


def _validate_and_filter_rubric(rubric_data, *, context_label: str) -> tuple[int, int, int]:
    """Drop dimensions that are invalid for the drift/grading pipeline, and
    any criterion that becomes empty as a result. Returns
    (dropped_insufficient, dropped_empty_text, dropped_crits).

    Rules, all applied in one pass:
      - INSUFFICIENT_EVIDENCE escape hatch -> drop.
      - Dim with neither `label` nor `description` text -> drop. The grader
        prompt substitutes these fields into the rubric it sends to Sonnet;
        an empty dim produces an "ambiguity note" that literally says
        "has no label or description," triggers a low_confidence drift
        panel, and the user sees a broken row. Validating here keeps the
        inferred rubric clean on first render.
      - Criterion left with zero dims -> drop.
    """
    dropped_insufficient = 0
    dropped_empty_text = 0
    for crit in rubric_data.get("rubric") or []:
        kept = []
        for dim in crit.get("dimensions") or []:
            ev = (dim.get("evidence") or "").strip().upper()
            if ev == "INSUFFICIENT_EVIDENCE":
                dropped_insufficient += 1
                continue
            label = (dim.get("label") or "").strip()
            description = (dim.get("description") or "").strip()
            if not label and not description:
                dropped_empty_text += 1
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "[%s] dropping dim %r under crit %r: empty label + empty description",
                    context_label, dim.get("id") or "(no id)",
                    (crit.get("name") or "(no crit name)"),
                )
                continue
            kept.append(dim)
        crit["dimensions"] = kept
    before_crit = len(rubric_data.get("rubric") or [])
    rubric_data["rubric"] = [
        c for c in (rubric_data.get("rubric") or []) if c.get("dimensions")
    ]
    dropped_crits = max(0, before_crit - len(rubric_data["rubric"]))
    if dropped_insufficient or dropped_empty_text or dropped_crits:
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[%s] filter: dropped %d insufficient-evidence, %d empty-text dim(s)%s.",
            context_label, dropped_insufficient, dropped_empty_text,
            f" and {dropped_crits} now-empty criterion" if dropped_crits else "",
        )
    return dropped_insufficient, dropped_empty_text, dropped_crits

def infer_rubric_only(messages):
    """Infer a rubric from conversation WITHOUT extracting decision points.

    Step 1 of the 5-step flow. DPs are extracted separately in step 3.
    Returns:
        dict with rubric data (no 'inference_decision_points'), or None on failure.
    """
    conversation_text = _build_conversation_text(messages)

    # Get the current active rubric to build upon
    active_rubric_dict, _, _ = get_active_rubric()
    previous_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
    previous_rubric_version = active_rubric_dict.get("version", 1) if active_rubric_dict else 1

    if previous_rubric:
        st.info(f"Using previous rubric v{previous_rubric_version}")
        previous_rubric_json = json.dumps(previous_rubric, ensure_ascii=False, indent=2)
    else:
        st.info("No previous rubric found - creating new rubric from scratch")
        previous_rubric_json = ""

    system_prompt = RUBRIC_INFER_ONLY_SYSTEM_PROMPT
    user_prompt = RUBRIC_infer_only_user_prompt(conversation_text, previous_rubric_json)

    max_retries = 3
    retry_delay = 5
    progress_placeholder = st.empty()

    for attempt in range(max_retries):
        try:
            thinking_text = ""
            response_text = ""

            if attempt > 0:
                progress_placeholder.info(f"Retry attempt {attempt + 1} of {max_retries}...")

            with client.messages.stream(
                model=MODEL_PRIMARY,
                max_tokens=32000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "adaptive"}
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            response_text += event.delta.text

            progress_placeholder.empty()
            response_text = response_text.strip()

            json_match = re.search(r'\{.*"rubric".*\}', response_text, re.DOTALL)
            if json_match:
                rubric_data = json.loads(json_match.group())
            else:
                rubric_data = json.loads(response_text)

            rubric_data["version"] = next_version_number()
            rubric_data["source"] = "inferred"
            rubric_data["conversation_id"] = st.session_state.get("selected_conversation")

            # Evidence gate + data-quality filter: drops dims marked
            # INSUFFICIENT_EVIDENCE and any dim with empty label + empty
            # description. See `_validate_and_filter_rubric`.
            _validate_and_filter_rubric(rubric_data, context_label="infer")

            # Normalize priorities to unique sequential 1..N
            _infer_criteria = rubric_data.get("rubric", [])
            if _infer_criteria:
                _indexed = [(i, c) for i, c in enumerate(_infer_criteria)]
                _indexed.sort(key=lambda x: (x[1].get("priority", 999), x[0]))
                for _rank, (_, _c) in enumerate(_indexed, 1):
                    _c["priority"] = _rank

            # Save to rubric history and activate
            rubric_history = load_rubric_history()
            rubric_history.append(rubric_data)
            save_rubric_history(rubric_history)
            st.session_state.active_rubric_idx = len(rubric_history) - 1
            invalidate_rubric_cache()

            rubric_list = rubric_data.get("rubric", [])
            st.session_state.rubric = rubric_list
            st.session_state.editing_criteria = copy.deepcopy(rubric_list)

            return rubric_data

        except Exception as e:
            error_str = str(e)
            if 'overloaded' in error_str.lower():
                if attempt < max_retries - 1:
                    countdown_placeholder = st.empty()
                    for remaining in range(retry_delay, 0, -1):
                        countdown_placeholder.warning(
                            f"Claude's servers are currently experiencing high demand. "
                            f"Retrying in {remaining} seconds... (Attempt {attempt + 1} of {max_retries})"
                        )
                        time.sleep(1)
                    countdown_placeholder.empty()
                    continue
                else:
                    st.error("Claude's servers are currently overloaded. Please try again in a few minutes.")
                    return None
            else:
                st.error(f"Error inferring rubric: {error_str}")
                return None

    return None


def extract_decision_points(messages, rubric_json, classification_feedback):
    """Extract decision points from conversation with classification context.

    Step 3 of the 5-step flow. Called after classification is confirmed.
    Args:
        messages: Conversation messages (same as infer_dp_messages)
        rubric_json: JSON string of the current rubric criteria
        classification_feedback: dict with 'classifications' (criterion_name -> stated/real/hallucinated)
    Returns:
        dict with 'parsed_data' containing decision_points, or None on failure.
    """
    conversation_text = _build_conversation_text(messages)
    classification_feedback_json = json.dumps(classification_feedback, ensure_ascii=False, indent=2)

    system_prompt = RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT
    user_prompt = RUBRIC_extract_dps_user_prompt(conversation_text, rubric_json, classification_feedback_json)

    max_retries = 3
    retry_delay = 5
    progress_placeholder = st.empty()

    for attempt in range(max_retries):
        try:
            thinking_text = ""
            response_text = ""

            if attempt > 0:
                progress_placeholder.info(f"Retry attempt {attempt + 1} of {max_retries}...")

            with client.messages.stream(
                model=MODEL_PRIMARY,
                max_tokens=16000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "adaptive"}
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            response_text += event.delta.text

            progress_placeholder.empty()
            response_text = response_text.strip()

            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)

            # Ensure parsed_data wrapper
            if "decision_points" in parsed:
                return {"parsed_data": parsed}
            elif "parsed_data" in parsed:
                return parsed
            else:
                return {"parsed_data": {"decision_points": []}}

        except Exception as e:
            error_str = str(e)
            if ('overloaded' in error_str.lower() or 'rate' in error_str.lower()) and attempt < max_retries - 1:
                countdown_placeholder = st.empty()
                for remaining in range(retry_delay, 0, -1):
                    countdown_placeholder.warning(f"Servers busy. Retrying in {remaining}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                countdown_placeholder.empty()
                continue
            else:
                st.error(f"Error extracting decision points: {error_str}")
                return None

    return None


def infer_final_rubric(messages, rubric_json, classification_feedback_json, corrected_dps_json, coldstart_text=""):
    """Infer the final rubric incorporating ALL accumulated feedback.

    Step 5 of the 5-step flow. Called after DP confirmation when hallucinated
    criteria existed or DP corrections were made.
    Args:
        messages: Conversation messages (same as infer_dp_messages)
        rubric_json: JSON string of the current rubric criteria
        classification_feedback_json: JSON string of classification feedback
        corrected_dps_json: JSON string of corrected DPs
        coldstart_text: User's cold-start preference description (optional)
    Returns:
        dict with rubric data + _change_explanation + _refinement_summary, or None.
    """
    conversation_text = _build_conversation_text(messages)

    system_prompt = RUBRIC_FINAL_INFER_SYSTEM_PROMPT
    user_prompt = RUBRIC_final_infer_user_prompt(
        conversation_text, rubric_json, classification_feedback_json,
        corrected_dps_json, coldstart_text
    )

    max_retries = 3
    retry_delay = 10
    progress_placeholder = st.empty()

    for attempt in range(max_retries):
        try:
            thinking_text = ""
            response_text = ""

            if attempt > 0:
                progress_placeholder.info(f"Retry attempt {attempt + 1} of {max_retries}...")

            with client.messages.stream(
                model=MODEL_PRIMARY,
                max_tokens=32000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "adaptive"}
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            response_text += event.delta.text

            progress_placeholder.empty()
            response_text = response_text.strip()

            json_match = re.search(r'\{.*"rubric".*\}', response_text, re.DOTALL)
            if json_match:
                rubric_data = json.loads(json_match.group())
            else:
                rubric_data = json.loads(response_text)

            # Extract explanations before cleaning
            change_explanation = rubric_data.pop("change_explanation", "")
            refinement_summary = rubric_data.get("refinement_summary", "")

            rubric_data["version"] = next_version_number()
            rubric_data["source"] = "inferred_final"
            rubric_data["conversation_id"] = st.session_state.get("selected_conversation")

            # Evidence gate + data-quality filter: same as infer_rubric_only.
            _validate_and_filter_rubric(rubric_data, context_label="infer_final")

            # Normalize priorities to unique sequential 1..N
            _final_criteria = rubric_data.get("rubric", [])
            if _final_criteria:
                _final_indexed = [(i, c) for i, c in enumerate(_final_criteria)]
                _final_indexed.sort(key=lambda x: (x[1].get("priority", 999), x[0]))
                for _final_rank, (_, _fc) in enumerate(_final_indexed, 1):
                    _fc["priority"] = _final_rank

            # Attach explanation so caller can display it
            rubric_data["_change_explanation"] = change_explanation
            rubric_data["_refinement_summary"] = refinement_summary

            # Save to rubric history and activate
            rubric_history = load_rubric_history()
            rubric_history.append(rubric_data)
            save_rubric_history(rubric_history)
            st.session_state.active_rubric_idx = len(rubric_history) - 1
            invalidate_rubric_cache()

            rubric_list = rubric_data.get("rubric", [])
            st.session_state.rubric = rubric_list
            st.session_state.editing_criteria = copy.deepcopy(rubric_list)

            return rubric_data

        except Exception as e:
            error_str = str(e)
            if ('overloaded' in error_str.lower() or 'rate' in error_str.lower()) and attempt < max_retries - 1:
                countdown_placeholder = st.empty()
                for remaining in range(retry_delay, 0, -1):
                    countdown_placeholder.warning(f"Servers busy. Retrying in {remaining}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                countdown_placeholder.empty()
                continue
            else:
                st.error(f"Error inferring final rubric: {error_str}")
                return None

    return None
