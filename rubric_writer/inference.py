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
from rubric_writer.widget_keys import project_scoped_key


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

def infer_rubric_pure(
    messages,
    previous_rubric,
    *,
    conversation_id=None,
    version_number=1,
    model=None,
    anthropic_client=None,
    max_retries=3,
    on_progress=None,
):
    """Pure inference: given messages and a previous rubric (may be empty),
    call the LLM, parse, filter, normalize priorities, and return the
    post-filter rubric_data.

    This function has zero dependencies on Streamlit, session_state, or
    Supabase. The synthetic-study pipeline calls it directly; the live
    `infer_rubric_only` wraps it with session-state plumbing.

    Args:
        messages: conversation messages (same shape as st.session_state.messages).
        previous_rubric: list of criteria dicts, or [] if no prior rubric.
        conversation_id: optional, stamped onto the returned rubric_data.
        version_number: integer to stamp as `version` on the rubric_data.
            The live caller supplies this from `next_version_number()`;
            the synthetic caller supplies it from its JSONL store.
        model: model id, defaults to MODEL_PRIMARY.
        anthropic_client: optional Anthropic client; defaults to live config client.
        max_retries: number of overload retries before raising.
        on_progress: optional callable(stage:str) for progress reporting.
            None means silent (synthetic pipeline). Live caller passes a
            Streamlit-aware callback.

    Returns:
        dict with keys: version, source, conversation_id, rubric (list),
        and any extra keys the LLM emitted (writing_type, user_goals_summary, ...).

    Raises:
        anthropic.APIStatusError on persistent overload after retries.
        ValueError on unparseable LLM output.
    """
    from rubric_writer.config import MODEL_PRIMARY as _MODEL_PRIMARY
    from rubric_writer.config import client as _default_client

    _client = anthropic_client or _default_client
    _model = model or _MODEL_PRIMARY

    conversation_text = _build_conversation_text(messages)
    if previous_rubric:
        previous_rubric_json = json.dumps(previous_rubric, ensure_ascii=False, indent=2)
    else:
        previous_rubric_json = ""

    user_prompt = RUBRIC_infer_only_user_prompt(conversation_text, previous_rubric_json)

    last_err = None
    for attempt in range(max_retries):
        if on_progress and attempt > 0:
            on_progress(f"retry_{attempt + 1}_of_{max_retries}")
        try:
            response_text = ""
            with _client.messages.stream(
                model=_model,
                max_tokens=32000,
                system=RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "adaptive"},
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            response_text += event.delta.text
                        # thinking deltas intentionally discarded — not used downstream

            response_text = response_text.strip()
            json_match = re.search(r'\{.*"rubric".*\}', response_text, re.DOTALL)
            if json_match:
                rubric_data = json.loads(json_match.group())
            else:
                rubric_data = json.loads(response_text)

            rubric_data["version"] = version_number
            rubric_data["source"] = "inferred"
            rubric_data["conversation_id"] = conversation_id

            _validate_and_filter_rubric(rubric_data, context_label="infer")

            # Normalize priorities to unique sequential 1..N (preserves
            # original order on tied priorities).
            _crits = rubric_data.get("rubric", [])
            if _crits:
                _indexed = [(i, c) for i, c in enumerate(_crits)]
                _indexed.sort(key=lambda x: (x[1].get("priority", 999), x[0]))
                for _rank, (_, _c) in enumerate(_indexed, 1):
                    _c["priority"] = _rank

            return rubric_data

        except Exception as e:
            last_err = e
            error_str = str(e).lower()
            if "overloaded" in error_str and attempt < max_retries - 1:
                if on_progress:
                    on_progress(f"overloaded_attempt_{attempt + 1}")
                time.sleep(5)
                continue
            raise

    if last_err:
        raise last_err
    raise RuntimeError("infer_rubric_pure: exhausted retries without result")


def infer_rubric_only(messages):
    """Infer a rubric from conversation WITHOUT extracting decision points.

    Step 1 of the 5-step flow. DPs are extracted separately in step 3.
    Returns:
        dict with rubric data (no 'inference_decision_points'), or None on failure.

    Live wrapper: handles session_state, Supabase persistence, and Streamlit
    progress UI. The pure inference logic lives in `infer_rubric_pure`.
    """
    # Get the current active rubric to build upon
    active_rubric_dict, _, _ = get_active_rubric()
    previous_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
    previous_rubric_version = active_rubric_dict.get("version", 1) if active_rubric_dict else 1

    if previous_rubric:
        st.info(f"Using previous rubric v{previous_rubric_version}")
    else:
        st.info("No previous rubric found - creating new rubric from scratch")

    progress_placeholder = st.empty()

    def _on_progress(stage: str) -> None:
        if stage.startswith("retry_"):
            progress_placeholder.info(stage.replace("_", " ").title())
        elif stage.startswith("overloaded_"):
            countdown_placeholder = st.empty()
            for remaining in range(5, 0, -1):
                countdown_placeholder.warning(
                    f"Claude's servers are currently experiencing high demand. "
                    f"Retrying in {remaining} seconds..."
                )
                time.sleep(1)
            countdown_placeholder.empty()

    try:
        rubric_data = infer_rubric_pure(
            messages,
            previous_rubric,
            conversation_id=st.session_state.get("selected_conversation"),
            version_number=next_version_number(),
            on_progress=_on_progress,
        )

        # Persistence + session-state activation must stay inside this try
        # block. Supabase writes can fail (network, auth, schema drift), and
        # if the exception escapes the spinner context Streamlit silently
        # reruns the page with no rubric saved — which the user sees as
        # "first click does nothing, second click works."
        rubric_history = load_rubric_history()
        rubric_history.append(rubric_data)
        save_rubric_history(rubric_history)
        st.session_state.active_rubric_idx = len(rubric_history) - 1
        invalidate_rubric_cache()

        rubric_list = rubric_data.get("rubric", [])
        st.session_state.rubric = rubric_list
        st.session_state.editing_criteria = copy.deepcopy(rubric_list)
        st.session_state.pop(project_scoped_key("rubric_version_selector"), None)

        return rubric_data
    except Exception as e:
        error_str = str(e)
        if "overloaded" in error_str.lower():
            st.error("Claude's servers are currently overloaded. Please try again in a few minutes.")
        else:
            st.error(f"Error inferring rubric: {error_str}")
        return None
    finally:
        progress_placeholder.empty()
