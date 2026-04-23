"""Research metrics logging for RQ1 (Preference Elicitation) and RQ2 (Rubric Refinement).

All data is stored in Supabase via save_project_data under metric-specific data_types.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any

import streamlit as st

from auth_supabase import save_project_data

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_id() -> str:
    return st.session_state.get("selected_conversation") or ""


def _save_metric(data_type: str, payload: dict[str, Any]) -> None:
    sb = st.session_state.get("supabase")
    pid = st.session_state.get("current_project_id")
    if sb and pid:
        try:
            save_project_data(sb, pid, data_type, payload)
        except Exception as e:
            _log.warning("Failed to save metric %s: %s", data_type, e)


# ---------------------------------------------------------------------------
# Metric 3: Confirmation Rate (RQ2)
# ---------------------------------------------------------------------------

def log_confirmation(
    source: str,
    draft_index: int,
    drift_type: str,
    dimension_id: str,
    dimension_text: str,
    grader_verdict: str,
    grader_confidence: str,
    user_response_raw: str,
    user_confirms_grader: bool,
) -> None:
    _save_metric("rq2_confirmation", {
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "phase": "rq2_confirmation",
        "source": source,
        "draft_index": draft_index,
        "drift_type": drift_type,
        "dimension_id": dimension_id,
        "dimension_text": dimension_text,
        "grader_verdict": grader_verdict,
        "grader_confidence": grader_confidence,
        "user_response_raw": user_response_raw,
        "user_confirms_grader": user_confirms_grader,
    })


# ---------------------------------------------------------------------------
# Metric 4: Drift Panel Fire Rate (RQ2)
# ---------------------------------------------------------------------------

def log_fire_rate(
    draft_index: int,
    total_dimensions: int,
    panels_fired: int,
    panels_by_type: dict[str, int],
    panels_suppressed: int,
    suppression_reasons: list[str],
) -> None:
    _save_metric("rq2_fire_rate", {
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "phase": "rq2_fire_rate",
        "draft_index": draft_index,
        "total_dimensions": total_dimensions,
        "panels_fired": panels_fired,
        "panels_by_type": panels_by_type,
        "panels_suppressed": panels_suppressed,
        "suppression_reasons": suppression_reasons,
    })


# ---------------------------------------------------------------------------
# Metric 5: Fresh-Generation Pairwise Preference (RQ2)
# ---------------------------------------------------------------------------

def log_pairwise_preference(
    early_rubric_version: int,
    late_rubric_version: int,
    early_rubric_label: str,
    late_rubric_label: str,
    user_preferred: str,
    user_preferred_rubric: str,
    user_reason: str = "",
) -> None:
    _save_metric("rq2_pairwise", {
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "phase": "rq2_pairwise",
        "early_rubric_version": early_rubric_version,
        "late_rubric_version": late_rubric_version,
        "early_rubric_label": early_rubric_label,
        "late_rubric_label": late_rubric_label,
        "user_preferred": user_preferred,
        "user_preferred_rubric": user_preferred_rubric,
        "user_reason": user_reason,
    })


def log_threeway_preference(
    *,
    task: str,
    early_rubric_version: int | None,
    late_rubric_version: int | None,
    label_to_arm: dict[str, str],   # e.g. {"A": "none", "B": "early", "C": "late"}
    best_label: str | None,          # "A" | "B" | "C" | None (if tie)
    worst_label: str | None,         # "A" | "B" | "C" | None (if tie)
    best_arm: str | None,            # "none" | "early" | "late" | None
    worst_arm: str | None,
    all_same: bool,
    user_reason: str = "",
) -> None:
    """Three-way blind preference: no-rubric vs first-inferred vs current-refined.

    `label_to_arm` records which draft label got which arm so we can decode
    position effects at analysis time. `best_label`/`worst_label` let us
    compute both (a) whether having any rubric helps (none vs others) and
    (b) whether refinement adds value (early vs late)."""
    _save_metric("rq2_threeway", {
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "phase": "rq2_threeway",
        "task": task,
        "early_rubric_version": early_rubric_version,
        "late_rubric_version": late_rubric_version,
        "label_to_arm": label_to_arm,
        "best_label": best_label,
        "worst_label": worst_label,
        "best_arm": best_arm,
        "worst_arm": worst_arm,
        "all_same": all_same,
        "user_reason": user_reason,
    })


# ---------------------------------------------------------------------------
# Spot-check injection for Metric 3
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Edit telemetry (for proposed rubric edit UI)
# ---------------------------------------------------------------------------

import time as _time_module

_EDIT_SHOWN_TS: dict[str, float] = {}


def log_edit_shown(
    edit_id: str,
    criterion_id: str,
    dimension_id: str,
    scope_change: str,
) -> None:
    """Fire when a proposed edit is first shown to the user."""
    _EDIT_SHOWN_TS[edit_id] = _time_module.monotonic()
    _save_metric("edit_telemetry", {
        "event": "edit_shown",
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "edit_id": edit_id,
        "criterion_id": criterion_id,
        "dimension_id": dimension_id,
        "scope_change": scope_change,
    })


def log_example_expanded(edit_id: str) -> None:
    """Fire when the user opens the expandable 'Example from your draft' section."""
    _save_metric("edit_telemetry", {
        "event": "example_expanded",
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "edit_id": edit_id,
    })


def log_edit_decision(
    edit_id: str,
    decision: str,  # "applied" | "modified" | "dismissed"
    scope_change: str = "",
    criterion_id: str = "",
    dimension_id: str = "",
) -> None:
    """Fire when the user applies, modifies, or dismisses a proposed edit.
    time_since_shown is in milliseconds from edit_shown to this event."""
    shown_ts = _EDIT_SHOWN_TS.get(edit_id)
    time_since_shown_ms: int | None = None
    if shown_ts is not None:
        time_since_shown_ms = int((_time_module.monotonic() - shown_ts) * 1000)
    _save_metric("edit_telemetry", {
        "event": "edit_decision",
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "edit_id": edit_id,
        "decision": decision,
        "scope_change": scope_change,
        "criterion_id": criterion_id,
        "dimension_id": dimension_id,
        "time_since_shown_ms": time_since_shown_ms,
    })


def should_inject_spot_check(draft_index: int, checkpoints: tuple[int, ...] = (4, 8)) -> bool:
    """Return True if this draft index is a spot-check checkpoint."""
    return draft_index in checkpoints


def sample_spot_check_dims_for_metric(
    current_grades: dict[str, Any],
    dims_surfaced_since_last_checkpoint: set[str],
    sample_size: int = 2,
) -> list[dict[str, Any]]:
    """Sample dimensions for metric spot-check. These get injected as
    low_confidence panels so the user can't distinguish them from organic ones."""
    candidates: list[dict[str, Any]] = []
    for c in current_grades.get("grades") or []:
        cname = (c.get("criterion_name") or "").strip()
        for d in c.get("dimension_grades") or []:
            did = (d.get("dimension_id") or "").strip()
            if not did or did in dims_surfaced_since_last_checkpoint:
                continue
            # Include high and medium confidence (skip low -- those already fire)
            if d.get("confidence") in ("high", "medium"):
                candidates.append({
                    "criterion": cname,
                    "dimension_id": did,
                    "grade": (d.get("grade") or "").upper(),
                    "confidence": d.get("confidence", "high"),
                    "evidence": d.get("evidence", ""),
                })
    if len(candidates) < 2:
        return candidates
    return random.sample(candidates, min(sample_size, len(candidates)))


# ---------------------------------------------------------------------------
# Metric 1: Precision (RQ1)
# ---------------------------------------------------------------------------

def log_dimension_recognition(
    dimension_id: str,
    dimension_text: str,
    criterion: str,
    category: str,
    presentation_order: int,
) -> None:
    """Log the user's categorization of an inferred dimension.
    category is one of: 'preference', 'not_my_preference'
    # Future: add 'implicit' category for implicitness rate
    """
    _save_metric("rq1_dimension_recognition", {
        "session_id": _get_session_id(),
        "timestamp": datetime.now().isoformat(),
        "phase": "rq1_dimension_recognition",
        "dimension_id": dimension_id,
        "dimension_text": dimension_text,
        "criterion": criterion,
        "category": category,
        "presentation_order": presentation_order,
    })


def render_dimension_recognition(rubric_dict: dict[str, Any]) -> None:
    """After rubric inference, present each dimension for precision validation.

    Precision = count(preference) / count(all dimensions)
    """
    rubric = rubric_dict.get("rubric") or []
    if not rubric:
        return

    all_dims: list[dict[str, Any]] = []
    for crit in rubric:
        cname = crit.get("name", "")
        for dim in crit.get("dimensions") or []:
            all_dims.append({
                "criterion": cname,
                "id": dim.get("id", ""),
                "label": dim.get("label") or dim.get("description") or dim.get("id", ""),
                "evidence": (dim.get("evidence") or "").strip(),
            })

    if not all_dims:
        return

    if "dim_recognition_results" not in st.session_state:
        st.session_state.dim_recognition_results = {}
    if "dim_recognition_done" not in st.session_state:
        st.session_state.dim_recognition_done = False

    if st.session_state.dim_recognition_done:
        # Already persisted as a system message -- nothing to render here
        return

    st.caption(
        "We inferred these dimensions from your conversation. "
        "**Click a dimension** to see the evidence we used to infer it. "
        "Click ❌ on any that are not your preference."
    )

    # Group dimensions by criterion
    from collections import OrderedDict
    grouped: OrderedDict[str, list[tuple[int, dict[str, Any]]]] = OrderedDict()
    for i, dim in enumerate(all_dims):
        grouped.setdefault(dim["criterion"], []).append((i, dim))

    for crit_name, dims_in_crit in grouped.items():
        st.markdown(f"**_{crit_name}_**")
        for i, dim in dims_in_crit:
            dim_key = f"{dim['criterion']}::{dim['id']}"
            rejected = st.session_state.dim_recognition_results.get(dim_key) == "not_my_preference"

            col_icon, col_text, col_btn = st.columns([0.5, 7, 1.5])
            with col_icon:
                st.markdown("❌" if rejected else "✅")
            with col_text:
                _ev = dim.get("evidence", "")
                # The dim label itself is the expander title -- click the dim
                # to reveal the inferrer's evidence. For rejected dims we
                # prefix the title so it's visually distinct and keep the
                # evidence available (user can still double-check why we
                # inferred the dim even after rejecting it).
                _title = (f"(rejected) {dim['label']}"
                          if rejected else dim["label"])
                if _ev:
                    with st.expander(_title, expanded=False):
                        st.caption(_ev)
                else:
                    # Legacy rubric with no evidence -- render the label as
                    # plain text so the row layout stays consistent. New
                    # rubrics always have evidence (enforced by the prompt
                    # + parser-level INSUFFICIENT_EVIDENCE reject).
                    if rejected:
                        st.caption(f"~~{dim['label']}~~")
                    else:
                        st.markdown(dim["label"])
            with col_btn:
                if rejected:
                    if st.button("🔁", key=f"recog_undo_{i}"):
                        del st.session_state.dim_recognition_results[dim_key]
                        st.rerun()
                else:
                    if st.button("❌", key=f"recog_reject_{i}"):
                        st.session_state.dim_recognition_results[dim_key] = "not_my_preference"
                        log_dimension_recognition(dim["id"], dim["label"], dim["criterion"], "not_my_preference", i)
                        st.rerun()

    if st.button("Confirm", type="primary", key="recog_confirm"):
        # Log all non-rejected dimensions as "preference"
        for dim in all_dims:
            dim_key = f"{dim['criterion']}::{dim['id']}"
            if dim_key not in st.session_state.dim_recognition_results:
                st.session_state.dim_recognition_results[dim_key] = "preference"
                log_dimension_recognition(dim["id"], dim["label"], dim["criterion"], "preference", all_dims.index(dim))

        results = st.session_state.dim_recognition_results
        rejected_ids = {
            dim["id"]
            for dim in all_dims
            if results.get(f"{dim['criterion']}::{dim['id']}") == "not_my_preference"
        }

        # Remove rejected dimensions from rubric and save as new version.
        saved_version: int | None = None
        current_version: int | None = None
        if rejected_ids:
            from rubric_writer.persistence import (
                get_active_rubric, load_rubric_history, save_rubric_history,
                invalidate_rubric_cache,
            )
            import copy

            rubric_dict_current, _, rubric_history = get_active_rubric()
            if rubric_dict_current and rubric_dict_current.get("rubric"):
                new_rubric = copy.deepcopy(rubric_dict_current)
                # Strip metadata that the DB assigns on save.
                new_rubric.pop("id", None)
                new_rubric.pop("version", None)
                new_rubric.pop("created_at", None)
                for crit in new_rubric.get("rubric") or []:
                    crit["dimensions"] = [
                        d for d in (crit.get("dimensions") or [])
                        if (d.get("id") or "").strip() not in rejected_ids
                    ]
                new_rubric["rubric"] = [
                    c for c in new_rubric["rubric"]
                    if c.get("dimensions")
                ]
                new_rubric["source"] = "user_validated"
                rubric_history.append(new_rubric)
                saved_version = save_rubric_history(rubric_history)
                invalidate_rubric_cache()

                # Refresh the shadow state the rubric configuration UI reads
                # from. Without this, the config tab keeps showing the OLD
                # version's dimensions until the page is reloaded.
                reloaded = load_rubric_history(force_reload=True)
                if reloaded:
                    new_active = reloaded[-1]
                    new_criteria = new_active.get("rubric", [])
                    st.session_state.rubric = new_criteria
                    try:
                        st.session_state.editing_criteria = copy.deepcopy(new_criteria)
                        st.session_state["editing_criteria_ui_version"] = (
                            st.session_state.get("editing_criteria_ui_version", 0) + 1
                        )
                    except Exception:
                        pass
                    st.session_state.active_rubric_idx = len(reloaded) - 1
                    if saved_version is None:
                        # Fall back to whatever the DB shows as the latest
                        # version number if save_rubric_history didn't return one.
                        saved_version = new_active.get("version")
                    # CLEAR the version-selector widget key. The selectbox will
                    # re-initialize on the next render using `index=active_idx`,
                    # which we just set to the new version's index. Setting the
                    # widget key directly is fragile -- Streamlit can raise
                    # StreamlitAPIException if the widget has already been
                    # instantiated in this session. Popping the key and letting
                    # the widget re-hydrate from `index` is more reliable.
                    try:
                        from rubric_writer.widget_keys import project_scoped_key
                        _rvk = project_scoped_key("rubric_version_selector")
                        st.session_state.pop(_rvk, None)
                    except Exception:
                        pass
        else:
            # No rejections — report the current rubric's version.
            try:
                from rubric_writer.persistence import get_active_rubric
                current_dict, _, _ = get_active_rubric()
                if current_dict:
                    current_version = current_dict.get("version")
            except Exception:
                current_version = None

        # Build compact system message
        if rejected_ids and saved_version is not None:
            content = (
                f"**Confirmed dimensions.** "
                f"Removed {len(rejected_ids)} dimension(s) you marked as not your "
                f"preference — new rubric **v{saved_version}** saved."
            )
        elif rejected_ids:
            content = (
                f"**Confirmed dimensions.** "
                f"Removed {len(rejected_ids)} dimension(s), but the new version "
                "couldn't be saved. Please try again."
            )
        else:
            ver_str = f"v{current_version}" if current_version is not None else "current version"
            content = (
                f"**Confirmed dimensions.** Everything was validated — "
                f"keeping the {ver_str} rubric as is."
            )

        import time as _time
        st.session_state.messages.append({
            "role": "system",
            "content": content,
            "message_id": f"dim_recog_{int(_time.time() * 1000000)}",
            "is_system_generated": True,
        })

        # Persist immediately so the system message survives a reload. Without
        # this, the message lives only in-memory until something else triggers
        # an auto-save, and a page refresh before that point loses it.
        try:
            from rubric_writer.persistence import _auto_save_conversation
            _auto_save_conversation()
        except Exception as e:
            _log.warning("auto-save after dim-recognition system message failed: %s", e)

        st.session_state.dim_recognition_done = True
        st.rerun()

    # --- Implicitness follow-up (commented out for now) ---
    # Future: after precision validation, for each confirmed dimension ask:
    # "Would you have thought to mention this before seeing any drafts?"
    # "Yes, I would have mentioned this" -> stated
    # "No, I wouldn't have thought of this" -> implicit
    # Implicitness rate = implicit / (stated + implicit)


# ---------------------------------------------------------------------------
# Pairwise Preference UI renderer
# ---------------------------------------------------------------------------

def render_pairwise_comparison() -> None:
    """Fresh-generation A/B comparison between early and late rubric versions.
    Shown at the start of a new conversation when a new rubric exists."""
    from rubric_writer.persistence import load_rubric_history
    from rubric_writer.config import MODEL_PRIMARY
    from rubric_writer.api_client import _api_call_with_retry

    hist = load_rubric_history()
    if len(hist) < 2:
        return

    # Use version 2 as "early" (first real inferred rubric) and latest as "late"
    early_idx = min(1, len(hist) - 1)
    late_idx = len(hist) - 1
    if early_idx == late_idx:
        return

    early_rubric = hist[early_idx]
    late_rubric = hist[late_idx]
    early_version = early_rubric.get("version", early_idx + 1)
    late_version = late_rubric.get("version", late_idx + 1)

    # Initialize state
    if "pairwise_done" not in st.session_state:
        st.session_state.pairwise_done = False
    if st.session_state.pairwise_done:
        return

    if "pairwise_task" not in st.session_state:
        st.session_state.pairwise_task = ""
    if "pairwise_drafts" not in st.session_state:
        st.session_state.pairwise_drafts = None
    if "pairwise_labels" not in st.session_state:
        # Randomize which rubric is A vs B
        if random.random() < 0.5:
            st.session_state.pairwise_labels = {"A": "early", "B": "late"}
        else:
            st.session_state.pairwise_labels = {"A": "late", "B": "early"}

    st.subheader("Rubric Comparison")
    st.caption(
        "Your rubric has been refined. Let's see if it makes a difference. "
        "Enter a writing task and we'll generate two drafts for you to compare."
    )

    task = st.text_area(
        "Writing task",
        value=st.session_state.pairwise_task,
        placeholder="e.g. Write a project update email to my team",
        key="pairwise_task_input",
    )

    if st.button("Generate drafts", key="pairwise_generate") and task.strip():
        st.session_state.pairwise_task = task.strip()
        labels = st.session_state.pairwise_labels

        with st.spinner("Generating Draft A..."):
            rubric_a = early_rubric if labels["A"] == "early" else late_rubric
            rubric_a_json = json.dumps(rubric_a.get("rubric", []), ensure_ascii=False, indent=2)
            resp_a = _api_call_with_retry(
                model=MODEL_PRIMARY, max_tokens=4000,
                messages=[{"role": "user", "content": (
                    f"Write the following based on this rubric.\n\n"
                    f"RUBRIC:\n{rubric_a_json}\n\n"
                    f"TASK: {task.strip()}\n\n"
                    f"Write only the draft, nothing else."
                )}],
            )
            draft_a = "".join(b.text for b in resp_a.content if b.type == "text")

        with st.spinner("Generating Draft B..."):
            rubric_b = late_rubric if labels["A"] == "early" else early_rubric
            rubric_b_json = json.dumps(rubric_b.get("rubric", []), ensure_ascii=False, indent=2)
            resp_b = _api_call_with_retry(
                model=MODEL_PRIMARY, max_tokens=4000,
                messages=[{"role": "user", "content": (
                    f"Write the following based on this rubric.\n\n"
                    f"RUBRIC:\n{rubric_b_json}\n\n"
                    f"TASK: {task.strip()}\n\n"
                    f"Write only the draft, nothing else."
                )}],
            )
            draft_b = "".join(b.text for b in resp_b.content if b.type == "text")

        st.session_state.pairwise_drafts = {"A": draft_a, "B": draft_b}
        st.rerun()

    # Show drafts if generated
    drafts = st.session_state.pairwise_drafts
    if drafts:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Draft A**")
            st.markdown(drafts["A"])
        with col_b:
            st.markdown("**Draft B**")
            st.markdown(drafts["B"])

        st.markdown("**Which draft better matches what you wanted?**")
        col1, col2 = st.columns(2)
        labels = st.session_state.pairwise_labels

        with col1:
            if st.button("Draft A", key="pairwise_pick_a"):
                preferred_rubric = labels["A"]
                early_label = "A" if labels["A"] == "early" else "B"
                late_label = "B" if labels["A"] == "early" else "A"
                log_pairwise_preference(
                    early_rubric_version=early_version,
                    late_rubric_version=late_version,
                    early_rubric_label=early_label,
                    late_rubric_label=late_label,
                    user_preferred="A",
                    user_preferred_rubric=preferred_rubric,
                )
                st.session_state.pairwise_done = True
                st.success("Thanks! Your preference has been recorded.")
                st.rerun()
        with col2:
            if st.button("Draft B", key="pairwise_pick_b"):
                preferred_rubric = labels["B"]
                early_label = "A" if labels["A"] == "early" else "B"
                late_label = "B" if labels["A"] == "early" else "A"
                log_pairwise_preference(
                    early_rubric_version=early_version,
                    late_rubric_version=late_version,
                    early_rubric_label=early_label,
                    late_rubric_label=late_label,
                    user_preferred="B",
                    user_preferred_rubric=preferred_rubric,
                )
                st.session_state.pairwise_done = True
                st.success("Thanks! Your preference has been recorded.")
                st.rerun()

        reason = st.text_input(
            "Why? (optional)",
            key="pairwise_reason",
            placeholder="What made you prefer that draft?",
        )
