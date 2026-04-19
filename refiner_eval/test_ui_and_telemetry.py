"""UI component tests for the proposed-edit panel and telemetry tests.

Run: python refiner_eval/test_ui_and_telemetry.py

We use mocks rather than a full Streamlit runtime. These test the functions'
contracts: what gets rendered (st.markdown/caption/expander calls), and what
telemetry fires with what payload.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# UI render tests
# ---------------------------------------------------------------------------

def _collect_ui_calls():
    """Patch streamlit calls and return a log of everything rendered."""
    calls: list[tuple[str, str]] = []

    class _FakeExpander:
        def __init__(self, title, **kw):
            calls.append(("expander", title))
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class _FakeCol:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    fake_st = MagicMock()
    fake_st.markdown = lambda text, **kw: calls.append(("markdown", str(text)))
    fake_st.caption = lambda text, **kw: calls.append(("caption", str(text)))
    fake_st.expander = _FakeExpander
    fake_st.warning = lambda text, **kw: calls.append(("warning", str(text)))
    fake_st.success = lambda text, **kw: calls.append(("success", str(text)))
    fake_st.button = lambda *a, **kw: False
    fake_st.columns = lambda spec: [_FakeCol() for _ in range(spec if isinstance(spec, int) else len(spec))]
    fake_st.divider = lambda: None
    fake_st.text_area = lambda *a, **kw: None
    fake_st.rerun = lambda: None
    fake_st.session_state = {}

    return fake_st, calls


def _render_suggestion(suggestion):
    """Helper to render a single suggestion with mocked streamlit."""
    fake_st, calls = _collect_ui_calls()
    with patch("rubric_writer.draft_grading_ui.st", fake_st):
        with patch("rubric_writer.draft_grading_ui._lookup_dimension_description",
                   return_value="the dimension description"):
            from rubric_writer.draft_grading_ui import _render_single_edit_suggestion
            _render_single_edit_suggestion(suggestion, 0)
    return calls


def test_renders_all_four_scope_values():
    base = {
        "criterion_name": "Tone",
        "dimension_id": "warmth",
        "before_wording": "old wording",
        "after_wording": "new wording",
        "reasoning": "You said this. The edit does that.",
        "example_annotation": "The passage applies like so.",
        "grader_evidence": "some evidence",
        "edit_id": "test-id",
    }
    for scope in ["expands", "narrows", "clarifies", "reframes"]:
        s = {**base, "scope_change": scope}
        calls = _render_suggestion(s)
        rendered_text = " ".join(c[1] for c in calls)
        # Badge appears somewhere via an HTML span
        assert any(f"{scope}" in c[1].lower() or "Expands" in c[1] or "Narrows" in c[1]
                   or "Clarifies" in c[1] or "Reframes" in c[1] for c in calls), (
            f"expected badge for {scope} in: {rendered_text}"
        )
        # Reasoning present
        assert any("Why this change" in c[1] for c in calls), f"missing reasoning label for {scope}"
        # Expandable example section present
        assert any(c[0] == "expander" and "Example from your draft" in c[1] for c in calls), (
            f"missing example expander for {scope}"
        )
        print(f"  scope={scope}: PASS")


def test_renders_diff_only_fallback_when_reasoning_missing():
    """If reasoning is empty (parse_status=missing_fields), UI should still render diff."""
    s = {
        "criterion_name": "Tone",
        "dimension_id": "warmth",
        "before_wording": "old wording",
        "after_wording": "new wording",
        "reasoning": "",
        "scope_change": "clarifies",
        "example_annotation": "",
        "grader_evidence": "",
        "parse_status": "missing_fields",
        "edit_id": "test-id",
    }
    calls = _render_suggestion(s)
    # Diff is present
    assert any("Before" in c[1] or "After" in c[1] for c in calls), "diff labels missing"
    # No crash, graceful handling
    print("  diff-only fallback: PASS")


def test_retry_warning_renders():
    s = {
        "criterion_name": "Tone", "dimension_id": "warmth",
        "before_wording": "a", "after_wording": "b",
        "reasoning": "You said so.", "scope_change": "clarifies",
        "example_annotation": "like so.", "grader_evidence": "x",
        "edit_id": "test-id",
        "is_retry": True,
        "previous_attempt": {"grader_grade": "NOT_MET", "user_expected": "MET"},
    }
    calls = _render_suggestion(s)
    assert any(c[0] == "warning" for c in calls), "retry warning should render"
    print("  retry warning: PASS")


# ---------------------------------------------------------------------------
# Telemetry tests
# ---------------------------------------------------------------------------

def test_log_edit_shown_fires_with_correct_payload():
    from rubric_writer import metrics

    captured = []
    with patch.object(metrics, "_save_metric",
                      lambda data_type, payload: captured.append((data_type, payload))):
        metrics.log_edit_shown(
            edit_id="edit-1",
            criterion_id="Tone",
            dimension_id="warmth",
            scope_change="expands",
        )
    assert len(captured) == 1
    data_type, payload = captured[0]
    assert data_type == "edit_telemetry"
    assert payload["event"] == "edit_shown"
    assert payload["edit_id"] == "edit-1"
    assert payload["criterion_id"] == "Tone"
    assert payload["dimension_id"] == "warmth"
    assert payload["scope_change"] == "expands"
    assert "timestamp" in payload
    print("  log_edit_shown: PASS")


def test_log_example_expanded_fires():
    from rubric_writer import metrics

    captured = []
    with patch.object(metrics, "_save_metric",
                      lambda data_type, payload: captured.append((data_type, payload))):
        metrics.log_example_expanded(edit_id="edit-1")
    assert len(captured) == 1
    data_type, payload = captured[0]
    assert data_type == "edit_telemetry"
    assert payload["event"] == "example_expanded"
    assert payload["edit_id"] == "edit-1"
    print("  log_example_expanded: PASS")


def test_log_edit_decision_includes_time_since_shown():
    import time
    from rubric_writer import metrics

    captured = []
    with patch.object(metrics, "_save_metric",
                      lambda data_type, payload: captured.append((data_type, payload))):
        metrics.log_edit_shown(edit_id="edit-2", criterion_id="c", dimension_id="d", scope_change="clarifies")
        time.sleep(0.1)  # 100ms
        metrics.log_edit_decision(
            edit_id="edit-2", decision="applied",
            scope_change="clarifies", criterion_id="c", dimension_id="d",
        )
    assert len(captured) == 2
    decision_payload = captured[1][1]
    assert decision_payload["event"] == "edit_decision"
    assert decision_payload["decision"] == "applied"
    assert decision_payload["edit_id"] == "edit-2"
    assert decision_payload["time_since_shown_ms"] is not None
    assert decision_payload["time_since_shown_ms"] >= 100
    print(f"  log_edit_decision (time_since_shown_ms={decision_payload['time_since_shown_ms']}): PASS")


def test_log_edit_decision_without_prior_shown():
    """If the edit was shown in a previous process and we don't have the timestamp,
    time_since_shown_ms should be None rather than crashing."""
    from rubric_writer import metrics

    captured = []
    with patch.object(metrics, "_save_metric",
                      lambda data_type, payload: captured.append((data_type, payload))):
        metrics.log_edit_decision(
            edit_id="ghost-edit",  # never was log_edit_shown'd
            decision="dismissed",
        )
    assert len(captured) == 1
    payload = captured[0][1]
    assert payload["time_since_shown_ms"] is None
    print("  log_edit_decision without prior shown: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("UI render tests:")
    test_renders_all_four_scope_values()
    test_renders_diff_only_fallback_when_reasoning_missing()
    test_retry_warning_renders()

    print("\nTelemetry tests:")
    test_log_edit_shown_fires_with_correct_payload()
    test_log_example_expanded_fires()
    test_log_edit_decision_includes_time_since_shown()
    test_log_edit_decision_without_prior_shown()

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
