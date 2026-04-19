"""Unit tests for refiner response schema parsing.
Run: python -m pytest refiner_eval/test_schema.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubric_writer.draft_grading_ui import _parse_refiner_response


def test_valid_full_response():
    raw = json.dumps({
        "criterion_name": "Tone",
        "dimension_id": "warmth",
        "before_wording": "Opens with a warm greeting.",
        "after_wording": "Opens with a warm greeting, appropriate to the context.",
        "reasoning": "You indicated the draft feels warm in context. This edit adds the context qualifier.",
        "scope_change": "expands",
        "example_annotation": "The opener 'Hi team' was flagged as too casual; the context qualifier covers it.",
    })
    parsed, status = _parse_refiner_response(raw)
    assert status == "ok"
    assert parsed["scope_change"] == "expands"
    assert parsed["before_wording"].startswith("Opens")


def test_case_insensitive_scope():
    raw = json.dumps({
        "before_wording": "a", "after_wording": "b",
        "reasoning": "you said so", "example_annotation": "x",
        "scope_change": "EXPANDS",
    })
    parsed, status = _parse_refiner_response(raw)
    assert status == "ok"
    assert parsed["scope_change"] == "expands"


def test_invalid_scope_defaults_to_clarifies():
    raw = json.dumps({
        "before_wording": "a", "after_wording": "b",
        "reasoning": "you said so", "example_annotation": "x",
        "scope_change": "makes_better",
    })
    parsed, status = _parse_refiner_response(raw)
    assert status == "invalid_scope"
    assert parsed["scope_change"] == "clarifies"


def test_no_change_needed():
    raw = json.dumps({"no_change_needed": True})
    parsed, status = _parse_refiner_response(raw)
    assert status == "no_change_needed"


def test_malformed_json():
    raw = "this is not json"
    parsed, status = _parse_refiner_response(raw)
    assert status == "malformed_json"
    assert parsed is None


def test_partial_json_embedded_in_prose():
    raw = (
        "Here is my response:\n\n"
        '{"before_wording": "a", "after_wording": "b", '
        '"reasoning": "You said so.", "scope_change": "clarifies", '
        '"example_annotation": "Covers the flagged sentence."}\n\n'
        "Let me know if you have questions."
    )
    parsed, status = _parse_refiner_response(raw)
    assert status == "ok"
    assert parsed["after_wording"] == "b"


def test_missing_reasoning_but_has_diff_returns_diff_only():
    raw = json.dumps({
        "before_wording": "a", "after_wording": "b",
    })
    parsed, status = _parse_refiner_response(raw)
    assert status == "missing_fields"
    assert parsed is not None
    assert parsed["before_wording"] == "a"
    assert parsed["after_wording"] == "b"


def test_legacy_old_text_new_text_compat():
    raw = json.dumps({
        "old_text": "a", "new_text": "b", "change_summary": "made it better",
    })
    parsed, status = _parse_refiner_response(raw)
    assert status == "missing_fields"
    assert parsed["before_wording"] == "a"
    assert parsed["after_wording"] == "b"


def test_missing_everything_returns_none():
    raw = json.dumps({"unrelated_field": "x"})
    parsed, status = _parse_refiner_response(raw)
    assert status == "missing_fields"
    assert parsed is None
