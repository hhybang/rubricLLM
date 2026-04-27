#!/usr/bin/env python3
"""
Replay verification: walk captured session data through the current code paths
to confirm recent changes don't break replay of older data.

This script does NOT start Streamlit. It exercises pure-function code paths
that touch session-shaped data: dim lookups, drift filters, inference
validation, audit-script joins. It reports any place where the new code
would have crashed, dropped data silently, or produced a different shape
from what the old code wrote.

Usage:
    python scripts/replay_sessions.py
    python scripts/replay_sessions.py --files user_chats/jianmc_chat.json user_chats/richag_chat.json

Reads from `user_chats/*_chat.json` by default.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Stub st.session_state so module imports don't blow up ──────────────────
# Several modules read st.session_state at import time; we don't want any
# Streamlit machinery active. Use a dict-backed shim that the few accessors
# we care about can read/write.
class _DummySessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


_ss = _DummySessionState()


# Try importing the real things we need; if a module touches Streamlit too
# aggressively we'll fall back to manual reimplementations.
_imports_ok = True
_import_errors: list[str] = []


def _try_import(label: str, fn):
    global _imports_ok
    try:
        return fn()
    except Exception as e:
        _imports_ok = False
        _import_errors.append(f"{label}: {e}")
        return None


# Defer streamlit import to a stub
try:
    import streamlit as st  # noqa
    # Substitute in our dummy session_state so any module-level access works.
    st.session_state = _ss  # type: ignore[attr-defined]
except Exception:
    pass


extract_primary_draft_text = _try_import(
    "extract_primary_draft_text",
    lambda: __import__("rubric_writer.draft_grading", fromlist=["extract_primary_draft_text"]).extract_primary_draft_text,
)


# Detector functions are pure of session_state, so they're safe to import.
detect_oscillation = _try_import(
    "detect_oscillation",
    lambda: __import__("rubric_writer.draft_grading", fromlist=["detect_oscillation"]).detect_oscillation,
)
detect_persistent_failure = _try_import(
    "detect_persistent_failure",
    lambda: __import__("rubric_writer.draft_grading", fromlist=["detect_persistent_failure"]).detect_persistent_failure,
)
detect_low_confidence_dims = _try_import(
    "detect_low_confidence_dims",
    lambda: __import__("rubric_writer.draft_grading", fromlist=["detect_low_confidence_dims"]).detect_low_confidence_dims,
)


# Inference filter for empty-text dims (fix from earlier today).
_validate_and_filter_rubric = _try_import(
    "_validate_and_filter_rubric",
    lambda: __import__("rubric_writer.inference", fromlist=["_validate_and_filter_rubric"])._validate_and_filter_rubric,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _parse_json_maybe(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def _draft_extract_local(content: str) -> str | None:
    """Local re-impl of extract_primary_draft_text for the case where the
    real one couldn't be imported."""
    m = re.search(r"<draft>(.*?)</draft>", content or "", re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_draft(content: str) -> str | None:
    if extract_primary_draft_text:
        return extract_primary_draft_text(content)
    return _draft_extract_local(content)


# ─── Per-session checks ─────────────────────────────────────────────────────

def check_drafts_have_content(messages: list[dict]) -> dict:
    """Every assistant message with a `draft_grade` should have extractable
    draft text. If not, the new code's anchor / drift-panel logic gates would
    silently render nothing."""
    issues = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        if not m.get("draft_grade"):
            continue
        draft = _extract_draft(m.get("content") or "")
        if not draft:
            issues.append({
                "msg_index": i,
                "message_id": m.get("message_id"),
                "issue": "graded message has no <draft>...</draft> in content",
            })
    return {"check": "drafts_have_content", "n_issues": len(issues), "issues": issues}


def check_drift_dim_ids_resolve(messages: list[dict], rubric_versions: list[dict]) -> dict:
    """Every dim_id mentioned in a persisted drift bundle should still resolve
    against SOME rubric version. If not, the panel renders `__: <raw_dim_id>`
    under the new render code."""
    # Build a "live ever" set of dim_ids = union of dim_ids across all rubric
    # versions. A dim is "ever-known" if any version had it.
    ever_known: set[str] = set()
    for rv in rubric_versions:
        rb = _parse_json_maybe(rv.get("rubric_data") or rv.get("rubric")) or {}
        for c in (rb.get("rubric") if isinstance(rb, dict) else []) or []:
            for d in c.get("dimensions") or []:
                did = (d.get("id") or "").strip().lower()
                if did:
                    ever_known.add(did)
    issues = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        drift = m.get("draft_drift") or {}
        kind = drift.get("kind") or "none"
        if kind in (None, "none"):
            continue
        entries_key = {
            "low_confidence": "low_confidence_dims",
            "spot_check": "spot_check_dims",
            "oscillation": "oscillations",
            "persistent_failure": "persistent_failure",
        }.get(kind)
        if not entries_key:
            continue
        for e in drift.get(entries_key) or []:
            did = (e.get("dimension_id") or "").strip().lower()
            if did and did not in ever_known:
                issues.append({
                    "msg_index": i,
                    "message_id": m.get("message_id"),
                    "drift_kind": kind,
                    "stale_dim_id": e.get("dimension_id"),
                    "issue": "drift bundle references dim_id not present in any rubric version",
                })
    return {"check": "drift_dim_ids_resolve", "n_issues": len(issues), "issues": issues}


def check_inference_filter_safety(rubric_versions: list[dict]) -> dict:
    """Apply the new `_validate_and_filter_rubric` to each saved rubric to see
    whether it would have dropped anything from rubrics that are now in
    production. If yes, flag the count -- replay would shrink that rubric."""
    if not _validate_and_filter_rubric:
        return {"check": "inference_filter_safety", "n_issues": 0,
                "skipped": "validator could not be imported"}
    issues = []
    for rv in rubric_versions:
        rb = _parse_json_maybe(rv.get("rubric_data") or rv.get("rubric"))
        if not isinstance(rb, dict):
            continue
        # Make a deep-ish copy so we don't mutate the source.
        rb_copy = json.loads(json.dumps(rb, default=str))
        try:
            insufficient, empty_text, crit_dropped = _validate_and_filter_rubric(
                rb_copy, context_label="replay",
            )
        except Exception as e:
            issues.append({
                "version": rv.get("version"),
                "issue": f"validator raised: {e}",
            })
            continue
        if insufficient or empty_text or crit_dropped:
            issues.append({
                "version": rv.get("version"),
                "would_drop_insufficient": insufficient,
                "would_drop_empty_text": empty_text,
                "would_drop_crits": crit_dropped,
                "issue": "new validator would shrink this rubric on replay",
            })
    return {"check": "inference_filter_safety", "n_issues": len(issues), "issues": issues}


def check_oscillation_replay(messages: list[dict]) -> dict:
    """For any draft where oscillation drift was persisted, run the new
    oscillation detector against the same dim history to see whether it
    still fires under the current rule (last-2-transitions both flips)."""
    if not detect_oscillation:
        return {"check": "oscillation_replay", "n_issues": 0,
                "skipped": "detector could not be imported"}
    # Reconstruct dim grade history across all assistant drafts in order.
    dim_history: dict[str, list[str]] = defaultdict(list)
    issues = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        dg = m.get("draft_grade") or {}
        for c in dg.get("grades") or []:
            for d in c.get("dimension_grades") or []:
                did = (d.get("dimension_id") or "").strip()
                grade = (d.get("grade") or "").upper()
                if did and grade in ("MET", "NOT_MET"):
                    dim_history[did].append(grade)
        # If this message had oscillation drift persisted, compare:
        drift = m.get("draft_drift") or {}
        if drift.get("kind") != "oscillation":
            continue
        persisted_dim_ids = {
            (o.get("dimension_id") or "").strip()
            for o in drift.get("oscillations") or []
        }
        try:
            new_oscs = detect_oscillation(dict(dim_history))
        except Exception as e:
            issues.append({
                "msg_index": i,
                "issue": f"detect_oscillation raised: {e}",
            })
            continue
        new_dim_ids = {(o.get("dimension_id") or "").strip() for o in new_oscs}
        diff = persisted_dim_ids - new_dim_ids
        if diff:
            issues.append({
                "msg_index": i,
                "message_id": m.get("message_id"),
                "issue": "persisted oscillation dims that the new detector would NOT fire on",
                "dropped_dims": sorted(diff),
            })
    return {"check": "oscillation_replay", "n_issues": len(issues), "issues": issues}


def check_audit_field_compatibility(messages: list[dict]) -> dict:
    """Quick smoke-test: walk every message's draft_grade / draft_drift /
    user_verdicts and confirm the audit-script extraction code's field
    accesses don't blow up."""
    issues = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        try:
            dg = m.get("draft_grade") or {}
            for c in dg.get("grades") or []:
                _ = c.get("criterion_name")
                _ = c.get("score")
                for d in c.get("dimension_grades") or []:
                    _ = d.get("dimension_id")
                    _ = (d.get("grade") or "").upper()
                    _ = (d.get("confidence") or "").lower()
            dd = m.get("draft_drift") or {}
            _ = dd.get("kind")
            uv = m.get("user_verdicts") or {}
            for v in uv.values():
                _ = v.get("user_grade")
        except Exception as e:
            issues.append({
                "msg_index": i,
                "issue": f"audit field access raised: {e}",
            })
    return {"check": "audit_field_compatibility", "n_issues": len(issues), "issues": issues}


# ─── Top-level driver ──────────────────────────────────────────────────────

def replay_session(path: str) -> dict:
    print(f"\n{'='*78}\nREPLAYING: {path}\n{'='*78}")
    with open(path) as f:
        convs = json.load(f)
    if not isinstance(convs, list):
        return {"path": path, "error": "expected a list of conversation dicts"}

    # Pull every message + every rubric snapshot referenced.
    all_messages: list[dict] = []
    rubric_snapshots: list[dict] = []
    for conv in convs:
        msgs = conv.get("messages")
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                msgs = []
        all_messages.extend(msgs or [])
        rubric = conv.get("rubric")
        if rubric:
            rubric_snapshots.append({
                "version": (rubric or {}).get("version") if isinstance(rubric, dict) else None,
                "rubric": rubric,
                "rubric_data": rubric,
            })

    print(f"  loaded {len(convs)} conv(s), {len(all_messages)} message(s), "
          f"{len(rubric_snapshots)} rubric snapshot(s)")

    results = []
    for fn in (
        check_drafts_have_content,
        check_audit_field_compatibility,
        check_oscillation_replay,
    ):
        r = fn(all_messages)
        results.append(r)
        print(f"  {r['check']}: {r['n_issues']} issue(s)")
    for fn in (check_drift_dim_ids_resolve,):
        r = fn(all_messages, rubric_snapshots)
        results.append(r)
        print(f"  {r['check']}: {r['n_issues']} issue(s)")
    for fn in (check_inference_filter_safety,):
        r = fn(rubric_snapshots)
        results.append(r)
        skipped = r.get("skipped")
        suffix = f"  [skipped: {skipped}]" if skipped else ""
        print(f"  {r['check']}: {r['n_issues']} issue(s){suffix}")
    return {"path": path, "results": results}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", default=None,
                   help="paths to *_chat.json files (default: glob user_chats/*_chat.json)")
    p.add_argument("--json-out", help="dump full report as JSON")
    args = p.parse_args()

    if not _imports_ok:
        print("[warn] some imports failed; replay will use local fallbacks where possible")
        for e in _import_errors:
            print(f"  - {e}")

    files = args.files or sorted(glob.glob("user_chats/*_chat.json"))
    if not files:
        print("No *_chat.json files found.")
        return 1

    report = []
    grand_issues = 0
    for path in files:
        r = replay_session(path)
        report.append(r)
        for ck in r.get("results", []):
            grand_issues += ck.get("n_issues", 0)

    print("\n" + "=" * 78)
    print(f"TOTAL ISSUES ACROSS ALL SESSIONS: {grand_issues}")
    print("=" * 78)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[full report → {args.json_out}]")

    return 0 if grand_issues == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
