"""
Background rubric grading (Sonnet) and drift detection.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import timedelta
from typing import Any

import streamlit as st

from rubric_writer.config import MODEL_LIGHT, client as default_anthropic_client

_log = logging.getLogger(__name__)

# Background grade sync interval (fragment). Env read at import; restart app to change.
_DRAFT_GRADE_POLL_EVERY = timedelta(
    seconds=float(os.environ.get("RUBRIC_GRADE_POLL_SEC", "4"))
)

GRADING_SYSTEM_PROMPT = """
You are grading a writing draft against a personalized rubric. The rubric represents one specific user's writing preferences — not generic standards.

For each criterion in the rubric, evaluate each dimension as MET or NOT_MET.

Rules:
- A dimension is MET only if the draft clearly and unambiguously satisfies what the dimension describes.
- A dimension is NOT_MET if the draft does not satisfy it, partially satisfies it, or if it's ambiguous.
- Grade what the dimension literally says. Do not grade based on your own opinion of "good writing."
- For each grade, cite the specific part of the draft (quote or reference) that justifies your decision.
- For each dimension, also report your confidence: "high", "medium", or "low".
  - "high": The dimension wording is clear and the draft unambiguously satisfies or fails it.
  - "medium": The dimension is somewhat open to interpretation, but one reading is more defensible.
  - "low": The dimension wording is genuinely ambiguous — two reasonable graders could disagree.
- When confidence is "low", add an "ambiguity_note" field (1 sentence) explaining what about the dimension wording caused the uncertainty.

Respond with ONLY valid JSON — no explanation, no preamble, no markdown fences.

For each criterion, the "score" field MUST be exactly the string \"MET_COUNT/TOTAL_COUNT\" (e.g. \"4/5\") matching your dimension_grades — no percentages, no decimals, no words.
"""

GRADING_USER_PROMPT = """
<rubric>
__RUBRIC_JSON__
</rubric>

<draft>
__DRAFT_TEXT__
</draft>

Grade every dimension of every criterion. Return:
{
  "grades": [
    {
      "criterion_name": "<name from rubric>",
      "criterion_priority": <priority from rubric>,
      "dimension_grades": [
        {
          "dimension_id": "<id from rubric>",
          "grade": "MET" | "NOT_MET",
          "confidence": "high" | "medium" | "low",
          "evidence": "<1 sentence: quote or cite the specific part of the draft>",
          "ambiguity_note": "<only when confidence is low: 1 sentence explaining what is ambiguous>"
        }
      ],
      "score": "<must equal MET count / number of dimensions above, format only: \"3/5\">"
    }
  ]
}
"""


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def load_grading_config() -> dict[str, Any]:
    return {
        "enabled": _env_bool("RUBRIC_GRADING_ENABLED", True),
        "model": os.environ.get("RUBRIC_GRADING_MODEL", MODEL_LIGHT),
        "temperature": float(os.environ.get("RUBRIC_GRADING_TEMPERATURE", "0")),
        "show_score_indicator": _env_bool("RUBRIC_SHOW_SCORE_INDICATOR", True),
        "show_drift_panels": _env_bool("RUBRIC_SHOW_DRIFT_PANELS", True),
        "grade_on_user_edit": _env_bool("RUBRIC_GRADE_ON_USER_EDIT", True),
        "grade_on_user_message": _env_bool("RUBRIC_GRADE_ON_USER_MESSAGE", True),
        "session_end_summary": _env_bool("RUBRIC_SESSION_END_SUMMARY", True),
    }


_inflight_lock = threading.Lock()
_inflight_message_ids: set[str] = set()


def extract_primary_draft_text(content: str) -> str | None:
    m = re.search(r"<draft>(.*?)</draft>", content or "", re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


_VALID_CONFIDENCE = {"high", "medium", "low"}


def normalize_grade_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Set each criterion's score to canonical \"MET/TOTAL\" from dimension_grades.
    Also normalizes confidence fields (defaults to "high" if missing/invalid)
    and strips ambiguity_note when confidence is not "low".
    """
    if not payload or not isinstance(payload, dict):
        return payload
    grades = payload.get("grades")
    if not isinstance(grades, list):
        return payload
    for crit in grades:
        if not isinstance(crit, dict):
            continue
        dims = crit.get("dimension_grades")
        if not isinstance(dims, list) or len(dims) == 0:
            continue
        total = len(dims)
        met = sum(1 for d in dims if (d.get("grade") or "").upper() == "MET")
        crit["score"] = f"{met}/{total}"
        for d in dims:
            conf = (d.get("confidence") or "").strip().lower()
            if conf not in _VALID_CONFIDENCE:
                conf = "high"
            d["confidence"] = conf
            if conf != "low":
                d.pop("ambiguity_note", None)
    return payload


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def grade_draft_sync(
    *,
    rubric_dict: dict[str, Any],
    draft_text: str,
    model: str | None = None,
    temperature: float = 0.0,
    anthropic_client=None,
) -> tuple[dict[str, Any] | None, int, str | None]:
    """
    Synchronous Sonnet grading. Returns (grades_payload_or_none, latency_ms, error).
    grades_payload matches API JSON with top-level "grades".
    """
    client = anthropic_client or default_anthropic_client
    model = model or MODEL_LIGHT
    rubric_min = {
        "writing_type": rubric_dict.get("writing_type"),
        "rubric": rubric_dict.get("rubric", []),
        "user_goals_summary": rubric_dict.get("user_goals_summary"),
    }
    rubric_json = json.dumps(rubric_min, ensure_ascii=False, indent=2)
    user = (
        GRADING_USER_PROMPT.replace("__RUBRIC_JSON__", rubric_json).replace(
            "__DRAFT_TEXT__", draft_text or ""
        )
    )
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=16000,
            temperature=temperature,
            system=GRADING_SYSTEM_PROMPT.strip(),
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        raw = "".join(parts)
        data = _extract_json_obj(raw)
        if not data or "grades" not in data:
            return None, latency_ms, "parse_error"
        normalize_grade_payload(data)
        return data, latency_ms, None
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _log.warning("grade_draft_sync failed: %s", e)
        return None, latency_ms, str(e)


def parse_score_pct(score_str: str | None) -> float | None:
    """
    Normalize rubric criterion score to a 0–1 fraction.

    The grader prompt asks for strings like '3/5', but models sometimes return
    '100%', '80%', or '0.8'. Previously we only parsed a/b, which made comparisons
    and drift logic wrong when formats mixed between drafts.
    """
    if not score_str:
        return None
    s = str(score_str).strip()
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b <= 0:
            return None
        return a / b
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$", s)
    if m:
        v = float(m.group(1))
        return max(0.0, min(1.0, v / 100.0))
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
    return None


def find_criterion(grades_payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for c in grades_payload.get("grades") or []:
        if (c.get("criterion_name") or "").strip() == (name or "").strip():
            return c
    return None


def was_met(prev_crit: dict[str, Any], dim_id: str) -> bool:
    for d in prev_crit.get("dimension_grades") or []:
        if (d.get("dimension_id") or "").strip() == (dim_id or "").strip():
            return (d.get("grade") or "").upper() == "MET"
    return False


def _criterion_fraction(crit: dict[str, Any]) -> float | None:
    return parse_score_pct(crit.get("score"))


# ---------------------------------------------------------------------------
# Drift heuristics (4 total, all dimension-level)
# ---------------------------------------------------------------------------

def detect_low_confidence_dims(grades: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract dimensions with low confidence from a single grade payload."""
    results: list[dict[str, Any]] = []
    for c in grades.get("grades") or []:
        cname = (c.get("criterion_name") or "").strip()
        for d in c.get("dimension_grades") or []:
            if d.get("confidence") == "low":
                results.append({
                    "criterion": cname,
                    "dimension_id": d.get("dimension_id", ""),
                    "grade": d.get("grade", ""),
                    "evidence": d.get("evidence", ""),
                    "ambiguity_note": d.get("ambiguity_note", ""),
                })
    return results


def detect_oscillation(
    dim_history: dict[str, list[str]], min_flips: int = 2,
) -> list[dict[str, Any]]:
    """Detect dimensions that flip MET<->NOT_MET repeatedly.

    dim_history: {dimension_id: [grade_str, ...]} across consecutive graded drafts.
    A dimension oscillates when it has >= min_flips direction changes in its history.
    """
    results: list[dict[str, Any]] = []
    for dim_id, grades in dim_history.items():
        if len(grades) < 3:
            continue
        flips = 0
        for i in range(1, len(grades)):
            if grades[i] != grades[i - 1]:
                flips += 1
        if flips >= min_flips:
            results.append({"dimension_id": dim_id, "flips": flips, "history": grades[-6:]})
    return results


def detect_persistent_failure(
    dim_grade_history: dict[str, list[str]],
    current: dict[str, Any],
    min_streak: int = 2,
) -> list[dict[str, Any]]:
    """Dimensions that have been NOT_MET for N consecutive drafts (from the end
    of their history). Covers all cases of dimensions stuck at NOT_MET.

    Uses dim_grade_history (accumulated across all graded drafts) to count the
    trailing NOT_MET streak for each dimension.
    """
    results: list[dict[str, Any]] = []
    # Build dim_id -> criterion lookup from current grades
    dim_to_crit: dict[str, str] = {}
    for c in current.get("grades") or []:
        cname = (c.get("criterion_name") or "").strip()
        for d in c.get("dimension_grades") or []:
            did = (d.get("dimension_id") or "").strip()
            if did:
                dim_to_crit[did] = cname

    for dim_id, grades in dim_grade_history.items():
        if len(grades) < min_streak:
            continue
        # Count trailing NOT_MET streak
        streak = 0
        for g in reversed(grades):
            if g == "NOT_MET":
                streak += 1
            else:
                break
        if streak >= min_streak:
            results.append({
                "dimension_id": dim_id,
                "criterion": dim_to_crit.get(dim_id, ""),
                "streak": streak,
            })
    return results


def detect_tradeoff(
    current_grades: dict[str, Any],
    previous_grades: dict[str, Any],
    delta_threshold: float = 0.02,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Criteria that improved vs slipped compared to the previous graded payload.
    Returns two lists: improvements and drops."""
    improvements: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for curr in current_grades.get("grades") or []:
        name = (curr.get("criterion_name") or "").strip()
        if not name:
            continue
        prev = find_criterion(previous_grades, name)
        if prev is None:
            continue
        c = parse_score_pct(curr.get("score"))
        p = parse_score_pct(prev.get("score"))
        if c is None or p is None:
            continue
        delta = c - p
        rec = {
            "name": name,
            "delta": delta,
            "from": p,
            "to": c,
            "score_str_prev": str(prev.get("score") or "").strip(),
            "score_str_curr": str(curr.get("score") or "").strip(),
        }
        if delta > delta_threshold:
            improvements.append(rec)
        elif delta < -delta_threshold:
            drops.append(rec)
    return improvements, drops


# ---------------------------------------------------------------------------
# Noise suppression
# ---------------------------------------------------------------------------

def _should_suppress_drift(
    current: dict[str, Any],
    draft_index: int | None,
    trigger: str | None,
) -> bool:
    """Suppress all drift heuristics (except low_confidence) when the signal
    would be unreliable: early drafts, user edits, or very small rubrics."""
    if draft_index is not None and int(draft_index) <= 2:
        return True
    if trigger == "user_edit":
        return True
    total_dims = sum(
        len(c.get("dimension_grades") or [])
        for c in (current.get("grades") or [])
    )
    if total_dims <= 3:
        return True
    return False


# ---------------------------------------------------------------------------
# Drift bundle
# ---------------------------------------------------------------------------

def _sample_spot_check_dims(
    current: dict[str, Any],
    dims_already_surfaced: set[str],
    sample_size: int = 2,
) -> list[dict[str, Any]]:
    """Randomly sample high/medium-confidence dimensions that have NOT been
    surfaced in any drift panel. Spot-checking catches silent misalignment
    where the grader is confident but wrong."""
    import random

    medium: list[dict[str, Any]] = []
    high: list[dict[str, Any]] = []
    for c in current.get("grades") or []:
        cname = (c.get("criterion_name") or "").strip()
        for d in c.get("dimension_grades") or []:
            did = (d.get("dimension_id") or "").strip()
            if not did or did in dims_already_surfaced:
                continue
            entry = {
                "criterion": cname,
                "dimension_id": did,
                "grade": (d.get("grade") or "").upper(),
                "confidence": d.get("confidence", "high"),
                "evidence": d.get("evidence", ""),
            }
            if d.get("confidence") == "medium":
                medium.append(entry)
            elif d.get("confidence") == "high":
                high.append(entry)
    # Prioritize medium confidence — more likely to be silently misaligned
    random.shuffle(medium)
    random.shuffle(high)
    pool = medium + high
    if not pool:
        return []
    return pool[:sample_size]


def compute_drift_bundle(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    cfg: dict[str, Any],
    dim_grade_history: dict[str, list[str]] | None = None,
    draft_index: int | None = None,
    trigger: str | None = None,
    dims_already_surfaced: set[str] | None = None,
) -> dict[str, Any]:
    """Compute drift signals for the UI. Five kinds, priority order:
    1. low_confidence -- grader can't interpret the rubric
    2. oscillation -- rubric wording is ambiguous (dimension keeps flipping)
    3. persistent_failure -- dimension NOT_MET for N consecutive drafts
    4. tradeoff -- some criteria improved while others dropped
    5. spot_check -- random sample of quiet dimensions to catch silent misalignment
    """
    low_conf = detect_low_confidence_dims(current) if current else []

    if not previous:
        bundle: dict[str, Any] = {"kind": "none"}
        if low_conf:
            bundle["kind"] = "low_confidence"
            bundle["low_confidence_dims"] = low_conf
        return bundle

    # Detect all signals
    oscillations = detect_oscillation(dim_grade_history or {})
    persistent = detect_persistent_failure(dim_grade_history or {}, current)
    tradeoff_improvements, tradeoff_drops = detect_tradeoff(current, previous)

    # Noise suppression
    suppressed = _should_suppress_drift(current, draft_index, trigger)

    # Priority order
    kind = "none"

    # Low-confidence always fires
    if low_conf:
        kind = "low_confidence"

    if not suppressed:
        if oscillations:
            kind = "oscillation"
        elif persistent:
            kind = "persistent_failure"
        elif tradeoff_improvements and tradeoff_drops:
            kind = "tradeoff"

    # If no heuristic fired, spot-check random quiet dimensions
    spot_check_dims: list[dict[str, Any]] = []
    if kind == "none" and not suppressed:
        # Collect all dims that have been in any drift signal
        surfaced = set(dims_already_surfaced or set())
        for lc in low_conf:
            surfaced.add(lc.get("dimension_id", ""))
        for o in oscillations:
            surfaced.add(o.get("dimension_id", ""))
        for p in persistent:
            surfaced.add(p.get("dimension_id", ""))
        spot_check_dims = _sample_spot_check_dims(current, surfaced)
        if spot_check_dims:
            kind = "spot_check"

    return {
        "kind": kind,
        "low_confidence_dims": low_conf,
        "oscillations": oscillations,
        "persistent_failure": persistent,
        "tradeoff_improvements": tradeoff_improvements,
        "tradeoff_drops": tradeoff_drops,
        "spot_check_dims": spot_check_dims,
        "suppressed": suppressed,
    }


def merge_draft_grades_into_messages(
    messages: list[dict[str, Any]], rows: list[dict[str, Any]], cfg: dict[str, Any]
) -> None:
    """Attach draft_grade + drift_ui to assistant messages by message_id.

    Past drafts that already have a draft_drift are treated as frozen historical
    records -- we don't recompute their drift. Only the latest draft (or drafts
    that have never been graded yet) get fresh drift computation. This keeps
    past interactions as stable logs even when the rubric changes.
    """
    # Pick the EARLIEST grade row for each message_id. Past drafts are
    # historical records -- their grade is a fact about the draft at the time
    # it was written, judged against the rubric in force then. If the same
    # draft was accidentally re-graded later (different rubric version, or a
    # duplicate insertion), the original row wins so past dots stay frozen.
    # `rows` comes back from the DB ordered by draft_index ascending, so we
    # just need to skip writes after the first for each mid.
    by_mid: dict[str, dict[str, Any]] = {}
    for r in rows:
        mid = r.get("message_id")
        if not mid:
            continue
        key = str(mid)
        if key in by_mid:
            continue  # keep the earliest (original) grade
        by_mid[key] = r

    # Identify the latest graded draft -- only it (and any ungraded-yet drafts)
    # should have drift recomputed. Earlier drafts stay frozen.
    latest_draft_mid: str | None = None
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        if not extract_primary_draft_text(m.get("content") or ""):
            continue
        latest_draft_mid = str(m.get("message_id") or "")
        break

    prev_payload: dict[str, Any] | None = None
    prev_rv: int | None = None
    dim_grade_history: dict[str, list[str]] = {}
    dims_already_surfaced: set[str] = set()

    for m in messages:
        # Skip messages with a debug drift override
        if m.get("_debug_drift_override"):
            continue
        mid = str(m.get("message_id") or "")
        row = by_mid.get(mid)
        if row and row.get("grades_json"):
            gj = row["grades_json"]
            if isinstance(gj, str):
                try:
                    gj = json.loads(gj)
                except json.JSONDecodeError:
                    gj = None
            if isinstance(gj, dict):
                normalize_grade_payload(gj)
                m["draft_grade"] = gj
                m["draft_grade_meta"] = {
                    "draft_index": row.get("draft_index"),
                    "rubric_version": row.get("rubric_version"),
                    "graded_at": row.get("graded_at"),
                    "latency_ms": row.get("latency_ms"),
                    "trigger": row.get("trigger"),
                }
                has_draft = bool(extract_primary_draft_text(m.get("content") or ""))
                if cfg["show_drift_panels"] and has_draft:
                    # PERSISTED drift wins: under Option B, drift is computed
                    # once in the background grading thread and stored in the
                    # DB row's drift_json column. If present, use it and never
                    # recompute -- the dots and panels become immutable once
                    # grading completes. This eliminates the "dots changed
                    # color" bug where drift could be recomputed differently
                    # across reruns.
                    persisted_drift = None
                    raw_drift = row.get("drift_json")
                    if raw_drift is not None:
                        try:
                            persisted_drift = (
                                json.loads(raw_drift)
                                if isinstance(raw_drift, str)
                                else raw_drift
                            )
                        except json.JSONDecodeError:
                            persisted_drift = None
                    if isinstance(persisted_drift, dict) and persisted_drift:
                        drift = persisted_drift
                    else:
                        # Legacy fallback: row has no persisted drift (either a
                        # legacy row from before Option B, or the thread's
                        # drift computation failed).
                        is_latest = (mid == latest_draft_mid)
                        has_existing_drift = bool(m.get("draft_drift"))
                        existing_drift = m.get("draft_drift") or {}

                        if existing_drift.get("kind") == "spot_check":
                            # Preserve spot_check to avoid re-randomizing on rerun
                            drift = existing_drift
                        elif not is_latest:
                            # PAST DRAFTS MUST NEVER GROW NEW DRIFT SIGNALS.
                            # If we already have drift in memory for this past
                            # draft, keep it. Otherwise freeze at "none" --
                            # do NOT recompute spot_check retroactively, which
                            # would make past drafts sprout new Quick-check
                            # panels (with different dims each render since
                            # spot_check samples randomly).
                            drift = existing_drift if has_existing_drift else {"kind": "none"}
                        else:
                            same_rubric = prev_rv == row.get("rubric_version")
                            drift = compute_drift_bundle(
                                gj,
                                prev_payload if same_rubric else None,
                                cfg,
                                dim_grade_history=dim_grade_history,
                                draft_index=row.get("draft_index"),
                                trigger=row.get("trigger"),
                                dims_already_surfaced=dims_already_surfaced,
                            )
                    m["draft_drift"] = drift
                    # Log fire rate (Metric 4)
                    try:
                        from rubric_writer.metrics import log_fire_rate
                        total_dims = sum(len(c.get("dimension_grades") or []) for c in gj.get("grades") or [])
                        drift_kind = drift.get("kind") or "none"
                        panels_fired = 0 if drift_kind == "none" else 1
                        suppression_reasons = []
                        if drift.get("suppressed"):
                            d_idx = row.get("draft_index")
                            if d_idx is not None and int(d_idx) <= 2:
                                suppression_reasons.append("early_draft")
                            if row.get("trigger") == "user_edit":
                                suppression_reasons.append("user_edit")
                            if total_dims <= 3:
                                suppression_reasons.append("small_rubric")
                        log_fire_rate(
                            draft_index=row.get("draft_index") or 0,
                            total_dimensions=total_dims,
                            panels_fired=panels_fired,
                            panels_by_type={
                                "low_confidence": 1 if drift_kind == "low_confidence" else 0,
                                "oscillation": 1 if drift_kind == "oscillation" else 0,
                                "persistent_failure": 1 if drift_kind == "persistent_failure" else 0,
                                "tradeoff": 1 if drift_kind == "tradeoff" else 0,
                                "spot_check": 1 if drift_kind == "spot_check" else 0,
                            },
                            panels_suppressed=1 if drift.get("suppressed") and drift_kind == "none" else 0,
                            suppression_reasons=suppression_reasons,
                        )
                    except Exception:
                        pass
                    # Track which dims have been surfaced so spot_check avoids them
                    for lc in drift.get("low_confidence_dims") or []:
                        dims_already_surfaced.add(lc.get("dimension_id", ""))
                    for o in drift.get("oscillations") or []:
                        dims_already_surfaced.add(o.get("dimension_id", ""))
                    for p in drift.get("persistent_failure") or []:
                        dims_already_surfaced.add(p.get("dimension_id", ""))
                    for sc in drift.get("spot_check_dims") or []:
                        dims_already_surfaced.add(sc.get("dimension_id", ""))
                elif row.get("drift_json"):
                    try:
                        m["draft_drift"] = (
                            json.loads(row["drift_json"])
                            if isinstance(row["drift_json"], str)
                            else row["drift_json"]
                        )
                    except json.JSONDecodeError:
                        m["draft_drift"] = {"kind": "none"}
                # Accumulate dimension grade history for subsequent drafts
                if has_draft:
                    for c in gj.get("grades") or []:
                        for d in c.get("dimension_grades") or []:
                            did = (d.get("dimension_id") or "").strip()
                            if did:
                                dim_grade_history.setdefault(did, []).append(
                                    (d.get("grade") or "").upper()
                                )
                    prev_payload = gj
                    prev_rv = row.get("rubric_version")


def sync_draft_grades_into_session() -> None:
    cfg = load_grading_config()
    if not cfg["enabled"]:
        return
    supabase = st.session_state.get("supabase")
    conv_id = st.session_state.get("selected_conversation")
    if not supabase or not conv_id:
        return
    try:
        from auth_supabase import fetch_draft_grades_for_conversation

        rows = fetch_draft_grades_for_conversation(supabase, conv_id)
        merge_draft_grades_into_messages(st.session_state.messages, rows, cfg)
    except Exception as e:
        _log.warning("sync_draft_grades_into_session: %s", e)


def grade_poll_fragment_enabled() -> bool:
    """Periodic Supabase sync while drafts lack grades. Set RUBRIC_GRADE_POLL_ENABLED=0 to disable."""
    return os.environ.get("RUBRIC_GRADE_POLL_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def count_pending_draft_grades(messages: list[dict[str, Any]] | None) -> int:
    """
    Assistant messages that show 'scoring in progress' — draft present, grade not merged yet.
    Only counts when grading could run (enabled + active rubric) to avoid infinite polling.
    """
    if not messages:
        return 0
    cfg = load_grading_config()
    if not cfg.get("enabled"):
        return 0
    try:
        from rubric_writer.persistence import get_active_rubric

        active, _, _ = get_active_rubric()
        if not active or not active.get("rubric"):
            return 0
    except Exception:
        return 0
    n = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        if not extract_primary_draft_text(m.get("content") or ""):
            continue
        if m.get("draft_grade"):
            continue
        n += 1
    return n


@st.fragment(run_every=_DRAFT_GRADE_POLL_EVERY)
def run_draft_grade_poll_fragment() -> None:
    """
    While grades are missing, re-fetch from Supabase on an interval.

    Calling st.rerun() on every partial merge (after < before) caused full-app
    rerun storms when many drafts finished one-by-one. We only rerun when new
    grades actually landed, and at most once per RERUN_MIN_INTERVAL seconds.
    """
    cfg = load_grading_config()
    if not cfg.get("enabled"):
        return
    if not st.session_state.get("supabase") or not st.session_state.get("selected_conversation"):
        return
    msgs = st.session_state.get("messages") or []
    before = count_pending_draft_grades(msgs)
    if before == 0:
        return
    sync_draft_grades_into_session()
    after = count_pending_draft_grades(st.session_state.get("messages") or [])
    now = time.monotonic()
    rerun_key = "_draft_grade_poll_rerun_ts"
    rerun_min_s = float(os.environ.get("RUBRIC_GRADE_POLL_RERUN_MIN_SEC", "10"))
    auto_rerun = os.environ.get("RUBRIC_GRADE_POLL_AUTO_RERUN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if auto_rerun and after < before:
        last = st.session_state.get(rerun_key)
        last_f = float(last) if isinstance(last, (int, float)) else 0.0
        if now - last_f >= rerun_min_s:
            st.session_state[rerun_key] = now
            st.rerun()
    st.empty()



def _compute_drift_for_persist(
    *,
    supabase: Any,
    conversation_id: str,
    current_grades: dict[str, Any],
    current_rubric_version: int,
    current_draft_index: int,
    trigger: str | None,
    fetch_fn,
) -> dict[str, Any] | None:
    """Compute a drift bundle for a just-graded draft using prior rows in the
    same conversation. Runs in the background grading thread (no Streamlit).

    Previous-draft data is reconstructed from `draft_grades` rows with lower
    draft_index. dim_grade_history and dims_already_surfaced are built from
    the full history so drift detection sees the same context the UI would
    have computed on its own.

    Returns a drift dict or None on any error -- None means we fall back to
    on-render computation (legacy behavior)."""
    try:
        cfg = load_grading_config()
        if not cfg.get("show_drift_panels"):
            return None

        rows = fetch_fn(supabase, conversation_id) or []
        # Only consider rows strictly before this draft and on the SAME rubric
        # version -- drift comparisons across rubric versions are meaningless.
        prior_rows = [
            r for r in rows
            if (r.get("draft_index") or 0) < current_draft_index
            and r.get("rubric_version") == current_rubric_version
        ]

        prev_payload = None
        if prior_rows:
            latest_prior = prior_rows[-1]
            gj = latest_prior.get("grades_json")
            if isinstance(gj, str):
                try:
                    gj = json.loads(gj)
                except json.JSONDecodeError:
                    gj = None
            if isinstance(gj, dict):
                prev_payload = gj

        # Rebuild dim_grade_history across all same-rubric prior rows
        # (earliest to latest).
        dim_grade_history: dict[str, list[str]] = {}
        dims_already_surfaced: set[str] = set()
        for r in prior_rows:
            gj = r.get("grades_json")
            if isinstance(gj, str):
                try:
                    gj = json.loads(gj)
                except json.JSONDecodeError:
                    continue
            if not isinstance(gj, dict):
                continue
            for c in gj.get("grades") or []:
                for d in c.get("dimension_grades") or []:
                    did = (d.get("dimension_id") or "").strip()
                    if did:
                        dim_grade_history.setdefault(did, []).append(
                            (d.get("grade") or "").upper()
                        )
            # Track dims that were already surfaced in past drafts so
            # spot_check avoids re-picking them.
            drift = r.get("drift_json")
            if isinstance(drift, str):
                try:
                    drift = json.loads(drift)
                except json.JSONDecodeError:
                    drift = None
            if isinstance(drift, dict):
                for lc in drift.get("low_confidence_dims") or []:
                    dims_already_surfaced.add(lc.get("dimension_id", ""))
                for o in drift.get("oscillations") or []:
                    dims_already_surfaced.add(o.get("dimension_id", ""))
                for p in drift.get("persistent_failure") or []:
                    dims_already_surfaced.add(p.get("dimension_id", ""))
                for sc in drift.get("spot_check_dims") or []:
                    dims_already_surfaced.add(sc.get("dimension_id", ""))

        return compute_drift_bundle(
            current_grades,
            prev_payload,
            cfg,
            dim_grade_history=dim_grade_history,
            draft_index=current_draft_index,
            trigger=trigger,
            dims_already_surfaced=dims_already_surfaced,
        )
    except Exception as e:
        _log.warning("_compute_drift_for_persist failed: %s", e)
        return None


def schedule_background_grade(
    *,
    supabase: Any,
    conversation_id: str | None,
    message_id: str,
    draft_text: str,
    rubric_version: int | None,
    rubric_dict: dict[str, Any] | None,
    trigger: str,
) -> None:
    cfg = load_grading_config()
    if not cfg["enabled"]:
        return
    if (
        not supabase
        or not conversation_id
        or not draft_text
        or not rubric_dict
        or rubric_version is None
    ):
        return
    if trigger == "user_message" and not cfg["grade_on_user_message"]:
        return
    if trigger == "user_edit" and not cfg["grade_on_user_edit"]:
        return
    try:
        _rv = int(rubric_version)
    except (TypeError, ValueError):
        return

    with _inflight_lock:
        if message_id in _inflight_message_ids:
            return
        _inflight_message_ids.add(message_id)

    _model = cfg["model"]
    _temp = cfg["temperature"]
    # Detach by deep-serializing. The display layer has historically attached
    # a `_diff` key with set() values to criteria; json.dumps chokes on sets.
    # Strip `_diff` and coerce any stray set() to a list before serializing.
    def _sanitize(x):
        if isinstance(x, dict):
            return {k: _sanitize(v) for k, v in x.items() if k != "_diff"}
        if isinstance(x, list):
            return [_sanitize(v) for v in x]
        if isinstance(x, set):
            return sorted(x)
        return x
    _rubric = json.loads(json.dumps(_sanitize(rubric_dict)))  # detach

    def _run():
        try:
            from auth_supabase import (
                insert_draft_grade, next_draft_grade_index,
                fetch_draft_grades_for_conversation,
            )

            # Belt-and-suspenders freeze: once a draft has a row in the DB,
            # we NEVER grade it again. The in-memory `_inflight_message_ids`
            # set only covers the current process; a DB check guarantees
            # immutability across sessions, reloads, and re-entrant calls.
            # This is what keeps past drafts' grades/dots frozen.
            try:
                existing_rows = fetch_draft_grades_for_conversation(
                    supabase, conversation_id
                ) or []
                if any(
                    str(r.get("message_id")) == str(message_id)
                    for r in existing_rows
                ):
                    _log.info(
                        "schedule_background_grade: message_id=%s already "
                        "has a grade row; skipping re-grade.", message_id,
                    )
                    return
            except Exception as e:
                _log.warning(
                    "schedule_background_grade: existing-row check failed "
                    "(%s); proceeding with grade.", e,
                )

            grades, latency_ms, err = grade_draft_sync(
                rubric_dict=_rubric,
                draft_text=draft_text,
                model=_model,
                temperature=_temp,
            )
            if err or not grades:
                return
            d_idx = next_draft_grade_index(supabase, conversation_id)

            # Option B: compute drift ONCE at grading time and persist it to
            # the DB so the UI never has to recompute. Inputs are reconstructed
            # from prior rows in this conversation's draft_grades table.
            drift_json = _compute_drift_for_persist(
                supabase=supabase,
                conversation_id=conversation_id,
                current_grades=grades,
                current_rubric_version=_rv,
                current_draft_index=d_idx,
                trigger=trigger,
                fetch_fn=fetch_draft_grades_for_conversation,
            )

            insert_draft_grade(
                supabase,
                conversation_id=conversation_id,
                message_id=message_id,
                draft_index=d_idx,
                draft_text=draft_text[:500000],
                rubric_version=_rv,
                grades_json=grades,
                model_used=_model,
                latency_ms=latency_ms,
                trigger=trigger,
                drift_json=drift_json,
            )
        except Exception as e:
            _log.warning("background grade thread: %s", e)
        finally:
            with _inflight_lock:
                _inflight_message_ids.discard(message_id)

    threading.Thread(target=_run, daemon=True).start()


def maybe_schedule_pending_grades(max_jobs: int = 2) -> None:
    """Best-effort: grade recent assistant drafts that have no DB row yet (e.g. after reload)."""
    cfg = load_grading_config()
    if not cfg["enabled"]:
        return
    supabase = st.session_state.get("supabase")
    conv_id = st.session_state.get("selected_conversation")
    if not supabase or not conv_id:
        return
    try:
        from auth_supabase import fetch_draft_grades_for_conversation

        existing = {str(r.get("message_id")) for r in fetch_draft_grades_for_conversation(supabase, conv_id)}
    except Exception:
        return
    from rubric_writer.persistence import get_active_rubric

    active_rubric, _, _ = get_active_rubric()
    if not active_rubric or not active_rubric.get("rubric"):
        return
    rv = active_rubric.get("version")
    n = 0
    for m in reversed(st.session_state.get("messages", [])):
        if n >= max_jobs:
            break
        if m.get("role") != "assistant":
            continue
        if m.get("_pre_rubric"):
            continue
        mid = str(m.get("message_id") or "")
        if not mid or mid in existing:
            continue
        dt = extract_primary_draft_text(m.get("content") or "")
        if not dt:
            continue
        schedule_background_grade(
            supabase=supabase,
            conversation_id=conv_id,
            message_id=mid,
            draft_text=dt,
            rubric_version=rv,
            rubric_dict=active_rubric,
            trigger="manual",
        )
        n += 1
