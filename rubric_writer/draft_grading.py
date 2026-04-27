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

For each dimension of each criterion, produce TWO independent judgments:

  1. Is the rubric DIMENSION WORDING clear?
     Ask: "If a second, equally-careful grader read ONLY this dimension's wording (no draft), would they understand what to check for in the same way I do?"
       - If yes → confidence "high"
       - If the wording leaves room for reasonable interpretation but one reading
         is clearly more defensible → confidence "medium"
       - If two careful graders could reasonably disagree on what the dimension
         even means, independent of any specific draft → confidence "low"

  2. Does the DRAFT satisfy the dimension, under your best reading of its wording?
       - "MET" — the draft clearly satisfies what you understand the dimension to ask for
       - "NOT_MET" — the draft does not satisfy it (absent, partially present, or falls short)

IMPORTANT: these two judgments are independent. A dimension can be:
  - MET with high confidence (clear wording, draft satisfies it)
  - NOT_MET with high confidence (clear wording, draft fails it)
  - MET with low confidence (wording is ambiguous but your best reading is satisfied)
  - NOT_MET with low confidence (wording is ambiguous AND draft doesn't clearly meet any reading)

Do NOT grade NOT_MET just because the dimension wording is unclear. If the wording
is unclear, that's a LOW-CONFIDENCE signal; you still commit to MET or NOT_MET
based on your best reading of the draft.

When confidence is "low", include a one-sentence "ambiguity_note" explaining
what about the DIMENSION WORDING (not the draft) is ambiguous. Examples:
  - "'clearly' is subjective — could mean plainly stated OR jargon-free"
  - "dimension doesn't specify how many examples count as 'several'"
  - "'professional but warm' sets two criteria that can conflict"

Other rules:
- Grade what the dimension literally says. Do not grade based on your own opinion of "good writing."
- For each grade, cite the specific part of the draft (quote or reference) that justifies your MET/NOT_MET decision.

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

# Serializes the "pick next draft_index + insert row" sequence across all
# grading threads in this process. Without this, two concurrent threads can
# both read next_draft_grade_index=1 from the DB before either inserts, and
# both end up assigning draft_idx=1 to different messages.
_index_insert_lock = threading.Lock()

# Process-local memo of message_ids that have been successfully graded by
# THIS process. Prevents maybe_schedule_pending_grades from re-scheduling a
# mid immediately after grading completes but before the DB fetch sees the
# row. Without this, we race: thread A finishes → removes itself from
# _inflight → Streamlit reruns → maybe_schedule_pending_grades fetches DB
# (cache miss) → re-schedules the same mid → duplicate grade. This set is
# checked alongside the DB fetch to cover the propagation window.
_recently_graded_mids: set[str] = set()

# Mids whose grades need to be rolled back into conversations.messages JSON
# (so a conversation reload can see the grade without re-querying
# draft_grades, which RLS may block). Populated by the grading thread;
# drained by the main thread on each render via flush_pending_conversation_save.
_pending_conversation_save_mids: set[str] = set()

# Process-local memo of duplicate-row warnings we've already emitted, to avoid
# spamming the terminal on every rerun of a conversation that has dups.
_DUP_WARNED: set[tuple] = set()


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
    dim_history: dict[str, list[str]],
    min_runs: int = 3,
) -> list[dict[str, Any]]:
    """Detect dimensions that flip MET<->NOT_MET repeatedly AND are still
    currently unstable.

    dim_history: {dimension_id: [grade_str, ...]} across consecutive graded drafts.

    Two requirements:
      1. At least min_runs distinct runs in the last 6 drafts (so there's
         been real back-and-forth, not just a single blip).
      2. At least one flip in the last 2 transitions (so the dim is still
         actually unstable, not recovering from earlier instability).

    Without requirement 2, sequences like [NOT_MET, MET, NOT_MET, NOT_MET,
    MET, MET] fire "oscillating" even though the last two drafts agree --
    which reads to the user as "this is stabilizing," not "this keeps
    flipping." We want the panel to say "the dim is unstable RIGHT NOW,"
    not "the dim was unstable at some point."

    Examples (min_runs=3):
      [MET, MET, NOT_MET]                          → 2 runs → ✗
      [MET, MET, MET, NOT_MET, MET]                → 3 runs, last flip at -1 → ✓
      [NOT_MET, MET, NOT_MET, MET]                 → 4 runs, flipping → ✓
      [NOT_MET, MET, NOT_MET, NOT_MET, MET, MET]   → 4 runs BUT last 2 trans agree → ✗
      [MET, MET, NOT_MET, MET, MET, MET]           → 3 runs BUT last 2 trans agree → ✗
    """
    results: list[dict[str, Any]] = []
    for dim_id, grades in dim_history.items():
        if len(grades) < min_runs:
            continue
        window = grades[-6:]
        runs = 0
        prev = None
        for g in window:
            if g != prev:
                runs += 1
                prev = g
        if runs < min_runs:
            continue
        # Require the last 3 grades to form a MET↔NOT_MET↔MET (or
        # NOT_MET↔MET↔NOT_MET) pattern -- i.e. both of the last two
        # transitions must be flips. That's the signature of "currently
        # oscillating." Anything else (flat tail, single flip recovering
        # from a long same-grade streak) isn't oscillation for panel
        # purposes.
        #
        #   MET → NOT_MET → MET       → flip, flip   → FIRE
        #   NOT_MET → MET → NOT_MET   → flip, flip   → FIRE
        #   NOT_MET → NOT_MET → MET   → same, flip   → don't fire (recovering)
        #   MET → NOT_MET → NOT_MET   → flip, same   → don't fire (stabilizing on NOT_MET)
        #   MET → MET → MET           → same, same   → don't fire (stable)
        if len(window) < 3:
            continue
        last_flip = window[-1] != window[-2]
        prev_flip = window[-2] != window[-3]
        if not (last_flip and prev_flip):
            continue
        flips = runs - 1
        results.append({"dimension_id": dim_id, "flips": flips, "history": window})
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
    """Suppress the MODEL-vs-rubric drift heuristics (oscillation, persistent
    failure, tradeoff) when the signal would be unreliable: early drafts,
    user edits, or very small rubrics.

    Note: low_confidence always fires regardless of suppression (it's about
    the grader's ambiguity, not about draft history). spot_check has its
    own suppression rule in `_should_suppress_spot_check` below -- it's
    more permissive because spot_check is a user-vs-grader calibration
    probe, not a model-vs-rubric signal, so user_edit drafts are fine
    (arguably better) candidates."""
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


def _should_suppress_spot_check(
    current: dict[str, Any],
    draft_index: int | None,
    trigger: str | None,
) -> bool:
    """Spot_check suppression is narrower than the general drift suppression.
    Spot_check asks the user directly about a dim on the current draft, so:

    - Early drafts (index <= 2): still suppressed. No history yet to inform
      the "this looks clean but is it really?" probe.
    - User edits: NOT suppressed. The user explicitly edited the text, so
      their opinion on whether it meets a dim is maximally informed.
    - Tiny rubrics (<=3 dims): still suppressed. Not enough quiet dims to
      sample from without re-asking about everything.
    """
    if draft_index is not None and int(draft_index) <= 2:
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
    """Sample dimensions for the spot-check probe.

    Strategy: **prefer high-confidence MET dims**, then high-confidence
    NOT_MET, then medium. The paper's framing is "silent misalignment" --
    catch the case where the grader is confidently wrong. High-confidence
    MET dims are the ones the grader is most sure about, so they're the
    ones where a user disagreement would be most informative for the
    confirmation-rate metric.

    Excludes any dim already surfaced by a drift panel earlier in the
    session (passed in via `dims_already_surfaced`), per the paper's
    "non-drift dimensions" requirement -- otherwise we'd be re-asking
    about contested dims and contaminating the uncontested-rate
    denominator.
    """
    import random

    high_met: list[dict[str, Any]] = []
    high_not_met: list[dict[str, Any]] = []
    medium: list[dict[str, Any]] = []
    for c in current.get("grades") or []:
        cname = (c.get("criterion_name") or "").strip()
        for d in c.get("dimension_grades") or []:
            did = (d.get("dimension_id") or "").strip()
            if not did or did in dims_already_surfaced:
                continue
            grade = (d.get("grade") or "").upper()
            conf = (d.get("confidence") or "high").lower()
            entry = {
                "criterion": cname,
                "dimension_id": did,
                "grade": grade,
                "confidence": conf,
                "evidence": d.get("evidence", ""),
            }
            if conf == "high" and grade == "MET":
                high_met.append(entry)
            elif conf == "high" and grade == "NOT_MET":
                high_not_met.append(entry)
            elif conf == "medium":
                medium.append(entry)
    random.shuffle(high_met)
    random.shuffle(high_not_met)
    random.shuffle(medium)
    pool = high_met + high_not_met + medium
    if not pool:
        return []
    return pool[:sample_size]


def _count_trailing_perfect_streak(
    dim_grade_history: dict[str, list[str]],
) -> int:
    """Count the number of consecutive draft positions (ending at the most
    recent draft) where every dim was MET. Used to throttle spot_check so
    it only fires every 3 perfect drafts, not on every clean draft."""
    if not dim_grade_history:
        return 0
    # Assume all dim histories are the same length (one entry per graded
    # draft in order). If not, use the min length as the shared range.
    lengths = [len(h) for h in dim_grade_history.values() if h]
    if not lengths:
        return 0
    shared = min(lengths)
    streak = 0
    for i in range(shared - 1, -1, -1):
        all_met = True
        for history in dim_grade_history.values():
            if i >= len(history):
                all_met = False
                break
            if (history[i] or "").upper() != "MET":
                all_met = False
                break
        if all_met:
            streak += 1
        else:
            break
    return streak


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

    # If no heuristic fired, spot-check random quiet dimensions.
    # Throttle: fire only once per 3 consecutive perfect-score drafts.
    # Without this, spot_check fires on every draft where nothing else
    # drifted, which is noisy when the user is happily producing clean
    # drafts. We only want to catch SILENT MISALIGNMENT across a streak
    # of 100%s, so once every 3 perfect drafts is the right cadence.
    spot_check_dims: list[dict[str, Any]] = []
    # Spot_check uses a narrower suppression rule than the other drift kinds:
    # it's a user-vs-grader calibration probe, so user-edited drafts are
    # valid (the user's opinion on whether the edited text meets a dim is
    # informed, not noise). Early drafts and tiny rubrics still suppress.
    spot_check_suppressed = _should_suppress_spot_check(current, draft_index, trigger)
    # dim_grade_history holds PRIOR drafts only (the current one hasn't been
    # appended yet at this point in the pipeline). Count the trailing perfect
    # streak of prior drafts, then add 1 for the current draft iff it is
    # also perfect (every current dim is MET).
    prior_streak = _count_trailing_perfect_streak(dim_grade_history or {})
    current_perfect = True
    for c in (current.get("grades") or []):
        for d in c.get("dimension_grades") or []:
            if (d.get("grade") or "").upper() != "MET":
                current_perfect = False
                break
        if not current_perfect:
            break
    perfect_streak = prior_streak + (1 if current_perfect else 0)
    # Two gate models, switched by the RUBRIC_SPOT_CHECK_GATE env var:
    #
    #   "streak" (default, production heuristic):
    #     Fire only after a clean 3-draft streak. Catches silent
    #     misalignment when the user is happily producing perfect drafts.
    #     Conservative; designed to not bother the user.
    #
    #   "scheduled" (study mode, paper §3.2):
    #     Fire at fixed checkpoints (mid-loop, end-loop) to guarantee
    #     uncontested-dim confirmation data every session, regardless of
    #     draft quality. Removes selection bias in the §4.2 confirmation
    #     rate metric.
    #
    # Both models still respect _should_suppress_spot_check (early drafts,
    # tiny rubrics).
    gate_model = (os.environ.get("RUBRIC_SPOT_CHECK_GATE", "scheduled") or "scheduled").strip().lower()
    if gate_model not in ("streak", "scheduled"):
        gate_model = "scheduled"

    # Streak gate: prior production behavior.
    streak_gate_passes = (
        current_perfect
        and perfect_streak >= 3
        and perfect_streak % 3 == 0
    )
    # Scheduled gate: fire at draft_index in {3, 5}. These are mid-loop and
    # end-loop checkpoints for the paper's planned 4-6 draft sessions. If a
    # session ends before draft 5, the end-loop probe is missing data --
    # report honestly via the audit script's gate_model partition.
    scheduled_gate_passes = (
        draft_index is not None and int(draft_index) in (3, 5)
    )

    if gate_model == "scheduled":
        gate_passes = scheduled_gate_passes
    else:
        gate_passes = streak_gate_passes

    _hist_len = min([len(h) for h in (dim_grade_history or {}).values()], default=0)
    _log.info(
        "[spot_check diag] gate_model=%s draft_index=%s trigger=%s kind=%s "
        "sc_suppressed=%s prior_streak=%s current_perfect=%s total_streak=%s "
        "streak_gate=%s scheduled_gate=%s gate_passes=%s "
        "dim_history_len=%s hist_keys=%d",
        gate_model, draft_index, trigger, kind, spot_check_suppressed,
        prior_streak, current_perfect, perfect_streak,
        streak_gate_passes, scheduled_gate_passes, gate_passes,
        _hist_len, len(dim_grade_history or {}),
    )
    if kind == "none" and not spot_check_suppressed and gate_passes:
        # Collect all dims that have been in any drift signal
        surfaced = set(dims_already_surfaced or set())
        for lc in low_conf:
            surfaced.add(lc.get("dimension_id", ""))
        for o in oscillations:
            surfaced.add(o.get("dimension_id", ""))
        for p in persistent:
            surfaced.add(p.get("dimension_id", ""))
        spot_check_dims = _sample_spot_check_dims(current, surfaced)
        _log.info(
            "[spot_check sampler] gate_model=%s returned %d dims (surfaced=%d, available_crits=%d)",
            gate_model, len(spot_check_dims), len(surfaced), len(current.get("grades") or []),
        )
        if spot_check_dims:
            kind = "spot_check"

    # --- P0.2b: structured per-draft heuristic diagnostic ---
    # One structured record per heuristic per draft. Lets post-hoc analysis
    # answer "how often did each heuristic's condition fire, and when it
    # fired, was it shown or suppressed by priority?" without guessing from
    # ad-hoc log lines. Attached to the returned drift bundle so
    # schedule_background_grade can persist it alongside drift_json.
    def _heuristic_record(name: str, condition_met: bool, shown: bool,
                          suppressed_reason: str | None) -> dict[str, Any]:
        return {
            "heuristic": name,
            "condition_met": condition_met,
            "shown": shown,
            "suppressed_reason": suppressed_reason,
        }

    # Priority-based "shown" determination: only the highest-priority
    # firing heuristic appears in the panel; the rest are suppressed by it.
    def _suppressed_by(higher: str | None) -> str | None:
        if higher is None:
            return "condition_not_met" if not True else None
        return f"priority:{higher}"

    heuristic_diagnostics = [
        _heuristic_record(
            "low_confidence",
            condition_met=bool(low_conf),
            shown=(kind == "low_confidence"),
            suppressed_reason=None if bool(low_conf) == (kind == "low_confidence")
                              else "condition_not_met" if not low_conf else f"priority:{kind}",
        ),
        _heuristic_record(
            "oscillation",
            condition_met=bool(oscillations),
            shown=(kind == "oscillation"),
            suppressed_reason=(
                None if not oscillations and kind != "oscillation"
                else None if oscillations and kind == "oscillation"
                else ("noise_suppression" if oscillations and suppressed
                      else f"priority:{kind}" if oscillations else "condition_not_met")
            ),
        ),
        _heuristic_record(
            "persistent_failure",
            condition_met=bool(persistent),
            shown=(kind == "persistent_failure"),
            suppressed_reason=(
                None if not persistent and kind != "persistent_failure"
                else None if persistent and kind == "persistent_failure"
                else ("noise_suppression" if persistent and suppressed
                      else f"priority:{kind}" if persistent else "condition_not_met")
            ),
        ),
        _heuristic_record(
            "tradeoff",
            condition_met=bool(tradeoff_improvements and tradeoff_drops),
            shown=(kind == "tradeoff"),
            suppressed_reason=(
                None if not (tradeoff_improvements and tradeoff_drops) and kind != "tradeoff"
                else None if tradeoff_improvements and tradeoff_drops and kind == "tradeoff"
                else ("noise_suppression" if tradeoff_improvements and tradeoff_drops and suppressed
                      else f"priority:{kind}" if tradeoff_improvements and tradeoff_drops
                      else "condition_not_met")
            ),
        ),
        _heuristic_record(
            "spot_check",
            condition_met=streak_gate_passes and not spot_check_suppressed,
            shown=(kind == "spot_check"),
            suppressed_reason=(
                None if kind == "spot_check"
                else "suppressed" if spot_check_suppressed
                else (f"{gate_model}_gate" if not gate_passes else
                      f"priority:{kind}" if kind != "none"
                      else "sampler_returned_empty")
            ),
        ),
    ]
    # Tag the spot_check diagnostic record with the gate_model in use, so
    # the audit script can partition data by gate cleanly. Other heuristics
    # are gate-model-agnostic, so we only tag spot_check.
    for _h in heuristic_diagnostics:
        if _h.get("heuristic") == "spot_check":
            _h["gate_model"] = gate_model

    # Single WARNING-level log line so the outcome is visible in the terminal
    # during live sessions without having to grep INFO output. Compact format.
    _log.warning(
        "[heuristic diag] draft_idx=%s trigger=%s shown_kind=%s :: %s",
        draft_index, trigger, kind,
        " | ".join(
            f"{h['heuristic']}={'*' if h['shown'] else '+' if h['condition_met'] else '.'}"
            + (f"({h['suppressed_reason']})" if h['suppressed_reason'] else "")
            for h in heuristic_diagnostics
        ),
    )

    return {
        "kind": kind,
        "low_confidence_dims": low_conf,
        "oscillations": oscillations,
        "persistent_failure": persistent,
        "tradeoff_improvements": tradeoff_improvements,
        "tradeoff_drops": tradeoff_drops,
        "spot_check_dims": spot_check_dims,
        "suppressed": suppressed,
        "heuristic_diagnostics": heuristic_diagnostics,
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
    _dup_mids: dict[str, int] = {}
    for r in rows:
        mid = r.get("message_id")
        if not mid:
            continue
        key = str(mid)
        if key in by_mid:
            # Duplicate row for same message_id. Keep the earliest (first
            # in the draft_index-ascending sort) since past drafts should
            # be historical records. Track for diagnostics -- if we see
            # these in the logs, something is re-grading past drafts.
            _dup_mids[key] = _dup_mids.get(key, 1) + 1
            continue
        by_mid[key] = r
    if _dup_mids:
        # Log once per process per unique-duplicate-set to avoid spamming the
        # terminal on every rerun of the same conversation.
        _key = tuple(sorted(_dup_mids.items()))
        if _key not in _DUP_WARNED:
            _DUP_WARNED.add(_key)
            _log.warning(
                "merge_draft_grades_into_messages: found duplicate draft_grades "
                "rows for message_id(s): %s. Keeping the earliest row for each. "
                "If you see past drafts' scores changing, this is likely the cause. "
                "(This warning is suppressed for the rest of the session.)",
                {k: v for k, v in _dup_mids.items()},
            )

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
                        # Drift panels represent actionable signals about the
                        # CURRENT state of the rubric. Past drafts' panels are
                        # a frozen record of what WAS true at grading time --
                        # which is confusing when the dim has since stabilized
                        # (e.g. oscillation panel lingers on draft #3 even
                        # though drafts #4, #5, #6 are all MET). The colored
                        # dots on each past draft already capture the
                        # historical record; the panel itself only belongs on
                        # the latest draft where the user can actually act on
                        # it. Strip the panel from non-latest drafts for every
                        # drift kind. This was previously only done for
                        # spot_check; extended to all kinds for the same
                        # reason.
                        is_latest = (mid == latest_draft_mid)
                        if not is_latest and drift.get("kind") not in (None, "none"):
                            drift = {"kind": "none"}
                    else:
                        # Legacy fallback: row has no persisted drift (either a
                        # legacy row from before Option B, or the thread's
                        # drift computation failed).
                        is_latest = (mid == latest_draft_mid)
                        has_existing_drift = bool(m.get("draft_drift"))
                        existing_drift = m.get("draft_drift") or {}

                        if has_existing_drift:
                            # ALWAYS preserve existing in-memory drift if we
                            # have it. This covers three cases:
                            # (1) spot_check: don't re-randomize on rerun.
                            # (2) past draft: frozen historical record.
                            # (3) latest draft with active drift: DON'T recompute
                            # just because the user clicked a drift-panel
                            # button -- that shouldn't dissolve the panel.
                            # The panel naturally clears when the user
                            # fully resolves all its dims.
                            drift = existing_drift
                        elif not is_latest:
                            # Past draft with no in-memory drift -- freeze at
                            # "none" instead of recomputing retroactively.
                            drift = {"kind": "none"}
                        else:
                            # Latest draft, no persisted drift, no in-memory
                            # drift -- this is the first render after grading
                            # completes for this draft. Compute drift fresh.
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
    current_grades: dict[str, Any],
    current_draft_index: int,
    trigger: str | None,
    prior_payloads: list[dict[str, Any]] | None = None,
    prior_drifts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Compute a drift bundle for a just-graded draft using prior-draft data
    captured from st.session_state.messages at schedule time.

    prior_payloads: list of grade payloads from earlier graded drafts in this
                    conversation, earliest to latest.
    prior_drifts:   list of drift bundles from earlier graded drafts, same
                    order, used to populate dims_already_surfaced so
                    spot_check doesn't re-ask about dims that were recently
                    surfaced in other drift panels.

    We don't hit the DB here because (a) the background thread's auth
    context can't be trusted to read draft_grades under RLS, and (b)
    session-state already has everything we need since the UI maintains it.
    Returns a drift dict or None on error."""
    try:
        cfg = load_grading_config()
        _log.info(
            "[drift persist] entered for draft_idx=%s. show_drift_panels=%s enabled=%s "
            "prior_payloads=%d prior_drifts=%d",
            current_draft_index, cfg.get("show_drift_panels"), cfg.get("enabled"),
            len(prior_payloads or []), len(prior_drifts or []),
        )
        if not cfg.get("show_drift_panels"):
            _log.info("[drift persist] skipping: show_drift_panels is False")
            return None

        prior_payloads = prior_payloads or []
        prior_drifts = prior_drifts or []

        # prev_payload is the most recent prior draft's grade (for tradeoff
        # and persistent-failure compute).
        prev_payload = prior_payloads[-1] if prior_payloads else None

        # Rebuild dim_grade_history across all prior payloads (earliest to
        # latest) so streak / oscillation / persistent-failure detection
        # sees full history.
        dim_grade_history: dict[str, list[str]] = {}
        for gj in prior_payloads:
            if not isinstance(gj, dict):
                continue
            for c in gj.get("grades") or []:
                for d in c.get("dimension_grades") or []:
                    did = (d.get("dimension_id") or "").strip()
                    if did:
                        dim_grade_history.setdefault(did, []).append(
                            (d.get("grade") or "").upper()
                        )

        # Build dims_already_surfaced from prior drift bundles so spot_check
        # doesn't re-probe the same dims twice.
        dims_already_surfaced: set[str] = set()
        for drift in prior_drifts:
            if not isinstance(drift, dict):
                continue
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

    # Capture auth tokens NOW (while we're on the main thread and have access
    # to st.session_state). The grading thread will re-apply these on the
    # supabase client right before its DB writes so RLS policies succeed.
    # Without this, the client's auth drifts during the ~20s Sonnet call
    # and inserts silently fail the with_check RLS clause despite the client
    # returning success at the API layer.
    _auth_tokens: tuple[str, str] | None = None
    try:
        _sess = st.session_state.get("auth_session") if hasattr(st, "session_state") else None
        if _sess and getattr(_sess, "access_token", None) and getattr(_sess, "refresh_token", None):
            _auth_tokens = (_sess.access_token, _sess.refresh_token)
    except Exception:
        _auth_tokens = None

    # Capture project_id so the grading thread can persist per-draft
    # heuristic diagnostics into project_data. Thread can't access
    # st.session_state directly.
    _project_id_hint: str | None = None
    try:
        _project_id_hint = st.session_state.get("current_project_id") if hasattr(st, "session_state") else None
    except Exception:
        _project_id_hint = None

    # Compute the draft index from session-state messages (authoritative for
    # this conversation). Aligned with `compute_draft_number` semantics:
    # only count prior assistant drafts that have a `draft_grade` attached,
    # then add 1 for the current draft we're about to grade. This keeps
    # the schedule-gate's "draft #3" aligned with the UI's "Draft 3" label
    # (which also only counts graded drafts) -- so spot_check fires on
    # the same draft the user sees as draft #3, not on the third raw draft
    # which might be the user's "draft #2" if a prior draft never got
    # graded.
    #
    # Also capture prior graded drafts' grade payloads + drift bundles so
    # the background thread's drift compute doesn't need to query the DB.
    _d_idx_hint: int = 1
    _prior_payloads: list[dict[str, Any]] = []
    _prior_drifts: list[dict[str, Any]] = []
    try:
        msgs = st.session_state.get("messages", []) or []
        graded_count = 0
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            if m.get("_pre_rubric"):
                continue
            if not extract_primary_draft_text(m.get("content") or ""):
                continue
            this_is_current = (str(m.get("message_id") or "") == str(message_id))
            if this_is_current:
                # This message is the one we're about to grade. It will become
                # the (graded_count + 1)-th graded draft once grading completes.
                _d_idx_hint = graded_count + 1
                break
            # Prior draft. Only include in the count + capture its grade
            # data if it has actually been graded already. Skip ungraded
            # prior drafts -- they won't appear as numbered drafts in the
            # UI yet either.
            dg = m.get("draft_grade")
            if isinstance(dg, dict):
                graded_count += 1
                _prior_payloads.append(dg)
                dd = m.get("draft_drift")
                if isinstance(dd, dict):
                    _prior_drifts.append(dd)
        else:
            # Current mid not found in messages (e.g. edit-render race).
            _d_idx_hint = max(graded_count + 1, 1)
    except Exception:
        _d_idx_hint = 1
        _prior_payloads = []
        _prior_drifts = []

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
            _log.info(
                "[grade thread] started for mid=%s trigger=%s conv_id=%s",
                message_id, trigger, conversation_id,
            )
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
            _log.info(
                "[grade thread] grade_draft_sync returned for mid=%s: err=%s grades_ok=%s latency=%sms",
                message_id, err, bool(grades), latency_ms,
            )
            if err or not grades:
                return
            # Serialize the "pick next draft_index + compute drift + insert
            # row" sequence. Two concurrent grading threads used to both
            # read draft_index=1 from the DB before either inserted, leading
            # to collisions on draft_index AND duplicate rows. Holding the
            # lock across the whole sequence ensures each thread sees a
            # consistent view of the current draft count.
            # Re-apply auth tokens on the shared supabase client BEFORE any
            # DB reads or writes the thread does. The client's token may
            # have drifted during Sonnet's ~20s call, and both the drift
            # compute (which fetches prior rows) and the insert (which
            # checks with_check) need auth.uid() to resolve to the project
            # owner for RLS to allow them.
            #
            # CRITICAL: `auth.set_session` updates the auth module's state,
            # but the PostgREST client (used by `.table(...).insert(...)`)
            # has its own Authorization header set at construction time from
            # the anon key. Without calling `postgrest.auth(access_token)`,
            # table inserts still send the anon key and RLS's auth.uid()
            # resolves to NULL -- silent with_check rejection, no row
            # written, no error raised at the API layer. This was the
            # root cause of Session 1's empty draft_grades table despite
            # all 12 drafts being graded successfully.
            if _auth_tokens:
                try:
                    supabase.auth.set_session(_auth_tokens[0], _auth_tokens[1])
                    # Propagate the access token to the PostgREST client so
                    # REST calls carry it in Authorization: Bearer <jwt>.
                    try:
                        supabase.postgrest.auth(_auth_tokens[0])
                    except Exception as _e2:
                        _log.warning("[grade thread] postgrest.auth failed: %s", _e2)
                except Exception as _e:
                    _log.warning("[grade thread] set_session failed: %s", _e)

            with _index_insert_lock:
                # Draft index came from session-state (authoritative count of
                # drafted assistant messages up to and including this one).
                # No DB round-trip needed, which would be blocked by RLS in
                # the background thread anyway.
                d_idx = _d_idx_hint
                _log.info(
                    "[index] conv=%s from_session=%s → assigning draft_idx=%s",
                    conversation_id, _d_idx_hint, d_idx,
                )

                # Option B: compute drift ONCE at grading time and persist it
                # to the DB so the UI never has to recompute. Inputs are
                # reconstructed from prior rows in this conversation's
                # draft_grades table.
                _log.info(
                    "[drift compute] calling _compute_drift_for_persist for mid=%s draft_idx=%s rv=%s",
                    message_id, d_idx, _rv,
                )
                drift_json = _compute_drift_for_persist(
                    current_grades=grades,
                    current_draft_index=d_idx,
                    trigger=trigger,
                    prior_payloads=_prior_payloads,
                    prior_drifts=_prior_drifts,
                )
                _log.info(
                    "[drift compute] returned for mid=%s: %s",
                    message_id,
                    "None" if drift_json is None else f"kind={drift_json.get('kind')!r}",
                )

                insert_ok = insert_draft_grade(
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

                # P0.2b: persist the per-draft heuristic diagnostic into
                # project_data so we can answer "across the session, how
                # often did each heuristic fire vs. get suppressed" without
                # grepping logs. Piggybacks on the drift_json bundle that
                # _compute_drift_for_persist already produced.
                try:
                    diags = (drift_json or {}).get("heuristic_diagnostics") or []
                    if diags and _project_id_hint:
                        from auth_supabase import save_project_data
                        save_project_data(
                            supabase,
                            _project_id_hint,
                            "heuristic_diagnostic",
                            {
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "conversation_id": conversation_id,
                                "message_id": message_id,
                                "draft_index": d_idx,
                                "shown_kind": (drift_json or {}).get("kind", "none"),
                                "heuristics": diags,
                            },
                        )
                except Exception as _e_diag:
                    _log.warning("[heuristic diag persist] failed: %s", _e_diag)
                # Mark this mid as freshly graded so the maybe_schedule_pending_grades
                # fallback doesn't re-schedule during the DB-propagation window.
                # (No need to bump any index counter -- draft_idx is computed
                # from st.session_state.messages at schedule time.)
                _ = insert_ok  # keep the symbol for clarity; value unused here
                with _inflight_lock:
                    _recently_graded_mids.add(message_id)
                    # Flag this mid so the main thread knows to rollup the
                    # fresh grade into conversations.messages JSON, preserving
                    # the grade + drift in the conversation record itself.
                    _pending_conversation_save_mids.add(message_id)
        except Exception as e:
            _log.warning("background grade thread: %s", e)
        finally:
            with _inflight_lock:
                _inflight_message_ids.discard(message_id)

    threading.Thread(target=_run, daemon=True).start()


def flush_pending_conversation_save() -> None:
    """If any mid was graded by a background thread since the last render,
    and its grade has already been attached to the message in session state
    (via merge_draft_grades_into_messages), roll it up into the persisted
    conversation record so the grade survives page reloads / login cycles.

    Called from the main chat render loop so it runs on every rerun.
    Silently no-ops when there's nothing to save."""
    global _pending_conversation_save_mids
    with _inflight_lock:
        pending = set(_pending_conversation_save_mids)
    if not pending:
        return
    msgs = st.session_state.get("messages", []) or []
    # Check whether any pending mid is now represented with a draft_grade
    # in session state. If yes, we have something worth saving.
    mids_ready = {
        str(m.get("message_id") or "") for m in msgs
        if m.get("role") == "assistant" and m.get("draft_grade")
    }
    ready_to_save = pending & mids_ready
    if not ready_to_save:
        return
    # Save (writes current st.session_state.messages as-is to the DB;
    # message dicts now carry draft_grade / draft_drift / draft_grade_meta
    # so the conversations.messages JSON preserves all of it).
    try:
        from rubric_writer.persistence import _auto_save_conversation
        _auto_save_conversation()
    except Exception as e:
        _log.warning("flush_pending_conversation_save: save failed: %s", e)
        return
    # Clear the pending set for everything we just saved.
    with _inflight_lock:
        _pending_conversation_save_mids -= ready_to_save


def maybe_schedule_pending_grades(max_jobs: int = 2) -> None:
    """Best-effort: grade recent assistant drafts that have no grade yet.

    Uses session-state `draft_grade` attribute as the source of truth
    (attached by merge_draft_grades_into_messages when the DB has a row).
    No DB fetch here -- RLS in the background thread can't be trusted, so
    we rely on what the UI's main thread has already loaded."""
    cfg = load_grading_config()
    if not cfg["enabled"]:
        return
    supabase = st.session_state.get("supabase")
    conv_id = st.session_state.get("selected_conversation")
    if not supabase or not conv_id:
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
        if not mid:
            continue
        # Skip if the message already has a grade attached in session state.
        # merge_draft_grades_into_messages sets `draft_grade` on any message
        # that has a row in the DB. If grade is attached, the draft has
        # already been graded -- no need to re-schedule.
        if m.get("draft_grade"):
            continue
        # Also skip if this message_id is currently being graded OR was just
        # graded by another thread. Process-local guards to prevent the same
        # mid from being scheduled twice within this Streamlit session.
        with _inflight_lock:
            if mid in _inflight_message_ids:
                continue
            if mid in _recently_graded_mids:
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
