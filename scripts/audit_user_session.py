#!/usr/bin/env python3
"""
Audit a user's session data for paper completeness.

Usage:
    python scripts/audit_user_session.py --email jian.chen@example.com
    python scripts/audit_user_session.py --user-id <uuid>
    python scripts/audit_user_session.py --search-name "Jian"
    python scripts/audit_user_session.py --email jian.chen@example.com --json-out /tmp/session.json

Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from .env (service-role
bypasses RLS so we can read any user's data without their password).

Reports, per project and per conversation:
  - conversations row: messages with draft_grade/draft_drift/user_verdicts
    embedded (the per-draft grader output lives here)
  - rubric_history: all rubric versions with timestamps + dim counts
  - project_data rollups for:
      * drift-panel feedback (oscillation_feedback, low_confidence_feedback,
        persistent_failure_feedback, spot_check_feedback)
      * tradeoff_preference
      * rubric_edit_applied (legacy) + rubric_edit_verification
      * rubric_edit_event (NEW: trigger-attributed version transitions)
      * refiner_proposal (NEW: proposed→applied/dismissed lifecycle)
      * heuristic_diagnostic (NEW: per-draft condition/suppressed counts)
      * rq2_threeway (blind three-way comparison)
      * surveys (task_a, task_b, final_review, responses)

Produces an end-of-report paper-readiness checklist. Optional --json-out
dumps the collected data for downstream analysis. Read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from typing import Any

# Load .env ------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from supabase import create_client, Client
except ImportError:
    sys.stderr.write("pip install supabase python-dotenv\n")
    sys.exit(1)


def _client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        sys.stderr.write(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env\n"
        )
        sys.exit(1)
    return create_client(url, key)


def _find_user(sb: Client, *, email: str | None, user_id: str | None,
               search_name: str | None) -> list[dict]:
    """Find candidate users. Uses auth.users admin API via service role."""
    if user_id:
        try:
            resp = sb.auth.admin.get_user_by_id(user_id)
            u = resp.user if hasattr(resp, "user") else resp
            if u:
                return [{"id": u.id, "email": u.email}]
        except Exception as e:
            print(f"[warn] get_user_by_id failed: {e}")
            return []

    all_users: list[dict] = []
    page = 1
    while True:
        try:
            resp = sb.auth.admin.list_users(page=page, per_page=200)
            users = resp if isinstance(resp, list) else getattr(resp, "users", [])
            if not users:
                break
            for u in users:
                all_users.append({"id": u.id, "email": u.email or ""})
            if len(users) < 200:
                break
            page += 1
        except Exception as e:
            print(f"[warn] list_users failed: {e}")
            break

    if email:
        return [u for u in all_users if (u["email"] or "").lower() == email.lower()]
    if search_name:
        tok = search_name.lower()
        return [u for u in all_users if tok in (u["email"] or "").lower()]
    return all_users


def _header(s: str, char: str = "=") -> None:
    print("\n" + char * 78)
    print(s)
    print(char * 78)


def _fmt_ts(ts: Any) -> str:
    return (str(ts) or "")[:19].replace("T", " ")


def _parse_json_maybe(val: Any) -> Any:
    """Coerce a possibly-JSON-string value to a Python object. Returns the
    value unchanged if it's already parsed or unparseable."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


# ────────────────────────────────────────────────────────────────────────────
# Per-conversation audit
# ────────────────────────────────────────────────────────────────────────────

def _extract_per_draft_data(messages: list[dict]) -> list[dict]:
    """Pull structured per-draft records out of the conversation JSON.

    Falls back to the embedded `draft_grade` / `draft_drift` / `user_verdicts`
    fields on each assistant message, so we have a queryable record even when
    the `draft_grades` table is missing rows."""
    out: list[dict] = []
    draft_idx = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if "<draft>" not in content:
            continue
        draft_idx += 1
        dg = m.get("draft_grade") or {}
        dd = m.get("draft_drift") or {}
        uv = m.get("user_verdicts") or {}
        # Flatten per-criterion scores.
        per_crit = []
        for c in dg.get("grades") or []:
            cname = c.get("criterion_name") or ""
            score = c.get("score") or ""
            dims = c.get("dimension_grades") or []
            per_crit.append({
                "criterion": cname,
                "score": score,
                "n_dims": len(dims),
                "n_met": sum(1 for d in dims if (d.get("grade") or "").upper() == "MET"),
                "n_low_conf": sum(1 for d in dims if (d.get("confidence") or "").lower() == "low"),
                "n_medium_conf": sum(1 for d in dims if (d.get("confidence") or "").lower() == "medium"),
            })
        total_dims = sum(pc["n_dims"] for pc in per_crit)
        total_met = sum(pc["n_met"] for pc in per_crit)
        out.append({
            "draft_number": draft_idx,
            "message_id": m.get("message_id"),
            "rubric_version": (m.get("draft_grade_meta") or {}).get("rubric_version"),
            "drift_kind": dd.get("kind") or "none",
            "total_dims": total_dims,
            "total_met": total_met,
            "pct_met": round(100.0 * total_met / total_dims, 1) if total_dims else None,
            "per_criterion": per_crit,
            "n_user_verdicts": len(uv),
            "user_verdicts": list(uv.values()),
        })
    return out


def audit_conversation(sb: Client, conv: dict) -> dict:
    conv_id = conv["id"]
    print(f"\n--- Conversation {conv_id} (created {_fmt_ts(conv.get('created_at'))}) ---")

    messages = _parse_json_maybe(conv.get("messages")) or []
    if not isinstance(messages, list):
        messages = []

    role_counts = Counter(m.get("role", "?") for m in messages)
    print(f"  messages: {len(messages)} total -- {dict(role_counts)}")

    per_draft = _extract_per_draft_data(messages)
    print(f"  drafts: {len(per_draft)}")
    if per_draft:
        drift_kinds = Counter(d["drift_kind"] for d in per_draft)
        met_pcts = [d["pct_met"] for d in per_draft if d["pct_met"] is not None]
        if met_pcts:
            print(f"    pct_met per draft: {met_pcts}")
        print(f"    drift panels by kind: {dict(drift_kinds)}")

    user_verdict_total = sum(d["n_user_verdicts"] for d in per_draft)
    print(f"    user_verdicts across drafts: {user_verdict_total}")

    return {
        "conv_id": conv_id,
        "created_at": conv.get("created_at"),
        "messages_total": len(messages),
        "role_counts": dict(role_counts),
        "drafts": per_draft,
    }


# ────────────────────────────────────────────────────────────────────────────
# project_data rollups
# ────────────────────────────────────────────────────────────────────────────

# data_type keys we care about. Grouped by what they measure.
_FEEDBACK_TYPES = [
    # Drift-panel button clicks
    "low_confidence_feedback",
    "oscillation_feedback",
    "persistent_failure_feedback",
    "spot_check_feedback",
    # Tradeoff
    "tradeoff_preference",
    # Rubric edits (legacy per-event and new structured)
    "rubric_edit_applied",
    "rubric_edit_verification",
    "rubric_edit_event",
    "rubric_dimension_removed",
    # Refiner proposal lifecycle (NEW)
    "refiner_proposal",
    # Heuristic diagnostic (NEW)
    "heuristic_diagnostic",
    # Research question logs
    "rq1_dimension_recognition",
    "rq2_threeway",
    "rq2_confirmation",
    "rq2_fire_rate",
    # Surveys
    "survey_task_a",
    "survey_task_b",
    "survey_final_review",
    "survey_responses",
    # Misc ops / config
    "rubric_classification_cleanup",
    "infer_conversation",
    "inline_rephrase",
    "log_changes",
]


def _flatten_data_items(rows: list[dict]) -> list:
    """project_data rows have a `data` column that's a JSON array (all items
    of that data_type concatenated). Flatten across rows."""
    items: list = []
    for r in rows:
        arr = _parse_json_maybe(r.get("data"))
        if isinstance(arr, list):
            items.extend(arr)
        elif arr is not None:
            items.append(arr)
    return items


def audit_project_data(sb: Client, project_id: str) -> tuple[dict, dict]:
    """Scan project_data by data_type.

    Returns:
      stats_count  : {data_type: int}     -- item count per type
      raw_data     : {data_type: [items]} -- full items for JSON export
    """
    _header("project_data rollup", "-")
    resp = sb.table("project_data").select("*").eq("project_id", project_id).execute()
    rows = resp.data or []
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r.get("data_type", "?")].append(r)

    stats_count: dict[str, int] = {}
    raw_data: dict[str, list] = {}

    # Print known types first (in order), then any unexpected types.
    known = set(_FEEDBACK_TYPES)
    all_types = list(_FEEDBACK_TYPES) + sorted(set(by_type.keys()) - known)
    for dt in all_types:
        entries = by_type.get(dt, [])
        items = _flatten_data_items(entries)
        stats_count[dt] = len(items)
        raw_data[dt] = items
        marker = "  "
        if len(items) == 0 and dt in known:
            marker = "  "
        print(f"{marker}{dt}: {len(items)} items across {len(entries)} row(s)")

    return stats_count, raw_data


# ────────────────────────────────────────────────────────────────────────────
# Derived rollups
# ────────────────────────────────────────────────────────────────────────────

def refiner_proposal_rollup(items: list[dict]) -> dict:
    """Group refiner_proposal items by edit_id and compute lifecycle stats."""
    by_edit: dict[str, list[dict]] = defaultdict(list)
    no_edit_id: list[dict] = []
    for it in items:
        eid = it.get("edit_id")
        if eid:
            by_edit[eid].append(it)
        else:
            no_edit_id.append(it)

    # Terminal disposition per edit_id (prefer the last non-"proposed" row).
    disposition_counts: Counter = Counter()
    per_drift_kind: Counter = Counter()
    for eid, rows in by_edit.items():
        # Sort by timestamp, take the latest non-proposed if any, else "proposed".
        rows_sorted = sorted(rows, key=lambda r: r.get("timestamp") or "")
        terminal = "proposed"
        for r in reversed(rows_sorted):
            d = r.get("disposition")
            if d and d != "proposed":
                terminal = d
                break
        disposition_counts[terminal] += 1
        # Take drift_kind from any row (should be consistent)
        dk = rows_sorted[0].get("drift_kind") or "?"
        per_drift_kind[dk] += 1

    # Add the standalone "dropped_unresolved" items (no edit_id)
    for it in no_edit_id:
        d = it.get("disposition") or "unknown"
        disposition_counts[d] += 1

    return {
        "n_unique_proposals": len(by_edit) + len(no_edit_id),
        "n_with_edit_id": len(by_edit),
        "disposition_counts": dict(disposition_counts),
        "per_drift_kind": dict(per_drift_kind),
    }


def rubric_edit_event_rollup(items: list[dict]) -> dict:
    """Group rubric_edit_event by trigger."""
    triggers = Counter(i.get("trigger") or "unknown" for i in items)
    n_added = sum(len((i.get("edit_summary") or {}).get("added_dims") or []) for i in items)
    n_removed = sum(len((i.get("edit_summary") or {}).get("removed_dims") or []) for i in items)
    n_modified = sum(len((i.get("edit_summary") or {}).get("modified_dims") or []) for i in items)
    return {
        "total_events": len(items),
        "by_trigger": dict(triggers),
        "cumulative_dims_added": n_added,
        "cumulative_dims_removed": n_removed,
        "cumulative_dims_modified": n_modified,
        "version_transitions": [
            {
                "from": i.get("from_version"),
                "to": i.get("to_version"),
                "trigger": i.get("trigger"),
                "timestamp": i.get("timestamp"),
            }
            for i in sorted(items, key=lambda x: x.get("timestamp") or "")
        ],
    }


# ────────────────────────────────────────────────────────────────────────────
# Reason codes (Option B: post-hoc mapping from button + drift kind)
# ────────────────────────────────────────────────────────────────────────────

# Maps (data_type, action_or_user_grade) → reason code. The reason code names
# the WHY behind the user's button click, derived from which drift panel
# fired and which option they chose. Implements §3.2 / §4.3 of the paper
# without requiring a UI change.
_REASON_CODE_MAP: dict[tuple[str, str], str] = {
    # Oscillation panel buttons.
    # Current UI: 3 buttons -- Subjective wording / No rubric change / Remove
    # (where "No rubric change" stages a radio with two reason sub-options:
    # rubric_fine and drafts_varying). Legacy `just_right` slug from sessions
    # 1-2 is preserved for back-compat.
    ("oscillation_feedback", "wording_subjective"): "rubric_ambiguous",
    ("oscillation_feedback", "drafts_varying"):     "not_a_rubric_problem",
    ("oscillation_feedback", "rubric_fine"):        "rubric_fine",
    ("oscillation_feedback", "just_right"):         "rubric_fine",  # legacy
    ("oscillation_feedback", "remove"):             "dim_not_wanted",
    # Persistent failure panel buttons (3 options + remove)
    ("persistent_failure_feedback", "MET"):           "grader_wrong",
    ("persistent_failure_feedback", "working_on_it"): "confirmed",
    ("persistent_failure_feedback", "remove"):        "dim_not_wanted",
    # Calibration buttons (used in some legacy paths)
    ("persistent_failure_feedback", "too_strict"):    "rubric_too_strict",
    ("persistent_failure_feedback", "too_vague"):     "rubric_ambiguous",
    ("persistent_failure_feedback", "just_right"):    "rubric_fine",
    # Low confidence panel
    ("low_confidence_feedback", "grade_correct"):              "confirmed_uncertain",
    ("low_confidence_feedback", "grade_flipped_to_MET"):       "grader_wrong",
    ("low_confidence_feedback", "grade_flipped_to_NOT_MET"):   "grader_wrong",
    # Spot check feedback
    ("spot_check_feedback", "MET"):     "confirmed_uncertain",
    ("spot_check_feedback", "NOT_MET"): "grader_wrong",
}


def reason_code_rollup(raw_data: dict[str, list]) -> dict:
    """Walk all drift-feedback project_data rows and map each user action
    to a reason code. Returns:
      total_actions, by_reason_code, by_drift_kind, by_drift_kind_x_reason
    so §4.3 can report the distribution.

    Logic: each item's `action` (oscillation/persistent_failure feedback) or
    `user_grade` (low_confidence/spot_check feedback) is the discriminator.
    """
    by_reason: Counter = Counter()
    by_drift_kind: Counter = Counter()
    by_kind_x_reason: dict[str, Counter] = defaultdict(Counter)
    unmapped: list[dict] = []
    total = 0
    feedback_keys = (
        "oscillation_feedback",
        "persistent_failure_feedback",
        "low_confidence_feedback",
        "spot_check_feedback",
    )
    drift_kind_for = {
        "oscillation_feedback": "oscillation",
        "persistent_failure_feedback": "persistent_failure",
        "low_confidence_feedback": "low_confidence",
        "spot_check_feedback": "spot_check",
    }
    for fk in feedback_keys:
        for item in raw_data.get(fk) or []:
            total += 1
            kind = drift_kind_for[fk]
            by_drift_kind[kind] += 1
            # If the row has an EXPLICIT reason_code (Option A — spot_check NO
            # radio collects this), prefer that. Falls back to the post-hoc
            # button-implied mapping (Option B) for older rows or rows where
            # the user didn't pick an option.
            explicit = (item.get("reason_code") or "").strip()
            if explicit:
                by_reason[explicit] += 1
                by_kind_x_reason[kind][explicit] += 1
                continue
            # The action discriminator differs by feedback type. For
            # spot_check, the row stores `user_grade` (MET/NOT_MET); for
            # low_confidence, `action` (grade_correct / grade_flipped_to_X);
            # for oscillation/persistent_failure, `action` (wording_subjective,
            # working_on_it, etc.). Try both fields.
            action = (item.get("action") or item.get("user_grade") or "").strip()
            key = (fk, action)
            reason = _REASON_CODE_MAP.get(key)
            if reason:
                by_reason[reason] += 1
                by_kind_x_reason[kind][reason] += 1
            else:
                unmapped.append({"data_type": fk, "action": action,
                                  "preview": str(item)[:120]})
    return {
        "total_disagreement_actions": total,
        "by_reason_code": dict(by_reason),
        "by_drift_kind": dict(by_drift_kind),
        "by_drift_kind_x_reason": {k: dict(v) for k, v in by_kind_x_reason.items()},
        "unmapped_actions": unmapped,
    }


def user_verdicts_rollup(conv_audits: list[dict]) -> dict:
    """Walk per-draft user_verdicts captured from conversation messages.

    user_verdicts is the actual on-the-wire record of drift-panel button
    clicks (the legacy *_feedback project_data rows are no longer written —
    the verdict ends up embedded on the assistant message that owns the
    panel). Each verdict has:
      - criterion / dimension_id  : what was being judged
      - user_grade                : the button label, e.g. MET, NOT_MET,
                                    working_on_it, remove, wording_subjective,
                                    drafts_varying, rubric_fine,
                                    grade_correct, grade_flipped_to_MET, ...

    We pair the verdict's user_grade with the draft's drift_kind so the same
    post-hoc reason-code mapping (`_REASON_CODE_MAP`) that the legacy
    feedback rows used can be applied here too.
    """
    by_action = Counter()
    by_drift_kind = Counter()
    pairs: list[tuple[str, str]] = []  # (drift_kind, user_grade)
    items_out: list[dict] = []
    for ca in conv_audits:
        for d in ca.get("drafts") or []:
            kind = d.get("drift_kind") or "none"
            for v in d.get("user_verdicts") or []:
                grade = (v.get("user_grade") or "").strip()
                if not grade:
                    continue
                by_action[grade] += 1
                by_drift_kind[kind] += 1
                pairs.append((kind, grade))
                items_out.append({
                    "conv_id": ca.get("conv_id"),
                    "draft_number": d.get("draft_number"),
                    "drift_kind": kind,
                    "criterion": v.get("criterion"),
                    "dimension_id": v.get("dimension_id"),
                    "user_grade": grade,
                })
    # Map each (drift_kind, action) to a reason code via the same table the
    # legacy *_feedback rollup uses. The mapping keys on data_type, so
    # synthesize one from drift_kind.
    drift_to_dt = {
        "oscillation": "oscillation_feedback",
        "persistent_failure": "persistent_failure_feedback",
        "low_confidence": "low_confidence_feedback",
        "spot_check": "spot_check_feedback",
    }
    by_reason: Counter = Counter()
    by_kind_x_reason: dict[str, Counter] = defaultdict(Counter)
    unmapped: list[dict] = []
    for kind, grade in pairs:
        dt = drift_to_dt.get(kind)
        if not dt:
            unmapped.append({"drift_kind": kind, "action": grade})
            continue
        reason = _REASON_CODE_MAP.get((dt, grade))
        if reason:
            by_reason[reason] += 1
            by_kind_x_reason[kind][reason] += 1
        else:
            unmapped.append({"drift_kind": kind, "action": grade})
    return {
        "total_verdicts": sum(by_action.values()),
        "by_action": dict(by_action),
        "by_drift_kind": dict(by_drift_kind),
        "by_reason_code": dict(by_reason),
        "by_drift_kind_x_reason": {k: dict(v) for k, v in by_kind_x_reason.items()},
        "unmapped": unmapped,
        "items": items_out,
    }


def heuristic_diagnostic_rollup(items: list[dict]) -> dict:
    """Aggregate heuristic_diagnostic entries across all drafts.

    Answers: for each heuristic, how many times was its condition met,
    how many times was it SHOWN (only one heuristic shows per draft due
    to priority), and what suppressed it when met-but-not-shown?

    For spot_check specifically, also partitions stats by gate_model
    ("streak" vs "scheduled"). Sessions run under different gate models
    are not directly comparable on the spot_check measure -- the audit
    script keeps them separate so the paper can report each denominator
    honestly. See §4.2 reporting in the spot-check scheduled-gate change."""
    per_heuristic: dict[str, dict] = defaultdict(lambda: {
        "condition_met": 0,
        "shown": 0,
        "suppressed_by_priority": 0,
        "suppressed_reasons": Counter(),
    })
    spot_check_by_gate: dict[str, dict] = defaultdict(lambda: {
        "condition_met": 0,
        "shown": 0,
        "n_records": 0,
    })
    draft_count = len(items)
    for rec in items:
        for h in rec.get("heuristics") or []:
            name = h.get("heuristic")
            if not name:
                continue
            bucket = per_heuristic[name]
            if h.get("condition_met"):
                bucket["condition_met"] += 1
            if h.get("shown"):
                bucket["shown"] += 1
            reason = h.get("suppressed_reason")
            if reason:
                bucket["suppressed_reasons"][reason] += 1
                if str(reason).startswith("priority:"):
                    bucket["suppressed_by_priority"] += 1
            # spot_check: also partition by gate_model.
            if name == "spot_check":
                gm = (h.get("gate_model") or "unspecified").lower()
                gb = spot_check_by_gate[gm]
                gb["n_records"] += 1
                if h.get("condition_met"):
                    gb["condition_met"] += 1
                if h.get("shown"):
                    gb["shown"] += 1
    # Convert Counter to dict for JSON serialization.
    for k in per_heuristic:
        per_heuristic[k]["suppressed_reasons"] = dict(per_heuristic[k]["suppressed_reasons"])
    return {
        "n_drafts_with_diagnostic": draft_count,
        "per_heuristic": dict(per_heuristic),
        "spot_check_by_gate": dict(spot_check_by_gate),
    }


# ────────────────────────────────────────────────────────────────────────────
# Rubric history
# ────────────────────────────────────────────────────────────────────────────

def audit_rubric_history(sb: Client, project_id: str) -> list[dict]:
    _header("rubric_history", "-")
    resp = sb.table("rubric_history").select("*").eq("project_id", project_id).order("version").execute()
    rows = resp.data or []
    print(f"  {len(rows)} version(s) total")
    versions: list[dict] = []
    for r in rows:
        rubric = _parse_json_maybe(r.get("rubric_data")) or {}
        crits = rubric.get("rubric", []) if isinstance(rubric, dict) else []
        n_dims = sum(len(c.get("dimensions") or []) for c in crits)
        source = (rubric.get("source") if isinstance(rubric, dict) else "") or ""
        print(f"    v{r.get('version')} ({_fmt_ts(r.get('created_at'))}) -- {len(crits)} criteria, {n_dims} dims  source={source}")
        versions.append({
            "version": r.get("version"),
            "created_at": r.get("created_at"),
            "source": source,
            "n_criteria": len(crits),
            "n_dims": n_dims,
            "criteria": [
                {
                    "name": c.get("name"),
                    "n_dims": len(c.get("dimensions") or []),
                    "dim_ids": [d.get("id") for d in c.get("dimensions") or []],
                }
                for c in crits
            ],
        })
    return versions


# ────────────────────────────────────────────────────────────────────────────
# Top-level audit
# ────────────────────────────────────────────────────────────────────────────

def audit_project(sb: Client, user_id: str, user_email: str) -> dict:
    _header(f"USER {user_email} ({user_id})")
    projects_resp = sb.table("projects").select("*").eq("user_id", user_id).order("created_at").execute()
    projects = projects_resp.data or []
    if not projects:
        print("  No projects.")
        return {"user_id": user_id, "user_email": user_email, "projects": []}

    out = {"user_id": user_id, "user_email": user_email, "projects": []}
    for p in projects:
        pid = p["id"]
        pname = p.get("name", "?")
        _header(f"PROJECT: {pname}  ({pid})")
        print(f"  created {_fmt_ts(p.get('created_at'))}")

        c_resp = sb.table("conversations").select("*").eq("project_id", pid).order("created_at").execute()
        convs = c_resp.data or []
        print(f"\n  conversations: {len(convs)}")
        conv_audits = [audit_conversation(sb, conv) for conv in convs]

        versions = audit_rubric_history(sb, pid)
        stats, raw_data = audit_project_data(sb, pid)

        # Derived rollups
        prop_rollup = refiner_proposal_rollup(raw_data.get("refiner_proposal") or [])
        edit_rollup = rubric_edit_event_rollup(raw_data.get("rubric_edit_event") or [])
        heur_rollup = heuristic_diagnostic_rollup(raw_data.get("heuristic_diagnostic") or [])
        verdict_rollup = user_verdicts_rollup(conv_audits)
        reason_rollup = reason_code_rollup(raw_data)

        _header(f"DERIVED ROLLUPS for {pname}", "-")
        print(f"  refiner_proposal lifecycle:")
        print(f"    unique proposals: {prop_rollup['n_unique_proposals']}")
        print(f"    disposition counts: {prop_rollup['disposition_counts']}")
        print(f"    by drift_kind: {prop_rollup['per_drift_kind']}")
        print(f"  rubric_edit_event:")
        print(f"    total events: {edit_rollup['total_events']}")
        print(f"    by trigger: {edit_rollup['by_trigger']}")
        print(f"    cumulative: +{edit_rollup['cumulative_dims_added']} -{edit_rollup['cumulative_dims_removed']} "
              f"~{edit_rollup['cumulative_dims_modified']} dims")
        print(f"  heuristic_diagnostic:")
        print(f"    drafts with diagnostic: {heur_rollup['n_drafts_with_diagnostic']}")
        for h_name, h_stats in heur_rollup["per_heuristic"].items():
            print(f"    {h_name}: condition_met={h_stats['condition_met']}  shown={h_stats['shown']}  "
                  f"suppressed_by_priority={h_stats['suppressed_by_priority']}")
        # spot_check partition by gate_model -- different gate models produce
        # incomparable measurement populations (streak gate samples from
        # uninterrupted clean streaks; scheduled gate samples at fixed
        # checkpoints). Reported separately for §4.2.
        sc_by_gate = heur_rollup.get("spot_check_by_gate") or {}
        if sc_by_gate:
            print(f"    spot_check by gate_model:")
            for gm, gs in sorted(sc_by_gate.items()):
                print(f"      {gm}: n_records={gs['n_records']}  "
                      f"condition_met={gs['condition_met']}  shown={gs['shown']}")
        # user_verdicts (drift-panel button clicks live on assistant messages
        # in the conversation JSON, not in *_feedback project_data rows)
        print(f"  user_verdicts (drift-panel clicks from conversation messages):")
        print(f"    total: {verdict_rollup['total_verdicts']}")
        if verdict_rollup["total_verdicts"]:
            print(f"    by action: {verdict_rollup['by_action']}")
            print(f"    by drift_kind: {verdict_rollup['by_drift_kind']}")
            if verdict_rollup["by_reason_code"]:
                print(f"    reason codes: {verdict_rollup['by_reason_code']}")
            if verdict_rollup["unmapped"]:
                print(f"    unmapped: {verdict_rollup['unmapped']}")
        # Reason-code distribution (legacy: post-hoc mapping from *_feedback
        # project_data rows; kept for back-compat with older sessions)
        print(f"  reason codes (legacy *_feedback rows):")
        print(f"    total disagreement actions: {reason_rollup['total_disagreement_actions']}")
        if reason_rollup["by_reason_code"]:
            print(f"    by reason: {reason_rollup['by_reason_code']}")
            print(f"    by drift_kind: {reason_rollup['by_drift_kind']}")
        if reason_rollup.get("unmapped_actions"):
            print(f"    unmapped actions: {len(reason_rollup['unmapped_actions'])}")
            for u in reason_rollup["unmapped_actions"][:3]:
                print(f"      - {u}")

        # Paper-readiness checklist
        _header(f"PAPER-READINESS CHECKLIST for {pname}", "=")
        n_drafts_in_convs = sum(len(c["drafts"]) for c in conv_audits)
        checks = [
            ("Rubric versions saved", len(versions)),
            ("Conversations", len(convs)),
            ("Drafts (in conversation JSON)", n_drafts_in_convs),
            ("heuristic_diagnostic rows", stats.get("heuristic_diagnostic", 0)),
            ("refiner_proposal events", stats.get("refiner_proposal", 0)),
            ("  unique proposals (edit_id)", prop_rollup["n_unique_proposals"]),
            ("rubric_edit_event rows", stats.get("rubric_edit_event", 0)),
            ("drift-panel verdicts (from messages)", verdict_rollup["total_verdicts"]),
            ("tradeoff_preference", stats.get("tradeoff_preference", 0)),
            ("rq2_threeway (blind comparison)", stats.get("rq2_threeway", 0)),
            ("rq1_dimension_recognition", stats.get("rq1_dimension_recognition", 0)),
            ("survey_task_a", stats.get("survey_task_a", 0)),
            ("survey_task_b", stats.get("survey_task_b", 0)),
            ("survey_final_review", stats.get("survey_final_review", 0)),
            ("survey_responses", stats.get("survey_responses", 0)),
        ]
        for name, val in checks:
            mark = "✓"
            if isinstance(val, int):
                mark = "✓" if val > 0 else "✗"
            elif isinstance(val, str) and "MISSING" in val:
                mark = "✗"
            print(f"  {mark} {name}: {val}")

        out["projects"].append({
            "project_id": pid,
            "project_name": pname,
            "created_at": p.get("created_at"),
            "conversations": conv_audits,
            "rubric_versions": versions,
            "project_data_counts": stats,
            "project_data_raw": raw_data,
            "derived": {
                "refiner_proposal": prop_rollup,
                "rubric_edit_event": edit_rollup,
                "heuristic_diagnostic": heur_rollup,
                "user_verdicts": verdict_rollup,
                "reason_codes": reason_rollup,
            },
        })

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--email", help="exact user email")
    g.add_argument("--user-id", help="user UUID")
    g.add_argument("--search-name", help="substring match on email")
    p.add_argument("--json-out", help="dump the full collected data as JSON to this path")
    args = p.parse_args()

    sb = _client()
    users = _find_user(sb, email=args.email, user_id=args.user_id, search_name=args.search_name)
    if not users:
        print("No matching users.")
        return 1
    if len(users) > 1:
        print("Multiple matches:")
        for u in users:
            print(f"  {u['id']}  {u['email']}")
        print("\nRe-run with --user-id or a more specific --email.")
        return 2
    u = users[0]
    result = audit_project(sb, u["id"], u["email"])

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[full result written to {args.json_out}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
