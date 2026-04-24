#!/usr/bin/env python3
"""
Audit a user's session data for paper completeness.

Usage:
    python scripts/audit_user_session.py --email jian.chen@example.com
    python scripts/audit_user_session.py --user-id <uuid>
    python scripts/audit_user_session.py --search-name "Jian"

Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from .env (service-role
bypasses RLS so we can read any user's data without their password).

Reports, per project and per conversation:
  - conversations row: messages with draft_grade/draft_drift/user_verdicts
    embedded; shows how many drafts, how many user_verdicts, whether grades
    survived the save
  - draft_grades rows: per-draft grader output + drift_json
  - project_data rows by data_type: drift-panel feedback, tradeoff prefs,
    rubric_edit_applied, rubric_edit_verification, rq2_threeway, etc.
  - rubric_history: all rubric versions with timestamps + dim counts

Flags gaps that would block paper analysis. Read-only.
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
    """Find candidate users. Uses auth.users admin API via service role.

    Returns list of {"id": ..., "email": ...} dicts.
    """
    if user_id:
        try:
            resp = sb.auth.admin.get_user_by_id(user_id)
            u = resp.user if hasattr(resp, "user") else resp
            if u:
                return [{"id": u.id, "email": u.email}]
        except Exception as e:
            print(f"[warn] get_user_by_id failed: {e}")
            return []

    # List and filter
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
        matches = [u for u in all_users if (u["email"] or "").lower() == email.lower()]
        return matches
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


def audit_conversation(sb: Client, conv: dict) -> dict:
    """Pull everything about one conversation. Returns a stats dict."""
    conv_id = conv["id"]
    print(f"\n--- Conversation {conv_id} (created {_fmt_ts(conv.get('created_at'))}) ---")

    raw_messages = conv.get("messages")
    try:
        messages = json.loads(raw_messages) if isinstance(raw_messages, str) else (raw_messages or [])
    except json.JSONDecodeError:
        messages = []

    # Role breakdown
    role_counts = Counter(m.get("role", "?") for m in messages)
    print(f"  messages: {len(messages)} total -- {dict(role_counts)}")

    # Drafts + grades attached
    assistant_drafts = [
        m for m in messages
        if m.get("role") == "assistant"
        and m.get("content", "")
        and "<draft>" in (m.get("content") or "")
    ]
    graded_drafts = [m for m in assistant_drafts if m.get("draft_grade")]
    drift_drafts = [m for m in assistant_drafts if (m.get("draft_drift") or {}).get("kind", "none") != "none"]
    print(f"  assistant drafts: {len(assistant_drafts)}  (with draft_grade embedded: {len(graded_drafts)})")
    print(f"  drafts with drift panel shown: {len(drift_drafts)}")
    if drift_drafts:
        drift_kinds = Counter((m.get("draft_drift") or {}).get("kind", "?") for m in drift_drafts)
        print(f"    drift kinds: {dict(drift_kinds)}")

    # User verdicts on dimensions (stored per-message)
    user_verdict_msgs = [m for m in assistant_drafts if m.get("user_verdicts")]
    total_verdicts = sum(len(m.get("user_verdicts") or {}) for m in user_verdict_msgs)
    print(f"  messages with user_verdicts: {len(user_verdict_msgs)}, total verdicts: {total_verdicts}")

    # draft_grades table
    dg_resp = sb.table("draft_grades").select("*").eq("conversation_id", conv_id).order("draft_index").execute()
    dg_rows = dg_resp.data or []
    print(f"  draft_grades rows: {len(dg_rows)}")
    if dg_rows:
        rv_counts = Counter(r.get("rubric_version") for r in dg_rows)
        print(f"    rubric_versions graded: {dict(rv_counts)}")
        drift_rows = sum(1 for r in dg_rows if r.get("drift_json"))
        print(f"    rows with drift_json persisted: {drift_rows}/{len(dg_rows)}")
        triggers = Counter(r.get("trigger") for r in dg_rows)
        print(f"    triggers: {dict(triggers)}")

    return {
        "conv_id": conv_id,
        "messages": len(messages),
        "drafts": len(assistant_drafts),
        "graded_drafts_embedded": len(graded_drafts),
        "drift_panels_rendered": len(drift_drafts),
        "user_verdicts_total": total_verdicts,
        "draft_grades_rows": len(dg_rows),
    }


_FEEDBACK_TYPES = [
    "low_confidence_feedback",
    "oscillation_feedback",
    "persistent_failure_feedback",
    "spot_check_feedback",
    "tradeoff_preference",
    "rubric_edit_applied",
    "rubric_edit_verification",
    "rq2_threeway",               # blind three-way comparison results
    "rubric_classification_cleanup",
]


def audit_project_data(sb: Client, project_id: str) -> dict:
    """Scan project_data by data_type."""
    _header("project_data (drift-panel telemetry, rubric edits, comparison)", "-")
    resp = sb.table("project_data").select("*").eq("project_id", project_id).execute()
    rows = resp.data or []
    # Group by data_type.
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r.get("data_type", "?")].append(r)

    stats = {}
    for dt in _FEEDBACK_TYPES + sorted(set(by_type.keys()) - set(_FEEDBACK_TYPES)):
        entries = by_type.get(dt, [])
        if not entries and dt in _FEEDBACK_TYPES:
            print(f"  {dt}: 0   [!!  MISSING  !!]" if dt in ("rq2_threeway",) else f"  {dt}: 0")
            stats[dt] = 0
            continue
        # Each row's `data` column is a JSON array; count items across rows.
        total_items = 0
        for r in entries:
            try:
                arr = json.loads(r["data"]) if isinstance(r.get("data"), str) else (r.get("data") or [])
                if isinstance(arr, list):
                    total_items += len(arr)
                else:
                    total_items += 1
            except json.JSONDecodeError:
                pass
        print(f"  {dt}: {total_items} items across {len(entries)} row(s)")
        stats[dt] = total_items

    return stats


def audit_rubric_history(sb: Client, project_id: str) -> dict:
    _header("rubric_history", "-")
    resp = sb.table("rubric_history").select("*").eq("project_id", project_id).order("version").execute()
    rows = resp.data or []
    print(f"  {len(rows)} version(s) total")
    for r in rows:
        try:
            rubric = json.loads(r["rubric_data"]) if isinstance(r.get("rubric_data"), str) else (r.get("rubric_data") or {})
        except Exception:
            rubric = {}
        crits = rubric.get("rubric", []) if isinstance(rubric, dict) else []
        n_dims = sum(len(c.get("dimensions") or []) for c in crits)
        source = (rubric.get("source") if isinstance(rubric, dict) else "") or ""
        print(f"    v{r.get('version')} ({_fmt_ts(r.get('created_at'))}) -- {len(crits)} criteria, {n_dims} dims  source={source}")
    return {"versions": len(rows)}


def audit_project(sb: Client, user_id: str, user_email: str) -> None:
    _header(f"USER {user_email} ({user_id})")
    projects_resp = sb.table("projects").select("*").eq("user_id", user_id).order("created_at").execute()
    projects = projects_resp.data or []
    if not projects:
        print("  No projects.")
        return
    for p in projects:
        pid = p["id"]
        pname = p.get("name", "?")
        _header(f"PROJECT: {pname}  ({pid})")
        print(f"  created {_fmt_ts(p.get('created_at'))}")
        # conversations
        c_resp = sb.table("conversations").select("*").eq("project_id", pid).order("created_at").execute()
        convs = c_resp.data or []
        print(f"\n  conversations: {len(convs)}")
        for conv in convs:
            audit_conversation(sb, conv)

        audit_rubric_history(sb, pid)
        stats = audit_project_data(sb, pid)

        # Paper-audit summary
        _header(f"SUMMARY for project {pname}", "=")
        # Required-for-paper checks
        checks = [
            ("Any rubric versions saved", stats_has_rubric := sb.table("rubric_history").select("id", count="exact").eq("project_id", pid).execute().count or 0),
            ("Any conversations", len(convs)),
            ("Any draft_grades rows (via conversations)", sum(
                (sb.table("draft_grades").select("id", count="exact").eq("conversation_id", c["id"]).execute().count or 0)
                for c in convs
            )),
            ("rq2_threeway preferences logged", stats.get("rq2_threeway", 0)),
            ("drift-panel feedback logged",
             stats.get("low_confidence_feedback", 0)
             + stats.get("oscillation_feedback", 0)
             + stats.get("persistent_failure_feedback", 0)
             + stats.get("spot_check_feedback", 0)),
            ("Rubric edits applied", stats.get("rubric_edit_applied", 0)),
            ("Rubric edit verifications", stats.get("rubric_edit_verification", 0)),
        ]
        for name, val in checks:
            mark = "✓" if val > 0 else "✗"
            print(f"  {mark} {name}: {val}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--email", help="exact user email")
    g.add_argument("--user-id", help="user UUID")
    g.add_argument("--search-name", help="substring match on email")
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
    audit_project(sb, u["id"], u["email"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
