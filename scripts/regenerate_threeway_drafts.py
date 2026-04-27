#!/usr/bin/env python3
"""
Regenerate the three-way comparison drafts for past sessions whose
rq2_threeway rows didn't capture draft text.

For each user, finds the latest project's rubric_history (early = first
inferred = v1 if more than one version, else v0; late = current = last
version), pulls the task text from the saved rq2_threeway row, and
regenerates one draft per arm using the same prompts the live Comparison
tab uses.

Output: one JSON file per user with {early_rubric, late_rubric, drafts_by_arm}
plus stdout printout for skim-reading.

Usage:
    python scripts/regenerate_threeway_drafts.py --email jianmc@mit.edu --email richag@mit.edu

Cost: ~3 Opus 4.7 calls per session (one per arm). 6 calls total for 2 users.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic
from supabase import create_client

# Reuse the EXACT chat system prompt the comparison tab uses, so the
# regeneration is apples-to-apples with what the user saw in their session.
from prompts import CHAT_build_system_prompt
from rubric_writer.config import MODEL_PRIMARY


# Same suffix the comparison tab applies. Keeps the regeneration aligned with
# the live behavior (force-draft, no clarifying questions).
_FORCE_DRAFT_SUFFIX = (
    "\n\nIMPORTANT: Produce a complete draft right now inside <draft>...</draft> "
    "tags. Do not ask clarifying questions and do not propose to write something "
    "different. If any details are missing, make reasonable assumptions and "
    "proceed -- the user wants to see a concrete attempt they can react to, not "
    "a clarification request."
)


def _strip_draft_tags(text: str) -> str:
    m = re.search(r"<draft>([\s\S]*?)</draft>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    text = re.sub(r"<probe_signal>[\s\S]*?</probe_signal>", "", text,
                  flags=re.IGNORECASE)
    return text.strip()


def _gen(client: anthropic.Anthropic, task: str, rubric_dict_or_empty) -> str:
    system_prompt = CHAT_build_system_prompt(rubric_dict_or_empty)
    resp = client.messages.create(
        model=MODEL_PRIMARY,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": task.strip() + _FORCE_DRAFT_SUFFIX}],
    )
    full = "".join(b.text for b in resp.content if b.type == "text")
    return _strip_draft_tags(full)


def _load_user_data(sb, email: str) -> dict | None:
    """Return {project_id, rubric_versions, threeway_rows} for the user."""
    # Find user
    page = 1
    target_uid = None
    while True:
        resp = sb.auth.admin.list_users(page=page, per_page=200)
        users = resp if isinstance(resp, list) else getattr(resp, "users", [])
        if not users:
            break
        for u in users:
            if (u.email or "").lower() == email.lower():
                target_uid = u.id
                break
        if target_uid or len(users) < 200:
            break
        page += 1
    if not target_uid:
        print(f"  [skip] no user matches {email}")
        return None
    projects = sb.table("projects").select("*").eq("user_id", target_uid).order("created_at").execute().data or []
    if not projects:
        print(f"  [skip] no projects for {email}")
        return None
    pid = projects[0]["id"]
    rubric_rows = sb.table("rubric_history").select("*").eq("project_id", pid).order("version").execute().data or []
    rubric_versions = []
    for r in rubric_rows:
        rd = r.get("rubric_data")
        if isinstance(rd, str):
            try:
                rd = json.loads(rd)
            except Exception:
                rd = {}
        rubric_versions.append({"version": r.get("version"), "rubric_data": rd})
    pd_rows = sb.table("project_data").select("*").eq("project_id", pid).eq("data_type", "rq2_threeway").execute().data or []
    threeway_items: list[dict] = []
    for row in pd_rows:
        arr = row.get("data")
        if isinstance(arr, str):
            try:
                arr = json.loads(arr)
            except Exception:
                arr = []
        if isinstance(arr, list):
            threeway_items.extend(arr)
    return {
        "user_id": target_uid,
        "project_id": pid,
        "project_name": projects[0].get("name", "?"),
        "rubric_versions": rubric_versions,
        "threeway_items": threeway_items,
    }


def regenerate_for_user(client: anthropic.Anthropic, sb, email: str) -> dict | None:
    print(f"\n{'='*78}\n{email}\n{'='*78}")
    data = _load_user_data(sb, email)
    if not data:
        return None
    rv = data["rubric_versions"]
    if not rv:
        print(f"  [skip] no rubric_history for {email}")
        return None
    # Match the comparison tab's logic: early = v1 if >= 2 versions, else v0;
    # late = last.
    early_rubric = rv[1] if len(rv) >= 2 else rv[0]
    late_rubric = rv[-1]

    if not data["threeway_items"]:
        print(f"  [skip] no rq2_threeway entries for {email}")
        return None
    threeway = data["threeway_items"][0]
    task = threeway.get("task", "")
    label_to_arm = threeway.get("label_to_arm") or {}
    print(f"  task: {task[:300]}{'...' if len(task) > 300 else ''}")
    print(f"  user picked: best_arm={threeway.get('best_arm')} worst_arm={threeway.get('worst_arm')}")
    print(f"  early rubric v{early_rubric['version']}, late rubric v{late_rubric['version']}")
    print(f"  generating drafts (3 Opus calls)...")

    drafts_by_arm = {}
    for arm in ("none", "early", "late"):
        if arm == "none":
            rb = []
        elif arm == "early":
            rb = early_rubric["rubric_data"]
        else:
            rb = late_rubric["rubric_data"]
        try:
            text = _gen(client, task, rb)
            drafts_by_arm[arm] = text
            print(f"    [{arm}] {len(text)} chars")
        except Exception as e:
            print(f"    [{arm}] FAILED: {e}")
            drafts_by_arm[arm] = f"[ERROR: {e}]"

    return {
        "email": email,
        "task": task,
        "user_choice": {
            "best_arm": threeway.get("best_arm"),
            "worst_arm": threeway.get("worst_arm"),
            "all_same": threeway.get("all_same"),
            "user_reason": threeway.get("user_reason", ""),
        },
        "early_rubric_version": early_rubric["version"],
        "late_rubric_version": late_rubric["version"],
        "drafts_by_arm": drafts_by_arm,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--email", action="append", required=True,
                   help="one or more --email flags")
    p.add_argument("--out", default="user_chats/regenerated_threeway.json",
                   help="output JSON path")
    args = p.parse_args()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")
        return 1
    sb = create_client(url, key)
    client = anthropic.Anthropic()

    results = []
    for email in args.email:
        r = regenerate_for_user(client, sb, email)
        if r:
            results.append(r)

    # Brief skim-readable preview to stdout
    print("\n" + "=" * 78)
    print("PREVIEWS (first 600 chars of each draft)")
    print("=" * 78)
    for r in results:
        print(f"\n>>> {r['email']}")
        print(f"    user picked: best={r['user_choice']['best_arm']}  worst={r['user_choice']['worst_arm']}")
        for arm, text in r["drafts_by_arm"].items():
            print(f"\n--- [{arm}] ---")
            print(text[:600] + ("...[truncated]" if len(text) > 600 else ""))

    # Full JSON for inspection
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\n[full result → {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
