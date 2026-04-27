#!/usr/bin/env python3
"""
Pull full conversations for one or more users from Supabase and write each
to user_chats/<username>_chat.json. Uses the service-role key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from supabase import create_client


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--email", action="append", required=True,
                   help="one or more --email flags")
    p.add_argument("--outdir", default="user_chats")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

    # Find each user
    users = []
    page = 1
    while True:
        resp = sb.auth.admin.list_users(page=page, per_page=200)
        ulist = resp if isinstance(resp, list) else getattr(resp, "users", [])
        if not ulist:
            break
        users.extend(ulist)
        if len(ulist) < 200:
            break
        page += 1

    for email in args.email:
        target = next((u for u in users if (u.email or "").lower() == email.lower()), None)
        if not target:
            print(f"[skip] no user matches {email}")
            continue
        slug = email.split("@")[0].replace(".", "_")
        # Get their projects
        projs = sb.table("projects").select("*").eq("user_id", target.id).execute().data or []
        out = []
        for pr in projs:
            convs = sb.table("conversations").select("*").eq("project_id", pr["id"]).order("created_at").execute().data or []
            for conv in convs:
                # messages is stored as JSON-string; parse once so the output
                # file has native JSON (not double-escaped).
                raw = conv.get("messages")
                if isinstance(raw, str):
                    try:
                        conv["messages"] = json.loads(raw)
                    except Exception:
                        pass
                raw_rb = conv.get("rubric")
                if isinstance(raw_rb, str):
                    try:
                        conv["rubric"] = json.loads(raw_rb)
                    except Exception:
                        pass
                out.append(conv)
        path = os.path.join(args.outdir, f"{slug}_chat.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[ok] {email} → {path}  ({len(out)} conversation(s), {os.path.getsize(path)} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
