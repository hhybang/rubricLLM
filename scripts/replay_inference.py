#!/usr/bin/env python3
"""
Replay rubric inference on a captured user conversation.

Pulls the first (or specified) conversation from user_chats/<slug>_chat.json,
runs the SAME inference prompt the live app uses (RUBRIC_INFER_ONLY_SYSTEM_PROMPT
+ RUBRIC_infer_only_user_prompt), parses, applies the data-quality filter,
and prints the inferred rubric with per-dim intent-invariance flagging so a
reviewer can eyeball whether any dim looks task-specific.

Usage:
    python scripts/replay_inference.py --slug richag --conv-index 0
    python scripts/replay_inference.py --slug jianmc

The default --conv-index 0 is the EARLIEST conversation by created_at.

Cost: 1 Opus 4.7 call per run (~$0.10 + a few cents).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic

# Reuse the live prompt + the builder + the data-quality filter so this
# script is faithful to what the app does at inference time.
from prompts import RUBRIC_INFER_ONLY_SYSTEM_PROMPT, RUBRIC_infer_only_user_prompt
from rubric_writer.config import MODEL_PRIMARY


# Local re-impl of `_build_conversation_text` from rubric_writer/persistence.py
# (the original imports streamlit so we mirror its formatting here).
def _build_conversation_text(messages: list[dict]) -> str:
    parts: list[str] = []
    msg_num = 1
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if msg.get("_synthetic_changelog"):
            parts.append(f"\n\n[Rubric Version Change]\n{content}")
            continue
        if role == "user":
            parts.append(f"\n\n[Message #{msg_num}] USER:\n{content}")
            msg_num += 1
        elif role == "assistant":
            parts.append(f"\n\n[Message #{msg_num}] ASSISTANT:\n{content}")
            msg_num += 1
        elif role == "system":
            parts.append(f"\n\n[Message #{msg_num}] SYSTEM:\n{content}")
            msg_num += 1
    return "".join(parts)


# Same data-quality filter the live inference applies. Reimplemented locally
# so we don't import streamlit-tainted modules.
def _validate_and_filter_rubric(rubric_data: dict) -> tuple[int, int, int]:
    """Returns (dropped_insufficient, dropped_empty_text, dropped_crits)."""
    dropped_insufficient = 0
    dropped_empty_text = 0
    for crit in rubric_data.get("rubric") or []:
        kept = []
        for dim in crit.get("dimensions") or []:
            ev = (dim.get("evidence") or "").strip().upper()
            if ev == "INSUFFICIENT_EVIDENCE":
                dropped_insufficient += 1
                continue
            label = (dim.get("label") or "").strip()
            description = (dim.get("description") or "").strip()
            if not label and not description:
                dropped_empty_text += 1
                continue
            kept.append(dim)
        crit["dimensions"] = kept
    before_crit = len(rubric_data.get("rubric") or [])
    rubric_data["rubric"] = [
        c for c in (rubric_data.get("rubric") or []) if c.get("dimensions")
    ]
    dropped_crits = max(0, before_crit - len(rubric_data["rubric"]))
    return dropped_insufficient, dropped_empty_text, dropped_crits


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True, help="filename slug (e.g. richag, jianmc)")
    p.add_argument("--conv-index", type=int, default=0,
                   help="0=earliest conversation; 1=second; etc.")
    p.add_argument("--max-tokens", type=int, default=20000)
    args = p.parse_args()

    path = f"user_chats/{args.slug}_chat.json"
    if not os.path.exists(path):
        print(f"missing: {path}")
        return 1
    with open(path) as f:
        convs = json.load(f)
    convs.sort(key=lambda c: c.get("created_at", ""))
    if args.conv_index >= len(convs):
        print(f"only {len(convs)} conversation(s) available; --conv-index must be < that")
        return 1
    conv = convs[args.conv_index]
    msgs = conv.get("messages") or []
    print(f"\n=== INFERRING from {args.slug}, conv {conv.get('id','?')[:8]} "
          f"({conv.get('created_at','')[:19]}, {len(msgs)} messages) ===\n")

    transcript = _build_conversation_text(msgs)
    print(f"transcript: {len(transcript)} chars")

    client = anthropic.Anthropic()
    user_prompt = RUBRIC_infer_only_user_prompt(transcript, "")

    print(f"calling {MODEL_PRIMARY}, max_tokens={args.max_tokens}...")
    resp = client.messages.create(
        model=MODEL_PRIMARY,
        max_tokens=args.max_tokens,
        system=RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    print(f"raw response: {len(raw)} chars")

    rubric_data = _extract_json(raw)
    if not rubric_data:
        print("\n[ERROR] couldn't parse JSON from response. Raw:")
        print(raw[:2000])
        return 2

    dropped_ins, dropped_empty, dropped_crits = _validate_and_filter_rubric(rubric_data)
    if dropped_ins or dropped_empty or dropped_crits:
        print(f"filter dropped: {dropped_ins} insufficient, {dropped_empty} empty-text, "
              f"{dropped_crits} now-empty crits")

    crits = rubric_data.get("rubric", [])
    print(f"\n--- INFERRED RUBRIC: {len(crits)} criteria, "
          f"{sum(len(c.get('dimensions') or []) for c in crits)} dims ---\n")
    for ci, c in enumerate(crits, 1):
        print(f"\n[C{ci}] {c.get('name','(no name)')}  (priority {c.get('priority','?')})")
        desc = (c.get("description") or "").strip()
        if desc:
            print(f"      {desc}")
        for di, d in enumerate(c.get("dimensions") or [], 1):
            label = d.get("label") or d.get("description") or "(no text)"
            evid = (d.get("evidence") or "").strip()[:200]
            conf = d.get("confidence", "?")
            print(f"      D{di}. {label}")
            print(f"          confidence={conf}  evidence={evid}")

    # Quick eyeball flag: any dim whose label or description contains
    # email-intent-specific terms. This is a heuristic, not a hard filter --
    # just to draw attention to candidates worth scrutinizing.
    intent_flag_terms = [
        "meeting time", "meeting times", "propose time", "propose a time",
        "propose specific", "specific times", "logistics", "schedule",
        "scheduling", "deadline", "ask is", "the ask",
        "concrete next step", "drives a response", "calendar",
    ]
    flagged = []
    for c in crits:
        for d in c.get("dimensions") or []:
            text = ((d.get("label") or "") + " " + (d.get("description") or "")).lower()
            for term in intent_flag_terms:
                if term in text:
                    flagged.append({
                        "criterion": c.get("name"),
                        "dimension": d.get("label") or d.get("description"),
                        "matched_term": term,
                    })
                    break
    if flagged:
        print("\n--- ⚠️  POSSIBLE INTENT-SPECIFIC DIMS (eyeball these) ---")
        for f in flagged:
            print(f"  - {f['criterion']} :: {f['dimension']}")
            print(f"      matched: {f['matched_term']!r}")
    else:
        print("\n--- ✓ no dims flagged by intent-specific heuristic ---")

    # Save full result for later inspection
    out_path = f"/tmp/replay_inference_{args.slug}_conv{args.conv_index}.json"
    with open(out_path, "w") as f:
        json.dump(rubric_data, f, indent=2, default=str)
    print(f"\n[full rubric → {out_path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
