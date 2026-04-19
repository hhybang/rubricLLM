"""Classify rubric edits and build edit-log messages."""
from rubric_writer.imports import *

def classify_rubric_edits(old_rubric_list, new_rubric_list):
    """
    Diff two rubric criteria lists and classify every change by type.
    Returns a structured dict of edits.
    """
    old_map = {}
    for c in (old_rubric_list or []):
        key = c.get('name', '').lower().strip()
        old_map[key] = c

    new_map = {}
    for c in (new_rubric_list or []):
        key = c.get('name', '').lower().strip()
        new_map[key] = c

    edits = {
        "added": [],
        "removed": [],
        "reweighted": [],
        "reworded": [],
        "dimensions_changed": []
    }

    # Find added and modified criteria
    for key, new_c in new_map.items():
        if key not in old_map:
            edits["added"].append({
                "name": new_c.get("name", ""),
                "description": new_c.get("description", ""),
                "weight": new_c.get("weight", new_c.get("priority", 0))
            })
        else:
            old_c = old_map[key]
            # Check weight/priority change
            old_w = old_c.get('weight', old_c.get('priority', 0))
            new_w = new_c.get('weight', new_c.get('priority', 0))
            if old_w != new_w:
                edits["reweighted"].append({
                    "name": new_c.get("name", ""),
                    "old_weight": old_w,
                    "new_weight": new_w
                })
            # Check description change (normalize whitespace to avoid false positives)
            old_desc = ' '.join(old_c.get('description', '').split())
            new_desc = ' '.join(new_c.get('description', '').split())
            if old_desc != new_desc:
                edits["reworded"].append({
                    "name": new_c.get("name", ""),
                    "field": "description",
                    "old": old_desc,
                    "new": new_desc
                })
            # Check dimension changes
            old_dims = set(d.get('label', '').strip() for d in old_c.get('dimensions', []) if d.get('label', '').strip())
            new_dims = set(d.get('label', '').strip() for d in new_c.get('dimensions', []) if d.get('label', '').strip())
            if old_dims != new_dims:
                edits["dimensions_changed"].append({
                    "name": new_c.get("name", ""),
                    "added_dims": list(new_dims - old_dims),
                    "removed_dims": list(old_dims - new_dims)
                })

    # Find removed criteria
    for key, old_c in old_map.items():
        if key not in new_map:
            edits["removed"].append({
                "name": old_c.get("name", ""),
                "description": old_c.get("description", "")
            })

    return edits

def format_edit_log_message(edit_classification, old_version, new_version, source):
    """
    Format a rubric edit classification into a conversation log message.
    Contains both human-readable text and a machine-parseable HTML comment.
    """
    lines = [f"📋 **(Temporary) Rubric Changes have been made:**"]

    edits = edit_classification
    if edits["added"]:
        for a in edits["added"]:
            lines.append(f"- **Added:** \"{a['name']}\"")
    if edits["removed"]:
        for r in edits["removed"]:
            lines.append(f"- **Removed:** \"{r['name']}\"")
    if edits["reweighted"]:
        for rw in edits["reweighted"]:
            lines.append(f"- **Reweighted:** \"{rw['name']}\" ({rw['old_weight']} → {rw['new_weight']})")
    if edits["reworded"]:
        for rw in edits["reworded"]:
            lines.append(f"- **Reworded:** \"{rw['name']}\" {rw['field']} changed")
    if edits["dimensions_changed"]:
        for dc in edits["dimensions_changed"]:
            parts = []
            if dc["added_dims"]:
                parts.append(f"+{', '.join(dc['added_dims'])}")
            if dc["removed_dims"]:
                parts.append(f"-{', '.join(dc['removed_dims'])}")
            lines.append(f"- **Dimensions:** \"{dc['name']}\" ({'; '.join(parts)})")

    if not any(edits[k] for k in edits):
        lines.append("- No substantive changes detected")

    # Machine-readable log embedded as HTML comment
    log_data = json.dumps({
        "version_from": old_version,
        "version_to": new_version,
        "edits": edit_classification,
        "source": source,
        "reverted": False
    })
    lines.append(f"<!--RUBRIC_EDIT_LOG:{log_data}-->")

    return "\n".join(lines)

def get_effective_edits_from_conversation(messages):
    """
    Walk conversation messages and extract rubric edit logs,
    excluding any that were subsequently reverted.
    Returns a list of edit log dicts (non-reverted only).
    """
    edit_logs = []  # list of (index, parsed_log_dict)
    reverted_versions = set()

    # First pass: collect all revert events
    for msg in messages:
        content = msg.get('content', '')
        if not isinstance(content, str):
            continue
        revert_match = re.search(r'<!--RUBRIC_REVERT_LOG:(.*?)-->', content)
        if revert_match:
            try:
                revert_data = json.loads(revert_match.group(1))
                reverted_to = revert_data.get("reverted_to_version")
                if reverted_to is not None:
                    reverted_versions.add(reverted_to)
            except json.JSONDecodeError:
                pass

    # Second pass: collect edit logs, mark reverted ones
    for msg in messages:
        content = msg.get('content', '')
        if not isinstance(content, str):
            continue
        edit_match = re.search(r'<!--RUBRIC_EDIT_LOG:(.*?)-->', content)
        if edit_match:
            try:
                log_data = json.loads(edit_match.group(1))
                # An edit is "reverted" if a later revert went back to the version before this edit
                version_from = log_data.get("version_from")
                if version_from in reverted_versions:
                    log_data["reverted"] = True
                edit_logs.append(log_data)
            except json.JSONDecodeError:
                pass

    # Return only non-reverted edits
    return [log for log in edit_logs if not log.get("reverted", False)]
