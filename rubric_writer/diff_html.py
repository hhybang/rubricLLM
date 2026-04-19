"""Annotated diff HTML and probe rubric marker."""
from rubric_writer.imports import *
from rubric_writer.persistence import _auto_save_conversation

def _annotated_diff_html(old_text: str, new_text: str, annotated_changes: list, message_id: str = "") -> str:
    """
    Generate diff HTML between old and new text, with [N] markers
    linked to annotated_changes reasons via hover tooltips.

    Uses a two-pass approach:
    1. Line-level diff to identify changed regions (prevents false "equal"
       matches when similar text appears in different positions).
    2. Word-level diff within each changed line pair for fine-grained display.
    """
    import difflib
    import html as html_lib

    if not old_text and not new_text:
        return '<span style="color:#9e9e9e;font-style:italic;">Not specified</span>'
    if not old_text:
        return f'<ins>{html_lib.escape(new_text)}</ins>'
    if not new_text:
        return f'<del>{html_lib.escape(old_text)}</del>'
    if old_text.strip() == new_text.strip():
        return html_lib.escape(new_text)

    # Split into lines for first pass
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    line_matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    # Build list of annotated_changes that haven't been matched yet
    _ac_available = list(range(len(annotated_changes)))

    def _match_change(old_fragment: str, new_fragment: str):
        """Try to match a diff region to an annotated_changes entry. Returns index or None."""
        old_frag_lower = old_fragment.lower().strip()
        new_frag_lower = new_fragment.lower().strip()
        best_idx = None
        best_score = 0.0
        for idx in _ac_available:
            ac = annotated_changes[idx]
            ac_orig = (ac.get('original_text', '') or '').strip().lower()
            ac_new = (ac.get('new_text', '') or '').strip().lower()
            score = 0.0
            if ac_orig and old_frag_lower:
                if ac_orig in old_frag_lower or old_frag_lower in ac_orig:
                    overlap = min(len(ac_orig), len(old_frag_lower)) / max(len(ac_orig), len(old_frag_lower), 1)
                    score += 0.5 + 0.5 * overlap
            if ac_new and new_frag_lower:
                if ac_new in new_frag_lower or new_frag_lower in ac_new:
                    overlap = min(len(ac_new), len(new_frag_lower)) / max(len(ac_new), len(new_frag_lower), 1)
                    score += 0.5 + 0.5 * overlap
            if ac_orig and not ac_new and old_frag_lower and not new_frag_lower:
                if ac_orig in old_frag_lower or old_frag_lower in ac_orig:
                    score += 0.5
            if ac_new and not ac_orig and new_frag_lower and not old_frag_lower:
                if ac_new in new_frag_lower or new_frag_lower in ac_new:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= 0.3:
            _ac_available.remove(best_idx)
            return best_idx
        return None

    def _marker_span(match_idx):
        """Build the [N] marker HTML for a matched annotated_change."""
        ac = annotated_changes[match_idx]
        reason = (ac.get('reason', '') or '').strip()
        reason_esc = html_lib.escape(reason)
        n = match_idx + 1
        return (f' <span class="rubric-marker" title="{reason_esc}" '
                f'style="background:#e3f2fd;color:#1565c0;padding:0 4px;'
                f'border-radius:3px;font-size:0.85em;cursor:default;">[{n}]</span>')

    def _word_diff_html(old_block: str, new_block: str, try_match: bool = True):
        """Word-level diff within a changed block. Returns HTML string."""
        old_w = old_block.split()
        new_w = new_block.split()
        w_matcher = difflib.SequenceMatcher(None, old_w, new_w)
        parts = []
        # Try to match the whole block to an annotated_change
        block_match = _match_change(old_block, new_block) if try_match else None
        for tag, i1, i2, j1, j2 in w_matcher.get_opcodes():
            if tag == 'equal':
                parts.append(html_lib.escape(' '.join(old_w[i1:i2])))
            elif tag == 'replace':
                parts.append(f'<del>{html_lib.escape(" ".join(old_w[i1:i2]))}</del>')
                parts.append(f'<ins>{html_lib.escape(" ".join(new_w[j1:j2]))}</ins>')
            elif tag == 'delete':
                parts.append(f'<del>{html_lib.escape(" ".join(old_w[i1:i2]))}</del>')
            elif tag == 'insert':
                parts.append(f'<ins>{html_lib.escape(" ".join(new_w[j1:j2]))}</ins>')
        result = ' '.join(parts)
        if block_match is not None:
            result += _marker_span(block_match)
        return result

    result_parts = []

    for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
        if tag == 'equal':
            result_parts.append(html_lib.escape(''.join(old_lines[i1:i2])))
        elif tag == 'replace':
            # For replaced line groups, do word-level diff within them
            old_block = ''.join(old_lines[i1:i2])
            new_block = ''.join(new_lines[j1:j2])
            result_parts.append(_word_diff_html(old_block, new_block))
        elif tag == 'delete':
            old_block = ''.join(old_lines[i1:i2])
            match_idx = _match_change(old_block, "")
            marker = _marker_span(match_idx) if match_idx is not None else ""
            result_parts.append(f'<del>{html_lib.escape(old_block)}</del>{marker}')
        elif tag == 'insert':
            new_block = ''.join(new_lines[j1:j2])
            match_idx = _match_change("", new_block)
            marker = _marker_span(match_idx) if match_idx is not None else ""
            result_parts.append(f'<ins>{html_lib.escape(new_block)}</ins>{marker}')

    diff_html = ''.join(result_parts)
    # Preserve line breaks in HTML
    diff_html = diff_html.replace('\n', '<br>')
    return (
        '<div class="annotated-draft" style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:12px;margin:8px 0;line-height:1.7;">'
        '<style>.annotated-draft ins { background:#c8e6c9; font-weight:bold; text-decoration:none; padding:0 2px; } '
        '.annotated-draft del { background:#ffcdd2; text-decoration:line-through; padding:0 2px; }</style>'
        f'{diff_html}</div>'
    )


def _word_level_diff(old_text: str, new_text: str) -> str:
    """
    Generate word-level diff HTML between old and new text.
    Unchanged words are shown normally, removed words have strikethrough in red,
    added words are shown in green.
    """
    import difflib

    if not old_text and not new_text:
        return '<span class="no-change-badge">Not specified</span>'
    if not old_text:
        return f'<span class="text-added">{new_text}</span>'
    if not new_text:
        return f'<span class="text-removed">{old_text}</span>'
    if old_text == new_text:
        return new_text

    # Split into words while preserving whitespace
    old_words = old_text.split()
    new_words = new_text.split()

    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, old_words, new_words)
    result = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words are the same
            result.append(' '.join(old_words[i1:i2]))
        elif tag == 'replace':
            # Words were replaced
            if i1 < i2:
                result.append(f'<span class="text-removed">{" ".join(old_words[i1:i2])}</span>')
            if j1 < j2:
                result.append(f'<span class="text-added">{" ".join(new_words[j1:j2])}</span>')
        elif tag == 'delete':
            # Words were deleted
            result.append(f'<span class="text-removed">{" ".join(old_words[i1:i2])}</span>')
        elif tag == 'insert':
            # Words were inserted
            result.append(f'<span class="text-added">{" ".join(new_words[j1:j2])}</span>')

    return ' '.join(result)


def _mark_probe_rubric_updated(criterion_name: str):
    """Mark the matching probe_results entry as rubric_updated=True, update the probe log message, and re-save to DB."""
    _pr_list = st.session_state.get("probe_results", [])
    _crit_lower = (criterion_name or "").lower().strip()
    for _pr in reversed(_pr_list):  # most recent first
        if _pr.get("criterion_name", "").lower().strip() == _crit_lower and not _pr.get("rubric_updated"):
            _pr["rubric_updated"] = True
            break
    # Also update the probe log message in the conversation
    for _msg in reversed(st.session_state.get("messages", [])):
        if _msg.get("is_probe_log"):
            _pld = _msg.get("probe_log_data", {})
            if _pld.get("criterion_name", "").lower().strip() == _crit_lower:
                _pld["rubric_applied"] = True
                break
    _auto_save_conversation()
    # Re-save full list to DB
    _save_sb = st.session_state.get("supabase")
    _save_pid = st.session_state.get("current_project_id")
    if _save_sb and _save_pid:
        try:
            from auth_supabase import save_project_data as _spd
            # Overwrite the full list (save_project_data appends, so we need to use update directly)
            existing = _save_sb.table("project_data").select("id").eq("project_id", _save_pid).eq("data_type", "probe_results").execute()
            if existing.data:
                _save_sb.table("project_data").update({
                    "data": json.dumps(_pr_list),
                    "updated_at": datetime.now().isoformat()
                }).eq("id", existing.data[0]["id"]).execute()
        except Exception as _e:
            # print(f"[PROBE] Failed to persist rubric_updated: {_e}")


            pass
