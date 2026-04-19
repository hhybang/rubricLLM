"""Compare two rubrics via LLM."""
from rubric_writer.imports import *
from rubric_writer.config import MODEL_PRIMARY, client

def _parse_compare_output(text: str):
    """
    Extract Base, Rubric A, Rubric B, Key Differences, Summary from the model's combined output.
    We look for the exact headings requested in the prompt.
    Returns dict with keys: base, a, b, key_diffs, summary (strings).
    """
    import re
    s = text.replace("\r\n", "\n")

    # Key differences (before Stage 1)
    key_pat = r"###\s*Key Rubric Differences\s*\n(.*?)(?=\n###\s*Stage 1|\Z)"
    # Stage 1 – Base
    base_pat = r"###\s*Stage 1\s*–\s*Base Draft\s*\n(.*?)(?:\n###\s*Stage 2|\Z)"
    # Stage 2 – Revisions: A and B
    a_pat    = r"####\s*Rubric A Revision[^\n]*\n(.*?)(?:\n####\s*Rubric B Revision|\Z)"
    b_pat    = r"####\s*Rubric B Revision[^\n]*\n(.*?)(?:\n###\s*Summary|\n###\s*Stage 3|\Z)"
    # Summary (at end)
    sum_pat  = r"###\s*Summary of Impact\s*\n(.*)\Z"

    key_m  = re.search(key_pat,  s, flags=re.S)
    base_m = re.search(base_pat, s, flags=re.S)
    a_m    = re.search(a_pat,    s, flags=re.S)
    b_m    = re.search(b_pat,    s, flags=re.S)
    sum_m  = re.search(sum_pat,  s, flags=re.S)

    key_diffs = (key_m.group(1).strip() if key_m else "")
    base      = (base_m.group(1).strip() if base_m else "")
    a_txt     = (a_m.group(1).strip() if a_m else "")
    b_txt     = (b_m.group(1).strip() if b_m else "")
    summary   = (sum_m.group(1).strip() if sum_m else "")

    return {"base": base, "a": a_txt, "b": b_txt, "key_diffs": key_diffs, "summary": summary}

def _md_diff_to_html_compare(marked_text: str) -> str:
    """
    Convert **bold** → green add span, ~~strike~~ → red del span.
    Keep paragraphs. Works on the revision texts or comparison cells.
    """
    if marked_text is None:
        return ""
    # Escape HTML special chars first
    esc = (marked_text
           .replace("&", "&amp;")
           .replace("<", "&lt;")
           .replace(">", "&gt;"))
    # Bold additions
    esc = re.sub(r"\*\*(.+?)\*\*", r"<span class='add'>\1</span>", esc)
    # Strikethrough deletions
    esc = re.sub(r"~~(.+?)~~", r"<span class='del'>\1</span>", esc)
    # Paragraphs
    parts = [f"<p>{p}</p>" for p in esc.split("\n\n") if p.strip()]
    return "<div class='diff-wrap'>" + "".join(parts) + "</div>"

def compare_rubrics(task, rubric_a, rubric_b):
    """Compare two rubrics using the RUBRIC_COMPARE_DRAFTS_PROMPT"""
    prompt = RUBRIC_COMPARE_DRAFTS_PROMPT.format(
        task=task,
        rubric_a=json.dumps(rubric_a, indent=2),
        rubric_b=json.dumps(rubric_b, indent=2)
    )

    # Call Claude API directly
    message = _api_call_with_retry(
        model=MODEL_PRIMARY,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"}
    )

    # Extract thinking and text from response
    thinking_text = ""
    response = ""
    for block in message.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            response = block.text

    # Parse using the exact same logic as the notebook
    parsed = _parse_compare_output(response)

    return {
        "base_txt": parsed["base"],
        "a_txt": parsed["a"],
        "b_txt": parsed["b"],
        "key_diffs": parsed["key_diffs"],
        "summary": parsed["summary"],
        "thinking": thinking_text
    }
